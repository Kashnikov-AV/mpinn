"""
Реализация мультидоменной ФИНС (MPINN).

MPINN координирует несколько независимых экземпляров PINN, каждый из которых обучается на своей области,
одновременно обеспечивая условия непрерывности на границах раздела доменов.

Каждый домен имеет свои:
- Геометрию (Interval для 1D)
- Функцию УЧП
- Конфигурации ГУ (только для ВНЕШНИХ границ, НЕ для интерфейсов)
- Коллокационные точки
- Веса для потерь УЧП и ГУ

Условия на интерфейсах обрабатываются отдельно через interface_loss.
"""

import time
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax import jit, value_and_grad

from .geom import Interval
from .pinn_core import PINN, FCNet


def compute_interface_loss(
    models: tuple[Any], interfaces: tuple[float], all_lambdas: tuple[float]
) -> list[float]:
    """
    Вычисление потерь на интерфейсах между соседними доменами.

    Обеспечивает непрерывность решения и потока на границах раздела доменов:
    - Непрерывность T: (T0 - T1)^2
    - Непрерывность потока: (lambda_left * dT/dx|left - lambda_right * dT/dx|right)^2

    Args:
        models: Кортеж нейросетевых моделей для каждого домена
        interfaces: Кортеж x-координат интерфейсов
        all_lambdas: Кортеж значений лямбда (теплопроводности) для каждого домена

    Returns:
        Список значений потерь на интерфейсах (по одному на каждый интерфейс)
    """
    interface_losses = []

    for i, x_int in enumerate(interfaces):
        m_l, m_r = models[i], models[i + 1]
        l_l, l_r = all_lambdas[i], all_lambdas[i + 1]

        # Evaluate solutions at interface
        t_l = m_l(jnp.array([[x_int]])).ravel()[0]
        t_r = m_r(jnp.array([[x_int]])).ravel()[0]
        cont_t = (t_l - t_r) ** 2

        # Evaluate gradients at interface using autograd
        def make_eval_fn(model):
            return lambda xv: model(jnp.array([[xv]])).ravel()[0]

        eval_l = make_eval_fn(m_l)
        eval_r = make_eval_fn(m_r)

        dt_l = jax.grad(eval_l)(x_int)
        dt_r = jax.grad(eval_r)(x_int)
        cont_f = (l_l * dt_l - l_r * dt_r) ** 2

        interface_losses.append(cont_t + cont_f)

    return interface_losses


class MPINN:
    """
    Мультидоменная физически информированная нейронная сеть (ФИНС).

    Координирует обучение нескольких экземпляров PINN на различных пространственных доменах,
    обеспечивая граничные условия только на внешних границах и ограничения непрерывности на интерфейсах.

    Attributes:
        boundaries: Кортеж координат границ доменов (включая интерфейсы)
        n_domains: Количество подобластей
        interfaces: Кортеж x-координат интерфейсов
        pinn_instances: Список объектов PINN, по одному на каждый домен
        domain_configs: Список конфигурационных словарей для каждого домена
    """

    def __init__(
        self,
        domain_configs: list[dict],
        interfaces: tuple[float, ...],
        all_lambdas: tuple[float, ...],
        interface_weight: float = 1.0,
        rng: jax.Array | None = None,
    ):
        """
        Инициализация MPINN с несколькими нейронными сетями для многодоменных задач.

        Каждый domain_config должен содержать:
        - 'geom': Геометрия Interval для домена
        - 'pde': Функция остатка УЧП для этого домена
        - 'bc_configs': Список конфигураций ГУ только для ВНЕШНИХ границ
        - 'n_points': Количество коллокационных точек во внутренней области
        - 'weights': Кортеж (pde_weight, bc_weight) для этого домена
        - 'net': Опционально предварительно созданная FCNet (если не предоставлена, будет использована архитектура по умолчанию)

        Args:
            domain_configs: Список словарей, по одному на каждый домен с геометрией, УЧП, конфигурациями ГУ, весами
            interfaces: Кортеж x-координат интерфейсов между доменами
            all_lambdas: Кортеж значений теплопроводности для каждого домена
            interface_weight: Вес для условий непрерывности на интерфейсах
            rng: Случайный ключ JAX
        """
        self.n_domains = len(domain_configs)
        self.interfaces = interfaces
        self.all_lambdas = all_lambdas
        self.interface_weight = interface_weight
        self.domain_configs = domain_configs

        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Build boundaries from geometries
        self.boundaries = tuple(
            (cfg['geom'].x0, cfg['geom'].x1) for cfg in domain_configs
        )
        
        # Generate collocation points for each domain
        col_keys = jax.random.split(rng, self.n_domains)
        self.x_collocation = tuple(
            cfg['geom'].sample_interior(cfg['n_points'], rng=k)
            for cfg, k in zip(domain_configs, col_keys)
        )

        # Create independent PINN instances for each domain with their own weights
        self.pinn_instances = []
        for cfg in domain_configs:
            net = cfg.get('net')
            if net is None:
                # Default network creation if not provided
                raise ValueError("Each domain_config must include a 'net' key with FCNet instance")
            pinn = PINN(net, cfg['opt'], weights=cfg['weights'])
            self.pinn_instances.append(pinn)

        self.graphdefs = tuple(p.graphdef for p in self.pinn_instances)
        self.params = tuple(p.params for p in self.pinn_instances)
        self.txs = tuple(p.tx for p in self.pinn_instances)
        self.opt_states = tuple(p.opt_state for p in self.pinn_instances)

    def create_loss_fn(self, phys: Any | None = None):
        """
        Создание составной функции потерь для многодоменного обучения.

        Полные потери включают:
        - Остатки УЧП в каждом домене (вычисляются через PINN.create_loss_fn)
        - Потери ГУ на ВНЕШНИХ границах только (через bc_configs в каждом домене)
        - Условия непрерывности на интерфейсах (решение и поток)

        Args:
            phys: Опциональный объект PhysicsParams (может использоваться для глобальных параметров)

        Returns:
            Функция потерь с сигнатурой (params_tuple,) -> (total_loss, aux_losses)
        """
        def total_loss(params_tuple):
            # Reconstruct models from graphdefs and parameters
            models = tuple(
                nnx.merge(g, p) for g, p in zip(self.graphdefs, params_tuple)
            )

            # Compute PDE + BC losses for each domain using their own loss_fn
            domain_losses = []
            all_pde_losses = {}
            all_bc_losses = {}

            for i, (pinn, model, x_d, cfg) in enumerate(
                zip(self.pinn_instances, models, self.x_collocation, self.domain_configs)
            ):
                # Get domain-specific loss function
                pde_fn = cfg['pde']
                bc_configs = cfg['bc_configs']
                
                # Create loss function for this domain
                domain_loss_fn = pinn.create_loss_fn(pde_fn, bc_configs, phys)
                
                # Compute loss for this domain
                domain_total, aux = domain_loss_fn(model, x_d)
                
                domain_losses.append(domain_total)
                all_pde_losses[f"pde_{i}"] = float(aux[0])
                
                # BC losses from aux[1:]
                for j, bc_loss in enumerate(aux[1:]):
                    all_bc_losses[f"bc_domain{i}_{j}"] = float(bc_loss)

            # Compute interface losses
            interface_losses = compute_interface_loss(
                models, self.interfaces, self.all_lambdas
            )
            all_interface_losses = {
                f"interface_{i}": float(loss) for i, loss in enumerate(interface_losses)
            }

            # Sum domain losses
            sum_domain_losses = sum(domain_losses)
            
            # Add weighted interface losses
            interface_loss_total = self.interface_weight * sum(interface_losses)
            
            total = sum_domain_losses + interface_loss_total

            # Collect all losses for logging
            all_losses = {
                **all_pde_losses,
                **all_bc_losses,
                **all_interface_losses,
            }

            # Return total loss and individual losses for logging
            aux_losses = (
                *domain_losses,
                *interface_losses,
            )
            return total, aux_losses

        return total_loss

    @partial(jit, static_argnames=["self"])
    def train_step(
        self,
        params: tuple,
        loss_fn: Callable,
    ):
        """
        Выполнение одного шага обучения для всех доменов.

        Args:
            params: Кортеж параметров для каждой PINN
            loss_fn: Функция потерь, созданная через create_loss_fn (уже имеет привязку x_collocation)

        Returns:
            Кортеж из (new_params, new_opt_states, total_loss, aux_losses)
        """
        def closure(p):
            return loss_fn(p)

        (total, aux), grads = value_and_grad(closure, has_aux=True)(params)

        # Update each domain's parameters independently
        new_params = []
        new_opt_states = []
        for p, g, tx, os in zip(params, grads, self.txs, self.opt_states):
            updates, new_os = tx.update(g, os)
            new_params.append(optax.apply_updates(p, updates))
            new_opt_states.append(new_os)

        return tuple(new_params), tuple(new_opt_states), total, aux

    def train_loop(
        self,
        num_steps: int,
        loss_fn: Callable,
        loss_names: tuple[str, ...],
        log_interval: int = 100,
    ):
        """
        Цикл обучения для MPINN.

        Args:
            num_steps: Количество шагов обучения
            loss_fn: Функция потерь (уже имеет привязку x_collocation)
            loss_names: Имена компонентов потерь для логирования
            log_interval: Частота логирования

        Returns:
            Словарь, содержащий историю обучения
        """
        history = {"steps": [], "total_loss": []}
        for name in loss_names:
            history[name] = []

        curr_params, curr_opt_states = self.params, self.opt_states
        for step in range(num_steps):
            curr_params, curr_opt_states, total, aux = self.train_step(
                curr_params, loss_fn
            )
            if step % log_interval == 0 or step == num_steps - 1:
                history["steps"].append(step)
                history["total_loss"].append(float(total))
                for name, val in zip(loss_names, aux):
                    history[name].append(float(val))

        self.params = curr_params
        self.opt_states = curr_opt_states
        return history

    def fit(
        self,
        phys: Any | None = None,
        epochs: int = 1000,
        log_interval: int = 100,
    ):
        """
        Обучение модели MPINN.

        Args:
            phys: Опциональный объект PhysicsParams, передаваемый в create_loss_fn
            epochs: Количество эпох обучения
            log_interval: Частота логирования

        Returns:
            Кортеж из (history_dict, training_time)
        """
        loss_fn = self.create_loss_fn(phys)
        loss_names = (
            *[f"domain_{i}_loss" for i in range(self.n_domains)],
            *[f"interface_{i}" for i in range(len(self.interfaces))],
        )

        start_time = time.perf_counter()
        history = self.train_loop(epochs, loss_fn, loss_names, log_interval)
        end_time = time.perf_counter()

        return history, end_time - start_time

    def predict(self, x_test):
        """
        Предсказание значений температуры для заданных координат x.

        Args:
            x_test: Входные координаты x (массивоподобный объект)

        Returns:
            Предсказанные значения температуры
        """
        models = tuple(nnx.merge(g, p) for g, p in zip(self.graphdefs, self.params))
        x_flat = jnp.atleast_1d(x_test.ravel())
        t_pred = jnp.zeros_like(x_flat)

        for i, (b0, b1) in enumerate(zip(self.boundaries[:-1], self.boundaries[1:])):
            if i == self.n_domains - 1:
                mask = (x_flat >= b0) & (x_flat <= b1)
            else:
                mask = (x_flat >= b0) & (x_flat < b1)

            if jnp.any(mask):
                x_dom = x_flat[mask].reshape(-1, 1)
                t_pred = t_pred.at[mask].set(models[i](x_dom).ravel())
        return t_pred.reshape(-1, 1)

    def compute_metrics(self, x_test, t_pred, t_exact):
        """
        Вычисление метрик ошибки между предсказанными и точными решениями.

        Делегирует PINN.compute_metrics для согласованности.

        Args:
            x_test: Входные координаты x
            t_pred: Предсказанные температуры
            t_exact: Точные температуры

        Returns:
            Словарь метрик ошибки (MAPE, MAE, MSE, RMSE, max_error)
        """
        # Use the first PINN instance's compute_metrics for consistency
        return self.pinn_instances[0].compute_metrics(x_test, t_pred, t_exact)

    def evaluate(self, x_test, exact_fn, phys, bc_names=None):
        """
        Оценка модели относительно точного решения.

        Args:
            x_test: Тестовые координаты x
            exact_fn: Функция вычисления точного решения
            phys: Объект PhysicsParams
            bc_names: Опциональные имена граничных условий

        Returns:
            Кортеж из (metrics_dict, predicted_values, exact_values)
        """
        t_pred = self.predict(x_test)
        t_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, t_pred, t_exact.reshape(-1, 1))
        if bc_names:
            metrics["bc_left"] = bc_names[0]
            metrics["bc_right"] = bc_names[1] if len(bc_names) > 1 else bc_names[0]
        return metrics, t_pred, t_exact

    def save_plot(
        self,
        x_test,
        t_pred,
        t_exact,
        phys,
        save_path,
        title="Сравнение MPINN и точного решения",
    ):
        """
        Сохранение графика сравнения предсказанного и точного решений.

        Использует унифицированный модуль построения графиков с маркерами интерфейсов.

        Args:
            x_test: Тестовые координаты x
            t_pred: Предсказанные температуры
            t_exact: Точные температуры
            phys: Объект PhysicsParams
            save_path: Путь для сохранения фигуры
            title: Заголовок графика
        """
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            x_test.ravel(), t_exact, "b-", label="Аналитическое решение", linewidth=2
        )
        ax.plot(x_test.ravel(), t_pred, "r:", label="ФИНС", linewidth=6)

        # Mark interfaces
        for x_int in phys.interfaces:
            ax.axvline(x=x_int, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("x, м")
        ax.set_ylabel("T, К")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
        plt.close()

    def show_plot(
        self, x_test, t_pred, t_exact, phys, title="Сравнение MPINN и точного решения"
    ):
        """
        Отображение графика сравнения предсказанного и точного решений.

        Использует унифицированный модуль построения графиков с маркерами интерфейсов.

        Args:
            x_test: Тестовые координаты x
            t_pred: Предсказанные температуры
            t_exact: Точные температуры
            phys: Объект PhysicsParams
            title: Заголовок графика
        """
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            x_test.ravel(), t_exact, "b-", label="Аналитическое решение", linewidth=2
        )
        ax.plot(x_test.ravel(), t_pred, "r:", label="ФИНС", linewidth=6)

        # Mark interfaces
        for x_int in phys.interfaces:
            ax.axvline(x=x_int, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("x, м")
        ax.set_ylabel("T, К")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
