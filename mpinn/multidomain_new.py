"""
Реализация мультидоменной ФИНС (MPINN) с поддержкой различных физических параметров для каждого домена.
"""

import time
from typing import List, Tuple, Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from functools import partial
import matplotlib.pyplot as plt

from .geom import GeometryBase
from .pinn_core import PINN, FCNet
from .config import PhysicsParams


class MPINN:
    """
    Мультидоменная физически информированная нейронная сеть.
    """

    def __init__(
        self,
        domain_configs: List[Dict[str, Any]],
        interface_pairs: List[Tuple[int, str, int, str]],  # (idx1, name1, idx2, name2)
        n_interface_points: int,
        interface_weight: float = 1.0,
        rng: Optional[jax.Array] = None,
    ):
        """
        Args:
            domain_configs: список конфигураций доменов. Каждая конфигурация:
                {
                    'geom': GeometryBase,
                    'phys': PhysicsParams,          # физические параметры домена
                    'pde_fn': Callable,             # функция PDE (model, x, phys) -> скаляр
                    'bc_configs': List[Dict],       # список ГУ (как в PINN)
                    'net': nnx.Module,              # нейросеть домена (уже обёрнута в ScaledNet)
                    'opt': optax.GradientTransformation,
                    'n_points': int,                # количество коллокационных точек
                    'weights': Tuple[float, float]  # (weight_pde, weight_bc)
                }
            interface_pairs: список кортежей (idx1, name1, idx2, name2),
                где idx1, idx2 — индексы доменов, name1, name2 — имена граней (например, 'right', 'left').
            n_interface_points: количество точек на каждом интерфейсе.
            interface_weight: вес потерь на интерфейсах.
            rng: ключ JAX.
        """
        self.n_domains = len(domain_configs)
        self.domain_configs = domain_configs
        self.interface_weight = interface_weight
        self.interface_pairs = interface_pairs

        if rng is None:
            rng = jax.random.PRNGKey(0)

        # Генерация коллокационных точек для каждого домена
        self.x_collocation = []
        for cfg in domain_configs:
            pts = cfg['geom'].sample_interior(cfg['n_points'], rng=rng)
            self.x_collocation.append(pts)

        # Разделение сетей на граф и параметры
        self.graphdefs = []
        self.params = []
        self.txs = []
        self.opt_states = []

        for cfg in domain_configs:
            net = cfg['net']
            g, p = nnx.split(net)
            self.graphdefs.append(g)
            self.params.append(p)
            tx = cfg['opt']
            self.txs.append(tx)
            self.opt_states.append(tx.init(p))

        # Генерация точек интерфейсов с использованием geom.sample_interface
        self.interface_data = []  # список кортежей (points, normals_self, normals_other)
        for idx1, name1, idx2, name2 in interface_pairs:
            geom1 = domain_configs[idx1]['geom']
            geom2 = domain_configs[idx2]['geom']
            pts, n1, n2 = geom1.sample_interface(
                other_geom=geom2,
                n_points=n_interface_points,
                self_interface_name=name1,
                other_interface_name=name2,
                rng=rng
            )
            self.interface_data.append((pts, n1, n2))

        # Для predict храним границы доменов
        self.boundaries = [(cfg['geom'].x0, cfg['geom'].x1) for cfg in domain_configs]

    def create_loss_fn(self):
        """Создаёт функцию потерь для многодоменной задачи."""
        def total_loss(params_tuple):
            # Собираем модели
            models = [nnx.merge(g, p) for g, p in zip(self.graphdefs, params_tuple)]

            domain_losses = []
            pde_losses = []
            bc_losses = []

            # Потери PDE и BC для каждого домена
            for i, (model, cfg, x_d) in enumerate(zip(models, self.domain_configs, self.x_collocation)):
                phys = cfg['phys']  # каждый домен имеет свой PhysicsParams
                # PDE loss
                pde_fn = cfg['pde_fn']
                loss_pde = pde_fn(model, x_d, phys)  # pde_fn возвращает скаляр (MSE)

                # BC losses
                loss_bc = 0.0
                for bc in cfg['bc_configs']:
                    fn = bc['fn']
                    points = bc['points']
                    params_bc = bc.get('params', {})
                    normals = bc.get('normals')
                    if normals is not None:
                        residuals = fn(model, points, normals, **params_bc)
                    else:
                        residuals = fn(model, points, **params_bc)
                    loss_bc += jnp.mean(residuals ** 2)

                # Взвешенная сумма
                w_pde, w_bc = cfg.get('weights', (1.0, 1.0))
                domain_loss = w_pde * loss_pde + w_bc * loss_bc
                domain_losses.append(domain_loss)
                pde_losses.append(loss_pde)
                bc_losses.append(loss_bc)

            # Потери на интерфейсах
            interface_losses = []
            for (pts, normals_self, normals_other), (idx1, name1, idx2, name2) in zip(
                self.interface_data, self.interface_pairs
            ):
                m1 = models[idx1]
                m2 = models[idx2]
                lam1 = self.domain_configs[idx1]['phys']._lambda
                lam2 = self.domain_configs[idx2]['phys']._lambda

                # Значения на интерфейсе
                T1 = m1(pts).ravel()
                T2 = m2(pts).ravel()
                cont_T = jnp.mean((T1 - T2) ** 2)

                # Производные по направлению нормали
                def grad_fn(model, pts, normals):
                    def predict_single(x):
                        return model(x.reshape(1, -1)).ravel()[0]
                    grad_fn_single = jax.grad(predict_single)
                    grads = jax.vmap(grad_fn_single)(pts)
                    return jnp.sum(grads * normals, axis=1)

                dT1n = grad_fn(m1, pts, normals_self)
                dT2n = grad_fn(m2, pts, normals_other)
                cont_flux = jnp.mean((lam1 * dT1n - lam2 * dT2n) ** 2)

                interface_loss = cont_T + cont_flux
                interface_losses.append(interface_loss)

            # Суммируем
            total_domain = sum(domain_losses)
            total_interface = self.interface_weight * sum(interface_losses)
            total = total_domain + total_interface

            # Вспомогательные значения для логирования
            aux = (*domain_losses, *interface_losses, *pde_losses, *bc_losses)
            return total, aux

        return total_loss

    @partial(jax.jit, static_argnames=['self'])
    def train_step(self, params_tuple, loss_fn):
        def closure(p):
            return loss_fn(p)

        (total, aux), grads = jax.value_and_grad(closure, has_aux=True)(params_tuple)

        new_params = []
        new_opt_states = []
        for p, g, tx, os in zip(params_tuple, grads, self.txs, self.opt_states):
            updates, new_os = tx.update(g, os)
            new_params.append(optax.apply_updates(p, updates))
            new_opt_states.append(new_os)

        return tuple(new_params), tuple(new_opt_states), total, aux

    def fit(self, epochs: int = 1000, log_interval: int = 100):
        """Обучение модели MPINN."""
        loss_fn = self.create_loss_fn()
        n_interfaces = len(self.interface_pairs)
        loss_names = (
            *[f"domain_{i}" for i in range(self.n_domains)],
            *[f"interface_{i}" for i in range(n_interfaces)],
            *[f"pde_{i}" for i in range(self.n_domains)],
            *[f"bc_{i}" for i in range(self.n_domains)],
        )

        history = {"steps": [], "total_loss": []}
        for name in loss_names:
            history[name] = []

        curr_params = self.params
        curr_opt_states = self.opt_states

        start_time = time.perf_counter()
        for step in range(epochs):
            curr_params, curr_opt_states, total, aux = self.train_step(
                curr_params, loss_fn
            )
            if step % log_interval == 0 or step == epochs - 1:
                history["steps"].append(step)
                history["total_loss"].append(float(total))
                for name, val in zip(loss_names, aux):
                    history[name].append(float(val))

        self.params = curr_params
        self.opt_states = curr_opt_states
        end_time = time.perf_counter()

        return history, end_time - start_time

    def predict(self, x_test):
        models = [nnx.merge(g, p) for g, p in zip(self.graphdefs, self.params)]
        x_flat = jnp.atleast_1d(x_test.ravel())
        t_pred = jnp.zeros_like(x_flat)

        for i, (b0, b1) in enumerate(self.boundaries):
            if i == self.n_domains - 1:
                mask = (x_flat >= b0) & (x_flat <= b1)
            else:
                mask = (x_flat >= b0) & (x_flat < b1)
            if jnp.any(mask):
                x_dom = x_flat[mask].reshape(-1, 1)
                t_pred = t_pred.at[mask].set(models[i](x_dom).ravel())
        return t_pred.reshape(-1, 1)

    def compute_metrics(self, x_test, t_pred, t_exact):
        diff = t_pred - t_exact
        mse = float(jnp.mean(diff**2))
        mae = float(jnp.mean(jnp.abs(diff)))
        rmse = float(jnp.sqrt(mse))
        max_error = float(jnp.max(jnp.abs(diff)))
        mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(t_exact) + 1e-8))))
        return {
            "mape": f"{mape:.4e}",
            "mae": f"{mae:.4e}",
            "mse": f"{mse:.4e}",
            "rmse": f"{rmse:.4e}",
            "max_error": f"{max_error:.4e}",
        }

    def evaluate(self, x_test, exact_fn, phys=None, bc_info=None):
        t_pred = self.predict(x_test).ravel()
        # Если exact_fn требует phys, но у нас разные phys для доменов, то нужно передавать что-то одно.
        # Для простоты предполагаем, что exact_fn может работать с любым phys (например, использует только координаты).
        # Если нужен phys конкретного домена, можно передать phys первого домена или объединить.
        # Здесь мы оставляем возможность передать phys в evaluate.
        t_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, t_pred, t_exact)
        if bc_info:
            for name, bc_type in bc_info.items():
                metrics[f"bc_{name}"] = bc_type
        return metrics, t_pred, t_exact

    def show_plot(self, x_test, t_pred, t_exact, title="Сравнение MPINN и точного решения"):
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), t_exact, "b-", label="Аналитическое решение", linewidth=2)
        ax.plot(x_test.ravel(), t_pred, "r:", label="ФИНС", linewidth=6)
        # Отмечаем интерфейсы
        for idx1, name1, idx2, name2 in self.interface_pairs:
            geom1 = self.domain_configs[idx1]['geom']
            if hasattr(geom1, 'x0') and hasattr(geom1, 'x1'):
                if name1 == 'right':
                    x_int = geom1.x1
                elif name1 == 'left':
                    x_int = geom1.x0
                else:
                    continue
                ax.axvline(x=x_int, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("x, м")
        ax.set_ylabel("T, К")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_plot(self, x_test, t_pred, t_exact, save_path, title="Сравнение MPINN и точного решения"):
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), t_exact, "b-", label="Аналитическое решение", linewidth=2)
        ax.plot(x_test.ravel(), t_pred, "r:", label="ФИНС", linewidth=6)
        for idx1, name1, idx2, name2 in self.interface_pairs:
            geom1 = self.domain_configs[idx1]['geom']
            if hasattr(geom1, 'x0') and hasattr(geom1, 'x1'):
                if name1 == 'right':
                    x_int = geom1.x1
                elif name1 == 'left':
                    x_int = geom1.x0
                else:
                    continue
                ax.axvline(x=x_int, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("x, м")
        ax.set_ylabel("T, К")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
        plt.close()