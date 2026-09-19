"""
Ядро реализации PINN на основе JAX и Flax NNX.
Поддерживает 1D, 2D и 3D задачи.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
from flax import nnx
from mpinn.config import normalize_coords, denormalize_temp

class FCNet(nnx.Module):
    """Полносвязная нейронная сеть для PINN."""

    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        num_layers: int,
        activation,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List(
            [
                nnx.Linear(din if i == 0 else dmid, dmid, rngs=rngs)
                for i in range(num_layers)
            ]
        )
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)
        self.num_layers = num_layers
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.linear_out(x)

class ScaledNet(nnx.Module):
    """Обёртка: нормализует вход, денормализует выход (T = T_norm * T_max)."""

    def __init__(self, base, x_min, x_max, T_max):
        self.base = base
        self.x_min = x_min
        self.x_max = x_max
        self.T_max = T_max

    def __call__(self, x):
        x_norm = normalize_coords(x, self.x_min, self.x_max)
        return denormalize_temp(self.base(x_norm), self.T_max)


class PINN:
    """Физически-информированная нейронная сеть (Функциональный JAX + Optax)."""

    def __init__(self, net, opt, weights, phys, pde_fn, bc_configs):
        # Разделяем модель на граф (структуру) и параметры (веса)
        self.graphdef, self.params = nnx.split(net)
        
        # Инициализируем optax оптимизатор и его состояние
        self.tx = opt  # Это должен быть optax.GradientTransformation
        self.opt_state = self.tx.init(self.params)
        
        self.phys = phys
        self.weights = weights
        self.pde_fn = pde_fn
        self.bc_configs = bc_configs
        self.loss_fn = self.create_loss_fn()

    def create_loss_fn(self):
        # Захватываем контекст в замыкание
        phys = self.phys
        pde_fn = self.pde_fn
        bc_configs = self.bc_configs
        weights = self.weights

        def total_loss(model, x_collocation):
            loss_pde = pde_fn(model, x_collocation, phys)

            loss_bcs = []
            for bc in bc_configs:
                fn = bc['fn']
                points = bc['points']
                params = bc.get('params', {})
                normals = bc.get('normals')

                if normals is not None:
                    residuals = fn(model, points, normals, **params)
                else:
                    residuals = fn(model, points, **params)

                loss_bcs.append(jnp.mean(residuals ** 2))

            loss_bc_total = sum(loss_bcs) if loss_bcs else 0.0
            total = weights[0] * loss_pde + weights[1] * loss_bc_total
            return total, (loss_pde, *loss_bcs)

        return total_loss

    @partial(jax.jit, static_argnames=['self', 'graphdef', 'tx', 'loss_fn'])
    def train_step(self, params, graphdef, x_collocation, tx, opt_state, loss_fn):
        def closure(p):
            # Собираем модель обратно из графа и параметров
            model = nnx.merge(graphdef, p)
            return loss_fn(model, x_collocation)
            
        # Считаем градиенты
        (total_loss, aux_losses), grads = jax.value_and_grad(closure, has_aux=True)(params)
        
        # Обновляем состояние оптимизатора (в новых optax нужно передавать params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        
        # Применяем обновления к весам
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, total_loss, aux_losses

    def train_loop(self, x_collocation, num_steps, log_interval=100, print_log=True):
        loss_fn = self.loss_fn

        n_bc = len(self.bc_configs)
        loss_names = ["pde", "bc_total"] + [f"bc_{i}" for i in range(n_bc)]
        history = {"total_loss": []}
        for name in loss_names:
            history[name] = []

        curr_params, curr_opt_state = self.params, self.opt_state
        
        for step in range(num_steps):
            # Явно передаем params и opt_state
            curr_params, curr_opt_state, total_loss, aux_losses = self.train_step(
                curr_params, self.graphdef, x_collocation, self.tx, curr_opt_state, loss_fn
            )
            
            if step % log_interval == 0 or step == num_steps - 1:
                pde_val = float(aux_losses[0])
                bc_vals = [float(v) for v in aux_losses[1:]]
                bc_total = float(sum(bc_vals)) if bc_vals else 0.0
                total_val = float(total_loss)

                # ---------- запись в историю ----------
                history["total_loss"].append(total_val)
                history["pde"].append(pde_val)
                history["bc_total"].append(bc_total)
                for i, val in enumerate(bc_vals):
                    history[f"bc_{i}"].append(val)

                if print_log:
                    line = (
                        f"\rstep: {step} | "
                        f"PDE: {pde_val:.3e} | "
                        f"BC: {bc_total:.3e}"
                    )
                    for i, v in enumerate(bc_vals):
                        line += f" | BC_{i}: {v:.3e}"
                    # добиваем пробелами, чтобы стереть хвост старой строки
                    print(line.ljust(120))
                    
        # Сохраняем финальное состояние
        self.params = curr_params
        self.opt_state = curr_opt_state
        return history

    def fit(self, x_collocation, epochs, log_interval=100):
        start_time = time.perf_counter()
        history = self.train_loop(x_collocation, epochs, log_interval)
        end_time = time.perf_counter()
        return history, end_time - start_time

    def predict(self, x_test):
        # Собираем модель для инференса
        model = nnx.merge(self.graphdef, self.params)
        return model(x_test)

    def compute_metrics(self, x_test, T_pred, T_exact):
        diff = T_pred - T_exact
        mse = float(jnp.mean(diff**2))
        mae = float(jnp.mean(jnp.abs(diff)))
        rmse = float(jnp.sqrt(mse))
        max_error = float(jnp.max(jnp.abs(diff)))
        mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(T_exact) + 1e-8))))
        return {
            "mape": f"{mape:.4e}",
            "mae": f"{mae:.4e}",
            "mse": f"{mse:.4e}",
            "rmse": f"{rmse:.4e}",
            "max_error": f"{max_error:.4e}",
        }

    def evaluate(self, x_test, exact_fn, phys, bc_info=None):
        T_pred = self.predict(x_test).ravel()
        T_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, T_pred, T_exact)
        if bc_info is not None:
            for name, bc_type in bc_info.items():
                metrics[f"bc_{name}"] = bc_type
        return metrics, T_pred, T_exact

    def _snapshot_params(self):
        """Независимая копия текущих параметров модели."""
        return jax.tree.map(jnp.copy, self.params)

    def _make_rl2_evaluator(self, x_test, exact_fn, phys):
        """
        Фабрика JIT-компилируемой функции оценки RL2.

        JAX кэширует компиляцию по object id функции, переданной в jit.
        Если создавать @jax.jit внутри цикла — каждый вызов порождает
        новый объект функции и новую компиляцию. Поэтому функцию нужно
        создать один раз за цикл и переиспользовать.
        """
        graphdef = self.graphdef

        @jax.jit
        def _rl2(params):
            model = nnx.merge(graphdef, params)
            T_pred = model(x_test).ravel()
            T_exact = exact_fn(x_test, phys).ravel()
            return jnp.linalg.norm(T_pred - T_exact) / (
                jnp.linalg.norm(T_exact) + 1e-12
            )

        return _rl2

    def _make_unweighted_loss_evaluator(self, x_collocation):
        """
        Фабрика JIT-компилируемой функции невзвешенного лосса.

        Считает pde + сумму всех bc без учёта config.weights: веса
        могут искажать ранжирование чекпоинтов между собой.
        """
        graphdef = self.graphdef
        loss_fn = self.loss_fn

        @jax.jit
        def _loss(params):
            model = nnx.merge(graphdef, params)
            _, (pde, *bcs) = loss_fn(model, x_collocation)
            bc_total = sum(bcs) if bcs else 0.0
            return pde + bc_total

        return _loss

    def fit_adaptive(
        self,
        x_collocation,
        x_test,
        exact_fn,
        phys,
        target_rl2: float = 1e-3,
        block_epochs: int = 50,
        max_epochs: int = 5000,
        log_interval: int = 100,
    ):
        """
        Обучает блоками по block_epochs эпох до достижения целевого RL2.

        После каждого блока считается RL2 на тестовых точках; параметры
        лучшей эпохи сохраняются. Обучение завершается при достижении
        target_rl2 либо при исчерпании max_epochs. По завершении
        восстанавливаются параметры лучшей эпохи.

        Примечание: opt_state не откатывается вместе с params — метод
        предназначен для получения финальной модели, а не для
        продолжения обучения с чекпоинта.
        """
        start_time = time.perf_counter()

        rl2_fn = self._make_rl2_evaluator(x_test, exact_fn, phys)
        history: dict[str, list] = {}
        total_epochs = 0
        best_rl2 = float("inf")
        best_params = self._snapshot_params()

        while total_epochs < max_epochs:
            block = min(block_epochs, max_epochs - total_epochs)
            block_history, _ = self.fit(x_collocation, block, log_interval)
            total_epochs += block

            for key, values in block_history.items():
                history.setdefault(key, []).extend(values)

            rl2 = float(rl2_fn(self.params))
            if rl2 < best_rl2:
                best_rl2 = rl2
                best_params = self._snapshot_params()

            if rl2 < target_rl2:
                break

        self.params = best_params
        return history, time.perf_counter() - start_time

    def fit_loss(
        self,
        x_collocation,
        epochs: int,
        log_interval: int = 100,
        checkpoint_every: int = 50,
    ):
        """
        Обучает ровно epochs эпох.

        Каждые checkpoint_every эпох сравнивает невзвешенную сумму
        pde + bc (без учёта config.weights — они могут искажать
        ранжирование чекпоинтов) и сохраняет параметры лучшей модели.
        По завершении восстанавливает параметры лучшего чекпоинта.

        Примечание: критерий чекпоинта (невзвешенный лосс) может
        отличаться от критерия обучения (взвешенный лосс). Если это
        критично — согласуйте веса или используйте fit_adaptive.
        """
        start_time = time.perf_counter()

        loss_fn = self._make_unweighted_loss_evaluator(x_collocation)
        history: dict[str, list] = {}
        total_epochs = 0
        best_loss = float("inf")
        best_params = self._snapshot_params()

        while total_epochs < epochs:
            block = min(checkpoint_every, epochs - total_epochs)
            block_history, _ = self.fit(x_collocation, block, log_interval)
            total_epochs += block

            for key, values in block_history.items():
                history.setdefault(key, []).extend(values)

            loss = float(loss_fn(self.params))
            if loss < best_loss:
                best_loss = loss
                best_params = self._snapshot_params()

        self.params = best_params
        return history, time.perf_counter() - start_time