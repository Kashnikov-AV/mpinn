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


class NormalizedNet(nnx.Module):
    """Обёртка для нормализации входа и выхода."""

    def __init__(self, base_net, x_min, x_max, T_min, T_max):
        self.base_net = base_net
        self.x_min = x_min
        self.x_max = x_max
        self.T_min = T_min
        self.T_max = T_max

    def __call__(self, x):
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-12)
        T_norm = self.base_net(x_norm)
        T = T_norm * (self.T_max - self.T_min) + self.T_min
        return T


class PINN:
    """Физически-информированная нейронная сеть (Функциональный JAX + Optax)."""

    def __init__(self, net, opt, weights, phys, pde_fn, bc_configs):
        # Разделяем модель на граф (структуру) и параметры (веса)
        self.graphdef, self.params = nnx.split(net)
        
        # Инициализируем optax оптимизатор и его состояние
        self.tx = opt  # Это должен быть optax.GradientTransformation
        self.opt_state = self.tx.init(self.params)
        
        self.weights = weights
        self.phys = phys
        self.pde_fn = pde_fn
        self.bc_configs = bc_configs

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

    def train_loop(self, x_collocation, num_steps, log_interval=100):
        loss_fn = self.create_loss_fn()
        
        n_bc = len(self.bc_configs)
        loss_names = ["pde", "bc_total"] + [f"bc_{i}" for i in range(n_bc)]
        history = {"steps": [], "total_loss": []}
        for name in loss_names:
            history[name] = []

        curr_params, curr_opt_state = self.params, self.opt_state
        
        for step in range(num_steps):
            # Явно передаем params и opt_state
            curr_params, curr_opt_state, total_loss, aux_losses = self.train_step(
                curr_params, self.graphdef, x_collocation, self.tx, curr_opt_state, loss_fn
            )
            
            if step % log_interval == 0 or step == num_steps - 1:
                history["steps"].append(step)
                history["total_loss"].append(float(total_loss))
                
                history["pde"].append(float(aux_losses[0]))
                bc_total = float(sum(aux_losses[1:])) if len(aux_losses) > 1 else 0.0
                history["bc_total"].append(bc_total)
                
                for i, val in enumerate(aux_losses[1:], start=1):
                    history[f"bc_{i-1}"].append(float(val))
                    
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