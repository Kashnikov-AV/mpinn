"""Core PINN implementation using JAX and Flax NNX."""

from __future__ import annotations

import time
from functools import partial

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax import value_and_grad


class FCNet(nnx.Module):
    """Fully connected neural network for PINN."""

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


class PINN:
    """Physics-Informed Neural Network trainer."""

    def __init__(self, net: FCNet, opt, weights):
        self.optimizer = nnx.Optimizer(net, opt)   # инкапсулирует модель и состояние
        self.weights = weights

    def create_loss_fn(self, pde_fn, bc_fns, phys):
        """Возвращает функцию total_loss(model, x_collocation) -> (total, aux)."""
        def total_loss(model, x_collocation):
            loss_pde = pde_fn(model, x_collocation, phys)
            loss_bcs = [bc_fn(model) for bc_fn in bc_fns]
            total = self.weights[0] * loss_pde
            for w, l_bc in zip(self.weights[1:], loss_bcs):
                total += w * l_bc
            return total, (loss_pde, *loss_bcs)
        return total_loss

    @nnx.jit
    def train_step(self, x_collocation, pde_fn, bc_fns, phys):
        loss_fn = self.create_loss_fn(pde_fn, bc_fns, phys)
        def loss_and_aux(model):
            return loss_fn(model, x_collocation)

        (total, aux), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(self.optimizer.model)
        self.optimizer.update(grads)

        losses = {"total_loss": float(total)}
        losses["pde"] = float(aux[0])
        for i, val in enumerate(aux[1:], start=1):
            losses[f"bc_{i-1}"] = float(val)   # или более осмысленные имена
        return losses

    def train_loop(self, x_collocation, pde_fn, bc_fns, phys, num_steps, log_interval=100):
        loss_names = ["pde"] + [f"bc_{i}" for i in range(len(bc_fns))]
        history = {"steps": [], "total_loss": []}
        for name in loss_names:
            history[name] = []

        for step in range(num_steps):
            losses = self.train_step(x_collocation, pde_fn, bc_fns, phys)
            if step % log_interval == 0 or step == num_steps - 1:
                history["steps"].append(step)
                history["total_loss"].append(losses["total_loss"])
                for name in loss_names:
                    history[name].append(losses.get(name, losses["total_loss"]))
        return history

    def fit(self, x_collocation, pde_fn, bc_fns, phys, epochs):
        start_time = time.perf_counter()
        history = self.train_loop(x_collocation, pde_fn, bc_fns, phys, epochs)
        end_time = time.perf_counter()
        return history, end_time - start_time

    def predict(self, x_test):
        return self.optimizer.model(x_test)

    def compute_metrics(self, x_test, T_pred, T_exact):
        """Compute error metrics between prediction and exact solution."""
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

    def evaluate(self, x_test, exact_fn, phys, bc_names=None):
        """Evaluate model against exact solution."""
        T_pred = self.predict(x_test).ravel()
        T_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, T_pred, T_exact)
        if bc_names:
            metrics["bc_left"] = bc_names[0]
            metrics["bc_right"] = bc_names[1] if len(bc_names) > 1 else bc_names[0]
        return metrics, T_pred, T_exact

    def save_plot(self, x_test, T_pred, T_exact, phys, save_path):
        """Save comparison plot to file."""
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            x_test.ravel(), T_exact, "b-", label="Аналитическое решение", linewidth=2
        )
        ax.plot(x_test.ravel(), T_pred, "r:", label="ФИНС", linewidth=6)
        ax.set_xlabel("x, м")
        ax.set_ylabel("T, К")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
        plt.close()

    def show_plot(self, x_test, T_pred, T_exact, phys):
        """Display comparison plot."""
        _fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            x_test.ravel(), T_exact, "b-", label="Аналитическое решение", linewidth=2
        )
        ax.plot(x_test.ravel(), T_pred, "r:", label="ФИНС", linewidth=6)
        ax.set_xlabel("x, м")
        ax.set_ylabel("T, К")
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def normalize(data, min_val, max_val):
    """
    Нормализация данных к диапазону [0, 1].

    Parameters
    ----------
    data : jax.Array | float
        Исходные данные.
    min_val : float
        Минимальное значение диапазона.
    max_val : float
        Максимальное значение диапазона.

    Returns
    -------
    jax.Array | float
        Нормализованные данные в диапазоне [0, 1].
    """
    return (data - min_val) / (max_val - min_val)


def denormalize(data_norm, min_val, max_val):
    """
    Денормализация данных из диапазона [0, 1] обратно в исходный диапазон.

    Parameters
    ----------
    data_norm : jax.Array | float
        Нормализованные данные в диапазоне [0, 1].
    min_val : float
        Минимальное значение исходного диапазона.
    max_val : float
        Максимальное значение исходного диапазона.

    Returns
    -------
    jax.Array | float
        Восстановленные физические данные.
    """
    return data_norm * (max_val - min_val) + min_val
