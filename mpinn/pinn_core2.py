"""
Ядро PINN на JAX + Flax NNX. 1D/2D/3D.
"""

from __future__ import annotations

import time
from typing import Callable

import jax.numpy as jnp
import optax
from flax import nnx

from mpinn.config import normalize_coords, denormalize_temp


class FCNet(nnx.Module):
    def __init__(self, din, dmid, dout, num_layers, activation, rngs: nnx.Rngs):
        super().__init__()
        self.layers = nnx.List(
            nnx.Linear(din if i == 0 else dmid, dmid, rngs=rngs)
            for i in range(num_layers)
        )
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)
        self.num_layers = num_layers
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.linear_out(x)


class ScaledNet(nnx.Module):
    def __init__(self, base, x_min, x_max, T_max):
        super().__init__()
        self.base = base
        self.x_min = jnp.asarray(x_min)
        self.x_max = jnp.asarray(x_max)
        self.T_max = jnp.asarray(T_max)

    def __call__(self, x):
        x_norm = normalize_coords(x, self.x_min, self.x_max)
        return denormalize_temp(self.base(x_norm), self.T_max)


def _weights_tuple(weights):
    return (weights["pde"], weights["bc"]) if isinstance(weights, dict) else tuple(weights[:2])


def _apply_bc(model, bc):
    fn, points = bc["fn"], bc["points"]
    params = bc.get("params", {})
    normals = bc.get("normals")
    r = fn(model, points, normals, **params) if normals is not None else fn(model, points, **params)
    return jnp.mean(r ** 2)


def _compute_loss(model, x_collocation, pde_fn, bc_configs, phys, weights):
    loss_pde = pde_fn(model, x_collocation, phys)
    bcs = [_apply_bc(model, bc) for bc in bc_configs]
    bc_total = sum(bcs) if bcs else jnp.zeros(())
    w_pde, w_bc = _weights_tuple(weights)
    return w_pde * loss_pde + w_bc * bc_total, (loss_pde, *bcs)


def _bc_residual_sum(model, x_collocation, pde_fn, bc_configs, phys):
    total = pde_fn(model, x_collocation, phys)
    for bc in bc_configs:
        total = total + _apply_bc(model, bc)
    return total


class PINN(nnx.Module):
    def __init__(self, net, tx, weights, phys, pde_fn, bc_configs):
        super().__init__()
        self.net = net
        self.optimizer = nnx.Optimizer(net, tx)
        self.weights = weights
        self.phys = phys
        self.pde_fn = pde_fn
        self.bc_configs = bc_configs
        self._train_step = self._make_train_step()
        self._loss_eval = self._make_loss_eval()

    def _make_train_step(self):
        pde_fn, bc_configs, phys, weights = (
            self.pde_fn, self.bc_configs, self.phys, self.weights,
        )

        @nnx.jit
        def train_step(net, optimizer, x_collocation):
            def loss_fn(model):
                return _compute_loss(model, x_collocation, pde_fn, bc_configs, phys, weights)

            (total, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(net)
            optimizer.update(grads)
            return total, aux

        return train_step

    def _make_loss_eval(self):
        pde_fn, bc_configs, phys = self.pde_fn, self.bc_configs, self.phys

        @nnx.jit
        def loss_eval(net, x_collocation):
            return _bc_residual_sum(net, x_collocation, pde_fn, bc_configs, phys)

        return loss_eval

    def _make_rl2_fn(self, x_test, exact_fn):
        phys = self.phys

        @nnx.jit
        def rl2_eval(net):
            T_pred = net(x_test).ravel()
            T_exact = exact_fn(x_test, phys).ravel()
            return jnp.linalg.norm(T_pred - T_exact) / (jnp.linalg.norm(T_exact) + 1e-12)

        return rl2_eval

    def train_loop(self, x_collocation, num_steps, log_interval=100, print_log=True):
        n_bc = len(self.bc_configs)
        history = {"total_loss": [], "pde": [], "bc_total": []}
        for i in range(n_bc):
            history[f"bc_{i}"] = []

        for step in range(num_steps):
            total_loss, aux = self._train_step(self.net, self.optimizer, x_collocation)

            if step % log_interval == 0 or step == num_steps - 1:
                pde_val = float(aux[0])
                bc_vals = [float(v) for v in aux[1:]]
                bc_total = float(sum(bc_vals)) if bc_vals else 0.0

                history["total_loss"].append(float(total_loss))
                history["pde"].append(pde_val)
                history["bc_total"].append(bc_total)
                for i, v in enumerate(bc_vals):
                    history[f"bc_{i}"].append(v)

                if print_log:
                    line = f"\rstep: {step} | PDE: {pde_val:.3e} | BC: {bc_total:.3e}"
                    for i, v in enumerate(bc_vals):
                        line += f" | BC_{i}: {v:.3e}"
                    print(line.ljust(120), end="", flush=True)

        if print_log:
            print()
        return history

    def fit(self, x_collocation, epochs, log_interval=100):
        start = time.perf_counter()
        history = self.train_loop(x_collocation, epochs, log_interval)
        return history, time.perf_counter() - start

    def _snapshot(self):
        return nnx.clone(self.net)

    def _restore(self, snapshot):
        nnx.update(self.net, nnx.state(snapshot))

    def predict(self, x_test):
        return self.net(x_test)

    def compute_metrics(self, x_test, T_pred, T_exact):
        diff = T_pred - T_exact
        mse = float(jnp.mean(diff ** 2))
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

    def _fit_blocks(self, x_collocation, total_steps, block_size, log_interval,
                    score_fn, stop_when=None):
        history = {}
        best_score = float("inf")
        best_net = self._snapshot()

        done = 0
        while done < total_steps:
            block = min(block_size, total_steps - done)
            block_history, _ = self.fit(x_collocation, block, log_interval)
            done += block

            for k, v in block_history.items():
                history.setdefault(k, []).extend(v)

            score = score_fn()
            if score < best_score:
                best_score, best_net = score, self._snapshot()

            if stop_when is not None and stop_when(score):
                break

        self._restore(best_net)
        return history, best_score

    def fit_adaptive(self, x_collocation, x_test, exact_fn, phys,
                     target_rl2=1e-3, block_epochs=50, max_epochs=5000,
                     log_interval=100):
        start = time.perf_counter()
        rl2_fn = self._make_rl2_fn(x_test, exact_fn)

        history, _ = self._fit_blocks(
            x_collocation,
            total_steps=max_epochs,
            block_size=block_epochs,
            log_interval=log_interval,
            score_fn=lambda: float(rl2_fn(self.net)),
            stop_when=lambda s: s < target_rl2,
        )
        return history, time.perf_counter() - start

    def fit_loss(self, x_collocation, epochs, log_interval=100, checkpoint_every=50):
        start = time.perf_counter()

        history, _ = self._fit_blocks(
            x_collocation,
            total_steps=epochs,
            block_size=checkpoint_every,
            log_interval=log_interval,
            score_fn=lambda: float(self._loss_eval(self.net, x_collocation)),
        )
        return history, time.perf_counter() - start