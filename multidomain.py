import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from functools import partial
from jax import jit, value_and_grad
import geom
import matplotlib.pyplot as plt
from pinn_core import FCNet, PINN

class MPINN:
    def __init__(self, nets, opt, weights, phys, n_collocation=100, rng=None):
        self.boundaries = (phys.x_left,) + tuple(phys.interfaces) + (phys.x_right,)
        self.n_domains = len(phys.all_lambdas)
        self.interfaces = tuple(phys.interfaces)
        self.weights = tuple(weights)

        if rng is None:
            rng = jax.random.PRNGKey(0)

        col_keys = jax.random.split(rng, self.n_domains)
        self.x_collocation = tuple(
            geom.Interval(b0, b1).sample_interior(n_collocation, rng=k)
            for b0, b1, k in zip(self.boundaries[:-1], self.boundaries[1:], col_keys)
        )

        self.pinn_instances = tuple(PINN(net, opt, weights=(1.0,)) for net in nets)

        self.graphdefs = tuple(p.graphdef for p in self.pinn_instances)
        self.params = tuple(p.params for p in self.pinn_instances)
        self.txs = tuple(p.tx for p in self.pinn_instances)
        self.opt_states = tuple(p.opt_state for p in self.pinn_instances)

    def create_loss_fn(self, pde_fn, bc_left_fn, bc_right_fn, phys):
        pde_bound = partial(pde_fn, phys=phys)

        def total_loss(params_tuple, x_collocation):
            models = tuple(nnx.merge(g, p) for g, p in zip(self.graphdefs, params_tuple))

            pde_losses = tuple(pde_bound(m, x_d) for m, x_d in zip(models, x_collocation))

            loss_bc_l = bc_left_fn(models[0])
            loss_bc_r = bc_right_fn(models[-1])

            interface_losses = []
            for i, x_int in enumerate(self.interfaces):
                m_l, m_r = models[i], models[i + 1]
                l_l, l_r = phys.all_lambdas[i], phys.all_lambdas[i + 1]

                t_l = m_l(jnp.array([[x_int]])).ravel()[0]
                t_r = m_r(jnp.array([[x_int]])).ravel()[0]
                cont_t = (t_l - t_r) ** 2

                def eval_l(xv): return m_l(jnp.array([[xv]])).ravel()[0]
                def eval_r(xv): return m_r(jnp.array([[xv]])).ravel()[0]

                dt_l = jax.grad(eval_l)(x_int)
                dt_r = jax.grad(eval_r)(x_int)
                cont_f = (l_l * dt_l - l_r * dt_r) ** 2

                interface_losses.append(cont_t + cont_f)

            total = sum(w * l for w, l in zip(self.weights[:self.n_domains], pde_losses))
            total += self.weights[self.n_domains] * loss_bc_l
            total += self.weights[self.n_domains + 1] * loss_bc_r
            for w, l_int in zip(self.weights[self.n_domains + 2:], interface_losses):
                total += w * l_int

            return total, (*pde_losses, loss_bc_l, loss_bc_r, *interface_losses)

        return total_loss

    @partial(jit, static_argnames=['self', 'txs'])
    def train_step(self, params, x_collocation, txs, opt_states, loss_fn):
        def closure(p):
            return loss_fn(p, x_collocation)

        (total, aux), grads = value_and_grad(closure, has_aux=True)(params)

        new_params = []
        new_opt_states = []
        for p, g, tx, os in zip(params, grads, txs, opt_states):
            updates, new_os = tx.update(g, os)
            new_params.append(optax.apply_updates(p, updates))
            new_opt_states.append(new_os)

        return tuple(new_params), tuple(new_opt_states), total, aux

    def train_loop(self, num_steps, loss_fn, loss_names, log_interval):
        history = {'steps': [], 'total_loss': []}
        for name in loss_names:
            history[name] = []

        curr_params, curr_opt_states = self.params, self.opt_states
        for step in range(num_steps):
            curr_params, curr_opt_states, total, aux = self.train_step(
                curr_params, self.x_collocation, self.txs, curr_opt_states, loss_fn
            )
            if step % log_interval == 0 or step == num_steps - 1:
                history['steps'].append(step)
                history['total_loss'].append(float(total))
                for name, val in zip(loss_names, aux):
                    history[name].append(float(val))

        self.params = curr_params
        self.opt_states = curr_opt_states
        return history

    def fit(self, pde_fn, bc_left_fn, bc_right_fn, phys, epochs, log_interval=100):
        loss_fn = self.create_loss_fn(pde_fn, bc_left_fn, bc_right_fn, phys)
        loss_names = (
            *[f'pde_{i}' for i in range(self.n_domains)],
            'bc_left', 'bc_right',
            *[f'interface_{i}' for i in range(len(self.interfaces))]
        )

        start_time = time.perf_counter()
        history = self.train_loop(epochs, loss_fn, loss_names, log_interval)
        end_time = time.perf_counter()

        return history, end_time - start_time

    def predict(self, x_test):
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
        diff = t_pred.ravel() - t_exact.ravel()
        mse = float(jnp.mean(diff ** 2))
        mae = float(jnp.mean(jnp.abs(diff)))
        rmse = float(jnp.sqrt(mse))
        max_error = float(jnp.max(jnp.abs(diff)))
        mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(t_exact.ravel()) + 1e-8))))
        return {
            'mape': f'{mape:.4e}', 'mae': f'{mae:.4e}', 'mse': f'{mse:.4e}',
            'rmse': f'{rmse:.4e}', 'max_error': f'{max_error:.4e}'
        }

    def evaluate(self, x_test, exact_fn, phys, bc_names=None):
        t_pred = self.predict(x_test)
        t_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, t_pred, t_exact.reshape(-1, 1))
        if bc_names:
            metrics['bc_left'] = bc_names[0]
            metrics['bc_right'] = bc_names[1] if len(bc_names) > 1 else bc_names[0]
        return metrics, t_pred, t_exact

    def save_plot(self, x_test, t_pred, t_exact, phys, save_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), t_exact, 'b-', label='Аналитическое решение', linewidth=2)
        ax.plot(x_test.ravel(), t_pred, 'r:', label='ФИНС', linewidth=6)
        for x_int in phys.interfaces:
            ax.axvline(x=x_int, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('x, м')
        ax.set_ylabel('T, К')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=72, bbox_inches='tight')
        plt.close()

    def show_plot(self, x_test, t_pred, t_exact, phys):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), t_exact, 'b-', label='Аналитическое решение', linewidth=2)
        ax.plot(x_test.ravel(), t_pred, 'r:', label='ФИНС', linewidth=6)
        for x_int in phys.interfaces:
            ax.axvline(x=x_int, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('x, м')
        ax.set_ylabel('T, К')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()