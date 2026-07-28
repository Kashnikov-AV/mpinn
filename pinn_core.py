import time
import jax
import flax.nnx as nnx
from jax import grad, jit, vmap, random, value_and_grad
import optax
from functools import partial
import matplotlib.pyplot as plt
import jax.numpy as jnp

class FCNet(nnx.Module):
    def __init__(self, din, dmid, dout, num_layers, activation, rngs: nnx.Rngs):
        self.layers = nnx.List([
            nnx.Linear(din if i == 0 else dmid, dmid, rngs=rngs) 
            for i in range(num_layers)
        ])
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)
        self.num_layers = num_layers
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.linear_out(x)

class PINN:
    def __init__(self, net: FCNet, opt, weights):
        self.net = net
        self.graphdef, self.params = nnx.split(net)
        self.tx = opt
        self.opt_state = self.tx.init(self.params)
        self.weights = weights

    def create_loss_fn(self, pde_fn, *bc_fns, phys):
        pde_bound = partial(pde_fn, phys=phys)
        bc_bounds = list(bc_fns)

        def total_loss(model, x_collocation):
            loss_pde = pde_bound(model, x_collocation)
            loss_bcs = [bc_b(model) for bc_b in bc_bounds]
            
            total = self.weights[0] * loss_pde
            for w, l_bc in zip(self.weights[1:], loss_bcs):
                total += w * l_bc
                
            return total, (loss_pde, *loss_bcs)
        return total_loss

    @partial(jit, static_argnames=['self', 'graphdef', 'tx', 'loss_fn'])
    def train_step(self, params, graphdef, x_collocation, tx, opt_state, loss_fn):
        def closure(p):
            model = nnx.merge(graphdef, p)
            return loss_fn(model, x_collocation)
            
        (total_loss, aux_losses), grads = value_and_grad(closure, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, aux_losses

    def train_loop(self, x_collocation, num_steps, loss_fn, loss_names, log_interval=100):
        history = {'steps': [], 'total_loss': []}
        for name in loss_names:
            history[name] = []
            
        curr_params, curr_opt_state = self.params, self.opt_state
        for step in range(num_steps):
            curr_params, curr_opt_state, total_loss, aux_losses = self.train_step(
                curr_params, self.graphdef, x_collocation, self.tx, curr_opt_state, loss_fn)
                
            if step % log_interval == 0 or step == num_steps - 1:
                history['steps'].append(step)
                history['total_loss'].append(float(total_loss))
                for name, val in zip(loss_names, aux_losses):
                    history[name].append(float(val))
                    
        self.params = curr_params
        self.opt_state = curr_opt_state
        return history

    def fit(self, x_collocation, pde_fn, bc_fns, phys, epochs):
        loss_fn = self.create_loss_fn(pde_fn, *bc_fns, phys=phys)
        loss_names = ['pde'] + [f'bc_{i}' for i in range(len(bc_fns))]
        
        start_time = time.perf_counter()
        history = self.train_loop(x_collocation, epochs, loss_fn, loss_names=loss_names)
        end_time = time.perf_counter()

        training_time = end_time - start_time
        
        return history, training_time

    def predict(self, x_test):
        model = nnx.merge(self.graphdef, self.params)
        return model(x_test)

    def compute_metrics(self, x_test, T_pred, T_exact):
        diff = T_pred - T_exact
        mse = float(jnp.mean(diff ** 2))
        mae = float(jnp.mean(jnp.abs(diff)))
        rmse = float(jnp.sqrt(mse))
        max_error = float(jnp.max(jnp.abs(diff)))
        mape = float(jnp.mean(jnp.abs(diff / (jnp.abs(T_exact) + 1e-8))))
        return {
            'mape': f'{mape:.4e}', 'mae': f'{mae:.4e}', 'mse': f'{mse:.4e}',
            'rmse': f'{rmse:.4e}', 'max_error': f'{max_error:.4e}'
        }

    def evaluate(self, x_test, exact_fn, phys, bc_names=None):
        T_pred = self.predict(x_test).ravel()
        T_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, T_pred, T_exact)
        if bc_names:
            metrics['bc_left'] = bc_names[0]
            metrics['bc_right'] = bc_names[1] if len(bc_names) > 1 else bc_names[0]
        return metrics, T_pred, T_exact

    def save_plot(self, x_test, T_pred, T_exact, phys, save_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), T_exact, 'b-', label='Аналитическое решение', linewidth=2)
        ax.plot(x_test.ravel(), T_pred, 'r:', label='ФИНС', linewidth=6)
        ax.set_xlabel('x, м')
        ax.set_ylabel('T, К')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=72, bbox_inches='tight')
        plt.close()

    def show_plot(self, x_test, T_pred, T_exact, phys):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), T_exact, 'b-', label='Аналитическое решение', linewidth=2)
        ax.plot(x_test.ravel(), T_pred, 'r:', label='ФИНС', linewidth=6)
        ax.set_xlabel('x, м')
        ax.set_ylabel('T, К')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def normalize(data, min_val, max_val):
    """
    нормализация данных к диапазону [0, 1].
    Parameters:
    -----------
    data : jax.Array или float
        Исходные данные.
    min_val : float
        Минимальное значение диапазона.
    max_val : float
        Максимальное значение диапазона.
    Returns:
    --------
    jax.Array или float
        Нормализованные данные в диапазоне [0, 1].
    """
    return (data - min_val) / (max_val - min_val)

def denormalize(data_norm, min_val, max_val):
    """
    денормализация данных из диапазона [0, 1] обратно в исходный диапазон.
    Parameters:
    -----------
    data_norm : jax.Array или float
        Нормализованные данные в диапазоне [0, 1].
    min_val : float
        Минимальное значение исходного диапазона.
    max_val : float
        Максимальное значение исходного диапазона.
    Returns:
    --------
    jax.Array или float
        Восстановленные физические данные.
    """
    return data_norm * (max_val - min_val) + min_val