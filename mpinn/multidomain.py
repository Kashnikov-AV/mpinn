"""
Multi-Domain PINN (MPINN) implementation.

MPINN coordinates multiple independent PINN instances, each trained on its own domain,
while enforcing continuity conditions at domain interfaces.
"""

import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from functools import partial
from jax import jit, value_and_grad
from typing import Dict, List, Tuple, Any, Optional, Callable
import matplotlib.pyplot as plt

from .geom import Interval
from .pinn_core import FCNet, PINN, normalize, denormalize
from .plotting import show_plot, save_plot, show_history
from .weight_strategies import BaseWeightStrategy, FixedWeightStrategy


def compute_interface_loss(
    models: Tuple[Any],
    interfaces: Tuple[float],
    all_lambdas: Tuple[float]
) -> List[float]:
    """
    Compute interface losses between adjacent domains.
    
    Enforces continuity of solution and flux at domain interfaces:
    - Continuity of T: (T_left - T_right)^2
    - Continuity of flux: (lambda_left * dT/dx|left - lambda_right * dT/dx|right)^2
    
    Args:
        models: Tuple of neural network models for each domain
        interfaces: Tuple of interface x-coordinates
        all_lambdas: Tuple of lambda (conductivity) values for each domain
        
    Returns:
        List of interface loss values (one per interface)
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
        def eval_l(xv): return m_l(jnp.array([[xv]])).ravel()[0]
        def eval_r(xv): return m_r(jnp.array([[xv]])).ravel()[0]
        
        dt_l = jax.grad(eval_l)(x_int)
        dt_r = jax.grad(eval_r)(x_int)
        cont_f = (l_l * dt_l - l_r * dt_r) ** 2
        
        interface_losses.append(cont_t + cont_f)
    
    return interface_losses


class MPINN:
    """
    Multi-Domain Physics-Informed Neural Network.
    
    Coordinates training of multiple PINN instances across different spatial domains,
    enforcing boundary conditions and interface continuity constraints.
    
    Attributes:
        boundaries: Tuple of domain boundary coordinates (including interfaces)
        n_domains: Number of subdomains
        interfaces: Tuple of interface x-coordinates
        pinn_instances: Tuple of PINN objects, one per domain
        weight_strategy: Strategy for computing loss weights
    """
    
    def __init__(
        self,
        nets: Tuple[FCNet, ...],
        opt: optax.GradientTransformation,
        phys: Any,
        n_collocation: int = 100,
        weight_strategy: Optional[BaseWeightStrategy] = None,
        rng: Optional[jax.Array] = None
    ):
        """
        Initialize MPINN with multiple neural networks for multi-domain problems.
        
        Args:
            nets: Tuple of FCNet instances, one per domain
            opt: Optax optimizer
            phys: PhysicsParams object containing domain configuration
            n_collocation: Number of collocation points per domain
            weight_strategy: Strategy for loss weighting (default: FixedWeightStrategy)
            rng: JAX random key
        """
        self.boundaries = (phys.x_left,) + tuple(phys.interfaces) + (phys.x_right,)
        self.n_domains = len(phys.all_lambdas)
        self.interfaces = tuple(phys.interfaces)
        self.all_lambdas = phys.all_lambdas
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        # Generate collocation points for each domain
        col_keys = jax.random.split(rng, self.n_domains)
        self.x_collocation = tuple(
            Interval(b0, b1).sample_interior(n_collocation, rng=k)
            for b0, b1, k in zip(self.boundaries[:-1], self.boundaries[1:], col_keys)
        )
        
        # Create independent PINN instances for each domain
        self.pinn_instances = tuple(PINN(net, opt, weights=(1.0,)) for net in nets)
        
        self.graphdefs = tuple(p.graphdef for p in self.pinn_instances)
        self.params = tuple(p.params for p in self.pinn_instances)
        self.txs = tuple(p.tx for p in self.pinn_instances)
        self.opt_states = tuple(p.opt_state for p in self.pinn_instances)
        
        # Initialize weight strategy
        self.weight_strategy = weight_strategy or FixedWeightStrategy()

    def create_loss_fn(
        self, 
        pde_fn: Callable, 
        bc_left_fn: Callable, 
        bc_right_fn: Callable, 
        phys: Any
    ):
        """
        Create a composite loss function for multi-domain training.
        
        The total loss includes:
        - PDE residuals in each domain
        - Boundary conditions at left and right boundaries
        - Interface continuity conditions (solution and flux)
        
        Args:
            pde_fn: PDE residual function
            bc_left_fn: Left boundary condition function
            bc_right_fn: Right boundary condition function
            phys: PhysicsParams object
            
        Returns:
            A loss function with signature (params_tuple, x_collocation) -> (total_loss, aux_losses)
        """
        pde_bound = partial(pde_fn, phys=phys)

        def total_loss(params_tuple, x_collocation):
            # Reconstruct models from graphdefs and parameters
            models = tuple(nnx.merge(g, p) for g, p in zip(self.graphdefs, params_tuple))

            # Compute PDE losses for each domain
            pde_losses = tuple(pde_bound(m, x_d) for m, x_d in zip(models, x_collocation))

            # Compute boundary condition losses
            loss_bc_l = bc_left_fn(models[0])
            loss_bc_r = bc_right_fn(models[-1])

            # Compute interface losses using extracted function
            interface_losses = compute_interface_loss(models, self.interfaces, self.all_lambdas)

            # Collect all raw losses
            all_losses = {
                'pde': dict(zip([f'pde_{i}' for i in range(self.n_domains)], pde_losses)),
                'bc': {'bc_left': loss_bc_l, 'bc_right': loss_bc_r},
                'interface': dict(zip([f'interface_{i}' for i in range(len(self.interfaces))], interface_losses))
            }
            
            # Compute weights using the strategy
            weights = self.weight_strategy.compute_weights(all_losses, step=0)
            
            # Aggregate weighted losses
            total = 0.0
            for domain, domain_weights in weights.items():
                for loss_name, weight in domain_weights.items():
                    if domain == 'pde':
                        total += weight * all_losses['pde'][loss_name]
                    elif domain == 'bc':
                        total += weight * all_losses['bc'][loss_name]
                    elif domain == 'interface':
                        total += weight * all_losses['interface'][loss_name]

            # Return total loss and individual losses for logging
            return total, (*pde_losses, loss_bc_l, loss_bc_r, *interface_losses)

        return total_loss

    @partial(jit, static_argnames=['self', 'txs'])
    def train_step(
        self, 
        params: Tuple, 
        x_collocation: Tuple, 
        txs: Tuple, 
        opt_states: Tuple, 
        loss_fn: Callable
    ):
        """
        Perform a single training step for all domains.
        
        Args:
            params: Tuple of parameters for each PINN
            x_collocation: Tuple of collocation points for each domain
            txs: Tuple of optimizers for each PINN
            opt_states: Tuple of optimizer states for each PINN
            loss_fn: Loss function created by create_loss_fn
            
        Returns:
            Tuple of (new_params, new_opt_states, total_loss, aux_losses)
        """
        def closure(p):
            return loss_fn(p, x_collocation)

        (total, aux), grads = value_and_grad(closure, has_aux=True)(params)

        # Update each domain's parameters independently
        new_params = []
        new_opt_states = []
        for p, g, tx, os in zip(params, grads, txs, opt_states):
            updates, new_os = tx.update(g, os)
            new_params.append(optax.apply_updates(p, updates))
            new_opt_states.append(new_os)

        return tuple(new_params), tuple(new_opt_states), total, aux

    def train_loop(
        self, 
        num_steps: int, 
        loss_fn: Callable, 
        loss_names: Tuple[str, ...], 
        log_interval: int = 100
    ):
        """
        Training loop for MPINN.
        
        Args:
            num_steps: Number of training steps
            loss_fn: Loss function
            loss_names: Names of loss components for logging
            log_interval: Frequency of logging
            
        Returns:
            Dictionary containing training history
        """
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

    def fit(
        self, 
        pde_fn: Callable, 
        bc_left_fn: Callable, 
        bc_right_fn: Callable, 
        phys: Any, 
        epochs: int, 
        log_interval: int = 100
    ):
        """
        Train the MPINN model.
        
        Args:
            pde_fn: PDE residual function
            bc_left_fn: Left boundary condition function
            bc_right_fn: Right boundary condition function
            phys: PhysicsParams object
            epochs: Number of training epochs
            log_interval: Frequency of logging
            
        Returns:
            Tuple of (history_dict, training_time)
        """
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
        """
        Predict temperature values for given x coordinates.
        
        Args:
            x_test: Input x coordinates (array-like)
            
        Returns:
            Predicted temperature values
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
        Compute error metrics between predicted and exact solutions.
        
        Delegates to PINN.compute_metrics for consistency.
        
        Args:
            x_test: Input x coordinates
            t_pred: Predicted temperatures
            t_exact: Exact temperatures
            
        Returns:
            Dictionary of error metrics (MAPE, MAE, MSE, RMSE, max_error)
        """
        # Use the first PINN instance's compute_metrics for consistency
        return self.pinn_instances[0].compute_metrics(x_test, t_pred, t_exact)

    def evaluate(self, x_test, exact_fn, phys, bc_names=None):
        """
        Evaluate the model against an exact solution.
        
        Args:
            x_test: Test x coordinates
            exact_fn: Function computing exact solution
            phys: PhysicsParams object
            bc_names: Optional boundary condition names
            
        Returns:
            Tuple of (metrics_dict, predicted_values, exact_values)
        """
        t_pred = self.predict(x_test)
        t_exact = exact_fn(x_test.ravel(), phys)
        metrics = self.compute_metrics(x_test, t_pred, t_exact.reshape(-1, 1))
        if bc_names:
            metrics['bc_left'] = bc_names[0]
            metrics['bc_right'] = bc_names[1] if len(bc_names) > 1 else bc_names[0]
        return metrics, t_pred, t_exact

    def save_plot(
        self, 
        x_test, 
        t_pred, 
        t_exact, 
        phys, 
        save_path, 
        title="Сравнение MPINN и точного решения"
    ):
        """
        Save a plot comparing predicted and exact solutions.
        
        Uses the unified plotting module with interface markers.
        
        Args:
            x_test: Test x coordinates
            t_pred: Predicted temperatures
            t_exact: Exact temperatures
            phys: PhysicsParams object
            save_path: Path to save the figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), t_exact, 'b-', label='Аналитическое решение', linewidth=2)
        ax.plot(x_test.ravel(), t_pred, 'r:', label='ФИНС', linewidth=6)
        
        # Mark interfaces
        for x_int in phys.interfaces:
            ax.axvline(x=x_int, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('x, м')
        ax.set_ylabel('T, К')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=72, bbox_inches='tight')
        plt.close()

    def show_plot(self, x_test, t_pred, t_exact, phys, title="Сравнение MPINN и точного решения"):
        """
        Display a plot comparing predicted and exact solutions.
        
        Uses the unified plotting module with interface markers.
        
        Args:
            x_test: Test x coordinates
            t_pred: Predicted temperatures
            t_exact: Exact temperatures
            phys: PhysicsParams object
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_test.ravel(), t_exact, 'b-', label='Аналитическое решение', linewidth=2)
        ax.plot(x_test.ravel(), t_pred, 'r:', label='ФИНС', linewidth=6)
        
        # Mark interfaces
        for x_int in phys.interfaces:
            ax.axvline(x=x_int, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('x, м')
        ax.set_ylabel('T, К')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()