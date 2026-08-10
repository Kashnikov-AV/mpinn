"""Tests for integration of multiple modules (PINN training loop)."""
import jax
import jax.numpy as jnp
import pytest

from mpinn.geom import Interval
from mpinn.pde import line_1d
from mpinn.bc import dirichlet_bc
from mpinn.pinn_core import FCNet, normalize, denormalize


class TestPINNIntegration:
    """Integration tests for complete PINN workflow."""

    def test_pinn_training_step(self, seed_key):
        """Test a single training step of PINN."""
        # Setup geometry
        geom = Interval(0.0, 1.0)
        x_collocation = geom.random_points(seed_key, num_points=50)
        x_bc_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_bc_right = jnp.array([[1.0]], dtype=jnp.float32)
        
        # Setup network
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[20, 20],
            activation='tanh'
        )
        
        variables = net.init(seed_key, x_collocation)
        
        # Define loss function
        def compute_loss(vars, x_col, x_left, x_right):
            # PDE residual loss
            pde_residual = line_1d(lambda x: net.apply(vars, x), x_col)
            pde_loss = jnp.mean(pde_residual ** 2)
            
            # BC losses
            bc_left_loss = dirichlet_bc(lambda x: net.apply(vars, x), x_left, 0.0)
            bc_right_loss = dirichlet_bc(lambda x: net.apply(vars, x), x_right, 1.0)
            
            total_loss = pde_loss + bc_left_loss + bc_right_loss
            return total_loss
        
        # Compute initial loss
        initial_loss = compute_loss(variables, x_collocation, x_bc_left, x_bc_right)
        
        assert jnp.isfinite(initial_loss), "Initial loss should be finite"
        assert initial_loss >= 0, "Loss should be non-negative"

    def test_pinn_gradient_descent_step(self, seed_key):
        """Test that gradients can be computed and applied."""
        geom = Interval(0.0, 1.0)
        x_collocation = geom.random_points(seed_key, num_points=30)
        x_bc_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_bc_right = jnp.array([[1.0]], dtype=jnp.float32)
        
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[16, 16],
            activation='tanh'
        )
        
        variables = net.init(seed_key, x_collocation)
        
        def loss_fn(vars):
            pde_residual = line_1d(lambda x: net.apply(vars, x), x_collocation)
            pde_loss = jnp.mean(pde_residual ** 2)
            
            bc_left_loss = dirichlet_bc(lambda x: net.apply(vars, x), x_bc_left, 0.0)
            bc_right_loss = dirichlet_bc(lambda x: net.apply(vars, x), x_bc_right, 1.0)
            
            return pde_loss + bc_left_loss + bc_right_loss
        
        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(variables)
        
        # Check gradient structure
        assert 'params' in grads
        assert len(grads['params']) > 0
        
        # Verify gradients are finite
        for layer_name, layer_grad in grads['params'].items():
            for param_name, param_grad in layer_grad.items():
                assert jnp.all(jnp.isfinite(param_grad)), f"Gradients should be finite for {layer_name}/{param_name}"

    def test_normalization_integration(self, seed_key):
        """Test normalization with network forward pass."""
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[10, 10],
            activation='tanh'
        )
        
        # Original data in [0, 10]
        x_original = jnp.linspace(0.0, 10.0, 20).reshape(-1, 1)
        
        # Normalize to [0, 1]
        x_norm = normalize(x_original, 0.0, 10.0)
        
        # Forward pass
        variables = net.init(seed_key, x_norm)
        y_norm = net.apply(variables, x_norm)
        
        # Denormalize output (assuming output also in [0, 10])
        y_original = denormalize(y_norm, 0.0, 10.0)
        
        assert y_original.shape == y_norm.shape
        assert jnp.all(jnp.isfinite(y_original))

    def test_jit_compiled_training(self, seed_key):
        """Test JIT-compiled training step."""
        geom = Interval(0.0, 1.0)
        x_collocation = geom.random_points(seed_key, num_points=20)
        x_bc_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_bc_right = jnp.array([[1.0]], dtype=jnp.float32)
        
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[12, 12],
            activation='tanh'
        )
        
        variables = net.init(seed_key, x_collocation)
        
        def loss_fn(vars):
            pde_residual = line_1d(lambda x: net.apply(vars, x), x_collocation)
            pde_loss = jnp.mean(pde_residual ** 2)
            bc_left_loss = dirichlet_bc(lambda x: net.apply(vars, x), x_bc_left, 0.0)
            bc_right_loss = dirichlet_bc(lambda x: net.apply(vars, x), x_bc_right, 1.0)
            return pde_loss + bc_left_loss + bc_right_loss
        
        # JIT compile
        jit_loss_fn = jax.jit(loss_fn)
        jit_grad_fn = jax.jit(jax.grad(loss_fn))
        
        # Compare JIT vs non-JIT
        loss_normal = loss_fn(variables)
        loss_jit = jit_loss_fn(variables)
        
        grads_normal = jax.grad(loss_fn)(variables)
        grads_jit = jit_grad_fn(variables)
        
        assert jnp.isclose(loss_normal, loss_jit, atol=1e-6), "JIT loss should match"
        
        # Check gradients match
        for layer_name in grads_normal['params']:
            for param_name in grads_normal['params'][layer_name]:
                normal_grad = grads_normal['params'][layer_name][param_name]
                jit_grad = grads_jit['params'][layer_name][param_name]
                assert jnp.allclose(normal_grad, jit_grad, atol=1e-6), \
                    f"JIT gradients should match for {layer_name}/{param_name}"
