"""Tests for PINN core module (FCNet, normalize, denormalize)."""
import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from mpinn.pinn_core import FCNet, normalize, denormalize


class TestNormalize:
    """Test data normalization functions."""

    def test_normalize_output_shape(self):
        """Check that normalize preserves shape."""
        x = jnp.linspace(0.0, 10.0, 20).reshape(-1, 1)
        x_min, x_max = 0.0, 10.0
        
        x_norm = normalize(x, x_min, x_max)
        
        assert x_norm.shape == x.shape

    def test_normalize_range(self):
        """Check that normalized data is in [0, 1]."""
        x = jnp.array([[0.0], [5.0], [10.0]], dtype=jnp.float32)
        x_min, x_max = 0.0, 10.0
        
        x_norm = normalize(x, x_min, x_max)
        
        assert jnp.all(x_norm >= 0.0 - 1e-6)
        assert jnp.all(x_norm <= 1.0 + 1e-6)

    def test_normalize_dtype(self):
        """Check dtype preservation."""
        x = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
        
        x_norm = normalize(x, 0.0, 10.0)
        
        assert x_norm.dtype == jnp.float32

    def test_denormalize_inverse(self):
        """Check that denormalize is inverse of normalize."""
        x = jnp.array([[2.5], [5.0], [7.5]], dtype=jnp.float32)
        x_min, x_max = 0.0, 10.0
        
        x_norm = normalize(x, x_min, x_max)
        x_denorm = denormalize(x_norm, x_min, x_max)
        
        assert jnp.allclose(x, x_denorm, atol=1e-6)


class TestFCNet:
    """Test fully connected neural network."""

    def test_fcnet_output_shape(self, seed_key):
        """Check network output shape matches expected."""
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[32, 32, 32],
            activation='tanh'
        )
        
        x = jnp.ones((10, 1), dtype=jnp.float32)
        variables = net.init(seed_key, x)
        y = net.apply(variables, x)
        
        assert y.shape == (10, 1), f"Expected (10, 1), got {y.shape}"

    def test_fcnet_dtype(self, seed_key):
        """Check network output dtype."""
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[16, 16],
            activation='relu'
        )
        
        x = jnp.ones((5, 1), dtype=jnp.float32)
        variables = net.init(seed_key, x)
        y = net.apply(variables, x)
        
        assert y.dtype == jnp.float32

    def test_fcnet_deterministic(self, seed_key):
        """Check network is deterministic with same weights."""
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[20, 20],
            activation='tanh'
        )
        
        x = jnp.linspace(0.0, 1.0, 8).reshape(-1, 1)
        variables = net.init(seed_key, x)
        
        y1 = net.apply(variables, x)
        y2 = net.apply(variables, x)
        
        assert jnp.allclose(y1, y2), "Network should be deterministic"

    def test_fcnet_different_activations(self, seed_key):
        """Test different activation functions."""
        activations = ['tanh', 'relu', 'sigmoid']
        
        for act in activations:
            net = FCNet(
                input_dim=1,
                output_dim=1,
                hidden_layers=[10, 10],
                activation=act
            )
            
            x = jnp.ones((4, 1), dtype=jnp.float32)
            variables = net.init(seed_key, x)
            y = net.apply(variables, x)
            
            assert y.shape == (4, 1), f"Failed for activation {act}"

    def test_fcnet_jit_compatible(self, seed_key):
        """Check network is compatible with JIT compilation."""
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[16, 16],
            activation='tanh'
        )
        
        x = jnp.ones((5, 1), dtype=jnp.float32)
        variables = net.init(seed_key, x)
        
        # JIT compile the apply function
        jit_apply = jax.jit(net.apply)
        y_jit = jit_apply(variables, x)
        
        # Compare with non-JIT
        y_normal = net.apply(variables, x)
        
        assert jnp.allclose(y_jit, y_normal, atol=1e-6), "JIT should produce same results"


class TestFCNetGradients:
    """Test gradient computation for PINN."""

    def test_gradient_computation(self, seed_key):
        """Check that gradients can be computed."""
        net = FCNet(
            input_dim=1,
            output_dim=1,
            hidden_layers=[20, 20],
            activation='tanh'
        )
        
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        variables = net.init(seed_key, x)
        
        def loss_fn(vars, x_batch):
            y = net.apply(vars, x_batch)
            return jnp.mean(y ** 2)
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(variables, x)
        
        # Check that gradients exist and have correct structure
        assert 'params' in grads
        assert len(grads['params']) > 0
