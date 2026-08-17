"""Tests for PINN core module (FCNet, normalize, denormalize)."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

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
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1,
            dmid=32,
            dout=1,
            num_layers=3,
            activation=nnx.tanh,
            rngs=rngs
        )
        
        x = jnp.ones((10, 1), dtype=jnp.float32)
        y = net(x)
        
        assert y.shape == (10, 1), f"Expected (10, 1), got {y.shape}"

    def test_fcnet_dtype(self, seed_key):
        """Check network output dtype."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1,
            dmid=16,
            dout=1,
            num_layers=2,
            activation=nnx.relu,
            rngs=rngs
        )
        
        x = jnp.ones((5, 1), dtype=jnp.float32)
        y = net(x)
        
        assert y.dtype == jnp.float32

    def test_fcnet_deterministic(self, seed_key):
        """Check network is deterministic with same weights."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1,
            dmid=20,
            dout=1,
            num_layers=2,
            activation=nnx.tanh,
            rngs=rngs
        )
        
        x = jnp.linspace(0.0, 1.0, 8).reshape(-1, 1)
        
        y1 = net(x)
        y2 = net(x)
        
        assert jnp.allclose(y1, y2), "Network should be deterministic"

    def test_fcnet_different_activations(self, seed_key):
        """Test different activation functions."""
        activations = [nnx.tanh, nnx.relu, nnx.sigmoid]
        
        for act in activations:
            rngs = nnx.Rngs(seed_key)
            net = FCNet(
                din=1,
                dmid=10,
                dout=1,
                num_layers=2,
                activation=act,
                rngs=rngs
            )
            
            x = jnp.ones((4, 1), dtype=jnp.float32)
            y = net(x)
            
            assert y.shape == (4, 1), f"Failed for activation {act}"

    def test_fcnet_jit_compatible(self, seed_key):
        """Check network is compatible with JIT compilation."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1,
            dmid=16,
            dout=1,
            num_layers=2,
            activation=nnx.tanh,
            rngs=rngs
        )
        
        x = jnp.ones((5, 1), dtype=jnp.float32)
        
        # JIT compile the network call
        jit_net = jax.jit(net.__call__)
        y_jit = jit_net(x)
        
        # Compare with non-JIT
        y_normal = net(x)
        
        assert jnp.allclose(y_jit, y_normal, atol=1e-6), "JIT should produce same results"


class TestFCNetGradients:
    """Test gradient computation for PINN."""

    def test_gradient_computation(self, seed_key):
        """Check that gradients can be computed."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1,
            dmid=20,
            dout=1,
            num_layers=2,
            activation=nnx.tanh,
            rngs=rngs
        )
        
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
        
        def loss_fn(model, x_batch):
            y = model(x_batch)
            return jnp.mean(y ** 2)
        
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(net, x)
        
        # Check that gradients exist and have correct structure
        assert grads is not None
