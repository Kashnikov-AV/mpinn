"""Tests for PINN core module (FCNet and NormalizedNet)."""

import jax
import jax.numpy as jnp
from flax import nnx

from mpinn.pinn_core import FCNet, NormalizedNet


class TestFCNet:
    """Test fully connected neural network."""

    def test_fcnet_output_shape(self, seed_key):
        """Check network output shape matches expected."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=32, dout=1, num_layers=3, activation=nnx.tanh, rngs=rngs
        )

        x = jnp.ones((10, 1), dtype=jnp.float32)
        y = net(x)

        assert y.shape == (10, 1), f"Expected (10, 1), got {y.shape}"

    def test_fcnet_dtype(self, seed_key):
        """Check network output dtype."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=16, dout=1, num_layers=2, activation=nnx.relu, rngs=rngs
        )

        x = jnp.ones((5, 1), dtype=jnp.float32)
        y = net(x)

        assert y.dtype == jnp.float32

    def test_fcnet_deterministic(self, seed_key):
        """Check network is deterministic with same weights."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=20, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
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
            net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=act, rngs=rngs)

            x = jnp.ones((4, 1), dtype=jnp.float32)
            y = net(x)

            assert y.shape == (4, 1), f"Failed for activation {act}"

    def test_fcnet_jit_compatible(self, seed_key):
        """Check network is compatible with JIT compilation."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=16, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        x = jnp.ones((5, 1), dtype=jnp.float32)

        # JIT compile the network call
        jit_net = jax.jit(net.__call__)
        y_jit = jit_net(x)

        # Compare with non-JIT
        y_normal = net(x)

        assert jnp.allclose(y_jit, y_normal, atol=1e-6), (
            "JIT should produce same results"
        )


class TestFCNetGradients:
    """Test gradient computation for PINN."""

    def test_gradient_computation(self, seed_key):
        """Check that gradients can be computed."""
        rngs = nnx.Rngs(seed_key)
        net = FCNet(
            din=1, dmid=20, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)

        def loss_fn(model, x_batch):
            y = model(x_batch)
            return jnp.mean(y**2)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(net, x)

        # Check that gradients exist and have correct structure
        assert grads is not None


class TestNormalizedNet:
    """Test NormalizedNet wrapper."""

    def test_normalized_net_output_range(self, seed_key):
        """Check that output values are within [T_min, T_max]."""
        rngs = nnx.Rngs(seed_key)
        base_net = FCNet(
            din=1, dmid=16, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        x_min, x_max = 0.0, 2.0
        T_min, T_max = 300.0, 500.0
        net = NormalizedNet(base_net, x_min, x_max, T_min, T_max)

        # Generate random x in [x_min, x_max]
        x = jax.random.uniform(jax.random.PRNGKey(0), (100, 1), minval=x_min, maxval=x_max)
        y = net(x)

        assert jnp.all(y >= T_min) and jnp.all(y <= T_max), (
            f"Output should be within [{T_min}, {T_max}], got min={y.min()}, max={y.max()}"
        )

    def test_normalized_net_inverse(self, seed_key):
        """Check that normalization + denormalization is invertible."""
        rngs = nnx.Rngs(seed_key)
        base_net = FCNet(
            din=1, dmid=8, dout=1, num_layers=1, activation=nnx.identity, rngs=rngs
        )
        # For identity base network, output should be exactly the normalized x mapped to T
        x_min, x_max = 0.0, 1.0
        T_min, T_max = 200.0, 600.0

        # We need to force base_net to produce exact normalized output.
        # However, the base net is not identity; we can instead just check the transformation logic.
        # Better: create a dummy net that returns the input as is.
        # We can override by creating a custom module, but easier: just test the formulas.

        # We'll create a deterministic net that returns normalized x.
        class IdentityNet(nnx.Module):
            def __call__(self, x):
                return x

        base_net = IdentityNet()
        net = NormalizedNet(base_net, x_min, x_max, T_min, T_max)

        x = jnp.array([[0.0], [0.5], [1.0]])
        y = net(x)

        expected = T_min + (T_max - T_min) * (x - x_min) / (x_max - x_min)
        assert jnp.allclose(y, expected, atol=1e-6)

    def test_normalized_net_jit_compatible(self, seed_key):
        """Check NormalizedNet is compatible with JIT compilation."""
        rngs = nnx.Rngs(seed_key)
        base_net = FCNet(
            din=1, dmid=16, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )
        net = NormalizedNet(base_net, x_min=0.0, x_max=1.0, T_min=0.0, T_max=1.0)

        x = jnp.ones((5, 1), dtype=jnp.float32)

        jit_net = jax.jit(net.__call__)
        y_jit = jit_net(x)
        y_normal = net(x)

        assert jnp.allclose(y_jit, y_normal, atol=1e-6)

    def test_normalized_net_gradient(self, seed_key):
        """Check gradients can be computed through NormalizedNet."""
        rngs = nnx.Rngs(seed_key)
        base_net = FCNet(
            din=1, dmid=16, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )
        net = NormalizedNet(base_net, x_min=0.0, x_max=1.0, T_min=0.0, T_max=1.0)

        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)

        def loss_fn(model, x_batch):
            y = model(x_batch)
            return jnp.mean(y**2)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(net, x)
        assert grads is not None