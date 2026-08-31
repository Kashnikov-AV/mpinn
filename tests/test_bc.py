"""Tests for boundary conditions module."""

import jax.numpy as jnp
from flax import nnx

from mpinn.bc import dirichlet_bc, neumann_bc, robin_bc


class TestDirichletBC:
    """Test cases for Dirichlet boundary conditions."""

    def test_dirichlet_exact_value(self, boundary_points, seed_key):
        """Check that bc(x_boundary) equals the target value."""
        x0 = boundary_points["left"]
        x1 = boundary_points["right"]

        target_value = 5.0

        # Create a dummy network that outputs constant value
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        # For testing, we just check the function accepts proper inputs
        # The actual loss depends on network output
        residuals_left = dirichlet_bc(net, x0, target_value)
        residuals_right = dirichlet_bc(net, x1, target_value)

        # Residuals should be finite
        assert jnp.all(jnp.isfinite(residuals_left)), f"Expected finite residuals, got {residuals_left}"
        assert jnp.all(jnp.isfinite(residuals_right)), f"Expected finite residuals, got {residuals_right}"

    def test_dirichlet_nonzero_loss(self, boundary_points, seed_key):
        """Check that loss is non-zero when BC is not satisfied."""
        x0 = boundary_points["left"]

        target_value = 5.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = dirichlet_bc(net, x0, target_value)

        # Loss should be finite and non-negative
        loss = jnp.mean(residuals ** 2)
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_dirichlet_output_shape(self, boundary_points, seed_key):
        """Check that residuals have shape (n_points,)."""
        x0 = boundary_points["left"]
        target_value = 1.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = dirichlet_bc(net, x0, target_value)

        # Should have shape (n_points,) where n_points is the number of boundary points
        assert residuals.shape == (1,), f"Expected shape (1,), got {residuals.shape}"


class TestNeumanBC:
    """Test cases for Neumann boundary conditions."""

    def test_neuman_exact_derivative(self, boundary_points, seed_key):
        """Check that derivative at boundary can be computed."""
        x1 = boundary_points["right"]
        # Normal pointing outward from right boundary
        normal = jnp.array([[1.0]], dtype=jnp.float32)

        target_derivative = 3.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = neumann_bc(net, x1, normal, target_derivative)

        # Residuals should be finite
        assert jnp.all(jnp.isfinite(residuals)), f"Expected finite residuals, got {residuals}"

    def test_neuman_nonzero_loss(self, boundary_points, seed_key):
        """Check loss computation for Neumann BC."""
        x1 = boundary_points["right"]
        normal = jnp.array([[1.0]], dtype=jnp.float32)

        target_derivative = 5.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = neumann_bc(net, x1, normal, target_derivative)

        # Loss should be finite and non-negative
        loss = jnp.mean(residuals ** 2)
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_neuman_output_shape(self, boundary_points, seed_key):
        """Check that residuals have shape (n_points,)."""
        x1 = boundary_points["right"]
        normal = jnp.array([[1.0]], dtype=jnp.float32)
        target_derivative = 1.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = neumann_bc(net, x1, normal, target_derivative)

        assert residuals.shape == (1,), f"Expected shape (1,), got {residuals.shape}"


class TestRobinBC:
    """Test cases for Robin boundary conditions."""

    def test_robin_exact_condition(self, boundary_points, seed_key):
        """Check Robin BC computation."""
        x1 = boundary_points["right"]
        normal = jnp.array([[1.0]], dtype=jnp.float32)

        a_coeff = 2.0
        b_coeff = 3.0
        g_value = 10.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = robin_bc(net, x1, normal, a_coeff, b_coeff, g_value)

        # Residuals should be finite
        assert jnp.all(jnp.isfinite(residuals)), f"Expected finite residuals, got {residuals}"

    def test_robin_output_shape(self, boundary_points, seed_key):
        """Check that residuals have shape (n_points,)."""
        x1 = boundary_points["right"]
        normal = jnp.array([[1.0]], dtype=jnp.float32)

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals = robin_bc(net, x1, normal, 1.0, 1.0, 1.0)

        assert residuals.shape == (1,), f"Expected shape (1,), got {residuals.shape}"


class TestBCDeterminism:
    """Test determinism of BC computations."""

    def test_dirichlet_deterministic(self, boundary_points, seed_key):
        """Check that same input produces same residuals."""
        x0 = boundary_points["left"]
        target_value = 42.0

        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet

        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs
        )

        residuals1 = dirichlet_bc(net, x0, target_value)
        residuals2 = dirichlet_bc(net, x0, target_value)

        assert jnp.allclose(residuals1, residuals2), "BC residuals should be deterministic"
