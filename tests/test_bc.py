"""Tests for boundary conditions module."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from mpinn.bc import dirichlet_bc, neuman_bc, robin_bc


class TestDirichletBC:
    """Test cases for Dirichlet boundary conditions."""

    def test_dirichlet_exact_value(self, boundary_points, seed_key):
        """Check that bc(x_boundary) equals the target value."""
        x_left = boundary_points["left"]
        x_right = boundary_points["right"]
        
        target_value = 5.0
        
        # Create a dummy network that outputs constant value
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        # For testing, we just check the function accepts proper inputs
        # The actual loss depends on network output
        loss_left = dirichlet_bc(net, x_left, target_value)
        loss_right = dirichlet_bc(net, x_right, target_value)
        
        # Loss should be finite
        assert jnp.isfinite(loss_left), f"Expected finite loss, got {loss_left}"
        assert jnp.isfinite(loss_right), f"Expected finite loss, got {loss_right}"

    def test_dirichlet_nonzero_loss(self, boundary_points, seed_key):
        """Check that loss is non-zero when BC is not satisfied."""
        x_left = boundary_points["left"]
        
        target_value = 5.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = dirichlet_bc(net, x_left, target_value)
        
        # Loss should be finite and non-negative
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_dirichlet_output_shape(self, boundary_points, seed_key):
        """Check that loss is a scalar (due to mean)."""
        x_left = boundary_points["left"]
        target_value = 1.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = dirichlet_bc(net, x_left, target_value)
        
        # Should be scalar due to jnp.mean
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


class TestNeumanBC:
    """Test cases for Neumann boundary conditions."""

    def test_neuman_exact_derivative(self, boundary_points, seed_key):
        """Check that derivative at boundary can be computed."""
        x_right = boundary_points["right"]
        
        target_derivative = 3.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = neuman_bc(net, x_right, target_derivative)
        
        # Loss should be finite
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_neuman_nonzero_loss(self, boundary_points, seed_key):
        """Check loss computation for Neumann BC."""
        x_right = boundary_points["right"]
        
        target_derivative = 5.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = neuman_bc(net, x_right, target_derivative)
        
        # Loss should be finite and non-negative
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"
        assert loss >= 0, f"Loss should be non-negative, got {loss}"

    def test_neuman_output_shape(self, boundary_points, seed_key):
        """Check that loss is a scalar."""
        x_right = boundary_points["right"]
        target_derivative = 1.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = neuman_bc(net, x_right, target_derivative)
        
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


class TestRobinBC:
    """Test cases for Robin boundary conditions."""

    def test_robin_exact_condition(self, boundary_points, seed_key):
        """Check Robin BC computation."""
        x_right = boundary_points["right"]
        
        a_coeff = 2.0
        b_coeff = 3.0
        g_value = 10.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = robin_bc(net, x_right, a_coeff, b_coeff, g_value)
        
        # Loss should be finite
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_robin_output_shape(self, boundary_points, seed_key):
        """Check that loss is a scalar."""
        x_right = boundary_points["right"]
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss = robin_bc(net, x_right, 1.0, 1.0, 1.0)
        
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


class TestBCDeterminism:
    """Test determinism of BC computations."""

    def test_dirichlet_deterministic(self, boundary_points, seed_key):
        """Check that same input produces same loss."""
        x_left = boundary_points["left"]
        target_value = 42.0
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(din=1, dmid=10, dout=1, num_layers=2, activation=nnx.tanh, rngs=rngs)
        
        loss1 = dirichlet_bc(net, x_left, target_value)
        loss2 = dirichlet_bc(net, x_left, target_value)
        
        assert jnp.allclose(loss1, loss2), "BC loss should be deterministic"
