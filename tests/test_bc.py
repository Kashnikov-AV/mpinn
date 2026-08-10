"""Tests for boundary conditions module."""
import jax
import jax.numpy as jnp
import pytest

from mpinn.bc import dirichlet_bc, neuman_bc, robin_bc


class TestDirichletBC:
    """Test cases for Dirichlet boundary conditions."""

    def test_dirichlet_exact_value(self, boundary_points):
        """Check that bc(x_boundary) equals the target value."""
        x_left = boundary_points["left"]
        x_right = boundary_points["right"]
        
        target_value = 5.0
        
        # Create a dummy network output that matches target at boundaries
        # For Dirichlet: loss = mean((u(x) - g)^2), should be 0 if u(x) == g
        def dummy_net(x):
            return jnp.full((x.shape[0], 1), target_value)
        
        loss_left = dirichlet_bc(dummy_net, x_left, target_value)
        loss_right = dirichlet_bc(dummy_net, x_right, target_value)
        
        # Loss should be very close to 0 when BC is satisfied
        assert jnp.isclose(loss_left, 0.0, atol=1e-6), f"Expected ~0, got {loss_left}"
        assert jnp.isclose(loss_right, 0.0, atol=1e-6), f"Expected ~0, got {loss_right}"

    def test_dirichlet_nonzero_loss(self, boundary_points):
        """Check that loss is non-zero when BC is not satisfied."""
        x_left = boundary_points["left"]
        
        target_value = 5.0
        
        def wrong_net(x):
            return jnp.full((x.shape[0], 1), 0.0)  # Returns 0 instead of 5
        
        loss = dirichlet_bc(wrong_net, x_left, target_value)
        
        # Loss should be (0 - 5)^2 = 25 (mean over points)
        expected_loss = 25.0
        assert jnp.isclose(loss, expected_loss, atol=1e-5), f"Expected {expected_loss}, got {loss}"

    def test_dirichlet_output_shape(self, boundary_points):
        """Check that loss is a scalar (due to mean)."""
        x_left = boundary_points["left"]
        target_value = 1.0
        
        def dummy_net(x):
            return jnp.full((x.shape[0], 1), target_value)
        
        loss = dirichlet_bc(dummy_net, x_left, target_value)
        
        # Should be scalar due to jnp.mean
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


class TestNeumanBC:
    """Test cases for Neumann boundary conditions."""

    def test_neuman_exact_derivative(self, boundary_points):
        """Check that derivative at boundary equals target derivative."""
        x_right = boundary_points["right"]
        
        target_derivative = 3.0
        
        # Network u(x) = target_derivative * x => du/dx = target_derivative
        def linear_net(x):
            return target_derivative * x
        
        loss = neuman_bc(linear_net, x_right, target_derivative)
        
        # Loss should be ~0 when derivative matches
        assert jnp.isclose(loss, 0.0, atol=1e-5), f"Expected ~0, got {loss}"

    def test_neuman_nonzero_loss(self, boundary_points):
        """Check non-zero loss when derivative doesn't match."""
        x_right = boundary_points["right"]
        
        target_derivative = 5.0
        
        # Network with zero derivative (constant)
        def constant_net(x):
            return jnp.ones_like(x)
        
        loss = neuman_bc(constant_net, x_right, target_derivative)
        
        # du/dx = 0, target = 5, loss = (0 - 5)^2 = 25
        expected_loss = 25.0
        assert jnp.isclose(loss, expected_loss, atol=1e-5), f"Expected {expected_loss}, got {loss}"

    def test_neuman_output_shape(self, boundary_points):
        """Check that loss is a scalar."""
        x_right = boundary_points["right"]
        target_derivative = 1.0
        
        def dummy_net(x):
            return x
        
        loss = neuman_bc(dummy_net, x_right, target_derivative)
        
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


class TestRobinBC:
    """Test cases for Robin boundary conditions."""

    def test_robin_exact_condition(self, boundary_points):
        """Check Robin BC: a*u + b*du/dn = g is satisfied."""
        x_right = boundary_points["right"]
        
        a_coeff = 2.0
        b_coeff = 3.0
        g_value = 10.0
        
        # Design a function that satisfies: a*u + b*du/dx = g
        # If u = c (constant), then du/dx = 0, so a*c = g => c = g/a
        constant_value = g_value / a_coeff
        
        def constant_net(x):
            return jnp.full_like(x, constant_value)
        
        loss = robin_bc(constant_net, x_right, a_coeff, b_coeff, g_value)
        
        # Should satisfy BC exactly
        assert jnp.isclose(loss, 0.0, atol=1e-5), f"Expected ~0, got {loss}"

    def test_robin_output_shape(self, boundary_points):
        """Check that loss is a scalar."""
        x_right = boundary_points["right"]
        
        def dummy_net(x):
            return x
        
        loss = robin_bc(dummy_net, x_right, 1.0, 1.0, 1.0)
        
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"


class TestBCDeterminism:
    """Test determinism of BC computations."""

    def test_dirichlet_deterministic(self, boundary_points):
        """Check that same input produces same loss."""
        x_left = boundary_points["left"]
        target_value = 42.0
        
        def dummy_net(x):
            return jnp.full((x.shape[0], 1), 10.0)
        
        loss1 = dirichlet_bc(dummy_net, x_left, target_value)
        loss2 = dirichlet_bc(dummy_net, x_left, target_value)
        
        assert jnp.allclose(loss1, loss2), "BC loss should be deterministic"
