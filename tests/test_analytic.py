"""Tests for analytic solutions module."""
import jax
import jax.numpy as jnp
import pytest

from mpinn.analytic import (
    line_1d_dirichlet_exact, cylinder_1d_dirichlet_exact, sphere_1d_dirichlet_exact
)


class DummyPhys:
    """Dummy physics object for testing."""
    def __init__(self, x_left=0.0, x_right=1.0, T_left=0.0, T_right=1.0):
        self.x_left = x_left
        self.x_right = x_right
        self.T_left = T_left
        self.T_right = T_right


class TestAnalyticLine1D:
    """Test analytic solution for 1D line geometry."""

    def test_line_solution_shape(self):
        """Check that solution has correct shape."""
        x = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        phys = DummyPhys(x_left=0.0, x_right=1.0, T_left=0.0, T_right=1.0)
        
        u = line_1d_dirichlet_exact(x, phys)
        
        assert u.shape[0] == x.shape[0], f"Expected {x.shape[0]}, got {u.shape[0]}"

    def test_line_solution_boundary_values(self):
        """Check that solution satisfies boundary conditions."""
        phys = DummyPhys(x_left=0.0, x_right=1.0, T_left=5.0, T_right=-3.0)
        
        x_left = jnp.array([[phys.x_left]], dtype=jnp.float32)
        x_right = jnp.array([[phys.x_right]], dtype=jnp.float32)
        
        u_left = line_1d_dirichlet_exact(x_left, phys)
        u_right = line_1d_dirichlet_exact(x_right, phys)
        
        assert jnp.isclose(u_left[0], phys.T_left, atol=1e-6), f"Left BC not satisfied"
        assert jnp.isclose(u_right[0], phys.T_right, atol=1e-6), f"Right BC not satisfied"

    def test_line_solution_dtype(self):
        """Check dtype of analytic solution."""
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1).astype(jnp.float32)
        phys = DummyPhys()
        
        u = line_1d_dirichlet_exact(x, phys)
        
        assert u.dtype == jnp.float32


class TestAnalyticCylinder1D:
    """Test analytic solution for 1D cylindrical geometry."""

    def test_cylinder_solution_shape(self):
        """Check solution shape for cylindrical case."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)  # Avoid r=0
        phys = DummyPhys(x_left=0.1, x_right=1.0, T_left=1.0, T_right=0.0)
        
        u = cylinder_1d_dirichlet_exact(x, phys)
        
        assert u.shape[0] == x.shape[0]

    def test_cylinder_solution_boundary_values(self):
        """Check BCs for cylindrical solution."""
        r_inner = 0.5
        r_outer = 2.0
        
        phys = DummyPhys(x_left=r_inner, x_right=r_outer, T_left=10.0, T_right=0.0)
        
        x_inner = jnp.array([[r_inner]], dtype=jnp.float32)
        x_outer = jnp.array([[r_outer]], dtype=jnp.float32)
        
        u_inner = cylinder_1d_dirichlet_exact(x_inner, phys)
        u_outer = cylinder_1d_dirichlet_exact(x_outer, phys)
        
        assert jnp.isclose(u_inner[0], phys.T_left, rtol=1e-5)
        assert jnp.isclose(u_outer[0], phys.T_right, rtol=1e-5)


class TestAnalyticSphere1D:
    """Test analytic solution for 1D spherical geometry."""

    def test_sphere_solution_shape(self):
        """Check solution shape for spherical case."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        phys = DummyPhys(x_left=0.1, x_right=1.0, T_left=1.0, T_right=0.0)
        
        u = sphere_1d_dirichlet_exact(x, phys)
        
        assert u.shape[0] == x.shape[0]

    def test_sphere_solution_boundary_values(self):
        """Check BCs for spherical solution."""
        r_inner = 1.0
        r_outer = 3.0
        
        phys = DummyPhys(x_left=r_inner, x_right=r_outer, T_left=100.0, T_right=50.0)
        
        x_inner = jnp.array([[r_inner]], dtype=jnp.float32)
        x_outer = jnp.array([[r_outer]], dtype=jnp.float32)
        
        u_inner = sphere_1d_dirichlet_exact(x_inner, phys)
        u_outer = sphere_1d_dirichlet_exact(x_outer, phys)
        
        assert jnp.isclose(u_inner[0], phys.T_left, rtol=1e-5)
        assert jnp.isclose(u_outer[0], phys.T_right, rtol=1e-5)


class TestAnalyticDeterminism:
    """Test determinism of analytic solutions."""

    def test_all_solutions_deterministic(self):
        """All analytic solutions should be deterministic."""
        x = jnp.linspace(0.1, 1.0, 15).reshape(-1, 1)
        phys = DummyPhys(x_left=0.1, x_right=1.0, T_left=1.0, T_right=0.0)
        
        # Line
        u1_line = line_1d_dirichlet_exact(x, phys)
        u2_line = line_1d_dirichlet_exact(x, phys)
        assert jnp.allclose(u1_line, u2_line)
        
        # Cylinder
        u1_cyl = cylinder_1d_dirichlet_exact(x, phys)
        u2_cyl = cylinder_1d_dirichlet_exact(x, phys)
        assert jnp.allclose(u1_cyl, u2_cyl)
        
        # Sphere
        u1_sph = sphere_1d_dirichlet_exact(x, phys)
        u2_sph = sphere_1d_dirichlet_exact(x, phys)
        assert jnp.allclose(u1_sph, u2_sph)
