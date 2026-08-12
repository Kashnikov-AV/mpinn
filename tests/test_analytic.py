"""Tests for analytic solutions module."""
import jax
import jax.numpy as jnp
import pytest

from mpinn.analytic import line_1d_solution, cylinder_1d_solution, sphere_1d_solution


class TestAnalyticLine1D:
    """Test analytic solution for 1D line geometry."""

    def test_line_solution_shape(self):
        """Check that solution has correct shape."""
        x = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
        
        # Example: u'' = 0 with u(0)=0, u(1)=1 => u(x) = x
        u = line_1d_solution(x, bc_left=0.0, bc_right=1.0)
        
        assert u.shape[0] == x.shape[0], f"Expected {x.shape[0]}, got {u.shape[0]}"

    def test_line_solution_boundary_values(self):
        """Check that solution satisfies boundary conditions."""
        x_left = jnp.array([[0.0]], dtype=jnp.float32)
        x_right = jnp.array([[1.0]], dtype=jnp.float32)
        
        bc_left = 5.0
        bc_right = -3.0
        
        u_left = line_1d_solution(x_left, bc_left=bc_left, bc_right=bc_right)
        u_right = line_1d_solution(x_right, bc_left=bc_left, bc_right=bc_right)
        
        assert jnp.isclose(u_left[0, 0], bc_left, atol=1e-6), f"Left BC not satisfied"
        assert jnp.isclose(u_right[0, 0], bc_right, atol=1e-6), f"Right BC not satisfied"

    def test_line_solution_dtype(self):
        """Check dtype of analytic solution."""
        x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1).astype(jnp.float32)
        
        u = line_1d_solution(x, bc_left=0.0, bc_right=1.0)
        
        assert u.dtype == jnp.float32


class TestAnalyticCylinder1D:
    """Test analytic solution for 1D cylindrical geometry."""

    def test_cylinder_solution_shape(self):
        """Check solution shape for cylindrical case."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)  # Avoid r=0
        
        u = cylinder_1d_solution(x, bc_inner=1.0, bc_outer=0.0, r_inner=0.1, r_outer=1.0)
        
        assert u.shape[0] == x.shape[0]

    def test_cylinder_solution_boundary_values(self):
        """Check BCs for cylindrical solution."""
        r_inner = 0.5
        r_outer = 2.0
        
        x_inner = jnp.array([[r_inner]], dtype=jnp.float32)
        x_outer = jnp.array([[r_outer]], dtype=jnp.float32)
        
        bc_inner = 10.0
        bc_outer = 0.0
        
        u_inner = cylinder_1d_solution(x_inner, bc_inner=bc_inner, bc_outer=bc_outer, 
                                        r_inner=r_inner, r_outer=r_outer)
        u_outer = cylinder_1d_solution(x_outer, bc_inner=bc_inner, bc_outer=bc_outer,
                                        r_inner=r_inner, r_outer=r_outer)
        
        assert jnp.isclose(u_inner[0, 0], bc_inner, rtol=1e-5)
        assert jnp.isclose(u_outer[0, 0], bc_outer, rtol=1e-5)


class TestAnalyticSphere1D:
    """Test analytic solution for 1D spherical geometry."""

    def test_sphere_solution_shape(self):
        """Check solution shape for spherical case."""
        x = jnp.linspace(0.1, 1.0, 20).reshape(-1, 1)
        
        u = sphere_1d_solution(x, bc_inner=1.0, bc_outer=0.0, r_inner=0.1, r_outer=1.0)
        
        assert u.shape[0] == x.shape[0]

    def test_sphere_solution_boundary_values(self):
        """Check BCs for spherical solution."""
        r_inner = 1.0
        r_outer = 3.0
        
        x_inner = jnp.array([[r_inner]], dtype=jnp.float32)
        x_outer = jnp.array([[r_outer]], dtype=jnp.float32)
        
        bc_inner = 100.0
        bc_outer = 50.0
        
        u_inner = sphere_1d_solution(x_inner, bc_inner=bc_inner, bc_outer=bc_outer,
                                      r_inner=r_inner, r_outer=r_outer)
        u_outer = sphere_1d_solution(x_outer, bc_inner=bc_inner, bc_outer=bc_outer,
                                      r_inner=r_inner, r_outer=r_outer)
        
        assert jnp.isclose(u_inner[0, 0], bc_inner, rtol=1e-5)
        assert jnp.isclose(u_outer[0, 0], bc_outer, rtol=1e-5)


class TestAnalyticDeterminism:
    """Test determinism of analytic solutions."""

    def test_all_solutions_deterministic(self):
        """All analytic solutions should be deterministic."""
        x = jnp.linspace(0.1, 1.0, 15).reshape(-1, 1)
        
        # Line
        u1_line = line_1d_solution(x, 0.0, 1.0)
        u2_line = line_1d_solution(x, 0.0, 1.0)
        assert jnp.allclose(u1_line, u2_line)
        
        # Cylinder
        u1_cyl = cylinder_1d_solution(x, 1.0, 0.0, 0.1, 1.0)
        u2_cyl = cylinder_1d_solution(x, 1.0, 0.0, 0.1, 1.0)
        assert jnp.allclose(u1_cyl, u2_cyl)
        
        # Sphere
        u1_sph = sphere_1d_solution(x, 1.0, 0.0, 0.1, 1.0)
        u2_sph = sphere_1d_solution(x, 1.0, 0.0, 0.1, 1.0)
        assert jnp.allclose(u1_sph, u2_sph)
