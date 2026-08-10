"""Tests for PDE module."""
import jax
import jax.numpy as jnp
import pytest

from mpinn.pde import line_1d, cylinder_1d, sphere_1d


class TestLine1D:
    """Test cases for 1D line (Cartesian) PDE."""

    def test_line_residual_shape(self, sample_interval_points):
        """Check that residual has correct shape."""
        x = sample_interval_points
        
        def dummy_net(x):
            return x ** 2  # Simple quadratic function
        
        residual = line_1d(dummy_net, x)
        
        # Residual should have same number of rows as input
        assert residual.shape[0] == x.shape[0], f"Expected {x.shape[0]}, got {residual.shape[0]}"
        assert residual.ndim == 1 or residual.shape[1] == 1, "Residual should be 1D or (N, 1)"

    def test_line_dtype(self, sample_interval_points):
        """Check that residual is float32."""
        x = sample_interval_points
        
        def dummy_net(x):
            return x ** 2
        
        residual = line_1d(dummy_net, x)
        
        assert residual.dtype == jnp.float32, f"Expected float32, got {residual.dtype}"

    def test_line_deterministic(self, sample_interval_points):
        """Check determinism of residual computation."""
        x = sample_interval_points
        
        def dummy_net(x):
            return jnp.sin(x)
        
        res1 = line_1d(dummy_net, x)
        res2 = line_1d(dummy_net, x)
        
        assert jnp.allclose(res1, res2), "PDE residual should be deterministic"

    def test_line_exact_solution(self, sample_interval_points):
        """Test with known solution u'' = 0 => u = ax + b."""
        x = sample_interval_points
        
        # Linear function has zero second derivative
        def linear_net(x):
            return 2.0 * x + 3.0
        
        residual = line_1d(linear_net, x)
        
        # Mean residual should be ~0
        mean_residual = jnp.mean(jnp.abs(residual))
        assert jnp.isclose(mean_residual, 0.0, atol=1e-5), f"Expected ~0, got {mean_residual}"


class TestCylinder1D:
    """Test cases for 1D cylindrical PDE."""

    def test_cylinder_residual_shape(self, sample_interval_points):
        """Check residual shape for cylindrical geometry."""
        x = sample_interval_points + 0.1  # Avoid r=0 singularity
        
        def dummy_net(x):
            return x ** 2
        
        residual = cylinder_1d(dummy_net, x)
        
        assert residual.shape[0] == x.shape[0]

    def test_cylinder_dtype(self, sample_interval_points):
        """Check dtype for cylindrical PDE."""
        x = sample_interval_points + 0.1
        
        def dummy_net(x):
            return x
        
        residual = cylinder_1d(dummy_net, x)
        
        assert residual.dtype == jnp.float32


class TestSphere1D:
    """Test cases for 1D spherical PDE."""

    def test_sphere_residual_shape(self, sample_interval_points):
        """Check residual shape for spherical geometry."""
        x = sample_interval_points + 0.1  # Avoid r=0 singularity
        
        def dummy_net(x):
            return x ** 2
        
        residual = sphere_1d(dummy_net, x)
        
        assert residual.shape[0] == x.shape[0]

    def test_sphere_dtype(self, sample_interval_points):
        """Check dtype for spherical PDE."""
        x = sample_interval_points + 0.1
        
        def dummy_net(x):
            return x
        
        residual = sphere_1d(dummy_net, x)
        
        assert residual.dtype == jnp.float32


class TestPDEDeterminism:
    """Test determinism across all PDE types."""

    def test_all_pdes_deterministic(self, sample_interval_points):
        """All PDE functions should be deterministic."""
        x = sample_interval_points + 0.1
        
        def test_net(x):
            return jnp.exp(-x)
        
        # Test line_1d
        res1_line = line_1d(test_net, x)
        res2_line = line_1d(test_net, x)
        assert jnp.allclose(res1_line, res2_line)
        
        # Test cylinder_1d
        res1_cyl = cylinder_1d(test_net, x)
        res2_cyl = cylinder_1d(test_net, x)
        assert jnp.allclose(res1_cyl, res2_cyl)
        
        # Test sphere_1d
        res1_sph = sphere_1d(test_net, x)
        res2_sph = sphere_1d(test_net, x)
        assert jnp.allclose(res1_sph, res2_sph)
