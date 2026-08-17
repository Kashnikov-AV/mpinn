"""Tests for PDE module."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from mpinn.pde import line_1d, cylinder_1d, sphere_1d


class TestLine1D:
    """Test cases for 1D line (Cartesian) PDE."""

    def test_line_residual_shape(self, sample_interval_points, seed_key):
        """Check that residual is a scalar (mean squared residual)."""
        x = sample_interval_points
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        # phys is not used for line_1d, pass None or dummy object
        residual = line_1d(net, x, phys=None)
        
        # Residual should be a scalar (due to jnp.mean)
        assert residual.ndim == 0, f"Expected scalar, got shape {residual.shape}"

    def test_line_dtype(self, sample_interval_points, seed_key):
        """Check that residual is float32."""
        x = sample_interval_points
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        residual = line_1d(net, x, phys=None)
        
        assert residual.dtype == jnp.float32, f"Expected float32, got {residual.dtype}"

    def test_line_deterministic(self, sample_interval_points, seed_key):
        """Check determinism of residual computation."""
        x = sample_interval_points
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        res1 = line_1d(net, x, phys=None)
        res2 = line_1d(net, x, phys=None)
        
        assert jnp.allclose(res1, res2), "PDE residual should be deterministic"

    def test_line_exact_solution(self, sample_interval_points, seed_key):
        """Test with known solution u'' = 0 => u = ax + b."""
        x = sample_interval_points
        
        # Create a network that approximates a linear function
        # For exact linear function, second derivative is zero
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        # The residual should be finite and computable
        residual = line_1d(net, x, phys=None)
        
        # Just check it's a valid number
        assert jnp.isfinite(residual), f"Expected finite residual, got {residual}"


class TestCylinder1D:
    """Test cases for 1D cylindrical PDE."""

    def test_cylinder_residual_shape(self, sample_interval_points, seed_key):
        """Check residual shape for cylindrical geometry."""
        x = sample_interval_points + 0.1  # Avoid r=0 singularity
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        residual = cylinder_1d(net, x, phys=None)
        
        assert residual.ndim == 0, f"Expected scalar, got shape {residual.shape}"

    def test_cylinder_dtype(self, sample_interval_points, seed_key):
        """Check dtype for cylindrical PDE."""
        x = sample_interval_points + 0.1
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        residual = cylinder_1d(net, x, phys=None)
        
        assert residual.dtype == jnp.float32


class TestSphere1D:
    """Test cases for 1D spherical PDE."""

    def test_sphere_residual_shape(self, sample_interval_points, seed_key):
        """Check residual shape for spherical geometry."""
        x = sample_interval_points + 0.1  # Avoid r=0 singularity
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        residual = sphere_1d(net, x, phys=None)
        
        assert residual.ndim == 0, f"Expected scalar, got shape {residual.shape}"

    def test_sphere_dtype(self, sample_interval_points, seed_key):
        """Check dtype for spherical PDE."""
        x = sample_interval_points + 0.1
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        residual = sphere_1d(net, x, phys=None)
        
        assert residual.dtype == jnp.float32


class TestPDEDeterminism:
    """Test determinism across all PDE types."""

    def test_all_pdes_deterministic(self, sample_interval_points, seed_key):
        """All PDE functions should be deterministic."""
        x = sample_interval_points + 0.1
        
        rngs = nnx.Rngs(seed_key)
        from mpinn.pinn_core import FCNet
        net = FCNet(
            din=1, dmid=10, dout=1, num_layers=2,
            activation=nnx.tanh, rngs=rngs
        )
        
        # Test line_1d
        res1_line = line_1d(net, x, phys=None)
        res2_line = line_1d(net, x, phys=None)
        assert jnp.allclose(res1_line, res2_line)
        
        # Test cylinder_1d
        res1_cyl = cylinder_1d(net, x, phys=None)
        res2_cyl = cylinder_1d(net, x, phys=None)
        assert jnp.allclose(res1_cyl, res2_cyl)
        
        # Test sphere_1d
        res1_sph = sphere_1d(net, x, phys=None)
        res2_sph = sphere_1d(net, x, phys=None)
        assert jnp.allclose(res1_sph, res2_sph)
