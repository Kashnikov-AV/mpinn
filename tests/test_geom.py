"""Tests for geometry module."""
import jax
import jax.numpy as jnp
import pytest

from mpinn.geom import Interval


class TestInterval:
    """Test cases for Interval geometry class."""

    def test_sample_interior_shape(self, seed_key):
        """Check that sample_interior returns correct shape (N, dim)."""
        geom = Interval(0.0, 1.0)
        points = geom.sample_interior(n_points=20, rng=seed_key)
        
        assert points.shape == (20, 1), f"Expected shape (20, 1), got {points.shape}"

    def test_sample_interior_dtype(self, seed_key):
        """Check that points are float32."""
        geom = Interval(0.0, 1.0)
        points = geom.sample_interior(n_points=10, rng=seed_key)
        
        assert points.dtype == jnp.float32, f"Expected float32, got {points.dtype}"

    def test_sample_boundary(self):
        """Check that boundary points are at interval edges."""
        geom = Interval(0.0, 1.0)
        boundaries = geom.sample_boundary()
        
        # Should have 2 boundary points (left and right)
        assert boundaries.shape == (2, 1)
        
        # Check values correspond to interval bounds
        assert jnp.isclose(boundaries[0, 0], 0.0)
        assert jnp.isclose(boundaries[1, 0], 1.0)

    def test_deterministic_sampling(self):
        """Check that same seed produces same points."""
        geom = Interval(0.0, 1.0)
        key1 = jax.random.PRNGKey(123)
        key2 = jax.random.PRNGKey(123)
        
        points1 = geom.sample_interior(n_points=15, rng=key1)
        points2 = geom.sample_interior(n_points=15, rng=key2)
        
        assert jnp.allclose(points1, points2), "Sampling should be deterministic with same seed"
