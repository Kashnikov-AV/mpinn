"""Tests for geometry module."""
import jax
import jax.numpy as jnp
import pytest

# Assuming the library is installed or in path
from mpinn.geom import Interval


class TestInterval:
    """Test cases for Interval geometry class."""

    def test_random_points_shape(self, seed_key):
        """Check that random_points returns correct shape (N, dim)."""
        geom = Interval(0.0, 1.0)
        points = geom.random_points(seed_key, num_points=20)
        
        assert points.shape == (20, 1), f"Expected shape (20, 1), got {points.shape}"

    def test_random_points_dtype(self, seed_key):
        """Check that points are float32."""
        geom = Interval(0.0, 1.0)
        points = geom.random_points(seed_key, num_points=10)
        
        assert points.dtype == jnp.float32, f"Expected float32, got {points.dtype}"

    def test_boundary_points(self):
        """Check that boundary points are at interval edges."""
        geom = Interval(0.0, 1.0)
        boundaries = geom.boundary_points(num_points_per_side=5)
        
        # Should have 2 sides * 5 points = 10 points total
        assert boundaries.shape[0] == 10
        assert boundaries.shape[1] == 1
        
        # Check min and max values correspond to interval bounds
        assert jnp.min(boundaries) >= 0.0
        assert jnp.max(boundaries) <= 1.0

    def test_deterministic_sampling(self):
        """Check that same seed produces same points."""
        geom = Interval(0.0, 1.0)
        key1 = jax.random.PRNGKey(123)
        key2 = jax.random.PRNGKey(123)
        
        points1 = geom.random_points(key1, num_points=15)
        points2 = geom.random_points(key2, num_points=15)
        
        assert jnp.allclose(points1, points2), "Sampling should be deterministic with same seed"
