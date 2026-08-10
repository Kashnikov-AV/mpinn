"""Configuration for pytest fixtures."""
import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def seed_key():
    """Provide a deterministic random key for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_interval_points():
    """Generate sample points from 1D interval geometry."""
    # Simulating Interval(0, 1).random_points(10)
    x = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)
    return x.astype(jnp.float32)


@pytest.fixture
def boundary_points():
    """Generate boundary points for testing BCs."""
    # Left and right boundaries for interval [0, 1]
    left = jnp.array([[0.0]], dtype=jnp.float32)
    right = jnp.array([[1.0]], dtype=jnp.float32)
    return {"left": left, "right": right}
