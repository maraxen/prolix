"""Tests for PBC module."""

import jax
import jax.numpy as jnp
import pytest
from prolix.physics import pbc

def test_periodic_displacement_wraps_correctly():
    """Test that displacement function respects minimum image convention."""
    box = jnp.array([10.0, 10.0, 10.0])
    displacement_fn, _ = pbc.create_periodic_space(box)
    
    # Points at 1.0 and 9.0 (dist 8.0 or 2.0 wrapped?)
    # 1.0 - 9.0 = -8.0 -> nearest image is +2.0
    r1 = jnp.array([1.0, 0.0, 0.0])
    r2 = jnp.array([9.0, 0.0, 0.0])
    
    dr = displacement_fn(r1, r2)
    assert jnp.allclose(dr, jnp.array([2.0, 0.0, 0.0]))
    
    # Distance
    dist = jnp.linalg.norm(dr)
    assert jnp.isclose(dist, 2.0)

def test_wrap_positions_stays_in_box():
    """Test that positions outside box are wrapped correctly."""
    box = jnp.array([10.0, 10.0, 10.0])
    
    r = jnp.array([
        [11.0, -1.0, 5.0],
        [25.0, 10.0, 0.0]
    ])
    
    wrapped = pbc.wrap_positions(r, box)
    
    expected = jnp.array([
        [1.0, 9.0, 5.0],
        [5.0, 0.0, 0.0]  # 10.0 % 10.0 == 0.0 usually, or 10.0 depending on implementation
    ])
    
    # Jax modulo handling: 10.0 % 10.0 -> 0.0
    assert jnp.allclose(wrapped, expected)
    assert jnp.all(wrapped >= 0.0)
    assert jnp.all(wrapped < box)

def test_minimum_image_distance():
    """Test explicit minimum image distance calculation."""
    box = jnp.array([10.0, 10.0, 10.0])
    r1 = jnp.array([1.0, 1.0, 1.0])
    r2 = jnp.array([9.0, 9.0, 9.0])
    
    # Dist vector: (-8, -8, -8) -> (2, 2, 2)
    # Norm: sqrt(12)
    
    dist = pbc.minimum_image_distance(r1, r2, box)
    expected = jnp.sqrt(12.0)
    
    assert jnp.isclose(dist, expected)
