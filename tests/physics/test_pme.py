"""Tests for PME module."""

import jax
import jax.numpy as jnp
import pytest
from prolix.physics import pme

def test_pme_energy_finite():
    """Test that PME energy function returns finite values."""
    box = jnp.array([20.0, 20.0, 20.0])
    charges = jnp.array([1.0, -1.0])
    
    pme_fn = pme.make_pme_energy_fn(charges, box, grid_points=32)
    
    r = jnp.array([
        [5.0, 5.0, 5.0],
        [15.0, 15.0, 15.0]
    ])
    
    e = pme_fn(r)
    assert jnp.isfinite(e)
    
    # Check gradients
    grad_fn = jax.grad(pme_fn)
    forces = -grad_fn(r)
    assert jnp.all(jnp.isfinite(forces))

def test_pme_symmetry():
    """Test invariance to translation (if box wrapped)."""
    box = jnp.array([10.0, 10.0, 10.0])
    charges = jnp.array([1.0, -1.0])
    pme_fn = pme.make_pme_energy_fn(charges, box, grid_points=32)
    
    r = jnp.array([[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]])
    e1 = pme_fn(r)
    
    # Translate by whole box (should match exactly if PME uses wrapped coords internally?)
    # jax_md.energy.coulomb_recip_pme docs say "map_charges_to_grid... fractional_coordinates=False"
    # User needs to ensure positions are reasonable?
    # Usually Ewald sums are periodic.
    
    r_shifted = r + box
    e2 = pme_fn(r_shifted)
    
    assert jnp.isclose(e1, e2)
    
    # Shift by small amount
    shift = jnp.array([1.0, 0.0, 0.0])
    e3 = pme_fn(r + shift)
    
    # Energy depends on relative distance, so shifting both particles nicely should be invariant
    # IF the grid is fine enough or PME is exact?
    # PME is approximate and grid dependent, so translational invariance is not perfect 
    # unless shift is integer number of grid spacings.
    # But for Recip space, it should be invariant?
    
    assert jnp.isclose(e1, e3, atol=1e-3)
