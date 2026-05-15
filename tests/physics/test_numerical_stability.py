import jax
import jax.numpy as jnp
import pytest
from prolix.physics.optimization import chunked_lj_energy

def test_lj_exclusion_precision():
    # Setup two atoms with a bond, distance 1.5, sigma 1.0, epsilon 1.0
    # LJ energy at 1.5 with sigma=1.0, eps=1.0 is 4 * ((1/1.5)**12 - (1/1.5)**6)
    # = 4 * (0.0152 - 0.1975) = -0.729
    
    # If they are "bonded" (excluded), energy should be 0.0.
    # We want to ensure that NO buffer is needed to avoid NaNs.
    
    pos = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float32)
    sigmas = jnp.array([1.0, 1.0], dtype=jnp.float32)
    epsilons = jnp.array([1.0, 1.0], dtype=jnp.float32)
    
    def displacement_fn(r1, r2):
        return r1 - r2
    
    # Standard energy (calculated to be ~ -0.3203)
    energy_no_excl = chunked_lj_energy(pos, sigmas, epsilons, displacement_fn, cutoff=2.0)
    assert jnp.abs(energy_no_excl + 0.3203) < 1e-4
    
    # With exclusion: we don't have direct API here but the current implementation 
    # of `chunked_lj_energy` uses `idx_j != idx_i`. 
    # Let's test `dist` being exactly 0 (which often produces NaN).
    
    pos_overlap = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    # Should not produce NaN if handled by mask
    energy_overlap = chunked_lj_energy(pos_overlap, sigmas, epsilons, displacement_fn, cutoff=2.0)
    assert not jnp.isnan(energy_overlap)
    assert energy_overlap == 0.0
