"""Verification test for SHAKE constraints.

Validates that SHAKE accurately preserves constrained bond lengths
over multiple molecular dynamics steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.physics import simulate, system

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

@pytest.mark.integration
class TestSHAKEGeometry:
  """Tests SHAKE constraint geometry preservation."""

  def test_shake_preserves_bond_lengths(self):
    """Test that SHAKE maintains constrained bond lengths over MD steps."""
    # 3-atom system (e.g., a water-like model or small peptide fragment)
    # Atoms 0-1 and 1-2 constrained.
    n_atoms = 3
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 0.866, 0.0]
    ], dtype=jnp.float64)
    
    masses = jnp.array([16.0, 1.0, 1.0], dtype=jnp.float64)
    
    # Constraints: (atom_i, atom_j), length
    pairs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    lengths = jnp.array([1.0, 1.0], dtype=jnp.float64)
    
    def energy_fn(r):
        return 0.001 * jnp.sum(r**2) # Weak harmonic to center
    
    displacement_fn, shift_fn = space.free()
    
    dt = 0.002 # 2 fs
    kT = 0.6 # ~300K
    
    init_fn, apply_fn = simulate.rattle_langevin(
        energy_fn,
        shift_fn=shift_fn,
        dt=dt,
        kT=kT,
        gamma=1.0,
        mass=masses,
        constraints=(pairs, lengths)
    )
    
    key = jax.random.PRNGKey(42)
    state = init_fn(key, positions)
    
    # Run 1000 steps
    for _ in range(1000):
        state = apply_fn(state)
        
    final_pos = state.position
    
    # Check bond lengths
    def get_length(idx1, idx2):
        dr = displacement_fn(final_pos[idx1], final_pos[idx2])
        return float(jnp.linalg.norm(dr))
    
    l01 = get_length(0, 1)
    l12 = get_length(1, 2)
    
    err01 = abs(l01 - 1.0)
    err12 = abs(l12 - 1.0)
    
    print(f"\nFinal bond lengths after 1000 steps:")
    print(f"  Bond 0-1: {l01:.10f} (error: {err01:.2e})")
    print(f"  Bond 1-2: {l12:.10f} (error: {err12:.2e})")
    
    # Target tolerance: 1e-5 A (standard for MD constraint convergence)
    assert err01 < 1e-5
    assert err12 < 1e-5

if __name__ == "__main__":
  pytest.main([__file__, "-v"])
