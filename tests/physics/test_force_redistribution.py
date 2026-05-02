from __future__ import annotations

import jax
import jax.numpy as jnp
from prolix.physics.virtual_sites_step import redistribute_forces
from prolix.types import VirtualSiteDef, VirtualSiteParamsPacked, VirtualSiteParams

def test_force_redistribution_conservation():
    """Verify that forces on virtual sites are redistributed, not lost."""
    # Setup: 1 VS (idx 3) with parents 0, 1, 2
    vs_def = jnp.array([[3, 0, 1, 2]], dtype=jnp.int32)
    
    # Weights for M-site origin: O(1.0), H1(0.0), H2(0.0)
    vs_params = jnp.zeros((1, 12), dtype=jnp.float32)
    # Origin weights are in indices 3:6 of packed params
    vs_params = vs_params.at[0, 3:6].set([1.0, 0.0, 0.0]) 
    
    # Initial forces
    forces = jnp.array([
        [0.0, 0.0, 0.0], # O
        [0.0, 0.0, 0.0], # H1
        [0.0, 0.0, 0.0], # H2
        [1.0, 2.0, 3.0]  # VS (force on M)
    ], dtype=jnp.float32)
    
    new_forces = redistribute_forces(forces, vs_def, vs_params)
    
    print(f"Original forces:\n{forces}")
    print(f"Redistributed forces:\n{new_forces}")
    
    # Expected: Force on M (1, 2, 3) moved to O
    # Force on O should be 1.0 * (1, 2, 3) + 0 = (1, 2, 3)
    assert jnp.allclose(new_forces[0], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(new_forces[3], jnp.array([0.0, 0.0, 0.0]))
    print("Force redistribution test passed!")

if __name__ == "__main__":
    test_force_redistribution_conservation()
