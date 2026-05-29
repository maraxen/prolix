from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from prolix.typing import IntegratorState
from prolix.typing import IntegratorParams, EnergyParams
from prolix.physics.virtual_sites_step import VirtualSiteReconstructionStep
from prolix.typing import VirtualSiteDef, VirtualSiteParamsPacked

def test_tip4p_reconstruction():
    """Verify TIP4P M-site reconstruction."""
    # Dummy system with 1 water molecule: O (0), H1 (1), H2 (2), M (3)
    # Positions in angstroms
    pos = jnp.array([
        [0.0, 0.0, 0.0],    # O
        [0.0, 0.757, 0.586], # H1
        [0.0, -0.757, 0.586], # H2
        [0.0, 0.0, 0.15]      # M (initial guess)
    ], dtype=jnp.float32)
    
    # Virtual Site Definition: [vs_idx, p1, p2, p3]
    vs_def = jnp.array([[3, 0, 1, 2]], dtype=jnp.int32)
    
    # TIP4P params: M-site is at distance 0.15 Angstrom from O along bisector.
    # We can simplify for this test by just making a rigid definition.
    # Weights for M-site (just Oxygen for now, for simplicity):
    # O=1.0, H1=0.0, H2=0.0, x_weights=... etc.
    # In reality TIP4P uses more complex geometry, but we want to test
    # that the reconstruction step *can* update site 3 from parents 0, 1, 2.
    
    # Let's use weights such that M = 1.0 * O
    vs_params = jnp.zeros((1, 12), dtype=jnp.float32)
    vs_params = vs_params.at[0, 0:3].set([0.0, 0.0, 0.15]) # p_local = (0, 0, 0.15)
    vs_params = vs_params.at[0, 3:6].set([1.0, 0.0, 0.0]) # origin = O
    vs_params = vs_params.at[0, 6:9].set([0.0, 1.0, 0.0]) # x_dir
    vs_params = vs_params.at[0, 9:12].set([0.0, 0.0, 1.0]) # y_dir
    
    # Create step
    step = VirtualSiteReconstructionStep(vs_def, vs_params)
    
    # Initial state
    state = IntegratorState(
        positions=pos,
        momentum=jnp.zeros_like(pos),
        force=jnp.zeros_like(pos),
        mass=jnp.ones((4, 1)),
        rng=jax.random.key(0)
    )
    
    params = IntegratorParams(
        dt=0.5,
        kT=2.49,
        gamma=1.0,
        energy_params=EnergyParams(params={})
    )
    
    # Apply step
    new_state = step.apply(state, params)
    
    # Check that virtual site M position changed
    # (previously [0.0, 0.0, 0.15], now should be [0.0, 0.0, 0.15] relative to O)
    # Let's change parents to make it more obvious
    
    new_pos = pos.at[0].set([1.0, 2.0, 3.0])
    state_new = state.__replace__(positions=new_pos)
    
    new_state = step.apply(state_new, params)
    
    # Expected: M = (1.0, 2.0, 3.0) + z_dir * 0.15 = (1.0, 2.0, 3.0) + (0, 0, 1) * 0.15 = (1.0, 2.0, 3.15)
    # The actual implementation: origin + x_dir * p0 + y_dir * p1 + z_dir * p2
    # Origin = O = (1, 2, 3)
    # x_dir = H1 - O (not exactly, it depends on weights)
    # With O_weights=(1,0,0), x_weights=(0,1,0), y_weights=(0,0,1):
    # Origin = 1.0 * O = (1, 2, 3)
    # x_dir = 1.0 * H1 = (0, 0.757, 0.586)
    # y_dir = 1.0 * H2 = (0, -0.757, 0.586)
    # M = (1, 2, 3) + x_dir * 0 + y_dir * 0 + z_dir * 0.15
    # Wait, vs_params.at[0, 0:3].set([0.0, 0.0, 0.15])
    # p_local = (0, 0, 0.15)
    # The code: origin + x_dir * p0 + y_dir * p1 + z_dir * p2
    # So M = (1, 2, 3) + x_dir * 0 + y_dir * 0 + z_dir * 0.15
    
    # Let's re-calculate z_dir:
    # v1 = H1 = (0, 0.757, 0.586)
    # v2 = H2 = (0, -0.757, 0.586)
    # v1 x v2 = (0.887, 0, 0)
    # z_dir = norm(v1 x v2) = (1, 0, 0)
    # So M = (1, 2, 3) + (1, 0, 0) * 0.15 = (1.15, 2, 3)
    
    expected_m = jnp.array([1.15, 2.0, 3.0])
    assert jnp.allclose(new_state.positions[3], expected_m)
    
    print("TIP4P reconstruction passed!")

if __name__ == "__main__":
    test_tip4p_reconstruction()
