from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array as ArrayType

from prolix.physics.step_system import Step, IntegratorState
from prolix.typing import IntegratorParams
from prolix.physics.virtual_sites import reconstruct_virtual_sites
from prolix.typing import VirtualSiteDef, VirtualSiteParamsPacked, VirtualSiteParams

class VirtualSiteReconstructionStep(Step):
  """Virtual site position reconstruction step."""
  vs_def: VirtualSiteDef
  vs_params: VirtualSiteParamsPacked

  def __init__(self, vs_def: VirtualSiteDef, vs_params: VirtualSiteParamsPacked):
    self.vs_def = jnp.asarray(vs_def)
    self.vs_params = jnp.asarray(vs_params)

  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Reconstruct virtual site positions."""
    new_positions = reconstruct_virtual_sites(
        state.positions, self.vs_def, self.vs_params
    )
    return state.__replace__(positions=new_positions)

def redistribute_forces(
    forces: ArrayType,
    vs_def: VirtualSiteDef,
    vs_params: VirtualSiteParamsPacked,
) -> ArrayType:
  """Redistribute forces from virtual sites to parent atoms using the transpose of the reconstruction weights."""
  
  new_forces = jnp.copy(forces)
  
  def apply_transpose_batch(vs_idx, p1_idx, p2_idx, p3_idx, vs_f, params):
    # Reconstruct the weight matrix W: [origin_w, x_w, y_w] which is 3x3
    # M = W^T * [pos1, pos2, pos3] + z_dir * p_local
    # This is complex because z_dir depends on positions.
    # For linear part (origin):
    # F_p = W * F_vs
    # But for full reconstruction, it includes rotators.
    # Standard approximation is to redistribute just the origin weights for now
    # if full rotation is too hard to linearize perfectly in one step.
    # However, let's look at the TIP4P definition.
    
    # Simple linear redistribution for the main components:
    # origin = weights * parents
    # F_parents += weights^T * F_vs
    
    # Just redistributing origin forces for now as a baseline for conservation.
    return (
      params.origin_weights * vs_f,
      jnp.stack([p1_idx, p2_idx, p3_idx]),
    )

  # For now, let's implement the simpler version: 
  # redistribute forces from vs to parents based on origin_weights
  # This is already a massive improvement over zeroing them.
  
  vs_indices = vs_def[:, 0]
  parents_indices = vs_def[:, 1:4]
  vs_forces = forces[vs_indices]
  
  # For each virtual site, distribute its force to its 3 parents
  def redistribute_one(vs_f, parents_idx, params_row):
    params = VirtualSiteParams.from_row(params_row)
    # The redistribution is: F_p_i += w_i * F_vs
    # w_i is in params.origin_weights (shape 3,)
    # F_vs is shape (3,)
    # We want to return (3, 3) where each row is the contribution to one parent
    return params.origin_weights[:, None] * vs_f[None, :]

  # redistributed will be (num_sites, 3, 3)
  redistributed = jax.vmap(redistribute_one)(vs_forces, parents_indices, vs_params)
  
  # Add back to forces
  for i in range(3):
    # redistributed[:, i, :] is (num_sites, 3)
    new_forces = new_forces.at[parents_indices[:, i]].add(redistributed[:, i, :])

  # Finally, zero out the virtual site forces
  new_forces = new_forces.at[vs_indices].set(jnp.zeros_like(vs_forces))
    
  return new_forces
