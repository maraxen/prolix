from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Float, Int

  from prolix.types import ArrayLike, Coordinates


def reconstruct_virtual_sites(
  positions: Coordinates,  # (N, 3)
  vs_def: Int[ArrayLike, "N_vs 4"],  # noqa: F722 # (N_vs, 4) [vs_idx, p1, p2, p3]
  vs_params: Float[ArrayLike, "N_vs 12"],  # noqa: F722 # (N_vs, 12)
) -> Coordinates:
  """Reconstruct virtual site positions from parent atoms.

  Args:
      positions: Atomic positions (Angstroms)
      vs_def: Definition of virtual sites. Each row:
              [vs_index, parent1_idx, parent2_idx, parent3_idx]
      vs_params: Parameters for each virtual site. Each row:
                 [p1, p2, p3, wo1..3, wx1..3, wy1..3]

  Returns:
      Updated positions array where virtual site coordinates are computed.

  """

  def _compute_single_vs(def_row, param_row, all_pos):
    vs_idx = def_row[0]
    i, j, k = def_row[1], def_row[2], def_row[3]

    # Origin weights
    wo = param_row[3:6]
    # X weights
    wx = param_row[6:9]
    # Y weights
    wy = param_row[9:12]

    # Local offset
    p_local = param_row[0:3]

    # Positions of parents
    r_i = all_pos[i]
    r_j = all_pos[j]
    r_k = all_pos[k]

    # Compute local frame
    # Origin
    origin = wo[0] * r_i + wo[1] * r_j + wo[2] * r_k

    # X-dir vector (before normalization)
    # Note: OpenMM defines it as weighted sum of coords?
    # Usually it's defined by vector difference.
    # But 'localCoords' uses generic weighted average of 3 points to define origin, x, y.
    # Let's check the XML parsing assumption.
    # For 'localCoords', the frame is defined by Origin, normalized X, normalized Y.
    # The axes vectors are computed as weighted sums.

    # Vector 1 (usually X direction)
    v1 = wx[0] * r_i + wx[1] * r_j + wx[2] * r_k
    # Vector 2 (usually Y direction)
    v2 = wy[0] * r_i + wy[1] * r_j + wy[2] * r_k

    # Construct orthogonal frame
    # Z = V1 x V2
    z_dir = jnp.cross(v1, v2)
    z_dir = z_dir / jnp.linalg.norm(z_dir)

    # X = V1 (normalized)? Or is V1 just direction?
    # Usually X is V1 normalized.
    x_dir = v1 / jnp.linalg.norm(v1)

    # Y = Z x X
    y_dir = jnp.cross(z_dir, x_dir)

    # Reconstruct P
    # R = Origin + x_dir*p1 + y_dir*p2 + z_dir*p3
    pos = origin + x_dir * p_local[0] + y_dir * p_local[1] + z_dir * p_local[2]

    return vs_idx, pos

  # Use map/scan?
  # Since virtual sites might depend on each other?
  # Usually valid VS don't depend on other VS (or are ordered).
  # Assuming no VS-VS dependencies for now (standard water models don't).

  # We can use vmap if we gather parents.
  # But parents are indices.

  # Compute all VS positions
  # vmap over defs and params
  # We need to pass 'positions' as closure or broadcast?
  # Indices are dynamic integers?
  # JAX likes static indices for gather usually, but here they are data.
  # positions[indices] works in JAX.

  # Let's write a batch function
  vs_def.shape[0]

  # If no virtual sites, return original
  # This check ideally outside JIT or using simple conditional
  # But JAX array shape 0 works.

  def body_fn(i, current_pos):
    d = vs_def[i]
    p = vs_params[i]

    idx, new_r = _compute_single_vs(d, p, current_pos)

    # Update ONLY if idx match?
    # Actually scatter update.
    return current_pos.at[idx].set(new_r)

  # If we assume VS don't depend on other VS, we can compute all in parallel
  # and then scatter update once.
  # _compute_single_vs needs full positions array to read parents.

  # VMAP approach:
  # Gather parents first
  # defs: (N_vs, 4)
  # idxs = vs_def[:, 1:4] # (N_vs, 3)
  # parents_pos = positions[idxs] # (N_vs, 3, 3)

  # Helper for VMAP
  def compute_batch(d_row, p_row, parents_pos_row):
    # Unpack logic from _compute_single_vs but with pre-gathered parents
    # parents_pos_row: (3, 3) corresponding to i, j, k

    wo = p_row[3:6]
    wx = p_row[6:9]
    wy = p_row[9:12]
    p_local = p_row[0:3]

    r_i, r_j, r_k = parents_pos_row[0], parents_pos_row[1], parents_pos_row[2]

    origin = wo[0] * r_i + wo[1] * r_j + wo[2] * r_k
    v1 = wx[0] * r_i + wx[1] * r_j + wx[2] * r_k
    v2 = wy[0] * r_i + wy[1] * r_j + wy[2] * r_k

    z_dir = jnp.cross(v1, v2)
    z_norm = jnp.linalg.norm(z_dir)
    # Avoid division by zero
    safe_z = jnp.where(z_norm > 1e-12, z_norm, 1.0)
    z_dir = z_dir / safe_z

    x_norm = jnp.linalg.norm(v1)
    safe_x = jnp.where(x_norm > 1e-12, x_norm, 1.0)
    x_dir = v1 / safe_x

    y_dir = jnp.cross(z_dir, x_dir)

    return origin + x_dir * p_local[0] + y_dir * p_local[1] + z_dir * p_local[2]

  # Gather parents
  # vs_def[:, 1], vs_def[:, 2], vs_def[:, 3]
  # But JAX advanced indexing works on arrays
  p1_idx = vs_def[:, 1]
  p2_idx = vs_def[:, 2]
  p3_idx = vs_def[:, 3]

  r1 = positions[p1_idx]
  r2 = positions[p2_idx]
  r3 = positions[p3_idx]

  parents_stack = jnp.stack([r1, r2, r3], axis=1)  # (N_vs, 3, 3)

  # Compute new positions
  new_vs_pos = jax.vmap(compute_batch)(vs_def, vs_params, parents_stack)

  # Scatter update
  vs_indices = vs_def[:, 0]
  return positions.at[vs_indices].set(new_vs_pos)
