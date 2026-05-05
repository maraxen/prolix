from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_md import util

from prolix.typing import VirtualSiteParams, VirtualSiteParamsPacked

Array = util.Array
Coordinates = Array
VirtualSiteDef = Array

def reconstruct_virtual_sites(
  positions: Coordinates,
  vs_def: VirtualSiteDef,
  vs_params: VirtualSiteParamsPacked,
) -> Array:
  r"""Reconstruct virtual site positions from parent atoms.
  """
  positions = jnp.asarray(positions)

  def compute_batch(p_row, parents_pos_row):
    params = VirtualSiteParams.from_row(p_row)
    pos_parent1, pos_parent2, pos_parent3 = (
      parents_pos_row[0],
      parents_pos_row[1],
      parents_pos_row[2],
    )

    origin = (
      params.origin_weights[0] * pos_parent1
      + params.origin_weights[1] * pos_parent2
      + params.origin_weights[2] * pos_parent3
    )
    v1 = (
      params.x_weights[0] * pos_parent1
      + params.x_weights[1] * pos_parent2
      + params.x_weights[2] * pos_parent3
    )
    v2 = (
      params.y_weights[0] * pos_parent1
      + params.y_weights[1] * pos_parent2
      + params.y_weights[2] * pos_parent3
    )

    z_dir = jnp.cross(v1, v2)
    z_norm = jnp.linalg.norm(z_dir)
    z_dir = z_dir / jnp.where(z_norm > 1e-12, z_norm, 1.0)

    x_norm = jnp.linalg.norm(v1)
    x_dir = v1 / jnp.where(x_norm > 1e-12, x_norm, 1.0)

    y_dir = jnp.cross(z_dir, x_dir)

    return (
      origin + x_dir * params.p_local[0] + y_dir * params.p_local[1] + z_dir * params.p_local[2]
    )

  parent1_idx, parent2_idx, parent3_idx = vs_def[:, 1], vs_def[:, 2], vs_def[:, 3]
  parents_stack = jnp.stack(
    [positions[parent1_idx], positions[parent2_idx], positions[parent3_idx]], axis=1
  )

  new_virtual_site_positions = jax.vmap(compute_batch)(vs_params, parents_stack)

  vs_indices = vs_def[:, 0]
  return positions.at[vs_indices].set(new_virtual_site_positions)
