from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax_md import util

from prolix.types import VirtualSiteParams

Array = util.Array

if TYPE_CHECKING:
  from prolix.types import Coordinates, VirtualSiteDef, VirtualSiteParamsPacked


def reconstruct_virtual_sites(
  positions: Coordinates,
  vs_def: VirtualSiteDef,
  vs_params: VirtualSiteParamsPacked,
) -> Array:
  r"""Reconstruct virtual site positions from parent atoms.

  Process:
  1.  **Gather Parents**: Extract coordinates $\vec{r}_i, \vec{r}_j, \vec{r}_k$ for each site.
  2.  **Origin**: Compute frame origin as weighted sum: $\vec{O} = \sum w_a \vec{r}_a$.
  3.  **Basis**: Compute reference vectors $\vec{v}_1, \vec{v}_2$ from weights.
  4.  **Orthonormalize**:
      -   $\hat{z} = \text{norm}(\vec{v}_1 \times \vec{v}_2)$
      -   $\hat{x} = \text{norm}(\vec{v}_1)$
      -   $\hat{y} = \hat{z} \times \hat{x}$
  5.  **Project**: $\vec{r}_{vs} = \vec{O} + x_{loc} \hat{x} + y_{loc} \hat{y} + z_{loc} \hat{z}$.

  Notes:
  Virtual sites defined by OpenMM `LocalCoordinatesSite` logic.
  Small epsilon ($10^{-12}$) added to norms to avoid division by zero.

  Args:
      positions: Atomic positions (N, 3).
      vs_def: Virtual site definitions (N_vs, 4) as [vs_idx, p1, p2, p3].
      vs_params: Packed parameters (N_vs, 12).

  Returns:
      Updated positions array with reconstructed virtual sites.
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
