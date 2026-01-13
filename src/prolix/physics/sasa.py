from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prolix.types import Coordinates, Energy, Radii


def _compute_spherical_cap_overlaps(dists: Array, radii: Radii, probe_radius: float) -> Array:
  """Calculates the overlap area of spherical caps between pairs of atoms.

  Args:
      dists: (N, N) pairwise distance matrix.
      radii: (N,) atomic radii.
      probe_radius: solvent probe radius.

  Returns:
      (N, N) matrix of overlap areas.
  """
  effective_radii = radii + probe_radius
  r_i = effective_radii[:, None]
  r_j = effective_radii[None, :]

  # Distance between centers
  d = dists

  # Conditions for spherical cap inclusion/exclusion
  no_overlap = d >= (r_i + r_j)
  i_inside_j = d <= (r_j - r_i)  # i is completely buried in j
  j_inside_i = d <= (r_i - r_j)  # j is completely internal to i

  # Cap Area Formula: A_cap = (pi * r_i / d) * (r_i + r_j - d) * (d + r_j - r_i)
  d_safe = jnp.where(d < 1e-6, 1.0, d)
  cap_area = (jnp.pi * r_i / d_safe) * (r_i + r_j - d) * (d + r_j - r_i)

  # Apply logic
  overlap_area = jnp.where(no_overlap, 0.0, cap_area)
  overlap_area = jnp.where(i_inside_j, 4 * jnp.pi * r_i**2, overlap_area)
  overlap_area = jnp.where(j_inside_i, 0.0, overlap_area)

  # Mask self-interaction
  mask = 1.0 - jnp.eye(radii.shape[0])
  return overlap_area * mask


def compute_sasa_energy_approx(
  positions: Coordinates,  # (N, 3)
  radii: Radii,  # (N,)
  gamma: float = 0.00542,  # kcal/mol/A^2
  offset: float = 0.92,  # kcal/mol
  probe_radius: float = 1.4,
) -> Energy:
  """Computes Non-Polar Solvation Energy using a differentiable SASA approximation.

  Formula: E = gamma * SASA + offset
  """
  # Pairwise distances
  diff = positions[:, None, :] - positions[None, :, :]
  dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

  # Get pairwise overlaps
  overlap_area = _compute_spherical_cap_overlaps(dists, radii, probe_radius)

  # Sum overlaps with scaling to account for multi-body overlap redundancy
  # A_i = Max(0, 4*pi*r_i^2 - S * Sum(Overlap))
  overlap_scale = 0.1
  total_overlap = jnp.sum(overlap_area, axis=1) * overlap_scale

  # Net area calculation
  effective_radii = radii + probe_radius
  surface_areas = 4 * jnp.pi * effective_radii**2
  net_area = jax.nn.relu(surface_areas - total_overlap)

  return gamma * jnp.sum(net_area) + offset
