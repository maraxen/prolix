"""Neighbor list management and sparse exclusion handling."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax_md import partition, space, util

from prolix.utils import topology

if TYPE_CHECKING:
  from collections.abc import Callable

  from proxide.md import SystemParams

Array = util.Array


@dataclasses.dataclass(frozen=True)
class ExclusionSpec:
  """Sparse specification of excluded interactions for neighbor lists.

  Instead of N×N dense matrices, we store pairs of atoms that should be
  excluded or scaled. This allows O(N×K) filtering in neighbor lists.
  """

  # 1-2 and 1-3 pairs (fully excluded)
  idx_12_13: Array  # (M1, 2) int32

  # 1-4 pairs (scaled)
  idx_14: Array  # (M2, 2) int32
  scale_14_elec: float
  scale_14_vdw: float

  # Total number of atoms (for validation/creation)
  n_atoms: int

  @classmethod
  def from_system_params(
    cls, system_params: SystemParams, coulomb14scale: float = 0.83333333, lj14scale: float = 0.5
  ) -> ExclusionSpec:
    """Build ExclusionSpec from standard system parameters."""
    charges = system_params["charges"]
    n_atoms = len(charges)
    bonds = system_params.get("bonds")

    if bonds is None or len(bonds) == 0:
      return cls(
        idx_12_13=jnp.zeros((0, 2), dtype=jnp.int32),
        idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
        scale_14_elec=coulomb14scale,
        scale_14_vdw=lj14scale,
        n_atoms=n_atoms,
      )

    excl = topology.find_bonded_exclusions(bonds, n_atoms)

    # Combine 1-2 and 1-3 (both fully excluded usually)
    idx_12_13 = jnp.concatenate([excl.idx_12, excl.idx_13], axis=0)

    return cls(
      idx_12_13=idx_12_13,
      idx_14=excl.idx_14,
      scale_14_elec=coulomb14scale,
      scale_14_vdw=lj14scale,
      n_atoms=n_atoms,
    )


def make_neighbor_list_fn(
  displacement_fn: space.DisplacementFn,
  box_size: Array,
  cutoff: float,
  capacity_multiplier: float = 1.25,
  disable_cell_list: bool = False,
) -> Callable[..., partition.NeighborList]:
  """Creates a neighbor list function optimized for solvent systems."""
  return partition.neighbor_list(
    displacement_fn,
    box_size,
    r_cutoff=cutoff,
    dr_threshold=0.5,  # buffer distance for updates
    capacity_multiplier=capacity_multiplier,
    disable_cell_list=disable_cell_list,
    mask_self=True,
    fractional_coordinates=False,
  )


def compute_exclusion_mask_neighbor_list(
  exclusion_spec: ExclusionSpec,
  neighbor_idx: Array,
  n_atoms: int,
) -> tuple[Array, Array, Array]:
  r"""Compute exclusion masks for a neighbor list.

  Process:
  1.  **Map**: Transform sparse `ExclusionSpec` to per-atom padded arrays.
  2.  **Lookup**: Retrieve scale factors for each neighbor in the list.
  3.  **Hard Mask**: Identify neighbors that are not fully excluded (1-2/1-3).

  Args:
      exclusion_spec: Sparse specification of excluded pairs.
      neighbor_idx: (N, K) neighbor indices.
      n_atoms: Total atomic count.

  Returns:
      (mask_vdw, mask_elec, mask_hard) arrays of shape (N, K).
  """
  excl_indices, excl_scales_vdw, excl_scales_elec = map_exclusions_to_dense_padded(exclusion_spec)

  mask_vdw, mask_elec = get_neighbor_exclusion_scales(
    excl_indices, excl_scales_vdw, excl_scales_elec, neighbor_idx
  )

  # Hard mask: 1.0 if NOT fully excluded (1-2/1-3)
  # We can detect this if either scale is exactly 0.0
  mask_hard = jnp.where((mask_vdw > 0) | (mask_elec > 0), 1.0, 0.0)

  return mask_vdw, mask_elec, mask_hard


def map_exclusions_to_dense_padded(
  exclusion_spec: ExclusionSpec,
  max_exclusions: int = 32,
) -> tuple[Array, Array, Array]:
  r"""Map pair exclusions to per-atom padded arrays for efficient lookup.

  Process:
  1.  **Allocate**: Create (N, max_exclusions) arrays for indices and scales.
  2.  **Collect**: Group exclusions by the first atom in each pair.
  3.  **Populate**: Scatter entries into the padded arrays.

  Args:
      exclusion_spec: Sparse exclusion data.
      max_exclusions: Capacity per atom for excluded neighbors.

  Returns:
      (excl_indices, excl_scales_vdw, excl_scales_elec) arrays.
  """
  N = exclusion_spec.n_atoms

  # Initialize with -1 (invalid index)
  excl_indices = jnp.full((N, max_exclusions), -1, dtype=jnp.int32)
  scales_vdw = jnp.ones((N, max_exclusions), dtype=jnp.float32)
  scales_elec = jnp.ones((N, max_exclusions), dtype=jnp.float32)

  # Helper to scatter pairs into the arrays
  # Since we can't use dynamic loops easily in JAX to fill this without scan,
  # and this is setup code, we can use numpy or simple loops if not traced?
  # Actually, this should be done during `make_energy_fn`, so numpy is fine!

  # Convert JAX arrays to numpy for population if they are arrays
  idx_12_13 = np.array(exclusion_spec.idx_12_13)
  idx_14 = np.array(exclusion_spec.idx_14)

  # Use lists to collect per atom
  atom_excls = [[] for _ in range(N)]

  for i, j in idx_12_13:
    if i < N:
      atom_excls[i].append((int(j), 0.0, 0.0))
    if j < N:
      atom_excls[j].append((int(i), 0.0, 0.0))

  for i, j in idx_14:
    s_v = exclusion_spec.scale_14_vdw
    s_e = exclusion_spec.scale_14_elec
    if i < N:
      atom_excls[i].append((int(j), s_v, s_e))
    if j < N:
      atom_excls[j].append((int(i), s_v, s_e))

  # Fill arrays
  excl_indices_np = np.full((N, max_exclusions), -1, dtype=np.int32)
  scales_vdw_np = np.ones((N, max_exclusions), dtype=np.float32)
  scales_elec_np = np.ones((N, max_exclusions), dtype=np.float32)

  for i in range(N):
    entries = atom_excls[i]
    # Sort by index for determinism
    entries.sort(key=lambda x: x[0])
    n_entries = min(len(entries), max_exclusions)
    for k in range(n_entries):
      excl_indices_np[i, k] = entries[k][0]
      scales_vdw_np[i, k] = entries[k][1]
      scales_elec_np[i, k] = entries[k][2]

  return jnp.array(excl_indices_np), jnp.array(scales_vdw_np), jnp.array(scales_elec_np)


def get_neighbor_exclusion_scales(
  excl_indices: Array,
  excl_scales_vdw: Array,
  excl_scales_elec: Array,
  neighbor_idx: Array,
) -> tuple[Array, Array]:
  r"""Compute scale factors for neighbors based on exclusion lists.

  Process:
  1.  **Broadcast**: Align neighbor and exclusion indices for comparison.
  2.  **Compare**: Identify matches between neighbor list and exclusion list.
  3.  **Select**: Extract the associated scale factors where matches exist.
  4.  **Default**: Return 1.0 for neighbors not found in the exclusion list.

  Args:
      excl_indices: (N, M) padded exclusion indices.
      excl_scales_vdw, excl_scales_elec: (N, M) padded scales.
      neighbor_idx: (N, K) neighbor indices.

  Returns:
      (scale_vdw, scale_elec) arrays of shape (N, K).
  """
  N, K = neighbor_idx.shape
  M = excl_indices.shape[1]

  # Broadcast to (N, K, M)
  # neighbor_idx: (N, K, 1)
  # excl_indices: (N, 1, M)

  n_idx_broad = neighbor_idx[:, :, None]
  excl_idx_broad = excl_indices[:, None, :]  # (N, 1, M)

  # Check for matches
  # match: (N, K, M) boolean
  # neighbor index == excluded index
  match = (n_idx_broad == excl_idx_broad) & (excl_idx_broad != -1)

  # If any match is found for a neighbor (k), we take the scale.
  # Since an atom appears at most once in exclusion list, max() works.

  # However, if NO match, scale should be 1.0.
  # Current arrays have scales, but we need to select them.

  # Extract scales where match is True
  # (N, K, M)
  vdw_matches = match * excl_scales_vdw[:, None, :]
  elec_matches = match * excl_scales_elec[:, None, :]

  # Sum over M (should be at most one match)
  # If no match, sum is 0.0. But we want 1.0 for non-excluded.
  # So we need to detect IF there was a match.

  has_match = jnp.any(match, axis=2)  # (N, K)

  sum_vdw = jnp.sum(vdw_matches, axis=2)  # (N, K)
  sum_elec = jnp.sum(elec_matches, axis=2)  # (N, K)

  final_vdw = jnp.where(has_match, sum_vdw, 1.0)
  final_elec = jnp.where(has_match, sum_elec, 1.0)

  return final_vdw, final_elec
