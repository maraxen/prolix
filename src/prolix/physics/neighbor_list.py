"""Neighbor list management and sparse exclusion handling."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space, util

from prolix.utils import topology

if TYPE_CHECKING:
  from collections.abc import Callable

  from proxide.core.containers import Protein

  from prolix.typing import SystemParams

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

  # Per-pair explicit exception data (from AMBER FF XML, row-aligned with pairs_14).
  # ALL pairs_14 are excluded from the main kernel (in idx_12_13).
  # These arrays hold only the nonzero-epsilon subset for the separate exception energy term.
  exception_pairs: Array        # (E, 2) int32
  exception_sigmas: Array       # (E,)   float32
  exception_epsilons: Array     # (E,)   float32
  exception_chargeprods: Array  # (E,)   float32

  @classmethod
  def from_protein(
    cls,
    protein: Protein,
    coulomb14scale: float | None = None,
    lj14scale: float | None = None,
  ) -> ExclusionSpec:
    """Build ExclusionSpec from standard protein structure.

    Handles explicit 1-4 exceptions from AMBER FF (e.g., Proline rings)
    by reading protein.pairs_14 and protein.exception_14_params if available.
    """
    charges = protein.charges
    if charges is None:
      raise ValueError("Protein must have charges to build ExclusionSpec")

    n_atoms = len(charges)
    bonds = protein.bonds

    # Static scales from protein if not provided
    c14 = coulomb14scale if coulomb14scale is not None else (
        protein.coulomb14scale if protein.coulomb14scale is not None else 0.83333333
    )
    l14 = lj14scale if lj14scale is not None else (
        protein.lj14scale if protein.lj14scale is not None else 0.5
    )

    if bonds is None or len(bonds) == 0:
      return cls(
        idx_12_13=jnp.zeros((0, 2), dtype=jnp.int32),
        idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
        scale_14_elec=c14,
        scale_14_vdw=l14,
        n_atoms=n_atoms,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=jnp.float32),
        exception_epsilons=jnp.zeros((0,), dtype=jnp.float32),
        exception_chargeprods=jnp.zeros((0,), dtype=jnp.float32),
      )

    excl = topology.find_bonded_exclusions(bonds, n_atoms)

    # Combine 1-2 and 1-3 (both fully excluded usually)
    idx_12_13 = jnp.concatenate([excl.idx_12, excl.idx_13], axis=0)

    exc_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
    exc_sigmas = jnp.zeros((0,), dtype=jnp.float32)
    exc_epsilons = jnp.zeros((0,), dtype=jnp.float32)
    exc_chargeprods = jnp.zeros((0,), dtype=jnp.float32)

    # When proxide supplies resolved per-pair 1-4 params, use them exclusively.
    # All pairs_14 are excluded (scale=0) from the main LJ/Coulomb kernels, and
    # their energy is computed by make_exception_pair_energy_fn instead.
    # We also clear idx_14 so the same pairs don't get re-added with the global
    # scale factors and cause double-counting.
    use_explicit_14 = (
      hasattr(protein, "pairs_14") and protein.pairs_14 is not None
      and len(protein.pairs_14) > 0
      and hasattr(protein, "exception_14_params") and protein.exception_14_params is not None
      and len(protein.exception_14_params) > 0
    )

    if use_explicit_14:
      pairs_np = np.asarray(protein.pairs_14, dtype=np.int32)       # (N, 2)
      params_np = np.asarray(protein.exception_14_params, dtype=np.float32)  # (N, 3)

      # All pairs_14 → fully excluded from main LJ/Coulomb kernels
      idx_12_13 = jnp.concatenate([idx_12_13, jnp.asarray(pairs_np, dtype=jnp.int32)], axis=0)

      # ALL pairs_14 → exception energy (handles both LJ and Coulomb per-pair)
      exc_pairs = jnp.asarray(pairs_np, dtype=jnp.int32)
      exc_chargeprods = jnp.asarray(params_np[:, 0], dtype=jnp.float32)
      exc_sigmas = jnp.asarray(params_np[:, 1], dtype=jnp.float32)
      exc_epsilons = jnp.asarray(params_np[:, 2], dtype=jnp.float32)

    # When explicit pairs_14 are available, clear idx_14 to avoid double-counting:
    # the topology-derived 1-4 pairs are the same set as pairs_14 and their energy
    # is now fully handled by the exception energy function.
    idx_14_out = jnp.zeros((0, 2), dtype=jnp.int32) if use_explicit_14 else excl.idx_14

    return cls(
      idx_12_13=idx_12_13,
      idx_14=idx_14_out,
      scale_14_elec=c14,
      scale_14_vdw=l14,
      n_atoms=n_atoms,
      exception_pairs=exc_pairs,
      exception_sigmas=exc_sigmas,
      exception_epsilons=exc_epsilons,
      exception_chargeprods=exc_chargeprods,
    )

  @classmethod
  def from_system_params(
    cls, system_params: SystemParams, coulomb14scale: float = 0.83333333, lj14scale: float = 0.5
  ) -> ExclusionSpec:
    """Build ExclusionSpec from standard system parameters (deprecated)."""
    from prolix.compat import system_params_to_protein

    protein = system_params_to_protein(system_params)
    return cls.from_protein(protein, coulomb14scale=coulomb14scale, lj14scale=lj14scale)


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


def max_exclusion_slots_needed(exclusion_spec: ExclusionSpec) -> int:
  """Maximum exclusion list length required for any single atom (before padding cap).

  ``map_exclusions_to_dense_padded`` truncates to ``max_exclusions`` (default 32). If
  this function returns a value greater than that cap, exclusion data will be
  silently dropped for that atom — call sites should assert or raise.

  Args:
      exclusion_spec: Sparse 1–2 / 1–3 / 1–4 pair lists from ``ExclusionSpec``.

  Returns:
      Maximum number of exclusion slots any atom needs.
  """
  n = exclusion_spec.n_atoms
  counts = np.zeros(n, dtype=np.int32)
  for i, j in np.asarray(exclusion_spec.idx_12_13):
    if 0 <= i < n:
      counts[i] += 1
    if 0 <= j < n:
      counts[j] += 1
  for i, j in np.asarray(exclusion_spec.idx_14):
    if 0 <= i < n:
      counts[i] += 1
    if 0 <= j < n:
      counts[j] += 1
  return int(np.max(counts)) if n > 0 else 0


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
      max_exclusions: Capacity per atom for excluded neighbors. If too small,
          entries are truncated — compare :func:`max_exclusion_slots_needed` first.

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


def pair_list_to_dense_padded(
  pair_indices: Array,
  pair_scales_vdw: Array,
  pair_scales_elec: Array,
  pair_mask: Array,
  n_atoms: int,
  max_exclusions: int = 32,
) -> tuple[Array, Array, Array]:
  r"""Convert a bundle-style pair-list exclusion set to per-atom padded arrays.

  Same output contract as :func:`map_exclusions_to_dense_padded` (the
  ``PhysicsSystem.excl_indices``/``excl_scales_vdw``/``excl_scales_elec``
  ``(N, max_exclusions)`` layout every neighbor-list/flash kernel in this
  codebase expects — ``system.py:152-153``, ``optimization.py``'s
  ``chunked_*_nl``, ``flash_explicit.py``, ``flash_nonbonded.py``), but takes
  an already-built pair list (``(E, 2)`` indices + ``(E,)`` per-pair scales +
  ``(E,)`` validity mask — the format ``MolecularBundle``/
  ``physics_system_from_bundle`` already carry) instead of an
  :class:`ExclusionSpec`, so bundle-derived systems don't need one
  reconstructed just for this.

  Host-side (numpy) by construction, same rationale as
  ``map_exclusions_to_dense_padded``: exclusion topology is a per-bundle
  constant fixed at construction time, never traced.

  Args:
      pair_indices: (E, 2) int, atom index pairs. Padding-row values are
          ignored via ``pair_mask``, not via a sentinel value in this array.
      pair_scales_vdw, pair_scales_elec: (E,) float, per-pair scale factors
          (0.0 = fully excluded, e.g. 0.5/1-1.2 = 1-4 scaled, matching the
          convention already used by ``_pme_exclusion_correction_from_pairs``).
      pair_mask: (E,) bool, True for real (non-padding) pair slots.
      n_atoms: N_padded — output arrays are always exactly this many rows,
          regardless of how many real atoms are populated (matches
          ``PhysicsSystem.n_padded_atoms``).
      max_exclusions: Capacity per atom. If a real atom needs more than this,
          entries are silently truncated (same caveat as
          ``map_exclusions_to_dense_padded`` — call sites needing a hard
          guarantee should check counts against this cap beforehand).

  Returns:
      (excl_indices, excl_scales_vdw, excl_scales_elec), each
      ``(n_atoms, max_exclusions)``, dtype int32/float32/float32.
  """
  idx_np = np.asarray(pair_indices)
  vdw_np = np.asarray(pair_scales_vdw)
  elec_np = np.asarray(pair_scales_elec)
  mask_np = np.asarray(pair_mask)

  atom_excls: list[list[tuple[int, float, float]]] = [[] for _ in range(n_atoms)]
  for k in range(idx_np.shape[0]):
    if not mask_np[k]:
      continue
    i, j = int(idx_np[k, 0]), int(idx_np[k, 1])
    sv, se = float(vdw_np[k]), float(elec_np[k])
    if 0 <= i < n_atoms:
      atom_excls[i].append((j, sv, se))
    if 0 <= j < n_atoms:
      atom_excls[j].append((i, sv, se))

  excl_indices_np = np.full((n_atoms, max_exclusions), -1, dtype=np.int32)
  scales_vdw_np = np.ones((n_atoms, max_exclusions), dtype=np.float32)
  scales_elec_np = np.ones((n_atoms, max_exclusions), dtype=np.float32)

  for i in range(n_atoms):
    entries = atom_excls[i]
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

  # Optimization: Avoid creating (N, K, M) tensor which is huge (O(N^2) effectively).
  # Instead, use a loop over M (exclusions) to update masks.
  # Since M is small (32), this is efficient and memory scaling is O(N*K).

  final_vdw = jnp.ones((N, K), dtype=jnp.float32)
  final_elec = jnp.ones((N, K), dtype=jnp.float32)

  # Prepare exclusion data (N, M)
  # broadcasting to (N, K) per iteration

  def loop_body(i, carrier):
    curr_vdw, curr_elec = carrier

    # Get i-th exclusion data for all atoms
    excl_idx_i = excl_indices[:, i]  # (N,)
    scale_v_i = excl_scales_vdw[:, i]  # (N,)
    scale_e_i = excl_scales_elec[:, i]  # (N,)

    # Broadcast to neighbors (N, K)
    # Check if neighbor_idx matches this exclusion
    # excl_idx_i[:, None] -> (N, 1)
    is_match = (neighbor_idx == excl_idx_i[:, None]) & (excl_idx_i[:, None] != -1)

    # Update scales where match found
    # If match, take the scale. Else keep current value.
    # Note: An atom appears only once in exclusion set for a given pair,
    # so we can just overwrite or multiply (since default is 1.0 and collisions only happen on 1.0)
    curr_vdw = jnp.where(is_match, scale_v_i[:, None], curr_vdw)
    curr_elec = jnp.where(is_match, scale_e_i[:, None], curr_elec)

    return curr_vdw, curr_elec

  # We can unroll this loop if M is small constant, or use fori_loop.
  # Since M is padded to 32, fori_loop is good to keep graph small?
  # Or simple range loop which unrolls in Python (better for constant folding if M small).
  # Let's use jax.lax.fori_loop for strictly static graph size,
  # or standard python loop if we suspect it helps.
  # For 32 iters, fori_loop is cleaner.

  final_vdw, final_elec = jax.lax.fori_loop(0, M, loop_body, (final_vdw, final_elec))

  return final_vdw, final_elec
