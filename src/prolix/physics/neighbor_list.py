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
  dr_threshold: float = 0.5,
) -> Callable[..., partition.NeighborList]:
  """Creates a neighbor list function optimized for solvent systems.

  Positions are wrapped into ``[0, box)`` before being handed to jax_md
  (backlog #4699). This is required for correctness, not just tidiness:
  ``jax_md.partition.cell_list_fn`` bins particles with

      indices = jnp.array(position / cell_size, dtype=i32)   # partition.py:391

  which truncates *toward zero* with no wrapping and no bounds check, so the
  cell list assumes positions already lie inside ``[0, box)``. prolix passes
  raw frame coordinates, which for a solvated protein are typically centred on
  the origin -- a negative coordinate then aliases onto the wrong cell
  (``int32(-0.4) == int32(0.4) == 0``) and a coordinate past ``L`` hashes
  outside ``cell_count``. Either way those particles' true neighbours are
  never enumerated and the pairs vanish with no error.

  Measured on solvated 1vii (43.3 A box, coords spanning -22.1 to +23.1):
  the cell list dropped 207,717 of 1,059,926 in-cutoff pairs (**19.6%**);
  with wrapping it returns exactly the brute-force candidate set (missing 0).
  See ``tests/physics/test_cell_list_pair_completeness.py``.

  Wrapping is physically a no-op for a periodic system -- it selects a
  different periodic image of the same configuration -- and is invisible to
  the neighbour list's own rebuild check, which measures displacement through
  the periodic (minimum-image) ``displacement_fn``. The returned ``idx`` is a
  list of atom *indices*, so callers may continue to evaluate forces with
  their original unwrapped coordinates.

  With the precondition now enforced, the cell list is safe and is enabled by
  default; ``disable_cell_list=True`` falls back to O(N^2) construction.
  """
  fns = partition.neighbor_list(
    displacement_fn,
    box_size,
    r_cutoff=cutoff,
    dr_threshold=dr_threshold,
    capacity_multiplier=capacity_multiplier,
    disable_cell_list=disable_cell_list,
    mask_self=True,
    fractional_coordinates=False,
  )

  box = jnp.asarray(box_size)

  def _wrap(position: Array) -> Array:
    return jnp.mod(position, box)

  def allocate(position: Array, **kwargs) -> partition.NeighborList:
    return fns.allocate(_wrap(position), **kwargs)

  def update(
    position: Array, neighbors: partition.NeighborList, **kwargs
  ) -> partition.NeighborList:
    return fns.update(_wrap(position), neighbors, **kwargs)

  return partition.NeighborListFns(allocate, update)


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
      max_exclusions: Capacity per atom for excluded neighbors. If any atom
          needs more slots, this raises (no silent truncation).

  Returns:
      (excl_indices, excl_scales_vdw, excl_scales_elec) arrays.
  """
  needed = max_exclusion_slots_needed(exclusion_spec)
  if needed > int(max_exclusions):
    raise ValueError(
        f"exclusion topology needs {needed} slots per atom, "
        f"max_exclusions={max_exclusions}"
    )
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

  max_needed = max((len(x) for x in atom_excls), default=0)
  if max_needed > int(max_exclusions):
    raise ValueError(
        f"exclusion topology needs {max_needed} slots per atom, "
        f"max_exclusions={max_exclusions}"
    )

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

  Uses a branchless $\lceil \log_2 M \rceil$-step binary search (5 steps for
  ``M = 32``) over sorted per-atom exclusion rows ``excl_indices``, avoiding
  both the $(N, K, M)$ broadcast tensor and the 32-step linear scan.

  Args:
      excl_indices: (N, M) padded exclusion indices (sorted ascending per row,
          unused slots padded with -1).
      excl_scales_vdw, excl_scales_elec: (N, M) padded scales.
      neighbor_idx: (N, K) neighbor indices.

  Returns:
      (scale_vdw, scale_elec) arrays of shape (N, K).
  """
  N, K = neighbor_idx.shape
  M = int(excl_indices.shape[1]) if excl_indices.ndim == 2 else 0
  vdw_dtype = excl_scales_vdw.dtype
  elec_dtype = excl_scales_elec.dtype

  ones_vdw = jnp.ones((N, K), dtype=vdw_dtype)
  ones_elec = jnp.ones((N, K), dtype=elec_dtype)
  if M == 0:
    return ones_vdw, ones_elec

  # Map -1 padding to INT_MAX so each row is monotonically non-decreasing.
  int_max = jnp.int32(2**30)
  excl_sorted = jnp.where(excl_indices >= 0, excl_indices.astype(jnp.int32), int_max)
  nbr_i32 = neighbor_idx.astype(jnp.int32)

  pos = jnp.zeros((N, K), dtype=jnp.int32)
  if M > 1:
    step = 1 << ((M - 1).bit_length() - 1)
    while step > 0:
      cand = pos + step
      cand_safe = jnp.minimum(cand, M - 1)
      val = jnp.take_along_axis(excl_sorted, cand_safe, axis=1)
      pos = jnp.where((cand < M) & (val <= nbr_i32), cand_safe, pos)
      step >>= 1

  matched_idx = jnp.take_along_axis(excl_indices, pos, axis=1)
  is_match = (matched_idx == nbr_i32) & (matched_idx >= 0) & (nbr_i32 < N)

  gathered_vdw = jnp.take_along_axis(excl_scales_vdw, pos, axis=1)
  gathered_elec = jnp.take_along_axis(excl_scales_elec, pos, axis=1)

  final_vdw = jnp.where(is_match, gathered_vdw, ones_vdw)
  final_elec = jnp.where(is_match, gathered_elec, ones_elec)
  return final_vdw, final_elec


def compute_nl_capacity(
  atom_bucket_size: int,
  box_size,
  cutoff: float,
  dr_threshold: float = 0.5,
  safety_factor: float = 1.5,
  round_to: int = 256,
) -> int:
  r"""Static neighbor-list capacity for a shape-class, from bucket constants only.

  Debt 760 (B1 connect-existing-NL-PME, `.praxia/docs/decisions/
  260717_b1-connect-existing-engines-scope.md`, decision D1): a single
  capacity per shape-class -- derived only from static, bucket-level
  quantities (padded atom count, box volume, cutoff), never from any
  particular bundle's actual measured occupancy -- so that two different
  real proteins sharing a shape bucket also always share the same neighbor-
  list capacity. Capacity computed per-bundle (e.g. via
  ``neighbor_fn.allocate()``'s own automatic sizing on real, per-protein
  positions) would fragment the cross-protein compile-sharing this session
  spent real effort confirming, since different proteins' local density can
  differ enough to pick different capacities even within the same atom
  bucket.

  Formula: ``K = ceil_to(round_to, safety_factor * rho_padded *
  (4/3) * pi * (cutoff + dr_threshold)**3)``, where ``rho_padded =
  atom_bucket_size / box_volume`` -- uniform padded-atom density times the
  volume of the effective neighbor-list sphere (cutoff + Verlet skin),
  times a safety margin. Ghost/padding atoms must already be spread
  uniformly across the box (debt 772, ``_pad_positions_with_ghost_lattice``
  in ``system.py``) for this density-based estimate to be valid -- it
  assumes no pathological clustering.

  Verified against real measurements (Opus-backed investigation, same
  decision doc): for the 5000-atom/32A-box/9.0A-cutoff shape class, this
  formula gives 822 (pre-rounding); jax_md's own `.allocate()` on the real,
  ghost-lattice-fixed 1VII bundle measured a required capacity of 895 (max
  occupancy 716) at the same cutoff -- comfortably under the ``round_to=256``
  rounded value of 1024, confirming the formula's safety margin holds in
  practice, not just in theory.

  Args:
      atom_bucket_size: Padded atom count for this shape class (e.g.
          `MolecularBundle.positions.shape[0]`, or equivalently the bucket
          size `ATOM_BUCKETS` resolved to for this class).
      box_size: (3,) box edge lengths (periodic systems only -- this
          function assumes a periodic box; callers must not call it for
          non-periodic bundles).
      cutoff: Direct-space nonbonded cutoff distance (Angstrom).
      dr_threshold: Verlet-list skin (Angstrom) -- matches jax_md's
          `dr_threshold` parameter to `partition.neighbor_list`; must be the
          same value used when actually allocating the neighbor list, or the
          capacity estimate and the real allocation diverge.
      safety_factor: Multiplicative headroom over the uniform-density
          estimate, to absorb the measured spread between different real
          proteins' local density within the same shape bucket (~11% in the
          1VII/2GB1 case) plus general safety margin.
      round_to: Round the final capacity up to a multiple of this value.
          256 (not 128) by default: aligns with both the O(N*K) direct-space
          kernels' 128-atom tile size (`optimization.py`) and
          `flash_explicit.py`'s 256-atom tile size (debt 763's tile-size-
          divisibility concern applies to K too, not just N).

  Returns:
      Static neighbor-list capacity K (Python int).
  """
  import math

  import numpy as np

  box_np = np.asarray(box_size, dtype=np.float64)
  box_volume = float(np.prod(box_np))
  if box_volume <= 0:
    msg = f"compute_nl_capacity: box_size must describe a positive-volume periodic box, got {box_size}"
    raise ValueError(msg)

  rho_padded = atom_bucket_size / box_volume
  sphere_volume = (4.0 / 3.0) * math.pi * (cutoff + dr_threshold) ** 3
  k_raw = safety_factor * rho_padded * sphere_volume

  return int(math.ceil(k_raw / round_to) * round_to)
