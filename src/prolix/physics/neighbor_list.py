"""Neighbor list management and sparse exclusion handling."""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax_md import partition, space, util

from prolix.compat import system_params_to_protein
from prolix.export import in_export_context
from prolix.physics.pbc import wrap_positions
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

    protein = system_params_to_protein(system_params)
    return cls.from_protein(protein, coulomb14scale=coulomb14scale, lj14scale=lj14scale)


def _should_use_pallas_nl_default(use_pallas_nl: bool | None = None) -> bool:
  if in_export_context():
    return False
  if use_pallas_nl is not None:
    return bool(use_pallas_nl)
  env = os.environ.get("PROLIX_USE_PALLAS", "").strip().lower()
  if env in ("0", "false", "no", "off"):
    return False
  if env in ("1", "true", "yes", "on"):
    return True
  return jax.default_backend() != "cpu"


def make_neighbor_list_fn(
  displacement_fn: space.DisplacementFn,
  box_size: Array,
  cutoff: float,
  capacity_multiplier: float = 1.25,
  disable_cell_list: bool = False,
  dr_threshold: float = 0.5,
  *,
  excl_indices: Array | None = None,
  excl_scales_vdw: Array | None = None,
  excl_scales_elec: Array | None = None,
  scale_14_vdw: float = 0.5,
  scale_14_elec: float = 0.83333333,
  use_pallas_nl: bool | None = None,
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
  box = jnp.asarray(box_size)
  box_diag = jnp.diag(box) if box.ndim == 2 else box
  if not isinstance(box, jax.core.Tracer):
    box_np = np.asarray(box)
    diag_np = np.diag(box_np) if box_np.ndim == 2 else box_np
    min_half_height = 0.5 * float(np.min(diag_np))
    if float(cutoff) > min_half_height + 1e-6:
      raise ValueError(
        f"Minimum-image violation: cutoff ({cutoff}) exceeds half the minimum box height ({min_half_height})"
      )
    if box_np.ndim == 2:
      if (
        abs(float(box_np[1, 0])) > 0.5 * float(box_np[0, 0]) + 1e-5
        or abs(float(box_np[2, 0])) > 0.5 * float(box_np[0, 0]) + 1e-5
        or abs(float(box_np[2, 1])) > 0.5 * float(box_np[1, 1]) + 1e-5
      ):
        raise ValueError(
          f"Triclinic box violates OpenMM reduced-basis conditions: {box_np}"
        )

  fns = partition.neighbor_list(
    displacement_fn,
    box_diag,
    r_cutoff=cutoff,
    dr_threshold=dr_threshold,
    capacity_multiplier=capacity_multiplier,
    disable_cell_list=disable_cell_list or (box.ndim == 2),
    mask_self=True,
    fractional_coordinates=False,
  )

  has_excl = (
    excl_indices is not None
    and excl_scales_vdw is not None
    and excl_scales_elec is not None
  )
  r_total_cutoff = float(cutoff) + float(dr_threshold)
  half_skin_sq = float((0.5 * float(dr_threshold)) ** 2)
  static_grid_dims = (
    None
    if isinstance(box_diag, jax.core.Tracer)
    else _compute_static_grid_dims(box_diag, r_total_cutoff)
  )

  def _wrap(position: Array, dyn_box: Array | None = None) -> Array:
    b = jnp.asarray(dyn_box) if dyn_box is not None else box
    if b.ndim == 2:
      return wrap_positions(position, b)
    return jnp.mod(position, b)

  def _encode_if_needed(idx: Array, n_atoms: int) -> Array:
    if not has_excl:
      return idx
    return encode_nl_exclusions(
      idx,
      excl_indices,
      excl_scales_vdw,
      excl_scales_elec,
      n_atoms=n_atoms,
      scale_14_vdw=scale_14_vdw,
      scale_14_elec=scale_14_elec,
    )

  def _rebuild_unconditional(
    position: Array, neighbors: partition.NeighborList, **kwargs
  ) -> partition.NeighborList:
    b = kwargs.get("box", None)
    b_eff = jnp.asarray(b) if b is not None else box
    pos_w = _wrap(position, b)
    can_pallas = (
      not disable_cell_list
      and b_eff.ndim == 1
      and getattr(neighbors, "cell_list_fn", None) is not None
      and getattr(neighbors, "cell_list_capacity", None) is not None
      and _should_use_pallas_nl_default(use_pallas_nl)
    )
    if can_pallas:
      N_atoms = int(pos_w.shape[-2])
      ei = excl_indices if has_excl else jnp.full((N_atoms, 1), -1, dtype=jnp.int32)
      ev = excl_scales_vdw if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
      ee = excl_scales_elec if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
      idx_new, occ_new = pallas_prune_and_encode_cell_list(
        pos_w,
        b_eff,
        ei,
        ev,
        ee,
        cell_size=float(neighbors.cell_size),
        cell_capacity=int(neighbors.cell_list_capacity),
        max_occupancy=int(neighbors.max_occupancy),
        r_cutoff=r_total_cutoff,
        scale_14_vdw=float(scale_14_vdw),
        scale_14_elec=float(scale_14_elec),
        encode_exclusions=has_excl,
        interpret=(jax.default_backend() == "cpu"),
        grid_dims=static_grid_dims,
      )
      err_new = neighbors.error.update(
        partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW,
        occ_new > int(neighbors.max_occupancy),
      )
      return dataclasses.replace(
        neighbors,
        idx=idx_new,
        reference_position=pos_w,
        error=err_new,
        update_fn=update,
      )
    # Fallback to jax_md update; force rebuild by shifting reference_position
    b_diag = jnp.diag(b_eff) if b_eff.ndim == 2 else b_eff
    forced_ref = pos_w + jnp.asarray(0.5, dtype=pos_w.dtype) * b_diag.astype(pos_w.dtype)
    nbr_forced = dataclasses.replace(neighbors, reference_position=forced_ref)
    nbr = fns.update(pos_w, nbr_forced, **kwargs)
    idx_enc = _encode_if_needed(nbr.idx, int(pos_w.shape[-2]))
    return dataclasses.replace(nbr, idx=idx_enc, update_fn=update)

  def update(
    position: Array, neighbors: partition.NeighborList, **kwargs
  ) -> partition.NeighborList:
    b = kwargs.get("box", None)
    b_eff = jnp.asarray(b) if b is not None else box
    pos_w = _wrap(position, b)
    can_pallas = (
      not disable_cell_list
      and b_eff.ndim == 1
      and getattr(neighbors, "cell_list_fn", None) is not None
      and getattr(neighbors, "cell_list_capacity", None) is not None
      and _should_use_pallas_nl_default(use_pallas_nl)
    )
    if can_pallas or has_excl:
      cell_size_f = float(neighbors.cell_size) if neighbors.cell_size is not None else r_total_cutoff
      cell_cap_i = int(neighbors.cell_list_capacity) if neighbors.cell_list_capacity is not None else 64
      max_occ_i = int(neighbors.max_occupancy)

      @jax.custom_batching.custom_vmap
      def _cond_rebuild(pos_in: Array, ref_in: Array, idx_in: Array, err_code_in: Array, box_in: Array):
        dr = jax.vmap(displacement_fn)(pos_in, ref_in)
        dr_sq = jnp.sum(dr * dr, axis=-1)
        should_upd = jnp.any(dr_sq > jnp.asarray(half_skin_sq, dtype=pos_in.dtype))

        def _do_one(operands):
          p_o, _r_o, _i_o, e_o, bx_o = operands
          nbr_tmp = dataclasses.replace(
            neighbors,
            idx=_i_o,
            reference_position=_r_o,
            error=partition.PartitionError(e_o),
          )
          if can_pallas:
            N_atoms = int(p_o.shape[0])
            ei = excl_indices if has_excl else jnp.full((N_atoms, 1), -1, dtype=jnp.int32)
            ev = excl_scales_vdw if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            ee = excl_scales_elec if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            idx_new, occ_new = pallas_prune_and_encode_cell_list(
              p_o,
              bx_o,
              ei,
              ev,
              ee,
              cell_size=cell_size_f,
              cell_capacity=cell_cap_i,
              max_occupancy=max_occ_i,
              r_cutoff=r_total_cutoff,
              scale_14_vdw=float(scale_14_vdw),
              scale_14_elec=float(scale_14_elec),
              encode_exclusions=has_excl,
              interpret=(jax.default_backend() == "cpu"),
              grid_dims=static_grid_dims,
            )
            err_new = nbr_tmp.error.update(
              partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW,
              occ_new > max_occ_i,
            )
            return idx_new, p_o, err_new.code
          b_diag = jnp.diag(bx_o) if bx_o.ndim == 2 else bx_o
          forced_ref = p_o + jnp.asarray(0.5, dtype=p_o.dtype) * b_diag.astype(p_o.dtype)
          nbr_forced = dataclasses.replace(nbr_tmp, reference_position=forced_ref)
          nbr_out = fns.update(p_o, nbr_forced, **kwargs)
          idx_enc = _encode_if_needed(nbr_out.idx, int(p_o.shape[0]))
          return idx_enc, nbr_out.reference_position, nbr_out.error.code

        def _keep_one(operands):
          _p_o, r_o, i_o, e_o, _bx_o = operands
          return i_o, r_o, e_o

        return jax.lax.cond(
          should_upd,
          _do_one,
          _keep_one,
          (pos_in, ref_in, idx_in, err_code_in, box_in),
        )

      @_cond_rebuild.def_vmap
      def _cond_rebuild_vmap(axis_size, in_batched, pos_in, ref_in, idx_in, err_code_in, box_in):
        B = int(axis_size)
        pos_b = pos_in if in_batched[0] else jnp.broadcast_to(pos_in[None, ...], (B,) + pos_in.shape)
        ref_b = ref_in if in_batched[1] else jnp.broadcast_to(ref_in[None, ...], (B,) + ref_in.shape)
        idx_b = idx_in if in_batched[2] else jnp.broadcast_to(idx_in[None, ...], (B,) + idx_in.shape)
        err_b = err_code_in if in_batched[3] else jnp.broadcast_to(err_code_in[None, ...], (B,) + err_code_in.shape)
        bx_b = box_in if in_batched[4] else jnp.broadcast_to(box_in[None, ...], (B,) + box_in.shape)

        dr_b = jax.vmap(jax.vmap(displacement_fn))(pos_b, ref_b)
        dr_sq_b = jnp.sum(dr_b * dr_b, axis=-1)
        any_upd = jnp.any(dr_sq_b > jnp.asarray(half_skin_sq, dtype=pos_b.dtype))

        def _do_batch(operands):
          p_b, _r_b, _i_b, e_b, b_b = operands
          if can_pallas:
            N_atoms = int(p_b.shape[1])
            ei = excl_indices if has_excl else jnp.full((N_atoms, 1), -1, dtype=jnp.int32)
            ev = excl_scales_vdw if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            ee = excl_scales_elec if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            idx_new_b, occ_new_b = _pallas_prune_and_encode_cell_list_batch(
              B,
              (True, True, False, False, False),
              p_b,
              b_b,
              ei,
              ev,
              ee,
              cell_size=cell_size_f,
              cell_capacity=cell_cap_i,
              max_occupancy=max_occ_i,
              r_cutoff=r_total_cutoff,
              scale_14_vdw=float(scale_14_vdw),
              scale_14_elec=float(scale_14_elec),
              encode_exclusions=has_excl,
              interpret=(jax.default_backend() == "cpu"),
              grid_dims=static_grid_dims,
            )
            overflow_bit = jnp.uint8(partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW)
            err_new_b = jnp.where(occ_new_b > max_occ_i, e_b | overflow_bit, e_b)
            return idx_new_b, p_b, err_new_b

          def _single_fallback(p_1, r_1, i_1, e_1, bx_1):
            b_diag = jnp.diag(bx_1) if bx_1.ndim == 2 else bx_1
            forced_ref = p_1 + jnp.asarray(0.5, dtype=p_1.dtype) * b_diag.astype(p_1.dtype)
            nbr_tmp = dataclasses.replace(
              neighbors,
              idx=i_1,
              reference_position=forced_ref,
              error=partition.PartitionError(e_1),
            )
            nbr_out = fns.update(p_1, nbr_tmp, **kwargs)
            idx_enc = _encode_if_needed(nbr_out.idx, int(p_1.shape[0]))
            return idx_enc, nbr_out.reference_position, nbr_out.error.code

          return jax.vmap(_single_fallback)(p_b, _r_b, _i_b, e_b, b_b)

        def _keep_batch(operands):
          _p_b, r_b, i_b, e_b, _b_b = operands
          return i_b, r_b, e_b

        out_idx_b, out_ref_b, out_err_b = jax.lax.cond(
          any_upd,
          _do_batch,
          _keep_batch,
          (pos_b, ref_b, idx_b, err_b, bx_b),
        )
        return (out_idx_b, out_ref_b, out_err_b), (True, True, True)

      idx_out, ref_out, err_code_out = _cond_rebuild(
        pos_w,
        neighbors.reference_position,
        neighbors.idx,
        neighbors.error.code,
        b_eff,
      )
      return dataclasses.replace(
        neighbors,
        idx=idx_out,
        reference_position=ref_out,
        error=partition.PartitionError(err_code_out),
        update_fn=update,
      )

    nbr = fns.update(pos_w, neighbors, **kwargs)
    return dataclasses.replace(nbr, update_fn=update)

  update._pre_encoded_exclusions = has_excl  # type: ignore[attr-defined]
  update._rebuild_unconditional = _rebuild_unconditional  # type: ignore[attr-defined]

  def allocate(position: Array, **kwargs) -> partition.NeighborList:
    b = kwargs.get("box", None)
    pos_w = _wrap(position, b)
    nbr = fns.allocate(pos_w, **kwargs)
    idx_enc = _encode_if_needed(nbr.idx, int(pos_w.shape[-2]))
    return dataclasses.replace(nbr, idx=idx_enc, update_fn=update)

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
  if box_np.ndim == 2:
    box_np = np.diag(box_np)
  box_volume = float(np.prod(box_np))
  if box_volume <= 0:
    msg = f"compute_nl_capacity: box_size must describe a positive-volume periodic box, got {box_size}"
    raise ValueError(msg)

  rho_padded = atom_bucket_size / box_volume
  sphere_volume = (4.0 / 3.0) * math.pi * (cutoff + dr_threshold) ** 3
  k_raw = safety_factor * rho_padded * sphere_volume

  return int(math.ceil(k_raw / round_to) * round_to)


def build_dense_exclusion_scales(
    excl_indices: Array,
    excl_scales: Array,
    N: int,
) -> Array:
    """Convert sparse (N, M) exclusion arrays to dense (N, N) scale matrix.

    Uses fori_loop over M (typically 32) exclusion slots. Each iteration
    scatters one column of exclusion data into the N×N output.

    Returns:
        (N, N) float32 scale matrix. 1.0 = full interaction, 0.0 = excluded,
        0.5 = 1-4 scaled (for vdw), 1/1.2 = 1-4 scaled (for elec).
    """
    M = excl_indices.shape[1]
    scale = jnp.ones((N, N), dtype=jnp.float32)

    row_idx = jnp.arange(N)

    def body(i, s):
        idx_col = excl_indices[:, i]   # (N,)
        sc = excl_scales[:, i]         # (N,)
        valid = idx_col >= 0           # (N,) bool
        safe_col = jnp.where(valid, idx_col, 0)
        # Scatter: for each row n, set s[n, safe_col[n]] = sc[n] if valid
        new_vals = jnp.where(valid, sc, s[row_idx, safe_col])
        s = s.at[row_idx, safe_col].set(new_vals)
        return s

    return jax.lax.fori_loop(0, M, body, scale)


# Backward-compatible private alias (previously defined in ``prolix.batched_energy``).
_build_dense_exclusion_scales = build_dense_exclusion_scales

# =============================================================================
# O1.1 & O5: Lossless 2-Bit Neighbor-List Exclusion Tagging & 3D Morton Sorting
# =============================================================================

_NB_EXCL_FULL_BIT: int = 1 << 30
_NB_EXCL_14_BIT: int = 1 << 29
_NB_INDEX_MASK: int = (1 << 29) - 1


def encode_nl_exclusions(
    neighbor_idx: Array,
    excl_indices: Array,
    excl_scales_vdw: Array,
    excl_scales_elec: Array,
    n_atoms: int | None = None,
    *,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    tol: float = 1e-4,
) -> Array:
    """Losslessly pre-encode 1-2/1-3 full exclusions and 1-4 scaled exclusions into ``neighbor_idx``.

    Preserves the underlying atom index ``j = raw_j & _NB_INDEX_MASK`` in the
    lower 29 bits while setting ``_NB_EXCL_FULL_BIT`` (bit 30) on full
    exclusions ``(w_vdw == 0, w_elec == 0)`` and ``_NB_EXCL_14_BIT`` (bit 29)
    on standard 1-4 exclusions ``(scale_14_vdw, scale_14_elec)``.

    Optimized (v2.2): Precomputes the 2-bit integer tag directly on the small
    ``(N, M)`` exclusion table and gates the binary search with ``[excl_min, excl_max]``
    bounds, avoiding any ``(N, K)`` float32 scale allocations and supporting both
    ``(N, K)`` and batched ``(B, N, K)`` ``neighbor_idx``.
    """
    nbr_i32 = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    N = int(nbr_i32.shape[-2]) if n_atoms is None else int(n_atoms)
    if N > _NB_INDEX_MASK:
        raise ValueError(
            f"encode_nl_exclusions requires n_atoms <= 2^29 - 1 ({_NB_INDEX_MASK}), got {N}"
        )
    M = int(excl_indices.shape[-1]) if excl_indices.ndim >= 2 else 0
    if M == 0:
        return jnp.where(
            (nbr_i32 >= 0) & ((nbr_i32 & jnp.int32(_NB_INDEX_MASK)) < N),
            nbr_i32 & jnp.int32(_NB_INDEX_MASK),
            nbr_i32,
        )

    ei_i32 = jnp.asarray(excl_indices, dtype=jnp.int32)
    ev = jnp.asarray(excl_scales_vdw)
    ee = jnp.asarray(excl_scales_elec)
    valid_e = (ei_i32 >= 0) & (ei_i32 < N)
    is_full_e = valid_e & (jnp.abs(ev) <= tol) & (jnp.abs(ee) <= tol)
    is_14_e = (
        valid_e
        & (~is_full_e)
        & (jnp.abs(ev - scale_14_vdw) <= tol)
        & (jnp.abs(ee - scale_14_elec) <= tol)
    )
    excl_tag = jnp.where(
        is_full_e,
        jnp.int32(_NB_EXCL_FULL_BIT),
        jnp.where(is_14_e, jnp.int32(_NB_EXCL_14_BIT), jnp.int32(0)),
    )

    int_max = jnp.int32(0x7FFFFFFF)
    excl_sorted = jnp.where(valid_e, ei_i32, int_max)
    excl_min = jnp.min(excl_sorted, axis=-1, keepdims=True)
    excl_max = jnp.max(jnp.where(valid_e, ei_i32, jnp.int32(-1)), axis=-1, keepdims=True)

    j_clean = jnp.where(nbr_i32 >= 0, nbr_i32 & jnp.int32(_NB_INDEX_MASK), nbr_i32)
    valid_j = (j_clean >= 0) & (j_clean < N)
    in_range = valid_j & (j_clean >= excl_min) & (j_clean <= excl_max)

    pos = jnp.zeros_like(j_clean)
    if M > 1:
        step = 1 << ((M - 1).bit_length() - 1)
        while step > 0:
            cand = pos + step
            cand_safe = jnp.minimum(cand, M - 1)
            val = jnp.take_along_axis(excl_sorted, cand_safe, axis=-1)
            pos = jnp.where(in_range & (cand < M) & (val <= j_clean), cand_safe, pos)
            step >>= 1

    matched_idx = jnp.take_along_axis(ei_i32, pos, axis=-1)
    is_match = in_range & (matched_idx == j_clean)
    gathered_tag = jnp.take_along_axis(excl_tag, pos, axis=-1)
    tagged = jnp.where(is_match, j_clean | gathered_tag, j_clean)
    return jnp.where(valid_j, tagged, nbr_i32)


def _expand_bits_10(v: Array) -> Array:
    """Expand a 10-bit integer in [0, 1023] to 30 interleaved bits."""
    v = v.astype(jnp.int32) & jnp.int32(0x3FF)
    v = (v | (v << 16)) & jnp.int32(0x030000FF)
    v = (v | (v << 8)) & jnp.int32(0x0300F00F)
    v = (v | (v << 4)) & jnp.int32(0x030C30C3)
    v = (v | (v << 2)) & jnp.int32(0x09249249)
    return v


def compute_morton_codes_3d(
    positions: Array,
    box: Array | None = None,
    atom_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Compute 30-bit 3D Morton (Z-order) permutation ``(perm, inv_perm)``.

    Padded/ghost atoms (``~atom_mask``) receive key ``0x7FFFFFFF`` so all real
    atoms sort contiguously into ``[0, N_real - 1]`` and ghost atoms remain at
    the tail ``[N_real, N - 1]``.
    """
    pos = jnp.asarray(positions)
    N = pos.shape[0]
    dtype = pos.dtype
    if atom_mask is None:
        mask_b = jnp.ones((N,), dtype=jnp.bool_)
    else:
        mask_b = jnp.asarray(atom_mask, dtype=jnp.bool_)

    if box is not None:
        b_arr = jnp.asarray(box, dtype=dtype)
        if b_arr.ndim == 2 and b_arr.shape == (3, 3):
            L = jnp.diag(b_arr)
        else:
            L = b_arr[:3]
        safe_L = jnp.where(L > 0, L, jnp.ones_like(L))
        pos_wrapped = jnp.where(L > 0, pos - L * jnp.floor(pos / safe_L), pos)
        u = jnp.clip(pos_wrapped / safe_L, 0.0, jnp.asarray(1.0 - 1e-7, dtype=dtype))
    else:
        first_valid = jnp.argmax(mask_b.astype(jnp.int32))
        anchor = pos[first_valid]
        pos_v = jnp.where(mask_b[:, None], pos, anchor[None, :])
        r_min = jnp.min(pos_v, axis=0)
        r_max = jnp.max(pos_v, axis=0)
        span = jnp.maximum(r_max - r_min, jnp.asarray(1e-4, dtype=dtype))
        u = jnp.clip((pos_v - r_min[None, :]) / span[None, :], 0.0, jnp.asarray(1.0 - 1e-7, dtype=dtype))

    q = jnp.clip(jnp.floor(u * jnp.asarray(1024.0, dtype=dtype)).astype(jnp.int32), 0, 1023)
    morton = (_expand_bits_10(q[:, 0]) << 2) | (_expand_bits_10(q[:, 1]) << 1) | _expand_bits_10(q[:, 2])
    morton = jnp.where(mask_b, morton, jnp.int32(0x7FFFFFFF))
    perm = jnp.argsort(morton, stable=True).astype(jnp.int32)
    inv_perm = jnp.argsort(perm).astype(jnp.int32)
    return perm, inv_perm


def permute_neighbor_list_morton(
    neighbor_idx: Array,
    perm: Array,
    inv_perm: Array,
    n_atoms: int | None = None,
) -> Array:
    """Permute ``neighbor_idx`` into Morton order while preserving sentinels and O1.1 tag bits."""
    nbr = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    N = int(nbr.shape[0]) if n_atoms is None else int(n_atoms)
    nbr_rows = nbr[perm]
    tag_bits = jnp.where(nbr_rows >= 0, nbr_rows & (~jnp.int32(_NB_INDEX_MASK)), jnp.int32(0))
    j_raw = jnp.where(nbr_rows >= 0, nbr_rows & jnp.int32(_NB_INDEX_MASK), nbr_rows)
    valid_j = (j_raw >= 0) & (j_raw < N)
    j_mapped = jnp.where(valid_j, inv_perm[jnp.clip(j_raw, 0, max(N - 1, 0))], nbr_rows)
    return jnp.where(valid_j, j_mapped | tag_bits, nbr_rows)


def permute_and_sort_exclusions(
    excl_indices: Array,
    excl_scales_vdw: Array,
    excl_scales_elec: Array,
    perm: Array,
    inv_perm: Array,
    n_atoms: int | None = None,
) -> tuple[Array, Array, Array]:
    """Permute per-atom exclusion tables by ``(perm, inv_perm)`` and re-sort each row ascending."""
    ei = jnp.asarray(excl_indices, dtype=jnp.int32)
    ev = jnp.asarray(excl_scales_vdw)
    ee = jnp.asarray(excl_scales_elec)
    N = int(ei.shape[0]) if n_atoms is None else int(n_atoms)
    if ei.shape[1] == 0:
        return ei[perm], ev[perm], ee[perm]

    ei_p = ei[perm]
    ev_p = ev[perm]
    ee_p = ee[perm]
    valid = (ei_p >= 0) & (ei_p < N)
    int_max = jnp.int32(0x7FFFFFFF)
    ei_mapped = jnp.where(valid, inv_perm[jnp.clip(ei_p, 0, max(N - 1, 0))], int_max)
    order = jnp.argsort(ei_mapped, axis=-1)
    ei_sorted = jnp.take_along_axis(ei_mapped, order, axis=-1)
    ei_final = jnp.where(ei_sorted < int_max, ei_sorted, jnp.int32(-1))
    ev_final = jnp.take_along_axis(ev_p, order, axis=-1)
    ee_final = jnp.take_along_axis(ee_p, order, axis=-1)
    return ei_final, ev_final, ee_final



# =============================================================================
# Pallas v2.2: Cell-List Stencil Pruning + In-Register Exclusion Encoding
# =============================================================================

_NL_NEIGHBOR_DELTAS_3D: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)


def _shift_cell_array_3d(arr: Array, dxyz: tuple[int, int, int]) -> Array:
    """Periodic 3D roll along axes (1, 2, 3) of a ``(B, Nx, Ny, Nz, ...)`` cell buffer."""
    dx, dy, dz = dxyz
    if dx != 0:
        arr = jnp.roll(arr, shift=dx, axis=1)
    if dy != 0:
        arr = jnp.roll(arr, shift=dy, axis=2)
    if dz != 0:
        arr = jnp.roll(arr, shift=dz, axis=3)
    return arr


def _compute_static_grid_dims(box: Array, cell_size: float) -> tuple[int, int, int]:
    """Compute static 3D cell grid dimensions ``(nx, ny, nz)`` on the host."""
    box_np = np.asarray(box, dtype=np.float64)
    if box_np.ndim == 3:
        box_np = box_np[0]
    if box_np.ndim == 2:
        box_np = np.diag(box_np)
    nx = max(1, int(np.floor(box_np[0] / float(cell_size))))
    ny = max(1, int(np.floor(box_np[1] / float(cell_size))))
    nz = max(1, int(np.floor(box_np[2] / float(cell_size))))
    return (nx, ny, nz)


@functools.lru_cache(maxsize=64)
def _build_stencil_cell_lut_np(grid_dims_static: tuple[int, int, int]) -> np.ndarray:
    """Precompute ``(G_cells * 27,)`` int32 neighbor-cell lookup table on the host.

    When any grid dimension is ``< 3``, periodic ``(-1, 0, +1)`` offsets wrap to the
    same cell; duplicate cell IDs within a 27-entry row are replaced with the sentinel
    cell index ``G_cells`` (whose count is ``0``) so no neighbor cell is scanned twice.
    """
    nx, ny, nz = int(grid_dims_static[0]), int(grid_dims_static[1]), int(grid_dims_static[2])
    G_cells = nx * ny * nz
    deltas = [(0, 0, 0)] + list(_NL_NEIGHBOR_DELTAS_3D)
    lut = np.full((G_cells, 27), G_cells, dtype=np.int32)
    for cx in range(nx):
        for cy in range(ny):
            for cz in range(nz):
                c_i = cx * (ny * nz) + cy * nz + cz
                seen: set[int] = set()
                slot = 0
                for dx, dy, dz in deltas:
                    nx_c = (cx + dx) % nx
                    ny_c = (cy + dy) % ny
                    nz_c = (cz + dz) % nz
                    n_cell = nx_c * (ny * nz) + ny_c * nz + nz_c
                    if n_cell not in seen:
                        seen.add(n_cell)
                        lut[c_i, slot] = n_cell
                        slot += 1
    return lut.reshape(-1)


def _build_cell_stencil_buffers_batched(
    position_b: Array,
    box_b: Array,
    cell_size: float,
    cell_capacity: int,
    grid_dims_static: tuple[int, int, int],
) -> tuple[Array, Array, Array, Array, Array, Array, int]:
    """Bin ``(B, N, 3)`` positions into compact ``(B, G_cells, C)`` L2-resident cell buffers."""
    B, N, _ = position_b.shape
    nx, ny, nz = int(grid_dims_static[0]), int(grid_dims_static[1]), int(grid_dims_static[2])
    G_cells = nx * ny * nz
    C = int(cell_capacity)

    grid_dims = jnp.asarray([nx, ny, nz], dtype=jnp.int32)
    box_3 = box_b[:, :3] if box_b.ndim == 2 else jnp.diag(box_b[0])[None, :]
    cell_dims = box_3 / grid_dims.astype(position_b.dtype)[None, :]
    safe_cell_dims = jnp.where(cell_dims > 0, cell_dims, jnp.ones_like(cell_dims))

    pos_wrapped = jnp.mod(position_b, box_3[:, None, :])
    coords = jnp.floor(pos_wrapped / safe_cell_dims[:, None, :]).astype(jnp.int32)
    coords = jnp.clip(coords, jnp.zeros((3,), dtype=jnp.int32), grid_dims - 1)
    cell_id = (
        coords[:, :, 0] * (ny * nz)
        + coords[:, :, 1] * nz
        + coords[:, :, 2]
    ).astype(jnp.int32)

    def _bin_single(c_id: Array, pos_s: Array) -> tuple[Array, Array, Array, Array, Array]:
        if N == 0:
            return (
                jnp.full((G_cells * C,), N, dtype=jnp.int32),
                jnp.zeros((G_cells * C,), dtype=jnp.float32),
                jnp.zeros((G_cells * C,), dtype=jnp.float32),
                jnp.zeros((G_cells * C,), dtype=jnp.float32),
                jnp.zeros((G_cells + 1,), dtype=jnp.int32),
            )
        order = jnp.argsort(c_id).astype(jnp.int32)
        sorted_c = c_id[order]
        idx_in_n = jnp.arange(N, dtype=jnp.int32)
        is_start = jnp.concatenate([jnp.array([True]), sorted_c[1:] != sorted_c[:-1]])
        is_end = jnp.concatenate([sorted_c[:-1] != sorted_c[1:], jnp.array([True])])

        start_dest = jnp.where(is_start, sorted_c, jnp.int32(G_cells))
        cell_start = jnp.zeros((G_cells + 1,), dtype=jnp.int32).at[start_dest].set(idx_in_n)
        group_start = cell_start[sorted_c]
        slot = idx_in_n - group_start

        valid_slot = slot < C
        flat_dest = jnp.where(valid_slot, sorted_c * C + slot, jnp.int32(G_cells * C))

        buf_id = jnp.full((G_cells * C + 1,), N, dtype=jnp.int32).at[flat_dest].set(order)[:-1]
        pos_sorted = pos_s[order]
        buf_x = jnp.zeros((G_cells * C + 1,), dtype=jnp.float32).at[flat_dest].set(pos_sorted[:, 0])[:-1]
        buf_y = jnp.zeros((G_cells * C + 1,), dtype=jnp.float32).at[flat_dest].set(pos_sorted[:, 1])[:-1]
        buf_z = jnp.zeros((G_cells * C + 1,), dtype=jnp.float32).at[flat_dest].set(pos_sorted[:, 2])[:-1]

        end_dest = jnp.where(is_end, sorted_c, jnp.int32(G_cells))
        cell_counts = (
            jnp.zeros((G_cells + 1,), dtype=jnp.int32)
            .at[end_dest]
            .set(jnp.minimum(slot + jnp.int32(1), jnp.int32(C)))
            .at[G_cells]
            .set(jnp.int32(0))
        )
        return buf_id, buf_x, buf_y, buf_z, cell_counts

    cell_atom_b, cell_x_b, cell_y_b, cell_z_b, cell_counts_b = jax.vmap(_bin_single)(
        cell_id, pos_wrapped
    )
    return cell_id, cell_atom_b, cell_x_b, cell_y_b, cell_z_b, cell_counts_b, G_cells


@functools.lru_cache(maxsize=64)
def _make_prune_and_encode_kernel(
    B: int,
    N: int,
    K: int,
    M: int,
    G_cells: int,
    cell_capacity: int,
    B_box: int,
    B_excl: int,
    r_sq_max_f: float,
    encode_exclusions: bool,
):
    """Build Pallas v2.3 kernel: L2-cached cell lookup, dynamic count-bounded tiling, gated exclusion match."""
    BLOCK_C = 64
    C_blocks = (cell_capacity + BLOCK_C - 1) // BLOCK_C
    C_pad = C_blocks * BLOCK_C
    K_blocks = (K + BLOCK_C - 1) // BLOCK_C
    M_BLOCK = 32

    def _kernel(
        px_ref,
        py_ref,
        pz_ref,
        cell_id_ref,
        cell_x_ref,
        cell_y_ref,
        cell_z_ref,
        cell_atom_ref,
        cell_count_ref,
        stencil_lut_ref,
        box_ref,
        excl_min_ref,
        excl_max_ref,
        excl_idx_ref,
        excl_tag_ref,
        nbr_out_ref,
        occ_out_ref,
    ):
        i_atom = pl.program_id(0)
        b_id = pl.program_id(1)
        r_sq_cutoff = jnp.float32(r_sq_max_f)
        n_sentinel = jnp.int32(N)
        lane_c = jnp.arange(BLOCK_C, dtype=jnp.int32)
        lane_m = jnp.arange(M_BLOCK, dtype=jnp.int32)

        base_atom = b_id * N + i_atom
        px_i = _pl_load(px_ref, (base_atom,))
        py_i = _pl_load(py_ref, (base_atom,))
        pz_i = _pl_load(pz_ref, (base_atom,))
        c_i = _pl_load(cell_id_ref, (base_atom,))

        base_box = (b_id * 3) if B_box > 1 else jnp.int32(0)
        Lx = _pl_load(box_ref, (base_box + 0,))
        Ly = _pl_load(box_ref, (base_box + 1,))
        Lz = _pl_load(box_ref, (base_box + 2,))
        inv_Lx = jnp.where(Lx > jnp.float32(0.0), jnp.float32(1.0) / Lx, jnp.float32(0.0))
        inv_Ly = jnp.where(Ly > jnp.float32(0.0), jnp.float32(1.0) / Ly, jnp.float32(0.0))
        inv_Lz = jnp.where(Lz > jnp.float32(0.0), jnp.float32(1.0) / Lz, jnp.float32(0.0))

        if encode_exclusions:
            base_excl_atom = (b_id * N + i_atom) if B_excl > 1 else i_atom
            e_min_i = _pl_load(excl_min_ref, (base_excl_atom,))
            e_max_i = _pl_load(excl_max_ref, (base_excl_atom,))
            base_excl_row = base_excl_atom * M_BLOCK
        else:
            e_min_i = jnp.int32(0x7FFFFFFF)
            e_max_i = jnp.int32(-1)
            base_excl_row = jnp.int32(0)

        base_b_cells = b_id * (G_cells * C_pad)
        base_b_counts = b_id * (G_cells + 1)
        base_c_lut = c_i * 27
        base_nbr_row = base_atom * K
        dummy_nbr_slot = jnp.int32(B * N * K)

        def _stencil_body(s_idx, count_acc):
            n_cell = _pl_load(stencil_lut_ref, (base_c_lut + s_idx,))
            n_cnt = _pl_load(cell_count_ref, (base_b_counts + n_cell,))
            n_tiles = (n_cnt + jnp.int32(BLOCK_C - 1)) // jnp.int32(BLOCK_C)
            base_cell_offs = base_b_cells + n_cell * C_pad

            def _tile_body(t_idx, c_acc):
                offs = base_cell_offs + t_idx * BLOCK_C + lane_c
                j_idx = _pl_load(cell_atom_ref, (offs,))
                jx = _pl_load(cell_x_ref, (offs,))
                jy = _pl_load(cell_y_ref, (offs,))
                jz = _pl_load(cell_z_ref, (offs,))

                dx = px_i - jx
                dy = py_i - jy
                dz = pz_i - jz
                dx = dx - Lx * jnp.floor(dx * inv_Lx + jnp.float32(0.5))
                dy = dy - Ly * jnp.floor(dy * inv_Ly + jnp.float32(0.5))
                dz = dz - Lz * jnp.floor(dz * inv_Lz + jnp.float32(0.5))
                r_sq = dx * dx + dy * dy + dz * dz

                valid = (j_idx < n_sentinel) & (j_idx != i_atom) & (r_sq < r_sq_cutoff)
                valid_i32 = valid.astype(jnp.int32)
                tile_hits = jnp.sum(valid_i32, dtype=jnp.int32)

                def _write_hits(c_in):
                    prefix = jnp.cumsum(valid_i32, axis=0, dtype=jnp.int32) - jnp.int32(1)
                    dest_slot = c_in + prefix
                    can_write = valid & (dest_slot < jnp.int32(K))

                    if encode_exclusions:
                        in_excl_range = can_write & (j_idx >= e_min_i) & (j_idx <= e_max_i)
                        has_any_excl = jnp.max(in_excl_range.astype(jnp.int32)) > jnp.int32(0)

                        def _apply_excl(j_in):
                            excl_row_idx = _pl_load(excl_idx_ref, (base_excl_row + lane_m,))
                            excl_row_tag = _pl_load(excl_tag_ref, (base_excl_row + lane_m,))
                            match_mat = in_excl_range[:, None] & (j_in[:, None] == excl_row_idx[None, :])
                            tag_bits = jnp.max(
                                jnp.where(match_mat, excl_row_tag[None, :], jnp.int32(0)),
                                axis=1,
                            )
                            return j_in | tag_bits

                        j_tagged = jax.lax.cond(has_any_excl, _apply_excl, lambda j_in: j_in, j_idx)
                    else:
                        j_tagged = j_idx

                    safe_dest = jnp.where(can_write, base_nbr_row + dest_slot, dummy_nbr_slot)
                    _pl_store(nbr_out_ref, (safe_dest,), j_tagged, mask=can_write)
                    return c_in + tile_hits

                return jax.lax.cond(tile_hits > jnp.int32(0), _write_hits, lambda c_in: c_in, c_acc)

            return jax.lax.fori_loop(jnp.int32(0), n_tiles, _tile_body, count_acc)

        total_hits = jax.lax.fori_loop(jnp.int32(0), jnp.int32(27), _stencil_body, jnp.int32(0))

        start_pad_tile = jnp.minimum(total_hits, jnp.int32(K)) // jnp.int32(BLOCK_C)
        sentinel_vec = jnp.full((BLOCK_C,), n_sentinel, dtype=jnp.int32)

        def _pad_body(pad_t, _):
            slot = pad_t * BLOCK_C + lane_c
            should_pad = (slot >= total_hits) & (slot < jnp.int32(K))
            safe_dest = jnp.where(should_pad, base_nbr_row + slot, dummy_nbr_slot)
            _pl_store(nbr_out_ref, (safe_dest,), sentinel_vec, mask=should_pad)
            return None

        jax.lax.fori_loop(start_pad_tile, jnp.int32(K_blocks), _pad_body, None)
        _pl_store(occ_out_ref, (base_atom,), total_hits)

    return _kernel


def _pallas_prune_and_encode_cell_list_batch(
    axis_size: int,
    in_batched: tuple[bool, ...],
    position: Array,
    box: Array,
    excl_indices: Array,
    excl_scales_vdw: Array,
    excl_scales_elec: Array,
    cell_size: float,
    cell_capacity: int,
    max_occupancy: int,
    r_cutoff: float,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    encode_exclusions: bool = True,
    interpret: bool = False,
    grid_dims: tuple[int, int, int] | None = None,
) -> tuple[Array, Array]:
    """Batched 2D-grid ``(N, B)`` Pallas cell-list pruning + in-register exclusion encoding."""
    B = int(axis_size)
    pos_b = position if in_batched[0] else jnp.broadcast_to(position[None, ...], (B,) + position.shape)
    bx_b = box if in_batched[1] else box[None, ...]
    ei_b = excl_indices if in_batched[2] else excl_indices[None, ...]
    ev_b = excl_scales_vdw if in_batched[3] else excl_scales_vdw[None, ...]
    ee_b = excl_scales_elec if in_batched[4] else excl_scales_elec[None, ...]

    N = int(pos_b.shape[1])
    K = int(max_occupancy)
    C = int(cell_capacity)
    BLOCK_C = 64
    C_blocks = (C + BLOCK_C - 1) // BLOCK_C
    C_pad = C_blocks * BLOCK_C
    M_BLOCK = 32

    pos_f32 = pos_b.astype(jnp.float32)
    bx_f32 = bx_b.astype(jnp.float32)
    if bx_f32.ndim == 3:
        bx_3 = jnp.stack([bx_f32[:, 0, 0], bx_f32[:, 1, 1], bx_f32[:, 2, 2]], axis=-1)
    else:
        bx_3 = bx_f32[:, :3]

    g_dims = grid_dims if grid_dims is not None else _compute_static_grid_dims(bx_3, float(cell_size))
    (
        cell_id_b,
        cell_atom_b,
        cell_x_b,
        cell_y_b,
        cell_z_b,
        cell_counts_b,
        G_cells,
    ) = _build_cell_stencil_buffers_batched(pos_f32, bx_3, float(cell_size), C_pad, g_dims)

    stencil_lut_flat = jnp.asarray(_build_stencil_cell_lut_np(g_dims), dtype=jnp.int32)
    cell_x_flat = cell_x_b.reshape(B * G_cells * C_pad)
    cell_y_flat = cell_y_b.reshape(B * G_cells * C_pad)
    cell_z_flat = cell_z_b.reshape(B * G_cells * C_pad)
    cell_atom_flat = cell_atom_b.reshape(B * G_cells * C_pad)
    cell_count_flat = cell_counts_b.reshape(B * (G_cells + 1))

    pos_w_f32 = jnp.mod(pos_f32, bx_3[:, None, :])
    px_flat = pos_w_f32[:, :, 0].reshape(B * N)
    py_flat = pos_w_f32[:, :, 1].reshape(B * N)
    pz_flat = pos_w_f32[:, :, 2].reshape(B * N)
    cell_id_flat = cell_id_b.reshape(B * N)
    box_flat = bx_3.reshape(bx_3.shape[0] * 3)

    B_excl = ei_b.shape[0]
    M_raw = int(ei_b.shape[2]) if ei_b.ndim == 3 else 0
    if encode_exclusions and M_raw > 0:
        ei_clip = ei_b[:, :, :M_BLOCK].astype(jnp.int32)
        ev_clip = ev_b[:, :, :M_BLOCK].astype(jnp.float32)
        ee_clip = ee_b[:, :, :M_BLOCK].astype(jnp.float32)
        pad_m = max(0, M_BLOCK - ei_clip.shape[2])
        if pad_m > 0:
            ei_clip = jnp.pad(ei_clip, ((0, 0), (0, 0), (0, pad_m)), constant_values=-1)
            ev_clip = jnp.pad(ev_clip, ((0, 0), (0, 0), (0, pad_m)), constant_values=1.0)
            ee_clip = jnp.pad(ee_clip, ((0, 0), (0, 0), (0, pad_m)), constant_values=1.0)
        valid_e = (ei_clip >= 0) & (ei_clip < N)
        tol = 1e-4
        is_full = valid_e & (jnp.abs(ev_clip) <= tol) & (jnp.abs(ee_clip) <= tol)
        is_14 = (
            valid_e
            & (~is_full)
            & (jnp.abs(ev_clip - float(scale_14_vdw)) <= tol)
            & (jnp.abs(ee_clip - float(scale_14_elec)) <= tol)
        )
        excl_tag = jnp.where(
            is_full,
            jnp.int32(_NB_EXCL_FULL_BIT),
            jnp.where(is_14, jnp.int32(_NB_EXCL_14_BIT), jnp.int32(0)),
        )
        int_max = jnp.int32(0x7FFFFFFF)
        excl_min = jnp.min(jnp.where(valid_e, ei_clip, int_max), axis=-1)
        excl_max = jnp.max(jnp.where(valid_e, ei_clip, jnp.int32(-1)), axis=-1)
        excl_min_flat = excl_min.reshape(B_excl * N)
        excl_max_flat = excl_max.reshape(B_excl * N)
        excl_idx_flat = ei_clip.reshape(B_excl * N * M_BLOCK)
        excl_tag_flat = excl_tag.reshape(B_excl * N * M_BLOCK)
        do_encode = True
    else:
        excl_min_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_max_flat = jnp.zeros((1,), dtype=jnp.int32)
        excl_idx_flat = jnp.zeros((M_BLOCK,), dtype=jnp.int32)
        excl_tag_flat = jnp.zeros((M_BLOCK,), dtype=jnp.int32)
        do_encode = False

    r_sq_max_f = float(r_cutoff) * float(r_cutoff)
    kernel_fn = _make_prune_and_encode_kernel(
        B,
        N,
        K,
        M_BLOCK,
        G_cells,
        C_pad,
        int(bx_3.shape[0]),
        int(B_excl),
        r_sq_max_f,
        do_encode,
    )
    out_shapes = (
        jax.ShapeDtypeStruct((B * N * K + 1,), jnp.int32),
        jax.ShapeDtypeStruct((B * N,), jnp.int32),
    )
    nbr_flat, occ_flat = pl.pallas_call(
        kernel_fn,
        out_shape=out_shapes,
        grid=(N, B),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_prune_and_encode_cell_list",
    )(
        px_flat,
        py_flat,
        pz_flat,
        cell_id_flat,
        cell_x_flat,
        cell_y_flat,
        cell_z_flat,
        cell_atom_flat,
        cell_count_flat,
        stencil_lut_flat,
        box_flat,
        excl_min_flat,
        excl_max_flat,
        excl_idx_flat,
        excl_tag_flat,
    )
    occ_b = jnp.max(occ_flat.reshape((B, N)), axis=-1)
    return nbr_flat[:-1].reshape((B, N, K)), occ_b


def pallas_prune_and_encode_cell_list(
    position: Array,
    box: Array,
    excl_indices: Array,
    excl_scales_vdw: Array,
    excl_scales_elec: Array,
    *,
    cell_size: float,
    cell_capacity: int,
    max_occupancy: int,
    r_cutoff: float,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    encode_exclusions: bool = True,
    interpret: bool = False,
    grid_dims: tuple[int, int, int] | None = None,
) -> tuple[Array, Array]:
    """Prune 3D cell-list stencil candidates and pre-encode exclusions in one Pallas kernel."""
    cs = float(cell_size)
    cc = int(cell_capacity)
    mo = int(max_occupancy)
    rc = float(r_cutoff)
    s14_v = float(scale_14_vdw)
    s14_e = float(scale_14_elec)
    enc = bool(encode_exclusions)
    interp = bool(interpret)
    g_dims = (
        grid_dims
        if grid_dims is not None
        else (None if isinstance(box, jax.core.Tracer) else _compute_static_grid_dims(box, cs))
    )

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, bx, ei, ev, ee):
        nbr_b, occ_b = _pallas_prune_and_encode_cell_list_batch(
            1,
            (False, False, False, False, False),
            pos,
            bx,
            ei,
            ev,
            ee,
            cs,
            cc,
            mo,
            rc,
            s14_v,
            s14_e,
            enc,
            interp,
            g_dims,
        )
        return nbr_b[0], occ_b[0]

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, bx, ei, ev, ee):
        nbr_b, occ_b = _pallas_prune_and_encode_cell_list_batch(
            int(axis_size),
            tuple(bool(b) for b in in_batched),
            pos,
            bx,
            ei,
            ev,
            ee,
            cs,
            cc,
            mo,
            rc,
            s14_v,
            s14_e,
            enc,
            interp,
            g_dims,
        )
        return (nbr_b, occ_b), (True, True)

    return _vmappable(position, box, excl_indices, excl_scales_vdw, excl_scales_elec)
