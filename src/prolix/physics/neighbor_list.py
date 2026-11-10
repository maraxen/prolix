"""Neighbor list management and sparse exclusion handling."""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax_md import dataclasses as jax_md_dataclasses
from jax_md import partition, space, util

from prolix.compat import system_params_to_protein
from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.pbc import wrap_positions
from prolix.utils import topology

if TYPE_CHECKING:
  from collections.abc import Callable

  from proxide.core.containers import Protein

  from prolix.typing import SystemParams

Array = util.Array

if not hasattr(partition.NeighborList, "max_k_tiles"):
  @jax_md_dataclasses.dataclass
  class _ProlixNeighborList(partition.NeighborList):
    max_k_tiles: Array | None = None

  _ProlixNeighborList.__name__ = "NeighborList"
  _ProlixNeighborList.__qualname__ = "NeighborList"
  partition.NeighborList = _ProlixNeighborList


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
  atom_mask: Array | None = None,
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
  precomp_excl_static = (
    precompute_static_nl_exclusion_tables(
      excl_indices,
      excl_scales_vdw,
      excl_scales_elec,
      int(excl_indices.shape[-2]),
      scale_14_vdw=float(scale_14_vdw),
      scale_14_elec=float(scale_14_elec),
    )
    if has_excl
    else None
  )

  def _wrap(position: Array, dyn_box: Array | None = None) -> Array:
    b = jnp.asarray(dyn_box) if dyn_box is not None else box
    if b.ndim == 2:
      return wrap_positions(position, b)
    return jnp.mod(position, b)

  def _encode_if_needed(idx: Array, n_atoms: int) -> Array:
    if not has_excl and atom_mask is None:
      return idx
    ei = excl_indices if has_excl else jnp.zeros((n_atoms, 0), dtype=jnp.int32)
    ev = excl_scales_vdw if has_excl else jnp.zeros((n_atoms, 0), dtype=jnp.float32)
    ee = excl_scales_elec if has_excl else jnp.zeros((n_atoms, 0), dtype=jnp.float32)
    return encode_nl_exclusions(
      idx,
      ei,
      ev,
      ee,
      n_atoms=n_atoms,
      scale_14_vdw=scale_14_vdw,
      scale_14_elec=scale_14_elec,
      atom_mask=atom_mask,
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
      idx_new, occ_new, mkt_new = pallas_prune_and_encode_cell_list(
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
        atom_mask=atom_mask,
        precomputed_excl=precomp_excl_static,
        return_max_k_tiles=True,
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
        max_k_tiles=mkt_new,
        update_fn=update,
      )
    # Fallback to jax_md update; force rebuild by shifting reference_position
    b_diag = jnp.diag(b_eff) if b_eff.ndim == 2 else b_eff
    forced_ref = pos_w + jnp.asarray(0.5, dtype=pos_w.dtype) * b_diag.astype(pos_w.dtype)
    nbr_forced = dataclasses.replace(neighbors, reference_position=forced_ref, max_k_tiles=None)
    nbr = fns.update(pos_w, nbr_forced, **kwargs)
    n_at = int(pos_w.shape[-2])
    idx_enc = _encode_if_needed(nbr.idx, n_at)
    mkt_enc = _max_k_tiles_from_idx(idx_enc, n_at)
    return dataclasses.replace(nbr, idx=idx_enc, max_k_tiles=mkt_enc, update_fn=update)

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
      n_at_u = int(pos_w.shape[-2])
      mkt_in = (
        neighbors.max_k_tiles
        if getattr(neighbors, "max_k_tiles", None) is not None
        else _max_k_tiles_from_idx(neighbors.idx, n_at_u)
      )

      @jax.custom_batching.custom_vmap
      def _cond_rebuild(
        pos_in: Array,
        ref_in: Array,
        idx_in: Array,
        err_code_in: Array,
        box_in: Array,
        mkt_in_arg: Array,
      ):
        dr = jax.vmap(displacement_fn)(pos_in, ref_in)
        dr_sq = jnp.sum(dr * dr, axis=-1)
        if atom_mask is not None:
          dr_sq = jnp.where(jnp.asarray(atom_mask, dtype=jnp.bool_), dr_sq, 0.0)
        should_upd = jnp.any(dr_sq > jnp.asarray(half_skin_sq, dtype=pos_in.dtype))

        def _do_one(operands):
          p_o, _r_o, _i_o, e_o, bx_o, _m_o = operands
          nbr_tmp = dataclasses.replace(
            neighbors,
            idx=_i_o,
            reference_position=_r_o,
            error=partition.PartitionError(e_o),
            max_k_tiles=None,
          )
          if can_pallas:
            N_atoms = int(p_o.shape[0])
            ei = excl_indices if has_excl else jnp.full((N_atoms, 1), -1, dtype=jnp.int32)
            ev = excl_scales_vdw if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            ee = excl_scales_elec if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            idx_new, occ_new, mkt_new = pallas_prune_and_encode_cell_list(
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
              atom_mask=atom_mask,
              precomputed_excl=precomp_excl_static,
              return_max_k_tiles=True,
            )
            err_new = nbr_tmp.error.update(
              partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW,
              occ_new > max_occ_i,
            )
            return idx_new, p_o.astype(_r_o.dtype), err_new.code, mkt_new
          b_diag = jnp.diag(bx_o) if bx_o.ndim == 2 else bx_o
          forced_ref = p_o + jnp.asarray(0.5, dtype=p_o.dtype) * b_diag.astype(p_o.dtype)
          nbr_forced = dataclasses.replace(nbr_tmp, reference_position=forced_ref, max_k_tiles=None)
          nbr_out = fns.update(p_o, nbr_forced, **kwargs)
          idx_enc = _encode_if_needed(nbr_out.idx, int(p_o.shape[0]))
          mkt_enc = _max_k_tiles_from_idx(idx_enc, int(p_o.shape[0]))
          return idx_enc, nbr_out.reference_position, nbr_out.error.code, mkt_enc

        def _keep_one(operands):
          _p_o, r_o, i_o, e_o, _bx_o, m_o = operands
          return i_o, r_o, e_o, m_o

        return jax.lax.cond(
          should_upd,
          _do_one,
          _keep_one,
          (pos_in, ref_in, idx_in, err_code_in, box_in, mkt_in_arg),
        )

      @_cond_rebuild.def_vmap
      def _cond_rebuild_vmap(
        axis_size, in_batched, pos_in, ref_in, idx_in, err_code_in, box_in, mkt_in_arg
      ):
        B = int(axis_size)
        pos_b = pos_in if in_batched[0] else jnp.broadcast_to(pos_in[None, ...], (B,) + pos_in.shape)
        ref_b = ref_in if in_batched[1] else jnp.broadcast_to(ref_in[None, ...], (B,) + ref_in.shape)
        idx_b = idx_in if in_batched[2] else jnp.broadcast_to(idx_in[None, ...], (B,) + idx_in.shape)
        err_b = err_code_in if in_batched[3] else jnp.broadcast_to(err_code_in[None, ...], (B,) + err_code_in.shape)
        bx_b = box_in if in_batched[4] else jnp.broadcast_to(box_in[None, ...], (B,) + box_in.shape)
        mkt_b = mkt_in_arg if in_batched[5] else jnp.broadcast_to(mkt_in_arg[None, ...], (B,) + mkt_in_arg.shape)

        dr_b = jax.vmap(jax.vmap(displacement_fn))(pos_b, ref_b)
        dr_sq_b = jnp.sum(dr_b * dr_b, axis=-1)
        if atom_mask is not None:
          dr_sq_b = jnp.where(jnp.asarray(atom_mask, dtype=jnp.bool_), dr_sq_b, 0.0)
        su_b = jnp.any(dr_sq_b > jnp.asarray(half_skin_sq, dtype=pos_b.dtype), axis=-1)
        any_upd = jnp.any(su_b)

        def _do_batch(operands):
          p_b, _r_b, _i_b, e_b, b_b, su_op, m_b = operands
          if can_pallas:
            N_atoms = int(p_b.shape[1])
            ei = excl_indices if has_excl else jnp.full((N_atoms, 1), -1, dtype=jnp.int32)
            ev = excl_scales_vdw if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            ee = excl_scales_elec if has_excl else jnp.ones((N_atoms, 1), dtype=jnp.float32)
            idx_new_b, occ_new_b, mkt_new_b = _pallas_prune_and_encode_cell_list_batch(
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
              atom_mask=atom_mask,
              mask_batched=(atom_mask is not None and atom_mask.ndim == 2),
              rebuild_mask=su_op,
              old_idx=_i_b,
              precomputed_excl=precomp_excl_static,
              return_max_k_tiles=True,
              old_max_k_tiles=m_b,
            )
            overflow_bit = jnp.uint8(partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW)
            err_new_b = jnp.where(su_op & (occ_new_b > max_occ_i), e_b | overflow_bit, e_b)
            ref_new_b = jnp.where(su_op[:, None, None], p_b.astype(_r_b.dtype), _r_b)
            return idx_new_b, ref_new_b, err_new_b, mkt_new_b

          def _single_fallback(p_1, r_1, i_1, e_1, bx_1, su_1, m_1):
            def _do_1(_):
              b_diag = jnp.diag(bx_1) if bx_1.ndim == 2 else bx_1
              forced_ref = p_1 + jnp.asarray(0.5, dtype=p_1.dtype) * b_diag.astype(p_1.dtype)
              nbr_tmp = dataclasses.replace(
                neighbors,
                idx=i_1,
                reference_position=forced_ref,
                error=partition.PartitionError(e_1),
                max_k_tiles=None,
              )
              nbr_out = fns.update(p_1, nbr_tmp, **kwargs)
              idx_enc = _encode_if_needed(nbr_out.idx, int(p_1.shape[0]))
              mkt_enc = _max_k_tiles_from_idx(idx_enc, int(p_1.shape[0]))
              return idx_enc, nbr_out.reference_position, nbr_out.error.code, mkt_enc

            return jax.lax.cond(
              su_1,
              _do_1,
              lambda _: (i_1, r_1, e_1, m_1),
              None,
            )

          return jax.vmap(_single_fallback)(p_b, _r_b, _i_b, e_b, b_b, su_op, m_b)

        def _keep_batch(operands):
          _p_b, r_b, i_b, e_b, _b_b, _su_op, m_b = operands
          return i_b, r_b, e_b, m_b

        out_idx_b, out_ref_b, out_err_b, out_mkt_b = jax.lax.cond(
          any_upd,
          _do_batch,
          _keep_batch,
          (pos_b, ref_b, idx_b, err_b, bx_b, su_b, mkt_b),
        )
        return (out_idx_b, out_ref_b, out_err_b, out_mkt_b), (True, True, True, True)

      idx_out, ref_out, err_code_out, mkt_out = _cond_rebuild(
        pos_w,
        neighbors.reference_position,
        neighbors.idx,
        neighbors.error.code,
        b_eff,
        mkt_in,
      )
      return dataclasses.replace(
        neighbors,
        idx=idx_out,
        reference_position=ref_out,
        error=partition.PartitionError(err_code_out),
        max_k_tiles=mkt_out,
        update_fn=update,
      )

    nbr_plain = dataclasses.replace(neighbors, max_k_tiles=None)
    nbr = fns.update(pos_w, nbr_plain, **kwargs)
    mkt_out = _max_k_tiles_from_idx(nbr.idx, int(pos_w.shape[-2]))
    return dataclasses.replace(nbr, max_k_tiles=mkt_out, update_fn=update)

  update._pre_encoded_exclusions = has_excl  # type: ignore[attr-defined]
  update._ghosts_filtered = atom_mask is not None  # type: ignore[attr-defined]
  update._rebuild_unconditional = _rebuild_unconditional  # type: ignore[attr-defined]

  def allocate(position: Array, **kwargs) -> partition.NeighborList:
    b = kwargs.get("box", None)
    b_eff = jnp.asarray(b) if b is not None else box
    pos_w = _wrap(position, b)
    nbr = fns.allocate(pos_w, **kwargs)
    can_pallas_alloc = (
      not disable_cell_list
      and b_eff.ndim == 1
      and getattr(nbr, "cell_list_fn", None) is not None
      and getattr(nbr, "cell_list_capacity", None) is not None
      and _should_use_pallas_nl_default(use_pallas_nl)
    )
    if can_pallas_alloc:
      return _rebuild_unconditional(pos_w, nbr, **kwargs)
    n_at_a = int(pos_w.shape[-2])
    idx_enc = _encode_if_needed(nbr.idx, n_at_a)
    mkt_enc = _max_k_tiles_from_idx(idx_enc, n_at_a)
    return dataclasses.replace(nbr, idx=idx_enc, max_k_tiles=mkt_enc, update_fn=update)

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
  jnp.full((N, max_exclusions), -1, dtype=jnp.int32)
  jnp.ones((N, max_exclusions), dtype=jnp.float32)
  jnp.ones((N, max_exclusions), dtype=jnp.float32)

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
    atom_mask: Array | None = None,
) -> Array:
    """Losslessly pre-encode 1-2/1-3 full exclusions and 1-4 scaled exclusions into ``neighbor_idx``.

    Preserves the underlying atom index ``j = raw_j & _NB_INDEX_MASK`` in the
    lower 29 bits while setting ``_NB_EXCL_FULL_BIT`` (bit 30) on full
    exclusions ``(w_vdw == 0, w_elec == 0)`` and ``_NB_EXCL_14_BIT`` (bit 29)
    on standard 1-4 exclusions ``(scale_14_vdw, scale_14_elec)``.

    Optimized (v2.2 / v2.4): Precomputes the 2-bit integer tag directly on the small
    ``(N, M)`` exclusion table and gates the binary search with ``[excl_min, excl_max]``
    bounds, avoiding any ``(N, K)`` float32 scale allocations and supporting both
    ``(N, K)`` and batched ``(B, N, K)`` ``neighbor_idx``. When ``atom_mask`` is
    provided, ghost padding atoms (``~atom_mask``) are replaced with sentinel ``N``
    and valid real neighbors are compacted to the front of each row so trailing
    32-neighbor tiles in ``pallas_nonbonded_nl`` exit in O(1).
    """
    nbr_i32 = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    N = int(nbr_i32.shape[-2]) if n_atoms is None else int(n_atoms)
    if N > _NB_INDEX_MASK:
        raise ValueError(
            f"encode_nl_exclusions requires n_atoms <= 2^29 - 1 ({_NB_INDEX_MASK}), got {N}"
        )
    M = int(excl_indices.shape[-1]) if excl_indices.ndim >= 2 else 0
    if M == 0:
        out = jnp.where(
            (nbr_i32 >= 0) & ((nbr_i32 & jnp.int32(_NB_INDEX_MASK)) < N),
            nbr_i32 & jnp.int32(_NB_INDEX_MASK),
            nbr_i32,
        )
    else:
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
        out = jnp.where(valid_j, tagged, nbr_i32)

    if N > 0:
        j_idx = jnp.where(out >= 0, out & jnp.int32(_NB_INDEX_MASK), out)
        valid_entry = (j_idx >= 0) & (j_idx < N)
        if atom_mask is not None:
            mask_b = jnp.asarray(atom_mask, dtype=jnp.bool_)
            safe_j = jnp.clip(j_idx, 0, max(N - 1, 0))
            if mask_b.ndim == 1:
                real_j = mask_b[safe_j]
                real_i = mask_b[:, None] if out.ndim == 2 else mask_b[None, :, None]
            else:
                real_j = jnp.take_along_axis(mask_b, safe_j, axis=-1)
                real_i = mask_b[:, :, None]
            keep = valid_entry & real_i & real_j
        else:
            keep = valid_entry
        out = jnp.where(keep, out, jnp.int32(N))
        perm = jnp.argsort((out == jnp.int32(N)).astype(jnp.int32), axis=-1)
        out = jnp.take_along_axis(out, perm, axis=-1)
    return out


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
        if (
            box_np.shape == (3, 3)
            and box_np[0, 1] == 0.0
            and box_np[0, 2] == 0.0
            and box_np[1, 2] == 0.0
        ):
            box_np = np.diag(box_np)
        else:
            box_np = box_np[0]
    nx = max(1, int(np.floor(box_np[0] / float(cell_size))))
    ny = max(1, int(np.floor(box_np[1] / float(cell_size))))
    nz = max(1, int(np.floor(box_np[2] / float(cell_size))))
    return (nx, ny, nz)


@functools.lru_cache(maxsize=64)
def _build_stencil_cell_lut_np(grid_dims_static: tuple[int, int, int]) -> np.ndarray:
    """Precompute ``((G_cells + 1) * 27,)`` int32 neighbor-cell lookup table on the host.

    When any grid dimension is ``< 3``, periodic ``(-1, 0, +1)`` offsets wrap to the
    same cell; duplicate cell IDs within a 27-entry row are replaced with the sentinel
    cell index ``G_cells`` (whose count is ``0``) so no neighbor cell is scanned twice.
    Row ``G_cells`` is also populated with ``G_cells`` so ghost atoms (``c_i == G_cells``)
    always have in-bounds LUT offsets.
    """
    nx, ny, nz = int(grid_dims_static[0]), int(grid_dims_static[1]), int(grid_dims_static[2])
    G_cells = nx * ny * nz
    deltas = [(0, 0, 0)] + list(_NL_NEIGHBOR_DELTAS_3D)
    lut = np.full((G_cells + 1, 27), G_cells, dtype=np.int32)
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


@functools.lru_cache(maxsize=64)
def _make_pallas_bin_cells_kernel(B: int, N: int, G_cells: int, C: int, BLOCK_BIN: int = 128):
    """MK-52: 2D-grid ``(B, cdiv(N, 128))`` Pallas atomic 3D cell-list binning kernel.

    Bins atoms in O(N) via ``plt.atomic_add`` on ``cell_counts`` and exits in O(1)
    for replicas where ``rebuild_mask[b] == False``.
    """

    def _bin_kernel(
        cell_id_ref,
        px_ref,
        py_ref,
        pz_ref,
        reb_ref,
        id_init_ref,
        x_init_ref,
        y_init_ref,
        z_init_ref,
        counts_init_ref,
        id_out_ref,
        x_out_ref,
        y_out_ref,
        z_out_ref,
        counts_out_ref,
    ):
        del id_init_ref, x_init_ref, y_init_ref, z_init_ref, counts_init_ref
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        reb_val = _pl_load(reb_ref, (b_id,))

        def _do_bin(_):
            offs = pid * BLOCK_BIN + jnp.arange(BLOCK_BIN, dtype=jnp.int32)
            in_bounds = offs < jnp.int32(N)
            base_atom = b_id * N + offs
            c_i = _pl_load(cell_id_ref, (base_atom,), mask=in_bounds, other=jnp.int32(G_cells))
            valid_cell = in_bounds & (c_i >= jnp.int32(0)) & (c_i < jnp.int32(G_cells))
            has_any = jnp.max(valid_cell.astype(jnp.int32)) > jnp.int32(0)

            def _scatter_active(_):
                zero_f = jnp.float32(0.0)
                px_i = _pl_load(px_ref, (base_atom,), mask=valid_cell, other=zero_f)
                py_i = _pl_load(py_ref, (base_atom,), mask=valid_cell, other=zero_f)
                pz_i = _pl_load(pz_ref, (base_atom,), mask=valid_cell, other=zero_f)
                safe_c = jnp.where(valid_cell, c_i, jnp.int32(G_cells))
                count_idx = b_id * (G_cells + 1) + safe_c
                ones = jnp.ones((BLOCK_BIN,), dtype=jnp.int32)
                slot = plt.atomic_add(counts_out_ref, (count_idx,), ones, mask=valid_cell)
                can_store = valid_cell & (slot >= jnp.int32(0)) & (slot < jnp.int32(C))
                dummy_dest = jnp.int32(B * G_cells * C)
                dest = jnp.where(
                    can_store,
                    b_id * (G_cells * C) + safe_c * C + slot,
                    dummy_dest,
                )
                _pl_store(id_out_ref, (dest,), offs, mask=can_store)
                _pl_store(x_out_ref, (dest,), px_i, mask=can_store)
                _pl_store(y_out_ref, (dest,), py_i, mask=can_store)
                _pl_store(z_out_ref, (dest,), pz_i, mask=can_store)

            jax.lax.cond(has_any, _scatter_active, lambda _: None, None)

        jax.lax.cond(reb_val > jnp.int32(0), _do_bin, lambda _: None, None)

    return _bin_kernel


def _resolve_nl_active_atoms(
    n_active_atoms: int | None, n_total: int, mask: Array | None = None
) -> int:
    if n_active_atoms is not None and int(n_active_atoms) > 0:
        return min(max(1, int(n_active_atoms)), max(1, int(n_total)))
    env_val = os.environ.get("PROLIX_ACTIVE_ATOMS", "").strip()
    if env_val:
        try:
            v = int(env_val)
            if v > 0:
                return min(v, max(1, int(n_total)))
        except ValueError:
            pass
    if mask is not None and not isinstance(mask, jax.core.Tracer):
        try:
            import numpy as _np

            m_np = _np.asarray(mask, dtype=bool)
            if m_np.ndim >= 2:
                m_np = _np.any(m_np, axis=0)
            nz = _np.flatnonzero(m_np)
            if nz.size > 0:
                return min(max(1, int(nz[-1]) + 1), max(1, int(n_total)))
        except Exception:
            pass
    return int(n_total)


def _build_cell_stencil_buffers_batched(
    position_b: Array,
    box_b: Array,
    cell_size: float,
    cell_capacity: int,
    grid_dims_static: tuple[int, int, int],
    atom_mask_b: Array | None = None,
    rebuild_mask_b: Array | None = None,
    interpret: bool = True,
    n_active_atoms: int | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, int]:
    """Bin ``(B, N, 3)`` positions into compact ``(B, G_cells, C)`` L2-resident cell buffers.

    When ``atom_mask_b`` is provided, ghost padding atoms (``~atom_mask_b``) are assigned
    sentinel cell ID ``G_cells`` and excluded from all 3D cell buffers.
    When ``rebuild_mask_b`` is provided (MK-47 / MK-52), replicas with ``~rebuild_mask_b[b]`` also
    receive sentinel cell ID ``G_cells`` for all atoms so their CTAs exit in O(1).
    """
    B, N, _ = position_b.shape
    N_act = _resolve_nl_active_atoms(n_active_atoms, N, atom_mask_b)
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
    if atom_mask_b is not None:
        mask_bool = jnp.asarray(atom_mask_b, dtype=jnp.bool_)
        if mask_bool.ndim == 1:
            mask_bool = mask_bool[None, :]
        cell_id = jnp.where(mask_bool, cell_id, jnp.int32(G_cells))
    if rebuild_mask_b is not None:
        reb_bool = jnp.asarray(rebuild_mask_b, dtype=jnp.bool_).reshape(B, 1)
        cell_id = jnp.where(reb_bool, cell_id, jnp.int32(G_cells))
        reb_i32 = reb_bool[:, 0].astype(jnp.int32)
    else:
        reb_i32 = jnp.ones((B,), dtype=jnp.int32)

    if not interpret and N > 0 and not in_export_context():
        BLOCK_BIN = 128
        n_bin_blocks = max(1, (N_act + BLOCK_BIN - 1) // BLOCK_BIN)
        bin_kernel = _make_pallas_bin_cells_kernel(B, N, G_cells, C, BLOCK_BIN)
        total_slots = B * G_cells * C + 1
        id_init = jnp.full((total_slots,), N, dtype=jnp.int32)
        x_init = jnp.zeros((total_slots,), dtype=jnp.float32)
        y_init = jnp.zeros((total_slots,), dtype=jnp.float32)
        z_init = jnp.zeros((total_slots,), dtype=jnp.float32)
        counts_init = jnp.zeros((B * (G_cells + 1),), dtype=jnp.int32)
        any_spec = pl.BlockSpec(memory_space=pl.ANY)
        out_shapes = (
            jax.ShapeDtypeStruct((total_slots,), jnp.int32),
            jax.ShapeDtypeStruct((total_slots,), jnp.float32),
            jax.ShapeDtypeStruct((total_slots,), jnp.float32),
            jax.ShapeDtypeStruct((total_slots,), jnp.float32),
            jax.ShapeDtypeStruct((B * (G_cells + 1),), jnp.int32),
        )
        id_flat, x_flat, y_flat, z_flat, counts_flat = pl.pallas_call(
            bin_kernel,
            out_shape=out_shapes,
            in_specs=[any_spec] * 10,
            out_specs=[any_spec] * 5,
            grid=(B, n_bin_blocks),
            input_output_aliases={5: 0, 6: 1, 7: 2, 8: 3, 9: 4},
            interpret=False,
            compiler_params=resolve_pallas_compiler_params("cell_prune", grid_blocks=B * n_bin_blocks),
            name="pallas_bin_cells_batch",
        )(
            cell_id.reshape(B * N),
            pos_wrapped[:, :, 0].astype(jnp.float32).reshape(B * N),
            pos_wrapped[:, :, 1].astype(jnp.float32).reshape(B * N),
            pos_wrapped[:, :, 2].astype(jnp.float32).reshape(B * N),
            reb_i32,
            id_init,
            x_init,
            y_init,
            z_init,
            counts_init,
        )
        cell_atom_b = id_flat[:-1].reshape((B, G_cells * C))
        cell_x_b = x_flat[:-1].reshape((B, G_cells * C))
        cell_y_b = y_flat[:-1].reshape((B, G_cells * C))
        cell_z_b = z_flat[:-1].reshape((B, G_cells * C))
        cell_counts_b = (
            jnp.minimum(counts_flat.reshape((B, G_cells + 1)), jnp.int32(C))
            .at[:, G_cells]
            .set(jnp.int32(0))
        )
        return cell_id, cell_atom_b, cell_x_b, cell_y_b, cell_z_b, cell_counts_b, G_cells

    def _bin_single(c_id: Array, pos_s: Array, reb_flag: Array) -> tuple[Array, Array, Array, Array, Array]:
        empty_res = (
            jnp.full((G_cells * C,), N, dtype=jnp.int32),
            jnp.zeros((G_cells * C,), dtype=jnp.float32),
            jnp.zeros((G_cells * C,), dtype=jnp.float32),
            jnp.zeros((G_cells * C,), dtype=jnp.float32),
            jnp.zeros((G_cells + 1,), dtype=jnp.int32),
        )
        if N == 0:
            return empty_res

        def _do_argsort(_):
            order = jnp.argsort(c_id).astype(jnp.int32)
            sorted_c = c_id[order]
            idx_in_n = jnp.arange(N, dtype=jnp.int32)
            is_start = jnp.concatenate([jnp.array([True]), sorted_c[1:] != sorted_c[:-1]])
            is_end = jnp.concatenate([sorted_c[:-1] != sorted_c[1:], jnp.array([True])])

            start_dest = jnp.where(is_start, sorted_c, jnp.int32(G_cells))
            cell_start = jnp.zeros((G_cells + 1,), dtype=jnp.int32).at[start_dest].set(idx_in_n)
            group_start = cell_start[sorted_c]
            slot = idx_in_n - group_start

            is_real_cell = sorted_c < jnp.int32(G_cells)
            valid_slot = (slot < C) & is_real_cell
            flat_dest = jnp.where(valid_slot, sorted_c * C + slot, jnp.int32(G_cells * C))

            buf_id = jnp.full((G_cells * C + 1,), N, dtype=jnp.int32).at[flat_dest].set(order)[:-1]
            pos_sorted = pos_s[order].astype(jnp.float32)
            buf_x = jnp.zeros((G_cells * C + 1,), dtype=jnp.float32).at[flat_dest].set(pos_sorted[:, 0])[:-1]
            buf_y = jnp.zeros((G_cells * C + 1,), dtype=jnp.float32).at[flat_dest].set(pos_sorted[:, 1])[:-1]
            buf_z = jnp.zeros((G_cells * C + 1,), dtype=jnp.float32).at[flat_dest].set(pos_sorted[:, 2])[:-1]

            end_dest = jnp.where(is_end & is_real_cell, sorted_c, jnp.int32(G_cells))
            cell_counts = (
                jnp.zeros((G_cells + 1,), dtype=jnp.int32)
                .at[end_dest]
                .set(jnp.minimum(slot + jnp.int32(1), jnp.int32(C)))
                .at[G_cells]
                .set(jnp.int32(0))
            )
            return buf_id, buf_x, buf_y, buf_z, cell_counts

        return jax.lax.cond(reb_flag > jnp.int32(0), _do_argsort, lambda _: empty_res, None)

    cell_atom_b, cell_x_b, cell_y_b, cell_z_b, cell_counts_b = jax.vmap(_bin_single)(
        cell_id, pos_wrapped, reb_i32
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
    use_scratch_slot: bool = False,
):
    """Build Pallas v3.0 (MK-47) kernel: L2-cached cell lookup, dynamic count-bounded tiling, gated exclusion match, and O(1) ghost/non-rebuilding CTA skip."""
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
        nbr_init_ref,
        nbr_out_ref,
        occ_out_ref,
    ):
        del nbr_init_ref
        i_atom = pl.program_id(0)
        b_id = pl.program_id(1)
        r_sq_cutoff = jnp.float32(r_sq_max_f)
        n_sentinel = jnp.int32(N)
        lane_c = jnp.arange(BLOCK_C, dtype=jnp.int32)
        lane_m = jnp.arange(M_BLOCK, dtype=jnp.int32)

        base_atom = b_id * N + i_atom
        c_i = _pl_load(cell_id_ref, (base_atom,))
        zero_hits = c_i - c_i

        def _run_active_atom(h0):
            px_i = _pl_load(px_ref, (base_atom,))
            py_i = _pl_load(py_ref, (base_atom,))
            pz_i = _pl_load(pz_ref, (base_atom,))

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
            dummy_nbr_slot = jnp.int32(B * N * K) if use_scratch_slot else jnp.int32(0)

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

            hits = jax.lax.fori_loop(jnp.int32(0), jnp.int32(27), _stencil_body, h0)
            start_pad_tile = jnp.minimum(hits, jnp.int32(K)) // jnp.int32(BLOCK_C)
            sentinel_vec = jnp.full((BLOCK_C,), n_sentinel, dtype=jnp.int32)

            def _pad_body(pad_t, _):
                slot = pad_t * BLOCK_C + lane_c
                should_pad = (slot >= hits) & (slot < jnp.int32(K))
                safe_dest = jnp.where(should_pad, base_nbr_row + slot, dummy_nbr_slot)
                _pl_store(nbr_out_ref, (safe_dest,), sentinel_vec, mask=should_pad)

            jax.lax.fori_loop(start_pad_tile, jnp.int32(K_blocks), _pad_body, None)
            return hits

        total_hits = jax.lax.cond(
            c_i < jnp.int32(G_cells),
            _run_active_atom,
            lambda h0: h0,
            zero_hits,
        )
        _pl_store(occ_out_ref, (base_atom,), total_hits)

    return _kernel


def _max_k_tiles_from_idx(idx: Array, n_atoms: int | None = None) -> Array:
    """Compute per-32-atom-block active K-tile count from an ``(N, K)`` or ``(B, N, K)`` neighbor array."""
    is_batched = idx.ndim == 3
    idx_b = idx if is_batched else idx[None, ...]
    B_i, N_i, K_i = int(idx_b.shape[0]), int(idx_b.shape[1]), int(idx_b.shape[2])
    eff_n = int(n_atoms) if n_atoms is not None else N_i
    n_atom_blocks = max(1, (N_i + 31) // 32)
    if K_i == 0 or N_i == 0:
        out = jnp.zeros((B_i, n_atom_blocks), dtype=jnp.int32)
        return out if is_batched else out[0]
    k_blocks = max(1, (K_i + 31) // 32)
    pad_n = n_atom_blocks * 32 - N_i
    pad_k = k_blocks * 32 - K_i
    if K_i >= 128:
        first_cols = idx_b[:, :, ::32].astype(jnp.int32)
        raw_first = jnp.where(first_cols >= 0, first_cols & jnp.int32(_NB_INDEX_MASK), first_cols)
        col_valid = (raw_first >= 0) & (raw_first < eff_n)
        if pad_n > 0:
            col_valid = jnp.pad(col_valid, ((0, 0), (0, pad_n), (0, 0)), constant_values=False)
        tile_valid = jnp.any(col_valid.reshape((B_i, n_atom_blocks, 32, k_blocks)), axis=2)
    else:
        raw_all = idx_b.astype(jnp.int32)
        raw_idx = jnp.where(raw_all >= 0, raw_all & jnp.int32(_NB_INDEX_MASK), raw_all)
        val_mask = (raw_idx >= 0) & (raw_idx < eff_n)
        if pad_n > 0 or pad_k > 0:
            val_mask = jnp.pad(val_mask, ((0, 0), (0, pad_n), (0, pad_k)), constant_values=False)
        tile_valid = jnp.any(val_mask.reshape((B_i, n_atom_blocks, 32, k_blocks, 32)), axis=(2, 4))
    tile_idx = jnp.arange(1, k_blocks + 1, dtype=jnp.int32)[None, None, :]
    mkt = jnp.max(jnp.where(tile_valid, tile_idx, jnp.int32(0)), axis=-1).astype(jnp.int32)
    return mkt if is_batched else mkt[0]


def precompute_static_nl_exclusion_tables(
    excl_indices: Array,
    excl_scales_vdw: Array,
    excl_scales_elec: Array,
    n_atoms: int,
    *,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> tuple[Array, Array, Array, Array]:
    """Precompute ``(excl_min, excl_max, excl_idx_32, excl_tag_32)`` once outside the MD step loop (MK-56)."""
    M_BLOCK = 32
    is_batched = excl_indices.ndim == 3
    ei_b = excl_indices if is_batched else excl_indices[None, ...]
    ev_b = excl_scales_vdw if is_batched else excl_scales_vdw[None, ...]
    ee_b = excl_scales_elec if is_batched else excl_scales_elec[None, ...]
    N = int(n_atoms)
    M_raw = int(ei_b.shape[2])
    if M_raw == 0:
        B_e = int(ei_b.shape[0])
        z_min = jnp.full((B_e, N), 0x7FFFFFFF, dtype=jnp.int32)
        z_max = jnp.full((B_e, N), -1, dtype=jnp.int32)
        z_idx = jnp.full((B_e, N, M_BLOCK), -1, dtype=jnp.int32)
        z_tag = jnp.zeros((B_e, N, M_BLOCK), dtype=jnp.int32)
        if is_batched:
            return z_min, z_max, z_idx, z_tag
        return z_min[0], z_max[0], z_idx[0], z_tag[0]

    ei_clip = ei_b[:, :, :M_BLOCK].astype(jnp.int32)
    ev_clip = ev_b[:, :, :M_BLOCK].astype(jnp.float32)
    ee_clip = ee_b[:, :, :M_BLOCK].astype(jnp.float32)
    pad_m = max(0, M_BLOCK - int(ei_clip.shape[2]))
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
    if is_batched:
        return excl_min, excl_max, ei_clip, excl_tag
    return excl_min[0], excl_max[0], ei_clip[0], excl_tag[0]


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
    atom_mask: Array | None = None,
    mask_batched: bool = False,
    rebuild_mask: Array | None = None,
    old_idx: Array | None = None,
    precomputed_excl: tuple[Array, Array, Array, Array] | None = None,
    return_max_k_tiles: bool = False,
    old_max_k_tiles: Array | None = None,
    n_active_atoms: int | None = None,
) -> tuple[Array, Array] | tuple[Array, Array, Array]:
    """Batched 2D-grid ``(N_act, B)`` Pallas cell-list pruning + in-register exclusion encoding."""
    B = int(axis_size)
    pos_b = position if in_batched[0] else jnp.broadcast_to(position[None, ...], (B,) + position.shape)
    bx_b = box if in_batched[1] else box[None, ...]
    ei_b = excl_indices if in_batched[2] else excl_indices[None, ...]
    ev_b = excl_scales_vdw if in_batched[3] else excl_scales_vdw[None, ...]
    ee_b = excl_scales_elec if in_batched[4] else excl_scales_elec[None, ...]
    if atom_mask is not None:
        mask_b = atom_mask if mask_batched else atom_mask[None, ...]
    else:
        mask_b = None

    N = int(pos_b.shape[1])
    N_act = _resolve_nl_active_atoms(n_active_atoms, N, mask_b)
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
    ) = _build_cell_stencil_buffers_batched(
        pos_f32,
        bx_3,
        float(cell_size),
        C_pad,
        g_dims,
        atom_mask_b=mask_b,
        rebuild_mask_b=rebuild_mask,
        interpret=interpret,
        n_active_atoms=N_act,
    )

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

    if encode_exclusions and precomputed_excl is not None:
        em_in, ex_in, ei32_in, et32_in = precomputed_excl
        em_b = em_in if em_in.ndim == 2 else em_in[None, ...]
        ex_b = ex_in if ex_in.ndim == 2 else ex_in[None, ...]
        ei32_b = ei32_in if ei32_in.ndim == 3 else ei32_in[None, ...]
        et32_b = et32_in if et32_in.ndim == 3 else et32_in[None, ...]
        B_excl = int(ei32_b.shape[0])
        excl_min_flat = em_b.astype(jnp.int32).reshape(B_excl * N)
        excl_max_flat = ex_b.astype(jnp.int32).reshape(B_excl * N)
        excl_idx_flat = ei32_b.astype(jnp.int32).reshape(B_excl * N * M_BLOCK)
        excl_tag_flat = et32_b.astype(jnp.int32).reshape(B_excl * N * M_BLOCK)
        do_encode = True
    else:
        B_excl = int(ei_b.shape[0])
        M_raw = int(ei_b.shape[2]) if ei_b.ndim == 3 else 0
        if encode_exclusions and M_raw > 0:
            excl_min, excl_max, ei_clip, excl_tag = precompute_static_nl_exclusion_tables(
                ei_b,
                ev_b,
                ee_b,
                N,
                scale_14_vdw=scale_14_vdw,
                scale_14_elec=scale_14_elec,
            )
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

    use_scratch_slot = bool(interpret)
    if use_scratch_slot:
        if old_idx is not None:
            nbr_init_flat = jnp.pad(
                jnp.asarray(old_idx, dtype=jnp.int32).reshape(B * N * K),
                (0, 1),
                constant_values=N,
            )
        else:
            nbr_init_flat = jnp.full((B * N * K + 1,), N, dtype=jnp.int32)
        nbr_out_len = B * N * K + 1
    else:
        if old_idx is not None:
            nbr_init_flat = jnp.asarray(old_idx, dtype=jnp.int32).reshape(B * N * K)
        else:
            nbr_init_flat = jnp.full((B * N * K,), N, dtype=jnp.int32)
        nbr_out_len = B * N * K

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
        use_scratch_slot,
    )
    out_shapes = (
        jax.ShapeDtypeStruct((nbr_out_len,), jnp.int32),
        jax.ShapeDtypeStruct((B * N,), jnp.int32),
    )
    nbr_flat, occ_flat = pl.pallas_call(
        kernel_fn,
        out_shape=out_shapes,
        grid=(N_act, B),
        input_output_aliases={15: 0},
        interpret=interpret,
        compiler_params=resolve_pallas_compiler_params("cell_prune", grid_blocks=N_act * B)
        if not interpret
        else None,
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
        nbr_init_flat,
    )
    occ_2d_raw = occ_flat.reshape((B, N))
    occ_2d_act = occ_2d_raw[:, :N_act]
    occ_b = jnp.max(occ_2d_act, axis=-1)
    nbr_out_b = (nbr_flat[:-1] if use_scratch_slot else nbr_flat).reshape((B, N, K))
    if return_max_k_tiles:
        n_atom_blocks = max(1, (N + 31) // 32)
        pad_n = n_atom_blocks * 32 - N_act
        occ_pad = jnp.pad(occ_2d_act, ((0, 0), (0, pad_n))) if pad_n > 0 else occ_2d_act[:, : n_atom_blocks * 32]
        max_occ_blk = jnp.minimum(
            jnp.max(occ_pad.reshape((B, n_atom_blocks, 32)), axis=-1),
            jnp.int32(K),
        )
        mkt_b = (max_occ_blk + jnp.int32(31)) // jnp.int32(32)
        if rebuild_mask is not None and old_max_k_tiles is not None:
            su_bool = jnp.asarray(rebuild_mask, dtype=jnp.bool_).reshape((B, 1))
            mkt_b = jnp.where(su_bool, mkt_b, jnp.asarray(old_max_k_tiles, dtype=jnp.int32))
        return nbr_out_b, occ_b, mkt_b
    return nbr_out_b, occ_b


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
    atom_mask: Array | None = None,
    precomputed_excl: tuple[Array, Array, Array, Array] | None = None,
    return_max_k_tiles: bool = False,
    n_active_atoms: int | None = None,
) -> tuple[Array, Array] | tuple[Array, Array, Array]:
    """Prune 3D cell-list stencil candidates and pre-encode exclusions in one Pallas kernel."""
    cs = float(cell_size)
    cc = int(cell_capacity)
    mo = int(max_occupancy)
    rc = float(r_cutoff)
    s14_v = float(scale_14_vdw)
    s14_e = float(scale_14_elec)
    enc = bool(encode_exclusions)
    interp = bool(interpret)
    ret_mkt = bool(return_max_k_tiles)
    n_act = None if n_active_atoms is None else int(n_active_atoms)
    g_dims = (
        grid_dims
        if grid_dims is not None
        else (None if isinstance(box, jax.core.Tracer) else _compute_static_grid_dims(box, cs))
    )

    if atom_mask is not None:
        @jax.custom_batching.custom_vmap
        def _vmappable_masked(pos, bx, ei, ev, ee, am):
            res = _pallas_prune_and_encode_cell_list_batch(
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
                atom_mask=am,
                mask_batched=False,
                precomputed_excl=precomputed_excl,
                return_max_k_tiles=ret_mkt,
                n_active_atoms=n_act,
            )
            if ret_mkt:
                return res[0][0], res[1][0], res[2][0]
            return res[0][0], res[1][0]

        @_vmappable_masked.def_vmap
        def _vmap_rule_masked(axis_size, in_batched, pos, bx, ei, ev, ee, am):
            res = _pallas_prune_and_encode_cell_list_batch(
                int(axis_size),
                tuple(bool(b) for b in in_batched[:5]),
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
                atom_mask=am,
                mask_batched=bool(in_batched[5]),
                precomputed_excl=precomputed_excl,
                return_max_k_tiles=ret_mkt,
                n_active_atoms=n_act,
            )
            if ret_mkt:
                return res, (True, True, True)
            return res, (True, True)

        return _vmappable_masked(
            position,
            box,
            excl_indices,
            excl_scales_vdw,
            excl_scales_elec,
            jnp.asarray(atom_mask, dtype=jnp.bool_),
        )

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, bx, ei, ev, ee):
        res = _pallas_prune_and_encode_cell_list_batch(
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
            precomputed_excl=precomputed_excl,
            return_max_k_tiles=ret_mkt,
            n_active_atoms=n_act,
        )
        if ret_mkt:
            return res[0][0], res[1][0], res[2][0]
        return res[0][0], res[1][0]

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, bx, ei, ev, ee):
        res = _pallas_prune_and_encode_cell_list_batch(
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
            precomputed_excl=precomputed_excl,
            return_max_k_tiles=ret_mkt,
            n_active_atoms=n_act,
        )
        if ret_mkt:
            return res, (True, True, True)
        return res, (True, True)

    return _vmappable(position, box, excl_indices, excl_scales_vdw, excl_scales_elec)


class TilePairNeighborList(NamedTuple):
    """Spatially sorted 32x32 tile-pair neighbor list with 32-bit exclusion bitmasks (MK-51 / MK-60)."""

    tile_pairs: Array
    excl_full_mask: Array
    excl_14_mask: Array
    num_valid_pairs: Array
    atom_perm: Array | None = None
    atom_inv_perm: Array | None = None


def _build_tile_pair_aabb_single(
    pos: Array,
    box: Array | None,
    excl_indices: Array | None,
    excl_scales_vdw: Array | None,
    excl_scales_elec: Array | None,
    atom_mask: Array | None,
    cutoff: float,
    max_tile_pairs: int,
    morton_sort: bool,
    n_active_atoms: int,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> TilePairNeighborList:
    """Single-replica GPU Morton-sorted 32-atom AABB tile-pair + 32-bit exclusion builder (MK-60)."""
    N = int(pos.shape[0])
    N_act = max(1, min(int(n_active_atoms), N)) if N > 0 else 0
    n_act_blocks = max(1, (N_act + 31) // 32)
    N_act_pad = n_act_blocks * 32
    pad_act = N_act_pad - N_act

    if atom_mask is None:
        am_full = jnp.arange(N, dtype=jnp.int32) < jnp.int32(N_act)
    else:
        am_full = jnp.asarray(atom_mask, dtype=jnp.bool_) & (
            jnp.arange(N, dtype=jnp.int32) < jnp.int32(N_act)
        )

    pos_f32 = pos.astype(jnp.float32)
    has_box = box is not None
    if has_box:
        b_arr = jnp.asarray(box, dtype=jnp.float32)
        L = jnp.diag(b_arr) if (b_arr.ndim == 2 and b_arr.shape == (3, 3)) else b_arr[:3]
        safe_L = jnp.where(L > 0.0, L, jnp.ones_like(L))
        inv_L = jnp.where(L > 0.0, 1.0 / safe_L, jnp.zeros_like(L))
    else:
        L = jnp.zeros((3,), dtype=jnp.float32)
        safe_L = jnp.ones((3,), dtype=jnp.float32)
        inv_L = jnp.zeros((3,), dtype=jnp.float32)

    perm_out: Array | None = None
    inv_perm_out: Array | None = None
    if morton_sort and N > 0:
        perm_act, inv_perm_act = compute_morton_codes_3d(
            pos_f32[:N_act],
            box=L if has_box else None,
            atom_mask=am_full[:N_act],
        )
        if N_act < N:
            tail = jnp.arange(N_act, N, dtype=jnp.int32)
            perm_out = jnp.concatenate([perm_act, tail], axis=0)
            inv_perm_out = jnp.concatenate([inv_perm_act, tail], axis=0)
        else:
            perm_out = perm_act
            inv_perm_out = inv_perm_act
        pos_act = pos_f32[perm_act]
        am_act = am_full[perm_act]
    else:
        pos_act = pos_f32[:N_act]
        am_act = am_full[:N_act]

    if has_box:
        pos_act = jnp.where(
            L[None, :] > 0.0,
            pos_act - L[None, :] * jnp.floor(pos_act * inv_L[None, :]),
            pos_act,
        )

    if pad_act > 0:
        pos_pad = jnp.pad(pos_act, ((0, pad_act), (0, 0)))
        am_pad = jnp.pad(am_act, (0, pad_act), constant_values=False)
    else:
        pos_pad = pos_act
        am_pad = am_act

    pos_tiles = pos_pad.reshape(n_act_blocks, 32, 3)
    mask_tiles = am_pad.reshape(n_act_blocks, 32)
    tile_has_real = jnp.any(mask_tiles, axis=1)

    first_real = jnp.argmax(mask_tiles.astype(jnp.int32), axis=1)
    anchor = jnp.take_along_axis(pos_tiles, first_real[:, None, None], axis=1)[:, 0, :]
    rel = pos_tiles - anchor[:, None, :]
    if has_box:
        rel = rel - L[None, None, :] * jnp.floor(rel * inv_L[None, None, :] + 0.5)
    rel = jnp.where(mask_tiles[:, :, None], rel, 0.0)
    rel_min = jnp.min(rel, axis=1)
    rel_max = jnp.max(rel, axis=1)
    center = anchor + 0.5 * (rel_min + rel_max)
    half_ext = 0.5 * (rel_max - rel_min)

    dc = center[:, None, :] - center[None, :, :]
    if has_box:
        dc = dc - L[None, None, :] * jnp.floor(dc * inv_L[None, None, :] + 0.5)
    gap = jnp.maximum(0.0, jnp.abs(dc) - (half_ext[:, None, :] + half_ext[None, :, :]))
    dist_sq = jnp.sum(gap * gap, axis=-1)

    t_arange = jnp.arange(n_act_blocks, dtype=jnp.int32)
    upper_tri = t_arange[:, None] <= t_arange[None, :]
    r_cut_sq = jnp.float32(float(cutoff) * float(cutoff))
    adj = upper_tri & tile_has_real[:, None] & tile_has_real[None, :] & (dist_sq <= r_cut_sq)

    M_pairs = max(1, int(max_tile_pairs))
    has_excl = excl_indices is not None and int(excl_indices.shape[-1]) > 0
    if has_excl and excl_indices is not None:
        M_excl = int(excl_indices.shape[-1])
        ei_raw = excl_indices.astype(jnp.int32)
        ev_raw = (
            jnp.zeros_like(ei_raw, dtype=jnp.float32)
            if excl_scales_vdw is None
            else excl_scales_vdw.astype(jnp.float32)
        )
        ee_raw = (
            jnp.zeros_like(ei_raw, dtype=jnp.float32)
            if excl_scales_elec is None
            else excl_scales_elec.astype(jnp.float32)
        )
        if perm_out is not None and inv_perm_out is not None:
            p_act = perm_out[:N_act]
            ei_act = ei_raw[p_act]
            ev_act = ev_raw[p_act]
            ee_act = ee_raw[p_act]
            valid_raw = (ei_act >= 0) & (ei_act < N) & am_act[:, None]
            ei_mapped = jnp.where(
                valid_raw,
                inv_perm_out[jnp.clip(ei_act, 0, max(N - 1, 0))],
                jnp.int32(-1),
            )
        else:
            ei_act = ei_raw[:N_act]
            ev_act = ev_raw[:N_act]
            ee_act = ee_raw[:N_act]
            ei_mapped = jnp.where(
                (ei_act >= 0) & (ei_act < N_act) & am_act[:, None],
                ei_act,
                jnp.int32(-1),
            )

        if pad_act > 0:
            ei_pad = jnp.pad(ei_mapped, ((0, pad_act), (0, 0)), constant_values=-1)
            ev_pad = jnp.pad(ev_act, ((0, pad_act), (0, 0)), constant_values=1.0)
            ee_pad = jnp.pad(ee_act, ((0, pad_act), (0, 0)), constant_values=1.0)
        else:
            ei_pad = ei_mapped
            ev_pad = ev_act
            ee_pad = ee_act

        valid_e = (ei_pad >= 0) & (ei_pad < N_act_pad)
        tol = 1e-4
        is_full_e = valid_e & (jnp.abs(ev_pad) <= tol) & (jnp.abs(ee_pad) <= tol)
        is_14_e = (
            valid_e
            & (~is_full_e)
            & (
                (jnp.abs(ev_pad - float(scale_14_vdw)) <= tol)
                | (jnp.abs(ee_pad - float(scale_14_elec)) <= tol)
                | (ev_pad < 1.0 - tol)
                | (ee_pad < 1.0 - tol)
            )
        )
        partner_tile = jnp.where(valid_e, ei_pad >> 5, jnp.int32(-2))
        partner_bit = jnp.where(valid_e, jnp.int32(1) << (ei_pad & jnp.int32(31)), jnp.int32(0))

        # Ensure any excluded atom pair's tile pair is included in adj
        t_row = (jnp.arange(N_act_pad, dtype=jnp.int32)[:, None] >> 5)
        t_col = jnp.where(valid_e, partner_tile, 0)
        t_lo = jnp.where(valid_e, jnp.minimum(t_row, t_col), 0)
        t_hi = jnp.where(valid_e, jnp.maximum(t_row, t_col), 0)
        adj = adj.at[t_lo.reshape(-1), t_hi.reshape(-1)].max(valid_e.reshape(-1))

    num_valid = jnp.sum(adj.astype(jnp.int32))
    ti_idx, tj_idx = jnp.nonzero(adj, size=M_pairs, fill_value=-1)
    valid_tp = jnp.arange(M_pairs, dtype=jnp.int32) < num_valid
    ti_idx = jnp.where(valid_tp, ti_idx.astype(jnp.int32), jnp.int32(-1))
    tj_idx = jnp.where(valid_tp, tj_idx.astype(jnp.int32), jnp.int32(-1))
    tile_pairs = jnp.stack([ti_idx, tj_idx], axis=-1)

    if has_excl and excl_indices is not None:
        safe_ti = jnp.clip(ti_idx, 0, n_act_blocks - 1)
        safe_tj = jnp.clip(tj_idx, 0, n_act_blocks - 1)
        pt_blk = partner_tile.reshape(n_act_blocks, 32, M_excl)
        pb_blk = partner_bit.reshape(n_act_blocks, 32, M_excl)
        full_blk = is_full_e.reshape(n_act_blocks, 32, M_excl)
        e14_blk = is_14_e.reshape(n_act_blocks, 32, M_excl)
        r_bits = jnp.int32(1) << jnp.arange(32, dtype=jnp.int32)[:, None]
        c_bits = jnp.int32(1) << jnp.arange(32, dtype=jnp.int32)[None, :]

        def _mask_for_pair(ti_p, tj_p, v_p):
            match_ij = pt_blk[ti_p] == tj_p
            f_ij = jnp.bitwise_or.reduce(
                jnp.where(match_ij & full_blk[ti_p], pb_blk[ti_p], jnp.int32(0)),
                axis=-1,
            )
            s_ij = jnp.bitwise_or.reduce(
                jnp.where(match_ij & e14_blk[ti_p], pb_blk[ti_p], jnp.int32(0)),
                axis=-1,
            )
            match_ji = pt_blk[tj_p] == ti_p
            f_ji_cols = jnp.bitwise_or.reduce(
                jnp.where(match_ji & full_blk[tj_p], pb_blk[tj_p], jnp.int32(0)),
                axis=-1,
            )
            s_ji_cols = jnp.bitwise_or.reduce(
                jnp.where(match_ji & e14_blk[tj_p], pb_blk[tj_p], jnp.int32(0)),
                axis=-1,
            )
            f_ji = jnp.bitwise_or.reduce(
                jnp.where((f_ji_cols[None, :] & r_bits) != 0, c_bits, jnp.int32(0)),
                axis=-1,
            )
            s_ji = jnp.bitwise_or.reduce(
                jnp.where((s_ji_cols[None, :] & r_bits) != 0, c_bits, jnp.int32(0)),
                axis=-1,
            )
            f_out = jnp.where(v_p, f_ij | f_ji, jnp.int32(0))
            s_out = jnp.where(v_p, (s_ij | s_ji) & (~f_out), jnp.int32(0))
            return f_out, s_out

        excl_full_mask, excl_14_mask = jax.vmap(_mask_for_pair)(safe_ti, safe_tj, valid_tp)
    else:
        excl_full_mask = jnp.zeros((M_pairs, 32), dtype=jnp.int32)
        excl_14_mask = jnp.zeros((M_pairs, 32), dtype=jnp.int32)

    pk_mod = sys.modules.get("prolix.pallas_kernels")
    tp_cls = getattr(pk_mod, "TilePairNeighborList", TilePairNeighborList)
    return tp_cls(
        tile_pairs=tile_pairs,
        excl_full_mask=excl_full_mask,
        excl_14_mask=excl_14_mask,
        num_valid_pairs=num_valid,
        atom_perm=perm_out,
        atom_inv_perm=inv_perm_out,
    )


def build_tile_pair_neighbor_list_gpu(
    positions: Array,
    box: Array | None = None,
    *,
    cutoff: float = 9.0,
    skin: float = 0.0,
    excl_indices: Array | None = None,
    excl_scales_vdw: Array | None = None,
    excl_scales_elec: Array | None = None,
    atom_mask: Array | None = None,
    max_tile_pairs: int | None = None,
    max_pairs: int | None = None,
    morton_sort: bool = True,
    n_active_atoms: int | None = None,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
) -> TilePairNeighborList:
    """Fast on-device GPU 3D Morton AABB tile-pair + 32-bit exclusion bitmask builder (MK-60).

    Constructs ``TilePairNeighborList`` directly from ``positions`` and ``box`` using
    32-atom Morton tile bounding boxes without materializing or permuting an ``(N, K)``
    atom-row neighbor list or sorting ``excl_indices``.
    """
    pos_arr = jnp.asarray(positions)
    is_batched = pos_arr.ndim == 3
    N = int(pos_arr.shape[-2])
    N_act = _resolve_nl_active_atoms(n_active_atoms, N, atom_mask)
    n_act_blocks = max(1, (N_act + 31) // 32)
    max_ut = n_act_blocks * (n_act_blocks + 1) // 2
    r_search = float(cutoff) + float(skin)

    if max_tile_pairs is None and max_pairs is not None:
        max_tile_pairs = int(max_pairs)
    if max_tile_pairs is None:
        # Conservative static upper bound for 3D Morton-sorted 32-atom tiles (~48 tile partners/tile)
        max_tile_pairs = min(max_ut, max(64, n_act_blocks * 48))

    if not is_batched:
        return _build_tile_pair_aabb_single(
            pos_arr,
            box,
            excl_indices,
            excl_scales_vdw,
            excl_scales_elec,
            atom_mask,
            r_search,
            int(max_tile_pairs),
            bool(morton_sort),
            int(N_act),
            scale_14_vdw=float(scale_14_vdw),
            scale_14_elec=float(scale_14_elec),
        )

    B = int(pos_arr.shape[0])
    b_bx = box is not None and ((box.ndim == 2 and box.shape != (3, 3)) or box.ndim == 3)
    b_ei = excl_indices is not None and excl_indices.ndim == 3
    b_ev = excl_scales_vdw is not None and excl_scales_vdw.ndim == 3
    b_ee = excl_scales_elec is not None and excl_scales_elec.ndim == 3
    b_am = atom_mask is not None and atom_mask.ndim == 2

    return jax.vmap(
        lambda b: _build_tile_pair_aabb_single(
            pos_arr[b],
            (box[b] if b_bx else box) if box is not None else None,
            (excl_indices[b] if b_ei else excl_indices) if excl_indices is not None else None,
            (excl_scales_vdw[b] if b_ev else excl_scales_vdw) if excl_scales_vdw is not None else None,
            (excl_scales_elec[b] if b_ee else excl_scales_elec) if excl_scales_elec is not None else None,
            (atom_mask[b] if b_am else atom_mask) if atom_mask is not None else None,
            r_search,
            int(max_tile_pairs),
            bool(morton_sort),
            int(N_act),
            scale_14_vdw=float(scale_14_vdw),
            scale_14_elec=float(scale_14_elec),
        )
    )(jnp.arange(B, dtype=jnp.int32))


