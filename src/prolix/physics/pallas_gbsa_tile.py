"""2-pass 32x32 symmetric tile-pair OBC2 GBSA + ACE nonpolar + LJ/Coulomb engine (MK-67, MK-73).

Eliminates row-NL gather overhead and 2x triangular CTA load imbalance on implicit-solvent
systems (e.g. ``5DFR_gbsa_implicit`` 2,489 atoms at ``B=1`` and multi-replica ``B=32``):

1. ``GBSATilePairList`` & ``build_gbsa_tile_pair_list``: Builds diagonal-striped dense
   upper-triangular ``(ti, tj)`` tile pairs for ``N <= 8,192`` or Morton-sorted AABB
   cutoff tile pairs for larger systems, encoding 1-2/1-3 and 1-4 exclusions as compact
   32-bit ``(M, 32)`` bitmasks and flattening multi-replica ragged batches onto 1D
   ``RaggedWorkQueue`` schedules with zero ghost CTAs.
2. ``pallas_gbsa_tile`` & ``pallas_gbsa_tile_energy_and_forces``:
   - **Pass 1**: 32x32 symmetric tile-pair Born integral ``I_i`` accumulation + fused 1D
     OBC2 ``tanh`` Born radii ``R_i``, ``dR_i/dI_i``, screened self-GB, and exact AKMA
     ACE nonpolar SASA energy ``4 * pi * sigma_np * (r_i + r_p)^2 * (r_i / R_i)^6``.
   - **Pass 2**: 32x32 symmetric tile-pair screened Generalized Born ``f_GB`` + direct
     Coulomb + softcore/switched LJ + 1-load 32-bit exclusion masks + symmetric Born
     chain-rule forces aliased in place.
   - **MK-73 Tile-Pair ``@jax.custom_vjp``**: Analytical ``jax.grad`` wrt ``positions``,
     ``charges``, ``radii``, ``scaled_radii``, ``sigmas``, and ``epsilons`` in
     ``O(tile_pairs)`` memory with zero dense ``(N, N)`` matrices and full
     ``prolix.export.in_export_context()`` pure-JAX fallback.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from proxide.physics import constants

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.generalized_born import (
    ALPHA_OBC,
    BETA_OBC,
    GAMMA_OBC,
    GB_COULOMB_CONSTANT,
    GB_KAPPA_SCALE,
    SURFACE_TENSION,
    compute_debye_kappa,
    compute_pair_integral,
)
from prolix.physics.neighbor_list import compute_morton_codes_3d
from prolix.physics.optimization import _lj_switch_weight_and_deriv
from prolix.physics.pallas_bonded import (
    _min_image3,
    _pl_atomic_add_1d,
    _triton_gpu_available,
)
from prolix.physics.pallas_implicit import _pair_integral_deriv
from prolix.physics.pallas_ragged_queue import (
    RaggedWorkQueue,
    build_ragged_work_queue,
)
from prolix.physics.pme import _is_perturbed, _unwrap_primal

_GB_TILE = 32


def _resolve_interpret(interpret: bool | None) -> bool:
    if interpret is not None:
        return bool(interpret)
    if os.environ.get("PROLIX_PALLAS_INTERPRET", "").strip() == "1":
        return True
    return not _triton_gpu_available()


def _any_block_spec() -> Any:
    bs_cls: Any = pl.BlockSpec
    any_mem: Any = pl.ANY
    return bs_cls(memory_space=any_mem)


def _normalize_box6_batch(
    box: jax.Array | np.ndarray | None, batch_size: int, dtype: jnp.dtype
) -> tuple[jax.Array, bool]:
    """Normalize optional box (None, (3,), (3,3), (B,3), (B,3,3), (B,6)) to ``(B, 6)``."""
    if box is None:
        return jnp.zeros((batch_size, 6), dtype=dtype), False
    b_arr = jnp.asarray(box, dtype=dtype)
    if b_arr.ndim == 1:
        if b_arr.shape[0] == 3:
            b6 = jnp.concatenate([b_arr, jnp.zeros((3,), dtype=dtype)], axis=0)[None, :]
        elif b_arr.shape[0] == 6:
            b6 = b_arr[None, :]
        else:
            raise ValueError(f"Unsupported 1D box shape {b_arr.shape}.")
    elif b_arr.ndim == 2 and b_arr.shape == (3, 3):
        b6 = jnp.stack(
            [b_arr[0, 0], b_arr[1, 1], b_arr[2, 2], b_arr[1, 0], b_arr[2, 0], b_arr[2, 1]]
        )[None, :]
    elif b_arr.ndim == 2 and b_arr.shape[1] in (3, 6):
        if b_arr.shape[1] == 3:
            b6 = jnp.concatenate([b_arr, jnp.zeros((b_arr.shape[0], 3), dtype=dtype)], axis=1)
        else:
            b6 = b_arr
    elif b_arr.ndim == 3 and b_arr.shape[1:] == (3, 3):
        b6 = jnp.stack(
            [
                b_arr[:, 0, 0],
                b_arr[:, 1, 1],
                b_arr[:, 2, 2],
                b_arr[:, 1, 0],
                b_arr[:, 2, 0],
                b_arr[:, 2, 1],
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported box shape {b_arr.shape}.")
    if b6.shape[0] == 1 and batch_size > 1:
        b6 = jnp.broadcast_to(b6, (batch_size, 6))
    return b6, True


def _normalize_box6_batch_np(
    box: jax.Array | np.ndarray | None,
    batch_size: int,
) -> tuple[np.ndarray, bool]:
    """Pure-NumPy host box normalization (safe inside ``jax.jit`` / ``jax.vmap``)."""
    if box is None or isinstance(box, jax.core.Tracer):
        return np.zeros((batch_size, 6), dtype=np.float64), box is not None
    b_arr = np.asarray(box, dtype=np.float64)
    if b_arr.ndim == 1:
        if b_arr.shape[0] == 3:
            b6 = np.concatenate([b_arr, np.zeros((3,), dtype=np.float64)], axis=0)[None, :]
        elif b_arr.shape[0] == 6:
            b6 = b_arr[None, :]
        else:
            raise ValueError(f"Unsupported 1D box shape {b_arr.shape}.")
    elif b_arr.ndim == 2 and b_arr.shape == (3, 3):
        b6 = np.asarray(
            [[b_arr[0, 0], b_arr[1, 1], b_arr[2, 2], b_arr[1, 0], b_arr[2, 0], b_arr[2, 1]]],
            dtype=np.float64,
        )
    elif b_arr.ndim == 2 and b_arr.shape[1] in (3, 6):
        if b_arr.shape[1] == 3:
            b6 = np.concatenate([b_arr, np.zeros((b_arr.shape[0], 3), dtype=np.float64)], axis=1)
        else:
            b6 = b_arr
    elif b_arr.ndim == 3 and b_arr.shape[1:] == (3, 3):
        b6 = np.stack(
            [
                b_arr[:, 0, 0],
                b_arr[:, 1, 1],
                b_arr[:, 2, 2],
                b_arr[:, 1, 0],
                b_arr[:, 2, 0],
                b_arr[:, 2, 1],
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported box shape {b_arr.shape}.")
    if b6.shape[0] == 1 and batch_size > 1:
        b6 = np.broadcast_to(b6, (batch_size, 6)).copy()
    return b6, True


@dataclass(frozen=True)
class GBSATilePairList:
    """Symmetric 32x32 tile-pair schedule + 32-bit exclusion bitmasks for OBC2 GBSA (MK-67)."""

    tile_pairs: jax.Array
    excl_full_mask: jax.Array
    excl_14_mask: jax.Array
    num_valid_pairs: jax.Array
    scale_14_vdw: float = 0.5
    scale_14_elec: float = 0.83333333
    atom_perm: jax.Array | None = None
    atom_inv_perm: jax.Array | None = None
    pair_queue: RaggedWorkQueue | None = None
    atom_queue: RaggedWorkQueue | None = None
    is_batched: bool = False

    def __init__(
        self,
        tile_pairs: jax.Array,
        excl_full_mask: jax.Array,
        excl_14_mask: jax.Array,
        num_valid_pairs: jax.Array,
        scale_14_vdw: float = 0.5,
        scale_14_elec: float = 0.83333333,
        atom_perm: jax.Array | None = None,
        atom_inv_perm: jax.Array | None = None,
        pair_queue: RaggedWorkQueue | None = None,
        atom_queue: RaggedWorkQueue | None = None,
        is_batched: bool = False,
    ) -> None:
        object.__setattr__(self, "tile_pairs", tile_pairs)
        object.__setattr__(self, "excl_full_mask", excl_full_mask)
        object.__setattr__(self, "excl_14_mask", excl_14_mask)
        object.__setattr__(self, "num_valid_pairs", num_valid_pairs)
        object.__setattr__(self, "scale_14_vdw", scale_14_vdw)
        object.__setattr__(self, "scale_14_elec", scale_14_elec)
        object.__setattr__(self, "atom_perm", atom_perm)
        object.__setattr__(self, "atom_inv_perm", atom_inv_perm)
        object.__setattr__(self, "pair_queue", pair_queue)
        object.__setattr__(self, "atom_queue", atom_queue)
        object.__setattr__(self, "is_batched", is_batched)

    def tree_flatten(self):
        children = (
            self.tile_pairs,
            self.excl_full_mask,
            self.excl_14_mask,
            self.num_valid_pairs,
            self.atom_perm,
            self.atom_inv_perm,
            self.pair_queue,
            self.atom_queue,
        )
        aux = (float(self.scale_14_vdw), float(self.scale_14_elec), bool(self.is_batched))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        s_vdw, s_elec, is_batched = aux
        tp, ef, e14, nvp, perm, inv_perm, pq, aq = children
        return cls(
            tile_pairs=tp,
            excl_full_mask=ef,
            excl_14_mask=e14,
            num_valid_pairs=nvp,
            scale_14_vdw=s_vdw,
            scale_14_elec=s_elec,
            atom_perm=perm,
            atom_inv_perm=inv_perm,
            pair_queue=pq,
            atom_queue=aq,
            is_batched=is_batched,
        )


jax.tree_util.register_pytree_node_class(GBSATilePairList)


def _pair_integral_param_derivs(
    r: jax.Array,
    or_a: jax.Array,
    sr_b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Analytical ``(dI/dor_a, dI/dsr_b)`` matching ``compute_pair_integral`` (MK-73)."""
    dtype = r.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    neg_one = jnp.asarray(-1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eps = jnp.asarray(1e-4, dtype=dtype)

    d_abs = jnp.abs(r - sr_b)
    L = jnp.maximum(or_a, d_abs)
    L_s = jnp.maximum(L, eps)
    U = r + sr_b
    U_s = jnp.maximum(U, eps)
    r_s = jnp.maximum(r, eps)

    inv_L = one / L_s
    inv_U = one / U_s
    inv_L2 = inv_L * inv_L
    inv_U2 = inv_U * inv_U
    inv_L3 = inv_L2 * inv_L
    inv_U3 = inv_U2 * inv_U
    sr2_sq = sr_b * sr_b
    r_minus_sr2_over_r = r - sr2_sq / r_s

    dtotal_dLs = (
        (zero - half) * inv_L2
        + quarter * inv_L / r_s
        + quarter * r_minus_sr2_over_r * inv_L3
    )
    dtotal_dUs = (
        half * inv_U2
        - quarter * inv_U / r_s
        - quarter * r_minus_sr2_over_r * inv_U3
    )
    valid = U > or_a

    # dL_s / dor_a = 1 iff or_a > |r - sr_b| and L > eps
    dor_mask = valid & (or_a > d_abs) & (L > eps)
    dI_dor_a = jnp.where(dor_mask, dtotal_dLs, zero)

    # dL_s / dsr_b = -sgn(r - sr_b) iff |r - sr_b| > or_a and L > eps
    sgn_d = jnp.where(r >= sr_b, one, neg_one)
    dLs_dsr = jnp.where((d_abs > or_a) & (L > eps), zero - sgn_d, zero)
    dUs_dsr = jnp.where(U > eps, one, zero)
    dexp_dsr = (zero - quarter) * (sr_b / r_s) * (inv_U2 - inv_L2)
    dI_dsr_b = jnp.where(valid, dtotal_dLs * dLs_dsr + dtotal_dUs * dUs_dsr + dexp_dsr, zero)
    return dI_dor_a, dI_dsr_b


def _extract_excl_pairs_np(
    n_atoms: int,
    idx_12_13: jax.Array | np.ndarray | None,
    idx_14: jax.Array | np.ndarray | None,
    excl_indices: jax.Array | np.ndarray | None,
    excl_scales_vdw: jax.Array | np.ndarray | None,
    excl_scales_elec: jax.Array | np.ndarray | None,
    scale_14_vdw: float,
    scale_14_elec: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalize either ``(idx_12_13, idx_14)`` or ``(excl_indices, excl_scales_*)`` into pair arrays."""
    s_vdw = float(scale_14_vdw)
    s_elec = float(scale_14_elec)
    full_list: list[np.ndarray] = []
    p14_list: list[np.ndarray] = []

    if idx_12_13 is not None:
        arr = np.asarray(idx_12_13, dtype=np.int32).reshape(-1, 2)
        if arr.size > 0:
            valid = (arr[:, 0] >= 0) & (arr[:, 1] >= 0) & (arr[:, 0] < n_atoms) & (arr[:, 1] < n_atoms)
            full_list.append(arr[valid])
    if idx_14 is not None:
        arr = np.asarray(idx_14, dtype=np.int32).reshape(-1, 2)
        if arr.size > 0:
            valid = (arr[:, 0] >= 0) & (arr[:, 1] >= 0) & (arr[:, 0] < n_atoms) & (arr[:, 1] < n_atoms)
            p14_list.append(arr[valid])

    if excl_indices is not None:
        ei = np.asarray(excl_indices, dtype=np.int32)
        K = ei.shape[-1]
        rows = np.broadcast_to(np.arange(n_atoms, dtype=np.int32)[:, None], (n_atoms, K))
        valid = (ei >= 0) & (ei < n_atoms) & (rows < ei)
        if np.any(valid):
            u_v = np.stack([rows[valid], ei[valid]], axis=-1)
            if excl_scales_vdw is not None or excl_scales_elec is not None:
                sv = (
                    np.asarray(excl_scales_vdw, dtype=np.float32)[valid]
                    if excl_scales_vdw is not None
                    else np.zeros((u_v.shape[0],), dtype=np.float32)
                )
                se = (
                    np.asarray(excl_scales_elec, dtype=np.float32)[valid]
                    if excl_scales_elec is not None
                    else np.zeros((u_v.shape[0],), dtype=np.float32)
                )
                is_zero = (np.abs(sv) < 1e-7) & (np.abs(se) < 1e-7)
                if np.any(is_zero):
                    full_list.append(u_v[is_zero])
                if np.any(~is_zero):
                    nz_sv = sv[~is_zero]
                    nz_se = se[~is_zero]
                    p14_list.append(u_v[~is_zero])
                    if idx_14 is None and nz_sv.size > 0:
                        s_vdw = float(nz_sv[0])
                    if idx_14 is None and nz_se.size > 0:
                        s_elec = float(nz_se[0])
            else:
                full_list.append(u_v)

    full_arr = np.concatenate(full_list, axis=0) if full_list else np.zeros((0, 2), dtype=np.int32)
    p14_arr = np.concatenate(p14_list, axis=0) if p14_list else np.zeros((0, 2), dtype=np.int32)
    return full_arr, p14_arr, s_vdw, s_elec


def _build_single_tile_pairs_np(
    pos_np: np.ndarray,
    mask_np: np.ndarray,
    box6_np: np.ndarray,
    has_box: bool,
    n_real: int,
    full_pairs: np.ndarray,
    p14_pairs: np.ndarray,
    cutoff: float | None,
    born_cutoff: float | None,
    skin: float,
    neighbor_idx: np.ndarray | None,
    use_morton: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray | None, np.ndarray | None]:
    n_atoms = int(pos_np.shape[0])
    n_act = max(0, min(int(n_real), n_atoms))
    if n_act == 0:
        return (
            np.full((1, 2), -1, dtype=np.int32),
            np.zeros((1, _GB_TILE), dtype=np.int32),
            np.zeros((1, _GB_TILE), dtype=np.int32),
            0,
            None,
            None,
        )

    perm_np: np.ndarray | None = None
    inv_perm_np: np.ndarray | None = None
    if use_morton:
        box_arg = jnp.asarray(box6_np[:3]) if has_box else None
        perm_j, inv_j = compute_morton_codes_3d(
            jnp.asarray(pos_np), box=box_arg, atom_mask=jnp.asarray(mask_np)
        )
        perm_np = np.asarray(perm_j, dtype=np.int32)
        inv_perm_np = np.asarray(inv_j, dtype=np.int32)
        pos_work = pos_np[perm_np]
        mask_work = mask_np[perm_np]
        n_act = int(np.sum(mask_work > 0))
    else:
        pos_work = pos_np
        mask_work = mask_np

    n_act_blocks = max(1, (max(1, n_act) + _GB_TILE - 1) // _GB_TILE)

    if neighbor_idx is not None:
        nl = np.asarray(neighbor_idx, dtype=np.int32)
        rows = np.broadcast_to(np.arange(n_atoms, dtype=np.int32)[:, None], nl.shape)
        valid = (nl >= 0) & (nl < n_atoms) & (mask_np[:, None] > 0)
        u = rows[valid]
        v = nl[valid]
        if inv_perm_np is not None:
            u = inv_perm_np[u]
            v = inv_perm_np[v]
        tu = u // _GB_TILE
        tv = v // _GB_TILE
        t_lo = np.minimum(tu, tv)
        t_hi = np.maximum(tu, tv)
        diag = np.arange(n_act_blocks, dtype=np.int32)
        all_ti = np.concatenate([diag, t_lo])
        all_tj = np.concatenate([diag, t_hi])
        uniq = np.unique(np.stack([all_ti, all_tj], axis=-1), axis=0)
        # Re-order diagonal-striped (d = tj - ti, ti) to eliminate L2 tile-0 contention
        order = np.lexsort((uniq[:, 0], uniq[:, 1] - uniq[:, 0]))
        pairs_np = uniq[order].astype(np.int32)
    elif cutoff is not None or born_cutoff is not None:
        c_list = [float(c) for c in (cutoff, born_cutoff) if c is not None]
        eff_cut = max(c_list) + max(0.0, float(skin))
        eff_cut_sq = eff_cut * eff_cut
        lo = np.full((n_act_blocks, 3), 1e30, dtype=np.float64)
        hi = np.full((n_act_blocks, 3), -1e30, dtype=np.float64)
        has_any = np.zeros((n_act_blocks,), dtype=bool)
        for t in range(n_act_blocks):
            s0 = t * _GB_TILE
            s1 = min(n_atoms, s0 + _GB_TILE)
            m_t = mask_work[s0:s1] > 0
            if np.any(m_t):
                pts = pos_work[s0:s1][m_t]
                lo[t] = np.min(pts, axis=0)
                hi[t] = np.max(pts, axis=0)
                has_any[t] = True
        pair_rows: list[tuple[int, int]] = []
        box3 = box6_np[:3]
        ortho_pbc = has_box and np.all(np.abs(box6_np[3:]) < 1e-8) and np.all(box3 > 0.0)
        for d in range(n_act_blocks):
            for ti in range(n_act_blocks - d):
                tj = ti + d
                if not (has_any[ti] and has_any[tj]):
                    continue
                if ti == tj:
                    pair_rows.append((ti, tj))
                    continue
                gap = np.maximum(0.0, np.maximum(lo[ti] - hi[tj], lo[tj] - hi[ti]))
                if ortho_pbc:
                    span = 0.5 * (hi[ti] - lo[ti] + hi[tj] - lo[tj])
                    dc = np.abs(0.5 * (lo[ti] + hi[ti]) - 0.5 * (lo[tj] + hi[tj]))
                    dc = dc - box3 * np.floor(dc / box3 + 0.5)
                    gap = np.maximum(0.0, np.abs(dc) - span)
                if float(np.sum(gap * gap)) <= eff_cut_sq:
                    pair_rows.append((ti, tj))
        pairs_np = (
            np.asarray(pair_rows, dtype=np.int32)
            if pair_rows
            else np.zeros((0, 2), dtype=np.int32)
        )
    else:
        pair_rows = [
            (ti, ti + d)
            for d in range(n_act_blocks)
            for ti in range(n_act_blocks - d)
        ]
        pairs_np = np.asarray(pair_rows, dtype=np.int32)

    m_valid = int(pairs_np.shape[0])
    m_alloc = max(1, m_valid)
    tp_pad = np.full((m_alloc, 2), -1, dtype=np.int32)
    if m_valid > 0:
        tp_pad[:m_valid] = pairs_np

    pair_lookup: dict[tuple[int, int], int] = {
        (int(pairs_np[k, 0]), int(pairs_np[k, 1])): k for k in range(m_valid)
    }
    ef_u32 = np.zeros((m_alloc, _GB_TILE), dtype=np.uint32)
    e14_u32 = np.zeros((m_alloc, _GB_TILE), dtype=np.uint32)

    def _scatter_bits(src_pairs: np.ndarray, dst_u32: np.ndarray) -> None:
        if src_pairs.size == 0 or m_valid == 0:
            return
        u = src_pairs[:, 0]
        v = src_pairs[:, 1]
        if inv_perm_np is not None:
            u = inv_perm_np[u]
            v = inv_perm_np[v]
        i_at = np.minimum(u, v)
        j_at = np.maximum(u, v)
        for ia, ja in zip(i_at.tolist(), j_at.tolist(), strict=True):
            if ia == ja or ia < 0 or ja >= n_atoms:
                continue
            ti, la = divmod(ia, _GB_TILE)
            tj, lb = divmod(ja, _GB_TILE)
            p_idx = pair_lookup.get((ti, tj))
            if p_idx is not None:
                dst_u32[p_idx, la] |= np.uint32(1) << np.uint32(lb)

    _scatter_bits(full_pairs, ef_u32)
    _scatter_bits(p14_pairs, e14_u32)
    return tp_pad, ef_u32.view(np.int32), e14_u32.view(np.int32), m_valid, perm_np, inv_perm_np


def _pack_traced_excl_bits_single(
    n_atoms: int,
    m_alloc: int,
    lut_j: jax.Array,
    idx_12_13: jax.Array | np.ndarray | None,
    idx_14: jax.Array | np.ndarray | None,
    excl_indices: jax.Array | np.ndarray | None,
    excl_scales_vdw: jax.Array | np.ndarray | None,
    excl_scales_elec: jax.Array | np.ndarray | None,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX 32-bit tile-pair exclusion bitmask builder for traced ``jit``/``vmap`` calls."""
    cnt_full = jnp.zeros((m_alloc, _GB_TILE, _GB_TILE), dtype=jnp.int32)
    cnt_14 = jnp.zeros((m_alloc, _GB_TILE, _GB_TILE), dtype=jnp.int32)

    def _accum_pairs(cnt: jax.Array, u: jax.Array, v: jax.Array, cond: jax.Array) -> jax.Array:
        ia = jnp.minimum(u, v)
        ja = jnp.maximum(u, v)
        valid = cond & (u >= 0) & (v >= 0) & (ja < n_atoms) & (ia != ja)
        ia_s = jnp.where(valid, ia, 0)
        ja_s = jnp.where(valid, ja, 0)
        ti, la = ia_s // _GB_TILE, ia_s % _GB_TILE
        tj, lb = ja_s // _GB_TILE, ja_s % _GB_TILE
        p_idx = lut_j[ti, tj]
        ok = valid & (p_idx >= 0)
        return cnt.at[jnp.maximum(p_idx, 0), la, lb].add(ok.astype(jnp.int32))

    if idx_12_13 is not None:
        arr = jnp.asarray(idx_12_13, dtype=jnp.int32).reshape(-1, 2)
        if arr.shape[0] > 0:
            cnt_full = _accum_pairs(cnt_full, arr[:, 0], arr[:, 1], jnp.ones((arr.shape[0],), dtype=jnp.bool_))
    if idx_14 is not None:
        arr = jnp.asarray(idx_14, dtype=jnp.int32).reshape(-1, 2)
        if arr.shape[0] > 0:
            cnt_14 = _accum_pairs(cnt_14, arr[:, 0], arr[:, 1], jnp.ones((arr.shape[0],), dtype=jnp.bool_))
    if excl_indices is not None:
        ei = jnp.asarray(excl_indices, dtype=jnp.int32)
        K = int(ei.shape[-1])
        if K > 0:
            rows = jnp.broadcast_to(jnp.arange(n_atoms, dtype=jnp.int32)[:, None], (n_atoms, K)).reshape(-1)
            cols = ei.reshape(-1)
            base_ok = (cols >= 0) & (cols < n_atoms) & (rows < cols)
            if excl_scales_vdw is not None or excl_scales_elec is not None:
                sv = (
                    jnp.asarray(excl_scales_vdw, dtype=jnp.float32).reshape(-1)
                    if excl_scales_vdw is not None
                    else jnp.zeros_like(rows, dtype=jnp.float32)
                )
                se = (
                    jnp.asarray(excl_scales_elec, dtype=jnp.float32).reshape(-1)
                    if excl_scales_elec is not None
                    else jnp.zeros_like(rows, dtype=jnp.float32)
                )
                is_zero = (jnp.abs(sv) < 1e-7) & (jnp.abs(se) < 1e-7)
                cnt_full = _accum_pairs(cnt_full, rows, cols, base_ok & is_zero)
                cnt_14 = _accum_pairs(cnt_14, rows, cols, base_ok & (~is_zero))
            else:
                cnt_full = _accum_pairs(cnt_full, rows, cols, base_ok)

    shifts = jnp.left_shift(jnp.uint32(1), jnp.arange(_GB_TILE, dtype=jnp.uint32))[None, None, :]
    ef_u32 = jnp.sum(jnp.where(cnt_full > 0, shifts, jnp.uint32(0)), axis=-1, dtype=jnp.uint32)
    e14_u32 = jnp.sum(jnp.where(cnt_14 > 0, shifts, jnp.uint32(0)), axis=-1, dtype=jnp.uint32)
    return (
        jax.lax.bitcast_convert_type(ef_u32, jnp.int32),
        jax.lax.bitcast_convert_type(e14_u32, jnp.int32),
    )


def build_gbsa_tile_pair_list(  # noqa: PLR0913
    positions: jax.Array | np.ndarray,
    *,
    atom_mask: jax.Array | np.ndarray | None = None,
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    n_active_atoms: int | None = None,
    idx_12_13: jax.Array | np.ndarray | None = None,
    idx_14: jax.Array | np.ndarray | None = None,
    excl_indices: jax.Array | np.ndarray | None = None,
    excl_scales_vdw: jax.Array | np.ndarray | None = None,
    excl_scales_elec: jax.Array | np.ndarray | None = None,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    box: jax.Array | np.ndarray | None = None,
    cutoff: float | None = None,
    born_cutoff: float | None = None,
    skin: float = 0.0,
    neighbor_idx: jax.Array | np.ndarray | None = None,
    use_morton: bool | None = None,
    dense_max_atoms: int = 8192,
) -> GBSATilePairList:
    """Build single-system or 1D-ragged multi-replica ``GBSATilePairList`` (MK-67)."""
    pos_shape = tuple(int(s) for s in np.shape(positions))
    is_batched = len(pos_shape) == 3
    B = pos_shape[0] if is_batched else 1
    n_atoms = pos_shape[-2]

    mask_is_traced = isinstance(atom_mask, jax.core.Tracer)
    if atom_mask is not None and not mask_is_traced:
        m_np = np.asarray(atom_mask, dtype=np.int32)
        mask_b = m_np if m_np.ndim == 2 else np.broadcast_to(m_np[None, :], (B, n_atoms))
    else:
        mask_b = np.ones((B, n_atoms), dtype=np.int32)

    if n_real_per_replica is not None and not isinstance(n_real_per_replica, jax.core.Tracer):
        nr_list = [min(max(0, int(x)), n_atoms) for x in np.asarray(n_real_per_replica).reshape(-1)]
    elif n_active_atoms is not None:
        nr_list = [min(max(0, int(n_active_atoms)), n_atoms)] * B
    elif atom_mask is not None and not mask_is_traced:
        nr_list = []
        for b in range(B):
            nz = np.nonzero(mask_b[b] > 0)[0]
            nr_list.append(int(nz[-1]) + 1 if nz.size > 0 else 0)
    else:
        nr_list = [n_atoms] * B

    do_morton = (
        bool(use_morton)
        if use_morton is not None
        else (
            cutoff is not None
            or born_cutoff is not None
            or neighbor_idx is not None
            or max(nr_list, default=0) > int(dense_max_atoms)
        )
    )
    is_all_pairs_nocutoff = (
        not do_morton and cutoff is None and born_cutoff is None and neighbor_idx is None
    )
    excl_is_traced = any(
        isinstance(x, jax.core.Tracer)
        for x in (idx_12_13, idx_14, excl_indices, excl_scales_vdw, excl_scales_elec)
        if x is not None
    )

    if is_all_pairs_nocutoff and excl_is_traced:
        n_act_max = max(1, max(nr_list, default=max(1, n_atoms)))
        n_blk = max(1, (n_act_max + _GB_TILE - 1) // _GB_TILE)
        pair_rows = [(ti, ti + d) for d in range(n_blk) for ti in range(n_blk - d)]
        m_valid = len(pair_rows)
        m_alloc = max(1, m_valid)
        tp_np = np.full((m_alloc, 2), -1, dtype=np.int32)
        lut_np = np.full((n_blk, n_blk), -1, dtype=np.int32)
        for k, (ti, tj) in enumerate(pair_rows):
            tp_np[k, 0] = ti
            tp_np[k, 1] = tj
            lut_np[ti, tj] = k
        lut_j = jnp.asarray(lut_np, dtype=jnp.int32)
        ef_j_list, e14_j_list = [], []
        for b in range(B):
            i12_b = idx_12_13[b] if (idx_12_13 is not None and idx_12_13.ndim == 3) else idx_12_13
            i14_b = idx_14[b] if (idx_14 is not None and idx_14.ndim == 3) else idx_14
            ei_b = excl_indices[b] if (excl_indices is not None and excl_indices.ndim == 3) else excl_indices
            ev_b = (
                excl_scales_vdw[b]
                if (excl_scales_vdw is not None and excl_scales_vdw.ndim == 3)
                else excl_scales_vdw
            )
            ee_b = (
                excl_scales_elec[b]
                if (excl_scales_elec is not None and excl_scales_elec.ndim == 3)
                else excl_scales_elec
            )
            ef_j, e14_j = _pack_traced_excl_bits_single(
                n_atoms, m_alloc, lut_j, i12_b, i14_b, ei_b, ev_b, ee_b
            )
            ef_j_list.append(ef_j)
            e14_j_list.append(e14_j)
        nvp_tr = [m_valid if nr > 0 else 0 for nr in nr_list]
        pair_q = build_ragged_work_queue(nvp_tr, tile_size=1, padded_size=m_alloc)
        atom_q = build_ragged_work_queue(nr_list, tile_size=_GB_TILE, padded_size=n_atoms)
        if not is_batched:
            return GBSATilePairList(
                tile_pairs=jnp.asarray(tp_np),
                excl_full_mask=ef_j_list[0],
                excl_14_mask=e14_j_list[0],
                num_valid_pairs=jnp.asarray(nvp_tr[0], dtype=jnp.int32),
                scale_14_vdw=float(scale_14_vdw),
                scale_14_elec=float(scale_14_elec),
                pair_queue=pair_q,
                atom_queue=atom_q,
                is_batched=False,
            )
        return GBSATilePairList(
            tile_pairs=jnp.broadcast_to(jnp.asarray(tp_np)[None, ...], (B, m_alloc, 2)),
            excl_full_mask=jnp.stack(ef_j_list, axis=0),
            excl_14_mask=jnp.stack(e14_j_list, axis=0),
            num_valid_pairs=jnp.asarray(nvp_tr, dtype=jnp.int32),
            scale_14_vdw=float(scale_14_vdw),
            scale_14_elec=float(scale_14_elec),
            pair_queue=pair_q,
            atom_queue=atom_q,
            is_batched=True,
        )

    if is_all_pairs_nocutoff:
        pos_b = np.zeros((B, n_atoms, 3), dtype=np.float32)
        box6_b = np.zeros((B, 6), dtype=np.float64)
        has_box = box is not None
    else:
        pos_np = np.asarray(positions, dtype=np.float32)
        pos_b = pos_np if is_batched else pos_np[None, ...]
        box6_b, has_box = _normalize_box6_batch_np(box, B)

    tp_list, ef_list, e14_list, nvp_list = [], [], [], []
    perm_list, inv_list = [], []
    eff_s_vdw = float(scale_14_vdw)
    eff_s_elec = float(scale_14_elec)

    for b in range(B):
        i12_b = idx_12_13[b] if (idx_12_13 is not None and idx_12_13.ndim == 3) else idx_12_13
        i14_b = idx_14[b] if (idx_14 is not None and idx_14.ndim == 3) else idx_14
        ei_b = excl_indices[b] if (excl_indices is not None and excl_indices.ndim == 3) else excl_indices
        ev_b = (
            excl_scales_vdw[b]
            if (excl_scales_vdw is not None and excl_scales_vdw.ndim == 3)
            else excl_scales_vdw
        )
        ee_b = (
            excl_scales_elec[b]
            if (excl_scales_elec is not None and excl_scales_elec.ndim == 3)
            else excl_scales_elec
        )
        nl_b = (
            np.asarray(neighbor_idx[b] if neighbor_idx.ndim == 3 else neighbor_idx)
            if neighbor_idx is not None
            else None
        )
        full_p, p14_p, s_v, s_e = _extract_excl_pairs_np(
            n_atoms, i12_b, i14_b, ei_b, ev_b, ee_b, scale_14_vdw, scale_14_elec
        )
        eff_s_vdw, eff_s_elec = s_v, s_e
        tp_b, ef_b, e14_b, mv_b, pm_b, ip_b = _build_single_tile_pairs_np(
            pos_b[b],
            mask_b[b],
            box6_b[b],
            has_box,
            nr_list[b],
            full_p,
            p14_p,
            cutoff,
            born_cutoff,
            skin,
            nl_b,
            do_morton,
        )
        tp_list.append(tp_b)
        ef_list.append(ef_b)
        e14_list.append(e14_b)
        nvp_list.append(mv_b)
        if pm_b is not None and ip_b is not None:
            perm_list.append(pm_b)
            inv_list.append(ip_b)

    m_max = max(1, max((t.shape[0] for t in tp_list), default=1))
    tp_all = np.full((B, m_max, 2), -1, dtype=np.int32)
    ef_all = np.zeros((B, m_max, _GB_TILE), dtype=np.int32)
    e14_all = np.zeros((B, m_max, _GB_TILE), dtype=np.int32)
    for b in range(B):
        mb = tp_list[b].shape[0]
        tp_all[b, :mb] = tp_list[b]
        ef_all[b, :mb] = ef_list[b]
        e14_all[b, :mb] = e14_list[b]

    pair_q = build_ragged_work_queue(nvp_list, tile_size=1, padded_size=m_max)
    atom_q = build_ragged_work_queue(nr_list, tile_size=_GB_TILE, padded_size=n_atoms)

    perm_out = jnp.asarray(np.stack(perm_list, axis=0)) if perm_list else None
    inv_out = jnp.asarray(np.stack(inv_list, axis=0)) if inv_list else None

    if not is_batched:
        return GBSATilePairList(
            tile_pairs=jnp.asarray(tp_all[0]),
            excl_full_mask=jnp.asarray(ef_all[0]),
            excl_14_mask=jnp.asarray(e14_all[0]),
            num_valid_pairs=jnp.asarray(nvp_list[0], dtype=jnp.int32),
            scale_14_vdw=eff_s_vdw,
            scale_14_elec=eff_s_elec,
            atom_perm=perm_out[0] if perm_out is not None else None,
            atom_inv_perm=inv_out[0] if inv_out is not None else None,
            pair_queue=pair_q,
            atom_queue=atom_q,
            is_batched=False,
        )

    return GBSATilePairList(
        tile_pairs=jnp.asarray(tp_all),
        excl_full_mask=jnp.asarray(ef_all),
        excl_14_mask=jnp.asarray(e14_all),
        num_valid_pairs=jnp.asarray(nvp_list, dtype=jnp.int32),
        scale_14_vdw=eff_s_vdw,
        scale_14_elec=eff_s_elec,
        atom_perm=perm_out,
        atom_inv_perm=inv_out,
        pair_queue=pair_q,
        atom_queue=atom_q,
        is_batched=True,
    )


def _perm_1d_or_2d(arr: jax.Array, perm: jax.Array) -> jax.Array:
    """Permute atom axis of ``(B, N)`` or ``(B, N, 3)`` by ``perm`` ``(B, N)`` or ``(N,)``."""
    p_b = perm if perm.ndim == 2 else jnp.broadcast_to(perm[None, :], (arr.shape[0], perm.shape[0]))
    if arr.shape[0] == 1 and p_b.shape[0] > 1:
        arr = jnp.broadcast_to(arr, (p_b.shape[0], *arr.shape[1:]))
    if arr.ndim == 3:
        return jnp.take_along_axis(arr, p_b[:, :, None], axis=1)
    return jnp.take_along_axis(arr, p_b, axis=1)


def _gbsa_tile_reference_single(  # noqa: PLR0913
    pos: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_pairs: jax.Array,
    excl_full_mask: jax.Array,
    excl_14_mask: jax.Array,
    box6: jax.Array,
    lam: jax.Array,
    has_box: bool,
    scale_14_vdw: float,
    scale_14_elec: float,
    solvent_dielectric: float,
    solute_dielectric: float,
    coulomb_dielectric: float,
    dielectric_offset: float,
    probe_radius: float,
    surface_tension: float,
    obc_alpha: float,
    obc_beta: float,
    obc_gamma: float,
    kappa: float,
    kappa_scale: float,
    cutoff: float | None,
    born_cutoff: float | None,
    switch_dist: float | None,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    include_gb: bool,
    include_ace: bool,
    include_coulomb: bool,
    include_lj: bool,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX ``O(M * 32^2)`` tile-pair GBSA energy + Born radii (zero ``(N, N)`` allocation)."""
    dtype = pos.dtype
    N = int(pos.shape[0])
    n_blocks = max(1, (N + _GB_TILE - 1) // _GB_TILE)
    n_pad = n_blocks * _GB_TILE
    pad = n_pad - N

    def _pad1(x, val=0):
        return jnp.pad(x, (0, pad), constant_values=val) if pad > 0 else x

    pos_p = jnp.pad(pos, ((0, pad), (0, 0))) if pad > 0 else pos
    msk_p = _pad1(atom_mask > 0, False)
    q_p = jnp.where(msk_p, _pad1(charges, 0.0), jnp.asarray(0.0, dtype=dtype))
    rad_p = jnp.where(msk_p, _pad1(radii, 1.5), jnp.asarray(1.5, dtype=dtype))
    sr_p = jnp.where(msk_p, _pad1(scaled_radii, 1.2), jnp.asarray(1.2, dtype=dtype))
    sig_p = jnp.where(msk_p, _pad1(sigmas, 1.0), jnp.asarray(1.0, dtype=dtype))
    eps_p = jnp.where(msk_p, _pad1(epsilons, 0.0), jnp.asarray(0.0, dtype=dtype))

    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    four = jnp.asarray(4.0, dtype=dtype)
    eps4 = jnp.asarray(1e-4, dtype=dtype)
    eps12 = jnp.asarray(1e-12, dtype=dtype)

    or_p = jnp.maximum(rad_p - jnp.asarray(dielectric_offset, dtype=dtype), eps4)
    lane = jnp.arange(_GB_TILE, dtype=jnp.int32)
    bit_b = jnp.left_shift(jnp.uint32(1), lane.astype(jnp.uint32))[None, :]
    box_tup = tuple(box6[k] for k in range(6))

    if include_gb:
        def _pass1_scan(i_acc, tp):
            ti, tj = tp[0], tp[1]
            valid_tp = (ti >= 0) & (tj >= 0)
            ti_s = jnp.maximum(ti, 0)
            tj_s = jnp.maximum(tj, 0)
            oi = ti_s * _GB_TILE + lane
            oj = tj_s * _GB_TILE + lane
            pi = pos_p[oi]
            pj = pos_p[oj]
            mi = msk_p[oi] & valid_tp
            mj = msk_p[oj] & valid_tp
            dx, dy, dz = _min_image3(
                (pi[:, 0:1] - pj[None, :, 0], pi[:, 1:2] - pj[None, :, 1], pi[:, 2:3] - pj[None, :, 2]),
                box_tup,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            active = mi[:, None] & mj[None, :] & jnp.where(ti_s == tj_s, lane[:, None] < lane[None, :], True)
            if born_cutoff is not None:
                active = active & (dist < jnp.asarray(float(born_cutoff), dtype=dtype))
            elif cutoff is not None:
                active = active & (dist < jnp.asarray(float(cutoff), dtype=dtype))
            I_ij = jnp.where(active, compute_pair_integral(dist, or_p[oi][:, None], sr_p[oj][None, :]), zero)
            I_ji = jnp.where(active, compute_pair_integral(dist, or_p[oj][None, :], sr_p[oi][:, None]), zero)
            i_acc = i_acc.at[oi].add(jnp.sum(I_ij, axis=1))
            i_acc = i_acc.at[oj].add(jnp.sum(I_ji, axis=0))
            return i_acc, None

        i_sum, _ = jax.lax.scan(_pass1_scan, jnp.zeros((n_pad,), dtype=dtype), tile_pairs)
        r_safe = jnp.maximum(rad_p, eps4)
        psi = i_sum * or_p
        psi2 = psi * psi
        arg = (
            jnp.asarray(obc_alpha, dtype=dtype) * psi
            - jnp.asarray(obc_beta, dtype=dtype) * psi2
            + jnp.asarray(obc_gamma, dtype=dtype) * psi2 * psi
        )
        inv_R = one / or_p - jnp.tanh(arg) / r_safe
        born_p = jnp.where(msk_p, one / jnp.maximum(inv_R, eps4), one)
    else:
        born_p = jnp.ones((n_pad,), dtype=dtype)

    coul_c = jnp.asarray(coulomb_constant, dtype=dtype)
    inv_eps_in = jnp.asarray(1.0 / float(solute_dielectric), dtype=dtype)
    inv_eps_out = jnp.asarray(1.0 / float(solvent_dielectric), dtype=dtype)
    inv_eps_coul = jnp.asarray(1.0 / float(coulomb_dielectric), dtype=dtype)
    kappa_eff = jnp.asarray(float(kappa) * float(kappa_scale), dtype=dtype)

    e_atom = zero
    if include_gb:
        tau_self = inv_eps_in - inv_eps_out * jnp.exp((zero - kappa_eff) * born_p)
        e_self = (zero - half) * coul_c * (q_p * q_p) * tau_self / born_p
        e_atom = e_atom + jnp.sum(jnp.where(msk_p, e_self, zero))
    if include_ace:
        rp = rad_p + jnp.asarray(probe_radius, dtype=dtype)
        e_ace = (
            jnp.asarray(4.0 * math.pi * float(surface_tension), dtype=dtype)
            * (rp * rp)
            * ((rad_p / born_p) ** 6)
        )
        e_atom = e_atom + jnp.sum(jnp.where(msk_p, e_ace, zero))

    soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * (
        jnp.maximum(one - lam, zero) ** int(softcore_exponent)
    )
    s14_v = jnp.asarray(scale_14_vdw, dtype=dtype)
    s14_e = jnp.asarray(scale_14_elec, dtype=dtype)

    def _pass2_scan(e_acc, scan_in):
        tp, m_full, m_14 = scan_in
        ti, tj = tp[0], tp[1]
        valid_tp = (ti >= 0) & (tj >= 0)
        ti_s = jnp.maximum(ti, 0)
        tj_s = jnp.maximum(tj, 0)
        oi = ti_s * _GB_TILE + lane
        oj = tj_s * _GB_TILE + lane
        pi, pj = pos_p[oi], pos_p[oj]
        mi = msk_p[oi] & valid_tp
        mj = msk_p[oj] & valid_tp
        dx, dy, dz = _min_image3(
            (pi[:, 0:1] - pj[None, :, 0], pi[:, 1:2] - pj[None, :, 1], pi[:, 2:3] - pj[None, :, 2]),
            box_tup,
            has_box,
        )
        r_sq = dx * dx + dy * dy + dz * dz + eps12
        dist = jnp.sqrt(r_sq)
        active = mi[:, None] & mj[None, :] & jnp.where(ti_s == tj_s, lane[:, None] < lane[None, :], True)
        qq = q_p[oi][:, None] * q_p[oj][None, :]
        e_pair = zero

        if include_gb:
            gb_act = (
                active & (dist < jnp.asarray(float(born_cutoff), dtype=dtype))
                if born_cutoff is not None
                else (active & (dist < jnp.asarray(float(cutoff), dtype=dtype)) if cutoff is not None else active)
            )
            p_ij = born_p[oi][:, None] * born_p[oj][None, :]
            u_ij = r_sq / (four * p_ij)
            f_gb = jnp.sqrt(r_sq + p_ij * jnp.exp(zero - u_ij))
            tau_ij = inv_eps_in - inv_eps_out * jnp.exp((zero - kappa_eff) * f_gb)
            e_gb = jnp.where(gb_act, (zero - coul_c) * qq * tau_ij / f_gb, zero)
            e_pair = e_pair + jnp.sum(e_gb)

        if include_coulomb or include_lj:
            nb_act = active & (dist < jnp.asarray(float(cutoff), dtype=dtype)) if cutoff is not None else active
            is_full = (m_full.astype(jnp.uint32)[:, None] & bit_b) != jnp.uint32(0)
            is_14 = (m_14.astype(jnp.uint32)[:, None] & bit_b) != jnp.uint32(0)
            if include_coulomb:
                w_e = jnp.where(is_full, zero, jnp.where(is_14, s14_e, one))
                valid_c = nb_act & (dist >= eps4)
                e_c = jnp.where(valid_c, coul_c * inv_eps_coul * qq * w_e / dist, zero)
                e_pair = e_pair + jnp.sum(e_c)
            if include_lj:
                w_v = jnp.where(is_full, zero, jnp.where(is_14, s14_v, one))
                sig_ij = half * (sig_p[oi][:, None] + sig_p[oj][None, :])
                sig_safe = jnp.where(nb_act, jnp.maximum(sig_ij, eps4), one)
                eps_prod = jnp.maximum(eps_p[oi][:, None] * eps_p[oj][None, :], zero)
                eps_ij = jnp.where(
                    eps_prod > zero, jnp.sqrt(jnp.where(eps_prod > zero, eps_prod, one)), zero
                )
                dist_safe = jnp.where(nb_act, dist, jnp.asarray(2.0, dtype=dtype))
                u6 = (dist_safe / sig_safe) ** 6
                x_inv6 = one / jnp.maximum(soft_term + u6, jnp.asarray(1e-6, dtype=dtype))
                sw_w = (
                    _lj_switch_weight_and_deriv(
                        dist, float(cutoff), max(0.0, float(cutoff) - float(switch_dist))
                    )[0]
                    if (cutoff is not None and switch_dist is not None)
                    else one
                )
                e_lj = jnp.where(nb_act, w_v * sw_w * four * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6), zero)
                e_pair = e_pair + jnp.sum(e_lj)

        return e_acc + e_pair, None

    e_pairs, _ = jax.lax.scan(_pass2_scan, zero, (tile_pairs, excl_full_mask, excl_14_mask))
    return e_atom + e_pairs, born_p[:N]


def gbsa_tile_reference(  # noqa: PLR0913
    positions: jax.Array | np.ndarray,
    charges: jax.Array | np.ndarray,
    radii: jax.Array | np.ndarray,
    scaled_radii: jax.Array | np.ndarray,
    sigmas: jax.Array | np.ndarray,
    epsilons: jax.Array | np.ndarray,
    tile_list: GBSATilePairList,
    *,
    atom_mask: jax.Array | np.ndarray | None = None,
    box: jax.Array | np.ndarray | None = None,
    softcore_lambda: float | jax.Array | np.ndarray = 1.0,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    coulomb_dielectric: float | None = None,
    dielectric_offset: float = 0.09,
    probe_radius: float = constants.PROBE_RADIUS,
    surface_tension: float = SURFACE_TENSION,
    obc_alpha: float = ALPHA_OBC,
    obc_beta: float = BETA_OBC,
    obc_gamma: float = GAMMA_OBC,
    kappa: float = 0.0,
    kappa_scale: float = GB_KAPPA_SCALE,
    cutoff: float | None = None,
    born_cutoff: float | None = None,
    switch_dist: float | None = None,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 1,
    coulomb_constant: float = GB_COULOMB_CONSTANT,
    include_gb: bool = True,
    include_ace: bool = True,
    include_coulomb: bool = True,
    include_lj: bool = True,
    return_forces: bool = False,
) -> Any:
    """Pure-JAX tile-pair GBSA reference for ``in_export_context()`` and ``float64`` checks."""
    pos_arr = jnp.asarray(positions)
    single_in = pos_arr.ndim == 2
    pos_b = pos_arr[None, ...] if single_in else pos_arr
    B, N, _ = pos_b.shape
    dtype = pos_b.dtype
    c_diel = float(solute_dielectric if coulomb_dielectric is None else coulomb_dielectric)

    def _bcast1(x):
        a = jnp.asarray(x, dtype=dtype)
        return jnp.broadcast_to(a[None, :] if a.ndim == 1 else a, (B, N))

    q_b = _bcast1(charges)
    rad_b = _bcast1(radii)
    sr_b = _bcast1(scaled_radii)
    sig_b = _bcast1(sigmas)
    eps_b = _bcast1(epsilons)
    msk_b = (
        jnp.ones((B, N), dtype=jnp.int32)
        if atom_mask is None
        else jnp.broadcast_to(
            jnp.asarray(atom_mask, dtype=jnp.int32)[None, :]
            if jnp.asarray(atom_mask).ndim == 1
            else jnp.asarray(atom_mask, dtype=jnp.int32),
            (B, N),
        )
    )
    box6_b, has_box = _normalize_box6_batch(box, B, dtype)
    lam_b = jnp.broadcast_to(jnp.asarray(softcore_lambda, dtype=dtype).reshape(-1), (B,))

    tp_b = tile_list.tile_pairs if tile_list.tile_pairs.ndim == 3 else jnp.broadcast_to(
        tile_list.tile_pairs[None, ...], (B, *tile_list.tile_pairs.shape)
    )
    ef_b = tile_list.excl_full_mask if tile_list.excl_full_mask.ndim == 3 else jnp.broadcast_to(
        tile_list.excl_full_mask[None, ...], (B, *tile_list.excl_full_mask.shape)
    )
    e14_b = tile_list.excl_14_mask if tile_list.excl_14_mask.ndim == 3 else jnp.broadcast_to(
        tile_list.excl_14_mask[None, ...], (B, *tile_list.excl_14_mask.shape)
    )

    def _eval_one(p_i, q_i, r_i, s_i, sg_i, ep_i, m_i, tp_i, ef_i, e14_i, bx_i, lm_i, perm_i):
        if perm_i is not None:
            p_w = p_i[perm_i]
            q_w, r_w, s_w = q_i[perm_i], r_i[perm_i], s_i[perm_i]
            sg_w, ep_w, m_w = sg_i[perm_i], ep_i[perm_i], m_i[perm_i]
        else:
            p_w, q_w, r_w, s_w, sg_w, ep_w, m_w = p_i, q_i, r_i, s_i, sg_i, ep_i, m_i
        return _gbsa_tile_reference_single(
            p_w, q_w, r_w, s_w, sg_w, ep_w, m_w, tp_i, ef_i, e14_i, bx_i, lm_i,
            has_box, tile_list.scale_14_vdw, tile_list.scale_14_elec,
            solvent_dielectric, solute_dielectric, c_diel, dielectric_offset,
            probe_radius, surface_tension, obc_alpha, obc_beta, obc_gamma,
            kappa, kappa_scale, cutoff, born_cutoff, switch_dist,
            softcore_alpha, softcore_exponent, coulomb_constant,
            include_gb, include_ace, include_coulomb, include_lj,
        )

    perm_b = (
        None
        if tile_list.atom_perm is None
        else (
            tile_list.atom_perm
            if tile_list.atom_perm.ndim == 2
            else jnp.broadcast_to(tile_list.atom_perm[None, :], (B, N))
        )
    )
    inv_b = (
        None
        if tile_list.atom_inv_perm is None
        else (
            tile_list.atom_inv_perm
            if tile_list.atom_inv_perm.ndim == 2
            else jnp.broadcast_to(tile_list.atom_inv_perm[None, :], (B, N))
        )
    )

    def _run_batch(pos_in):
        if perm_b is None:
            e_out, br_out = jax.vmap(
                lambda p, q, r, s, sg, ep, m, tp, ef, e14, bx, lm: _eval_one(
                    p, q, r, s, sg, ep, m, tp, ef, e14, bx, lm, None
                )
            )(pos_in, q_b, rad_b, sr_b, sig_b, eps_b, msk_b, tp_b, ef_b, e14_b, box6_b, lam_b)
        else:
            e_out, br_s = jax.vmap(_eval_one)(
                pos_in, q_b, rad_b, sr_b, sig_b, eps_b, msk_b, tp_b, ef_b, e14_b, box6_b, lam_b, perm_b
            )
            br_out = _perm_1d_or_2d(br_s, inv_b) if inv_b is not None else br_s
        return e_out, br_out

    if not return_forces:
        e_b, _ = _run_batch(pos_b)
        return e_b[0] if single_in else e_b

    (_, (e_b, br_b)), grad_pos = jax.value_and_grad(
        lambda p: (jnp.sum(_run_batch(p)[0]), _run_batch(p)), has_aux=True
    )(pos_b)
    forces_b = -grad_pos
    if single_in:
        return e_b[0], forces_b[0], br_b[0]
    return e_b, forces_b, br_b


def _run_gbsa_tile_pallas_fwd(  # noqa: PLR0913
    pos_b: jax.Array,
    q_b: jax.Array,
    rad_b: jax.Array,
    sr_b: jax.Array,
    sig_b: jax.Array,
    eps_b: jax.Array,
    msk_b: jax.Array,
    tile_list: GBSATilePairList,
    box6_b: jax.Array,
    lam_b: jax.Array,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    coulomb_dielectric: float,
    dielectric_offset: float,
    probe_radius: float,
    surface_tension: float,
    obc_alpha: float,
    obc_beta: float,
    obc_gamma: float,
    kappa: float,
    kappa_scale: float,
    cutoff: float | None,
    born_cutoff: float | None,
    switch_dist: float | None,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    include_gb: bool,
    include_ace: bool,
    include_coulomb: bool,
    include_lj: bool,
    compute_forces: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Core 1D-ragged 32x32 symmetric tile-pair Pallas forward pass (MK-67)."""
    B, N, _ = pos_b.shape
    orig_dtype = pos_b.dtype
    dtype = jnp.dtype(jnp.float32) if (not interpret and orig_dtype == jnp.float64) else orig_dtype

    if N == 0 or B == 0:
        zf = jnp.zeros((B, N, 3), dtype=orig_dtype)
        zv = jnp.zeros((B, N), dtype=orig_dtype)
        return jnp.zeros((B,), dtype=orig_dtype), zf, zv, zv, zv, zv, zv

    perm = tile_list.atom_perm
    inv_perm = tile_list.atom_inv_perm
    if perm is not None:
        pos_w = _perm_1d_or_2d(pos_b, perm)
        q_w = _perm_1d_or_2d(q_b, perm)
        rad_w = _perm_1d_or_2d(rad_b, perm)
        sr_w = _perm_1d_or_2d(sr_b, perm)
        sig_w = _perm_1d_or_2d(sig_b, perm)
        eps_w = _perm_1d_or_2d(eps_b, perm)
        msk_w = _perm_1d_or_2d(msk_b, perm)
    else:
        pos_w, q_w, rad_w, sr_w, sig_w, eps_w, msk_w = pos_b, q_b, rad_b, sr_b, sig_b, eps_b, msk_b

    tp_b = (
        tile_list.tile_pairs
        if tile_list.tile_pairs.ndim == 3
        else jnp.broadcast_to(tile_list.tile_pairs[None, ...], (B, *tile_list.tile_pairs.shape))
    )
    ef_b = (
        tile_list.excl_full_mask
        if tile_list.excl_full_mask.ndim == 3
        else jnp.broadcast_to(tile_list.excl_full_mask[None, ...], (B, *tile_list.excl_full_mask.shape))
    )
    e14_b = (
        tile_list.excl_14_mask
        if tile_list.excl_14_mask.ndim == 3
        else jnp.broadcast_to(tile_list.excl_14_mask[None, ...], (B, *tile_list.excl_14_mask.shape))
    )
    M = int(tp_b.shape[1])

    pair_q = tile_list.pair_queue
    if pair_q is None or pair_q.batch_size != B:
        nvp = (
            np.broadcast_to(np.asarray(tile_list.num_valid_pairs).reshape(-1), (B,))
            if not isinstance(tile_list.num_valid_pairs, jax.core.Tracer)
            else [M] * B
        )
        pair_q = build_ragged_work_queue(nvp, tile_size=1, padded_size=M)

    atom_q = tile_list.atom_queue
    if atom_q is None or atom_q.batch_size != B:
        atom_q = build_ragged_work_queue([N] * B, tile_size=_GB_TILE, padded_size=N)

    n_blocks_pad = max(1, (N + _GB_TILE - 1) // _GB_TILE)
    n_stride = n_blocks_pad * _GB_TILE if interpret else N
    pad_n = n_stride - N

    def _to_flat(arr, pad_val=0.0):
        a = arr.astype(dtype) if arr.dtype != jnp.int32 else arr
        if pad_n > 0:
            a = jnp.pad(a, ((0, 0), (0, pad_n)), constant_values=pad_val)
        return jnp.asarray(a).reshape(-1)

    pos_f = pos_w.astype(dtype)
    if pad_n > 0:
        pos_f = jnp.pad(pos_f, ((0, 0), (0, pad_n), (0, 0)))
    px_flat = jnp.asarray(pos_f[:, :, 0]).reshape(-1)
    py_flat = jnp.asarray(pos_f[:, :, 1]).reshape(-1)
    pz_flat = jnp.asarray(pos_f[:, :, 2]).reshape(-1)

    B_q, B_rad, B_sr = int(q_w.shape[0]), int(rad_w.shape[0]), int(sr_w.shape[0])
    B_sig, B_eps, B_msk = int(sig_w.shape[0]), int(eps_w.shape[0]), int(msk_w.shape[0])
    B_box, B_lam = int(box6_b.shape[0]), int(lam_b.shape[0])

    q_flat = _to_flat(q_w, 0.0)
    rad_flat = _to_flat(rad_w, 1.5)
    sr_flat = _to_flat(sr_w, 1.2)
    sig_flat = _to_flat(sig_w, 1.0)
    eps_flat = _to_flat(eps_w, 0.0)
    msk_flat = _to_flat(msk_w.astype(jnp.int32), 0)
    box_flat = box6_b.astype(dtype).reshape(-1)
    lam_flat = lam_b.astype(dtype).reshape(-1)
    tp_flat = tp_b.astype(jnp.int32).reshape(-1)
    ef_flat = ef_b.astype(jnp.int32).reshape(-1)
    e14_flat = e14_b.astype(jnp.int32).reshape(-1)

    cparams = resolve_pallas_compiler_params("megakernel")
    born_cut = born_cutoff if born_cutoff is not None else cutoff
    has_born_cut = born_cut is not None
    born_cut_val = float(born_cut) if born_cut is not None else 0.0
    has_nb_cut = cutoff is not None
    has_switch = cutoff is not None and switch_dist is not None

    # --- Pass 1a: 32x32 Symmetric Tile-Pair Born Integral Accumulation ---
    i_sum_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    if include_gb and pair_q.total_blocks > 0:
        def _pass1a_kernel(
            px_r, py_r, pz_r, rad_r, sr_r, msk_r, box_r, tp_r, cta_rep_r, cta_blk_r, i_in_r, i_out_r
        ):
            del i_in_r
            cta = pl.program_id(0)
            b_id = _pl_load(cta_rep_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            p_local = _pl_load(cta_blk_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            tp_base = (b_id * M + p_local) * 2
            ti = _pl_load(tp_r, (jnp.full((1,), tp_base, dtype=jnp.int32),))[0]
            tj = _pl_load(tp_r, (jnp.full((1,), tp_base + 1, dtype=jnp.int32),))[0]
            valid_tp = (ti >= 0) & (tj >= 0)
            ti_s, tj_s = jnp.maximum(ti, 0), jnp.maximum(tj, 0)

            lane = jnp.arange(_GB_TILE, dtype=jnp.int32)
            oi = ti_s * _GB_TILE + lane
            oj = tj_s * _GB_TILE + lane
            bi_ok = (oi < N) & valid_tp
            bj_ok = (oj < N) & valid_tp

            b_pos = b_id * n_stride
            b_rad = (b_id if B_rad > 1 else jnp.int32(0)) * n_stride
            b_sr = (b_id if B_sr > 1 else jnp.int32(0)) * n_stride
            b_msk = (b_id if B_msk > 1 else jnp.int32(0)) * n_stride
            b_box = (b_id if B_box > 1 else jnp.int32(0)) * 6

            zero = jnp.asarray(0.0, dtype=dtype)
            eps4 = jnp.asarray(1e-4, dtype=dtype)
            eps12 = jnp.asarray(1e-12, dtype=dtype)
            d_off = jnp.asarray(float(dielectric_offset), dtype=dtype)

            mi = (_pl_load(msk_r, (b_msk + oi,), mask=bi_ok, other=0) > 0) & bi_ok
            mj = (_pl_load(msk_r, (b_msk + oj,), mask=bj_ok, other=0) > 0) & bj_ok
            pxi = _pl_load(px_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pyi = _pl_load(py_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pzi = _pl_load(pz_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pxj = _pl_load(px_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pyj = _pl_load(py_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pzj = _pl_load(pz_r, (b_pos + oj,), mask=bj_ok, other=zero)
            safe_rad = jnp.asarray(1.5, dtype=dtype)
            safe_sr = jnp.asarray(1.2, dtype=dtype)
            ri_raw = jnp.where(mi, _pl_load(rad_r, (b_rad + oi,), mask=bi_ok, other=safe_rad), safe_rad)
            rj_raw = jnp.where(mj, _pl_load(rad_r, (b_rad + oj,), mask=bj_ok, other=safe_rad), safe_rad)
            ori = jnp.maximum(ri_raw - d_off, eps4)
            orj = jnp.maximum(rj_raw - d_off, eps4)
            sri = jnp.where(mi, _pl_load(sr_r, (b_sr + oi,), mask=bi_ok, other=safe_sr), safe_sr)
            srj = jnp.where(mj, _pl_load(sr_r, (b_sr + oj,), mask=bj_ok, other=safe_sr), safe_sr)

            box6 = tuple(
                _pl_load(box_r, (jnp.full((1,), b_box + k, dtype=jnp.int32),))[0] for k in range(6)
            )
            dx, dy, dz = _min_image3(
                (pxi[:, None] - pxj[None, :], pyi[:, None] - pyj[None, :], pzi[:, None] - pzj[None, :]),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            active = mi[:, None] & mj[None, :] & jnp.where(ti_s == tj_s, lane[:, None] < lane[None, :], True)
            if has_born_cut:
                active = active & (dist < jnp.asarray(born_cut_val, dtype=dtype))

            I_ij = jnp.where(active, compute_pair_integral(dist, ori[:, None], srj[None, :]), zero)
            I_ji = jnp.where(active, compute_pair_integral(dist, orj[None, :], sri[:, None]), zero)
            acc_i = jnp.sum(I_ij, axis=1)
            acc_j = jnp.sum(I_ji, axis=0)
            is_diag = ti_s == tj_s
            _pl_atomic_add_1d(
                i_out_r, b_pos + oi, jnp.where(is_diag, acc_i + acc_j, acc_i), mi, interpret
            )
            _pl_atomic_add_1d(
                i_out_r, b_pos + oj, acc_j, mj & (~is_diag), interpret
            )

        any_spec = _any_block_spec()
        i_sum_flat = pl.pallas_call(
            _pass1a_kernel,
            out_shape=jax.ShapeDtypeStruct(i_sum_flat.shape, dtype),
            in_specs=[any_spec] * 11,
            out_specs=any_spec,
            grid=(pair_q.total_blocks,),
            input_output_aliases={10: 0},
            interpret=interpret,
            compiler_params=cparams if not interpret else None,
        )(
            px_flat, py_flat, pz_flat, rad_flat, sr_flat, msk_flat, box_flat, tp_flat,
            pair_q.cta_to_replica, pair_q.cta_to_local_block, i_sum_flat,
        )

    # --- Pass 1b: Fused 1D OBC2 Born Radii + Screened Self-GB + Exact AKMA ACE ---
    born_flat = jnp.ones((B * n_stride,), dtype=dtype)
    db_di_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    dedb_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    gq_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    e_out = jnp.zeros((B,), dtype=dtype)

    if (include_gb or include_ace) and atom_q.total_blocks > 0:
        def _pass1b_kernel(  # noqa: PLR0913
            i_sum_r, q_r, rad_r, msk_r, cta_rep_r, cta_blk_r,
            b_in_r, db_in_r, dedb_in_r, gq_in_r, e_in_r,
            b_out_r, db_out_r, dedb_out_r, gq_out_r, e_out_r,
        ):
            del b_in_r, db_in_r, dedb_in_r, gq_in_r, e_in_r
            cta = pl.program_id(0)
            b_id = _pl_load(cta_rep_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            blk = _pl_load(cta_blk_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            offs = blk * _GB_TILE + jnp.arange(_GB_TILE, dtype=jnp.int32)
            in_b = offs < N

            b_out = b_id * n_stride
            b_q = (b_id if B_q > 1 else jnp.int32(0)) * n_stride
            b_rad = (b_id if B_rad > 1 else jnp.int32(0)) * n_stride
            b_msk = (b_id if B_msk > 1 else jnp.int32(0)) * n_stride

            zero = jnp.asarray(0.0, dtype=dtype)
            one = jnp.asarray(1.0, dtype=dtype)
            half = jnp.asarray(0.5, dtype=dtype)
            eps4 = jnp.asarray(1e-4, dtype=dtype)

            mi = (_pl_load(msk_r, (b_msk + offs,), mask=in_b, other=0) > 0) & in_b
            ri = jnp.where(mi, _pl_load(rad_r, (b_rad + offs,), mask=in_b, other=one), jnp.asarray(1.5, dtype=dtype))
            qi = jnp.where(mi, _pl_load(q_r, (b_q + offs,), mask=in_b, other=zero), zero)
            ii = _pl_load(i_sum_r, (b_out + offs,), mask=in_b, other=zero)

            if include_gb:
                ori = jnp.maximum(ri - jnp.asarray(float(dielectric_offset), dtype=dtype), eps4)
                r_safe = jnp.maximum(ri, eps4)
                psi = ii * ori
                psi2 = psi * psi
                a_c = jnp.asarray(float(obc_alpha), dtype=dtype)
                b_c = jnp.asarray(float(obc_beta), dtype=dtype)
                g_c = jnp.asarray(float(obc_gamma), dtype=dtype)
                arg = a_c * psi - b_c * psi2 + g_c * psi2 * psi
                th = jnp.tanh(arg)
                inv_R = one / ori - th / r_safe
                R_i = jnp.where(mi, one / jnp.maximum(inv_R, eps4), one)
                darg = a_c - jnp.asarray(2.0, dtype=dtype) * b_c * psi + jnp.asarray(3.0, dtype=dtype) * g_c * psi2
                db_di = jnp.where(mi, R_i * R_i * (one - th * th) * darg * (ori / r_safe), zero)
            else:
                R_i = one
                db_di = zero

            coul_c = jnp.asarray(float(coulomb_constant), dtype=dtype)
            inv_ein = jnp.asarray(1.0 / float(solute_dielectric), dtype=dtype)
            inv_eout = jnp.asarray(1.0 / float(solvent_dielectric), dtype=dtype)
            keff = jnp.asarray(float(kappa) * float(kappa_scale), dtype=dtype)

            e_atom = zero
            dedb_atom = zero
            gq_atom = zero
            if include_gb:
                exp_s = jnp.exp((zero - keff) * R_i)
                tau_s = inv_ein - inv_eout * exp_s
                e_self = (zero - half) * coul_c * qi * qi * tau_s / R_i
                dedb_s = (
                    half
                    * coul_c
                    * qi
                    * qi
                    * (inv_ein - inv_eout * (one + keff * R_i) * exp_s)
                    / (R_i * R_i)
                )
                gq_s = (zero - coul_c) * qi * tau_s / R_i
                e_atom = e_atom + jnp.where(mi, e_self, zero)
                dedb_atom = dedb_atom + jnp.where(mi, dedb_s, zero)
                gq_atom = jnp.where(mi, gq_s, zero)

            if include_ace:
                rp = ri + jnp.asarray(float(probe_radius), dtype=dtype)
                _rr = ri / R_i
                _rr2 = _rr * _rr
                e_ace = (
                    jnp.asarray(4.0 * math.pi * float(surface_tension), dtype=dtype)
                    * (rp * rp)
                    * (_rr2 * _rr2 * _rr2)
                )
                dedb_ace = jnp.asarray(-6.0, dtype=dtype) * e_ace / R_i
                e_atom = e_atom + jnp.where(mi, e_ace, zero)
                dedb_atom = dedb_atom + jnp.where(mi, dedb_ace, zero)

            _pl_store(b_out_r, (b_out + offs,), R_i, mask=in_b)
            _pl_store(db_out_r, (b_out + offs,), db_di, mask=in_b)
            _pl_store(dedb_out_r, (b_out + offs,), dedb_atom, mask=in_b)
            _pl_store(gq_out_r, (b_out + offs,), gq_atom, mask=in_b)
            e_blk = jnp.full((1,), jnp.sum(e_atom), dtype=dtype)
            _pl_atomic_add_1d(
                e_out_r, jnp.full((1,), b_id, dtype=jnp.int32), e_blk, jnp.ones((1,), dtype=jnp.bool_), interpret
            )

        any_spec = _any_block_spec()
        born_flat, db_di_flat, dedb_flat, gq_flat, e_out = pl.pallas_call(
            _pass1b_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(born_flat.shape, dtype),
                jax.ShapeDtypeStruct(db_di_flat.shape, dtype),
                jax.ShapeDtypeStruct(dedb_flat.shape, dtype),
                jax.ShapeDtypeStruct(gq_flat.shape, dtype),
                jax.ShapeDtypeStruct(e_out.shape, dtype),
            ),
            in_specs=[any_spec] * 11,
            out_specs=(any_spec,) * 5,
            grid=(atom_q.total_blocks,),
            input_output_aliases={6: 0, 7: 1, 8: 2, 9: 3, 10: 4},
            interpret=interpret,
            compiler_params=cparams if not interpret else None,
        )(
            i_sum_flat, q_flat, rad_flat, msk_flat,
            atom_q.cta_to_replica, atom_q.cta_to_local_block,
            born_flat, db_di_flat, dedb_flat, gq_flat, e_out,
        )

    return _run_gbsa_tile_pass2(
        B=B, N=N, M=M, n_stride=n_stride, orig_dtype=orig_dtype, dtype=dtype,
        B_q=B_q, B_rad=B_rad, B_sr=B_sr, B_sig=B_sig, B_eps=B_eps, B_msk=B_msk,
        B_box=B_box, B_lam=B_lam,
        px_flat=px_flat, py_flat=py_flat, pz_flat=pz_flat, q_flat=q_flat,
        rad_flat=rad_flat, sr_flat=sr_flat, sig_flat=sig_flat, eps_flat=eps_flat,
        msk_flat=msk_flat, box_flat=box_flat, lam_flat=lam_flat,
        tp_flat=tp_flat, ef_flat=ef_flat, e14_flat=e14_flat,
        born_flat=born_flat, i_sum_flat=i_sum_flat, db_di_flat=db_di_flat,
        dedb_flat=dedb_flat, gq_flat=gq_flat, e_out=e_out,
        pair_q=pair_q, inv_perm=inv_perm,
        scale_14_vdw=tile_list.scale_14_vdw, scale_14_elec=tile_list.scale_14_elec,
        has_box=has_box, solvent_dielectric=solvent_dielectric,
        solute_dielectric=solute_dielectric, coulomb_dielectric=coulomb_dielectric,
        dielectric_offset=dielectric_offset, kappa=kappa, kappa_scale=kappa_scale,
        cutoff=cutoff, born_cut=born_cut, switch_dist=switch_dist,
        has_born_cut=has_born_cut, has_nb_cut=has_nb_cut, has_switch=has_switch,
        softcore_alpha=softcore_alpha, softcore_exponent=softcore_exponent,
        coulomb_constant=coulomb_constant, include_gb=include_gb,
        include_coulomb=include_coulomb, include_lj=include_lj,
        compute_forces=compute_forces, interpret=interpret, cparams=cparams,
    )


def _run_gbsa_tile_pass2(  # noqa: PLR0913
    *,
    B: int,
    N: int,
    M: int,
    n_stride: int,
    orig_dtype: jnp.dtype,
    dtype: jnp.dtype,
    B_q: int,
    B_rad: int,
    B_sr: int,
    B_sig: int,
    B_eps: int,
    B_msk: int,
    B_box: int,
    B_lam: int,
    px_flat: jax.Array,
    py_flat: jax.Array,
    pz_flat: jax.Array,
    q_flat: jax.Array,
    rad_flat: jax.Array,
    sr_flat: jax.Array,
    sig_flat: jax.Array,
    eps_flat: jax.Array,
    msk_flat: jax.Array,
    box_flat: jax.Array,
    lam_flat: jax.Array,
    tp_flat: jax.Array,
    ef_flat: jax.Array,
    e14_flat: jax.Array,
    born_flat: jax.Array,
    i_sum_flat: jax.Array,
    db_di_flat: jax.Array,
    dedb_flat: jax.Array,
    gq_flat: jax.Array,
    e_out: jax.Array,
    pair_q: RaggedWorkQueue,
    inv_perm: jax.Array | None,
    scale_14_vdw: float,
    scale_14_elec: float,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    coulomb_dielectric: float,
    dielectric_offset: float,
    kappa: float,
    kappa_scale: float,
    cutoff: float | None,
    born_cut: float | None,
    switch_dist: float | None,
    has_born_cut: bool,
    has_nb_cut: bool,
    has_switch: bool,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    include_gb: bool,
    include_coulomb: bool,
    include_lj: bool,
    compute_forces: bool,
    interpret: bool,
    cparams: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pass 2a (symmetric tile-pair GB + Coulomb + LJ) & Pass 2b (Born chain-rule forces)."""
    fx_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    fy_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    fz_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    born_cut_val = float(born_cut) if born_cut is not None else 0.0
    nb_cut_val = float(cutoff) if cutoff is not None else 0.0

    if (include_gb or include_coulomb or include_lj) and pair_q.total_blocks > 0:
        def _pass2a_kernel(  # noqa: PLR0913
            px_r, py_r, pz_r, q_r, sig_r, eps_r, born_r, msk_r, box_r, lam_r,
            tp_r, ef_r, e14_r, cta_rep_r, cta_blk_r,
            fx_in_r, fy_in_r, fz_in_r, dedb_in_r, gq_in_r, e_in_r,
            fx_out_r, fy_out_r, fz_out_r, dedb_out_r, gq_out_r, e_out_r,
        ):
            del fx_in_r, fy_in_r, fz_in_r, dedb_in_r, gq_in_r, e_in_r
            cta = pl.program_id(0)
            b_id = _pl_load(cta_rep_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            p_local = _pl_load(cta_blk_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            pair_abs = b_id * M + p_local
            tp_base = pair_abs * 2
            ti = _pl_load(tp_r, (jnp.full((1,), tp_base, dtype=jnp.int32),))[0]
            tj = _pl_load(tp_r, (jnp.full((1,), tp_base + 1, dtype=jnp.int32),))[0]
            valid_tp = (ti >= 0) & (tj >= 0)
            ti_s, tj_s = jnp.maximum(ti, 0), jnp.maximum(tj, 0)

            lane = jnp.arange(_GB_TILE, dtype=jnp.int32)
            oi = ti_s * _GB_TILE + lane
            oj = tj_s * _GB_TILE + lane
            bi_ok = (oi < N) & valid_tp
            bj_ok = (oj < N) & valid_tp
            is_diag = ti_s == tj_s

            b_pos = b_id * n_stride
            b_q = (b_id if B_q > 1 else jnp.int32(0)) * n_stride
            b_sig = (b_id if B_sig > 1 else jnp.int32(0)) * n_stride
            b_eps = (b_id if B_eps > 1 else jnp.int32(0)) * n_stride
            b_msk = (b_id if B_msk > 1 else jnp.int32(0)) * n_stride
            b_box = (b_id if B_box > 1 else jnp.int32(0)) * 6
            b_lam = b_id if B_lam > 1 else jnp.int32(0)

            zero = jnp.asarray(0.0, dtype=dtype)
            one = jnp.asarray(1.0, dtype=dtype)
            two = jnp.asarray(2.0, dtype=dtype)
            four = jnp.asarray(4.0, dtype=dtype)
            half = jnp.asarray(0.5, dtype=dtype)
            quarter = jnp.asarray(0.25, dtype=dtype)
            eps4 = jnp.asarray(1e-4, dtype=dtype)
            eps12 = jnp.asarray(1e-12, dtype=dtype)

            mi = (_pl_load(msk_r, (b_msk + oi,), mask=bi_ok, other=0) > 0) & bi_ok
            mj = (_pl_load(msk_r, (b_msk + oj,), mask=bj_ok, other=0) > 0) & bj_ok
            pxi = _pl_load(px_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pyi = _pl_load(py_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pzi = _pl_load(pz_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pxj = _pl_load(px_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pyj = _pl_load(py_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pzj = _pl_load(pz_r, (b_pos + oj,), mask=bj_ok, other=zero)
            qi = _pl_load(q_r, (b_q + oi,), mask=bi_ok, other=zero)
            qj = _pl_load(q_r, (b_q + oj,), mask=bj_ok, other=zero)

            box6 = tuple(
                _pl_load(box_r, (jnp.full((1,), b_box + k, dtype=jnp.int32),))[0] for k in range(6)
            )
            dx, dy, dz = _min_image3(
                (pxi[:, None] - pxj[None, :], pyi[:, None] - pyj[None, :], pzi[:, None] - pzj[None, :]),
                box6,
                has_box,
            )
            r_sq = dx * dx + dy * dy + dz * dz + eps12
            dist = jnp.sqrt(r_sq)
            active = mi[:, None] & mj[None, :] & jnp.where(is_diag, lane[:, None] < lane[None, :], True)
            qq = qi[:, None] * qj[None, :]

            coul_c = jnp.asarray(float(coulomb_constant), dtype=dtype)
            e_tot = zero
            f_tot = zero
            gq_mat = zero

            if include_gb:
                Ri = _pl_load(born_r, (b_pos + oi,), mask=bi_ok, other=one)
                Rj = _pl_load(born_r, (b_pos + oj,), mask=bj_ok, other=one)
                gb_act = (
                    active & (dist < jnp.asarray(born_cut_val, dtype=dtype)) if has_born_cut else active
                )
                p_ij = Ri[:, None] * Rj[None, :]
                u_ij = r_sq / (four * p_ij)
                e_ij = jnp.exp(zero - u_ij)
                f_gb_sq = r_sq + p_ij * e_ij
                f_gb = jnp.sqrt(f_gb_sq)
                keff = jnp.asarray(float(kappa) * float(kappa_scale), dtype=dtype)
                inv_ein = jnp.asarray(1.0 / float(solute_dielectric), dtype=dtype)
                inv_eout = jnp.asarray(1.0 / float(solvent_dielectric), dtype=dtype)
                exp_kf = jnp.exp((zero - keff) * f_gb)
                h_f = (inv_ein - inv_eout * exp_kf) / f_gb
                gf_f = (inv_ein - inv_eout * (one + keff * f_gb) * exp_kf) / f_gb_sq

                e_gb = jnp.where(gb_act, (zero - coul_c) * qq * h_f, zero)
                f_gb_sc = jnp.where(
                    gb_act, (zero - coul_c) * qq * gf_f * (one - quarter * e_ij) / f_gb, zero
                )
                dedb_c = jnp.where(
                    gb_act, half * coul_c * qq * gf_f * e_ij * (one + u_ij) / f_gb, zero
                )
                gq_gb = jnp.where(gb_act, (zero - coul_c) * h_f, zero)

                dedb_i = jnp.sum(dedb_c * Rj[None, :], axis=1)
                dedb_j = jnp.sum(dedb_c * Ri[:, None], axis=0)
                _pl_atomic_add_1d(
                    dedb_out_r, b_pos + oi, jnp.where(is_diag, dedb_i + dedb_j, dedb_i), mi, interpret
                )
                _pl_atomic_add_1d(dedb_out_r, b_pos + oj, dedb_j, mj & (~is_diag), interpret)
                e_tot = e_tot + e_gb
                f_tot = f_tot + f_gb_sc
                gq_mat = gq_mat + gq_gb

            if include_coulomb or include_lj:
                nb_act = active & (dist < jnp.asarray(nb_cut_val, dtype=dtype)) if has_nb_cut else active
                m_base = pair_abs * _GB_TILE + lane
                m_full = _pl_load(ef_r, (m_base,), mask=valid_tp, other=0)
                m_14 = _pl_load(e14_r, (m_base,), mask=valid_tp, other=0)
                bit_b = jnp.left_shift(jnp.uint32(1), lane.astype(jnp.uint32))[None, :]
                is_full = (m_full.astype(jnp.uint32)[:, None] & bit_b) != jnp.uint32(0)
                is_14 = (m_14.astype(jnp.uint32)[:, None] & bit_b) != jnp.uint32(0)

                if include_coulomb:
                    s14_e = jnp.asarray(float(scale_14_elec), dtype=dtype)
                    w_e = jnp.where(is_full, zero, jnp.where(is_14, s14_e, one))
                    valid_c = nb_act & (dist >= eps4)
                    inv_d = jnp.where(valid_c, one / dist, zero)
                    c_sc = coul_c * jnp.asarray(1.0 / float(coulomb_dielectric), dtype=dtype)
                    e_c = jnp.where(valid_c, c_sc * qq * w_e * inv_d, zero)
                    f_c = jnp.where(valid_c, c_sc * qq * w_e * inv_d * (inv_d * inv_d), zero)
                    gq_c = jnp.where(valid_c, c_sc * w_e * inv_d, zero)
                    e_tot = e_tot + e_c
                    f_tot = f_tot + f_c
                    gq_mat = gq_mat + gq_c

                if include_lj:
                    s14_v = jnp.asarray(float(scale_14_vdw), dtype=dtype)
                    w_v = jnp.where(is_full, zero, jnp.where(is_14, s14_v, one))
                    sigi = _pl_load(sig_r, (b_sig + oi,), mask=bi_ok, other=one)
                    sigj = _pl_load(sig_r, (b_sig + oj,), mask=bj_ok, other=one)
                    epsi = _pl_load(eps_r, (b_eps + oi,), mask=bi_ok, other=zero)
                    epsj = _pl_load(eps_r, (b_eps + oj,), mask=bj_ok, other=zero)
                    lam = _pl_load(lam_r, (jnp.full((1,), b_lam, dtype=jnp.int32),))[0]
                    _oml_gb = jnp.maximum(one - lam, zero)
                    _oml_gb_pow = one
                    for _ in range(int(softcore_exponent)):
                        _oml_gb_pow = _oml_gb_pow * _oml_gb
                    soft_t = jnp.asarray(float(softcore_alpha), dtype=dtype) * _oml_gb_pow
                    sig_ij = half * (sigi[:, None] + sigj[None, :])
                    sig_s = jnp.where(nb_act, jnp.maximum(sig_ij, eps4), one)
                    eps_p = jnp.maximum(epsi[:, None] * epsj[None, :], zero)
                    eps_ij = jnp.where(eps_p > zero, jnp.sqrt(eps_p + jnp.asarray(1e-30, dtype=dtype)), zero)
                    _u_gb = dist / sig_s
                    _u2_gb = _u_gb * _u_gb
                    u6 = _u2_gb * _u2_gb * _u2_gb
                    x_inv6 = one / jnp.maximum(soft_t + u6, jnp.asarray(1e-6, dtype=dtype))
                    e_lj_u = four * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)
                    f_lj_u = (
                        jnp.asarray(24.0, dtype=dtype)
                        * eps_ij
                        * lam
                        * (two * x_inv6 - one)
                        * (x_inv6 * x_inv6)
                        * u6
                        / r_sq
                    )
                    if has_switch and cutoff is not None and switch_dist is not None:
                        sw_w, dsw_dr = _lj_switch_weight_and_deriv(
                            dist, float(cutoff), max(0.0, float(cutoff) - float(switch_dist))
                        )
                        e_lj = jnp.where(nb_act, w_v * sw_w * e_lj_u, zero)
                        f_lj = jnp.where(nb_act, w_v * (sw_w * f_lj_u - dsw_dr * e_lj_u / dist), zero)
                    else:
                        e_lj = jnp.where(nb_act, w_v * e_lj_u, zero)
                        f_lj = jnp.where(nb_act, w_v * f_lj_u, zero)
                    e_tot = e_tot + e_lj
                    f_tot = f_tot + f_lj

            e_blk = jnp.full((1,), jnp.sum(e_tot), dtype=dtype)
            _pl_atomic_add_1d(
                e_out_r, jnp.full((1,), b_id, dtype=jnp.int32), e_blk, valid_tp[None], interpret
            )

            gqi = jnp.sum(gq_mat * qj[None, :], axis=1)
            gqj = jnp.sum(gq_mat * qi[:, None], axis=0)
            _pl_atomic_add_1d(gq_out_r, b_pos + oi, jnp.where(is_diag, gqi + gqj, gqi), mi, interpret)
            _pl_atomic_add_1d(gq_out_r, b_pos + oj, gqj, mj & (~is_diag), interpret)

            if compute_forces:
                fxi = jnp.sum(f_tot * dx, axis=1)
                fyi = jnp.sum(f_tot * dy, axis=1)
                fzi = jnp.sum(f_tot * dz, axis=1)
                fxj = zero - jnp.sum(f_tot * dx, axis=0)
                fyj = zero - jnp.sum(f_tot * dy, axis=0)
                fzj = zero - jnp.sum(f_tot * dz, axis=0)
                _pl_atomic_add_1d(fx_out_r, b_pos + oi, jnp.where(is_diag, fxi + fxj, fxi), mi, interpret)
                _pl_atomic_add_1d(fy_out_r, b_pos + oi, jnp.where(is_diag, fyi + fyj, fyi), mi, interpret)
                _pl_atomic_add_1d(fz_out_r, b_pos + oi, jnp.where(is_diag, fzi + fzj, fzi), mi, interpret)
                _pl_atomic_add_1d(fx_out_r, b_pos + oj, fxj, mj & (~is_diag), interpret)
                _pl_atomic_add_1d(fy_out_r, b_pos + oj, fyj, mj & (~is_diag), interpret)
                _pl_atomic_add_1d(fz_out_r, b_pos + oj, fzj, mj & (~is_diag), interpret)

        any_spec = _any_block_spec()
        fx_flat, fy_flat, fz_flat, dedb_flat, gq_flat, e_out = pl.pallas_call(
            _pass2a_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(fx_flat.shape, dtype),
                jax.ShapeDtypeStruct(fy_flat.shape, dtype),
                jax.ShapeDtypeStruct(fz_flat.shape, dtype),
                jax.ShapeDtypeStruct(dedb_flat.shape, dtype),
                jax.ShapeDtypeStruct(gq_flat.shape, dtype),
                jax.ShapeDtypeStruct(e_out.shape, dtype),
            ),
            in_specs=[any_spec] * 21,
            out_specs=(any_spec,) * 6,
            grid=(pair_q.total_blocks,),
            input_output_aliases={15: 0, 16: 1, 17: 2, 18: 3, 19: 4, 20: 5},
            interpret=interpret,
            compiler_params=cparams if not interpret else None,
        )(
            px_flat, py_flat, pz_flat, q_flat, sig_flat, eps_flat, born_flat, msk_flat,
            box_flat, lam_flat, tp_flat, ef_flat, e14_flat,
            pair_q.cta_to_replica, pair_q.cta_to_local_block,
            fx_flat, fy_flat, fz_flat, dedb_flat, gq_flat, e_out,
        )

    # --- Pass 2b: Symmetric Tile-Pair Born Chain-Rule Forces (aliased in-place) ---
    if compute_forces and include_gb and pair_q.total_blocks > 0:
        w_flat = dedb_flat * db_di_flat

        def _pass2b_chain_kernel(  # noqa: PLR0913
            px_r, py_r, pz_r, rad_r, sr_r, w_r, msk_r, box_r, tp_r, cta_rep_r, cta_blk_r,
            fx_in_r, fy_in_r, fz_in_r, fx_out_r, fy_out_r, fz_out_r,
        ):
            del fx_in_r, fy_in_r, fz_in_r
            cta = pl.program_id(0)
            b_id = _pl_load(cta_rep_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            p_local = _pl_load(cta_blk_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            tp_base = (b_id * M + p_local) * 2
            ti = _pl_load(tp_r, (jnp.full((1,), tp_base, dtype=jnp.int32),))[0]
            tj = _pl_load(tp_r, (jnp.full((1,), tp_base + 1, dtype=jnp.int32),))[0]
            valid_tp = (ti >= 0) & (tj >= 0)
            ti_s, tj_s = jnp.maximum(ti, 0), jnp.maximum(tj, 0)
            is_diag = ti_s == tj_s

            lane = jnp.arange(_GB_TILE, dtype=jnp.int32)
            oi = ti_s * _GB_TILE + lane
            oj = tj_s * _GB_TILE + lane
            bi_ok = (oi < N) & valid_tp
            bj_ok = (oj < N) & valid_tp

            b_pos = b_id * n_stride
            b_rad = (b_id if B_rad > 1 else jnp.int32(0)) * n_stride
            b_sr = (b_id if B_sr > 1 else jnp.int32(0)) * n_stride
            b_msk = (b_id if B_msk > 1 else jnp.int32(0)) * n_stride
            b_box = (b_id if B_box > 1 else jnp.int32(0)) * 6

            zero = jnp.asarray(0.0, dtype=dtype)
            eps4 = jnp.asarray(1e-4, dtype=dtype)
            eps12 = jnp.asarray(1e-12, dtype=dtype)
            d_off = jnp.asarray(float(dielectric_offset), dtype=dtype)

            mi = (_pl_load(msk_r, (b_msk + oi,), mask=bi_ok, other=0) > 0) & bi_ok
            mj = (_pl_load(msk_r, (b_msk + oj,), mask=bj_ok, other=0) > 0) & bj_ok
            pxi = _pl_load(px_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pyi = _pl_load(py_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pzi = _pl_load(pz_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pxj = _pl_load(px_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pyj = _pl_load(py_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pzj = _pl_load(pz_r, (b_pos + oj,), mask=bj_ok, other=zero)
            safe_rad = jnp.asarray(1.5, dtype=dtype)
            safe_sr = jnp.asarray(1.2, dtype=dtype)
            ri_raw = jnp.where(mi, _pl_load(rad_r, (b_rad + oi,), mask=bi_ok, other=safe_rad), safe_rad)
            rj_raw = jnp.where(mj, _pl_load(rad_r, (b_rad + oj,), mask=bj_ok, other=safe_rad), safe_rad)
            ori = jnp.maximum(ri_raw - d_off, eps4)
            orj = jnp.maximum(rj_raw - d_off, eps4)
            sri = jnp.where(mi, _pl_load(sr_r, (b_sr + oi,), mask=bi_ok, other=safe_sr), safe_sr)
            srj = jnp.where(mj, _pl_load(sr_r, (b_sr + oj,), mask=bj_ok, other=safe_sr), safe_sr)
            wi = _pl_load(w_r, (b_pos + oi,), mask=bi_ok, other=zero)
            wj = _pl_load(w_r, (b_pos + oj,), mask=bj_ok, other=zero)

            box6 = tuple(
                _pl_load(box_r, (jnp.full((1,), b_box + k, dtype=jnp.int32),))[0] for k in range(6)
            )
            dx, dy, dz = _min_image3(
                (pxi[:, None] - pxj[None, :], pyi[:, None] - pyj[None, :], pzi[:, None] - pzj[None, :]),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            active = mi[:, None] & mj[None, :] & jnp.where(is_diag, lane[:, None] < lane[None, :], True)
            if has_born_cut:
                active = active & (dist < jnp.asarray(born_cut_val, dtype=dtype))

            dI_ij = _pair_integral_deriv(dist, ori[:, None], srj[None, :])
            dI_ji = _pair_integral_deriv(dist, orj[None, :], sri[:, None])
            f_sc = jnp.where(active, (zero - (wi[:, None] * dI_ij + wj[None, :] * dI_ji)) / dist, zero)

            fxi = jnp.sum(f_sc * dx, axis=1)
            fyi = jnp.sum(f_sc * dy, axis=1)
            fzi = jnp.sum(f_sc * dz, axis=1)
            fxj = zero - jnp.sum(f_sc * dx, axis=0)
            fyj = zero - jnp.sum(f_sc * dy, axis=0)
            fzj = zero - jnp.sum(f_sc * dz, axis=0)
            _pl_atomic_add_1d(fx_out_r, b_pos + oi, jnp.where(is_diag, fxi + fxj, fxi), mi, interpret)
            _pl_atomic_add_1d(fy_out_r, b_pos + oi, jnp.where(is_diag, fyi + fyj, fyi), mi, interpret)
            _pl_atomic_add_1d(fz_out_r, b_pos + oi, jnp.where(is_diag, fzi + fzj, fzi), mi, interpret)
            _pl_atomic_add_1d(fx_out_r, b_pos + oj, fxj, mj & (~is_diag), interpret)
            _pl_atomic_add_1d(fy_out_r, b_pos + oj, fyj, mj & (~is_diag), interpret)
            _pl_atomic_add_1d(fz_out_r, b_pos + oj, fzj, mj & (~is_diag), interpret)

        any_spec = _any_block_spec()
        fx_flat, fy_flat, fz_flat = pl.pallas_call(
            _pass2b_chain_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(fx_flat.shape, dtype),
                jax.ShapeDtypeStruct(fy_flat.shape, dtype),
                jax.ShapeDtypeStruct(fz_flat.shape, dtype),
            ),
            in_specs=[any_spec] * 14,
            out_specs=(any_spec,) * 3,
            grid=(pair_q.total_blocks,),
            input_output_aliases={11: 0, 12: 1, 13: 2},
            interpret=interpret,
            compiler_params=cparams if not interpret else None,
        )(
            px_flat, py_flat, pz_flat, rad_flat, sr_flat, w_flat, msk_flat, box_flat, tp_flat,
            pair_q.cta_to_replica, pair_q.cta_to_local_block, fx_flat, fy_flat, fz_flat,
        )

    forces_w = jnp.stack(
        [
            fx_flat.reshape(B, n_stride)[:, :N],
            fy_flat.reshape(B, n_stride)[:, :N],
            fz_flat.reshape(B, n_stride)[:, :N],
        ],
        axis=-1,
    ).astype(orig_dtype)
    born_w = born_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    gq_w = gq_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    i_sum_w = i_sum_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    dedb_w = dedb_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    db_di_w = db_di_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)

    if inv_perm is not None:
        forces_out = _perm_1d_or_2d(forces_w, inv_perm)
        born_out = _perm_1d_or_2d(born_w, inv_perm)
        gq_out = _perm_1d_or_2d(gq_w, inv_perm)
        i_sum_out = _perm_1d_or_2d(i_sum_w, inv_perm)
        dedb_out = _perm_1d_or_2d(dedb_w, inv_perm)
        db_di_out = _perm_1d_or_2d(db_di_w, inv_perm)
    else:
        forces_out, born_out, gq_out = forces_w, born_w, gq_w
        i_sum_out, dedb_out, db_di_out = i_sum_w, dedb_w, db_di_w

    return e_out.astype(orig_dtype), forces_out, born_out, gq_out, i_sum_out, dedb_out, db_di_out


def _run_gbsa_tile_param_bwd_pallas(  # noqa: PLR0913
    ct_e: jax.Array,
    pos_b: jax.Array,
    rad_b: jax.Array,
    sr_b: jax.Array,
    sig_b: jax.Array,
    eps_b: jax.Array,
    msk_b: jax.Array,
    born_b: jax.Array,
    i_sum_b: jax.Array,
    dedb_b: jax.Array,
    db_di_b: jax.Array,
    tile_list: GBSATilePairList,
    box6_b: jax.Array,
    lam_b: jax.Array,
    has_box: bool,
    dielectric_offset: float,
    probe_radius: float,
    surface_tension: float,
    obc_alpha: float,
    obc_beta: float,
    obc_gamma: float,
    cutoff: float | None,
    born_cutoff: float | None,
    switch_dist: float | None,
    softcore_alpha: float,
    softcore_exponent: int,
    include_gb: bool,
    include_ace: bool,
    include_lj: bool,
    need_rad: bool,
    need_sr: bool,
    need_sig: bool,
    need_eps: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """MK-73 32x32 symmetric tile-pair Pallas parameter VJP (zero ``(N, N)`` matrices)."""
    B, N, _ = pos_b.shape
    orig_dtype = pos_b.dtype
    dtype = jnp.dtype(jnp.float32) if (not interpret and orig_dtype == jnp.float64) else orig_dtype

    perm = tile_list.atom_perm
    inv_perm = tile_list.atom_inv_perm
    if perm is not None:
        pos_w = _perm_1d_or_2d(pos_b, perm)
        rad_w = _perm_1d_or_2d(rad_b, perm)
        sr_w = _perm_1d_or_2d(sr_b, perm)
        sig_w = _perm_1d_or_2d(sig_b, perm)
        eps_w = _perm_1d_or_2d(eps_b, perm)
        msk_w = _perm_1d_or_2d(msk_b, perm)
        born_w = _perm_1d_or_2d(born_b, perm)
        i_sum_w = _perm_1d_or_2d(i_sum_b, perm)
        dedb_w = _perm_1d_or_2d(dedb_b, perm)
        db_di_w = _perm_1d_or_2d(db_di_b, perm)
    else:
        pos_w, rad_w, sr_w, sig_w, eps_w, msk_w = pos_b, rad_b, sr_b, sig_b, eps_b, msk_b
        born_w, i_sum_w, dedb_w, db_di_w = born_b, i_sum_b, dedb_b, db_di_b

    g_col = ct_e.astype(orig_dtype)[:, None]
    msk_bool = jnp.broadcast_to(msk_w > 0, (B, N))
    rad_full = jnp.where(
        msk_bool,
        jnp.broadcast_to(rad_w, (B, N)).astype(orig_dtype),
        jnp.asarray(1.5, dtype=orig_dtype),
    )
    zero_o = jnp.asarray(0.0, dtype=orig_dtype)
    one_o = jnp.asarray(1.0, dtype=orig_dtype)
    eps4_o = jnp.asarray(1e-4, dtype=orig_dtype)

    grad_rad_diag = jnp.zeros((B, N), dtype=orig_dtype)
    if need_rad:
        if include_gb:
            d_off_o = jnp.asarray(float(dielectric_offset), dtype=orig_dtype)
            dor_dr = jnp.where(rad_full - d_off_o > eps4_o, one_o, zero_o)
            drs_dr = jnp.where(rad_full > eps4_o, one_o, zero_o)
            ori_o = jnp.maximum(rad_full - d_off_o, eps4_o)
            rs_o = jnp.maximum(rad_full, eps4_o)
            psi = i_sum_w * ori_o
            psi2 = psi * psi
            a_o = jnp.asarray(float(obc_alpha), dtype=orig_dtype)
            b_o = jnp.asarray(float(obc_beta), dtype=orig_dtype)
            g_o = jnp.asarray(float(obc_gamma), dtype=orig_dtype)
            th = jnp.tanh(a_o * psi - b_o * psi2 + g_o * psi2 * psi)
            darg = a_o - jnp.asarray(2.0, dtype=orig_dtype) * b_o * psi + jnp.asarray(3.0, dtype=orig_dtype) * g_o * psi2
            inv_r_raw = one_o / ori_o - th / rs_o
            inv_ok = jnp.where(inv_r_raw > eps4_o, one_o, zero_o)
            dR_dr = (
                inv_ok
                * (born_w * born_w)
                * (
                    dor_dr / (ori_o * ori_o)
                    + (one_o - th * th) * darg * i_sum_w * dor_dr / rs_o
                    - th * drs_dr / (rs_o * rs_o)
                )
            )
            grad_rad_diag = grad_rad_diag + g_col * dedb_w * dR_dr
        if include_ace:
            rp = rad_full + jnp.asarray(float(probe_radius), dtype=orig_dtype)
            c_ace = jnp.asarray(4.0 * math.pi * float(surface_tension), dtype=orig_dtype)
            ratio = rad_full / born_w
            de_ace_dr = c_ace * (ratio ** 5) * (
                jnp.asarray(2.0, dtype=orig_dtype) * rp * ratio
                + jnp.asarray(6.0, dtype=orig_dtype) * (rp * rp) / born_w
            )
            grad_rad_diag = grad_rad_diag + g_col * de_ace_dr
        grad_rad_diag = jnp.where(msk_bool, grad_rad_diag, zero_o)

    tp_b = (
        tile_list.tile_pairs
        if tile_list.tile_pairs.ndim == 3
        else jnp.broadcast_to(tile_list.tile_pairs[None, ...], (B, *tile_list.tile_pairs.shape))
    )
    ef_b = (
        tile_list.excl_full_mask
        if tile_list.excl_full_mask.ndim == 3
        else jnp.broadcast_to(tile_list.excl_full_mask[None, ...], (B, *tile_list.excl_full_mask.shape))
    )
    e14_b = (
        tile_list.excl_14_mask
        if tile_list.excl_14_mask.ndim == 3
        else jnp.broadcast_to(tile_list.excl_14_mask[None, ...], (B, *tile_list.excl_14_mask.shape))
    )
    M = int(tp_b.shape[1])
    pair_q = tile_list.pair_queue
    if pair_q is None or pair_q.batch_size != B:
        nvp = (
            np.broadcast_to(np.asarray(tile_list.num_valid_pairs).reshape(-1), (B,))
            if not isinstance(tile_list.num_valid_pairs, jax.core.Tracer)
            else [M] * B
        )
        pair_q = build_ragged_work_queue(nvp, tile_size=1, padded_size=M)

    n_blocks_pad = max(1, (N + _GB_TILE - 1) // _GB_TILE)
    n_stride = n_blocks_pad * _GB_TILE if interpret else N
    pad_n = n_stride - N

    def _flat(arr, pad_val=0.0):
        a = arr.astype(dtype) if arr.dtype != jnp.int32 else arr
        if pad_n > 0:
            a = jnp.pad(a, ((0, 0), (0, pad_n)), constant_values=pad_val)
        return jnp.asarray(a).reshape(-1)

    pos_f = pos_w.astype(dtype)
    if pad_n > 0:
        pos_f = jnp.pad(pos_f, ((0, 0), (0, pad_n), (0, 0)))
    px_flat = jnp.asarray(pos_f[:, :, 0]).reshape(-1)
    py_flat = jnp.asarray(pos_f[:, :, 1]).reshape(-1)
    pz_flat = jnp.asarray(pos_f[:, :, 2]).reshape(-1)

    B_rad, B_sr = int(rad_w.shape[0]), int(sr_w.shape[0])
    B_sig, B_eps, B_msk = int(sig_w.shape[0]), int(eps_w.shape[0]), int(msk_w.shape[0])
    B_box, B_lam = int(box6_b.shape[0]), int(lam_b.shape[0])

    rad_flat = _flat(rad_w, 1.5)
    sr_flat = _flat(sr_w, 1.2)
    sig_flat = _flat(sig_w, 1.0)
    eps_flat = _flat(eps_w, 0.0)
    msk_flat = _flat(msk_w.astype(jnp.int32), 0)
    w_flat = _flat(g_col * dedb_w * db_di_w, 0.0)
    g_flat = ct_e.astype(dtype).reshape(-1)
    box_flat = box6_b.astype(dtype).reshape(-1)
    lam_flat = lam_b.astype(dtype).reshape(-1)
    tp_flat = tp_b.astype(jnp.int32).reshape(-1)
    ef_flat = ef_b.astype(jnp.int32).reshape(-1)
    e14_flat = e14_b.astype(jnp.int32).reshape(-1)

    grad_r_flat = _flat(grad_rad_diag, 0.0)
    grad_sr_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    grad_sig_flat = jnp.zeros((B * n_stride,), dtype=dtype)
    grad_eps_flat = jnp.zeros((B * n_stride,), dtype=dtype)

    born_cut = born_cutoff if born_cutoff is not None else cutoff
    has_born_cut = born_cut is not None
    born_cut_val = float(born_cut) if born_cut is not None else 0.0
    has_nb_cut = cutoff is not None
    nb_cut_val = float(cutoff) if cutoff is not None else 0.0
    has_switch = cutoff is not None and switch_dist is not None
    run_gb_bwd = include_gb and (need_rad or need_sr)
    run_lj_bwd = include_lj and (need_sig or need_eps)

    if (run_gb_bwd or run_lj_bwd) and pair_q.total_blocks > 0:
        def _param_bwd_kernel(  # noqa: PLR0913
            px_r, py_r, pz_r, rad_r, sr_r, sig_r, eps_r, w_r, g_r, msk_r, box_r, lam_r,
            tp_r, ef_r, e14_r, cta_rep_r, cta_blk_r,
            gr_in, gsr_in, gsig_in, geps_in,
            gr_out, gsr_out, gsig_out, geps_out,
        ):
            del gr_in, gsr_in, gsig_in, geps_in
            cta = pl.program_id(0)
            b_id = _pl_load(cta_rep_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            p_local = _pl_load(cta_blk_r, (jnp.full((1,), cta, dtype=jnp.int32),))[0]
            pair_abs = b_id * M + p_local
            tp_base = pair_abs * 2
            ti = _pl_load(tp_r, (jnp.full((1,), tp_base, dtype=jnp.int32),))[0]
            tj = _pl_load(tp_r, (jnp.full((1,), tp_base + 1, dtype=jnp.int32),))[0]
            valid_tp = (ti >= 0) & (tj >= 0)
            ti_s, tj_s = jnp.maximum(ti, 0), jnp.maximum(tj, 0)
            is_diag = ti_s == tj_s

            lane = jnp.arange(_GB_TILE, dtype=jnp.int32)
            oi = ti_s * _GB_TILE + lane
            oj = tj_s * _GB_TILE + lane
            bi_ok = (oi < N) & valid_tp
            bj_ok = (oj < N) & valid_tp

            b_pos = b_id * n_stride
            b_rad = (b_id if B_rad > 1 else jnp.int32(0)) * n_stride
            b_sr = (b_id if B_sr > 1 else jnp.int32(0)) * n_stride
            b_sig = (b_id if B_sig > 1 else jnp.int32(0)) * n_stride
            b_eps = (b_id if B_eps > 1 else jnp.int32(0)) * n_stride
            b_msk = (b_id if B_msk > 1 else jnp.int32(0)) * n_stride
            b_box = (b_id if B_box > 1 else jnp.int32(0)) * 6
            b_lam = b_id if B_lam > 1 else jnp.int32(0)

            zero = jnp.asarray(0.0, dtype=dtype)
            one = jnp.asarray(1.0, dtype=dtype)
            two = jnp.asarray(2.0, dtype=dtype)
            half = jnp.asarray(0.5, dtype=dtype)
            eps4 = jnp.asarray(1e-4, dtype=dtype)
            eps12 = jnp.asarray(1e-12, dtype=dtype)

            mi = (_pl_load(msk_r, (b_msk + oi,), mask=bi_ok, other=0) > 0) & bi_ok
            mj = (_pl_load(msk_r, (b_msk + oj,), mask=bj_ok, other=0) > 0) & bj_ok
            pxi = _pl_load(px_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pyi = _pl_load(py_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pzi = _pl_load(pz_r, (b_pos + oi,), mask=bi_ok, other=zero)
            pxj = _pl_load(px_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pyj = _pl_load(py_r, (b_pos + oj,), mask=bj_ok, other=zero)
            pzj = _pl_load(pz_r, (b_pos + oj,), mask=bj_ok, other=zero)

            box6 = tuple(
                _pl_load(box_r, (jnp.full((1,), b_box + k, dtype=jnp.int32),))[0] for k in range(6)
            )
            dx, dy, dz = _min_image3(
                (pxi[:, None] - pxj[None, :], pyi[:, None] - pyj[None, :], pzi[:, None] - pzj[None, :]),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            active = mi[:, None] & mj[None, :] & jnp.where(is_diag, lane[:, None] < lane[None, :], True)

            if run_gb_bwd:
                d_off = jnp.asarray(float(dielectric_offset), dtype=dtype)
                safe_rad = jnp.asarray(1.5, dtype=dtype)
                safe_sr = jnp.asarray(1.2, dtype=dtype)
                ri_raw = jnp.where(mi, _pl_load(rad_r, (b_rad + oi,), mask=bi_ok, other=safe_rad), safe_rad)
                rj_raw = jnp.where(mj, _pl_load(rad_r, (b_rad + oj,), mask=bj_ok, other=safe_rad), safe_rad)
                ori = jnp.maximum(ri_raw - d_off, eps4)
                orj = jnp.maximum(rj_raw - d_off, eps4)
                sri = jnp.where(mi, _pl_load(sr_r, (b_sr + oi,), mask=bi_ok, other=safe_sr), safe_sr)
                srj = jnp.where(mj, _pl_load(sr_r, (b_sr + oj,), mask=bj_ok, other=safe_sr), safe_sr)
                wi = _pl_load(w_r, (b_pos + oi,), mask=bi_ok, other=zero)
                wj = _pl_load(w_r, (b_pos + oj,), mask=bj_ok, other=zero)
                gb_act = (
                    active & (dist < jnp.asarray(born_cut_val, dtype=dtype)) if has_born_cut else active
                )
                dI_ij_dor, dI_ij_dsr = _pair_integral_param_derivs(dist, ori[:, None], srj[None, :])
                dI_ji_dor, dI_ji_dsr = _pair_integral_param_derivs(dist, orj[None, :], sri[:, None])
                if need_rad:
                    dor_i = jnp.where(ri_raw - d_off > eps4, one, zero)
                    dor_j = jnp.where(rj_raw - d_off > eps4, one, zero)
                    gri = dor_i * jnp.sum(jnp.where(gb_act, wi[:, None] * dI_ij_dor, zero), axis=1)
                    grj = dor_j * jnp.sum(jnp.where(gb_act, wj[None, :] * dI_ji_dor, zero), axis=0)
                    _pl_atomic_add_1d(gr_out, b_pos + oi, jnp.where(is_diag, gri + grj, gri), mi, interpret)
                    _pl_atomic_add_1d(gr_out, b_pos + oj, grj, mj & (~is_diag), interpret)
                if need_sr:
                    gsri = jnp.sum(jnp.where(gb_act, wj[None, :] * dI_ji_dsr, zero), axis=1)
                    gsrj = jnp.sum(jnp.where(gb_act, wi[:, None] * dI_ij_dsr, zero), axis=0)
                    _pl_atomic_add_1d(gsr_out, b_pos + oi, jnp.where(is_diag, gsri + gsrj, gsri), mi, interpret)
                    _pl_atomic_add_1d(gsr_out, b_pos + oj, gsrj, mj & (~is_diag), interpret)

            if run_lj_bwd:
                nb_act = active & (dist < jnp.asarray(nb_cut_val, dtype=dtype)) if has_nb_cut else active
                m_base = pair_abs * _GB_TILE + lane
                m_full = _pl_load(ef_r, (m_base,), mask=valid_tp, other=0)
                m_14 = _pl_load(e14_r, (m_base,), mask=valid_tp, other=0)
                bit_b = jnp.left_shift(jnp.uint32(1), lane.astype(jnp.uint32))[None, :]
                is_full = (m_full.astype(jnp.uint32)[:, None] & bit_b) != jnp.uint32(0)
                is_14 = (m_14.astype(jnp.uint32)[:, None] & bit_b) != jnp.uint32(0)
                w_v = jnp.where(
                    is_full, zero, jnp.where(is_14, jnp.asarray(float(tile_list.scale_14_vdw), dtype=dtype), one)
                )
                sigi = _pl_load(sig_r, (b_sig + oi,), mask=bi_ok, other=one)
                sigj = _pl_load(sig_r, (b_sig + oj,), mask=bj_ok, other=one)
                epsi = _pl_load(eps_r, (b_eps + oi,), mask=bi_ok, other=zero)
                epsj = _pl_load(eps_r, (b_eps + oj,), mask=bj_ok, other=zero)
                lam = _pl_load(lam_r, (jnp.full((1,), b_lam, dtype=jnp.int32),))[0]
                g_b = _pl_load(g_r, (jnp.full((1,), b_id, dtype=jnp.int32),))[0]
                _oml_bwd = jnp.maximum(one - lam, zero)
                _oml_bwd_pow = one
                for _ in range(int(softcore_exponent)):
                    _oml_bwd_pow = _oml_bwd_pow * _oml_bwd
                soft_t = jnp.asarray(float(softcore_alpha), dtype=dtype) * _oml_bwd_pow
                sig_ij = half * (sigi[:, None] + sigj[None, :])
                sig_ok = sig_ij > eps4
                sig_s = jnp.where(nb_act, jnp.maximum(sig_ij, eps4), one)
                eps_p = jnp.maximum(epsi[:, None] * epsj[None, :], zero)
                eps_pos = eps_p > zero
                eps_ij = jnp.where(eps_pos, jnp.sqrt(eps_p + jnp.asarray(1e-30, dtype=dtype)), one)
                _u_pb = dist / sig_s
                _u2_pb = _u_pb * _u_pb
                u6 = _u2_pb * _u2_pb * _u2_pb
                x_inv6 = one / jnp.maximum(soft_t + u6, jnp.asarray(1e-6, dtype=dtype))
                sw_w = (
                    _lj_switch_weight_and_deriv(
                        dist, float(cutoff), max(0.0, float(cutoff) - float(switch_dist))
                    )[0]
                    if (has_switch and cutoff is not None and switch_dist is not None)
                    else one
                )
                if need_sig:
                    dsig_mat = jnp.where(
                        nb_act & sig_ok & eps_pos,
                        g_b
                        * jnp.asarray(12.0, dtype=dtype)
                        * w_v
                        * sw_w
                        * eps_ij
                        * lam
                        * (two * x_inv6 - one)
                        * (x_inv6 * x_inv6)
                        * u6
                        / sig_s,
                        zero,
                    )
                    gsigi = jnp.sum(dsig_mat, axis=1)
                    gsigj = jnp.sum(dsig_mat, axis=0)
                    _pl_atomic_add_1d(gsig_out, b_pos + oi, jnp.where(is_diag, gsigi + gsigj, gsigi), mi, interpret)
                    _pl_atomic_add_1d(gsig_out, b_pos + oj, gsigj, mj & (~is_diag), interpret)
                if need_eps:
                    deps_c = jnp.where(
                        nb_act & eps_pos,
                        g_b * two * w_v * sw_w * lam * (x_inv6 * x_inv6 - x_inv6) / eps_ij,
                        zero,
                    )
                    gepsi = jnp.sum(deps_c * epsj[None, :], axis=1)
                    gepsj = jnp.sum(deps_c * epsi[:, None], axis=0)
                    _pl_atomic_add_1d(geps_out, b_pos + oi, jnp.where(is_diag, gepsi + gepsj, gepsi), mi, interpret)
                    _pl_atomic_add_1d(geps_out, b_pos + oj, gepsj, mj & (~is_diag), interpret)

        any_spec = _any_block_spec()
        grad_r_flat, grad_sr_flat, grad_sig_flat, grad_eps_flat = pl.pallas_call(
            _param_bwd_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(grad_r_flat.shape, dtype),
                jax.ShapeDtypeStruct(grad_sr_flat.shape, dtype),
                jax.ShapeDtypeStruct(grad_sig_flat.shape, dtype),
                jax.ShapeDtypeStruct(grad_eps_flat.shape, dtype),
            ),
            in_specs=[any_spec] * 21,
            out_specs=(any_spec,) * 4,
            grid=(pair_q.total_blocks,),
            input_output_aliases={17: 0, 18: 1, 19: 2, 20: 3},
            interpret=interpret,
            compiler_params=resolve_pallas_compiler_params("megakernel") if not interpret else None,
        )(
            px_flat, py_flat, pz_flat, rad_flat, sr_flat, sig_flat, eps_flat,
            w_flat, g_flat, msk_flat, box_flat, lam_flat, tp_flat, ef_flat, e14_flat,
            pair_q.cta_to_replica, pair_q.cta_to_local_block,
            grad_r_flat, grad_sr_flat, grad_sig_flat, grad_eps_flat,
        )

    gr_w = grad_r_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    gsr_w = grad_sr_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    gsig_w = grad_sig_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)
    geps_w = grad_eps_flat.reshape(B, n_stride)[:, :N].astype(orig_dtype)

    if inv_perm is not None:
        return (
            _perm_1d_or_2d(gr_w, inv_perm),
            _perm_1d_or_2d(gsr_w, inv_perm),
            _perm_1d_or_2d(gsig_w, inv_perm),
            _perm_1d_or_2d(geps_w, inv_perm),
        )
    return gr_w, gsr_w, gsig_w, geps_w


def _prepare_gbsa_batch_inputs(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    box6: jax.Array,
    lam: jax.Array,
) -> tuple[
    bool,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    tuple[bool, bool, bool, bool, bool],
]:
    single_in = positions.ndim == 2
    pos_b = positions[None, ...] if single_in else positions
    B, N, _ = pos_b.shape
    dtype = pos_b.dtype

    def _norm1(x):
        is_b = x.ndim == 2 and not single_in
        xb = x if is_b else (x[None, :] if x.ndim == 1 else x[:1])
        return xb.astype(dtype), is_b

    q_b, b_q = _norm1(charges)
    rad_b, b_rad = _norm1(radii)
    sr_b, b_sr = _norm1(scaled_radii)
    sig_b, b_sig = _norm1(sigmas)
    eps_b, b_eps = _norm1(epsilons)
    msk_b = (
        atom_mask
        if (atom_mask.ndim == 2 and not single_in)
        else (atom_mask[None, :] if atom_mask.ndim == 1 else atom_mask[:1])
    ).astype(jnp.int32)
    bx_b = box6 if (box6.ndim == 2 and box6.shape[0] in (1, B)) else box6.reshape(1, 6)
    lm_arr = jnp.asarray(lam, dtype=dtype).reshape(-1)
    lm_b = lm_arr if lm_arr.shape[0] in (1, B) else jnp.broadcast_to(lm_arr[:1], (B,))
    return (
        single_in,
        pos_b,
        q_b,
        rad_b,
        sr_b,
        sig_b,
        eps_b,
        msk_b,
        bx_b.astype(dtype),
        lm_b,
        (b_q, b_rad, b_sr, b_sig, b_eps),
    )


def _make_gbsa_vmap_op(static_cfg: tuple):
    (
        has_box,
        solvent_dielectric,
        solute_dielectric,
        coulomb_dielectric,
        dielectric_offset,
        probe_radius,
        surface_tension,
        obc_alpha,
        obc_beta,
        obc_gamma,
        kappa,
        kappa_scale,
        cutoff,
        born_cutoff,
        switch_dist,
        softcore_alpha,
        softcore_exponent,
        coulomb_constant,
        include_gb,
        include_ace,
        include_coulomb,
        include_lj,
        compute_forces,
        interpret,
    ) = static_cfg

    @jax.custom_batching.custom_vmap
    def _eval(pos, q, rad, sr, sig, eps, msk, tile_list, box6, lam):
        single_in, pos_b, q_b, rad_b, sr_b, sig_b, eps_b, msk_b, bx_b, lm_b, _ = (
            _prepare_gbsa_batch_inputs(pos, q, rad, sr, sig, eps, msk, box6, lam)
        )
        outs = _run_gbsa_tile_pallas_fwd(
            pos_b, q_b, rad_b, sr_b, sig_b, eps_b, msk_b, tile_list, bx_b, lm_b,
            has_box, solvent_dielectric, solute_dielectric, coulomb_dielectric,
            dielectric_offset, probe_radius, surface_tension,
            obc_alpha, obc_beta, obc_gamma, kappa, kappa_scale,
            cutoff, born_cutoff, switch_dist, softcore_alpha, softcore_exponent,
            coulomb_constant, include_gb, include_ace, include_coulomb, include_lj,
            compute_forces, interpret,
        )
        if single_in:
            return tuple(o[0] for o in outs)
        return outs

    @_eval.def_vmap
    def _eval_vmap(axis_size, in_batched, pos, q, rad, sr, sig, eps, msk, tile_list, box6, lam):
        pos_b = pos if in_batched[0] else jnp.broadcast_to(pos[None, ...], (axis_size, *pos.shape))
        q_b = q if in_batched[1] else q[None, ...]
        rad_b = rad if in_batched[2] else rad[None, ...]
        sr_b = sr if in_batched[3] else sr[None, ...]
        sig_b = sig if in_batched[4] else sig[None, ...]
        eps_b = eps if in_batched[5] else eps[None, ...]
        msk_b = msk if in_batched[6] else msk[None, ...]
        bx_b = box6 if in_batched[8] else box6[None, ...]
        lm_b = lam.reshape(-1) if in_batched[9] else lam.reshape(1)
        tp_v = tile_list
        if any(in_batched[7] if isinstance(in_batched[7], tuple) else [in_batched[7]]):
            tp_v = GBSATilePairList(
                tile_pairs=tile_list.tile_pairs,
                excl_full_mask=tile_list.excl_full_mask,
                excl_14_mask=tile_list.excl_14_mask,
                num_valid_pairs=tile_list.num_valid_pairs,
                scale_14_vdw=tile_list.scale_14_vdw,
                scale_14_elec=tile_list.scale_14_elec,
                atom_perm=tile_list.atom_perm,
                atom_inv_perm=tile_list.atom_inv_perm,
                pair_queue=None,
                atom_queue=None,
                is_batched=True,
            )
        outs = _run_gbsa_tile_pallas_fwd(
            pos_b, q_b, rad_b, sr_b, sig_b, eps_b, msk_b, tp_v, bx_b, lm_b,
            has_box, solvent_dielectric, solute_dielectric, coulomb_dielectric,
            dielectric_offset, probe_radius, surface_tension,
            obc_alpha, obc_beta, obc_gamma, kappa, kappa_scale,
            cutoff, born_cutoff, switch_dist, softcore_alpha, softcore_exponent,
            coulomb_constant, include_gb, include_ace, include_coulomb, include_lj,
            compute_forces, interpret,
        )
        return outs, (True,) * 7

    return _eval


@functools.partial(jax.custom_vjp, nondiff_argnums=(10,))
def _gbsa_tile_energy_diff(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_list: GBSATilePairList,
    box6: jax.Array,
    lam: jax.Array,
    static_cfg: tuple,
) -> jax.Array:
    op = _make_gbsa_vmap_op((*static_cfg[:-2], False, static_cfg[-1]))
    return op(positions, charges, radii, scaled_radii, sigmas, epsilons, atom_mask, tile_list, box6, lam)[0]


def _gbsa_tile_energy_fwd(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_list: GBSATilePairList,
    box6: jax.Array,
    lam: jax.Array,
    static_cfg: tuple,
):
    pos_v = _unwrap_primal(positions)
    q_v = _unwrap_primal(charges)
    rad_v = _unwrap_primal(radii)
    sr_v = _unwrap_primal(scaled_radii)
    sig_v = _unwrap_primal(sigmas)
    eps_v = _unwrap_primal(epsilons)
    msk_v = _unwrap_primal(atom_mask)
    tl_v = jax.tree_util.tree_map(_unwrap_primal, tile_list)
    bx_v = _unwrap_primal(box6)
    lm_v = _unwrap_primal(lam)

    pert_flags = (
        _is_perturbed(positions),
        _is_perturbed(charges),
        _is_perturbed(radii),
        _is_perturbed(scaled_radii),
        _is_perturbed(sigmas),
        _is_perturbed(epsilons),
    )
    op = _make_gbsa_vmap_op((*static_cfg[:-2], True, static_cfg[-1]))
    e_out, f_out, br_out, gq_out, i_sum_out, dedb_out, db_di_out = op(
        pos_v, q_v, rad_v, sr_v, sig_v, eps_v, msk_v, tl_v, bx_v, lm_v
    )
    residuals = (
        pos_v, q_v, rad_v, sr_v, sig_v, eps_v, msk_v, tl_v, bx_v, lm_v,
        f_out, br_out, gq_out, i_sum_out, dedb_out, db_di_out, pert_flags,
    )
    return e_out, residuals


def _gbsa_tile_energy_bwd(static_cfg: tuple, residuals, g):
    (
        pos_v, q_v, rad_v, sr_v, sig_v, eps_v, msk_v, tl_v, bx_v, lm_v,
        f_out, br_out, gq_out, i_sum_out, dedb_out, db_di_out, pert_flags,
    ) = residuals
    p_pos, p_q, p_rad, p_sr, p_sig, p_eps = pert_flags

    def _f0_tree(tree):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x, dtype=jax.dtypes.float0)
            if jnp.issubdtype(x.dtype, jnp.integer) or jnp.issubdtype(x.dtype, jnp.bool_)
            else jnp.zeros_like(x),
            tree,
        )

    z_msk = _f0_tree(msk_v)
    z_tl = _f0_tree(tl_v)
    z_bx = jnp.zeros_like(bx_v)
    z_lm = jnp.zeros_like(lm_v)

    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return (
            jnp.zeros_like(pos_v),
            jnp.zeros_like(q_v),
            jnp.zeros_like(rad_v),
            jnp.zeros_like(sr_v),
            jnp.zeros_like(sig_v),
            jnp.zeros_like(eps_v),
            z_msk,
            z_tl,
            z_bx,
            z_lm,
        )

    (
        has_box,
        solvent_dielectric,
        solute_dielectric,
        coulomb_dielectric,
        dielectric_offset,
        probe_radius,
        surface_tension,
        obc_alpha,
        obc_beta,
        obc_gamma,
        kappa,
        kappa_scale,
        cutoff,
        born_cutoff,
        switch_dist,
        softcore_alpha,
        softcore_exponent,
        coulomb_constant,
        include_gb,
        include_ace,
        include_coulomb,
        include_lj,
        _cf,
        interpret,
    ) = static_cfg

    if in_export_context():
        def _pure_fn(p_, q_, r_, s_, sg_, ep_):
            return gbsa_tile_reference(
                p_, q_, r_, s_, sg_, ep_, tl_v,
                atom_mask=msk_v, box=bx_v if has_box else None, softcore_lambda=lm_v,
                solvent_dielectric=solvent_dielectric, solute_dielectric=solute_dielectric,
                coulomb_dielectric=coulomb_dielectric, dielectric_offset=dielectric_offset,
                probe_radius=probe_radius, surface_tension=surface_tension,
                obc_alpha=obc_alpha, obc_beta=obc_beta, obc_gamma=obc_gamma,
                kappa=kappa, kappa_scale=kappa_scale, cutoff=cutoff, born_cutoff=born_cutoff,
                switch_dist=switch_dist, softcore_alpha=softcore_alpha,
                softcore_exponent=softcore_exponent, coulomb_constant=coulomb_constant,
                include_gb=include_gb, include_ace=include_ace,
                include_coulomb=include_coulomb, include_lj=include_lj, return_forces=False,
            )

        _, vjp_fn = jax.vjp(_pure_fn, pos_v, q_v, rad_v, sr_v, sig_v, eps_v)
        dp, dq, dr, dsr, dsig, deps = vjp_fn(g)
        return dp, dq, dr, dsr, dsig, deps, z_msk, z_tl, z_bx, z_lm

    single_in, pos_b, _q_b, rad_b, sr_b, sig_b, eps_b, msk_b, bx_b, lm_b, b_flags = (
        _prepare_gbsa_batch_inputs(pos_v, q_v, rad_v, sr_v, sig_v, eps_v, msk_v, bx_v, lm_v)
    )
    b_q, b_rad, b_sr, b_sig, b_eps = b_flags
    B = int(pos_b.shape[0])
    dtype = pos_v.dtype
    g_vec = jnp.asarray(g, dtype=dtype).reshape((1,) if single_in else (B,))

    f_b = f_out[None, ...] if single_in else f_out
    gq_b = gq_out[None, ...] if single_in else gq_out
    br_b = br_out[None, ...] if single_in else br_out
    is_b = i_sum_out[None, ...] if single_in else i_sum_out
    dedb_b = dedb_out[None, ...] if single_in else dedb_out
    dbdi_b = db_di_out[None, ...] if single_in else db_di_out

    def _fit_param(grad_bn: jax.Array, orig_arr: jax.Array, was_batched: bool) -> jax.Array:
        if single_in:
            res = grad_bn[0]
            return res.reshape(orig_arr.shape).astype(orig_arr.dtype)
        if was_batched:
            return grad_bn.astype(orig_arr.dtype)
        red = jnp.sum(grad_bn, axis=0)
        return red.reshape(orig_arr.shape).astype(orig_arr.dtype)

    dpos = (
        ((0.0 - g_vec[:, None, None]) * f_b)[0].astype(pos_v.dtype)
        if single_in
        else ((0.0 - g_vec[:, None, None]) * f_b).astype(pos_v.dtype)
    ) if p_pos else jnp.zeros_like(pos_v)
    dq = _fit_param(g_vec[:, None] * gq_b, q_v, b_q) if p_q else jnp.zeros_like(q_v)

    if p_rad or p_sr or p_sig or p_eps:
        gr_b, gsr_b, gsig_b, geps_b = _run_gbsa_tile_param_bwd_pallas(
            g_vec, pos_b, rad_b, sr_b, sig_b, eps_b, msk_b,
            br_b, is_b, dedb_b, dbdi_b, tl_v, bx_b, lm_b,
            has_box, dielectric_offset, probe_radius, surface_tension,
            obc_alpha, obc_beta, obc_gamma, cutoff, born_cutoff, switch_dist,
            softcore_alpha, softcore_exponent, include_gb, include_ace, include_lj,
            p_rad, p_sr, p_sig, p_eps, interpret,
        )
        dr = _fit_param(gr_b, rad_v, b_rad) if p_rad else jnp.zeros_like(rad_v)
        dsr = _fit_param(gsr_b, sr_v, b_sr) if p_sr else jnp.zeros_like(sr_v)
        dsig = _fit_param(gsig_b, sig_v, b_sig) if p_sig else jnp.zeros_like(sig_v)
        deps = _fit_param(geps_b, eps_v, b_eps) if p_eps else jnp.zeros_like(eps_v)
    else:
        dr = jnp.zeros_like(rad_v)
        dsr = jnp.zeros_like(sr_v)
        dsig = jnp.zeros_like(sig_v)
        deps = jnp.zeros_like(eps_v)

    return dpos, dq, dr, dsr, dsig, deps, z_msk, z_tl, z_bx, z_lm


_gbsa_tile_energy_diff.defvjp(_gbsa_tile_energy_fwd, _gbsa_tile_energy_bwd, symbolic_zeros=True)


def pallas_gbsa_tile(  # noqa: PLR0913
    positions: jax.Array | np.ndarray,
    charges: jax.Array | np.ndarray,
    radii: jax.Array | np.ndarray,
    scaled_radii: jax.Array | np.ndarray,
    sigmas: jax.Array | np.ndarray | None = None,
    epsilons: jax.Array | np.ndarray | None = None,
    tile_list: GBSATilePairList | None = None,
    *,
    atom_mask: jax.Array | np.ndarray | None = None,
    idx_12_13: jax.Array | np.ndarray | None = None,
    idx_14: jax.Array | np.ndarray | None = None,
    excl_indices: jax.Array | np.ndarray | None = None,
    excl_scales_vdw: jax.Array | np.ndarray | None = None,
    excl_scales_elec: jax.Array | np.ndarray | None = None,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.83333333,
    box: jax.Array | np.ndarray | None = None,
    softcore_lambda: float | jax.Array | np.ndarray = 1.0,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    coulomb_dielectric: float | None = None,
    dielectric_offset: float = 0.09,
    probe_radius: float = constants.PROBE_RADIUS,
    surface_tension: float = SURFACE_TENSION,
    obc_alpha: float = ALPHA_OBC,
    obc_beta: float = BETA_OBC,
    obc_gamma: float = GAMMA_OBC,
    kappa: float = 0.0,
    kappa_scale: float = GB_KAPPA_SCALE,
    salt_concentration_m: float = 0.0,
    temperature_k: float = 298.15,
    cutoff: float | None = None,
    born_cutoff: float | None = None,
    switch_dist: float | None = None,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 1,
    coulomb_constant: float = GB_COULOMB_CONSTANT,
    include_gb: bool = True,
    include_ace: bool = True,
    include_coulomb: bool = True,
    include_lj: bool = True,
    return_forces: bool = False,
    interpret: bool | None = None,
) -> Any:
    """2-pass 32x32 symmetric tile-pair OBC2 GBSA + ACE + Coulomb/LJ megakernel (MK-67, MK-73).

    Returns scalar/``(B,)`` energy when ``return_forces=False`` (supporting ``jax.grad`` wrt
    ``positions, charges, radii, scaled_radii, sigmas, epsilons``), or
    ``(energy, forces, born_radii)`` when ``return_forces=True``.
    """
    pos_arr = jnp.asarray(positions)
    single_in = pos_arr.ndim == 2
    B = 1 if single_in else int(pos_arr.shape[0])
    N = int(pos_arr.shape[-2])
    dtype = pos_arr.dtype

    sig_arr = jnp.ones((N,), dtype=dtype) if sigmas is None else jnp.asarray(sigmas, dtype=dtype)
    eps_arr = jnp.zeros((N,), dtype=dtype) if epsilons is None else jnp.asarray(epsilons, dtype=dtype)
    do_lj = bool(include_lj and epsilons is not None)

    msk_arr = (
        jnp.ones((N,) if single_in else (B, N), dtype=jnp.int32)
        if atom_mask is None
        else jnp.asarray(atom_mask, dtype=jnp.int32)
    )
    eff_kappa = (
        float(kappa)
        if float(kappa) > 0.0
        else (
            compute_debye_kappa(salt_concentration_m, solvent_dielectric, temperature_k)
            if float(salt_concentration_m) > 0.0
            else 0.0
        )
    )
    c_diel = float(solute_dielectric if coulomb_dielectric is None else coulomb_dielectric)

    if tile_list is None:
        tile_list = build_gbsa_tile_pair_list(
            pos_arr,
            atom_mask=msk_arr,
            idx_12_13=idx_12_13,
            idx_14=idx_14,
            excl_indices=excl_indices,
            excl_scales_vdw=excl_scales_vdw,
            excl_scales_elec=excl_scales_elec,
            scale_14_vdw=scale_14_vdw,
            scale_14_elec=scale_14_elec,
            box=box,
            cutoff=cutoff,
            born_cutoff=born_cutoff,
        )

    if in_export_context():
        return gbsa_tile_reference(
            pos_arr, charges, radii, scaled_radii, sig_arr, eps_arr, tile_list,
            atom_mask=msk_arr, box=box, softcore_lambda=softcore_lambda,
            solvent_dielectric=solvent_dielectric, solute_dielectric=solute_dielectric,
            coulomb_dielectric=c_diel, dielectric_offset=dielectric_offset,
            probe_radius=probe_radius, surface_tension=surface_tension,
            obc_alpha=obc_alpha, obc_beta=obc_beta, obc_gamma=obc_gamma,
            kappa=eff_kappa, kappa_scale=kappa_scale, cutoff=cutoff, born_cutoff=born_cutoff,
            switch_dist=switch_dist, softcore_alpha=softcore_alpha,
            softcore_exponent=softcore_exponent, coulomb_constant=coulomb_constant,
            include_gb=include_gb, include_ace=include_ace,
            include_coulomb=include_coulomb, include_lj=do_lj, return_forces=return_forces,
        )

    use_interpret = _resolve_interpret(interpret)
    box6_b, has_box = _normalize_box6_batch(box, B, dtype)
    lam_arr = jnp.asarray(softcore_lambda, dtype=dtype)

    static_cfg = (
        has_box,
        float(solvent_dielectric),
        float(solute_dielectric),
        float(c_diel),
        float(dielectric_offset),
        float(probe_radius),
        float(surface_tension),
        float(obc_alpha),
        float(obc_beta),
        float(obc_gamma),
        float(eff_kappa),
        float(kappa_scale),
        cutoff,
        born_cutoff,
        switch_dist,
        float(softcore_alpha),
        int(softcore_exponent),
        float(coulomb_constant),
        bool(include_gb),
        bool(include_ace),
        bool(include_coulomb),
        bool(do_lj),
        bool(return_forces),
        bool(use_interpret),
    )

    if not return_forces:
        return _gbsa_tile_energy_diff(
            pos_arr,
            jnp.asarray(charges, dtype=dtype),
            jnp.asarray(radii, dtype=dtype),
            jnp.asarray(scaled_radii, dtype=dtype),
            sig_arr,
            eps_arr,
            msk_arr,
            tile_list,
            box6_b[0] if single_in else box6_b,
            lam_arr,
            static_cfg,
        )

    op = _make_gbsa_vmap_op(static_cfg)
    e_out, f_out, br_out, _, _, _, _ = op(
        pos_arr,
        jnp.asarray(charges, dtype=dtype),
        jnp.asarray(radii, dtype=dtype),
        jnp.asarray(scaled_radii, dtype=dtype),
        sig_arr,
        eps_arr,
        msk_arr,
        tile_list,
        box6_b[0] if single_in else box6_b,
        lam_arr,
    )
    return e_out, f_out, br_out


def pallas_gbsa_tile_energy_and_forces(
    positions: jax.Array | np.ndarray,
    charges: jax.Array | np.ndarray,
    radii: jax.Array | np.ndarray,
    scaled_radii: jax.Array | np.ndarray,
    sigmas: jax.Array | np.ndarray | None = None,
    epsilons: jax.Array | np.ndarray | None = None,
    tile_list: GBSATilePairList | None = None,
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """Convenience wrapper returning ``(energy, forces)`` from ``pallas_gbsa_tile``."""
    kwargs["return_forces"] = True
    energy, forces, _born_radii = pallas_gbsa_tile(
        positions, charges, radii, scaled_radii, sigmas, epsilons, tile_list, **kwargs
    )
    return energy, forces
