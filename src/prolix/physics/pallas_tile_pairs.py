"""Morton 32x32 tile-pair skin-rebuild, fused bonded+tile-pair megakernel, and param VJP (MK-65/71/72/73)."""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.scipy.special import erf, erfc

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.optimization import _lj_switch_weight_and_deriv
from prolix.physics.pallas_bonded import (
    _add3,
    _cross3,
    _dot3,
    _eval_cmap_horner_cell,
    _min_image3,
    _pl_atomic_add_1d,
    _scale3,
    _sub3,
    _torsion_angle_and_atom_grads_vec3,
    _triton_erf,
    _triton_erfc,
    _triton_gpu_available,
)

COULOMB_CONSTANT: float = 332.06371329919216
_NB_BLOCK_N: int = 32
_BLOCK_B: int = 64


def _tile_pairs_compiler_params(grid_blocks: int = 0) -> plt.CompilerParams:
    """MK-72: Optimal 32x32 tile-pair / megakernel Triton launch parameters.

    Forces ``num_warps=4, num_stages=1`` across Ada/Hopper/Blackwell so Hopper
    ``sm_90`` never software-pipelines pointer-mode ``fori_loop`` bodies into
    double registers (unless overridden via ``PROLIX_PALLAS_NUM_WARPS/STAGES``).
    """
    env_w = os.environ.get("PROLIX_PALLAS_NUM_WARPS", "").strip()
    env_s = os.environ.get("PROLIX_PALLAS_NUM_STAGES", "").strip()
    if env_w or env_s:
        return resolve_pallas_compiler_params("megakernel", grid_blocks=grid_blocks)
    return plt.CompilerParams(num_warps=4, num_stages=1)


class DedupExclEdges(NamedTuple):
    """Precomputed deduplicated undirected excluded pairs ``(u < v)`` for a topology."""

    full_u: jax.Array
    full_v: jax.Array
    n_full: int
    excl14_u: jax.Array
    excl14_v: jax.Array
    n_14: int
    scale_14_vdw: float
    scale_14_elec: float


class MortonTilePairState:
    """Dynamic GPU-resident Morton 32x32 tile-pair neighbor state for ``EnsemblePlan``."""

    def __init__(
        self,
        perm: jax.Array,
        inv_perm: jax.Array,
        q_m: jax.Array,
        sig_m: jax.Array,
        eps_m: jax.Array,
        mask_m: jax.Array,
        pair_ti: jax.Array,
        pair_tj: jax.Array,
        excl_full_bits: jax.Array,
        excl_14_bits: jax.Array,
        n_valid_pairs: jax.Array,
        ref_positions: jax.Array,
        *,
        n_act: int,
        n_act_blocks: int,
        max_pairs: int,
    ) -> None:
        self.perm = perm
        self.inv_perm = inv_perm
        self.q_m = q_m
        self.sig_m = sig_m
        self.eps_m = eps_m
        self.mask_m = mask_m
        self.pair_ti = pair_ti
        self.pair_tj = pair_tj
        self.excl_full_bits = excl_full_bits
        self.excl_14_bits = excl_14_bits
        self.n_valid_pairs = n_valid_pairs
        self.ref_positions = ref_positions
        self.n_act = n_act
        self.n_act_blocks = n_act_blocks
        self.max_pairs = max_pairs


def _morton_tp_flatten(st: MortonTilePairState):
    children = (
        st.perm,
        st.inv_perm,
        st.q_m,
        st.sig_m,
        st.eps_m,
        st.mask_m,
        st.pair_ti,
        st.pair_tj,
        st.excl_full_bits,
        st.excl_14_bits,
        st.n_valid_pairs,
        st.ref_positions,
    )
    return children, (st.n_act, st.n_act_blocks, st.max_pairs)


def _morton_tp_unflatten(aux, children) -> MortonTilePairState:
    n_act, n_act_blocks, max_pairs = aux
    return MortonTilePairState(*children, n_act=n_act, n_act_blocks=n_act_blocks, max_pairs=max_pairs)


jax.tree_util.register_pytree_node(MortonTilePairState, _morton_tp_flatten, _morton_tp_unflatten)


def estimate_max_tile_pairs(n_act_blocks: int, cutoff: float = 9.0, skin: float = 2.0) -> int:
    """Conservative upper-triangular 32x32 tile-pair capacity per replica."""
    nb = max(1, int(n_act_blocks))
    upper_all = nb * (nb + 1) // 2
    r_eff = max(float(cutoff) + float(skin), 6.0)
    # ~0.10 atoms/A^3 -> 1 tile ~ 320 A^3 (~6.8 A); sphere of (r_eff + 7.5 A)
    est_per_tile = int(max(96.0, 0.5 * (4.189 * ((r_eff + 7.5) ** 3) / 320.0) * 2.8))
    cap = max(2048, nb * est_per_tile)
    cap = ((min(upper_all, cap) + 63) // 64) * 64
    return max(64, cap)


def precompute_dedup_excl_edges(
    excl_indices: np.ndarray | jax.Array | None,
    excl_scales_vdw: np.ndarray | jax.Array | None,
    excl_scales_elec: np.ndarray | jax.Array | None,
    n_act: int,
    *,
    default_14_vdw: float = 0.5,
    default_14_elec: float = 1.0 / 1.2,
) -> DedupExclEdges:
    """Extract deduplicated undirected ``u < v`` full & 1-4 excluded edges on host."""
    if excl_indices is None:
        z = jnp.zeros((1,), dtype=jnp.int32)
        return DedupExclEdges(z, z, 0, z, z, 0, float(default_14_vdw), float(default_14_elec))

    idx_np = np.asarray(excl_indices, dtype=np.int32)
    if idx_np.ndim == 3:
        idx_np = idx_np[0]
    n_rows = min(int(n_act), idx_np.shape[0])
    idx_np = idx_np[:n_rows]
    vdw_np = (
        np.asarray(excl_scales_vdw[0] if np.ndim(excl_scales_vdw) == 3 else excl_scales_vdw, dtype=np.float32)[:n_rows]
        if excl_scales_vdw is not None
        else np.zeros_like(idx_np, dtype=np.float32)
    )
    elec_np = (
        np.asarray(excl_scales_elec[0] if np.ndim(excl_scales_elec) == 3 else excl_scales_elec, dtype=np.float32)[:n_rows]
        if excl_scales_elec is not None
        else np.zeros_like(idx_np, dtype=np.float32)
    )

    rows = np.broadcast_to(np.arange(n_rows, dtype=np.int32)[:, None], idx_np.shape)
    valid = (idx_np >= 0) & (idx_np < n_act) & (idx_np != rows)
    u_raw = rows[valid]
    v_raw = idx_np[valid]
    sv_raw = vdw_np[valid]
    se_raw = elec_np[valid]
    keep_ut = u_raw < v_raw
    # Also catch asymmetric entries where only (u > v) was populated
    u_min = np.minimum(u_raw, v_raw)
    v_max = np.maximum(u_raw, v_raw)
    is_non_unit = (sv_raw < 1.0 - 1e-6) | (se_raw < 1.0 - 1e-6)
    u_min = u_min[is_non_unit]
    v_max = v_max[is_non_unit]
    sv_sub = sv_raw[is_non_unit]
    se_sub = se_raw[is_non_unit]
    del keep_ut

    if u_min.size == 0:
        z = jnp.zeros((1,), dtype=jnp.int32)
        return DedupExclEdges(z, z, 0, z, z, 0, float(default_14_vdw), float(default_14_elec))

    keys = u_min.astype(np.int64) * np.int64(max(n_act, 1)) + v_max.astype(np.int64)
    _, uniq_idx = np.unique(keys, return_index=True)
    u_u = u_min[uniq_idx]
    v_u = v_max[uniq_idx]
    sv_u = sv_sub[uniq_idx]
    se_u = se_sub[uniq_idx]

    is_full = (sv_u <= 1e-6) & (se_u <= 1e-6)
    is_14 = ~is_full
    s14_vdw = float(np.median(sv_u[is_14])) if np.any(is_14) else float(default_14_vdw)
    s14_elec = float(np.median(se_u[is_14])) if np.any(is_14) else float(default_14_elec)

    fu = jnp.asarray(u_u[is_full] if np.any(is_full) else np.zeros((1,), dtype=np.int32), dtype=jnp.int32)
    fv = jnp.asarray(v_u[is_full] if np.any(is_full) else np.zeros((1,), dtype=np.int32), dtype=jnp.int32)
    u14 = jnp.asarray(u_u[is_14] if np.any(is_14) else np.zeros((1,), dtype=np.int32), dtype=jnp.int32)
    v14 = jnp.asarray(v_u[is_14] if np.any(is_14) else np.zeros((1,), dtype=np.int32), dtype=jnp.int32)
    return DedupExclEdges(fu, fv, int(np.sum(is_full)), u14, v14, int(np.sum(is_14)), s14_vdw, s14_elec)


def _spread_bits_10(v: jax.Array) -> jax.Array:
    x = v & jnp.int32(0x3FF)
    x = (x | (x << 16)) & jnp.int32(0x030000FF)
    x = (x | (x << 8)) & jnp.int32(0x0300F00F)
    x = (x | (x << 4)) & jnp.int32(0x030C30C3)
    x = (x | (x << 2)) & jnp.int32(0x09249249)
    return x


def _morton3d_encode_jax(pos: jax.Array, box: jax.Array, mask_bool: jax.Array) -> jax.Array:
    """Compute 30-bit 3D Z-order keys for ``(B, N_pad, 3)``; inactive atoms get ``0x7FFFFFFF``."""
    safe_box = jnp.where(box > 1e-4, box, jnp.ones_like(box))
    pos_w = jnp.mod(pos, safe_box[:, None, :])
    scaled = jnp.floor((pos_w / safe_box[:, None, :]) * 1024.0).astype(jnp.int32)
    q = jnp.clip(scaled, 0, 1023)
    code = _spread_bits_10(q[..., 0]) | (_spread_bits_10(q[..., 1]) << 1) | (_spread_bits_10(q[..., 2]) << 2)
    return jnp.where(mask_bool, code, jnp.int32(0x7FFFFFFF))


def _compute_tile_aabbs(
    pos_m: jax.Array,
    mask_m: jax.Array,
    safe_box: jax.Array,
    n_act_blocks: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute tight PBC-unwrapped 32-atom tile AABBs ``(center, half_ext, has_tile)``."""
    B = pos_m.shape[0]
    box_b = safe_box[:, None, None, :]
    pos_t = jnp.mod(pos_m.reshape(B, n_act_blocks, _NB_BLOCK_N, 3), box_b)
    msk_t = (mask_m.reshape(B, n_act_blocks, _NB_BLOCK_N) > 0)
    lane = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)[None, None, :]
    first_lane = jnp.min(jnp.where(msk_t, lane, _NB_BLOCK_N), axis=2, keepdims=True)
    is_first = lane == jnp.minimum(first_lane, _NB_BLOCK_N - 1)
    ref_p = jnp.sum(jnp.where(is_first[..., None], pos_t, 0.0), axis=2, keepdims=True)
    dr = pos_t - ref_p
    unwrapped = ref_p + (dr - box_b * jnp.round(dr / box_b))
    pts = jnp.where(msk_t[..., None], unwrapped, ref_p)
    lo = jnp.min(pts, axis=2)
    hi = jnp.max(pts, axis=2)
    center = jnp.mod(0.5 * (lo + hi), safe_box[:, None, :])
    half_ext = 0.5 * (hi - lo)
    has_tile = jnp.any(msk_t, axis=2).astype(jnp.int32)
    return center, half_ext, has_tile


def _build_tile_pairs_jax_fallback(
    center: jax.Array,
    half_ext: jax.Array,
    has_tile: jax.Array,
    safe_box: jax.Array,
    inv_perm: jax.Array,
    dedup_excl: DedupExclEdges,
    r_eff_sq: float,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure-JAX tile-pair & bitmask builder for CPU ``interpret=True`` and StableHLO export."""
    B, nb, _ = center.shape

    def _one_rep(c_b, h_b, has_b, box_b, inv_b):
        dc = jnp.abs(c_b[:, None, :] - c_b[None, :, :])
        dc = jnp.minimum(dc, box_b[None, None, :] - dc)
        gap = jnp.maximum(0.0, dc - (h_b[:, None, :] + h_b[None, :, :]))
        dist_sq = jnp.sum(gap * gap, axis=-1)
        idx_i = jnp.arange(nb, dtype=jnp.int32)
        valid = (
            (has_b[:, None] > 0)
            & (has_b[None, :] > 0)
            & (idx_i[:, None] <= idx_i[None, :])
            & (dist_sq <= r_eff_sq)
        )
        flat_v = valid.reshape(-1)
        n_val = jnp.minimum(
            jnp.sum(flat_v.astype(jnp.int32), dtype=jnp.int32),
            jnp.int32(max_pairs),
        ).astype(jnp.int32)
        order_raw = jnp.argsort(~flat_v)[:max_pairs].astype(jnp.int32)
        pad_ord = max_pairs - int(order_raw.shape[0])
        order = (
            jnp.pad(order_raw, (0, pad_ord), constant_values=0)
            if pad_ord > 0
            else order_raw
        )
        p_idx = jnp.arange(max_pairs, dtype=jnp.int32)
        keep = p_idx < n_val
        ti = jnp.where(keep, order // jnp.int32(nb), jnp.int32(-1)).astype(jnp.int32)
        tj = jnp.where(keep, order % jnp.int32(nb), jnp.int32(-1)).astype(jnp.int32)

        pair_lookup = jnp.full((nb * nb,), -1, dtype=jnp.int32)
        safe_flat = jnp.where(keep, ti * jnp.int32(nb) + tj, jnp.int32(0)).astype(jnp.int32)
        pair_lookup = pair_lookup.at[safe_flat].set(jnp.where(keep, p_idx, jnp.int32(-1)))
        # Ensure slot 0 wasn't clobbered if pair (0, 0) was valid at p_idx == 0
        pair_lookup = jnp.where(
            (jnp.arange(nb * nb, dtype=jnp.int32) == (ti[0] * jnp.int32(nb) + tj[0])) & (n_val > 0),
            jnp.int32(0),
            pair_lookup,
        )

        def _stamp(u_arr, v_arr, count):
            bits = jnp.zeros((max_pairs, _NB_BLOCK_N), dtype=jnp.int32)
            if count <= 0:
                return bits
            mu = inv_b[u_arr]
            mv = inv_b[v_arr]
            m_lo = jnp.minimum(mu, mv)
            m_hi = jnp.maximum(mu, mv)
            eti = m_lo >> 5
            eli = m_lo & 31
            etj = m_hi >> 5
            elj = m_hi & 31
            p_slot = pair_lookup[eti * nb + etj]
            ok = p_slot >= 0
            sp = jnp.where(ok, p_slot, 0)
            b_val = jnp.where(ok, jnp.int32(1) << elj, jnp.int32(0))
            return bits.at[sp, eli].add(b_val)

        ef = _stamp(dedup_excl.full_u, dedup_excl.full_v, dedup_excl.n_full)
        e14 = _stamp(dedup_excl.excl14_u, dedup_excl.excl14_v, dedup_excl.n_14)
        return ti, tj, ef, e14, n_val

    return jax.vmap(_one_rep)(center, half_ext, has_tile, safe_box, inv_perm)


def _stamp_excl_bitmasks_gpu(
    B: int,
    nb: int,
    max_pairs: int,
    n_act_pad: int,
    inv_perm: jax.Array,
    row_start: jax.Array,
    row_len: jax.Array,
    pair_tj: jax.Array,
    dedup_excl: DedupExclEdges,
) -> tuple[jax.Array, jax.Array]:
    """O(N_atoms) Pallas GPU kernel stamping 32x32 exclusion bitmasks into tile pairs."""
    ef_zero = jnp.zeros((B * max_pairs * _NB_BLOCK_N,), dtype=jnp.int32)
    e14_zero = jnp.zeros((B * max_pairs * _NB_BLOCK_N,), dtype=jnp.int32)
    inv_flat = inv_perm.reshape(-1)
    rs_flat = row_start.reshape(-1)
    rl_flat = row_len.reshape(-1)
    tj_flat = pair_tj.reshape(-1)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)

    def _run_stamp(u_arr: jax.Array, v_arr: jax.Array, n_edges: int, buf_zero: jax.Array) -> jax.Array:
        if n_edges <= 0:
            return buf_zero.reshape(B, max_pairs, _NB_BLOCK_N)
        blk_e = 64
        n_eblk = (n_edges + blk_e - 1) // blk_e

        def _stamp_kernel(u_ref, v_ref, inv_ref, rs_ref, rl_ref, tj_ref, out_in_ref, out_ref):
            del out_in_ref
            b_id = pl.program_id(0)
            pid = pl.program_id(1)
            offs = pid * blk_e + jnp.arange(blk_e, dtype=jnp.int32)
            in_e = offs < n_edges
            u_idx = _pl_load(u_ref, (offs,), mask=in_e, other=0)
            v_idx = _pl_load(v_ref, (offs,), mask=in_e, other=0)
            inv_base = b_id * n_act_pad
            mu = _pl_load(inv_ref, (inv_base + u_idx,), mask=in_e, other=0)
            mv = _pl_load(inv_ref, (inv_base + v_idx,), mask=in_e, other=0)
            m_lo = jnp.minimum(mu, mv)
            m_hi = jnp.maximum(mu, mv)
            eti = m_lo >> 5
            eli = m_lo & 31
            etj = m_hi >> 5
            elj = m_hi & 31

            rs = _pl_load(rs_ref, (b_id * nb + eti,), mask=in_e, other=0)
            rl = _pl_load(rl_ref, (b_id * nb + eti,), mask=in_e, other=0)
            max_rl = jnp.max(jnp.where(in_e, rl, jnp.int32(0))).astype(jnp.int32)
            init_slot = jnp.full((blk_e,), -1, dtype=jnp.int32)

            def _scan_step(k, found_slot):
                still = in_e & (k < rl) & (found_slot < 0)
                cand_idx = (rs + k).astype(jnp.int32)
                in_cap = still & (cand_idx < max_pairs)
                cand_tj = _pl_load(tj_ref, (b_id * max_pairs + cand_idx,), mask=in_cap, other=-2)
                return jnp.where(in_cap & (cand_tj == etj), cand_idx, found_slot).astype(jnp.int32)

            slot = jax.lax.fori_loop(jnp.int32(0), max_rl, _scan_step, init_slot)
            ok = in_e & (slot >= 0) & (slot < max_pairs)
            safe_slot = jnp.where(ok, slot, jnp.int32(0)).astype(jnp.int32)
            bit_val = jnp.where(ok, jnp.int32(1) << elj, jnp.int32(0))
            dst = ((b_id * max_pairs + safe_slot) * _NB_BLOCK_N + eli).astype(jnp.int32)
            plt.atomic_or(out_ref, (dst,), bit_val, mask=ok)

        res = pl.pallas_call(
            _stamp_kernel,
            out_shape=jax.ShapeDtypeStruct((B * max_pairs * _NB_BLOCK_N,), jnp.int32),
            in_specs=[any_spec] * 7,
            out_specs=any_spec,
            grid=(B, n_eblk),
            compiler_params=plt.CompilerParams(num_warps=2, num_stages=1),
            input_output_aliases={6: 0},
            name="morton_stamp_excl_bits",
        )(u_arr, v_arr, inv_flat, rs_flat, rl_flat, tj_flat, buf_zero)
        return res.reshape(B, max_pairs, _NB_BLOCK_N)

    ef = _run_stamp(dedup_excl.full_u, dedup_excl.full_v, dedup_excl.n_full, ef_zero)
    e14 = _run_stamp(dedup_excl.excl14_u, dedup_excl.excl14_v, dedup_excl.n_14, e14_zero)
    return ef, e14


def _discover_pairs_gpu(
    B: int,
    nb: int,
    max_pairs: int,
    center: jax.Array,
    half_ext: jax.Array,
    has_tile: jax.Array,
    safe_box: jax.Array,
    r_eff: float,
    host_min_box: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Two-pass GPU Pallas tile-pair discovery returning ``(ti, tj, row_start, row_len, n_valid)``."""
    r_eff_sq = float(r_eff * r_eff)
    cx_f = center[:, :, 0].reshape(-1).astype(jnp.float32)
    cy_f = center[:, :, 1].reshape(-1).astype(jnp.float32)
    cz_f = center[:, :, 2].reshape(-1).astype(jnp.float32)
    hx_f = half_ext[:, :, 0].reshape(-1).astype(jnp.float32)
    hy_f = half_ext[:, :, 1].reshape(-1).astype(jnp.float32)
    hz_f = half_ext[:, :, 2].reshape(-1).astype(jnp.float32)
    has_f = has_tile.reshape(-1).astype(jnp.int32)
    box_f = safe_box.reshape(-1).astype(jnp.float32)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    chunk_w = 64
    n_chunks = (nb + chunk_w - 1) // chunk_w

    use_cell_grid = (nb > 1024) and (host_min_box >= 2.0 * (r_eff + 3.0))
    g_cell = int(np.clip(int(host_min_box // max(r_eff + 4.5, 12.0)), 4, 20)) if use_cell_grid else 1
    n_cells = g_cell * g_cell * g_cell

    if use_cell_grid:
        cell_len = safe_box / float(g_cell)
        h_lim = 0.5 * jnp.maximum(cell_len - float(r_eff), 0.0)
        is_wide = (has_tile > 0) & jnp.any(half_ext > h_lim[:, None, :], axis=-1)
        is_narrow = (has_tile > 0) & (~is_wide)
        g_coord = jnp.clip(
            jnp.floor(center / jnp.maximum(cell_len[:, None, :], 1e-4)).astype(jnp.int32),
            0,
            g_cell - 1,
        )
        cid = ((g_coord[..., 0] * g_cell + g_coord[..., 1]) * g_cell + g_coord[..., 2]).astype(jnp.int32)
        cid_key = jnp.where(is_narrow, cid, jnp.int32(n_cells)).astype(jnp.int32)
        narrow_sorted = jnp.argsort(cid_key, axis=-1).astype(jnp.int32)
        sorted_cids = jnp.take_along_axis(cid_key, narrow_sorted, axis=-1)
        q_cids = jnp.arange(n_cells + 1, dtype=jnp.int32)
        bounds = jax.vmap(lambda sc: jnp.searchsorted(sc, q_cids, side="left"))(sorted_cids).astype(jnp.int32)
        c_start = bounds[:, :-1].reshape(-1)
        c_end = bounds[:, 1:].reshape(-1)
        wide_sorted = jnp.argsort(~is_wide, axis=-1).astype(jnp.int32).reshape(-1)
        n_wide = jnp.sum(is_wide.astype(jnp.int32), axis=-1, dtype=jnp.int32).astype(jnp.int32)
        is_wide_flat = is_wide.astype(jnp.int32).reshape(-1)
        cid_flat = cid.reshape(-1)
        ns_flat = narrow_sorted.reshape(-1)
    else:
        c_start = jnp.zeros((1,), dtype=jnp.int32)
        c_end = jnp.zeros((1,), dtype=jnp.int32)
        wide_sorted = jnp.zeros((1,), dtype=jnp.int32)
        n_wide = jnp.zeros((B,), dtype=jnp.int32)
        is_wide_flat = jnp.zeros((1,), dtype=jnp.int32)
        cid_flat = jnp.zeros((1,), dtype=jnp.int32)
        ns_flat = jnp.zeros((1,), dtype=jnp.int32)

    def _run_pass(emit: bool, row_start_in: jax.Array | None = None):
        rs_arg = row_start_in.reshape(-1) if row_start_in is not None else jnp.zeros((B * nb,), dtype=jnp.int32)
        ti_init = jnp.full((B * max_pairs,), -1, dtype=jnp.int32) if emit else jnp.zeros((1,), dtype=jnp.int32)
        tj_init = jnp.full((B * max_pairs,), -1, dtype=jnp.int32) if emit else jnp.zeros((1,), dtype=jnp.int32)

        def _pair_kernel(  # noqa: PLR0913
            cx_r, cy_r, cz_r, hx_r, hy_r, hz_r, has_r, box_r,
            cs_r, ce_r, ns_r, ws_r, nw_r, iw_r, cid_r, rs_r, ti_in_r, tj_in_r, *out_refs
        ):
            del ti_in_r, tj_in_r
            b_id = pl.program_id(0)
            ti = pl.program_id(1)
            base_t = b_id * nb
            z0 = jnp.zeros((1,), dtype=jnp.int32)
            has_i = _pl_load(has_r, (base_t + ti + z0,))[0] > 0
            bx = _pl_load(box_r, (b_id * 3 + z0,))[0]
            by = _pl_load(box_r, (b_id * 3 + jnp.ones((1,), dtype=jnp.int32),))[0]
            bz = _pl_load(box_r, (b_id * 3 + jnp.full((1,), 2, dtype=jnp.int32),))[0]
            cxi = _pl_load(cx_r, (base_t + ti + z0,))[0]
            cyi = _pl_load(cy_r, (base_t + ti + z0,))[0]
            czi = _pl_load(cz_r, (base_t + ti + z0,))[0]
            hxi = _pl_load(hx_r, (base_t + ti + z0,))[0]
            hyi = _pl_load(hy_r, (base_t + ti + z0,))[0]
            hzi = _pl_load(hz_r, (base_t + ti + z0,))[0]
            rs_i = _pl_load(rs_r, (base_t + ti + z0,))[0]
            r2_lim = jnp.asarray(r_eff_sq, dtype=jnp.float32)
            zf = jnp.asarray(0.0, dtype=jnp.float32)

            def _test_vec(tj_v, in_m):
                ok_j = in_m & (tj_v >= ti) & (tj_v < nb)
                stj = jnp.where(ok_j, tj_v, 0)
                hj = _pl_load(has_r, (base_t + stj,), mask=ok_j, other=0) > 0
                cxj = _pl_load(cx_r, (base_t + stj,), mask=ok_j, other=zf)
                cyj = _pl_load(cy_r, (base_t + stj,), mask=ok_j, other=zf)
                czj = _pl_load(cz_r, (base_t + stj,), mask=ok_j, other=zf)
                hxj = _pl_load(hx_r, (base_t + stj,), mask=ok_j, other=zf)
                hyj = _pl_load(hy_r, (base_t + stj,), mask=ok_j, other=zf)
                hzj = _pl_load(hz_r, (base_t + stj,), mask=ok_j, other=zf)
                dcx = jnp.abs(cxi - cxj)
                dcx = jnp.minimum(dcx, bx - dcx)
                dcy = jnp.abs(cyi - cyj)
                dcy = jnp.minimum(dcy, by - dcy)
                dcz = jnp.abs(czi - czj)
                dcz = jnp.minimum(dcz, bz - dcz)
                gx = jnp.maximum(zf, dcx - (hxi + hxj))
                gy = jnp.maximum(zf, dcy - (hyi + hyj))
                gz = jnp.maximum(zf, dcz - (hzi + hzj))
                return ok_j & hj & ((gx * gx + gy * gy + gz * gz) <= r2_lim)

            def _store_vec(cur_cnt, tj_v, v_mask):
                n_add = jnp.sum(v_mask.astype(jnp.int32), dtype=jnp.int32).astype(jnp.int32)
                if emit:
                    loc = (jnp.cumsum(v_mask.astype(jnp.int32), dtype=jnp.int32) - jnp.int32(1)).astype(jnp.int32)
                    dst = (rs_i + cur_cnt + loc).astype(jnp.int32)
                    w_ok = v_mask & (dst < max_pairs)
                    s_dst = (b_id * max_pairs + jnp.where(w_ok, dst, jnp.int32(0))).astype(jnp.int32)
                    _pl_store(out_refs[0], (s_dst,), jnp.broadcast_to(ti, tj_v.shape).astype(jnp.int32), mask=w_ok)
                    _pl_store(out_refs[1], (s_dst,), tj_v.astype(jnp.int32), mask=w_ok)
                return (cur_cnt + n_add).astype(jnp.int32)

            def _full_scan(_):
                lanes = jnp.arange(chunk_w, dtype=jnp.int32)
                return jax.lax.fori_loop(
                    (ti // chunk_w).astype(jnp.int32),
                    jnp.int32(n_chunks),
                    lambda ck, acc: _store_vec(acc, ck * chunk_w + lanes, _test_vec(ck * chunk_w + lanes, (ck * chunk_w + lanes) < nb)),
                    (ti - ti).astype(jnp.int32),
                )

            def _active_ti(_):
                if not use_cell_grid:
                    return _full_scan(None)
                wide_i = _pl_load(iw_r, (base_t + ti + z0,))[0] > 0

                def _narrow_scan(_):
                    ci = _pl_load(cid_r, (base_t + ti + z0,))[0]
                    ix0 = ci // (g_cell * g_cell)
                    iy0 = (ci // g_cell) % g_cell
                    iz0 = ci % g_cell
                    lanes32 = jnp.arange(32, dtype=jnp.int32)

                    def _nbr_body(nb_k, acc0):
                        dx_c = (nb_k // 9) - 1
                        dy_c = ((nb_k // 3) % 3) - 1
                        dz_c = (nb_k % 3) - 1
                        nc = (
                            jnp.mod(ix0 + dx_c, g_cell) * (g_cell * g_cell)
                            + jnp.mod(iy0 + dy_c, g_cell) * g_cell
                            + jnp.mod(iz0 + dz_c, g_cell)
                        ).astype(jnp.int32)
                        s_ptr = _pl_load(cs_r, (b_id * n_cells + nc + z0,))[0]
                        e_ptr = _pl_load(ce_r, (b_id * n_cells + nc + z0,))[0]
                        n_in_c = (e_ptr - s_ptr).astype(jnp.int32)
                        n_c_blk = ((n_in_c + 31) // 32).astype(jnp.int32)

                        def _cell_chunk(cb, acc_c):
                            pos32 = (s_ptr + cb * 32 + lanes32).astype(jnp.int32)
                            in_c = pos32 < e_ptr
                            tj32 = _pl_load(ns_r, (base_t + jnp.where(in_c, pos32, jnp.int32(0)),), mask=in_c, other=-1)
                            return _store_vec(acc_c, tj32, _test_vec(tj32, in_c))

                        return jax.lax.fori_loop(jnp.int32(0), n_c_blk, _cell_chunk, acc0)

                    cnt_nb = jax.lax.fori_loop(
                        jnp.int32(0), jnp.int32(27), _nbr_body, (ti - ti).astype(jnp.int32)
                    )
                    nw_b = _pl_load(nw_r, (b_id + z0,))[0]
                    n_w_blk = ((nw_b + 31) // 32).astype(jnp.int32)

                    def _wide_chunk(wb, acc_w):
                        wpos = (wb * 32 + lanes32).astype(jnp.int32)
                        in_w = wpos < nw_b
                        tjw = _pl_load(ws_r, (base_t + jnp.where(in_w, wpos, jnp.int32(0)),), mask=in_w, other=-1)
                        return _store_vec(acc_w, tjw, _test_vec(tjw, in_w))

                    return jax.lax.fori_loop(jnp.int32(0), n_w_blk, _wide_chunk, cnt_nb)

                return jax.lax.cond(wide_i, _full_scan, _narrow_scan, None)

            cnt_out = jax.lax.cond(has_i, _active_ti, lambda z: z, (ti - ti).astype(jnp.int32))
            if not emit:
                _pl_store(out_refs[0], (base_t + ti + z0,), cnt_out[None])

        in_args = (
            cx_f, cy_f, cz_f, hx_f, hy_f, hz_f, has_f, box_f,
            c_start, c_end, ns_flat, wide_sorted, n_wide, is_wide_flat, cid_flat, rs_arg, ti_init, tj_init,
        )
        if not emit:
            rl = pl.pallas_call(
                _pair_kernel,
                out_shape=jax.ShapeDtypeStruct((B * nb,), jnp.int32),
                in_specs=[any_spec] * 18,
                out_specs=any_spec,
                grid=(B, nb),
                compiler_params=plt.CompilerParams(num_warps=2, num_stages=1),
                name="morton_count_tile_pairs",
            )(*in_args)
            return rl.reshape(B, nb)

        ti_out, tj_out = pl.pallas_call(
            _pair_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((B * max_pairs,), jnp.int32),
                jax.ShapeDtypeStruct((B * max_pairs,), jnp.int32),
            ],
            in_specs=[any_spec] * 18,
            out_specs=[any_spec, any_spec],
            grid=(B, nb),
            compiler_params=plt.CompilerParams(num_warps=2, num_stages=1),
            input_output_aliases={16: 0, 17: 1},
            name="morton_emit_tile_pairs",
        )(*in_args)
        return ti_out.reshape(B, max_pairs), tj_out.reshape(B, max_pairs)

    row_len = _run_pass(emit=False)
    row_end = jnp.cumsum(row_len, axis=1, dtype=jnp.int32).astype(jnp.int32)
    row_start = (row_end - row_len).astype(jnp.int32)
    n_valid = jnp.minimum(row_end[:, -1], jnp.int32(max_pairs)).astype(jnp.int32)
    pair_ti, pair_tj = _run_pass(emit=True, row_start_in=row_start)
    return pair_ti, pair_tj, row_start, row_len, n_valid


def pallas_build_morton_tile_pairs_gpu(
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    box: jax.Array,
    dedup_excl: DedupExclEdges,
    *,
    n_act: int,
    max_pairs: int,
    cutoff: float = 9.0,
    skin: float = 2.0,
    host_min_box: float = 30.0,
    interpret: bool = False,
) -> MortonTilePairState:
    """MK-52 / MK-65: GPU Morton sort + O(N_tiles) tile-pair & O(N) bitmask rebuild."""
    single_in = positions.ndim == 2
    pos_b = positions[None, ...] if single_in else positions
    B, n_atoms, _ = pos_b.shape
    q_b = jnp.broadcast_to(charges[None, :] if charges.ndim == 1 else charges, (B, n_atoms))
    sig_b = jnp.broadcast_to(sigmas[None, :] if sigmas.ndim == 1 else sigmas, (B, n_atoms))
    eps_b = jnp.broadcast_to(epsilons[None, :] if epsilons.ndim == 1 else epsilons, (B, n_atoms))
    msk_b = jnp.broadcast_to(atom_mask[None, :] if atom_mask.ndim == 1 else atom_mask, (B, n_atoms))
    box_arr = jnp.asarray(box, dtype=jnp.float32)
    if box_arr.ndim == 1:
        box_b = jnp.broadcast_to(box_arr[:3][None, :], (B, 3))
    elif box_arr.ndim == 2 and box_arr.shape == (3, 3):
        box_b = jnp.broadcast_to(jnp.diag(box_arr)[None, :], (B, 3))
    elif box_arr.ndim == 3:
        box_b = jax.vmap(jnp.diag)(box_arr)
    else:
        box_b = jnp.broadcast_to(box_arr[:, :3], (B, 3))
    safe_box = jnp.where(box_b > 1e-4, box_b, jnp.ones_like(box_b))

    n_act_eff = min(int(n_act), n_atoms)
    n_act_blocks = max(1, (n_act_eff + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    n_act_pad = n_act_blocks * _NB_BLOCK_N
    pad = n_act_pad - n_act_eff

    pos_f32 = pos_b[:, :n_act_eff, :].astype(jnp.float32)
    q_f32 = q_b[:, :n_act_eff].astype(jnp.float32)
    sig_f32 = sig_b[:, :n_act_eff].astype(jnp.float32)
    eps_f32 = eps_b[:, :n_act_eff].astype(jnp.float32)
    msk_bool = jnp.asarray(msk_b[:, :n_act_eff], dtype=jnp.bool_)
    if pad > 0:
        pos_f32 = jnp.pad(pos_f32, ((0, 0), (0, pad), (0, 0)))
        q_f32 = jnp.pad(q_f32, ((0, 0), (0, pad)))
        sig_f32 = jnp.pad(sig_f32, ((0, 0), (0, pad)), constant_values=1.0)
        eps_f32 = jnp.pad(eps_f32, ((0, 0), (0, pad)))
        msk_bool = jnp.pad(msk_bool, ((0, 0), (0, pad)), constant_values=False)

    codes = _morton3d_encode_jax(pos_f32, safe_box, msk_bool)
    perm = jnp.argsort(codes, axis=-1).astype(jnp.int32)
    inv_perm = jnp.argsort(perm, axis=-1).astype(jnp.int32)

    pos_m = jnp.take_along_axis(pos_f32, perm[:, :, None], axis=1)
    q_m = jnp.take_along_axis(q_f32, perm, axis=1)
    sig_m = jnp.take_along_axis(sig_f32, perm, axis=1)
    eps_m = jnp.take_along_axis(eps_f32, perm, axis=1)
    mask_m = jnp.take_along_axis(msk_bool.astype(jnp.int32), perm, axis=1)

    center, half_ext, has_tile = _compute_tile_aabbs(pos_m, mask_m, safe_box, n_act_blocks)
    r_eff = float(cutoff) + float(skin)

    use_gpu = (not interpret) and (not in_export_context()) and _triton_gpu_available()
    if use_gpu:
        pair_ti, pair_tj, row_start, row_len, n_valid = _discover_pairs_gpu(
            B, n_act_blocks, max_pairs, center, half_ext, has_tile, safe_box, r_eff, host_min_box
        )
        ef_bits, e14_bits = _stamp_excl_bitmasks_gpu(
            B, n_act_blocks, max_pairs, n_act_pad, inv_perm, row_start, row_len, pair_tj, dedup_excl
        )
    else:
        pair_ti, pair_tj, ef_bits, e14_bits, n_valid = _build_tile_pairs_jax_fallback(
            center, half_ext, has_tile, safe_box, inv_perm, dedup_excl, r_eff * r_eff, max_pairs
        )

    return MortonTilePairState(
        perm=perm,
        inv_perm=inv_perm,
        q_m=q_m,
        sig_m=sig_m,
        eps_m=eps_m,
        mask_m=mask_m,
        pair_ti=pair_ti,
        pair_tj=pair_tj,
        excl_full_bits=ef_bits,
        excl_14_bits=e14_bits,
        n_valid_pairs=n_valid,
        ref_positions=pos_b,
        n_act=n_act_eff,
        n_act_blocks=n_act_blocks,
        max_pairs=max_pairs,
    )


class FusedBondedBuffers(NamedTuple):
    """Pre-flattened SoA bonded topology & parameter buffers with MK-64 stride=0 deduplication."""

    b_i: jax.Array
    b_j: jax.Array
    b_r0: jax.Array
    b_k: jax.Array
    b_msk: jax.Array
    B_real: int
    s_bi: int
    s_bp: int
    s_bm: int
    a_i: jax.Array
    a_j: jax.Array
    a_k: jax.Array
    a_th0: jax.Array
    a_kval: jax.Array
    a_msk: jax.Array
    A_real: int
    s_ai: int
    s_ap: int
    s_am: int
    d_i: jax.Array
    d_j: jax.Array
    d_k: jax.Array
    d_l: jax.Array
    d_p0: jax.Array
    d_p1: jax.Array
    d_p2: jax.Array
    d_msk: jax.Array
    D_real: int
    s_di: int
    s_dp: int
    s_dm: int
    h_i: jax.Array
    h_j: jax.Array
    h_k: jax.Array
    h_l: jax.Array
    h_kval: jax.Array
    h_phi0: jax.Array
    h_msk: jax.Array
    H_real: int
    s_hi: int
    s_hp: int
    s_hm: int
    e_i: jax.Array
    e_j: jax.Array
    e_sig: jax.Array
    e_eps: jax.Array
    e_qp: jax.Array
    e_msk: jax.Array
    E_real: int
    s_ep: int
    s_es: int
    s_ee: int
    s_eq: int
    s_em: int
    c_0: jax.Array
    c_1: jax.Array
    c_2: jax.Array
    c_3: jax.Array
    c_4: jax.Array
    c_midx: jax.Array
    c_coeffs: jax.Array
    c_msk: jax.Array
    C_real: int
    G_cmap: int
    N_maps: int
    s_ct: int
    s_cmi: int
    s_cc: int
    s_cm: int


def _prep_term(arr: jax.Array | None, base_ndim: int, B: int, dtype) -> tuple[jax.Array, int, int]:
    """Normalize an optional bonded array to ``(1_or_B, K, ...)`` with stride ``0`` or ``K``."""
    if arr is None:
        z = jnp.zeros((1, 1), dtype=dtype)
        return z, 0, 0
    a = jnp.asarray(arr, dtype=dtype)
    if a.shape[1 if a.ndim > base_ndim else 0] == 0:
        z = jnp.zeros((1, 1), dtype=dtype)
        return z, 0, 0
    batched = a.ndim > base_ndim
    if not batched:
        a = a[None, ...]
    k_len = int(a.shape[1])
    stride = k_len if (batched and int(a.shape[0]) == B and B > 1) else 0
    if stride == 0 and int(a.shape[0]) > 1:
        a = a[:1]
    return a, k_len, stride


def _cat_bonded_pair(
    a0: jax.Array, a1: jax.Array, base_ndim: int, B: int, cat_ax: int
) -> jax.Array:
    """Concatenate two bonded arrays, broadcasting any unbatched partner when the other is batched."""
    if a0.ndim == base_ndim and a1.ndim == base_ndim:
        return jnp.concatenate([a0, a1], axis=cat_ax)
    b_dim = max(
        1,
        int(B),
        int(a0.shape[0]) if a0.ndim > base_ndim else 1,
        int(a1.shape[0]) if a1.ndim > base_ndim else 1,
    )
    if a0.ndim == base_ndim:
        a0 = jnp.broadcast_to(a0[None, ...], (b_dim, *a0.shape))
    elif int(a0.shape[0]) == 1 and b_dim > 1:
        a0 = jnp.broadcast_to(a0, (b_dim, *a0.shape[1:]))
    if a1.ndim == base_ndim:
        a1 = jnp.broadcast_to(a1[None, ...], (b_dim, *a1.shape))
    elif int(a1.shape[0]) == 1 and b_dim > 1:
        a1 = jnp.broadcast_to(a1, (b_dim, *a1.shape[1:]))
    return jnp.concatenate([a0, a1], axis=cat_ax)


def prepare_fused_bonded_buffers(  # noqa: PLR0913
    B: int,
    *,
    bonds: jax.Array | None = None,
    bond_params: jax.Array | None = None,
    bond_mask: jax.Array | None = None,
    urey_bradley_bonds: jax.Array | None = None,
    urey_bradley_params: jax.Array | None = None,
    urey_bradley_mask: jax.Array | None = None,
    angles: jax.Array | None = None,
    angle_params: jax.Array | None = None,
    angle_mask: jax.Array | None = None,
    dihedrals: jax.Array | None = None,
    dihedral_params: jax.Array | None = None,
    dihedral_mask: jax.Array | None = None,
    impropers: jax.Array | None = None,
    improper_params: jax.Array | None = None,
    improper_mask: jax.Array | None = None,
    improper_is_harmonic: bool = False,
    exception_pairs: jax.Array | None = None,
    exception_sigmas: jax.Array | None = None,
    exception_epsilons: jax.Array | None = None,
    exception_chargeprods: jax.Array | None = None,
    exception_mask: jax.Array | None = None,
    cmap_torsions: jax.Array | None = None,
    cmap_indices: jax.Array | None = None,
    cmap_coeffs: jax.Array | None = None,
    cmap_mask: jax.Array | None = None,
) -> FusedBondedBuffers:
    """Pack all 6 bonded groups into flat f32/i32 SoA buffers with MK-64 stride=0 deduplication."""
    zi = jnp.zeros((1,), dtype=jnp.int32)
    zf = jnp.zeros((1,), dtype=jnp.float32)

    # Merge Urey-Bradley 1-3 harmonic pairs into 2-body bonds
    if urey_bradley_bonds is not None and jnp.asarray(urey_bradley_bonds).size > 0:
        ub_idx = jnp.asarray(urey_bradley_bonds, dtype=jnp.int32)
        if ub_idx.shape[-1] == 3:
            ub_idx = ub_idx[..., [0, 2]]
        ub_prm = jnp.asarray(urey_bradley_params, dtype=jnp.float32)
        ub_msk = (
            jnp.asarray(urey_bradley_mask, dtype=jnp.float32)
            if urey_bradley_mask is not None
            else jnp.ones(ub_idx.shape[:-1], dtype=jnp.float32)
        )
        if bonds is not None and jnp.asarray(bonds).size > 0:
            b_idx_0 = jnp.asarray(bonds, dtype=jnp.int32)
            b_prm_0 = jnp.asarray(bond_params, dtype=jnp.float32)
            b_msk_0 = (
                jnp.asarray(bond_mask, dtype=jnp.float32)
                if bond_mask is not None
                else jnp.ones(b_idx_0.shape[:-1], dtype=jnp.float32)
            )
            bonds = _cat_bonded_pair(b_idx_0, ub_idx, 2, B, -2)
            bond_params = _cat_bonded_pair(b_prm_0, ub_prm, 2, B, -2)
            bond_mask = _cat_bonded_pair(b_msk_0, ub_msk, 1, B, -1)
        else:
            bonds, bond_params, bond_mask = ub_idx, ub_prm, ub_msk

    # Split periodic vs harmonic impropers
    if impropers is not None and jnp.asarray(impropers).size > 0:
        if improper_is_harmonic:
            harm_idx, harm_prm, harm_msk = impropers, improper_params, improper_mask
        else:
            imp_i = jnp.asarray(impropers, dtype=jnp.int32)
            imp_p = jnp.asarray(improper_params, dtype=jnp.float32)
            imp_m = (
                jnp.asarray(improper_mask, dtype=jnp.float32)
                if improper_mask is not None
                else jnp.ones(imp_i.shape[:-1], dtype=jnp.float32)
            )
            if dihedrals is not None and jnp.asarray(dihedrals).size > 0:
                d_i0 = jnp.asarray(dihedrals, dtype=jnp.int32)
                d_p0 = jnp.asarray(dihedral_params, dtype=jnp.float32)
                d_m0 = (
                    jnp.asarray(dihedral_mask, dtype=jnp.float32)
                    if dihedral_mask is not None
                    else jnp.ones(d_i0.shape[:-1], dtype=jnp.float32)
                )
                dihedrals = _cat_bonded_pair(d_i0, imp_i, 2, B, -2)
                dihedral_params = _cat_bonded_pair(d_p0, imp_p, 2, B, -2)
                dihedral_mask = _cat_bonded_pair(d_m0, imp_m, 1, B, -1)
            else:
                dihedrals, dihedral_params, dihedral_mask = imp_i, imp_p, imp_m
            harm_idx = harm_prm = harm_msk = None
    else:
        harm_idx = harm_prm = harm_msk = None

    bi_a, B_real, s_bi = _prep_term(bonds, 2, B, jnp.int32)
    bp_a, _, s_bp = _prep_term(bond_params, 2, B, jnp.float32)
    bm_a, _, s_bm = _prep_term(
        bond_mask if bond_mask is not None else (jnp.ones(bi_a.shape[:2], jnp.float32)[0] if B_real > 0 else None),
        1, B, jnp.float32,
    )
    if B_real > 0 and not isinstance(bm_a, jax.core.Tracer) and float(np.max(np.asarray(bm_a))) <= 0.0:
        B_real = 0

    ai_a, A_real, s_ai = _prep_term(angles, 2, B, jnp.int32)
    ap_a, _, s_ap = _prep_term(angle_params, 2, B, jnp.float32)
    am_a, _, s_am = _prep_term(
        angle_mask if angle_mask is not None else (jnp.ones(ai_a.shape[:2], jnp.float32)[0] if A_real > 0 else None),
        1, B, jnp.float32,
    )
    if A_real > 0 and not isinstance(am_a, jax.core.Tracer) and float(np.max(np.asarray(am_a))) <= 0.0:
        A_real = 0

    di_a, D_real, s_di = _prep_term(dihedrals, 2, B, jnp.int32)
    dp_a, _, s_dp = _prep_term(dihedral_params, 2, B, jnp.float32)
    dm_a, _, s_dm = _prep_term(
        dihedral_mask if dihedral_mask is not None else (jnp.ones(di_a.shape[:2], jnp.float32)[0] if D_real > 0 else None),
        1, B, jnp.float32,
    )
    if D_real > 0 and not isinstance(dm_a, jax.core.Tracer) and float(np.max(np.asarray(dm_a))) <= 0.0:
        D_real = 0

    hi_a, H_real, s_hi = _prep_term(harm_idx, 2, B, jnp.int32)
    hp_a, _, s_hp = _prep_term(harm_prm, 2, B, jnp.float32)
    hm_a, _, s_hm = _prep_term(
        harm_msk if harm_msk is not None else (jnp.ones(hi_a.shape[:2], jnp.float32)[0] if H_real > 0 else None),
        1, B, jnp.float32,
    )
    if H_real > 0 and not isinstance(hm_a, jax.core.Tracer) and float(np.max(np.asarray(hm_a))) <= 0.0:
        H_real = 0

    ep_a, E_real, s_ep = _prep_term(exception_pairs, 2, B, jnp.int32)
    es_a, _, s_es = _prep_term(exception_sigmas, 1, B, jnp.float32)
    ee_a, _, s_ee = _prep_term(exception_epsilons, 1, B, jnp.float32)
    eq_a, _, s_eq = _prep_term(exception_chargeprods, 1, B, jnp.float32)
    em_a, _, s_em = _prep_term(
        exception_mask if exception_mask is not None else (jnp.ones(ep_a.shape[:2], jnp.float32)[0] if E_real > 0 else None),
        1, B, jnp.float32,
    )
    if E_real > 0 and not isinstance(em_a, jax.core.Tracer) and float(np.max(np.asarray(em_a))) <= 0.0:
        E_real = 0

    if cmap_torsions is not None and cmap_coeffs is not None and jnp.asarray(cmap_torsions).size > 0 and jnp.asarray(cmap_coeffs).size > 0:
        ct_raw = jnp.asarray(cmap_torsions, dtype=jnp.int32)
        if ct_raw.shape[-1] == 8:
            ct_raw = ct_raw[..., [0, 1, 2, 3, 7]]
        ct_a, C_real, s_ct = _prep_term(ct_raw, 2, B, jnp.int32)
        cmi_a, _, s_cmi = _prep_term(cmap_indices, 1, B, jnp.int32)
        cm_a, _, s_cm = _prep_term(
            cmap_mask if cmap_mask is not None else jnp.ones(ct_raw.shape[:-1], jnp.float32),
            1, B, jnp.float32,
        )
        cc_raw = jnp.asarray(cmap_coeffs, dtype=jnp.float32)
        cc_b = cc_raw if cc_raw.ndim == 5 else cc_raw[None, ...]
        N_maps, G_cmap = int(cc_b.shape[1]), int(cc_b.shape[2])
        s_cc = int(N_maps * G_cmap * G_cmap * 16) if (cc_raw.ndim == 5 and int(cc_b.shape[0]) == B and B > 1) else 0
        if s_cc == 0 and int(cc_b.shape[0]) > 1:
            cc_b = cc_b[:1]
        if not isinstance(cm_a, jax.core.Tracer) and float(np.max(np.asarray(cm_a))) <= 0.0:
            C_real = 0
    else:
        ct_a = cmi_a = cm_a = cc_b = None
        C_real = G_cmap = N_maps = s_ct = s_cmi = s_cc = s_cm = 0
        G_cmap, N_maps = 24, 1

    return FusedBondedBuffers(
        b_i=bi_a[..., 0].reshape(-1) if B_real > 0 else zi,
        b_j=bi_a[..., 1].reshape(-1) if B_real > 0 else zi,
        b_r0=bp_a[..., 0].reshape(-1) if B_real > 0 else zf,
        b_k=bp_a[..., 1].reshape(-1) if B_real > 0 else zf,
        b_msk=bm_a.reshape(-1) if B_real > 0 else zf,
        B_real=B_real, s_bi=s_bi, s_bp=s_bp, s_bm=s_bm,
        a_i=ai_a[..., 0].reshape(-1) if A_real > 0 else zi,
        a_j=ai_a[..., 1].reshape(-1) if A_real > 0 else zi,
        a_k=ai_a[..., 2].reshape(-1) if A_real > 0 else zi,
        a_th0=ap_a[..., 0].reshape(-1) if A_real > 0 else zf,
        a_kval=ap_a[..., 1].reshape(-1) if A_real > 0 else zf,
        a_msk=am_a.reshape(-1) if A_real > 0 else zf,
        A_real=A_real, s_ai=s_ai, s_ap=s_ap, s_am=s_am,
        d_i=di_a[..., 0].reshape(-1) if D_real > 0 else zi,
        d_j=di_a[..., 1].reshape(-1) if D_real > 0 else zi,
        d_k=di_a[..., 2].reshape(-1) if D_real > 0 else zi,
        d_l=di_a[..., 3].reshape(-1) if D_real > 0 else zi,
        d_p0=dp_a[..., 0].reshape(-1) if D_real > 0 else zf,
        d_p1=dp_a[..., 1].reshape(-1) if D_real > 0 else zf,
        d_p2=dp_a[..., 2].reshape(-1) if D_real > 0 else zf,
        d_msk=dm_a.reshape(-1) if D_real > 0 else zf,
        D_real=D_real, s_di=s_di, s_dp=s_dp, s_dm=s_dm,
        h_i=hi_a[..., 0].reshape(-1) if H_real > 0 else zi,
        h_j=hi_a[..., 1].reshape(-1) if H_real > 0 else zi,
        h_k=hi_a[..., 2].reshape(-1) if H_real > 0 else zi,
        h_l=hi_a[..., 3].reshape(-1) if H_real > 0 else zi,
        h_kval=hp_a[..., 0].reshape(-1) if H_real > 0 else zf,
        h_phi0=hp_a[..., 1].reshape(-1) if H_real > 0 else zf,
        h_msk=hm_a.reshape(-1) if H_real > 0 else zf,
        H_real=H_real, s_hi=s_hi, s_hp=s_hp, s_hm=s_hm,
        e_i=ep_a[..., 0].reshape(-1) if E_real > 0 else zi,
        e_j=ep_a[..., 1].reshape(-1) if E_real > 0 else zi,
        e_sig=es_a.reshape(-1) if E_real > 0 else zf,
        e_eps=ee_a.reshape(-1) if E_real > 0 else zf,
        e_qp=eq_a.reshape(-1) if E_real > 0 else zf,
        e_msk=em_a.reshape(-1) if E_real > 0 else zf,
        E_real=E_real, s_ep=s_ep, s_es=s_es, s_ee=s_ee, s_eq=s_eq, s_em=s_em,
        c_0=ct_a[..., 0].reshape(-1) if C_real > 0 and ct_a is not None else zi,
        c_1=ct_a[..., 1].reshape(-1) if C_real > 0 and ct_a is not None else zi,
        c_2=ct_a[..., 2].reshape(-1) if C_real > 0 and ct_a is not None else zi,
        c_3=ct_a[..., 3].reshape(-1) if C_real > 0 and ct_a is not None else zi,
        c_4=ct_a[..., 4].reshape(-1) if C_real > 0 and ct_a is not None else zi,
        c_midx=cmi_a.reshape(-1) if C_real > 0 and cmi_a is not None else zi,
        c_coeffs=cc_b.reshape(-1) if C_real > 0 and cc_b is not None else zf,
        c_msk=cm_a.reshape(-1) if C_real > 0 and cm_a is not None else zf,
        C_real=C_real, G_cmap=G_cmap, N_maps=N_maps, s_ct=s_ct, s_cmi=s_cmi, s_cc=s_cc, s_cm=s_cm,
    )


def _pallas_bonded_and_tile_pairs_soa(  # noqa: PLR0913
    B: int,
    px_m: jax.Array,
    py_m: jax.Array,
    pz_m: jax.Array,
    q_m: jax.Array,
    sig_m: jax.Array,
    eps_m: jax.Array,
    mask_m: jax.Array,
    inv_perm_flat: jax.Array,
    tp_i_flat: jax.Array,
    tp_j_flat: jax.Array,
    ef_flat: jax.Array,
    e14_flat: jax.Array,
    n_valid_flat: jax.Array,
    box_flat: jax.Array,
    lam_flat: jax.Array,
    bonded: FusedBondedBuffers,
    *,
    n_act: int,
    n_act_blocks: int,
    max_pairs: int,
    pme_alpha: float = 0.0,
    cutoff: float = 9.0,
    coulomb_constant: float = COULOMB_CONSTANT,
    lj_switch_width: float = 0.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    include_pme_excl_erf: bool = False,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 1.0 / 1.2,
    dist_eps: float = 1e-10,
    b0_is_ji: bool = True,
    angle_regularization_mode: str = "regularized_cos",
    cmap_first_is_psi: bool = True,
    tp_group: int | None = None,
    compute_energy: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """MK-65/71: Single-launch fused Morton 32x32 tile-pairs + all 6 bonded CTA groups."""
    dtype = jnp.float32
    n_act_pad = n_act_blocks * _NB_BLOCK_N
    eff_tp_group = max(1, int(tp_group)) if tp_group is not None else (8 if (B == 1 and max_pairs >= 16384) else 4)
    n_tp_blocks = max(1, (max_pairs + eff_tp_group - 1) // eff_tp_group)

    n_blk_b = (bonded.B_real + _BLOCK_B - 1) // _BLOCK_B if bonded.B_real > 0 else 0
    n_blk_a = (bonded.A_real + _BLOCK_B - 1) // _BLOCK_B if bonded.A_real > 0 else 0
    n_blk_d = (bonded.D_real + _BLOCK_B - 1) // _BLOCK_B if bonded.D_real > 0 else 0
    n_blk_h = (bonded.H_real + _BLOCK_B - 1) // _BLOCK_B if bonded.H_real > 0 else 0
    n_blk_e = (bonded.E_real + _BLOCK_B - 1) // _BLOCK_B if bonded.E_real > 0 else 0
    n_blk_c = (bonded.C_real + _BLOCK_B - 1) // _BLOCK_B if bonded.C_real > 0 else 0
    o_b0, o_b1 = 0, n_blk_b
    o_a0, o_a1 = o_b1, o_b1 + n_blk_a
    o_d0, o_d1 = o_a1, o_a1 + n_blk_d
    o_h0, o_h1 = o_d1, o_d1 + n_blk_h
    o_e0, o_e1 = o_h1, o_h1 + n_blk_e
    o_c0, o_c1 = o_e1, o_e1 + n_blk_c
    total_bonded_blocks = o_c1
    total_blocks = n_tp_blocks + total_bonded_blocks
    e_out_dim = total_blocks if compute_energy else 1
    max_atom = max(n_act - 1, 0)

    s_q = n_act_pad if int(q_m.size) == B * n_act_pad and B > 1 else (n_act_pad if B == 1 else 0)
    s_sig = n_act_pad if int(sig_m.size) == B * n_act_pad and B > 1 else (n_act_pad if B == 1 else 0)
    s_eps = n_act_pad if int(eps_m.size) == B * n_act_pad and B > 1 else (n_act_pad if B == 1 else 0)
    s_msk = n_act_pad if int(mask_m.size) == B * n_act_pad and B > 1 else (n_act_pad if B == 1 else 0)
    s_inv = n_act_pad if int(inv_perm_flat.size) == B * n_act_pad and B > 1 else (n_act_pad if B == 1 else 0)
    s_box = 3 if int(box_flat.size) == B * 3 and B > 1 else (3 if B == 1 else 0)
    s_lam = 1 if int(lam_flat.size) == B and B > 1 else 0

    fx0 = jnp.zeros((B * n_act_pad,), dtype=dtype)
    fy0 = jnp.zeros((B * n_act_pad,), dtype=dtype)
    fz0 = jnp.zeros((B * n_act_pad,), dtype=dtype)

    def _fused_kernel(  # noqa: PLR0913
        px_r, py_r, pz_r, q_r, sig_r, eps_r, msk_r, inv_r,
        tpi_r, tpj_r, ef_r, e14_r, nval_r, box_r, lam_r,
        bi_r, bj_r, br0_r, bk_r, bm_r,
        ai_r, aj_r, ak_r, ath0_r, akv_r, am_r,
        di_r, dj_r, dk_r, dl_r, dp0_r, dp1_r, dp2_r, dm_r,
        hi_r, hj_r, hk_r, hl_r, hkv_r, hph_r, hm_r,
        ei_r, ej_r, es_r, ee_r, eq_r, em_r,
        c0_r, c1_r, c2_r, c3_r, c4_r, cm_idx_r, cc_r, cm_r,
        fx_in_r, fy_in_r, fz_in_r,
        e_out_r, fx_out_r, fy_out_r, fz_out_r,
    ):
        del fx_in_r, fy_in_r, fz_in_r
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        pos_base = b_id * n_act_pad
        f_base = b_id * n_act_pad
        z_idx = jnp.zeros((1,), dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)
        six = jnp.asarray(6.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)

        bx = _pl_load(box_r, (b_id * s_box + z_idx,))[0]
        by = _pl_load(box_r, (b_id * s_box + jnp.ones((1,), dtype=jnp.int32),))[0]
        bz = _pl_load(box_r, (b_id * s_box + jnp.full((1,), 2, dtype=jnp.int32),))[0]
        box_xyz = (bx, by, bz)
        lam_v = _pl_load(lam_r, (b_id * s_lam + z_idx,))[0]

        def _do_tile_pairs(tp_pid):
            p0 = tp_pid * eff_tp_group
            n_val_b = _pl_load(nval_r, (b_id + z_idx,))[0]

            def _active_group(_):
                inv_bx = one / jnp.maximum(bx, jnp.asarray(1e-4, dtype=dtype))
                inv_by = one / jnp.maximum(by, jnp.asarray(1e-4, dtype=dtype))
                inv_bz = one / jnp.maximum(bz, jnp.asarray(1e-4, dtype=dtype))
                _oml_tp = jnp.maximum(one - lam_v, zero)
                _oml_pow = one
                for _ in range(int(softcore_exponent)):
                    _oml_pow = _oml_pow * _oml_tp
                soft_lj = jnp.asarray(softcore_alpha, dtype=dtype) * _oml_pow
                s14_v = jnp.asarray(scale_14_vdw, dtype=dtype)
                s14_e = jnp.asarray(scale_14_elec, dtype=dtype)
                cut_sq = jnp.asarray(cutoff * cutoff, dtype=dtype)
                coul_c = jnp.asarray(coulomb_constant, dtype=dtype)
                alpha_c = jnp.asarray(pme_alpha, dtype=dtype)
                two_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)
                lane32 = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
                ut_mask = lane32[:, None] < lane32[None, :]
                bit_sh = lane32[None, :]
                zv = (lane32 - lane32).astype(dtype)
                zi32 = lane32 - lane32
                base_tp = b_id * max_pairs
                base_ef = b_id * max_pairs * _NB_BLOCK_N

                def _flush_i(cti, m_i32, afx, afy, afz):
                    def _do_fl(z):
                        oi = cti * _NB_BLOCK_N + lane32
                        mi = m_i32 > 0
                        _pl_atomic_add_1d(fx_out_r, f_base + oi, afx, mi, interpret)
                        _pl_atomic_add_1d(fy_out_r, f_base + oi, afy, mi, interpret)
                        _pl_atomic_add_1d(fz_out_r, f_base + oi, afz, mi, interpret)
                        return z

                    _ = jax.lax.cond(cti >= 0, _do_fl, lambda z: z, (tp_pid - tp_pid).astype(jnp.int32))

                def _pair_step(step_k, carry):
                    cti, pxi, pyi, pzi, qi, sigi, epsi, mi_i32, afx, afy, afz, e_acc = carry
                    p_idx = p0 + step_k
                    in_p = p_idx < n_val_b
                    sp_idx = jnp.where(in_p, p_idx, 0)
                    ti = _pl_load(tpi_r, (base_tp + sp_idx + z_idx,))[0]
                    tj = _pl_load(tpj_r, (base_tp + sp_idx + z_idx,))[0]
                    v_tp = in_p & (ti >= 0) & (tj >= 0) & (ti < n_act_blocks) & (tj < n_act_blocks)

                    def _run_tp(c_in):
                        cti0, pxi0, pyi0, pzi0, qi0, sigi0, epsi0, mi0, afx0, afy0, afz0, ea0 = c_in
                        sti = jnp.clip(ti, 0, n_act_blocks - 1)
                        stj = jnp.clip(tj, 0, n_act_blocks - 1)

                        def _reload_i(c_prev):
                            cti_p, _, _, _, _, _, _, mi_p, afx_p, afy_p, afz_p = c_prev
                            _flush_i(cti_p, mi_p, afx_p, afy_p, afz_p)
                            oi = sti * _NB_BLOCK_N + lane32
                            zf_v = afx_p - afx_p
                            return (
                                sti,
                                _pl_load(px_r, (pos_base + oi,)),
                                _pl_load(py_r, (pos_base + oi,)),
                                _pl_load(pz_r, (pos_base + oi,)),
                                _pl_load(q_r, (b_id * s_q + oi,)),
                                _pl_load(sig_r, (b_id * s_sig + oi,)),
                                _pl_load(eps_r, (b_id * s_eps + oi,)),
                                _pl_load(msk_r, (b_id * s_msk + oi,)),
                                zf_v, zf_v, zf_v,
                            )

                        cti1, pxi1, pyi1, pzi1, qi1, sigi1, epsi1, mi1, afx1, afy1, afz1 = jax.lax.cond(
                            sti != cti0,
                            _reload_i,
                            lambda c_prev: c_prev,
                            (cti0, pxi0, pyi0, pzi0, qi0, sigi0, epsi0, mi0, afx0, afy0, afz0),
                        )
                        oj = stj * _NB_BLOCK_N + lane32
                        pxj = _pl_load(px_r, (pos_base + oj,))
                        pyj = _pl_load(py_r, (pos_base + oj,))
                        pzj = _pl_load(pz_r, (pos_base + oj,))
                        mj_bool = _pl_load(msk_r, (b_id * s_msk + oj,)) > 0
                        mi_bool = mi1 > 0

                        fb = _pl_load(ef_r, (base_ef + sp_idx * _NB_BLOCK_N + lane32,))
                        is_full = ((fb[:, None] >> bit_sh) & jnp.int32(1)) != 0
                        is_diag = sti == stj
                        geom_ok = jnp.where(is_diag, ut_mask, jnp.bool_(True))
                        vp = mi_bool[:, None] & mj_bool[None, :] & geom_ok & (
                            True if include_pme_excl_erf else (~is_full)
                        )
                        rdx = pxi1[:, None] - pxj[None, :]
                        rdy = pyi1[:, None] - pyj[None, :]
                        rdz = pzi1[:, None] - pzj[None, :]
                        dx = rdx - bx * jnp.floor(rdx * inv_bx + half)
                        dy = rdy - by * jnp.floor(rdy * inv_by + half)
                        dz = rdz - bz * jnp.floor(rdz * inv_bz + half)
                        r2_raw = dx * dx + dy * dy + dz * dz
                        act = vp & (r2_raw < cut_sq)

                        def _eval_act(a_in):
                            ax_i, ay_i, az_i, ea_i = a_in
                            qj = _pl_load(q_r, (b_id * s_q + oj,))
                            sigj = _pl_load(sig_r, (b_id * s_sig + oj,))
                            epsj = _pl_load(eps_r, (b_id * s_eps + oj,))
                            b14 = _pl_load(e14_r, (base_ef + sp_idx * _NB_BLOCK_N + lane32,))
                            is_14 = ((b14[:, None] >> bit_sh) & jnp.int32(1)) != 0

                            r2 = r2_raw + jnp.asarray(1e-12, dtype=dtype)
                            dist = jnp.sqrt(r2)
                            inv_r = jnp.where(act, one / dist, zero)
                            inv_r2 = inv_r * inv_r
                            w_v = jnp.where(is_full, zero, jnp.where(is_14, s14_v, one))
                            w_e = jnp.where(is_full, zero, jnp.where(is_14, s14_e, one))

                            sig_ij = half * (sigi1[:, None] + sigj[None, :])
                            sig_s = jnp.where(act, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
                            eps_ij = jnp.sqrt(jnp.maximum(epsi1[:, None] * epsj[None, :], zero) + jnp.asarray(1e-30, dtype=dtype))
                            _u_rat = dist / sig_s
                            _u2 = _u_rat * _u_rat
                            u6 = _u2 * _u2 * _u2
                            xi6 = one / jnp.maximum(soft_lj + u6, jnp.asarray(1e-6, dtype=dtype))
                            v_lj = four * eps_ij * lam_v * (xi6 * xi6 - xi6)
                            dv_lj = jnp.asarray(-24.0, dtype=dtype) * eps_ij * lam_v * (two * xi6 - one) * (xi6 * xi6) * u6 * inv_r
                            sw, dsw = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
                            e_lj = w_v * v_lj * sw
                            de_lj = w_v * (dv_lj * sw + v_lj * dsw)

                            qq = coul_c * qi1[:, None] * qj[None, :]
                            if pme_alpha > 0.0:
                                ar = alpha_c * dist
                                ex_t = jnp.exp(zero - ar * ar)
                                if include_pme_excl_erf:
                                    erf_v = _triton_erf(ar)
                                    ew = w_e - erf_v
                                    e_c = qq * ew * inv_r
                                    de_c = qq * ((zero - ew) * inv_r2 - (alpha_c * two_sqrt_pi * ex_t) * inv_r)
                                else:
                                    erfc_v = _triton_erfc(ar)
                                    e_c = qq * w_e * erfc_v * inv_r
                                    de_c = qq * w_e * ((zero - erfc_v) * inv_r2 - (alpha_c * two_sqrt_pi * ex_t) * inv_r)
                            else:
                                vc = dist >= jnp.asarray(1e-4, dtype=dtype)
                                e_c = jnp.where(vc, qq * w_e * inv_r, zero)
                                de_c = jnp.where(vc, (zero - qq) * w_e * inv_r2, zero)

                            f_sc = jnp.where(act, (zero - (de_lj + de_c)) * inv_r, zero)
                            fxm = f_sc * dx
                            fym = f_sc * dy
                            fzm = f_sc * dz
                            fxi = jnp.sum(fxm, axis=1)
                            fyi = jnp.sum(fym, axis=1)
                            fzi = jnp.sum(fzm, axis=1)
                            fxj = zero - jnp.sum(fxm, axis=0)
                            fyj = zero - jnp.sum(fym, axis=0)
                            fzj = zero - jnp.sum(fzm, axis=0)

                            def _add_j(_):
                                _pl_atomic_add_1d(fx_out_r, f_base + oj, fxj, mj_bool, interpret)
                                _pl_atomic_add_1d(fy_out_r, f_base + oj, fyj, mj_bool, interpret)
                                _pl_atomic_add_1d(fz_out_r, f_base + oj, fzj, mj_bool, interpret)
                                return (ax_i + fxi, ay_i + fyi, az_i + fzi)

                            nx_i, ny_i, nz_i = jax.lax.cond(
                                is_diag,
                                lambda _: (ax_i + fxi + fxj, ay_i + fyi + fyj, az_i + fzi + fzj),
                                _add_j,
                                None,
                            )
                            ne = (ea_i + jnp.sum(jnp.where(act, e_lj + e_c, zero))) if compute_energy else ea_i
                            return (nx_i, ny_i, nz_i, ne)

                        afx2, afy2, afz2, ea2 = jax.lax.cond(
                            jnp.max(act.astype(jnp.int32)) > 0,
                            _eval_act,
                            lambda a_in: a_in,
                            (afx1, afy1, afz1, ea0),
                        )
                        return (cti1, pxi1, pyi1, pzi1, qi1, sigi1, epsi1, mi1, afx2, afy2, afz2, ea2)

                    return jax.lax.cond(v_tp, _run_tp, lambda c_in: c_in, carry)

                init_c = (
                    (tp_pid - tp_pid) - jnp.int32(1),
                    zv, zv, zv, zv, zv, zv, zi32, zv, zv, zv,
                    (tp_pid - tp_pid).astype(dtype),
                )
                fin = jax.lax.fori_loop(jnp.int32(0), jnp.int32(eff_tp_group), _pair_step, init_c)
                _flush_i(fin[0], fin[7], fin[8], fin[9], fin[10])
                return fin[11]

            e_tp = jax.lax.cond(p0 < n_val_b, _active_group, lambda z: z, (tp_pid - tp_pid).astype(dtype))
            if compute_energy:
                _pl_store(e_out_r, (b_id * e_out_dim + tp_pid + z_idx,), e_tp[None])
            return tp_pid - tp_pid

        def _map_m(s_atom, v_m):
            return _pl_load(inv_r, (b_id * s_inv + s_atom,), mask=v_m, other=0)

        def _load_r(m_idx):
            return (
                _pl_load(px_r, (pos_base + m_idx,)),
                _pl_load(py_r, (pos_base + m_idx,)),
                _pl_load(pz_r, (pos_base + m_idx,)),
            )

        def _add_f(m_idx, f_vec, v_m):
            _pl_atomic_add_1d(fx_out_r, f_base + m_idx, f_vec[0], v_m, interpret)
            _pl_atomic_add_1d(fy_out_r, f_base + m_idx, f_vec[1], v_m, interpret)
            _pl_atomic_add_1d(fz_out_r, f_base + m_idx, f_vec[2], v_m, interpret)

        def _do_bonds(p_val):
            offs = (p_val - o_b0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            in_b = offs < bonded.B_real
            so = jnp.minimum(offs, max(bonded.B_real - 1, 0))
            mv = _pl_load(bm_r, (b_id * bonded.s_bm + so,), mask=in_b, other=zero)

            def _ev(_):
                ai = _pl_load(bi_r, (b_id * bonded.s_bi + so,), mask=in_b, other=-1)
                aj = _pl_load(bj_r, (b_id * bonded.s_bi + so,), mask=in_b, other=-1)
                v = in_b & (mv > zero) & (ai >= 0) & (ai < n_act) & (aj >= 0) & (aj < n_act)
                me = jnp.where(v, mv, zero)
                mi = _map_m(jnp.where(v, jnp.clip(ai, 0, max_atom), 0), v)
                mj = _map_m(jnp.where(v, jnp.clip(aj, 0, max_atom), 0), v)
                dv = _min_image3(_sub3(_load_r(mi), _load_r(mj)), box_xyz, True)
                d2 = _dot3(dv, dv)
                dist = jnp.sqrt(d2 + jnp.asarray(dist_eps, dtype=dtype))
                r0 = _pl_load(br0_r, (b_id * bonded.s_bp + so,), mask=in_b, other=zero)
                kb = _pl_load(bk_r, (b_id * bonded.s_bp + so,), mask=in_b, other=zero)
                df = dist - r0
                fc = jnp.where(v & (dist > zero), (zero - me * kb * df) / jnp.maximum(dist, jnp.asarray(1e-12, dtype=dtype)), zero)
                fi = _scale3(fc, dv)
                _add_f(mi, fi, v)
                _add_f(mj, _scale3(zero - one, fi), v)
                return jnp.sum(half * me * kb * df * df)

            return jax.lax.cond(
                jnp.max((in_b & (mv > zero)).astype(jnp.int32)) > 0,
                _ev,
                lambda z: z,
                (p_val - p_val).astype(dtype),
            )

        def _do_angles(p_val):
            offs = (p_val - o_a0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            in_a = offs < bonded.A_real
            so = jnp.minimum(offs, max(bonded.A_real - 1, 0))
            mv = _pl_load(am_r, (b_id * bonded.s_am + so,), mask=in_a, other=zero)

            def _ev(_):
                ai = _pl_load(ai_r, (b_id * bonded.s_ai + so,), mask=in_a, other=-1)
                aj = _pl_load(aj_r, (b_id * bonded.s_ai + so,), mask=in_a, other=-1)
                ak = _pl_load(ak_r, (b_id * bonded.s_ai + so,), mask=in_a, other=-1)
                v = in_a & (mv > zero) & (ai >= 0) & (ai < n_act) & (aj >= 0) & (aj < n_act) & (ak >= 0) & (ak < n_act)
                me = jnp.where(v, mv, zero)
                mi = _map_m(jnp.where(v, jnp.clip(ai, 0, max_atom), 0), v)
                mj = _map_m(jnp.where(v, jnp.clip(aj, 0, max_atom), 0), v)
                mk = _map_m(jnp.where(v, jnp.clip(ak, 0, max_atom), 0), v)
                rj = _load_r(mj)
                v_ji = _min_image3(_sub3(_load_r(mi), rj), box_xyz, True)
                v_jk = _min_image3(_sub3(_load_r(mk), rj), box_xyz, True)
                eps_v = jnp.asarray(max(dist_eps, 1e-20), dtype=dtype)
                d_ji2 = _dot3(v_ji, v_ji) + eps_v
                d_jk2 = _dot3(v_jk, v_jk) + eps_v
                d_ji = jnp.sqrt(d_ji2)
                d_jk = jnp.sqrt(d_jk2)
                dot_ik = _dot3(v_ji, v_jk)
                cp = _cross3(v_ji, v_jk)
                th0 = _pl_load(ath0_r, (b_id * bonded.s_ap + so,), mask=in_a, other=zero)
                ka = _pl_load(akv_r, (b_id * bonded.s_ap + so,), mask=in_a, other=zero)
                if angle_regularization_mode == "openmm_cross":
                    cp2 = _dot3(cp, cp)
                    rcp = jnp.sqrt(jnp.maximum(cp2, jnp.asarray(1e-24, dtype=dtype)))
                    dth = jnp.arctan2(rcp, dot_ik) - th0
                    nd = jnp.where(v & (cp2 > jnp.asarray(1e-24, dtype=dtype)), zero - me * ka * dth, zero)
                    rcp_d = rcp
                else:
                    den = d_ji * d_jk + jnp.asarray(1e-8, dtype=dtype)
                    cr = dot_ik / den
                    clim = jnp.asarray(0.999999, dtype=dtype)
                    cc = jnp.clip(cr, zero - clim, clim)
                    dth = jnp.arccos(cc) - th0
                    sth = jnp.sqrt(jnp.maximum(one - cc * cc, jnp.asarray(1e-12, dtype=dtype)))
                    nd = jnp.where(v & (cr > (zero - clim)) & (cr < clim), zero - me * ka * dth, zero)
                    rcp_d = den * sth
                fi = _scale3(nd / (d_ji2 * rcp_d), _cross3(v_ji, cp))
                fk = _scale3(nd / (d_jk2 * rcp_d), _cross3(cp, v_jk))
                _add_f(mi, fi, v)
                _add_f(mj, _scale3(zero - one, _add3(fi, fk)), v)
                _add_f(mk, fk, v)
                return jnp.sum(half * me * ka * dth * dth)

            return jax.lax.cond(
                jnp.max((in_a & (mv > zero)).astype(jnp.int32)) > 0,
                _ev,
                lambda z: z,
                (p_val - p_val).astype(dtype),
            )

        def _do_torsions(p_val):
            offs = (p_val - o_d0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            in_d = offs < bonded.D_real
            so = jnp.minimum(offs, max(bonded.D_real - 1, 0))
            mv = _pl_load(dm_r, (b_id * bonded.s_dm + so,), mask=in_d, other=zero)

            def _ev(_):
                ai = _pl_load(di_r, (b_id * bonded.s_di + so,), mask=in_d, other=-1)
                aj = _pl_load(dj_r, (b_id * bonded.s_di + so,), mask=in_d, other=-1)
                ak = _pl_load(dk_r, (b_id * bonded.s_di + so,), mask=in_d, other=-1)
                al = _pl_load(dl_r, (b_id * bonded.s_di + so,), mask=in_d, other=-1)
                v = in_d & (mv > zero) & (ai >= 0) & (ai < n_act) & (aj >= 0) & (aj < n_act) & (ak >= 0) & (ak < n_act) & (al >= 0) & (al < n_act)
                me = jnp.where(v, mv, zero)
                mi = _map_m(jnp.where(v, jnp.clip(ai, 0, max_atom), 0), v)
                mj = _map_m(jnp.where(v, jnp.clip(aj, 0, max_atom), 0), v)
                mk = _map_m(jnp.where(v, jnp.clip(ak, 0, max_atom), 0), v)
                ml = _map_m(jnp.where(v, jnp.clip(al, 0, max_atom), 0), v)
                phi, gi, gj, gk, gl = _torsion_angle_and_atom_grads_vec3(
                    _load_r(mi), _load_r(mj), _load_r(mk), _load_r(ml), v, box_xyz, True, b0_is_ji
                )
                p0_d = _pl_load(dp0_r, (b_id * bonded.s_dp + so,), mask=in_d, other=zero)
                p1_d = _pl_load(dp1_r, (b_id * bonded.s_dp + so,), mask=in_d, other=zero)
                p2_d = _pl_load(dp2_r, (b_id * bonded.s_dp + so,), mask=in_d, other=zero)
                arg = p0_d * phi - p1_d
                ng = jnp.where(v, me * p2_d * p0_d * jnp.sin(arg), zero)
                for m_idx, gv in ((mi, gi), (mj, gj), (mk, gk), (ml, gl)):
                    _add_f(m_idx, _scale3(ng, gv), v)
                return jnp.sum(me * p2_d * (one + jnp.cos(arg)))

            return jax.lax.cond(
                jnp.max((in_d & (mv > zero)).astype(jnp.int32)) > 0,
                _ev,
                lambda z: z,
                (p_val - p_val).astype(dtype),
            )

        def _do_impropers(p_val):
            offs = (p_val - o_h0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            in_h = offs < bonded.H_real
            so = jnp.minimum(offs, max(bonded.H_real - 1, 0))
            mv = _pl_load(hm_r, (b_id * bonded.s_hm + so,), mask=in_h, other=zero)

            def _ev(_):
                ai = _pl_load(hi_r, (b_id * bonded.s_hi + so,), mask=in_h, other=-1)
                aj = _pl_load(hj_r, (b_id * bonded.s_hi + so,), mask=in_h, other=-1)
                ak = _pl_load(hk_r, (b_id * bonded.s_hi + so,), mask=in_h, other=-1)
                al = _pl_load(hl_r, (b_id * bonded.s_hi + so,), mask=in_h, other=-1)
                v = in_h & (mv > zero) & (ai >= 0) & (ai < n_act) & (aj >= 0) & (aj < n_act) & (ak >= 0) & (ak < n_act) & (al >= 0) & (al < n_act)
                me = jnp.where(v, mv, zero)
                mi = _map_m(jnp.where(v, jnp.clip(ai, 0, max_atom), 0), v)
                mj = _map_m(jnp.where(v, jnp.clip(aj, 0, max_atom), 0), v)
                mk = _map_m(jnp.where(v, jnp.clip(ak, 0, max_atom), 0), v)
                ml = _map_m(jnp.where(v, jnp.clip(al, 0, max_atom), 0), v)
                phi, gi, gj, gk, gl = _torsion_angle_and_atom_grads_vec3(
                    _load_r(mi), _load_r(mj), _load_r(mk), _load_r(ml), v, box_xyz, True, b0_is_ji
                )
                kh = _pl_load(hkv_r, (b_id * bonded.s_hp + so,), mask=in_h, other=zero)
                ph0 = _pl_load(hph_r, (b_id * bonded.s_hp + so,), mask=in_h, other=zero)
                drw = phi - ph0 + pi
                dph = drw - two_pi * jnp.floor(drw / two_pi) - pi
                ng = jnp.where(v, zero - two * me * kh * dph, zero)
                for m_idx, gv in ((mi, gi), (mj, gj), (mk, gk), (ml, gl)):
                    _add_f(m_idx, _scale3(ng, gv), v)
                return jnp.sum(me * kh * dph * dph)

            return jax.lax.cond(
                jnp.max((in_h & (mv > zero)).astype(jnp.int32)) > 0,
                _ev,
                lambda z: z,
                (p_val - p_val).astype(dtype),
            )

        def _do_exc14(p_val):
            offs = (p_val - o_e0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            in_e = offs < bonded.E_real
            so = jnp.minimum(offs, max(bonded.E_real - 1, 0))
            mv = _pl_load(em_r, (b_id * bonded.s_em + so,), mask=in_e, other=zero)

            def _ev(_):
                coul = jnp.asarray(coulomb_constant, dtype=dtype)
                ai = _pl_load(ei_r, (b_id * bonded.s_ep + so,), mask=in_e, other=-1)
                aj = _pl_load(ej_r, (b_id * bonded.s_ep + so,), mask=in_e, other=-1)
                v = in_e & (mv > zero) & (ai >= 0) & (ai < n_act) & (aj >= 0) & (aj < n_act) & (ai != aj)
                me = jnp.where(v, mv, zero)
                mi = _map_m(jnp.where(v, jnp.clip(ai, 0, max_atom), 0), v)
                mj = _map_m(jnp.where(v, jnp.clip(aj, 0, max_atom), 0), v)
                dv = _min_image3(_sub3(_load_r(mi), _load_r(mj)), box_xyz, True)
                r2 = _dot3(dv, dv) + jnp.asarray(1e-12, dtype=dtype)
                dist = jnp.sqrt(r2)
                se = _pl_load(es_r, (b_id * bonded.s_es + so,), mask=in_e, other=one)
                ee = _pl_load(ee_r, (b_id * bonded.s_ee + so,), mask=in_e, other=zero)
                qe = _pl_load(eq_r, (b_id * bonded.s_eq + so,), mask=in_e, other=zero)
                ss = jnp.where(v, jnp.maximum(se, jnp.asarray(1e-4, dtype=dtype)), one)
                _oml_e = jnp.maximum(one - lam_v, zero)
                _oml_e_pow = one
                for _ in range(int(softcore_exponent)):
                    _oml_e_pow = _oml_e_pow * _oml_e
                st = jnp.asarray(softcore_alpha, dtype=dtype) * _oml_e_pow
                _r_rat = dist / ss
                _r2_lj = _r_rat * _r_rat
                r6 = _r2_lj * _r2_lj * _r2_lj
                id1 = one / (st + r6 + jnp.asarray(1e-12, dtype=dtype))
                id2 = id1 * id1
                ec = coul * qe / dist
                de_r = (me / r2) * (four * ee * lam_v * (id2 - two * id2 * id1) * (six * r6) - ec)
                fi = _scale3(jnp.where(v, zero - de_r, zero), dv)
                _add_f(mi, fi, v)
                _add_f(mj, _scale3(zero - one, fi), v)
                return jnp.sum(me * (four * ee * lam_v * (id2 - id1) + ec))

            return jax.lax.cond(
                jnp.max((in_e & (mv > zero)).astype(jnp.int32)) > 0,
                _ev,
                lambda z: z,
                (p_val - p_val).astype(dtype),
            )

        def _do_cmap(p_val):
            offs = (p_val - o_c0) * _BLOCK_B + jnp.arange(_BLOCK_B, dtype=jnp.int32)
            in_c = offs < bonded.C_real
            so = jnp.minimum(offs, max(bonded.C_real - 1, 0))
            mv = _pl_load(cm_r, (b_id * bonded.s_cm + so,), mask=in_c, other=zero)

            def _ev(_):
                inv_del = one / jnp.asarray(2.0 * jnp.pi / bonded.G_cmap, dtype=dtype)
                i0 = _pl_load(c0_r, (b_id * bonded.s_ct + so,), mask=in_c, other=-1)
                i1 = _pl_load(c1_r, (b_id * bonded.s_ct + so,), mask=in_c, other=-1)
                i2 = _pl_load(c2_r, (b_id * bonded.s_ct + so,), mask=in_c, other=-1)
                i3 = _pl_load(c3_r, (b_id * bonded.s_ct + so,), mask=in_c, other=-1)
                i4 = _pl_load(c4_r, (b_id * bonded.s_ct + so,), mask=in_c, other=-1)
                v = in_c & (mv > zero) & (i0 >= 0) & (i0 < n_act) & (i1 >= 0) & (i1 < n_act) & (i2 >= 0) & (i2 < n_act) & (i3 >= 0) & (i3 < n_act) & (i4 >= 0) & (i4 < n_act)
                me = jnp.where(v, mv, zero)
                m0 = _map_m(jnp.where(v, jnp.clip(i0, 0, max_atom), 0), v)
                m1 = _map_m(jnp.where(v, jnp.clip(i1, 0, max_atom), 0), v)
                m2 = _map_m(jnp.where(v, jnp.clip(i2, 0, max_atom), 0), v)
                m3 = _map_m(jnp.where(v, jnp.clip(i3, 0, max_atom), 0), v)
                m4 = _map_m(jnp.where(v, jnp.clip(i4, 0, max_atom), 0), v)
                r0, r1, r2_p, r3, r4 = _load_r(m0), _load_r(m1), _load_r(m2), _load_r(m3), _load_r(m4)
                phi, dp0, dp1, dp2, dp3 = _torsion_angle_and_atom_grads_vec3(r0, r1, r2_p, r3, v, box_xyz, True, False)
                psi, ds1, ds2, ds3, ds4 = _torsion_angle_and_atom_grads_vec3(r1, r2_p, r3, r4, v, box_xyz, True, False)
                th1, th2 = (psi, phi) if cmap_first_is_psi else (phi, psi)
                t1s = th1 + two_pi
                t2s = th2 + two_pi
                q1 = (t1s - two_pi * jnp.floor(t1s / two_pi)) * inv_del
                q2 = (t2s - two_pi * jnp.floor(t2s / two_pi)) * inv_del
                g1r = jnp.floor(q1).astype(jnp.int32)
                g2r = jnp.floor(q2).astype(jnp.int32)
                ms = jnp.where(v, jnp.clip(_pl_load(cm_idx_r, (b_id * bonded.s_cmi + so,), mask=in_c, other=0), 0, max(bonded.N_maps - 1, 0)), 0)
                b_idx = (b_id * bonded.s_cc) + ((ms * bonded.G_cmap + (g1r % bonded.G_cmap)) * bonded.G_cmap + (g2r % bonded.G_cmap)) * 16
                ep, du, dv_c = _eval_cmap_horner_cell(cc_r, b_idx, q1 - g1r.astype(dtype), q2 - g2r.astype(dtype), dtype)
                dt1 = me * du * inv_del
                dt2 = me * dv_c * inv_del
                dph, dps = (dt2, dt1) if cmap_first_is_psi else (dt1, dt2)
                nph = jnp.where(v, zero - dph, zero)
                nps = jnp.where(v, zero - dps, zero)
                _add_f(m0, _scale3(nph, dp0), v)
                _add_f(m1, _add3(_scale3(nph, dp1), _scale3(nps, ds1)), v)
                _add_f(m2, _add3(_scale3(nph, dp2), _scale3(nps, ds2)), v)
                _add_f(m3, _add3(_scale3(nph, dp3), _scale3(nps, ds3)), v)
                _add_f(m4, _scale3(nps, ds4), v)
                return jnp.sum(me * ep)

            return jax.lax.cond(
                jnp.max((in_c & (mv > zero)).astype(jnp.int32)) > 0,
                _ev,
                lambda z: z,
                (p_val - p_val).astype(dtype),
            )

        groups = []
        if n_blk_b > 0:
            groups.append((o_b1, _do_bonds))
        if n_blk_a > 0:
            groups.append((o_a1, _do_angles))
        if n_blk_d > 0:
            groups.append((o_d1, _do_torsions))
        if n_blk_h > 0:
            groups.append((o_h1, _do_impropers))
        if n_blk_e > 0:
            groups.append((o_e1, _do_exc14))
        if n_blk_c > 0:
            groups.append((o_c1, _do_cmap))

        def _disp_b(gi: int, pv: jax.Array) -> jax.Array:
            if gi == len(groups) - 1:
                return groups[gi][1](pv)
            o_end, fn_g = groups[gi]
            return jax.lax.cond(pv < o_end, fn_g, lambda p: _disp_b(gi + 1, p), pv)

        def _do_bonded_all(b_pid):
            es = _disp_b(0, b_pid)
            if compute_energy:
                _pl_store(e_out_r, (b_id * e_out_dim + n_tp_blocks + b_pid + z_idx,), es[None])
            return b_pid - b_pid

        if total_bonded_blocks > 0:
            _ = jax.lax.cond(pid < n_tp_blocks, _do_tile_pairs, lambda p: _do_bonded_all(p - n_tp_blocks), pid)
        else:
            _ = _do_tile_pairs(pid)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    e_blk, fx_m, fy_m, fz_m = pl.pallas_call(
        _fused_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B * e_out_dim,), dtype),
            jax.ShapeDtypeStruct((B * n_act_pad,), dtype),
            jax.ShapeDtypeStruct((B * n_act_pad,), dtype),
            jax.ShapeDtypeStruct((B * n_act_pad,), dtype),
        ],
        in_specs=[any_spec] * 58,
        out_specs=[any_spec] * 4,
        grid=(B, total_blocks),
        input_output_aliases={55: 1, 56: 2, 57: 3},
        interpret=interpret,
        compiler_params=_tile_pairs_compiler_params(B * total_blocks) if not interpret else None,
        name="pallas_bonded_and_tile_pairs_fused",
    )(
        px_m, py_m, pz_m, q_m, sig_m, eps_m, mask_m, inv_perm_flat,
        tp_i_flat, tp_j_flat, ef_flat, e14_flat, n_valid_flat, box_flat, lam_flat,
        bonded.b_i, bonded.b_j, bonded.b_r0, bonded.b_k, bonded.b_msk,
        bonded.a_i, bonded.a_j, bonded.a_k, bonded.a_th0, bonded.a_kval, bonded.a_msk,
        bonded.d_i, bonded.d_j, bonded.d_k, bonded.d_l, bonded.d_p0, bonded.d_p1, bonded.d_p2, bonded.d_msk,
        bonded.h_i, bonded.h_j, bonded.h_k, bonded.h_l, bonded.h_kval, bonded.h_phi0, bonded.h_msk,
        bonded.e_i, bonded.e_j, bonded.e_sig, bonded.e_eps, bonded.e_qp, bonded.e_msk,
        bonded.c_0, bonded.c_1, bonded.c_2, bonded.c_3, bonded.c_4, bonded.c_midx, bonded.c_coeffs, bonded.c_msk,
        fx0, fy0, fz0,
    )
    e_sum = jnp.sum(e_blk.reshape(B, e_out_dim), axis=-1) if compute_energy else jnp.zeros((B,), dtype=dtype)
    return e_sum, fx_m, fy_m, fz_m


def pallas_tile_pairs_param_grads(  # noqa: PLR0913
    B: int,
    in_b: tuple[bool, ...],
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_pairs: jax.Array,
    excl_full_mask: jax.Array,
    excl_14_mask: jax.Array,
    box: jax.Array,
    lam_arr: jax.Array,
    upstream_g: jax.Array,
    *,
    has_box: bool,
    pme_alpha: float,
    cutoff: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
    scale_14_vdw: float,
    scale_14_elec: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """MK-73: Native O(N) memory 2D-grid Pallas parameter-gradient kernel for 32x32 tile pairs."""
    b_pos, b_q, b_sig, b_eps, b_mask, b_tp, b_ef, b_e14, b_box, b_lam = in_b
    N_orig = int(positions.shape[1] if b_pos else positions.shape[0])
    M_pairs = int(tile_pairs.shape[1] if b_tp else tile_pairs.shape[0])
    n_blocks = max(1, (N_orig + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    N = n_blocks * _NB_BLOCK_N
    pad_n = N - N_orig
    dtype = jnp.float32

    def _pad_a(a, batched, fill=0):
        x = a if batched else a[None, ...]
        if pad_n > 0:
            pw = ((0, 0), (0, pad_n)) + ((0, 0),) * (x.ndim - 2)
            x = jnp.pad(x, pw, constant_values=fill)
        return x

    pos_p = _pad_a(positions.astype(dtype), b_pos, 0.0)
    q_p = _pad_a(charges.astype(dtype), b_q, 0.0)
    sig_p = _pad_a(sigmas.astype(dtype), b_sig, 1.0)
    eps_p = _pad_a(epsilons.astype(dtype), b_eps, 0.0)
    msk_p = _pad_a(atom_mask.astype(jnp.int32), b_mask, 0)
    tp_b = tile_pairs if b_tp else tile_pairs[None, ...]
    ef_b = excl_full_mask if b_ef else excl_full_mask[None, ...]
    e14_b = excl_14_mask if b_e14 else excl_14_mask[None, ...]
    box_b = (box if b_box else box[None, ...]).astype(dtype)
    lam_b = (lam_arr.reshape(B) if b_lam else lam_arr.reshape(1)).astype(dtype)
    g_b = jnp.broadcast_to(jnp.asarray(upstream_g, dtype=dtype).reshape(-1), (B,))

    tp_grp = 4
    n_pblk = max(1, (M_pairs + tp_grp - 1) // tp_grp)
    z_dq = jnp.zeros((B * N,), dtype=dtype)
    z_ds = jnp.zeros((B * N,), dtype=dtype)
    z_de = jnp.zeros((B * N,), dtype=dtype)

    def _bwd_kernel(  # noqa: PLR0913
        px_r, py_r, pz_r, q_r, sig_r, eps_r, msk_r,
        tpi_r, tpj_r, ef_r, e14_r, box_r, lam_r, g_r,
        dq_in, ds_in, de_in,
        dq_out, ds_out, de_out, dl_out,
    ):
        del dq_in, ds_in, de_in
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        z0 = jnp.zeros((1,), dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)

        bp = (b_id if b_pos else 0) * N
        bq = (b_id if b_q else 0) * N
        bsg = (b_id if b_sig else 0) * N
        bep = (b_id if b_eps else 0) * N
        bmk = (b_id if b_mask else 0) * N
        btp = (b_id if b_tp else 0) * M_pairs
        bef = (b_id if b_ef else 0) * M_pairs * _NB_BLOCK_N
        be14 = (b_id if b_e14 else 0) * M_pairs * _NB_BLOCK_N
        bbx = (b_id if b_box else 0) * 6
        blm = b_id if b_lam else 0
        out_b = b_id * N

        gv = _pl_load(g_r, (b_id + z0,))[0]
        ax = _pl_load(box_r, (bbx + z0,))[0]
        by = _pl_load(box_r, (bbx + jnp.ones((1,), dtype=jnp.int32),))[0]
        cz = _pl_load(box_r, (bbx + jnp.full((1,), 2, dtype=jnp.int32),))[0]
        bx_t = _pl_load(box_r, (bbx + jnp.full((1,), 3, dtype=jnp.int32),))[0]
        cx_t = _pl_load(box_r, (bbx + jnp.full((1,), 4, dtype=jnp.int32),))[0]
        cy_t = _pl_load(box_r, (bbx + jnp.full((1,), 5, dtype=jnp.int32),))[0]
        inv_ax = jnp.where(ax > zero, one / jnp.maximum(ax, jnp.asarray(1e-4, dtype=dtype)), zero) if has_box else zero
        inv_by = jnp.where(by > zero, one / jnp.maximum(by, jnp.asarray(1e-4, dtype=dtype)), zero) if has_box else zero
        inv_cz = jnp.where(cz > zero, one / jnp.maximum(cz, jnp.asarray(1e-4, dtype=dtype)), zero) if has_box else zero

        lam = _pl_load(lam_r, (blm + z0,))[0]
        oml = jnp.maximum(one - lam, zero)
        sc_p = int(softcore_exponent)
        oml_p = one
        oml_pm1 = one
        for _k in range(sc_p):
            if _k == sc_p - 1:
                oml_pm1 = oml_p
            oml_p = oml_p * oml
        soft_lj = jnp.asarray(softcore_alpha, dtype=dtype) * oml_p
        dsoft_dlam = (
            jnp.asarray(-float(softcore_alpha * sc_p), dtype=dtype) * oml_pm1
            if sc_p > 0
            else zero
        )
        s14_v = jnp.asarray(scale_14_vdw, dtype=dtype)
        s14_e = jnp.asarray(scale_14_elec, dtype=dtype)
        cut_sq = jnp.asarray(cutoff * cutoff, dtype=dtype)
        coul_c = jnp.asarray(coulomb_constant, dtype=dtype)
        alpha_c = jnp.asarray(pme_alpha, dtype=dtype)
        lane32 = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
        ut_mask = lane32[:, None] < lane32[None, :]
        bit_sh = lane32[None, :]

        def _step(sk, dlam_acc):
            p_idx = pid * tp_grp + sk
            in_p = p_idx < M_pairs
            sp = jnp.where(in_p, p_idx, 0)
            ti = _pl_load(tpi_r, (btp + sp + z0,))[0]
            tj = _pl_load(tpj_r, (btp + sp + z0,))[0]
            v_tp = in_p & (ti >= 0) & (tj >= 0) & (ti < n_blocks) & (tj < n_blocks)

            def _do_p(dl_in):
                sti = jnp.clip(ti, 0, n_blocks - 1)
                stj = jnp.clip(tj, 0, n_blocks - 1)
                oi = sti * _NB_BLOCK_N + lane32
                oj = stj * _NB_BLOCK_N + lane32
                mi = (_pl_load(msk_r, (bmk + oi,)) > 0) & (oi < N_orig)
                mj = (_pl_load(msk_r, (bmk + oj,)) > 0) & (oj < N_orig)
                pxi = _pl_load(px_r, (bp + oi,))
                pyi = _pl_load(py_r, (bp + oi,))
                pzi = _pl_load(pz_r, (bp + oi,))
                pxj = _pl_load(px_r, (bp + oj,))
                pyj = _pl_load(py_r, (bp + oj,))
                pzj = _pl_load(pz_r, (bp + oj,))
                fb = _pl_load(ef_r, (bef + sp * _NB_BLOCK_N + lane32,))
                is_full = ((fb[:, None] >> bit_sh) & jnp.int32(1)) != 0
                is_diag = sti == stj
                vp = mi[:, None] & mj[None, :] & jnp.where(is_diag, ut_mask, jnp.bool_(True)) & (
                    True if include_pme_excl_erf else (~is_full)
                )
                rdx = pxi[:, None] - pxj[None, :]
                rdy = pyi[:, None] - pyj[None, :]
                rdz = pzi[:, None] - pzj[None, :]
                if has_box:
                    sz = jnp.floor(rdz * inv_cz + half)
                    dxz = rdx - sz * cx_t
                    dyz = rdy - sz * cy_t
                    dz = rdz - sz * cz
                    sy = jnp.floor(dyz * inv_by + half)
                    dxy = dxz - sy * bx_t
                    dy = dyz - sy * by
                    dx = dxy - jnp.floor(dxy * inv_ax + half) * ax
                else:
                    dx, dy, dz = rdx, rdy, rdz
                r2_raw = dx * dx + dy * dy + dz * dz
                act = vp & (r2_raw < cut_sq)

                def _act_bwd(dl0):
                    qi = _pl_load(q_r, (bq + oi,))
                    qj = _pl_load(q_r, (bq + oj,))
                    sigi = _pl_load(sig_r, (bsg + oi,))
                    sigj = _pl_load(sig_r, (bsg + oj,))
                    epsi = _pl_load(eps_r, (bep + oi,))
                    epsj = _pl_load(eps_r, (bep + oj,))
                    b14 = _pl_load(e14_r, (be14 + sp * _NB_BLOCK_N + lane32,))
                    is_14 = ((b14[:, None] >> bit_sh) & jnp.int32(1)) != 0

                    dist = jnp.sqrt(r2_raw + jnp.asarray(1e-12, dtype=dtype))
                    inv_r = jnp.where(act, one / dist, zero)
                    wv = jnp.where(is_full, zero, jnp.where(is_14, s14_v, one))
                    we = jnp.where(is_full, zero, jnp.where(is_14, s14_e, one))
                    sw, _ = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
                    w_lj = jnp.where(act, gv * wv * sw, zero)

                    sig_ij = half * (sigi[:, None] + sigj[None, :])
                    sig_s = jnp.where(act, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
                    ep_prod = jnp.maximum(epsi[:, None] * epsj[None, :], zero)
                    eps_ij = jnp.sqrt(ep_prod + jnp.asarray(1e-30, dtype=dtype))
                    _u_bwd = dist / sig_s
                    _u2_bwd = _u_bwd * _u_bwd
                    u6 = _u2_bwd * _u2_bwd * _u2_bwd
                    xi6 = one / jnp.maximum(soft_lj + u6, jnp.asarray(1e-6, dtype=dtype))
                    x2_x = xi6 * xi6 - xi6
                    dx_dd = (zero - xi6 * xi6) * (two * xi6 - one)

                    # dE/dsigma
                    de_dsig_mat = w_lj * four * eps_ij * lam * dx_dd * (jnp.asarray(-6.0, dtype=dtype) * u6 / sig_s) * half
                    # dE/deps
                    has_ep = act & (epsi[:, None] > zero) & (epsj[None, :] > zero)
                    inv_2eij = jnp.where(has_ep, half / jnp.maximum(eps_ij, jnp.asarray(1e-15, dtype=dtype)), zero)
                    base_de_deij = w_lj * four * lam * x2_x * inv_2eij
                    de_depsi = jnp.sum(base_de_deij * epsj[None, :], axis=1)
                    de_depsj = jnp.sum(base_de_deij * epsi[:, None], axis=0)
                    # dE/dlam
                    de_dlam_mat = w_lj * four * eps_ij * (x2_x + lam * dx_dd * dsoft_dlam)

                    # dE/dq
                    if pme_alpha > 0.0:
                        ar = alpha_c * dist
                        if include_pme_excl_erf:
                            erf_v = erf(ar) if (interpret and dtype == jnp.float64) else _triton_erf(ar)
                            cfac = (we - erf_v) * inv_r
                        else:
                            erfc_v = erfc(ar) if (interpret and dtype == jnp.float64) else _triton_erfc(ar)
                            cfac = we * erfc_v * inv_r
                    else:
                        cfac = jnp.where(dist >= jnp.asarray(1e-4, dtype=dtype), we * inv_r, zero)
                    c_mat = jnp.where(act, gv * coul_c * cfac, zero)
                    dqi = jnp.sum(c_mat * qj[None, :], axis=1)
                    dqj = jnp.sum(c_mat * qi[:, None], axis=0)
                    dsi = jnp.sum(de_dsig_mat, axis=1)
                    dsj = jnp.sum(de_dsig_mat, axis=0)

                    def _wr_diag(z):
                        _pl_atomic_add_1d(dq_out, out_b + oi, dqi + dqj, mi, interpret)
                        _pl_atomic_add_1d(ds_out, out_b + oi, dsi + dsj, mi, interpret)
                        _pl_atomic_add_1d(de_out, out_b + oi, de_depsi + de_depsj, mi, interpret)
                        return z

                    def _wr_off(z):
                        _pl_atomic_add_1d(dq_out, out_b + oi, dqi, mi, interpret)
                        _pl_atomic_add_1d(ds_out, out_b + oi, dsi, mi, interpret)
                        _pl_atomic_add_1d(de_out, out_b + oi, de_depsi, mi, interpret)
                        _pl_atomic_add_1d(dq_out, out_b + oj, dqj, mj, interpret)
                        _pl_atomic_add_1d(ds_out, out_b + oj, dsj, mj, interpret)
                        _pl_atomic_add_1d(de_out, out_b + oj, de_depsj, mj, interpret)
                        return z

                    _ = jax.lax.cond(is_diag, _wr_diag, _wr_off, (pid - pid).astype(jnp.int32))
                    return dl0 + jnp.sum(de_dlam_mat)

                return jax.lax.cond(jnp.max(act.astype(jnp.int32)) > 0, _act_bwd, lambda d: d, dl_in)

            return jax.lax.cond(v_tp, _do_p, lambda d: d, dlam_acc)

        dl_tot = jax.lax.fori_loop(jnp.int32(0), jnp.int32(tp_grp), _step, (pid - pid).astype(dtype))
        _pl_store(dl_out, (b_id * n_pblk + pid + z0,), dl_tot[None])

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    dq_f, ds_f, de_f, dl_f = pl.pallas_call(
        _bwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B * N,), dtype),
            jax.ShapeDtypeStruct((B * N,), dtype),
            jax.ShapeDtypeStruct((B * N,), dtype),
            jax.ShapeDtypeStruct((B * n_pblk,), dtype),
        ],
        in_specs=[any_spec] * 17,
        out_specs=[any_spec] * 4,
        grid=(B, n_pblk),
        input_output_aliases={14: 0, 15: 1, 16: 2},
        interpret=interpret,
        compiler_params=_tile_pairs_compiler_params(B * n_pblk) if not interpret else None,
        name="pallas_tile_pairs_param_grads",
    )(
        pos_p[:, :, 0].reshape(-1), pos_p[:, :, 1].reshape(-1), pos_p[:, :, 2].reshape(-1),
        q_p.reshape(-1), sig_p.reshape(-1), eps_p.reshape(-1), msk_p.reshape(-1),
        tp_b[..., 0].reshape(-1), tp_b[..., 1].reshape(-1),
        ef_b.reshape(-1), e14_b.reshape(-1), box_b.reshape(-1), lam_b.reshape(-1), g_b,
        z_dq, z_ds, z_de,
    )
    dq_out = dq_f.reshape(B, N)[:, :N_orig]
    ds_out = ds_f.reshape(B, N)[:, :N_orig]
    de_out = de_f.reshape(B, N)[:, :N_orig]
    dl_out = jnp.sum(dl_f.reshape(B, n_pblk), axis=-1)
    grad_q = (dq_out if b_q else jnp.sum(dq_out, axis=0)).astype(charges.dtype)
    grad_s = (ds_out if b_sig else jnp.sum(ds_out, axis=0)).astype(sigmas.dtype)
    grad_e = (de_out if b_eps else jnp.sum(de_out, axis=0)).astype(epsilons.dtype)
    grad_l = (dl_out.reshape(lam_arr.shape) if b_lam else jnp.sum(dl_out).reshape(lam_arr.shape)).astype(lam_arr.dtype)
    return grad_q, grad_s, grad_e, grad_l
