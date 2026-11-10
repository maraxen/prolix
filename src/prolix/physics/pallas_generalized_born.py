"""Pallas 2D-grid (B, n_act_blocks) kernels for implicit-solvent OBC-II GBSA + LJ/Coulomb (MK-61).

Provides:
  - ``pallas_gbsa_dense``: 2D-grid ``(B, n_act_blocks)`` coalesced 32x32 tile kernel for
    implicit-solvent OBC-II Generalized Born + ACE nonpolar (+ optional fused dense LJ &
    Coulomb with sparse 32-slot gated exclusions) with ``return_forces=True`` and ``custom_vjp``.
  - ``pallas_gbsa_energy_and_forces``: convenience wrapper returning ``(energy, forces)``.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_vjp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from proxide.physics import constants

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.generalized_born import (
    ALPHA_OBC,
    BETA_OBC,
    GAMMA_OBC,
    SURFACE_TENSION,
    compute_pair_integral,
)
from prolix.physics.pallas_bonded import (
    _min_image3,
    _normalize_box,
    _triton_gpu_available,
)
from prolix.physics.pallas_implicit import (
    _pair_integral_deriv,
    pallas_gb_ace_nl,
)

_GB_BLOCK = 32
_MAX_EXCL = 32


def _resolve_gb_active_atoms(
    n_active_atoms: int | None,
    n_total: int,
    atom_mask: jax.Array | None = None,
) -> int:
    if n_active_atoms is not None and int(n_active_atoms) > 0:
        return min(int(n_active_atoms), int(n_total))
    env_val = os.environ.get("PROLIX_ACTIVE_ATOMS", "").strip()
    if env_val:
        try:
            v = int(env_val)
            if v > 0:
                return min(v, int(n_total))
        except ValueError:
            pass
    if atom_mask is not None and not isinstance(atom_mask, jax.core.Tracer):
        try:
            am_np = np.asarray(atom_mask, dtype=bool)
            if am_np.size > 0:
                am_any = np.any(am_np, axis=tuple(range(am_np.ndim - 1))) if am_np.ndim > 1 else am_np
                nz = np.nonzero(am_any)[0]
                if nz.size > 0:
                    return min(max(1, int(nz[-1]) + 1), int(n_total))
        except Exception:
            pass
    return int(n_total)


def _pallas_gbsa_dense_batch(
    axis_size: int,
    in_batched: tuple[bool, ...],
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    lam_arr: jax.Array,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    dielectric_offset: float,
    surface_tension: float,
    include_ace: bool,
    include_lj_coulomb: bool,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    interpret: bool,
    n_active_atoms: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """2D-grid ``(B, n_act_blocks)`` upper-triangular symmetric 32x32 tile OBC-II GBSA + LJ/Coulomb kernel (MK-61)."""
    B = int(axis_size)
    pos_b = positions if in_batched[0] else positions[None, ...]
    q_b = charges if in_batched[1] else charges[None, ...]
    rad_b = radii if in_batched[2] else radii[None, ...]
    sr_b = scaled_radii if in_batched[3] else scaled_radii[None, ...]
    sig_b = sigmas if in_batched[4] else sigmas[None, ...]
    eps_b = epsilons if in_batched[5] else epsilons[None, ...]
    msk_b = atom_mask if in_batched[6] else atom_mask[None, ...]
    ei_b = excl_indices if in_batched[7] else excl_indices[None, ...]
    ev_b = excl_scales_vdw if in_batched[8] else excl_scales_vdw[None, ...]
    ee_b = excl_scales_elec if in_batched[9] else excl_scales_elec[None, ...]
    bx_b = box if in_batched[10] else box[None, ...]
    lam_b = lam_arr.reshape((B if in_batched[11] else 1, -1))[:, 0]

    N = int(pos_b.shape[1])
    orig_dtype = pos_b.dtype
    if N == 0:
        return (
            jnp.zeros((B,), dtype=orig_dtype),
            jnp.zeros((B, 0, 3), dtype=orig_dtype),
            jnp.zeros((B, 0), dtype=charges.dtype),
        )

    dtype = jnp.float32 if (not interpret and orig_dtype == jnp.float64) else orig_dtype
    N_act = _resolve_gb_active_atoms(n_active_atoms, N, msk_b)
    n_blocks = max(1, (N + _GB_BLOCK - 1) // _GB_BLOCK)
    n_act_blocks = max(1, (N_act + _GB_BLOCK - 1) // _GB_BLOCK)
    N_pad = n_blocks * _GB_BLOCK
    pad_n = N_pad - N

    pos_b = pos_b.astype(dtype)
    q_b = q_b.astype(dtype)
    rad_b = rad_b.astype(dtype)
    sr_b = sr_b.astype(dtype)
    sig_b = sig_b.astype(dtype)
    eps_b = eps_b.astype(dtype)
    bx_b = bx_b.astype(dtype)
    lam_b = lam_b.astype(dtype)

    B_pos = pos_b.shape[0]
    B_q = q_b.shape[0]
    B_rad = rad_b.shape[0]
    B_sr = sr_b.shape[0]
    B_sig = sig_b.shape[0]
    B_eps = eps_b.shape[0]
    B_msk = msk_b.shape[0]
    B_ei = ei_b.shape[0]
    B_ev = ev_b.shape[0]
    B_ee = ee_b.shape[0]
    B_box = bx_b.shape[0]
    B_lam = lam_b.shape[0]

    if pad_n > 0:
        pos_pad = jnp.pad(pos_b, ((0, 0), (0, pad_n), (0, 0)))
        q_pad = jnp.pad(q_b, ((0, 0), (0, pad_n)))
        rad_pad = jnp.pad(rad_b, ((0, 0), (0, pad_n)), constant_values=1.5)
        sr_pad = jnp.pad(sr_b, ((0, 0), (0, pad_n)), constant_values=1.2)
        sig_pad = jnp.pad(sig_b, ((0, 0), (0, pad_n)), constant_values=1.0)
        eps_pad = jnp.pad(eps_b, ((0, 0), (0, pad_n)))
        mask_pad = jnp.pad(msk_b.astype(jnp.int32), ((0, 0), (0, pad_n)), constant_values=0)
    else:
        pos_pad = pos_b
        q_pad = q_b
        rad_pad = rad_b
        sr_pad = sr_b
        sig_pad = sig_b
        eps_pad = eps_b
        mask_pad = msk_b.astype(jnp.int32)

    M_raw = int(ei_b.shape[2]) if ei_b.ndim == 3 else 0
    if include_lj_coulomb and M_raw > 0:
        ei_clip = ei_b[:, :, :_MAX_EXCL].astype(jnp.int32)
        ev_clip = ev_b[:, :, :_MAX_EXCL].astype(dtype)
        ee_clip = ee_b[:, :, :_MAX_EXCL].astype(dtype)
        pad_m = max(0, _MAX_EXCL - int(ei_clip.shape[2]))
        pad_n_e = N_pad - int(ei_clip.shape[1])
        if pad_n_e > 0 or pad_m > 0:
            ei_clip = jnp.pad(ei_clip, ((0, 0), (0, pad_n_e), (0, pad_m)), constant_values=-1)
            ev_clip = jnp.pad(ev_clip, ((0, 0), (0, pad_n_e), (0, pad_m)), constant_values=1.0)
            ee_clip = jnp.pad(ee_clip, ((0, 0), (0, pad_n_e), (0, pad_m)), constant_values=1.0)
        valid_e = (ei_clip >= 0) & (ei_clip < N)
        int_max = jnp.int32(0x7FFFFFFF)
        excl_min_pad = jnp.min(jnp.where(valid_e, ei_clip, int_max), axis=-1)
        excl_max_pad = jnp.max(jnp.where(valid_e, ei_clip, jnp.int32(-1)), axis=-1)
        excl_min_flat = jnp.asarray(excl_min_pad.reshape(B_ei * N_pad))
        excl_max_flat = jnp.asarray(excl_max_pad.reshape(B_ei * N_pad))
        excl_idx_flat = jnp.asarray(ei_clip.reshape(B_ei * N_pad * _MAX_EXCL))
        excl_vdw_flat = jnp.asarray(ev_clip.reshape(B_ev * N_pad * _MAX_EXCL))
        excl_elec_flat = jnp.asarray(ee_clip.reshape(B_ee * N_pad * _MAX_EXCL))
        has_excl = True
    else:
        excl_min_flat = jnp.full((1,), 0x7FFFFFFF, dtype=jnp.int32)
        excl_max_flat = jnp.full((1,), -1, dtype=jnp.int32)
        excl_idx_flat = jnp.full((_MAX_EXCL,), -1, dtype=jnp.int32)
        excl_vdw_flat = jnp.ones((_MAX_EXCL,), dtype=dtype)
        excl_elec_flat = jnp.ones((_MAX_EXCL,), dtype=dtype)
        has_excl = False

    px_flat = jnp.asarray(pos_pad[:, :, 0].reshape(B_pos * N_pad))
    py_flat = jnp.asarray(pos_pad[:, :, 1].reshape(B_pos * N_pad))
    pz_flat = jnp.asarray(pos_pad[:, :, 2].reshape(B_pos * N_pad))
    q_flat = jnp.asarray(q_pad.reshape(B_q * N_pad))
    rad_flat = jnp.asarray(rad_pad.reshape(B_rad * N_pad))
    sr_flat = jnp.asarray(sr_pad.reshape(B_sr * N_pad))
    sig_flat = jnp.asarray(sig_pad.reshape(B_sig * N_pad))
    eps_flat = jnp.asarray(eps_pad.reshape(B_eps * N_pad))
    mask_flat = jnp.asarray(mask_pad.reshape(B_msk * N_pad))
    box_flat = jnp.asarray(bx_b.reshape(B_box * 6))
    lam_flat = jnp.asarray(lam_b.reshape(B_lam))

    tau = (1.0 / float(solute_dielectric)) - (1.0 / float(solvent_dielectric))
    any_spec = pl.BlockSpec(memory_space=pl.ANY)

    # --- Pass 1: 2D-grid (B, n_act_blocks) Atomic-Free Row Born Pair Integrals ---
    def _pass1_born_dense_kernel(
        px_ref,
        py_ref,
        pz_ref,
        rad_ref,
        sr_ref,
        mask_ref,
        box_ref,
        i_out_ref,
    ):
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        lane = jnp.arange(_GB_BLOCK, dtype=jnp.int32)
        offs_i = pid * _GB_BLOCK + lane

        base_pos = (b_id * N_pad) if B_pos > 1 else jnp.int32(0)
        base_rad = (b_id * N_pad) if B_rad > 1 else jnp.int32(0)
        base_sr = (b_id * N_pad) if B_sr > 1 else jnp.int32(0)
        base_msk = (b_id * N_pad) if B_msk > 1 else jnp.int32(0)
        base_box = (b_id * 6) if B_box > 1 else jnp.int32(0)
        base_out = b_id * N_pad

        zero = jnp.asarray(0.0, dtype=dtype)
        eps4 = jnp.asarray(1e-4, dtype=dtype)
        eps6 = jnp.asarray(1e-6, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        offset = jnp.asarray(dielectric_offset, dtype=dtype)

        ax = _pl_load(box_ref, (base_box + 0,))
        by = _pl_load(box_ref, (base_box + 1,))
        cz = _pl_load(box_ref, (base_box + 2,))
        bx_t = _pl_load(box_ref, (base_box + 3,))
        cx_t = _pl_load(box_ref, (base_box + 4,))
        cy_t = _pl_load(box_ref, (base_box + 5,))
        box6 = (ax, by, cz, bx_t, cx_t, cy_t)

        mask_i = (_pl_load(mask_ref, (base_msk + offs_i,)) > 0) & (offs_i < N_act)
        px_i = _pl_load(px_ref, (base_pos + offs_i,))
        py_i = _pl_load(py_ref, (base_pos + offs_i,))
        pz_i = _pl_load(pz_ref, (base_pos + offs_i,))
        rad_i = jnp.maximum(_pl_load(rad_ref, (base_rad + offs_i,)), eps6)
        or_i = jnp.maximum(rad_i - offset, eps4)

        def _tile_body(t_j, acc_I):
            offs_j = t_j * _GB_BLOCK + lane
            px_j = _pl_load(px_ref, (base_pos + offs_j,))
            py_j = _pl_load(py_ref, (base_pos + offs_j,))
            pz_j = _pl_load(pz_ref, (base_pos + offs_j,))
            sr_j = _pl_load(sr_ref, (base_sr + offs_j,))
            mask_j = (_pl_load(mask_ref, (base_msk + offs_j,)) > 0) & (offs_j < N_act)

            active = mask_i[:, None] & mask_j[None, :] & (offs_i[:, None] != offs_j[None, :])
            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j[None, :], py_i[:, None] - py_j[None, :], pz_i[:, None] - pz_j[None, :]),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            I_ij = compute_pair_integral(dist, or_i[:, None], sr_j[None, :])
            return acc_I + jnp.sum(jnp.where(active, I_ij, zero), axis=1)

        I_i = jax.lax.fori_loop(
            0, n_act_blocks, _tile_body, jnp.zeros((_GB_BLOCK,), dtype=dtype)
        )
        _pl_store(i_out_ref, (base_out + offs_i,), jnp.where(mask_i, I_i, zero))

    i_flat = pl.pallas_call(
        _pass1_born_dense_kernel,
        out_shape=jax.ShapeDtypeStruct((B * N_pad,), dtype),
        in_specs=[any_spec] * 7,
        out_specs=any_spec,
        grid=(B, n_act_blocks),
        interpret=interpret,
        compiler_params=resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=B * n_act_blocks)
        if not interpret
        else None,
        name="pallas_gbsa_pass1_born_dense",
    )(px_flat, py_flat, pz_flat, rad_flat, sr_flat, mask_flat, box_flat)

    i_pad = i_flat.reshape(B, N_pad)

    # Host O(B * N) OBC-II Born radii, dB/dI, self GB + ACE nonpolar energy and d(E_self + E_ace)/dB
    one_f = jnp.asarray(1.0, dtype=dtype)
    zero_f = jnp.asarray(0.0, dtype=dtype)
    eps4_f = jnp.asarray(1e-4, dtype=dtype)
    eps6_f = jnp.asarray(1e-6, dtype=dtype)
    offset_f = jnp.asarray(dielectric_offset, dtype=dtype)
    act_mask_b = (mask_pad > 0) & (jnp.arange(N_pad, dtype=jnp.int32)[None, :] < N_act)
    mask_f = act_mask_b.astype(dtype)

    rad_safe_pad = jnp.maximum(rad_pad, eps6_f)
    or_pad = jnp.maximum(rad_safe_pad - offset_f, eps4_f)
    psi_pad = or_pad * i_pad
    alpha_obc = jnp.asarray(ALPHA_OBC, dtype=dtype)
    beta_obc = jnp.asarray(BETA_OBC, dtype=dtype)
    gamma_obc = jnp.asarray(GAMMA_OBC, dtype=dtype)
    arg_pad = alpha_obc * psi_pad - beta_obc * (psi_pad * psi_pad) + gamma_obc * (psi_pad * psi_pad * psi_pad)
    th_pad = jnp.tanh(arg_pad)
    r_safe_pad = jnp.maximum(rad_safe_pad, eps4_f)
    b_inv_raw_pad = one_f / or_pad - th_pad / r_safe_pad
    b_inv_safe_pad = jnp.maximum(b_inv_raw_pad, eps4_f)
    born_pad = jnp.where(act_mask_b, one_f / b_inv_safe_pad, one_f)
    darg_dpsi_pad = (
        alpha_obc
        - jnp.asarray(2.0, dtype=dtype) * beta_obc * psi_pad
        + jnp.asarray(3.0, dtype=dtype) * gamma_obc * (psi_pad * psi_pad)
    )
    db_di_pad = jnp.where(
        act_mask_b & (b_inv_raw_pad > eps4_f),
        (born_pad * born_pad) * (or_pad / r_safe_pad) * (one_f - th_pad * th_pad) * darg_dpsi_pad,
        zero_f,
    )
    born_flat = jnp.asarray(born_pad.reshape(B * N_pad))

    prefactor = jnp.asarray(-0.5 * float(coulomb_constant) * tau, dtype=dtype)

    e_gb_self_vec = prefactor * mask_f * (q_pad * q_pad) / born_pad
    de_self_db = -e_gb_self_vec / born_pad

    if include_ace:
        st_scale = float(surface_tension) / float(SURFACE_TENSION) if SURFACE_TENSION > 0.0 else 1.0
        radius_nm = (rad_pad - jnp.asarray(dielectric_offset, dtype=dtype)) * jnp.asarray(
            0.1, dtype=dtype
        ) + jnp.asarray(0.009, dtype=dtype)
        born_nm = born_pad * jnp.asarray(0.1, dtype=dtype)
        e_ace_vec = (
            mask_f
            * jnp.asarray(28.3919551 * 0.239006 * st_scale, dtype=dtype)
            * ((radius_nm + jnp.asarray(0.14, dtype=dtype)) ** 2)
            * ((radius_nm / born_nm) ** 6)
        )
        de_ace_db = -jnp.asarray(6.0, dtype=dtype) * e_ace_vec / born_pad
    else:
        e_ace_vec = jnp.zeros_like(born_pad)
        de_ace_db = jnp.zeros_like(born_pad)

    # --- Pass 2: 2D-grid (B, n_act_blocks) Atomic-Free Row Pair GB (+ optional LJ & Coulomb) ---
    def _pass2_pair_dense_kernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        born_ref,
        sig_ref,
        eps_ref,
        mask_ref,
        excl_min_ref,
        excl_max_ref,
        excl_idx_ref,
        excl_vdw_ref,
        excl_elec_ref,
        box_ref,
        lam_ref,
        e_out_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
        dedb_out_ref,
        gq_out_ref,
    ):
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        lane = jnp.arange(_GB_BLOCK, dtype=jnp.int32)
        offs_i = pid * _GB_BLOCK + lane

        base_pos = (b_id * N_pad) if B_pos > 1 else jnp.int32(0)
        base_q = (b_id * N_pad) if B_q > 1 else jnp.int32(0)
        base_born = b_id * N_pad
        base_sig = (b_id * N_pad) if B_sig > 1 else jnp.int32(0)
        base_eps = (b_id * N_pad) if B_eps > 1 else jnp.int32(0)
        base_msk = (b_id * N_pad) if B_msk > 1 else jnp.int32(0)
        base_ei = (b_id * N_pad) if (has_excl and B_ei > 1) else jnp.int32(0)
        base_ev = (b_id * N_pad * _MAX_EXCL) if (has_excl and B_ev > 1) else jnp.int32(0)
        base_ee = (b_id * N_pad * _MAX_EXCL) if (has_excl and B_ee > 1) else jnp.int32(0)
        base_box = (b_id * 6) if B_box > 1 else jnp.int32(0)
        base_lam = b_id if B_lam > 1 else jnp.int32(0)
        base_out = b_id * N_pad

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)
        quarter = jnp.asarray(0.25, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        pref = jnp.asarray(-0.5 * float(coulomb_constant) * tau, dtype=dtype)
        coul_c = jnp.asarray(float(coulomb_constant), dtype=dtype)

        ax = _pl_load(box_ref, (base_box + 0,))
        by = _pl_load(box_ref, (base_box + 1,))
        cz = _pl_load(box_ref, (base_box + 2,))
        bx_t = _pl_load(box_ref, (base_box + 3,))
        cx_t = _pl_load(box_ref, (base_box + 4,))
        cy_t = _pl_load(box_ref, (base_box + 5,))
        box6 = (ax, by, cz, bx_t, cx_t, cy_t)

        lam = _pl_load(lam_ref, (base_lam,))
        soft_term = jnp.asarray(float(softcore_alpha), dtype=dtype) * (
            jnp.maximum(one - lam, zero) ** int(softcore_exponent)
        )

        mask_i = (_pl_load(mask_ref, (base_msk + offs_i,)) > 0) & (offs_i < N_act)
        px_i = _pl_load(px_ref, (base_pos + offs_i,))
        py_i = _pl_load(py_ref, (base_pos + offs_i,))
        pz_i = _pl_load(pz_ref, (base_pos + offs_i,))
        q_i = _pl_load(q_ref, (base_q + offs_i,))
        b_i = _pl_load(born_ref, (base_born + offs_i,))

        if include_lj_coulomb:
            sig_i = _pl_load(sig_ref, (base_sig + offs_i,))
            eps_i = _pl_load(eps_ref, (base_eps + offs_i,))
            if has_excl:
                e_min_i = _pl_load(excl_min_ref, (base_ei + offs_i,))
                e_max_i = _pl_load(excl_max_ref, (base_ei + offs_i,))
            else:
                e_min_i = jnp.full((_GB_BLOCK,), 0x7FFFFFFF, dtype=jnp.int32)
                e_max_i = jnp.full((_GB_BLOCK,), -1, dtype=jnp.int32)
        else:
            sig_i = zero
            eps_i = zero
            e_min_i = jnp.full((_GB_BLOCK,), 0x7FFFFFFF, dtype=jnp.int32)
            e_max_i = jnp.full((_GB_BLOCK,), -1, dtype=jnp.int32)

        def _compute_tile_excl_weights(t_j, offs_j):
            if not (include_lj_coulomb and has_excl):
                return one, one
            j_lo = t_j * _GB_BLOCK
            j_hi = j_lo + (_GB_BLOCK - 1)
            row_has_excl = mask_i & (e_min_i <= j_hi) & (e_max_i >= j_lo)
            tile_has_excl = jnp.max(row_has_excl.astype(jnp.int32)) > jnp.int32(0)

            def _compute_excl_weights(_):
                ones_mat = jnp.ones((_GB_BLOCK, _GB_BLOCK), dtype=dtype)

                def _slot_body(m_slot, w_carry):
                    w_v_acc, w_e_acc = w_carry
                    e_offs = offs_i * _MAX_EXCL + m_slot
                    e_idx_m = _pl_load(excl_idx_ref, (base_ei * _MAX_EXCL + e_offs,))
                    e_v_m = _pl_load(excl_vdw_ref, (base_ev + e_offs,))
                    e_e_m = _pl_load(excl_elec_ref, (base_ee + e_offs,))
                    match_m = row_has_excl[:, None] & (e_idx_m[:, None] == offs_j[None, :])
                    w_v_acc = jnp.where(match_m, e_v_m[:, None], w_v_acc)
                    w_e_acc = jnp.where(match_m, e_e_m[:, None], w_e_acc)
                    return w_v_acc, w_e_acc

                return jax.lax.fori_loop(0, _MAX_EXCL, _slot_body, (ones_mat, ones_mat))

            return jax.lax.cond(
                tile_has_excl,
                _compute_excl_weights,
                lambda _: (
                    jnp.ones((_GB_BLOCK, _GB_BLOCK), dtype=dtype),
                    jnp.ones((_GB_BLOCK, _GB_BLOCK), dtype=dtype),
                ),
                None,
            )

        z_vec = jnp.zeros((_GB_BLOCK,), dtype=dtype)
        init_carry = (z_vec, z_vec, z_vec, z_vec, z_vec, z_vec)

        def _tile_body(t_j, carry):
            acc_e, acc_fx, acc_fy, acc_fz, acc_dedb, acc_gq = carry
            offs_j = t_j * _GB_BLOCK + lane
            px_j = _pl_load(px_ref, (base_pos + offs_j,))
            py_j = _pl_load(py_ref, (base_pos + offs_j,))
            pz_j = _pl_load(pz_ref, (base_pos + offs_j,))
            q_j = _pl_load(q_ref, (base_q + offs_j,))
            b_j = _pl_load(born_ref, (base_born + offs_j,))
            mask_j = (_pl_load(mask_ref, (base_msk + offs_j,)) > 0) & (offs_j < N_act)

            active = mask_i[:, None] & mask_j[None, :] & (offs_i[:, None] != offs_j[None, :])
            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j[None, :], py_i[:, None] - py_j[None, :], pz_i[:, None] - pz_j[None, :]),
                box6,
                has_box,
            )
            r_sq = dx * dx + dy * dy + dz * dz + eps12

            p_ij = b_i[:, None] * b_j[None, :]
            u_ij = r_sq / (four * p_ij)
            e_ij = jnp.exp(zero - u_ij)
            f_gb_sq = r_sq + p_ij * e_ij
            f_gb = jnp.sqrt(f_gb_sq)
            f_gb_cubed = f_gb * f_gb_sq

            qq = q_i[:, None] * q_j[None, :]
            e_gb_pair = jnp.where(active, pref * qq / f_gb, zero)
            f_gb_coeff = jnp.where(
                active,
                two * pref * qq * (one - quarter * e_ij) / f_gb_cubed,
                zero,
            )
            dedb_common = jnp.where(
                active,
                (zero - pref) * qq * e_ij * (one + u_ij) / f_gb_cubed,
                zero,
            )
            gq_common_gb = jnp.where(active, two * pref / f_gb, zero)

            if include_lj_coulomb:
                sig_j = _pl_load(sig_ref, (base_sig + offs_j,))
                eps_j = _pl_load(eps_ref, (base_eps + offs_j,))
                dist = jnp.sqrt(r_sq)
                inv_dist = jnp.where(active, one / dist, zero)
                inv_r_sq = inv_dist * inv_dist
                w_vdw, w_elec = _compute_tile_excl_weights(t_j, offs_j)

                sig_ij = half * (sig_i[:, None] + sig_j[None, :])
                sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
                eps_ij = jnp.sqrt(
                    jnp.maximum(eps_i[:, None] * eps_j[None, :], zero) + jnp.asarray(1e-30, dtype=dtype)
                )
                u6 = (dist / sig_safe) ** 6
                x_inv6 = one / jnp.maximum(soft_term + u6, jnp.asarray(1e-6, dtype=dtype))
                e_lj = jnp.where(
                    active,
                    half * w_vdw * four * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6),
                    zero,
                )
                f_lj_scalar = jnp.where(
                    active,
                    w_vdw
                    * jnp.asarray(24.0, dtype=dtype)
                    * eps_ij
                    * lam
                    * (two * x_inv6 - one)
                    * (x_inv6 * x_inv6)
                    * u6
                    * inv_r_sq,
                    zero,
                )
                valid_coul = active & (dist >= jnp.asarray(1e-4, dtype=dtype))
                e_coul = jnp.where(valid_coul, half * coul_c * qq * w_elec * inv_dist, zero)
                f_coul_scalar = jnp.where(
                    valid_coul,
                    coul_c * qq * w_elec * inv_dist * inv_r_sq,
                    zero,
                )
                gq_common_coul = jnp.where(valid_coul, coul_c * w_elec * inv_dist, zero)

                e_tot = e_gb_pair + e_lj + e_coul
                f_tot_coeff = f_gb_coeff + f_lj_scalar + f_coul_scalar
                gq_common = gq_common_gb + gq_common_coul
            else:
                e_tot = e_gb_pair
                f_tot_coeff = f_gb_coeff
                gq_common = gq_common_gb

            return (
                acc_e + jnp.sum(e_tot, axis=1),
                acc_fx + jnp.sum(f_tot_coeff * dx, axis=1),
                acc_fy + jnp.sum(f_tot_coeff * dy, axis=1),
                acc_fz + jnp.sum(f_tot_coeff * dz, axis=1),
                acc_dedb + jnp.sum(dedb_common * b_j[None, :], axis=1),
                acc_gq + jnp.sum(gq_common * q_j[None, :], axis=1),
            )

        e_p, fx_d, fy_d, fz_d, dedb_p, gq_p = jax.lax.fori_loop(
            0, n_act_blocks, _tile_body, init_carry
        )
        gq_self = jnp.where(mask_i, two * pref * (q_i / b_i), zero)

        _pl_store(e_out_ref, (base_out + offs_i,), jnp.where(mask_i, e_p, zero))
        _pl_store(fx_out_ref, (base_out + offs_i,), jnp.where(mask_i, fx_d, zero))
        _pl_store(fy_out_ref, (base_out + offs_i,), jnp.where(mask_i, fy_d, zero))
        _pl_store(fz_out_ref, (base_out + offs_i,), jnp.where(mask_i, fz_d, zero))
        _pl_store(dedb_out_ref, (base_out + offs_i,), jnp.where(mask_i, dedb_p, zero))
        _pl_store(gq_out_ref, (base_out + offs_i,), jnp.where(mask_i, gq_p + gq_self, zero))

    (
        e_pair_flat,
        fx_dir_flat,
        fy_dir_flat,
        fz_dir_flat,
        de_pair_db_flat,
        grad_q_flat,
    ) = pl.pallas_call(
        _pass2_pair_dense_kernel,
        out_shape=[jax.ShapeDtypeStruct((B * N_pad,), dtype) for _ in range(6)],
        in_specs=[any_spec] * 15,
        out_specs=[any_spec] * 6,
        grid=(B, n_act_blocks),
        interpret=interpret,
        compiler_params=resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=B * n_act_blocks)
        if not interpret
        else None,
        name="pallas_gbsa_pass2_pair_dense",
    )(
        px_flat,
        py_flat,
        pz_flat,
        q_flat,
        born_flat,
        sig_flat,
        eps_flat,
        mask_flat,
        excl_min_flat,
        excl_max_flat,
        excl_idx_flat,
        excl_vdw_flat,
        excl_elec_flat,
        box_flat,
        lam_flat,
    )

    e_pair_pad = e_pair_flat.reshape(B, N_pad)
    de_pair_db_pad = de_pair_db_flat.reshape(B, N_pad)
    grad_q_pad = grad_q_flat.reshape(B, N_pad)

    # --- Pass 3: 2D-grid (B, n_act_blocks) Atomic-Free Born Chain-Rule Forces + Direct Force Sum ---
    w_pad = (de_self_db + de_ace_db + de_pair_db_pad) * db_di_pad
    w_flat = jnp.asarray(w_pad.reshape(B * N_pad))
    or_flat = jnp.asarray(or_pad.reshape(B_rad * N_pad))

    def _pass3_born_chain_dense_kernel(
        px_ref,
        py_ref,
        pz_ref,
        or_ref,
        sr_ref,
        w_ref,
        mask_ref,
        box_ref,
        fx_dir_ref,
        fy_dir_ref,
        fz_dir_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        lane = jnp.arange(_GB_BLOCK, dtype=jnp.int32)
        offs_i = pid * _GB_BLOCK + lane

        base_pos = (b_id * N_pad) if B_pos > 1 else jnp.int32(0)
        base_or = (b_id * N_pad) if B_rad > 1 else jnp.int32(0)
        base_sr = (b_id * N_pad) if B_sr > 1 else jnp.int32(0)
        base_w = b_id * N_pad
        base_msk = (b_id * N_pad) if B_msk > 1 else jnp.int32(0)
        base_box = (b_id * 6) if B_box > 1 else jnp.int32(0)
        base_out = b_id * N_pad

        zero = jnp.asarray(0.0, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)

        ax = _pl_load(box_ref, (base_box + 0,))
        by = _pl_load(box_ref, (base_box + 1,))
        cz = _pl_load(box_ref, (base_box + 2,))
        bx_t = _pl_load(box_ref, (base_box + 3,))
        cx_t = _pl_load(box_ref, (base_box + 4,))
        cy_t = _pl_load(box_ref, (base_box + 5,))
        box6 = (ax, by, cz, bx_t, cx_t, cy_t)

        mask_i = (_pl_load(mask_ref, (base_msk + offs_i,)) > 0) & (offs_i < N_act)
        px_i = _pl_load(px_ref, (base_pos + offs_i,))
        py_i = _pl_load(py_ref, (base_pos + offs_i,))
        pz_i = _pl_load(pz_ref, (base_pos + offs_i,))
        or_i = _pl_load(or_ref, (base_or + offs_i,))
        sr_i = _pl_load(sr_ref, (base_sr + offs_i,))
        w_i = _pl_load(w_ref, (base_w + offs_i,))

        init_fx = _pl_load(fx_dir_ref, (base_out + offs_i,))
        init_fy = _pl_load(fy_dir_ref, (base_out + offs_i,))
        init_fz = _pl_load(fz_dir_ref, (base_out + offs_i,))

        def _tile_body(t_j, carry):
            acc_fx, acc_fy, acc_fz = carry
            offs_j = t_j * _GB_BLOCK + lane
            px_j = _pl_load(px_ref, (base_pos + offs_j,))
            py_j = _pl_load(py_ref, (base_pos + offs_j,))
            pz_j = _pl_load(pz_ref, (base_pos + offs_j,))
            or_j = _pl_load(or_ref, (base_or + offs_j,))
            sr_j = _pl_load(sr_ref, (base_sr + offs_j,))
            w_j = _pl_load(w_ref, (base_w + offs_j,))
            mask_j = (_pl_load(mask_ref, (base_msk + offs_j,)) > 0) & (offs_j < N_act)

            active = mask_i[:, None] & mask_j[None, :] & (offs_i[:, None] != offs_j[None, :])
            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j[None, :], py_i[:, None] - py_j[None, :], pz_i[:, None] - pz_j[None, :]),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            dI_ij_dr = _pair_integral_deriv(dist, or_i[:, None], sr_j[None, :])
            dI_ji_dr = _pair_integral_deriv(dist, or_j[None, :], sr_i[:, None])

            f_scalar = jnp.where(
                active,
                (zero - (w_i[:, None] * dI_ij_dr + w_j[None, :] * dI_ji_dr)) / dist,
                zero,
            )
            return (
                acc_fx + jnp.sum(f_scalar * dx, axis=1),
                acc_fy + jnp.sum(f_scalar * dy, axis=1),
                acc_fz + jnp.sum(f_scalar * dz, axis=1),
            )

        fx_b, fy_b, fz_b = jax.lax.fori_loop(
            0, n_act_blocks, _tile_body, (init_fx, init_fy, init_fz)
        )
        _pl_store(fx_out_ref, (base_out + offs_i,), jnp.where(mask_i, fx_b, zero))
        _pl_store(fy_out_ref, (base_out + offs_i,), jnp.where(mask_i, fy_b, zero))
        _pl_store(fz_out_ref, (base_out + offs_i,), jnp.where(mask_i, fz_b, zero))

    fx_tot_flat, fy_tot_flat, fz_tot_flat = pl.pallas_call(
        _pass3_born_chain_dense_kernel,
        out_shape=[jax.ShapeDtypeStruct((B * N_pad,), dtype) for _ in range(3)],
        in_specs=[any_spec] * 11,
        out_specs=[any_spec] * 3,
        grid=(B, n_act_blocks),
        interpret=interpret,
        compiler_params=resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=B * n_act_blocks)
        if not interpret
        else None,
        name="pallas_gbsa_pass3_born_chain_dense",
    )(
        px_flat,
        py_flat,
        pz_flat,
        or_flat,
        sr_flat,
        w_flat,
        mask_flat,
        box_flat,
        fx_dir_flat,
        fy_dir_flat,
        fz_dir_flat,
    )

    total_energy = jnp.sum(
        (e_gb_self_vec[:, :N_act] + e_ace_vec[:, :N_act] + e_pair_pad[:, :N_act]).astype(orig_dtype),
        axis=-1,
    )
    forces = jnp.stack(
        [
            fx_tot_flat.reshape(B, N_pad)[:, :N],
            fy_tot_flat.reshape(B, N_pad)[:, :N],
            fz_tot_flat.reshape(B, N_pad)[:, :N],
        ],
        axis=-1,
    ).astype(orig_dtype)
    return total_energy, forces, grad_q_pad[:, :N].astype(charges.dtype)


def _pallas_gbsa_dense_vmappable(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    lam_arr: jax.Array,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    dielectric_offset: float,
    surface_tension: float,
    include_ace: bool,
    include_lj_coulomb: bool,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    interpret: bool,
    n_active_atoms: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    static_args = (
        bool(has_box),
        float(solvent_dielectric),
        float(solute_dielectric),
        float(dielectric_offset),
        float(surface_tension),
        bool(include_ace),
        bool(include_lj_coulomb),
        float(softcore_alpha),
        int(softcore_exponent),
        float(coulomb_constant),
        bool(interpret),
        None if n_active_atoms is None else int(n_active_atoms),
    )

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, rad, sr, sig, eps, msk, ei, ev, ee, bx, lm):
        e_b, f_b, gq_b = _pallas_gbsa_dense_batch(
            1,
            (False,) * 12,
            pos,
            q,
            rad,
            sr,
            sig,
            eps,
            msk,
            ei,
            ev,
            ee,
            bx,
            lm,
            *static_args,
        )
        return e_b[0], f_b[0], gq_b[0]

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, rad, sr, sig, eps, msk, ei, ev, ee, bx, lm):
        e_b, f_b, gq_b = _pallas_gbsa_dense_batch(
            int(axis_size),
            tuple(bool(b) for b in in_batched),
            pos,
            q,
            rad,
            sr,
            sig,
            eps,
            msk,
            ei,
            ev,
            ee,
            bx,
            lm,
            *static_args,
        )
        return (e_b, f_b, gq_b), (True, True, True)

    if positions.ndim == 3:
        B = int(positions.shape[0])
        in_b = (
            True,
            charges.ndim == 2,
            radii.ndim == 2,
            scaled_radii.ndim == 2,
            sigmas.ndim == 2,
            epsilons.ndim == 2,
            atom_mask.ndim == 2,
            excl_indices.ndim == 3,
            excl_scales_vdw.ndim == 3,
            excl_scales_elec.ndim == 3,
            box.ndim == 2,
            lam_arr.shape[0] > 1,
        )
        return _pallas_gbsa_dense_batch(
            B,
            in_b,
            positions,
            charges,
            radii,
            scaled_radii,
            sigmas,
            epsilons,
            atom_mask,
            excl_indices,
            excl_scales_vdw,
            excl_scales_elec,
            box,
            lam_arr,
            *static_args,
        )

    return _vmappable(
        positions,
        charges,
        radii,
        scaled_radii,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        lam_arr,
    )


@functools.partial(custom_vjp, nondiff_argnums=tuple(range(12, 24)))
def _pallas_gbsa_dense_diff(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    excl_indices: jax.Array,
    excl_scales_vdw: jax.Array,
    excl_scales_elec: jax.Array,
    box: jax.Array,
    lam_arr: jax.Array,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    dielectric_offset: float,
    surface_tension: float,
    include_ace: bool,
    include_lj_coulomb: bool,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    interpret: bool,
    n_active_atoms: int | None,
) -> jax.Array:
    e, _, _ = _pallas_gbsa_dense_vmappable(
        positions,
        charges,
        radii,
        scaled_radii,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        lam_arr,
        has_box,
        solvent_dielectric,
        solute_dielectric,
        dielectric_offset,
        surface_tension,
        include_ace,
        include_lj_coulomb,
        softcore_alpha,
        softcore_exponent,
        coulomb_constant,
        interpret,
        n_active_atoms,
    )
    return e


def _unwrap_primal(x):
    return x.value if hasattr(x, "perturbed") else x


def _is_perturbed(x, default: bool = True) -> bool:
    return bool(x.perturbed) if hasattr(x, "perturbed") else bool(default)


def _gbsa_dense_param_energy_one(
    pos_1: jax.Array,
    q_1: jax.Array,
    rad_1: jax.Array,
    sr_1: jax.Array,
    sig_1: jax.Array,
    eps_1: jax.Array,
    m_1: jax.Array,
    ei_1: jax.Array,
    ev_1: jax.Array,
    box_1: jax.Array,
    lam_1: jax.Array,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    dielectric_offset: float,
    surface_tension: float,
    include_ace: bool,
    include_lj_coulomb: bool,
    softcore_alpha: float,
    softcore_exponent: int,
    coulomb_constant: float,
    n_active_atoms: int | None,
) -> jax.Array:
    N = int(pos_1.shape[0])
    dtype = pos_1.dtype
    n_act = N if n_active_atoms is None else min(int(n_active_atoms), N)
    idx = jnp.arange(N, dtype=jnp.int32)
    m_bool = (m_1 > 0) & (idx < n_act)
    box6 = (box_1[0], box_1[1], box_1[2], box_1[3], box_1[4], box_1[5])
    dx, dy, dz = _min_image3(
        (
            pos_1[:, None, 0] - pos_1[None, :, 0],
            pos_1[:, None, 1] - pos_1[None, :, 1],
            pos_1[:, None, 2] - pos_1[None, :, 2],
        ),
        box6,
        has_box,
    )
    r_sq = dx * dx + dy * dy + dz * dz + jnp.asarray(1e-12, dtype=dtype)
    dist = jnp.sqrt(r_sq)
    active = m_bool[:, None] & m_bool[None, :] & (idx[:, None] != idx[None, :])
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    rad_safe = jnp.maximum(rad_1.astype(dtype), jnp.asarray(1e-6, dtype=dtype))
    or_1 = jnp.maximum(
        rad_safe - jnp.asarray(dielectric_offset, dtype=dtype),
        jnp.asarray(1e-4, dtype=dtype),
    )
    sr_w = sr_1.astype(dtype)
    I_ij = compute_pair_integral(dist, or_1[:, None], sr_w[None, :])
    I_i = jnp.sum(jnp.where(active, I_ij, zero), axis=1)
    psi = or_1 * I_i
    arg = (
        jnp.asarray(ALPHA_OBC, dtype=dtype) * psi
        - jnp.asarray(BETA_OBC, dtype=dtype) * (psi * psi)
        + jnp.asarray(GAMMA_OBC, dtype=dtype) * (psi * psi * psi)
    )
    th = jnp.tanh(arg)
    r_safe = jnp.maximum(rad_safe, jnp.asarray(1e-4, dtype=dtype))
    b_inv = jnp.maximum(one / or_1 - th / r_safe, jnp.asarray(1e-4, dtype=dtype))
    born_r = jnp.where(m_bool, one / b_inv, one)

    tau = (1.0 / float(solute_dielectric)) - (1.0 / float(solvent_dielectric))
    pref = jnp.asarray(-0.5 * float(coulomb_constant) * tau, dtype=dtype)
    q_w = q_1.astype(dtype)
    e_gb_self = jnp.sum(jnp.where(m_bool, pref * (q_w * q_w) / born_r, zero))
    p_ij = born_r[:, None] * born_r[None, :]
    u_ij = r_sq / (jnp.asarray(4.0, dtype=dtype) * p_ij)
    f_gb = jnp.sqrt(r_sq + p_ij * jnp.exp(zero - u_ij))
    e_gb_pair = jnp.sum(jnp.where(active, pref * (q_w[:, None] * q_w[None, :]) / f_gb, zero))
    e_tot = e_gb_self + e_gb_pair

    if include_ace:
        st_scale = float(surface_tension) / float(SURFACE_TENSION) if SURFACE_TENSION > 0.0 else 1.0
        radius_nm = (rad_1.astype(dtype) - jnp.asarray(dielectric_offset, dtype=dtype)) * jnp.asarray(
            0.1, dtype=dtype
        ) + jnp.asarray(0.009, dtype=dtype)
        born_nm = born_r * jnp.asarray(0.1, dtype=dtype)
        e_ace = jnp.sum(
            jnp.where(
                m_bool,
                jnp.asarray(28.3919551 * 0.239006 * st_scale, dtype=dtype)
                * ((radius_nm + jnp.asarray(0.14, dtype=dtype)) ** 2)
                * ((radius_nm / born_nm) ** 6),
                zero,
            )
        )
        e_tot = e_tot + e_ace

    if include_lj_coulomb:
        if ei_1.shape[-1] > 0:
            valid_e = (ei_1 >= 0) & (ei_1 < N)
            match_e = valid_e[:, :, None] & (ei_1[:, :, None] == idx[None, None, :])
            any_e = jnp.any(match_e, axis=1)
            w_vdw = jnp.where(
                any_e,
                jnp.max(jnp.where(match_e, ev_1.astype(dtype)[:, :, None], zero), axis=1),
                one,
            )
        else:
            w_vdw = one
        sig_w = sig_1.astype(dtype)
        eps_w = eps_1.astype(dtype)
        sig_ij = jnp.asarray(0.5, dtype=dtype) * (sig_w[:, None] + sig_w[None, :])
        sig_safe = jnp.where(active, jnp.maximum(sig_ij, jnp.asarray(1e-4, dtype=dtype)), one)
        eps_ij = jnp.sqrt(
            jnp.maximum(eps_w[:, None] * eps_w[None, :], zero) + jnp.asarray(1e-30, dtype=dtype)
        )
        lam = lam_1.reshape(()).astype(dtype)
        soft_term = jnp.asarray(float(softcore_alpha), dtype=dtype) * (
            jnp.maximum(one - lam, zero) ** int(softcore_exponent)
        )
        u6 = (dist / sig_safe) ** 6
        x_inv6 = one / jnp.maximum(soft_term + u6, jnp.asarray(1e-6, dtype=dtype))
        e_lj = jnp.sum(
            jnp.where(
                active,
                jnp.asarray(0.5, dtype=dtype)
                * w_vdw
                * jnp.asarray(4.0, dtype=dtype)
                * eps_ij
                * lam
                * (x_inv6 * x_inv6 - x_inv6),
                zero,
            )
        )
        e_tot = e_tot + e_lj

    return e_tot.astype(dtype)


def _pallas_gbsa_dense_fwd(
    positions,
    charges,
    radii,
    scaled_radii,
    sigmas,
    epsilons,
    atom_mask,
    excl_indices,
    excl_scales_vdw,
    excl_scales_elec,
    box,
    lam_arr,
    has_box,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
    include_lj_coulomb,
    softcore_alpha,
    softcore_exponent,
    coulomb_constant,
    interpret,
    n_active_atoms,
):
    pos_v = _unwrap_primal(positions)
    q_v = _unwrap_primal(charges)
    rad_v = _unwrap_primal(radii)
    sr_v = _unwrap_primal(scaled_radii)
    sig_v = _unwrap_primal(sigmas)
    eps_v = _unwrap_primal(epsilons)
    msk_v = _unwrap_primal(atom_mask)
    ei_v = _unwrap_primal(excl_indices)
    ev_v = _unwrap_primal(excl_scales_vdw)
    ee_v = _unwrap_primal(excl_scales_elec)
    box_v = _unwrap_primal(box)
    lam_v = _unwrap_primal(lam_arr)
    params_perturbed = any(
        _is_perturbed(x, default=True) for x in (radii, scaled_radii, sigmas, epsilons)
    )
    e, forces, grad_q = _pallas_gbsa_dense_vmappable(
        pos_v,
        q_v,
        rad_v,
        sr_v,
        sig_v,
        eps_v,
        msk_v,
        ei_v,
        ev_v,
        ee_v,
        box_v,
        lam_v,
        has_box,
        solvent_dielectric,
        solute_dielectric,
        dielectric_offset,
        surface_tension,
        include_ace,
        include_lj_coulomb,
        softcore_alpha,
        softcore_exponent,
        coulomb_constant,
        interpret,
        n_active_atoms,
    )
    res = (
        forces,
        grad_q,
        pos_v,
        q_v,
        rad_v,
        sr_v,
        sig_v,
        eps_v,
        msk_v,
        ei_v,
        ev_v,
        ee_v,
        box_v,
        lam_v,
        params_perturbed,
    )
    return e, res


def _pallas_gbsa_dense_bwd(
    has_box,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
    include_lj_coulomb,
    softcore_alpha,
    softcore_exponent,
    coulomb_constant,
    interpret,
    n_active_atoms,
    res,
    g,
):
    del interpret
    (
        forces,
        grad_q,
        positions,
        charges,
        radii,
        scaled_radii,
        sigmas,
        epsilons,
        atom_mask,
        excl_indices,
        excl_scales_vdw,
        excl_scales_elec,
        box,
        lam_arr,
        params_perturbed,
    ) = res
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return (
            jnp.zeros_like(positions),
            jnp.zeros_like(charges),
            jnp.zeros_like(radii),
            jnp.zeros_like(scaled_radii),
            jnp.zeros_like(sigmas),
            jnp.zeros_like(epsilons),
            jnp.zeros_like(atom_mask),
            jnp.zeros_like(excl_indices),
            jnp.zeros_like(excl_scales_vdw),
            jnp.zeros_like(excl_scales_elec),
            jnp.zeros_like(box),
            jnp.zeros_like(lam_arr),
        )
    g_arr = jnp.asarray(g, dtype=positions.dtype)
    grad_pos = -g_arr[..., None, None] * forces if g_arr.ndim > 0 else -g_arr * forces
    gq_scaled = g_arr[..., None] * grad_q if g_arr.ndim > 0 else g_arr * grad_q
    while gq_scaled.ndim > charges.ndim:
        gq_scaled = jnp.sum(gq_scaled, axis=0)

    if params_perturbed:
        if positions.ndim == 2:
            _, vjp_fn = jax.vjp(
                lambda r_in, sr_in, sg_in, ep_in: _gbsa_dense_param_energy_one(
                    positions,
                    charges,
                    r_in,
                    sr_in,
                    sg_in,
                    ep_in,
                    atom_mask,
                    excl_indices,
                    excl_scales_vdw,
                    box,
                    lam_arr,
                    has_box,
                    solvent_dielectric,
                    solute_dielectric,
                    dielectric_offset,
                    surface_tension,
                    include_ace,
                    include_lj_coulomb,
                    softcore_alpha,
                    softcore_exponent,
                    coulomb_constant,
                    n_active_atoms,
                ),
                radii,
                scaled_radii,
                sigmas,
                epsilons,
            )
            drad, dsr, dsig, deps = vjp_fn(g_arr)
        else:
            B = positions.shape[0]
            b_q = charges.ndim > 1
            b_rad = radii.ndim > 1
            b_sr = scaled_radii.ndim > 1
            b_sig = sigmas.ndim > 1
            b_eps = epsilons.ndim > 1
            b_msk = atom_mask.ndim > 1
            b_ei = excl_indices.ndim > 2
            b_ev = excl_scales_vdw.ndim > 2
            b_bx = box.ndim > 1
            b_lm = lam_arr.shape[0] > 1

            def _step(b):
                _, vjp_b = jax.vjp(
                    lambda r_in, sr_in, sg_in, ep_in: _gbsa_dense_param_energy_one(
                        positions[b],
                        charges[b] if b_q else charges,
                        r_in,
                        sr_in,
                        sg_in,
                        ep_in,
                        atom_mask[b] if b_msk else atom_mask,
                        excl_indices[b] if b_ei else excl_indices,
                        excl_scales_vdw[b] if b_ev else excl_scales_vdw,
                        box[b] if b_bx else box,
                        lam_arr[b] if b_lm else lam_arr,
                        has_box,
                        solvent_dielectric,
                        solute_dielectric,
                        dielectric_offset,
                        surface_tension,
                        include_ace,
                        include_lj_coulomb,
                        softcore_alpha,
                        softcore_exponent,
                        coulomb_constant,
                        n_active_atoms,
                    ),
                    radii[b] if b_rad else radii,
                    scaled_radii[b] if b_sr else scaled_radii,
                    sigmas[b] if b_sig else sigmas,
                    epsilons[b] if b_eps else epsilons,
                )
                return vjp_b(g_arr[b] if g_arr.ndim > 0 else g_arr)

            drad_all, dsr_all, dsig_all, deps_all = jax.lax.map(_step, jnp.arange(B))
            drad = drad_all if b_rad else jnp.sum(drad_all, axis=0)
            dsr = dsr_all if b_sr else jnp.sum(dsr_all, axis=0)
            dsig = dsig_all if b_sig else jnp.sum(dsig_all, axis=0)
            deps = deps_all if b_eps else jnp.sum(deps_all, axis=0)
        drad = drad.astype(radii.dtype)
        dsr = dsr.astype(scaled_radii.dtype)
        dsig = dsig.astype(sigmas.dtype)
        deps = deps.astype(epsilons.dtype)
    else:
        drad = jnp.zeros_like(radii)
        dsr = jnp.zeros_like(scaled_radii)
        dsig = jnp.zeros_like(sigmas)
        deps = jnp.zeros_like(epsilons)

    return (
        grad_pos,
        gq_scaled.astype(charges.dtype),
        drad,
        dsr,
        dsig,
        deps,
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(excl_indices),
        jnp.zeros_like(excl_scales_vdw),
        jnp.zeros_like(excl_scales_elec),
        jnp.zeros_like(box),
        jnp.zeros_like(lam_arr),
    )


_pallas_gbsa_dense_diff.defvjp(
    _pallas_gbsa_dense_fwd, _pallas_gbsa_dense_bwd, symbolic_zeros=True
)


def pallas_gbsa_dense(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array | None = None,
    atom_mask: jax.Array | None = None,
    *,
    sigmas: jax.Array | None = None,
    epsilons: jax.Array | None = None,
    excl_indices: jax.Array | None = None,
    excl_scales_vdw: jax.Array | None = None,
    excl_scales_elec: jax.Array | None = None,
    box: jax.Array | None = None,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    dielectric_offset: float = 0.09,
    surface_tension: float = SURFACE_TENSION,
    include_ace: bool = True,
    include_lj_coulomb: bool = False,
    soft_core_lambda: float | jax.Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    coulomb_constant: float = constants.COULOMB_CONSTANT,
    return_forces: bool = False,
    interpret: bool | None = None,
    n_active_atoms: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """2D-grid ``(B, n_act_blocks)`` coalesced 32x32 tile OBC-II GBSA (+ optional LJ/Coulomb) kernel."""
    dtype = positions.dtype
    N = int(positions.shape[-2])
    if N == 0:
        e_z = jnp.zeros(positions.shape[:-2], dtype=dtype)
        if return_forces:
            return e_z, jnp.zeros_like(positions)
        return e_z

    q_arr = jnp.asarray(charges, dtype=dtype)
    rad_arr = jnp.asarray(radii, dtype=dtype)
    sr_arr = rad_arr if scaled_radii is None else jnp.asarray(scaled_radii, dtype=dtype)
    sig_arr = (
        jnp.ones_like(q_arr)
        if sigmas is None
        else jnp.asarray(sigmas, dtype=dtype)
    )
    eps_arr = (
        jnp.zeros_like(q_arr)
        if epsilons is None
        else jnp.asarray(epsilons, dtype=dtype)
    )
    msk_arr = (
        jnp.ones(positions.shape[:-1], dtype=jnp.bool_)
        if atom_mask is None
        else jnp.asarray(atom_mask, dtype=jnp.bool_)
    )
    if excl_indices is not None and excl_indices.shape[-1] > 0:
        ei_arr = jnp.asarray(excl_indices, dtype=jnp.int32)
        ev_arr = (
            jnp.ones_like(ei_arr, dtype=dtype)
            if excl_scales_vdw is None
            else jnp.asarray(excl_scales_vdw, dtype=dtype)
        )
        ee_arr = (
            jnp.ones_like(ei_arr, dtype=dtype)
            if excl_scales_elec is None
            else jnp.asarray(excl_scales_elec, dtype=dtype)
        )
    else:
        ei_arr = jnp.full((N, 1), -1, dtype=jnp.int32)
        ev_arr = jnp.ones((N, 1), dtype=dtype)
        ee_arr = jnp.ones((N, 1), dtype=dtype)

    box_arr, has_box = _normalize_box(box, positions)
    lam_raw = jnp.asarray(soft_core_lambda, dtype=dtype)
    lam_arr = lam_raw.reshape((1,)) if lam_raw.ndim == 0 else lam_raw.reshape((-1, 1))

    if interpret is None:
        interpret = not _triton_gpu_available() or in_export_context()

    eff_active_atoms = _resolve_gb_active_atoms(n_active_atoms, N, atom_mask)

    if return_forces:
        e_out, f_out, _ = _pallas_gbsa_dense_vmappable(
            positions,
            q_arr,
            rad_arr,
            sr_arr,
            sig_arr,
            eps_arr,
            msk_arr,
            ei_arr,
            ev_arr,
            ee_arr,
            box_arr,
            lam_arr,
            bool(has_box),
            float(solvent_dielectric),
            float(solute_dielectric),
            float(dielectric_offset),
            float(surface_tension),
            bool(include_ace),
            bool(include_lj_coulomb),
            float(softcore_alpha),
            int(softcore_exponent),
            float(coulomb_constant),
            bool(interpret),
            n_active_atoms=eff_active_atoms,
        )
        return e_out, f_out

    return _pallas_gbsa_dense_diff(
        positions,
        q_arr,
        rad_arr,
        sr_arr,
        sig_arr,
        eps_arr,
        msk_arr,
        ei_arr,
        ev_arr,
        ee_arr,
        box_arr,
        lam_arr,
        bool(has_box),
        float(solvent_dielectric),
        float(solute_dielectric),
        float(dielectric_offset),
        float(surface_tension),
        bool(include_ace),
        bool(include_lj_coulomb),
        float(softcore_alpha),
        int(softcore_exponent),
        float(coulomb_constant),
        bool(interpret),
        int(eff_active_atoms),
    )


def pallas_gbsa_energy_and_forces(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    *,
    scaled_radii: jax.Array | None = None,
    sigmas: jax.Array | None = None,
    epsilons: jax.Array | None = None,
    atom_mask: jax.Array | None = None,
    excl_indices: jax.Array | None = None,
    excl_scales_vdw: jax.Array | None = None,
    excl_scales_elec: jax.Array | None = None,
    box: jax.Array | None = None,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    dielectric_offset: float = 0.09,
    surface_tension: float = SURFACE_TENSION,
    include_ace: bool = True,
    include_lj_coulomb: bool = True,
    soft_core_lambda: float | jax.Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    coulomb_constant: float = constants.COULOMB_CONSTANT,
    interpret: bool | None = None,
    n_active_atoms: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate implicit-solvent OBC-II GBSA (+ optional fused LJ/Coulomb) energy and forces."""
    res = pallas_gbsa_dense(
        positions,
        charges,
        radii,
        scaled_radii=scaled_radii,
        sigmas=sigmas,
        epsilons=epsilons,
        atom_mask=atom_mask,
        excl_indices=excl_indices,
        excl_scales_vdw=excl_scales_vdw,
        excl_scales_elec=excl_scales_elec,
        box=box,
        solvent_dielectric=solvent_dielectric,
        solute_dielectric=solute_dielectric,
        dielectric_offset=dielectric_offset,
        surface_tension=surface_tension,
        include_ace=include_ace,
        include_lj_coulomb=include_lj_coulomb,
        soft_core_lambda=soft_core_lambda,
        softcore_alpha=softcore_alpha,
        softcore_exponent=softcore_exponent,
        coulomb_constant=coulomb_constant,
        return_forces=True,
        interpret=interpret,
        n_active_atoms=n_active_atoms,
    )
    assert isinstance(res, tuple)
    return res


__all__ = [
    "pallas_gb_ace_nl",
    "pallas_gbsa_dense",
    "pallas_gbsa_energy_and_forces",
]
