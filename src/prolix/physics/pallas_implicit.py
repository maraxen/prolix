"""Pallas kernels for implicit-solvent GB/OBC-II + ACE nonpolar + SASA over symmetric neighbor lists (R6).

Provides:
  - ``pallas_gb_ace_nl``: 3-pass atomic-free (32, 32) OBC-II Generalized Born + ACE nonpolar kernel
  - ``pallas_sasa_nl``: 2-pass atomic-free (32, 32) probabilistic SASA nonpolar kernel
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from proxide.physics import constants

from prolix.export import in_export_context
from prolix.physics.generalized_born import (
    ALPHA_OBC,
    BETA_OBC,
    GAMMA_OBC,
    SURFACE_TENSION,
    compute_ace_nonpolar_energy,
    compute_gb_energy_neighbor_list,
    compute_nonpolar_energy,
    compute_pair_integral,
)
from prolix.physics.pallas_bonded import (
    _load_box6,
    _min_image3,
    _normalize_box,
    _triton_gpu_available,
)

_GB_BLOCK_N = 32
_GB_BLOCK_K = 32
_NB_INDEX_MASK = (1 << 29) - 1


def _pair_integral_deriv(r: jax.Array, or_a: jax.Array, sr_b: jax.Array) -> jax.Array:
    """Analytical derivative dI(r, or_a, sr_b)/dr matching compute_pair_integral (Section 3.7.1)."""
    dtype = r.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    neg_one = jnp.asarray(-1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eighth = jnp.asarray(0.125, dtype=dtype)
    eps = jnp.asarray(1e-4, dtype=dtype)

    D = jnp.abs(r - sr_b)
    L = jnp.maximum(or_a, D)
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
    dtotal_drs = (
        (zero - quarter) * jnp.log(L_s / U_s) / (r_s * r_s)
        + eighth * (sr2_sq / (r_s * r_s)) * (inv_U2 - inv_L2)
    )
    dtotal_dr_exp = eighth * (inv_U2 - inv_L2)

    sgn_D = jnp.where(r >= sr_b, one, neg_one)
    dLs_dr = jnp.where((D > or_a) & (L > eps), sgn_D, zero)
    dUs_dr = jnp.where(U > eps, one, zero)
    drs_dr = jnp.where(r > eps, one, zero)

    dI_dr = (
        dtotal_dLs * dLs_dr
        + dtotal_dUs * dUs_dr
        + dtotal_drs * drs_dr
        + dtotal_dr_exp
    )
    return jnp.where(U > or_a, dI_dr, zero)


# =============================================================================
# R6.1: 3-Pass OBC-II Generalized Born + ACE Nonpolar (pallas_gb_ace_nl)
# =============================================================================


def _pallas_gb_ace_nl_single(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    scaled_radii: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    box: jax.Array,
    has_box: bool,
    solvent_dielectric: float,
    solute_dielectric: float,
    dielectric_offset: float,
    surface_tension: float,
    include_ace: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    N = positions.shape[0]
    K = neighbor_idx.shape[1]
    dtype = positions.dtype

    n_blocks = max(1, (N + _GB_BLOCK_N - 1) // _GB_BLOCK_N)
    N_pad = n_blocks * _GB_BLOCK_N
    pad_n = N_pad - N

    k_blocks = max(1, (K + _GB_BLOCK_K - 1) // _GB_BLOCK_K)
    K_pad = k_blocks * _GB_BLOCK_K
    pad_k = K_pad - K
    pad_n_nbr = N_pad - neighbor_idx.shape[0]

    if pad_n > 0:
        pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0)))
        q_pad = jnp.pad(charges.astype(dtype), (0, pad_n))
        rad_pad = jnp.pad(radii.astype(dtype), (0, pad_n), constant_values=1.5)
        sr_pad = jnp.pad(scaled_radii.astype(dtype), (0, pad_n), constant_values=1.2)
        mask_pad = jnp.pad(atom_mask.astype(jnp.int32), (0, pad_n), constant_values=0)
    else:
        pos_pad = positions
        q_pad = charges.astype(dtype)
        rad_pad = radii.astype(dtype)
        sr_pad = scaled_radii.astype(dtype)
        mask_pad = atom_mask.astype(jnp.int32)

    nbr_clean = jnp.where(
        neighbor_idx >= 0,
        neighbor_idx.astype(jnp.int32) & _NB_INDEX_MASK,
        neighbor_idx.astype(jnp.int32),
    )
    nbr_pad = nbr_clean
    if pad_n_nbr > 0 or pad_k > 0:
        nbr_pad = jnp.pad(nbr_pad, ((0, pad_n_nbr), (0, pad_k)), constant_values=N)

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    nbr_flat = jnp.asarray(nbr_pad.reshape(N_pad * K_pad))
    box_vec = jnp.asarray(box.astype(dtype))

    # --- Pass 1: Born radii B_i and dB_i / dI_i ---
    def _pass1_born_kernel(
        px_ref,
        py_ref,
        pz_ref,
        rad_ref,
        sr_ref,
        mask_ref,
        nbr_flat_ref,
        box_ref,
        b_out_ref,
        db_di_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _GB_BLOCK_N + jnp.arange(_GB_BLOCK_N, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        eps4 = jnp.asarray(1e-4, dtype=dtype)
        eps6 = jnp.asarray(1e-6, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        offset = jnp.asarray(dielectric_offset, dtype=dtype)
        box6 = _load_box6(box_ref)

        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)
        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        rad_i = jnp.maximum(_pl_load(rad_ref, (offs_i,)), eps6)
        or_i = jnp.maximum(rad_i - offset, eps4)

        def _k_body(k_tile, acc_I):
            offs_k = k_tile * _GB_BLOCK_K + jnp.arange(_GB_BLOCK_K, dtype=jnp.int32)
            flat_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(_GB_BLOCK_N * _GB_BLOCK_K)
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_offs,))
            j_idx = j_idx_1d.reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            sr_j = _pl_load(sr_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            mask_j = _pl_load(mask_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K) > 0

            active = (
                mask_i[:, None]
                & mask_j
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
            )

            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j, py_i[:, None] - py_j, pz_i[:, None] - pz_j),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            I_ij = compute_pair_integral(dist, or_i[:, None], sr_j)
            return acc_I + jnp.sum(jnp.where(active, I_ij, zero), axis=1)

        I_i = jax.lax.fori_loop(0, k_blocks, _k_body, jnp.zeros((_GB_BLOCK_N,), dtype=dtype))
        psi_i = or_i * I_i
        alpha_obc = jnp.asarray(ALPHA_OBC, dtype=dtype)
        beta_obc = jnp.asarray(BETA_OBC, dtype=dtype)
        gamma_obc = jnp.asarray(GAMMA_OBC, dtype=dtype)
        arg_i = alpha_obc * psi_i - beta_obc * (psi_i * psi_i) + gamma_obc * (psi_i * psi_i * psi_i)
        th_i = jnp.tanh(arg_i)
        r_safe_i = jnp.maximum(rad_i, eps4)
        b_inv_raw = one / or_i - th_i / r_safe_i
        b_inv_safe = jnp.maximum(b_inv_raw, eps4)
        b_i = one / b_inv_safe

        darg_dpsi = alpha_obc - jnp.asarray(2.0, dtype=dtype) * beta_obc * psi_i + jnp.asarray(3.0, dtype=dtype) * gamma_obc * (psi_i * psi_i)
        db_di = jnp.where(
            mask_i & (b_inv_raw > eps4),
            (b_i * b_i) * (or_i / r_safe_i) * (one - th_i * th_i) * darg_dpsi,
            zero,
        )
        _pl_store(b_out_ref, (offs_i,), b_i)
        _pl_store(db_di_out_ref, (offs_i,), db_di)

    born_pad, db_di_pad = pl.pallas_call(
        _pass1_born_kernel,
        out_shape=[jax.ShapeDtypeStruct((N_pad,), dtype), jax.ShapeDtypeStruct((N_pad,), dtype)],
        in_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(8)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(2)],
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_gb_pass1_born_nl",
    )(px, py, pz, rad_pad, sr_pad, mask_pad, nbr_flat, box_vec)

    # --- Pass 2: Host O(N) self GB + ACE, plus Pallas O(N*K) pair GB, direct forces, dE_pair/dB, grad_q ---
    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    prefactor = jnp.asarray(-0.5 * constants.COULOMB_CONSTANT * tau, dtype=dtype)
    mask_f = (mask_pad > 0).astype(dtype)

    e_gb_self_vec = prefactor * mask_f * (q_pad * q_pad) / born_pad
    de_self_db = -e_gb_self_vec / born_pad

    if include_ace:
        st_scale = float(surface_tension) / float(SURFACE_TENSION) if SURFACE_TENSION > 0.0 else 1.0
        radius_nm = (rad_pad - jnp.asarray(dielectric_offset, dtype=dtype)) * jnp.asarray(0.1, dtype=dtype) + jnp.asarray(0.009, dtype=dtype)
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

    def _pass2_gb_pair_kernel(
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        born_ref,
        mask_ref,
        nbr_flat_ref,
        box_ref,
        e_pair_out_ref,
        fx_dir_out_ref,
        fy_dir_out_ref,
        fz_dir_out_ref,
        de_pair_db_out_ref,
        grad_q_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _GB_BLOCK_N + jnp.arange(_GB_BLOCK_N, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)
        quarter = jnp.asarray(0.25, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        pref = jnp.asarray(-0.5 * constants.COULOMB_CONSTANT * tau, dtype=dtype)
        box6 = _load_box6(box_ref)

        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)
        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        q_i = _pl_load(q_ref, (offs_i,))
        b_i = _pl_load(born_ref, (offs_i,))

        def _k_body(k_tile, carry):
            acc_e, acc_fx, acc_fy, acc_fz, acc_dedb, acc_gq = carry
            offs_k = k_tile * _GB_BLOCK_K + jnp.arange(_GB_BLOCK_K, dtype=jnp.int32)
            flat_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(_GB_BLOCK_N * _GB_BLOCK_K)
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_offs,))
            j_idx = j_idx_1d.reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            q_j = _pl_load(q_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            b_j = _pl_load(born_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            mask_j = _pl_load(mask_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K) > 0

            active = (
                mask_i[:, None]
                & mask_j
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
            )

            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j, py_i[:, None] - py_j, pz_i[:, None] - pz_j),
                box6,
                has_box,
            )
            r_sq = dx * dx + dy * dy + dz * dz + eps12

            p_ij = b_i[:, None] * b_j
            u_ij = r_sq / (four * p_ij)
            e_ij = jnp.exp(zero - u_ij)
            f_gb_sq = r_sq + p_ij * e_ij
            f_gb = jnp.sqrt(f_gb_sq)
            f_gb_cubed = f_gb * f_gb_sq

            qq = q_i[:, None] * q_j
            e_term = jnp.where(active, pref * qq / f_gb, zero)
            f_coeff = jnp.where(
                active,
                two * pref * qq * (one - quarter * e_ij) / f_gb_cubed,
                zero,
            )
            dedb_term = jnp.where(
                active,
                (zero - pref) * qq * b_j * e_ij * (one + u_ij) / f_gb_cubed,
                zero,
            )
            gq_term = jnp.where(active, q_j / f_gb, zero)

            return (
                acc_e + jnp.sum(e_term, axis=1),
                acc_fx + jnp.sum(f_coeff * dx, axis=1),
                acc_fy + jnp.sum(f_coeff * dy, axis=1),
                acc_fz + jnp.sum(f_coeff * dz, axis=1),
                acc_dedb + jnp.sum(dedb_term, axis=1),
                acc_gq + jnp.sum(gq_term, axis=1),
            )

        init_carry = tuple(jnp.zeros((_GB_BLOCK_N,), dtype=dtype) for _ in range(6))
        e_p, fx_d, fy_d, fz_d, dedb_p, gq_p = jax.lax.fori_loop(0, k_blocks, _k_body, init_carry)
        gq_tot = jnp.where(mask_i, two * pref * (q_i / b_i + gq_p), zero)

        _pl_store(e_pair_out_ref, (offs_i,), e_p)
        _pl_store(fx_dir_out_ref, (offs_i,), fx_d)
        _pl_store(fy_dir_out_ref, (offs_i,), fy_d)
        _pl_store(fz_dir_out_ref, (offs_i,), fz_d)
        _pl_store(de_pair_db_out_ref, (offs_i,), dedb_p)
        _pl_store(grad_q_out_ref, (offs_i,), gq_tot)

    e_pair_pad, fx_dir_pad, fy_dir_pad, fz_dir_pad, de_pair_db_pad, grad_q_pad = pl.pallas_call(
        _pass2_gb_pair_kernel,
        out_shape=[jax.ShapeDtypeStruct((N_pad,), dtype) for _ in range(6)],
        in_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(8)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(6)],
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_gb_pass2_pair_nl",
    )(px, py, pz, q_pad, born_pad, mask_pad, nbr_flat, box_vec)

    # --- Pass 3: Atomic-Free Symmetric-NL Born Chain-Rule Forces ---
    w_pad = (de_self_db + de_ace_db + de_pair_db_pad) * db_di_pad
    or_pad = jnp.maximum(
        jnp.maximum(rad_pad, jnp.asarray(1e-6, dtype=dtype)) - jnp.asarray(dielectric_offset, dtype=dtype),
        jnp.asarray(1e-4, dtype=dtype),
    )

    def _pass3_born_chain_kernel(
        px_ref,
        py_ref,
        pz_ref,
        or_ref,
        sr_ref,
        w_ref,
        mask_ref,
        nbr_flat_ref,
        box_ref,
        fx_born_out_ref,
        fy_born_out_ref,
        fz_born_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _GB_BLOCK_N + jnp.arange(_GB_BLOCK_N, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        box6 = _load_box6(box_ref)

        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)
        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        or_i = _pl_load(or_ref, (offs_i,))
        sr_i = _pl_load(sr_ref, (offs_i,))
        w_i = _pl_load(w_ref, (offs_i,))

        def _k_body(k_tile, carry):
            acc_fx, acc_fy, acc_fz = carry
            offs_k = k_tile * _GB_BLOCK_K + jnp.arange(_GB_BLOCK_K, dtype=jnp.int32)
            flat_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(_GB_BLOCK_N * _GB_BLOCK_K)
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_offs,))
            j_idx = j_idx_1d.reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            or_j = _pl_load(or_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            sr_j = _pl_load(sr_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            w_j = _pl_load(w_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            mask_j = _pl_load(mask_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K) > 0

            active = (
                mask_i[:, None]
                & mask_j
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
            )

            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j, py_i[:, None] - py_j, pz_i[:, None] - pz_j),
                box6,
                has_box,
            )
            dist = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)

            dI_ij_dr = _pair_integral_deriv(dist, or_i[:, None], sr_j)
            dI_ji_dr = _pair_integral_deriv(dist, or_j, sr_i[:, None])

            f_scalar = jnp.where(
                active,
                (zero - (w_i[:, None] * dI_ij_dr + w_j * dI_ji_dr)) / dist,
                zero,
            )
            return (
                acc_fx + jnp.sum(f_scalar * dx, axis=1),
                acc_fy + jnp.sum(f_scalar * dy, axis=1),
                acc_fz + jnp.sum(f_scalar * dz, axis=1),
            )

        init_carry = tuple(jnp.zeros((_GB_BLOCK_N,), dtype=dtype) for _ in range(3))
        fx_b, fy_b, fz_b = jax.lax.fori_loop(0, k_blocks, _k_body, init_carry)
        _pl_store(fx_born_out_ref, (offs_i,), fx_b)
        _pl_store(fy_born_out_ref, (offs_i,), fy_b)
        _pl_store(fz_born_out_ref, (offs_i,), fz_b)

    fx_born_pad, fy_born_pad, fz_born_pad = pl.pallas_call(
        _pass3_born_chain_kernel,
        out_shape=[jax.ShapeDtypeStruct((N_pad,), dtype) for _ in range(3)],
        in_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(9)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(3)],
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_gb_pass3_born_chain_nl",
    )(px, py, pz, or_pad, sr_pad, w_pad, mask_pad, nbr_flat, box_vec)

    total_energy = jnp.sum(e_gb_self_vec[:N] + e_ace_vec[:N] + e_pair_pad[:N])
    forces = jnp.stack(
        [
            (fx_dir_pad + fx_born_pad)[:N],
            (fy_dir_pad + fy_born_pad)[:N],
            (fz_dir_pad + fz_born_pad)[:N],
        ],
        axis=-1,
    )
    return total_energy, forces, grad_q_pad[:N].astype(charges.dtype)


def _gb_ace_nl_ref_one(
    pos_1,
    q_1,
    rad_1,
    sr_1,
    m_1,
    nbr_1,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
):
    N = pos_1.shape[0]
    m_bool = jnp.asarray(m_1, dtype=jnp.bool_)
    nbr_clean = jnp.where(nbr_1 >= 0, nbr_1 & _NB_INDEX_MASK, nbr_1)
    safe_j = jnp.clip(nbr_clean, 0, max(N - 1, 0))
    valid_pair = (
        (nbr_clean >= 0)
        & (nbr_clean < N)
        & (nbr_clean != jnp.arange(N, dtype=jnp.int32)[:, None])
        & m_bool[:, None]
        & m_bool[safe_j]
    )
    nbr_masked = jnp.where(valid_pair, nbr_clean, N)
    q_masked = jnp.where(m_bool, q_1, 0.0)
    e_gb, born_r = compute_gb_energy_neighbor_list(
        pos_1,
        q_masked,
        rad_1,
        nbr_masked,
        solvent_dielectric=solvent_dielectric,
        solute_dielectric=solute_dielectric,
        dielectric_offset=dielectric_offset,
        scaled_radii=sr_1,
    )
    if include_ace:
        st_scale = float(surface_tension) / float(SURFACE_TENSION) if SURFACE_TENSION > 0.0 else 1.0
        e_ace = st_scale * jnp.sum(
            jnp.where(
                m_bool,
                compute_ace_nonpolar_energy(rad_1, born_r, dielectric_offset=dielectric_offset),
                0.0,
            )
        )
        return e_gb + e_ace
    return e_gb


def _pallas_gb_ace_nl_vmappable(
    positions,
    charges,
    radii,
    scaled_radii,
    atom_mask,
    neighbor_idx,
    box,
    has_box,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
    interpret,
):
    hb = bool(has_box)
    sd = float(solvent_dielectric)
    ud = float(solute_dielectric)
    do = float(dielectric_offset)
    st = float(surface_tension)
    ia = bool(include_ace)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, rad, sr, msk, nbr, bx):
        return _pallas_gb_ace_nl_single(pos, q, rad, sr, msk, nbr, bx, hb, sd, ud, do, st, ia, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, rad, sr, msk, nbr, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                q[b_idx] if in_batched[1] else q,
                rad[b_idx] if in_batched[2] else rad,
                sr[b_idx] if in_batched[3] else sr,
                msk[b_idx] if in_batched[4] else msk,
                nbr[b_idx] if in_batched[5] else nbr,
                bx[b_idx] if in_batched[6] else bx,
            )

        e_b, f_b, gq_b = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_b, f_b, gq_b), (True, True, True)

    return _vmappable(positions, charges, radii, scaled_radii, atom_mask, neighbor_idx, box)


@functools.partial(custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _pallas_gb_ace_nl_diff(
    positions,
    charges,
    radii,
    scaled_radii,
    atom_mask,
    neighbor_idx,
    box,
    has_box,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
    interpret,
):
    energy, _, _ = _pallas_gb_ace_nl_vmappable(
        positions,
        charges,
        radii,
        scaled_radii,
        atom_mask,
        neighbor_idx,
        box,
        has_box,
        solvent_dielectric,
        solute_dielectric,
        dielectric_offset,
        surface_tension,
        include_ace,
        interpret,
    )
    return energy


def _pallas_gb_ace_nl_fwd(
    positions,
    charges,
    radii,
    scaled_radii,
    atom_mask,
    neighbor_idx,
    box,
    has_box,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
    interpret,
):
    energy, forces, grad_q = _pallas_gb_ace_nl_vmappable(
        positions,
        charges,
        radii,
        scaled_radii,
        atom_mask,
        neighbor_idx,
        box,
        has_box,
        solvent_dielectric,
        solute_dielectric,
        dielectric_offset,
        surface_tension,
        include_ace,
        interpret,
    )
    return energy, (forces, grad_q, positions, charges, radii, scaled_radii, atom_mask, neighbor_idx, box)


def _pallas_gb_ace_nl_bwd(
    has_box,
    solvent_dielectric,
    solute_dielectric,
    dielectric_offset,
    surface_tension,
    include_ace,
    interpret,
    res,
    g,
):
    del has_box, interpret
    forces, grad_q, positions, charges, radii, scaled_radii, atom_mask, neighbor_idx, box = res
    grad_pos = -g[..., None, None] * forces if g.ndim > 0 else -g * forces
    gq_scaled = g[..., None] * grad_q if g.ndim > 0 else g * grad_q
    while gq_scaled.ndim > charges.ndim:
        gq_scaled = jnp.sum(gq_scaled, axis=0)

    if positions.ndim == 2:
        _, vjp_fn = jax.vjp(
            lambda r_in, sr_in: _gb_ace_nl_ref_one(
                positions,
                charges,
                r_in,
                sr_in,
                atom_mask,
                neighbor_idx,
                solvent_dielectric,
                solute_dielectric,
                dielectric_offset,
                surface_tension,
                include_ace,
            ),
            radii,
            scaled_radii,
        )
        drad, dsr = vjp_fn(g)
    else:
        B = positions.shape[0]
        b_q = charges.ndim > 1
        b_rad = radii.ndim > 1
        b_sr = scaled_radii.ndim > 1
        b_msk = atom_mask.ndim > 1
        b_nbr = neighbor_idx.ndim > 2

        def _step(b):
            _, vjp_b = jax.vjp(
                lambda r_in, sr_in: _gb_ace_nl_ref_one(
                    positions[b],
                    charges[b] if b_q else charges,
                    r_in,
                    sr_in,
                    atom_mask[b] if b_msk else atom_mask,
                    neighbor_idx[b] if b_nbr else neighbor_idx,
                    solvent_dielectric,
                    solute_dielectric,
                    dielectric_offset,
                    surface_tension,
                    include_ace,
                ),
                radii[b] if b_rad else radii,
                scaled_radii[b] if b_sr else scaled_radii,
            )
            return vjp_b(g[b] if g.ndim > 0 else g)

        drad_all, dsr_all = jax.lax.map(_step, jnp.arange(B))
        drad = drad_all if b_rad else jnp.sum(drad_all, axis=0)
        dsr = dsr_all if b_sr else jnp.sum(dsr_all, axis=0)

    return (
        grad_pos,
        gq_scaled.astype(charges.dtype),
        drad.astype(radii.dtype),
        dsr.astype(scaled_radii.dtype),
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(neighbor_idx),
        jnp.zeros_like(box),
    )


_pallas_gb_ace_nl_diff.defvjp(_pallas_gb_ace_nl_fwd, _pallas_gb_ace_nl_bwd)


def pallas_gb_ace_nl(
    positions: jax.Array,
    charges: jax.Array,
    radii: jax.Array,
    neighbor_idx: jax.Array,
    atom_mask: jax.Array | None = None,
    scaled_radii: jax.Array | None = None,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
    dielectric_offset: float = 0.09,
    surface_tension: float = SURFACE_TENSION,
    include_ace: bool = True,
    box: jax.Array | None = None,
    *,
    interpret: bool | None = None,
) -> jax.Array:
    """3-pass atomic-free (32, 32) OBC-II GB + ACE nonpolar Pallas kernel over symmetric neighbor list."""
    if positions.shape[-2] == 0:
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    rad_arr = jnp.asarray(radii, dtype=positions.dtype)
    sr_arr = rad_arr if scaled_radii is None else jnp.asarray(scaled_radii, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(positions.shape[:-1], dtype=jnp.bool_)
        if atom_mask is None
        else jnp.asarray(atom_mask)
    )
    q_arr = jnp.asarray(charges, dtype=positions.dtype)
    nbr_arr = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    nbr_arr = jnp.where(nbr_arr >= 0, nbr_arr & _NB_INDEX_MASK, nbr_arr)

    if in_export_context():
        if positions.ndim == 3:
            return jax.vmap(
                lambda b: _gb_ace_nl_ref_one(
                    positions[b],
                    q_arr[b] if q_arr.ndim == 2 else q_arr,
                    rad_arr[b] if rad_arr.ndim == 2 else rad_arr,
                    sr_arr[b] if sr_arr.ndim == 2 else sr_arr,
                    msk_arr[b] if msk_arr.ndim == 2 else msk_arr,
                    nbr_arr[b] if nbr_arr.ndim == 3 else nbr_arr,
                    solvent_dielectric,
                    solute_dielectric,
                    dielectric_offset,
                    surface_tension,
                    include_ace,
                )
            )(jnp.arange(positions.shape[0]))
        return _gb_ace_nl_ref_one(
            positions,
            q_arr,
            rad_arr,
            sr_arr,
            msk_arr,
            nbr_arr,
            solvent_dielectric,
            solute_dielectric,
            dielectric_offset,
            surface_tension,
            include_ace,
        )

    if interpret is None:
        interpret = not _triton_gpu_available()
    box_arr, has_box = _normalize_box(box, positions)
    return _pallas_gb_ace_nl_diff(
        positions,
        q_arr,
        rad_arr,
        sr_arr,
        msk_arr,
        nbr_arr,
        box_arr,
        bool(has_box),
        float(solvent_dielectric),
        float(solute_dielectric),
        float(dielectric_offset),
        float(surface_tension),
        bool(include_ace),
        bool(interpret),
    )


# =============================================================================
# R6.2: 2-Pass Probabilistic SASA Nonpolar Kernel (pallas_sasa_nl)
# =============================================================================


def _pallas_sasa_nl_single(
    positions: jax.Array,
    radii: jax.Array,
    atom_mask: jax.Array,
    neighbor_idx: jax.Array,
    box: jax.Array,
    has_box: bool,
    surface_tension: float,
    probe_radius: float,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    N = positions.shape[0]
    K = neighbor_idx.shape[1]
    dtype = positions.dtype

    n_blocks = max(1, (N + _GB_BLOCK_N - 1) // _GB_BLOCK_N)
    N_pad = n_blocks * _GB_BLOCK_N
    pad_n = N_pad - N

    k_blocks = max(1, (K + _GB_BLOCK_K - 1) // _GB_BLOCK_K)
    K_pad = k_blocks * _GB_BLOCK_K
    pad_k = K_pad - K
    pad_n_nbr = N_pad - neighbor_idx.shape[0]

    if pad_n > 0:
        pos_pad = jnp.pad(positions, ((0, pad_n), (0, 0)))
        rad_pad = jnp.pad(radii.astype(dtype), (0, pad_n), constant_values=1.5)
        mask_pad = jnp.pad(atom_mask.astype(jnp.int32), (0, pad_n), constant_values=0)
    else:
        pos_pad = positions
        rad_pad = radii.astype(dtype)
        mask_pad = atom_mask.astype(jnp.int32)

    nbr_clean = jnp.where(
        neighbor_idx >= 0,
        neighbor_idx.astype(jnp.int32) & _NB_INDEX_MASK,
        neighbor_idx.astype(jnp.int32),
    )
    nbr_pad = nbr_clean
    if pad_n_nbr > 0 or pad_k > 0:
        nbr_pad = jnp.pad(nbr_pad, ((0, pad_n_nbr), (0, pad_k)), constant_values=N)

    px = jnp.asarray(pos_pad[:, 0])
    py = jnp.asarray(pos_pad[:, 1])
    pz = jnp.asarray(pos_pad[:, 2])
    nbr_flat = jnp.asarray(nbr_pad.reshape(N_pad * K_pad))
    box_vec = jnp.asarray(box.astype(dtype))
    R_pad = rad_pad + jnp.asarray(probe_radius, dtype=dtype)
    S_pad = jnp.asarray(4.0 * jnp.pi, dtype=dtype) * (R_pad * R_pad)

    # --- Pass 1: Per-atom exposed surface area A_i and zero_count_i ---
    def _pass1_sasa_area_kernel(
        px_ref,
        py_ref,
        pz_ref,
        R_ref,
        S_ref,
        mask_ref,
        nbr_flat_ref,
        box_ref,
        area_out_ref,
        zcount_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _GB_BLOCK_N + jnp.arange(_GB_BLOCK_N, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        eps6 = jnp.asarray(1e-6, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        box6 = _load_box6(box_ref)

        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)
        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        R_i = _pl_load(R_ref, (offs_i,))
        S_i = _pl_load(S_ref, (offs_i,))

        def _k_body(k_tile, carry):
            acc_log, acc_zc = carry
            offs_k = k_tile * _GB_BLOCK_K + jnp.arange(_GB_BLOCK_K, dtype=jnp.int32)
            flat_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(_GB_BLOCK_N * _GB_BLOCK_K)
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_offs,))
            j_idx = j_idx_1d.reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            R_j = _pl_load(R_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            mask_j = _pl_load(mask_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K) > 0

            active = (
                mask_i[:, None]
                & mask_j
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
            )

            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j, py_i[:, None] - py_j, pz_i[:, None] - pz_j),
                box6,
                has_box,
            )
            d_ij = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            d_safe = d_ij + eps6

            R_i_m = R_i[:, None]
            S_i_m = S_i[:, None]
            b_raw = (pi * R_i_m / d_safe) * (R_j * R_j - (d_ij - R_i_m) * (d_ij - R_i_m))
            no_ov = d_ij >= (R_i_m + R_j)
            i_in_j = R_j >= (R_i_m + d_ij)
            j_in_i = R_i_m >= (R_j + d_ij)

            b_ij = jnp.where(no_ov, zero, b_raw)
            b_ij = jnp.where(i_in_j, S_i_m, b_ij)
            b_ij = jnp.where(j_in_i, zero, b_ij)
            b_ij = jnp.where(active, b_ij, zero)

            f_ij = jnp.maximum(zero, one - b_ij / S_i_m)
            is_zero = active & (f_ij <= eps12)
            log_f = jnp.where(active & (~is_zero), jnp.log(jnp.maximum(f_ij, eps12)), zero)
            return (
                acc_log + jnp.sum(log_f, axis=1),
                acc_zc + jnp.sum(is_zero.astype(jnp.int32), axis=1, dtype=jnp.int32),
            )

        log_sum, z_cnt = jax.lax.fori_loop(
            0,
            k_blocks,
            _k_body,
            (jnp.zeros((_GB_BLOCK_N,), dtype=dtype), jnp.zeros((_GB_BLOCK_N,), dtype=jnp.int32)),
        )
        A_i = jnp.where(mask_i & (z_cnt == 0), S_i * jnp.exp(log_sum), zero)
        _pl_store(area_out_ref, (offs_i,), A_i)
        _pl_store(zcount_out_ref, (offs_i,), z_cnt)

    area_pad, zcount_pad = pl.pallas_call(
        _pass1_sasa_area_kernel,
        out_shape=[jax.ShapeDtypeStruct((N_pad,), dtype), jax.ShapeDtypeStruct((N_pad,), jnp.int32)],
        in_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(8)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(2)],
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_sasa_pass1_area_nl",
    )(px, py, pz, R_pad, S_pad, mask_pad, nbr_flat, box_vec)

    # --- Pass 2: Atomic-Free Symmetric-NL SASA Forces ---
    def _pass2_sasa_forces_kernel(
        px_ref,
        py_ref,
        pz_ref,
        R_ref,
        S_ref,
        area_ref,
        zcount_ref,
        mask_ref,
        nbr_flat_ref,
        box_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
    ):
        pid = pl.program_id(0)
        offs_i = pid * _GB_BLOCK_N + jnp.arange(_GB_BLOCK_N, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        pi = jnp.asarray(jnp.pi, dtype=dtype)
        eps6 = jnp.asarray(1e-6, dtype=dtype)
        eps12 = jnp.asarray(1e-12, dtype=dtype)
        gamma = jnp.asarray(surface_tension, dtype=dtype)
        box6 = _load_box6(box_ref)

        mask_i = (_pl_load(mask_ref, (offs_i,)) > 0) & (offs_i < N)
        px_i = _pl_load(px_ref, (offs_i,))
        py_i = _pl_load(py_ref, (offs_i,))
        pz_i = _pl_load(pz_ref, (offs_i,))
        R_i = _pl_load(R_ref, (offs_i,))
        S_i = _pl_load(S_ref, (offs_i,))
        A_i = _pl_load(area_ref, (offs_i,))
        zc_i = _pl_load(zcount_ref, (offs_i,))

        def _k_body(k_tile, carry):
            acc_fx, acc_fy, acc_fz = carry
            offs_k = k_tile * _GB_BLOCK_K + jnp.arange(_GB_BLOCK_K, dtype=jnp.int32)
            flat_offs = (offs_i[:, None] * K_pad + offs_k[None, :]).reshape(_GB_BLOCK_N * _GB_BLOCK_K)
            j_idx_1d = _pl_load(nbr_flat_ref, (flat_offs,))
            j_idx = j_idx_1d.reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            safe_j_1d = jnp.clip(j_idx_1d, 0, N_pad - 1)

            px_j = _pl_load(px_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            py_j = _pl_load(py_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            pz_j = _pl_load(pz_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            R_j = _pl_load(R_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            S_j = _pl_load(S_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            A_j = _pl_load(area_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            zc_j = _pl_load(zcount_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K)
            mask_j = _pl_load(mask_ref, (safe_j_1d,)).reshape(_GB_BLOCK_N, _GB_BLOCK_K) > 0

            active = (
                mask_i[:, None]
                & mask_j
                & (j_idx >= 0)
                & (j_idx < N)
                & (j_idx != offs_i[:, None])
            )

            dx, dy, dz = _min_image3(
                (px_i[:, None] - px_j, py_i[:, None] - py_j, pz_i[:, None] - pz_j),
                box6,
                has_box,
            )
            d_ij = jnp.sqrt(dx * dx + dy * dy + dz * dz + eps12)
            d_safe = d_ij + eps6
            d_safe_sq = d_safe * d_safe

            R_i_m = R_i[:, None]
            S_i_m = S_i[:, None]
            A_i_m = A_i[:, None]
            zc_i_m = zc_i[:, None]

            no_ov = d_ij >= (R_i_m + R_j)
            i_in_j = R_j >= (R_i_m + d_ij)
            j_in_i = R_i_m >= (R_j + d_ij)
            smooth = active & (~no_ov) & (~i_in_j) & (~j_in_i)

            # (i, j) buried area and derivative
            term_ij = R_j * R_j - (d_ij - R_i_m) * (d_ij - R_i_m)
            b_ij = jnp.where(smooth, (pi * R_i_m / d_safe) * term_ij, zero)
            f_ij = jnp.maximum(zero, one - b_ij / S_i_m)
            db_ij_dd = jnp.where(
                smooth,
                pi * R_i_m * ((zero - term_ij) / d_safe_sq - two * (d_ij - R_i_m) / d_safe),
                zero,
            )
            dA_i_dd = jnp.where(
                smooth & (zc_i_m == 0) & (f_ij > eps12),
                (zero - A_i_m) / (S_i_m * jnp.maximum(f_ij, eps12)) * db_ij_dd,
                zero,
            )

            # (j, i) buried area and derivative
            term_ji = R_i_m * R_i_m - (d_ij - R_j) * (d_ij - R_j)
            b_ji = jnp.where(smooth, (pi * R_j / d_safe) * term_ji, zero)
            f_ji = jnp.maximum(zero, one - b_ji / S_j)
            db_ji_dd = jnp.where(
                smooth,
                pi * R_j * ((zero - term_ji) / d_safe_sq - two * (d_ij - R_j) / d_safe),
                zero,
            )
            dA_j_dd = jnp.where(
                smooth & (zc_j == 0) & (f_ji > eps12),
                (zero - A_j) / (S_j * jnp.maximum(f_ji, eps12)) * db_ji_dd,
                zero,
            )

            f_scalar = jnp.where(
                active,
                (zero - gamma) * (dA_i_dd + dA_j_dd) / d_ij,
                zero,
            )
            return (
                acc_fx + jnp.sum(f_scalar * dx, axis=1),
                acc_fy + jnp.sum(f_scalar * dy, axis=1),
                acc_fz + jnp.sum(f_scalar * dz, axis=1),
            )

        init_carry = tuple(jnp.zeros((_GB_BLOCK_N,), dtype=dtype) for _ in range(3))
        fx_s, fy_s, fz_s = jax.lax.fori_loop(0, k_blocks, _k_body, init_carry)
        _pl_store(fx_out_ref, (offs_i,), fx_s)
        _pl_store(fy_out_ref, (offs_i,), fy_s)
        _pl_store(fz_out_ref, (offs_i,), fz_s)

    fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _pass2_sasa_forces_kernel,
        out_shape=[jax.ShapeDtypeStruct((N_pad,), dtype) for _ in range(3)],
        in_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(10)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY) for _ in range(3)],
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pallas_sasa_pass2_forces_nl",
    )(px, py, pz, R_pad, S_pad, area_pad, zcount_pad, mask_pad, nbr_flat, box_vec)

    total_energy = jnp.asarray(surface_tension, dtype=dtype) * jnp.sum(area_pad[:N])
    forces = jnp.stack([fx_pad[:N], fy_pad[:N], fz_pad[:N]], axis=-1)
    return total_energy, forces


def _sasa_nl_ref_one(pos_1, rad_1, m_1, nbr_1, surface_tension, probe_radius):
    nbr_clean = jnp.where(nbr_1 >= 0, nbr_1 & _NB_INDEX_MASK, nbr_1)
    return compute_nonpolar_energy(
        pos_1,
        rad_1,
        surface_tension=surface_tension,
        probe_radius=probe_radius,
        neighbor_idx=nbr_clean,
        atom_mask=m_1,
    )


def _pallas_sasa_nl_vmappable(
    positions,
    radii,
    atom_mask,
    neighbor_idx,
    box,
    has_box,
    surface_tension,
    probe_radius,
    interpret,
):
    hb = bool(has_box)
    st = float(surface_tension)
    pr = float(probe_radius)
    interp = bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, rad, msk, nbr, bx):
        return _pallas_sasa_nl_single(pos, rad, msk, nbr, bx, hb, st, pr, interp)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, rad, msk, nbr, bx):
        def _step(b_idx):
            return _vmappable(
                pos[b_idx] if in_batched[0] else pos,
                rad[b_idx] if in_batched[1] else rad,
                msk[b_idx] if in_batched[2] else msk,
                nbr[b_idx] if in_batched[3] else nbr,
                bx[b_idx] if in_batched[4] else bx,
            )

        e_b, f_b = jax.lax.map(_step, jnp.arange(axis_size))
        return (e_b, f_b), (True, True)

    return _vmappable(positions, radii, atom_mask, neighbor_idx, box)


@functools.partial(custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _pallas_sasa_nl_diff(
    positions,
    radii,
    atom_mask,
    neighbor_idx,
    box,
    has_box,
    surface_tension,
    probe_radius,
    interpret,
):
    energy, _ = _pallas_sasa_nl_vmappable(
        positions, radii, atom_mask, neighbor_idx, box, has_box, surface_tension, probe_radius, interpret
    )
    return energy


def _pallas_sasa_nl_fwd(
    positions,
    radii,
    atom_mask,
    neighbor_idx,
    box,
    has_box,
    surface_tension,
    probe_radius,
    interpret,
):
    energy, forces = _pallas_sasa_nl_vmappable(
        positions, radii, atom_mask, neighbor_idx, box, has_box, surface_tension, probe_radius, interpret
    )
    return energy, (forces, positions, radii, atom_mask, neighbor_idx, box)


def _pallas_sasa_nl_bwd(has_box, surface_tension, probe_radius, interpret, res, g):
    del has_box, interpret
    forces, positions, radii, atom_mask, neighbor_idx, box = res
    grad_pos = -g[..., None, None] * forces if g.ndim > 0 else -g * forces

    if positions.ndim == 2:
        _, vjp_fn = jax.vjp(
            lambda r_in: _sasa_nl_ref_one(
                positions, r_in, atom_mask, neighbor_idx, surface_tension, probe_radius
            ),
            radii,
        )
        (drad,) = vjp_fn(g)
    else:
        B = positions.shape[0]
        b_rad = radii.ndim > 1
        b_msk = atom_mask.ndim > 1
        b_nbr = neighbor_idx.ndim > 2

        def _step(b):
            _, vjp_b = jax.vjp(
                lambda r_in: _sasa_nl_ref_one(
                    positions[b],
                    r_in,
                    atom_mask[b] if b_msk else atom_mask,
                    neighbor_idx[b] if b_nbr else neighbor_idx,
                    surface_tension,
                    probe_radius,
                ),
                radii[b] if b_rad else radii,
            )
            return vjp_b(g[b] if g.ndim > 0 else g)[0]

        drad_all = jax.lax.map(_step, jnp.arange(B))
        drad = drad_all if b_rad else jnp.sum(drad_all, axis=0)

    return (
        grad_pos,
        drad.astype(radii.dtype),
        jnp.zeros_like(atom_mask),
        jnp.zeros_like(neighbor_idx),
        jnp.zeros_like(box),
    )


_pallas_sasa_nl_diff.defvjp(_pallas_sasa_nl_fwd, _pallas_sasa_nl_bwd)


def pallas_sasa_nl(
    positions: jax.Array,
    radii: jax.Array,
    neighbor_idx: jax.Array,
    atom_mask: jax.Array | None = None,
    surface_tension: float = SURFACE_TENSION,
    probe_radius: float = constants.PROBE_RADIUS,
    dielectric_offset: float = 0.09,
    box: jax.Array | None = None,
    *,
    interpret: bool | None = None,
) -> jax.Array:
    """2-pass atomic-free (32, 32) SASA nonpolar Pallas kernel over symmetric neighbor list."""
    del dielectric_offset
    if positions.shape[-2] == 0:
        return jnp.zeros(positions.shape[:-2], dtype=positions.dtype)

    rad_arr = jnp.asarray(radii, dtype=positions.dtype)
    msk_arr = (
        jnp.ones(positions.shape[:-1], dtype=jnp.bool_)
        if atom_mask is None
        else jnp.asarray(atom_mask)
    )
    nbr_arr = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    nbr_arr = jnp.where(nbr_arr >= 0, nbr_arr & _NB_INDEX_MASK, nbr_arr)

    if in_export_context():
        if positions.ndim == 3:
            return jax.vmap(
                lambda b: _sasa_nl_ref_one(
                    positions[b],
                    rad_arr[b] if rad_arr.ndim == 2 else rad_arr,
                    msk_arr[b] if msk_arr.ndim == 2 else msk_arr,
                    nbr_arr[b] if nbr_arr.ndim == 3 else nbr_arr,
                    surface_tension,
                    probe_radius,
                )
            )(jnp.arange(positions.shape[0]))
        return _sasa_nl_ref_one(positions, rad_arr, msk_arr, nbr_arr, surface_tension, probe_radius)

    if interpret is None:
        interpret = not _triton_gpu_available()
    box_arr, has_box = _normalize_box(box, positions)
    return _pallas_sasa_nl_diff(
        positions,
        rad_arr,
        msk_arr,
        nbr_arr,
        box_arr,
        bool(has_box),
        float(surface_tension),
        float(probe_radius),
        bool(interpret),
    )


__all__ = [
    "pallas_gb_ace_nl",
    "pallas_sasa_nl",
]
