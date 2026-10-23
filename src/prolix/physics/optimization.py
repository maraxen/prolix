import functools
from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

from .lj_switching import lj_switch_weight
from .tiling import (
    compute_dense_tiling_dims,
    pad_to_tile,
    tile_reduction,
    tile_reduction_nl,
)

T = TypeVar("T")


def _lj_switch_weight_and_deriv(
    r: jnp.ndarray,
    cutoff: float,
    switch_width: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (S(r), dS/dr) for the C2 smooth LJ switching polynomial."""
    sw = None if switch_width == 0.0 else switch_width
    s_val = lj_switch_weight(r, cutoff, sw)
    if switch_width <= 0.0:
        return s_val, jnp.zeros_like(r)
    r_cut = jnp.asarray(cutoff, dtype=r.dtype)
    w = jnp.asarray(switch_width, dtype=r.dtype)
    r_sw = r_cut - w
    one = jnp.asarray(1.0, dtype=r.dtype)
    zero = jnp.asarray(0.0, dtype=r.dtype)
    z = jnp.clip((r - r_sw) / w, zero, one)
    ds_dr = jnp.where(
        (r > r_sw) & (r < r_cut),
        -jnp.asarray(30.0, dtype=r.dtype) * (z ** 2) * ((one - z) ** 2) / w,
        zero,
    )
    return s_val, ds_dr


# ============================================================================
# LJ Kernels (Dense & NL)
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 5, 6, 7))
def chunked_lj_energy(
    r: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    excl_indices: jnp.ndarray,
    excl_scales: jnp.ndarray,
    displacement_fn: Callable,
    cutoff: float = 9.0,
    tile_size: int = 128,
) -> jnp.ndarray:
    """Computes dense LJ energy using FlashMD tiling O(N^2)."""
    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(r.shape[0]), int(excl_indices.shape[0]), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    sig_pad, _ = pad_to_tile(sigmas, pad_dim)
    eps_pad, _ = pad_to_tile(epsilons, pad_dim)
    excl_i_pad = jnp.pad(excl_indices, ((0, pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1))
        ds = dist
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)

        matches = (idx_j[None, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        s_ij = 0.5 * (sig_j[None, :] + sig_i[:, None])
        e_ij = jnp.sqrt(jnp.maximum(eps_j[None, :] * eps_i[:, None], 0.0))

        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0:
            m = m & (ds < cutoff)

        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        inv_r6 = jnp.where(m & (ds > 0.0), (s_ij * inv_ds) ** 6, 0.0)
        energy = jnp.where(m & (ds > 0.0), 4.0 * e_ij * (inv_r6 ** 2 - inv_r6), 0.0)
        return 0.5 * jnp.sum(energy * pair_scale)

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)


def _chunked_lj_fwd(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, cutoff, tile_size):
    return (
        chunked_lj_energy(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, cutoff, tile_size),
        (r, sigmas, epsilons, excl_scales),
    )


def _chunked_lj_bwd(excl_indices, displacement_fn, cutoff, tile_size, res, g):
    r, sigmas, epsilons, excl_scales = res
    N = r.shape[0]
    n_excl_rows, M = excl_indices.shape
    dtype = r.dtype

    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(N), int(n_excl_rows), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    sig_pad, _ = pad_to_tile(sigmas, pad_dim)
    eps_pad, _ = pad_to_tile(epsilons, pad_dim)
    excl_i_pad = jnp.pad(excl_indices, ((0, pad_dim - n_excl_rows), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        d2 = jnp.sum(dr ** 2, axis=-1)
        d = jnp.sqrt(d2 + jnp.asarray(1e-30, dtype=dtype))

        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, M))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, M))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)

        # matches: (inner_tile_size, M, tile_size)
        matches = (idx_j[None, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None]) & (d2 > 0.0)
        if cutoff > 0:
            m = m & (d < cutoff)

        s_ij = 0.5 * (sig_i[:, None] + sig_j[None, :])
        s_ij_safe = jnp.where(m & (s_ij > 0.0), s_ij, 1.0)
        e_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j[None, :], 0.0) + jnp.asarray(1e-30, dtype=dtype))

        inv_d = jnp.where(m, 1.0 / d, 0.0)
        u = jnp.where(m, (s_ij * inv_d) ** 6, 0.0)
        v_unsw = jnp.where(m, 4.0 * e_ij * (u * u - u), 0.0)

        # Position gradient: dE/dr_i = sum_j scale * 4*e_ij * (-12*u^2 + 6*u) / d * (dr / d)
        de_dr_over_r = jnp.where(
            m,
            pair_scale * 4.0 * e_ij * (-12.0 * u * u + 6.0 * u) * (inv_d * inv_d),
            0.0,
        )
        grad_r_tile = jnp.sum(de_dr_over_r[:, :, None] * dr, axis=1)

        # Sigma gradient: dE/d(sigma_i) = sum_j scale * 4*e_ij * (12*u^2 - 6*u) * (1 / (2*s_ij))
        grad_sig_tile = jnp.sum(
            jnp.where(m, pair_scale * 2.0 * e_ij * (12.0 * u * u - 6.0 * u) / s_ij_safe, 0.0),
            axis=1,
        )

        # Epsilon gradient: dE/d(eps_i) = sum_j scale * 2*(u^2 - u) * eps_j / e_ij
        grad_eps_tile = jnp.sum(
            jnp.where(m, pair_scale * 2.0 * (u * u - u) * eps_j[None, :] / e_ij, 0.0),
            axis=1,
        )

        # Exclusion scale gradient: dE/d(excl_scales[i, m]) = 0.5 * sum_j I(matches[i, m, j]) * v_unsw[i, j]
        grad_excl_tile = 0.5 * jnp.sum(
            jnp.where(matches & m[:, None, :], v_unsw[:, None, :], 0.0),
            axis=2,
        )

        return jnp.concatenate(
            [grad_r_tile, grad_sig_tile[:, None], grad_eps_tile[:, None], grad_excl_tile],
            axis=1,
        )

    init_combined = jnp.zeros((pad_dim, 5 + M), dtype=dtype)
    combined = tile_reduction(
        r_pad, mask_pad, f_tile_grad, init_combined, tile_size, inner_tile_size=inner_tile_size
    )
    grad_r = combined[:N, :3]
    grad_sig = combined[:N, 3]
    grad_eps = combined[:N, 4]
    grad_excl = combined[: excl_scales.shape[0], 5:]
    return (g * grad_r, g * grad_sig, g * grad_eps, g * grad_excl)


chunked_lj_energy.defvjp(_chunked_lj_fwd, _chunked_lj_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def chunked_lj_energy_nl(
    r: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    excl_indices: jnp.ndarray,
    excl_scales: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    cutoff: float = 9.0,
    tile_size: int = 32,
    switch_width: float = 0.0,
) -> jnp.ndarray:
    """Computes LJ energy over neighbor list."""
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    sig_pad, _ = pad_to_tile(sigmas, inner_tile_size)
    eps_pad, _ = pad_to_tile(epsilons, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1))
        ds = dist
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        safe_nb = jnp.where(mask_j, nb_idx_tile, 0)
        sig_j = sig_pad[safe_nb]
        eps_j = eps_pad[safe_nb]

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))

        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        s_ij = 0.5 * (sig_i[:, None] + sig_j)
        e_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j, 0.0))
        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        inv_r6 = jnp.where(ds > 0.0, (s_ij * inv_ds) ** 6, 0.0)
        idx_i = (start_i + jnp.arange(inner_tile_size))[:, None]
        m = mask_i[:, None] & mask_j & (nb_idx_tile != idx_i) & (ds > 0.0)
        if cutoff > 0:
            m = m & (ds < cutoff)
        sw = None if switch_width == 0.0 else switch_width
        switch = lj_switch_weight(ds, cutoff, sw)
        energy = jnp.where(m, 4.0 * e_ij * (inv_r6 ** 2 - inv_r6) * switch, 0.0)
        return 0.5 * jnp.sum(energy * pair_scale)

    return tile_reduction_nl(r_pad, neighbor_idx, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)


def _chunked_lj_nl_fwd(r, sigmas, epsilons, excl_indices, excl_scales, nb_idx, disp_fn, cutoff, tile_size, switch_width):
    return (
        chunked_lj_energy_nl(r, sigmas, epsilons, excl_indices, excl_scales, nb_idx, disp_fn, cutoff, tile_size, switch_width),
        (r, sigmas, epsilons, excl_indices, excl_scales, nb_idx),
    )


def _chunked_lj_nl_bwd(displacement_fn, cutoff, tile_size, switch_width, res, g):
    r, sigmas, epsilons, excl_indices, excl_scales, nb_idx = res
    N = r.shape[0]
    M = excl_indices.shape[1]
    dtype = r.dtype

    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    sig_pad, _ = pad_to_tile(sigmas, inner_tile_size)
    eps_pad, _ = pad_to_tile(epsilons, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        d2 = jnp.sum(dr ** 2, axis=-1)
        d = jnp.sqrt(d2 + jnp.asarray(1e-30, dtype=dtype))

        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        safe_nb = jnp.where(mask_j, nb_idx_tile, 0)
        sig_j = sig_pad[safe_nb]
        eps_j = eps_pad[safe_nb]

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, M))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, M))

        # matches: (inner_tile_size, M, tile_size)
        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        idx_i = (start_i + jnp.arange(inner_tile_size))[:, None]
        m = mask_i[:, None] & mask_j & (nb_idx_tile != idx_i) & (d2 > 0.0)
        if cutoff > 0:
            m = m & (d < cutoff)

        s_ij = 0.5 * (sig_i[:, None] + sig_j)
        s_ij_safe = jnp.where(m & (s_ij > 0.0), s_ij, 1.0)
        e_ij = jnp.sqrt(jnp.maximum(eps_i[:, None] * eps_j, 0.0) + jnp.asarray(1e-30, dtype=dtype))

        inv_d = jnp.where(m, 1.0 / d, 0.0)
        u = jnp.where(m, (s_ij * inv_d) ** 6, 0.0)
        v_unsw = jnp.where(m, 4.0 * e_ij * (u * u - u), 0.0)
        dv_unsw_dr = jnp.where(m, 4.0 * e_ij * (-12.0 * u * u + 6.0 * u) * inv_d, 0.0)

        s_w, ds_dr = _lj_switch_weight_and_deriv(d, cutoff, switch_width)
        v_sw = v_unsw * s_w

        # Position gradient: dE/dr_i = sum_k scale * (dv_unsw_dr * S(r) + v_unsw * S'(r)) * (dr / d)
        de_dr_over_r = jnp.where(m, pair_scale * (dv_unsw_dr * s_w + v_unsw * ds_dr) * inv_d, 0.0)
        grad_r_tile = jnp.sum(de_dr_over_r[:, :, None] * dr, axis=1)

        # Sigma gradient
        grad_sig_tile = jnp.sum(
            jnp.where(m, pair_scale * s_w * 2.0 * e_ij * (12.0 * u * u - 6.0 * u) / s_ij_safe, 0.0),
            axis=1,
        )

        # Epsilon gradient
        grad_eps_tile = jnp.sum(
            jnp.where(m, pair_scale * s_w * 2.0 * (u * u - u) * eps_j / e_ij, 0.0),
            axis=1,
        )

        # Exclusion scale gradient: 0.5 * sum_k I(matches[i, m, k]) * v_sw[i, k]
        grad_excl_tile = 0.5 * jnp.sum(
            jnp.where(matches & m[:, None, :], v_sw[:, None, :], 0.0),
            axis=2,
        )

        return jnp.concatenate(
            [grad_r_tile, grad_sig_tile[:, None], grad_eps_tile[:, None], grad_excl_tile],
            axis=1,
        )

    init_combined = jnp.zeros((excl_pad_dim, 5 + M), dtype=dtype)
    combined = tile_reduction_nl(
        r_pad, nb_idx, mask_pad, f_tile_grad, init_combined, tile_size, inner_tile_size=inner_tile_size
    )
    grad_r = combined[:N, :3]
    grad_sig = combined[:N, 3]
    grad_eps = combined[:N, 4]
    grad_excl = combined[: excl_scales.shape[0], 5:]
    return (
        g * grad_r,
        g * grad_sig,
        g * grad_eps,
        jnp.zeros_like(excl_indices),
        g * grad_excl,
        jnp.zeros_like(nb_idx),
    )


chunked_lj_energy_nl.defvjp(_chunked_lj_nl_fwd, _chunked_lj_nl_bwd)


# ============================================================================
# Coulomb Kernels (Dense & NL)
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 4, 5, 6, 7, 8))
def chunked_coulomb_energy(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    excl_indices: jnp.ndarray,
    excl_scales: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    cutoff: float = 9.0,
    tile_size: int = 128,
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy using FlashMD tiling."""
    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(r.shape[0]), int(excl_indices.shape[0]), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    q_pad, _ = pad_to_tile(charges, pad_dim)
    excl_i_pad = jnp.pad(excl_indices, ((0, pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1))
        ds = dist
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        q_ij = q_j[None, :] * q_i[:, None]

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)

        matches = (idx_j[None, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0:
            m &= ds < cutoff

        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        e_pair = coulomb_constant * q_ij * inv_ds * (pair_scale - jax.scipy.special.erf(pme_alpha * ds))
        return 0.5 * jnp.sum(jnp.where(m & (ds > 0.0), e_pair, 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)


def _chunked_coulomb_fwd(r, charges, excl_indices, excl_scales, displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return (
        chunked_coulomb_energy(r, charges, excl_indices, excl_scales, displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size),
        (r, charges, excl_indices, excl_scales, pme_alpha),
    )


def _chunked_coulomb_bwd(excl_indices, displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size, res, g):
    r, charges, excl_idx_orig, excl_scales, pme_alpha = res
    N = r.shape[0]
    n_excl_rows, M = excl_idx_orig.shape
    dtype = r.dtype

    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(N), int(n_excl_rows), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    q_pad, _ = pad_to_tile(charges, pad_dim)
    excl_i_pad = jnp.pad(excl_idx_orig, ((0, pad_dim - n_excl_rows), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        d2 = jnp.sum(dr ** 2, axis=-1)
        d = jnp.sqrt(d2 + jnp.asarray(1e-30, dtype=dtype))

        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, M))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, M))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)

        matches = (idx_j[None, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None]) & (d2 > 0.0)
        if cutoff > 0:
            m = m & (d < cutoff)

        inv_d = jnp.where(m, 1.0 / d, 0.0)
        inv_d2 = inv_d * inv_d
        alpha_d = pme_alpha * d
        erf_factor = jax.scipy.special.erf(alpha_d)
        derf_factor = (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-(alpha_d ** 2)) * pme_alpha
        eff_scale = pair_scale - erf_factor

        q_ij = q_i[:, None] * q_j[None, :]
        dE_dd = jnp.where(m, coulomb_constant * q_ij * (-eff_scale * inv_d2 - derf_factor * inv_d), 0.0)
        grad_r_tile = jnp.sum((dE_dd * inv_d)[:, :, None] * dr, axis=1)

        # Charge gradient: dE/dq_i = sum_j C * q_j * (pair_scale - erf(alpha * d)) / d
        grad_q_tile = jnp.sum(
            jnp.where(m, coulomb_constant * q_j[None, :] * eff_scale * inv_d, 0.0),
            axis=1,
        )

        # Exclusion scale gradient: 0.5 * sum_j I(matches[i, m, j]) * C * q_i * q_j / d
        base_coul = jnp.where(m, coulomb_constant * q_ij * inv_d, 0.0)
        grad_excl_tile = 0.5 * jnp.sum(
            jnp.where(matches & m[:, None, :], base_coul[:, None, :], 0.0),
            axis=2,
        )

        return jnp.concatenate([grad_r_tile, grad_q_tile[:, None], grad_excl_tile], axis=1)

    init_combined = jnp.zeros((pad_dim, 4 + M), dtype=dtype)
    combined = tile_reduction(
        r_pad, mask_pad, f_tile_grad, init_combined, tile_size, inner_tile_size=inner_tile_size
    )
    grad_r = combined[:N, :3]
    grad_q = combined[:N, 3]
    grad_excl = combined[: excl_scales.shape[0], 4:]
    return (g * grad_r, g * grad_q, g * grad_excl)


chunked_coulomb_energy.defvjp(_chunked_coulomb_fwd, _chunked_coulomb_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def chunked_coulomb_energy_nl(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    excl_indices: jnp.ndarray,
    excl_scales: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    cutoff: float = 9.0,
    tile_size: int = 32,
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy over neighbor list."""
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    q_pad, _ = pad_to_tile(charges, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1))
        ds = dist
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        safe_nb = jnp.where(mask_j, nb_idx_tile, 0)
        q_j = q_pad[safe_nb]
        q_ij = q_i[:, None] * q_j
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
        idx_i = (start_i + jnp.arange(inner_tile_size))[:, None]
        m = mask_i[:, None] & mask_j & (nb_idx_tile != idx_i) & (ds > 0.0)
        if cutoff > 0:
            m = m & (ds < cutoff)
        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        e_pair = coulomb_constant * q_ij * inv_ds * (pair_scale - jax.scipy.special.erf(pme_alpha * ds))
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    return tile_reduction_nl(r_pad, neighbor_idx, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)


def _chunked_coulomb_nl_fwd(r, charges, excl_indices, excl_scales, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return (
        chunked_coulomb_energy_nl(r, charges, excl_indices, excl_scales, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size),
        (r, charges, excl_indices, excl_scales, nb_idx, pme_alpha),
    )


def _chunked_coulomb_nl_bwd(displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size, res, g):
    r, charges, excl_indices, excl_scales, nb_idx, pme_alpha = res
    N = r.shape[0]
    M = excl_indices.shape[1]
    dtype = r.dtype

    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    q_pad, _ = pad_to_tile(charges, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        d2 = jnp.sum(dr ** 2, axis=-1)
        d = jnp.sqrt(d2 + jnp.asarray(1e-30, dtype=dtype))

        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        safe_nb = jnp.where(mask_j, nb_idx_tile, 0)
        q_j = q_pad[safe_nb]
        q_ij = q_i[:, None] * q_j

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, M))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, M))
        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None]) & (excl_i[:, :, None] >= 0)
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        idx_i = (start_i + jnp.arange(inner_tile_size))[:, None]
        m = mask_i[:, None] & mask_j & (nb_idx_tile != idx_i) & (d2 > 0.0)
        if cutoff > 0:
            m = m & (d < cutoff)

        inv_d = jnp.where(m, 1.0 / d, 0.0)
        inv_d2 = inv_d * inv_d
        alpha_d = pme_alpha * d
        erf_factor = jax.scipy.special.erf(alpha_d)
        derf_factor = (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-(alpha_d ** 2)) * pme_alpha
        eff_scale = pair_scale - erf_factor

        dE_dd = jnp.where(m, coulomb_constant * q_ij * (-eff_scale * inv_d2 - derf_factor * inv_d), 0.0)
        grad_r_tile = jnp.sum((dE_dd * inv_d)[:, :, None] * dr, axis=1)

        grad_q_tile = jnp.sum(
            jnp.where(m, coulomb_constant * q_j * eff_scale * inv_d, 0.0),
            axis=1,
        )

        base_coul = jnp.where(m, coulomb_constant * q_ij * inv_d, 0.0)
        grad_excl_tile = 0.5 * jnp.sum(
            jnp.where(matches & m[:, None, :], base_coul[:, None, :], 0.0),
            axis=2,
        )

        return jnp.concatenate([grad_r_tile, grad_q_tile[:, None], grad_excl_tile], axis=1)

    init_combined = jnp.zeros((excl_pad_dim, 4 + M), dtype=dtype)
    combined = tile_reduction_nl(
        r_pad, nb_idx, mask_pad, f_tile_grad, init_combined, tile_size, inner_tile_size=inner_tile_size
    )
    grad_r = combined[:N, :3]
    grad_q = combined[:N, 3]
    grad_excl = combined[: excl_scales.shape[0], 4:]
    return (
        g * grad_r,
        g * grad_q,
        jnp.zeros_like(excl_indices),
        g * grad_excl,
        jnp.zeros_like(nb_idx),
    )


chunked_coulomb_energy_nl.defvjp(_chunked_coulomb_nl_fwd, _chunked_coulomb_nl_bwd)
