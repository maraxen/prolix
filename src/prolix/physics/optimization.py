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
    tile_size: int = 128
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
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)
        
        matches = (idx_j[None, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
        
        s_ij = 0.5 * (sig_j[None, :] + sig_i[:, None])
        e_ij = jnp.sqrt(eps_j[None, :] * eps_i[:, None])
        
        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0: m = m & (ds < cutoff)
        
        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        inv_r6 = jnp.where(m & (ds > 0.0), (s_ij * inv_ds)**6, 0.0)
        energy = jnp.where(m & (ds > 0.0), 4.0 * e_ij * (inv_r6**2 - inv_r6), 0.0)
        return 0.5 * jnp.sum(energy * pair_scale)

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

def _chunked_lj_fwd(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, cutoff, tile_size):
    return chunked_lj_energy(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, cutoff, tile_size), (r, sigmas, epsilons, excl_scales)

def _chunked_lj_bwd(excl_indices, displacement_fn, cutoff, tile_size, res, g):
    r, sigmas, epsilons, excl_scales = res
    N = r.shape[0]
    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(N), int(excl_indices.shape[0]), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    sig_pad, _ = pad_to_tile(sigmas, pad_dim)
    eps_pad, _ = pad_to_tile(epsilons, pad_dim)
    excl_i_pad = jnp.pad(excl_indices, ((0, pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)
        
        matches = (idx_j[None, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        def pair_vals(ri, rj, si, sj, ei, ej, scale):
            dr = displacement_fn(ri, rj); d2 = jnp.sum(dr**2); d = jnp.sqrt(d2)
            inv_d = jnp.where(d > 0.0, 1.0 / d, 0.0)
            inv_d2 = jnp.where(d > 0.0, 1.0 / d2, 0.0)
            s = 0.5*(si+sj); e = jnp.sqrt(ei*ej)
            inv_r6 = jnp.where(d > 0.0, (s/d)**6, 0.0)
            dE_ds = 4.0 * e / s * (12.0 * inv_r6**2 - 6.0 * inv_r6)
            f_mag = dE_ds * s * inv_d2 * scale
            return jnp.where(d > 0.0, f_mag * dr, 0.0)
        
        forces = jax.vmap(jax.vmap(pair_vals, (None, 0, None, 0, None, 0, 0)), (0, None, 0, None, 0, None, 0))(
            pos_i, pos_j, sig_i, sig_j, eps_i, eps_j, pair_scale)
        
        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0:
            dist = jnp.sqrt(jnp.sum(jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)**2, axis=-1))
            m = m & (dist < cutoff)
        
        f_on_i = jnp.sum(jnp.where(m[..., None], forces, 0.0), axis=1)
        return f_on_i

    f_res = tile_reduction(r_pad, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size, inner_tile_size=inner_tile_size)
    return (-g * f_res[:N], g * jnp.zeros_like(sigmas), g * jnp.zeros_like(epsilons), g * jnp.zeros_like(excl_scales))

chunked_lj_energy.defvjp(_chunked_lj_fwd, _chunked_lj_bwd)

def _chunked_lj_energy_nl_core(
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
    """Core LJ energy computation (non-custom_vjp). Used by both forward and backward."""
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    sig_pad, _ = pad_to_tile(sigmas, inner_tile_size)
    eps_pad, _ = pad_to_tile(epsilons, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = sig_pad[nb_idx_tile]
        eps_j = eps_pad[nb_idx_tile]

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))

        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        s_ij = 0.5 * (sig_i[:, None] + sig_j)
        e_ij = jnp.sqrt(eps_i[:, None] * eps_j)
        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        inv_r6 = jnp.where(ds > 0.0, (s_ij * inv_ds)**6, 0.0)
        m = mask_i[:, None] & mask_j & (ds > 0.0)
        if cutoff > 0: m = m & (ds < cutoff)
        sw = None if switch_width == 0.0 else switch_width
        switch = lj_switch_weight(ds, cutoff, sw)
        energy = jnp.where(m, 4.0 * e_ij * (inv_r6**2 - inv_r6) * switch, 0.0)
        return 0.5 * jnp.sum(energy * pair_scale)

    return tile_reduction_nl(r_pad, neighbor_idx, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)


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
    """Computes LJ energy over neighbor list.

    custom_vjp backward (debt 760 / #4623 / #4793): an earlier attempt at
    this analytic backward pass (commit 7f7e147) had a copy-paste bug
    (hardcoded zero position gradient) that was correctly fixed and
    validated (finite-difference, tests/physics/test_nl_kernel_gradient_correctness.py)
    -- but that fix lived on a branch that a parallel debt-4383 branch's
    "remove custom_vjp entirely" fix clobbered on merge without ever seeing
    it (commit 223590b's "before" state matches 7f7e147's "before" state,
    not its "after"). Restored 2026-09 (#4793/#4623): generic autodiff
    through the gather-heavy tiled NL forward pass forces JAX to synthesize
    an expensive scatter-add backward (measured ~20x slower than dense's
    analytic-custom_vjp force path at n_padded=5000); this restores the
    analytic rule, matching the dense kernel's own custom_vjp pattern.
    """
    return _chunked_lj_energy_nl_core(r, sigmas, epsilons, excl_indices, excl_scales, neighbor_idx,
                                       displacement_fn, cutoff, tile_size, switch_width)


def _chunked_lj_nl_fwd(r, sigmas, epsilons, excl_indices, excl_scales, nb_idx, disp_fn, cutoff, tile_size, switch_width):
    return _chunked_lj_energy_nl_core(r, sigmas, epsilons, excl_indices, excl_scales, nb_idx, disp_fn, cutoff, tile_size, switch_width), (r, sigmas, epsilons, excl_indices, excl_scales, nb_idx)


def _chunked_lj_nl_bwd(displacement_fn, cutoff, tile_size, switch_width, res, g):
    # excl_indices and neighbor_idx (nb_idx) are differentiable-position
    # arguments (not nondiff_argnums, debt 802) purely so they can be
    # jax.vmap/jax.lax.while_loop tracers (per-replica topology under a
    # stacked/vmapped EnsemblePlan dispatch, or a periodically-updated
    # neighbor list) -- gradient w.r.t. an integer index array is always zero.
    r, sigmas, epsilons, excl_indices, excl_scales, nb_idx = res
    N = r.shape[0]
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    sig_pad, _ = pad_to_tile(sigmas, inner_tile_size)
    eps_pad, _ = pad_to_tile(epsilons, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = sig_pad[nb_idx_tile]
        eps_j = eps_pad[nb_idx_tile]

        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))

        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        def pair_vals(ri, rj, si, sj, ei, ej, scale):
            dr = displacement_fn(ri, rj); d2 = jnp.sum(dr**2); d = jnp.sqrt(d2)
            inv_d = jnp.where(d > 0.0, 1.0 / d, 0.0)
            inv_d2 = jnp.where(d > 0.0, 1.0 / d2, 0.0)
            s = 0.5*(si+sj); e = jnp.sqrt(ei*ej)
            inv_r6 = jnp.where(d > 0.0, (s/d)**6, 0.0)

            # Unswitched LJ force
            dE_ds = 4.0 * e / s * (12.0 * inv_r6**2 - 6.0 * inv_r6)
            f_mag_unswitched = dE_ds * s * inv_d2 * scale
            f_pair_unswitched = jnp.where(d > 0.0, f_mag_unswitched * dr, 0.0)

            # Switch function chain-rule term
            if switch_width is not None and switch_width > 0.0:
                r_cut = jnp.asarray(cutoff, dtype=d.dtype)
                width = jnp.asarray(switch_width, dtype=d.dtype)
                r_switch = r_cut - width
                z = (d - r_switch) / jnp.maximum(width, jnp.asarray(1e-12, dtype=d.dtype))

                # dS_dd = (-30*z^2 + 60*z^3 - 30*z^4) / width
                dS_dd = (-30.0 * z**2 + 60.0 * z**3 - 30.0 * z**4) / jnp.maximum(width, jnp.asarray(1e-12, dtype=d.dtype))
                # dS_dd is nonzero only inside switch window: r_switch < d < r_cut
                dS_dd = jnp.where((d > r_switch) & (d < r_cut), dS_dd, 0.0)

                # Unswitched LJ pair energy
                E_lj_unswitched = 4.0 * e * (inv_r6**2 - inv_r6)

                # S(d) multiplier on the unswitched force
                switch = lj_switch_weight(d, cutoff, switch_width)

                # Full switched per-pair force: F_pair = S(d)*F_pair_unswitched - E_lj_unswitched * dS_dd * (dr/d)
                f_pair_switched = switch * f_pair_unswitched - E_lj_unswitched * dS_dd * jnp.where(d > 0.0, dr * inv_d, 0.0)
                return jnp.where(d > 0.0, f_pair_switched, 0.0)
            else:
                return f_pair_unswitched

        forces = jax.vmap(jax.vmap(pair_vals, (None, 0, None, 0, None, 0, 0)), (0, 0, 0, 0, 0, 0, 0))(
            pos_i, pos_j, sig_i, sig_j, eps_i, eps_j, pair_scale)

        # Compute distance for cutoff check
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist

        m = mask_i[:, None] & mask_j & (ds > 0.0)
        if cutoff > 0: m = m & (ds < cutoff)

        f_on_i = jnp.sum(jnp.where(m[..., None], forces, 0.0), axis=1)
        return f_on_i

    f_res = tile_reduction_nl(r_pad, nb_idx, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size, inner_tile_size=inner_tile_size)
    return (
        -g * f_res[:N],
        g * jnp.zeros_like(sigmas),
        g * jnp.zeros_like(epsilons),
        jnp.zeros_like(excl_indices),
        g * jnp.zeros_like(excl_scales),
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
    tile_size: int = 128
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy using FlashMD tiling."""
    # pad_dim / inner_tile_size via XR-BUCKET helper (#746).
    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(r.shape[0]), int(excl_indices.shape[0]), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    q_pad, _ = pad_to_tile(charges, pad_dim)
    excl_i_pad = jnp.pad(excl_indices, ((0, pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        q_ij = q_j[None, :] * q_i[:, None]
        
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)
        
        matches = (idx_j[None, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
        
        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0: m &= (ds < cutoff)
        
        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        e_pair = coulomb_constant * q_ij * inv_ds * (pair_scale - jax.scipy.special.erf(pme_alpha * ds))
        return 0.5 * jnp.sum(jnp.where(m & (ds > 0.0), e_pair, 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

def _chunked_coulomb_fwd(r, charges, excl_indices, excl_scales, displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return chunked_coulomb_energy(r, charges, excl_indices, excl_scales, displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size), (r, charges, excl_indices, excl_scales, pme_alpha)

def _chunked_coulomb_bwd(excl_indices, displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size, res, g):
    r, charges, excl_idx_orig, excl_scales, pme_alpha = res
    N = r.shape[0]
    pad_dim, inner_tile_size = compute_dense_tiling_dims(
        int(N), int(excl_idx_orig.shape[0]), tile_size
    )
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    q_pad, _ = pad_to_tile(charges, pad_dim)
    excl_i_pad = jnp.pad(excl_idx_orig, ((0, pad_dim - excl_idx_orig.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_idx_orig.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_idx_orig.shape[1]))
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)

        matches = (idx_j[None, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        def pair_vals(ri, rj, qi, qj, scale):
            dr = displacement_fn(ri, rj); d2 = jnp.sum(dr**2); d = jnp.sqrt(d2)
            inv_d = jnp.where(d > 0.0, 1.0/d, 0.0)
            inv_d2 = jnp.where(d > 0.0, 1.0/d2, 0.0)
            alpha_d = pme_alpha * d
            erf_factor = jax.scipy.special.erf(alpha_d)
            derf_factor = (2.0/jnp.sqrt(jnp.pi)) * jnp.exp(-alpha_d**2) * pme_alpha
            dE_dd = coulomb_constant * qi * qj * (-scale * inv_d2 + erf_factor * inv_d2 - derf_factor * inv_d)
            return jnp.where(d > 0.0, dE_dd * dr * inv_d, 0.0)

        forces = jax.vmap(jax.vmap(pair_vals, (None, 0, None, 0, 0)), (0, None, 0, None, 0))(
            pos_i, pos_j, q_i, q_j, pair_scale)
        
        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0:
            dist = jnp.sqrt(jnp.sum(jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)**2, axis=-1))
            m &= (dist < cutoff)
        
        f_on_i = jnp.sum(jnp.where(m[..., None], forces, 0.0), axis=1)
        return f_on_i

    f_res = tile_reduction(r_pad, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size, inner_tile_size=inner_tile_size)
    return (g * f_res[:N], g * jnp.zeros_like(charges), g * jnp.zeros_like(excl_scales))

chunked_coulomb_energy.defvjp(_chunked_coulomb_fwd, _chunked_coulomb_bwd)

def _chunked_coulomb_energy_nl_core(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    excl_indices: jnp.ndarray,
    excl_scales: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    cutoff: float = 9.0,
    tile_size: int = 32
) -> jnp.ndarray:
    """Core Coulomb energy computation (non-custom_vjp). Used by both forward and backward."""
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    q_pad, _ = pad_to_tile(charges, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = q_pad[nb_idx_tile]
        q_ij = q_i[:, None] * q_j
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
        m = mask_i[:, None] & mask_j & (ds > 0.0)
        if cutoff > 0: m = m & (ds < cutoff)
        inv_ds = 1.0 / jnp.where(ds > 0.0, ds, 1.0)
        e_pair = coulomb_constant * q_ij * inv_ds * (pair_scale - jax.scipy.special.erf(pme_alpha * ds))
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    return tile_reduction_nl(r_pad, neighbor_idx, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

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
    tile_size: int = 32
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy over neighbor list.

    custom_vjp backward: see chunked_lj_energy_nl's docstring (#4623/#4793) --
    same restoration, same rationale.
    """
    return _chunked_coulomb_energy_nl_core(r, charges, excl_indices, excl_scales, neighbor_idx,
                                           displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size)


def _chunked_coulomb_nl_fwd(r, charges, excl_indices, excl_scales, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return _chunked_coulomb_energy_nl_core(r, charges, excl_indices, excl_scales, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size), (r, charges, excl_indices, excl_scales, nb_idx, pme_alpha)


def _chunked_coulomb_nl_bwd(displacement_fn, pme_alpha, coulomb_constant, cutoff, tile_size, res, g):
    # excl_indices and neighbor_idx (nb_idx) are differentiable-position
    # arguments (not nondiff_argnums, debt 802) purely so they can be
    # jax.vmap/jax.lax.while_loop tracers (per-replica topology under a
    # stacked/vmapped EnsemblePlan dispatch, or a periodically-updated
    # neighbor list) -- gradient w.r.t. an integer index array is always zero.
    r, charges, excl_indices, excl_scales, nb_idx, pme_alpha = res
    N = r.shape[0]
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    q_pad, _ = pad_to_tile(charges, inner_tile_size)
    excl_pad_dim = r_pad.shape[0]
    excl_i_pad = jnp.pad(excl_indices, ((0, excl_pad_dim - excl_indices.shape[0]), (0, 0)), constant_values=-1)
    excl_s_pad = jnp.pad(excl_scales, ((0, excl_pad_dim - excl_scales.shape[0]), (0, 0)), constant_values=1.0)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = q_pad[nb_idx_tile]
        excl_i = jax.lax.dynamic_slice(excl_i_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))
        excl_s = jax.lax.dynamic_slice(excl_s_pad, (start_i, 0), (inner_tile_size, excl_indices.shape[1]))

        matches = (nb_idx_tile[:, None, :] == excl_i[:, :, None])
        pair_scale = jnp.sum(jnp.where(matches, excl_s[:, :, None], 0.0), axis=1)
        pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)

        def pair_vals(ri, rj, qi, qj, scale):
            dr = displacement_fn(ri, rj); d2 = jnp.sum(dr**2); d = jnp.sqrt(d2)
            inv_d = jnp.where(d > 0.0, 1.0/d, 0.0)
            inv_d2 = jnp.where(d > 0.0, 1.0/d2, 0.0)
            alpha_d = pme_alpha * d
            erf_factor = jax.scipy.special.erf(alpha_d)
            derf_factor = (2.0/jnp.sqrt(jnp.pi)) * jnp.exp(-alpha_d**2) * pme_alpha
            dE_dd = coulomb_constant * qi * qj * (-scale * inv_d2 + erf_factor * inv_d2 - derf_factor * inv_d)
            return jnp.where(d > 0.0, dE_dd * dr * inv_d, 0.0)

        forces = jax.vmap(jax.vmap(pair_vals, (None, 0, None, 0, 0)), (0, 0, 0, 0, 0))(
            pos_i, pos_j, q_i, q_j, pair_scale)

        # Compute distance for cutoff check
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist

        m = mask_i[:, None] & mask_j & (ds > 0.0)
        if cutoff > 0: m = m & (ds < cutoff)

        f_on_i = jnp.sum(jnp.where(m[..., None], forces, 0.0), axis=1)
        return f_on_i

    f_res = tile_reduction_nl(r_pad, nb_idx, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size, inner_tile_size=inner_tile_size)
    return (
        g * f_res[:N],
        g * jnp.zeros_like(charges),
        jnp.zeros_like(excl_indices),
        g * jnp.zeros_like(excl_scales),
        jnp.zeros_like(nb_idx),
    )


chunked_coulomb_energy_nl.defvjp(_chunked_coulomb_nl_fwd, _chunked_coulomb_nl_bwd)
