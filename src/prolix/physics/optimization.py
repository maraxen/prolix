import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from typing import Any, Callable, TypeVar, Tuple
from prolix.typing import PhysicsSystem, DifferentiableParams
from .tiling import tile_reduction, tile_reduction_nl, pad_to_tile

T = TypeVar("T")

# ============================================================================
# LJ Kernels (Dense & NL)
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def chunked_lj_energy(
    r: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    displacement_fn: Callable,
    cutoff: float = 9.0,
    tile_size: int = 128
) -> jnp.ndarray:
    """Computes dense LJ energy using FlashMD tiling O(N^2)."""
    inner_tile_size = 1024
    pad_dim = max(tile_size, inner_tile_size)
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    sig_pad, _ = pad_to_tile(sigmas, pad_dim)
    eps_pad, _ = pad_to_tile(epsilons, pad_dim)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        
        s_ij = 0.5 * (sig_j[None, :] + sig_i[:, None])
        e_ij = jnp.sqrt(eps_j[None, :] * eps_i[:, None])
        
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)
        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        
        if cutoff > 0: m = m & (ds < cutoff)
        inv_r6 = (s_ij / ds)**6
        return 0.5 * jnp.sum(jnp.where(m, 4.0 * e_ij * (inv_r6**2 - inv_r6), 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

def _chunked_lj_fwd(r, sigmas, epsilons, disp_fn, cutoff, tile_size):
    return chunked_lj_energy(r, sigmas, epsilons, disp_fn, cutoff, tile_size), (r, sigmas, epsilons)

def _chunked_lj_bwd(disp_fn, cutoff, tile_size, res, g):
    r, sigmas, epsilons = res
    N = r.shape[0]
    inner_tile_size = 1024
    pad_dim = max(tile_size, inner_tile_size)
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    sig_pad, _ = pad_to_tile(sigmas, pad_dim)
    eps_pad, _ = pad_to_tile(epsilons, pad_dim)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        
        def pair_vals(ri, rj, si, sj, ei, ej):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; d = jnp.sqrt(d2); inv_d = 1.0/d
            s = 0.5*(si+sj); e = jnp.sqrt(ei*ej); inv_r6 = (s/d)**6
            dE_de = 4.0 * (inv_r6**2 - inv_r6)
            dE_ds = 4.0 * e / s * (12.0 * inv_r6**2 - 6.0 * inv_r6)
            # f = -dE/dr = -dE/dd * dr/d. 
            # dE/dd = -dE/ds * (s/d) => f = dE/ds * (s/d) * dr/d = dE/ds * s / d^2 * dr
            f_mag = dE_ds * s * inv_d**2
            g_ei = dE_de * 0.5 * jnp.sqrt(ej/(ei+1e-12))
            g_si = dE_ds * 0.5
            return f_mag * dr, g_si, g_ei
        
        res = jax.vmap(jax.vmap(pair_vals, (None, 0, None, 0, None, 0)), (0, None, 0, None, 0, None))(pos_i, pos_j, sig_i, sig_j, eps_i, eps_j)
        forces, g_si, g_ei = res
        
        idx_i = start_i + jnp.arange(inner_tile_size)
        idx_j = start_j + jnp.arange(tile_size)
        m = (idx_j[None, :] != idx_i[:, None]) & (mask_j[None, :] & mask_i[:, None])
        if cutoff > 0:
            dist = jnp.sqrt(jnp.sum(jax.vmap(jax.vmap(disp_fn, (None, 0)), (0, None))(pos_i, pos_j)**2, axis=-1))
            m = m & (dist < cutoff)
        
        f_on_i = jnp.sum(jnp.where(m[..., None], forces, 0.0), axis=1)
        gs_i = jnp.sum(jnp.where(m, g_si, 0.0), axis=1)
        ge_i = jnp.sum(jnp.where(m, g_ei, 0.0), axis=1)
        return f_on_i, gs_i, ge_i

    f_res = tile_reduction(r_pad, mask_pad, lambda *a: f_tile_grad(*a)[0], jnp.zeros_like(r_pad), tile_size, inner_tile_size=inner_tile_size)
    gs_res = tile_reduction(r_pad, mask_pad, lambda *a: f_tile_grad(*a)[1], jnp.zeros_like(sig_pad), tile_size, inner_tile_size=inner_tile_size)
    ge_res = tile_reduction(r_pad, mask_pad, lambda *a: f_tile_grad(*a)[2], jnp.zeros_like(eps_pad), tile_size, inner_tile_size=inner_tile_size)
    
    return (-g * f_res[:N], g * gs_res[:N], g * ge_res[:N])

chunked_lj_energy.defvjp(_chunked_lj_fwd, _chunked_lj_bwd)

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def chunked_lj_energy_nl(
    r: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    cutoff: float = 9.0,
    tile_size: int = 32
) -> jnp.ndarray:
    """Computes LJ energy over neighbor list."""
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    sig_pad, _ = pad_to_tile(sigmas, inner_tile_size)
    eps_pad, _ = pad_to_tile(epsilons, inner_tile_size)
    
    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        sig_i = jax.lax.dynamic_slice(sig_pad, (start_i,), (inner_tile_size,))
        eps_i = jax.lax.dynamic_slice(eps_pad, (start_i,), (inner_tile_size,))
        sig_j = sig_pad[nb_idx_tile]
        eps_j = eps_pad[nb_idx_tile]
        s_ij = 0.5 * (sig_i[:, None] + sig_j)
        e_ij = jnp.sqrt(eps_i[:, None] * eps_j)
        inv_r6 = (s_ij / ds)**6
        m = mask_i[:, None] & mask_j
        if cutoff > 0: m = m & (ds < cutoff)
        return 0.5 * jnp.sum(jnp.where(m, 4.0 * e_ij * (inv_r6**2 - inv_r6), 0.0))

    return tile_reduction_nl(r_pad, neighbor_idx, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

def _chunked_lj_nl_fwd(r, sigmas, epsilons, nb_idx, disp_fn, cutoff, tile_size):
    return chunked_lj_energy_nl(r, sigmas, epsilons, nb_idx, disp_fn, cutoff, tile_size), (r, sigmas, epsilons, nb_idx)

def _chunked_lj_nl_bwd(disp_fn, cutoff, tile_size, res, g):
    # Simplified placeholder
    return (-g * jnp.zeros_like(res[0]), g * jnp.zeros_like(res[1]), g * jnp.zeros_like(res[2]))

chunked_lj_energy_nl.defvjp(_chunked_lj_nl_fwd, _chunked_lj_nl_bwd)

# ============================================================================
# Coulomb Kernels (Dense & NL)
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 4, 5, 6))
def chunked_coulomb_energy(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    cutoff: float = 9.0,
    tile_size: int = 128
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy using FlashMD tiling."""
    inner_tile_size = 1024
    pad_dim = max(tile_size, inner_tile_size)
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    q_pad, _ = pad_to_tile(charges, pad_dim)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        q_ij = q_j[None, :] * q_i[:, None]
        m = (start_j + jnp.arange(tile_size))[None, :] != (start_i + jnp.arange(inner_tile_size))[:, None]
        m &= mask_j[None, :] & mask_i[:, None]
        if cutoff > 0: m &= (ds < cutoff)
        e_pair = coulomb_constant * (q_ij / ds) * jax.scipy.special.erfc(pme_alpha * ds)
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

def _chunked_coulomb_fwd(r, charges, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return chunked_coulomb_energy(r, charges, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size), (r, charges, pme_alpha)

def _chunked_coulomb_bwd(disp_fn, coulomb_constant, cutoff, tile_size, res, g):
    r, charges, pme_alpha = res
    N = r.shape[0]
    inner_tile_size = 1024
    pad_dim = max(tile_size, inner_tile_size)
    r_pad, mask_pad = pad_to_tile(r, pad_dim)
    q_pad, _ = pad_to_tile(charges, pad_dim)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_i, start_j):
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        
        def pair_vals(ri, rj, qi, qj):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; d = jnp.sqrt(d2)
            f_mag = coulomb_constant * qi * qj * (jax.scipy.special.erfc(pme_alpha*d)/(d**3) + (2.0*pme_alpha/(jnp.sqrt(jnp.pi)*d**2))*jnp.exp(-(pme_alpha*d)**2))
            g_qi = coulomb_constant * qj / d * jax.scipy.special.erfc(pme_alpha * d)
            g_alpha = -2.0 * coulomb_constant * qi * qj / jnp.sqrt(jnp.pi) * jnp.exp(-(pme_alpha * d)**2)
            return f_mag * dr, g_qi, g_alpha
            
        forces, g_qi_vals, g_a_vals = jax.vmap(jax.vmap(pair_vals, (None, 0, None, 0)), (0, None, 0, None))(pos_i, pos_j, q_i, q_j)
        m = (start_j + jnp.arange(tile_size))[None, :] != (start_i + jnp.arange(inner_tile_size))[:, None]
        m &= mask_j[None, :] & mask_i[:, None]
        if cutoff > 0:
            dist = jnp.sqrt(jnp.sum(jax.vmap(jax.vmap(disp_fn, (None, 0)), (0, None))(pos_i, pos_j)**2, axis=-1))
            m &= (dist < cutoff)
        
        f_on_i = jnp.sum(jnp.where(m[..., None], forces, 0.0), axis=1)
        gq_i = jnp.sum(jnp.where(m, g_qi_vals, 0.0), axis=1)
        ga = jnp.sum(jnp.where(m, g_a_vals, 0.0))
        return f_on_i, gq_i, ga

    f_res = tile_reduction(r_pad, mask_pad, lambda *a: f_tile_grad(*a)[0], jnp.zeros_like(r_pad), tile_size, inner_tile_size=inner_tile_size)
    gq_res = tile_reduction(r_pad, mask_pad, lambda *a: f_tile_grad(*a)[1], jnp.zeros_like(q_pad), tile_size, inner_tile_size=inner_tile_size)
    ga_res = tile_reduction(r_pad, mask_pad, lambda *a: f_tile_grad(*a)[2], 0.0, tile_size, inner_tile_size=inner_tile_size)
    
    return (-g * f_res[:N], g * gq_res[:N], g * ga_res * 0.5)

chunked_coulomb_energy.defvjp(_chunked_coulomb_fwd, _chunked_coulomb_bwd)

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 5, 6, 7))
def chunked_coulomb_energy_nl(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0637,
    cutoff: float = 9.0,
    tile_size: int = 32
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy over neighbor list."""
    inner_tile_size = 1024
    r_pad, mask_pad = pad_to_tile(r, inner_tile_size)
    q_pad, _ = pad_to_tile(charges, inner_tile_size)
    
    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1)); ds = dist
        q_i = jax.lax.dynamic_slice(q_pad, (start_i,), (inner_tile_size,))
        q_j = q_pad[nb_idx_tile]
        q_ij = q_i[:, None] * q_j
        m = mask_i[:, None] & mask_j
        if cutoff > 0: m = m & (ds < cutoff)
        e_pair = coulomb_constant * (q_ij / ds) * jax.scipy.special.erfc(pme_alpha * ds)
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    return tile_reduction_nl(r_pad, neighbor_idx, mask_pad, f_tile, 0.0, tile_size, inner_tile_size=inner_tile_size)

def _chunked_coulomb_nl_fwd(r, charges, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return chunked_coulomb_energy_nl(r, charges, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size), (r, charges, nb_idx, pme_alpha)

def _chunked_coulomb_nl_bwd(disp_fn, coulomb_constant, cutoff, tile_size, res, g):
    return (-g * jnp.zeros_like(res[0]), g * jnp.zeros_like(res[1]), g * 0.0)

chunked_coulomb_energy_nl.defvjp(_chunked_coulomb_nl_fwd, _chunked_coulomb_nl_bwd)
