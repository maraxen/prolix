import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from typing import Any, Callable, TypeVar, Tuple
from .types import PhysicsSystem, EnergyParams
from .tiling import tile_reduction, pad_to_tile

T = TypeVar("T")

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def chunked_lj_energy(
    r: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    displacement_fn: Callable,
    tile_size: int = 128
) -> jnp.ndarray:
    """Computes LJ energy using configurable FlashMD tiling."""
    N = r.shape[0]
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    sig_pad, _ = pad_to_tile(sigmas, tile_size)
    eps_pad, _ = pad_to_tile(epsilons, tile_size)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_j, pos_i)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
        ds = dist + 1e-6
        
        # Slice static params for tile j
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        
        s_ij = 0.5 * (sig_j[:, None] + sig_pad[None, :])
        e_ij = jnp.sqrt(eps_j[:, None] * eps_pad[None, :])
        
        idx_i = jnp.arange(r_pad.shape[0])
        idx_j = start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        
        inv_r6 = (s_ij / ds)**6
        return 0.5 * jnp.sum(jnp.where(m, 4.0 * e_ij * (inv_r6**2 - inv_r6), 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size)

def _chunked_lj_fwd(r, sigmas, epsilons, disp_fn, tile_size):
    return chunked_lj_energy(r, sigmas, epsilons, disp_fn, tile_size), (r, sigmas, epsilons)

def _chunked_lj_bwd(disp_fn, tile_size, res, g):
    r, sigmas, epsilons = res
    N = r.shape[0]
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    sig_pad, _ = pad_to_tile(sigmas, tile_size)
    eps_pad, _ = pad_to_tile(epsilons, tile_size)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_j):
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        
        def pair_force(ri, rj, si, sj, ei, ej):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; s = 0.5*(si+sj); e = jnp.sqrt(ei*ej)
            inv_r2 = 1.0/d2; inv_r6 = (s**2*inv_r2)**3
            return (24.0*e*(2.0*inv_r6**2-inv_r6)*inv_r2) * dr

        forces = jax.vmap(jax.vmap(pair_force, (None,0,None,0,None,0)), (0,None,0,None,0,None))(pos_j, pos_i, sig_j, sig_pad, eps_j, eps_pad)
        
        idx_i = jnp.arange(r_pad.shape[0])
        idx_j = start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        forces = jnp.where(m[..., None], forces, 0.0)
        
        f_on_j = jnp.sum(forces, axis=1)
        f_on_all = -jnp.sum(forces, axis=0)
        
        val = jnp.zeros_like(pos_i)
        val = jax.lax.dynamic_update_slice(val, f_on_j, (start_j, 0))
        return val + f_on_all

    forces_pad = tile_reduction(r_pad, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size)
    return (-g * forces_pad[:N] * 0.5, jnp.zeros_like(sigmas), jnp.zeros_like(epsilons))

chunked_lj_energy.defvjp(_chunked_lj_fwd, _chunked_lj_bwd)

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def chunked_coulomb_energy(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0636,
    tile_size: int = 128
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy using configurable FlashMD tiling."""
    N = r.shape[0]
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    q_pad, _ = pad_to_tile(charges, tile_size)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_j, pos_i)
        dist = jnp.sqrt(jnp.sum(dr**2) + 1e-12); ds = dist + 1e-6
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        q_ij = q_j[:, None] * q_pad[None, :]
        idx_i, idx_j = jnp.arange(r_pad.shape[0]), start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        e_pair = coulomb_constant * (q_ij / ds) * jax.scipy.special.erfc(pme_alpha * ds)
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size)

def _chunked_coulomb_fwd(r, charges, disp_fn, pme_alpha, coulomb_constant, tile_size):
    return chunked_coulomb_energy(r, charges, disp_fn, pme_alpha, coulomb_constant, tile_size), (r, charges)

def _chunked_coulomb_bwd(disp_fn, pme_alpha, coulomb_constant, tile_size, res, g):
    r, charges = res
    N = r.shape[0]
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    q_pad, _ = pad_to_tile(charges, tile_size)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_j):
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        def pair_force(ri, rj, qi, qj):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; d = jnp.sqrt(d2)
            f_mag = coulomb_constant * qi * qj * (jax.scipy.special.erfc(pme_alpha*d)/(d**3) + (2.0*pme_alpha/(jnp.sqrt(jnp.pi)*d**2))*jnp.exp(-(pme_alpha*d)**2))
            return f_mag * dr
        forces = jax.vmap(jax.vmap(pair_force, (None,0,None,0)), (0,None,0,None))(pos_j, pos_i, q_j, q_pad)
        idx_i, idx_j = jnp.arange(r_pad.shape[0]), start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        forces = jnp.where(m[..., None], forces, 0.0)
        f_on_j = jnp.sum(forces, axis=1); f_on_all = -jnp.sum(forces, axis=0)
        val = jnp.zeros_like(pos_i); val = jax.lax.dynamic_update_slice(val, f_on_j, (start_j, 0))
        return val + f_on_all

    forces_pad = tile_reduction(r_pad, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size)
    return (-g * forces_pad[:N] * 0.5, jnp.zeros_like(charges))

chunked_coulomb_energy.defvjp(_chunked_coulomb_fwd, _chunked_coulomb_bwd)
