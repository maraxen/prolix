import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from typing import Any, Callable, TypeVar, Tuple
from .types import PhysicsSystem, EnergyParams
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
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    sig_pad, _ = pad_to_tile(sigmas, tile_size)
    eps_pad, _ = pad_to_tile(epsilons, tile_size)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_j, pos_i)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12); ds = dist + 1e-6
        sig_j = jax.lax.dynamic_slice(sig_pad, (start_j,), (tile_size,))
        eps_j = jax.lax.dynamic_slice(eps_pad, (start_j,), (tile_size,))
        s_ij = 0.5 * (sig_j[:, None] + sig_pad[None, :])
        e_ij = jnp.sqrt(eps_j[:, None] * eps_pad[None, :])
        idx_i, idx_j = jnp.arange(r_pad.shape[0]), start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        if cutoff > 0: m = m & (ds < cutoff)
        inv_r6 = (s_ij / ds)**6
        return 0.5 * jnp.sum(jnp.where(m, 4.0 * e_ij * (inv_r6**2 - inv_r6), 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size)

def _chunked_lj_fwd(r, sigmas, epsilons, disp_fn, cutoff, tile_size):
    return chunked_lj_energy(r, sigmas, epsilons, disp_fn, cutoff, tile_size), (r, sigmas, epsilons)

def _chunked_lj_bwd(disp_fn, cutoff, tile_size, res, g):
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
            f_mag = (24.0*e*(2.0*inv_r6**2-inv_r6)*inv_r2)
            if cutoff > 0: f_mag = jnp.where(jnp.sqrt(d2) < cutoff, f_mag, 0.0)
            return f_mag * dr
        forces = jax.vmap(jax.vmap(pair_force, (None,0,None,0,None,0)), (0,None,0,None,0,None))(pos_j, pos_i, sig_j, sig_pad, eps_j, eps_pad)
        idx_i, idx_j = jnp.arange(r_pad.shape[0]), start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        forces = jnp.where(m[..., None], forces, 0.0)
        f_on_j = jnp.sum(forces, axis=1); f_on_all = -jnp.sum(forces, axis=0)
        val = jnp.zeros_like(pos_i); val = jax.lax.dynamic_update_slice(val, f_on_j, (start_j, 0))
        return val + f_on_all

    forces_pad = tile_reduction(r_pad, mask_pad, f_tile_grad, jnp.zeros_like(r_pad), tile_size)
    return (-g * forces_pad[:N] * 0.5, jnp.zeros_like(sigmas), jnp.zeros_like(epsilons))

chunked_lj_energy.defvjp(_chunked_lj_fwd, _chunked_lj_bwd)

@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def chunked_lj_energy_nl(
    r: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    cutoff: float = 9.0,
    tile_size: int = 32
) -> jnp.ndarray:
    """Computes LJ energy over neighbor list using O(N*K) tiling."""
    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_idx):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12); ds = dist + 1e-6
        sig_j = sigmas[nb_idx_tile]; eps_j = epsilons[nb_idx_tile]
        s_ij = 0.5 * (sigmas[:, None] + sig_j); e_ij = jnp.sqrt(epsilons[:, None] * eps_j)
        inv_r6 = (s_ij / ds)**6
        m = mask_i[:, None] & mask_j
        if cutoff > 0: m = m & (ds < cutoff)
        return 0.5 * jnp.sum(jnp.where(m, 4.0 * e_ij * (inv_r6**2 - inv_r6), 0.0))

    atom_mask = jnp.ones(r.shape[0], dtype=bool) 
    return tile_reduction_nl(r, neighbor_idx, atom_mask, f_tile, 0.0, tile_size)

def _chunked_lj_nl_fwd(r, sigmas, epsilons, nb_idx, disp_fn, cutoff, tile_size):
    return chunked_lj_energy_nl(r, sigmas, epsilons, nb_idx, disp_fn, cutoff, tile_size), (r, sigmas, epsilons, nb_idx)

def _chunked_lj_nl_bwd(disp_fn, cutoff, tile_size, res, g):
    r, sigmas, epsilons, nb_idx = res
    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_idx):
        def pair_force(ri, rj, si, sj, ei, ej):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; s = 0.5*(si+sj); e = jnp.sqrt(ei*ej)
            inv_r2 = 1.0/d2; inv_r6 = (s**2*inv_r2)**3
            f_mag = (24.0*e*(2.0*inv_r6**2-inv_r6)*inv_r2)
            if cutoff > 0: f_mag = jnp.where(jnp.sqrt(d2) < cutoff, f_mag, 0.0)
            return f_mag * dr
        sig_j, eps_j = sigmas[nb_idx_tile], epsilons[nb_idx_tile]
        forces = jax.vmap(jax.vmap(pair_force, (None, 0, None, 0, None, 0)), (0, 0, 0, 0, 0, 0))(pos_i, pos_j, sigmas, sig_j, epsilons, eps_j)
        m = mask_i[:, None] & mask_j
        forces = jnp.where(m[..., None], forces, 0.0)
        f_on_i = jnp.sum(forces, axis=1)
        f_on_j_contrib = -forces
        nb_flat = nb_idx_tile.reshape(-1)
        f_on_j_flat = f_on_j_contrib.reshape(-1, 3)
        f_total = jnp.zeros_like(pos_i).at[jnp.arange(pos_i.shape[0])].add(f_on_i)
        f_total = f_total.at[nb_flat].add(f_on_j_flat)
        return f_total

    atom_mask = jnp.ones(r.shape[0], dtype=bool)
    forces = tile_reduction_nl(r, nb_idx, atom_mask, f_tile_grad, jnp.zeros_like(r), tile_size)
    return (-g * forces * 0.5, jnp.zeros_like(sigmas), jnp.zeros_like(epsilons), None)

chunked_lj_energy_nl.defvjp(_chunked_lj_nl_fwd, _chunked_lj_nl_bwd)

# ============================================================================
# Coulomb Kernels (Dense & NL)
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def chunked_coulomb_energy(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0636,
    cutoff: float = 9.0,
    tile_size: int = 128
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy using FlashMD tiling O(N^2)."""
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    q_pad, _ = pad_to_tile(charges, tile_size)

    def f_tile(pos_i, pos_j, mask_i, mask_j, start_j):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(pos_j, pos_i)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12); ds = dist + 1e-6
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        q_ij = q_j[:, None] * q_pad[None, :]
        idx_i, idx_j = jnp.arange(r_pad.shape[0]), start_j + jnp.arange(tile_size)
        m = (idx_j[:, None] != idx_i[None, :]) & (mask_j[:, None] & mask_i[None, :])
        if cutoff > 0: m = m & (ds < cutoff)
        e_pair = coulomb_constant * (q_ij / ds) * jax.scipy.special.erfc(pme_alpha * ds)
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    return tile_reduction(r_pad, mask_pad, f_tile, 0.0, tile_size)

def _chunked_coulomb_fwd(r, charges, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return chunked_coulomb_energy(r, charges, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size), (r, charges)

def _chunked_coulomb_bwd(disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size, res, g):
    r, charges = res
    N = r.shape[0]
    r_pad, mask_pad = pad_to_tile(r, tile_size)
    q_pad, _ = pad_to_tile(charges, tile_size)

    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, start_j):
        q_j = jax.lax.dynamic_slice(q_pad, (start_j,), (tile_size,))
        def pair_force(ri, rj, qi, qj):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; d = jnp.sqrt(d2)
            f_mag = coulomb_constant * qi * qj * (jax.scipy.special.erfc(pme_alpha*d)/(d**3) + (2.0*pme_alpha/(jnp.sqrt(jnp.pi)*d**2))*jnp.exp(-(pme_alpha*d)**2))
            if cutoff > 0: f_mag = jnp.where(d < cutoff, f_mag, 0.0)
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

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def chunked_coulomb_energy_nl(
    r: jnp.ndarray,
    charges: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    displacement_fn: Callable,
    pme_alpha: float = 0.34,
    coulomb_constant: float = 332.0636,
    cutoff: float = 9.0,
    tile_size: int = 32
) -> jnp.ndarray:
    """Computes direct-space Coulomb energy over neighbor list using O(N*K) tiling."""
    def f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_idx):
        dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(pos_i, pos_j)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12); ds = dist + 1e-6
        q_j = charges[nb_idx_tile]
        q_ij = charges[:, None] * q_j
        m = mask_i[:, None] & mask_j
        if cutoff > 0: m = m & (ds < cutoff)
        e_pair = coulomb_constant * (q_ij / ds) * jax.scipy.special.erfc(pme_alpha * ds)
        return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))

    atom_mask = jnp.ones(r.shape[0], dtype=bool)
    return tile_reduction_nl(r, neighbor_idx, atom_mask, f_tile, 0.0, tile_size)

def _chunked_coulomb_nl_fwd(r, charges, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size):
    return chunked_coulomb_energy_nl(r, charges, nb_idx, disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size), (r, charges, nb_idx)

def _chunked_coulomb_nl_bwd(disp_fn, pme_alpha, coulomb_constant, cutoff, tile_size, res, g):
    r, charges, nb_idx = res
    def f_tile_grad(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_idx):
        def pair_force(ri, rj, qi, qj):
            dr = disp_fn(ri, rj); d2 = jnp.sum(dr**2) + 1e-12; d = jnp.sqrt(d2)
            f_mag = coulomb_constant * qi * qj * (jax.scipy.special.erfc(pme_alpha*d)/(d**3) + (2.0*pme_alpha/(jnp.sqrt(jnp.pi)*d**2))*jnp.exp(-(pme_alpha*d)**2))
            if cutoff > 0: f_mag = jnp.where(d < cutoff, f_mag, 0.0)
            return f_mag * dr
        q_j = charges[nb_idx_tile]
        forces = jax.vmap(jax.vmap(pair_force, (None, 0, None, 0)), (0, 0, 0, 0))(pos_i, pos_j, charges, q_j)
        m = mask_i[:, None] & mask_j
        forces = jnp.where(m[..., None], forces, 0.0)
        f_on_i = jnp.sum(forces, axis=1)
        f_on_j_contrib = -forces
        nb_flat = nb_idx_tile.reshape(-1)
        f_on_j_flat = f_on_j_contrib.reshape(-1, 3)
        f_total = jnp.zeros_like(pos_i).at[jnp.arange(pos_i.shape[0])].add(f_on_i)
        f_total = f_total.at[nb_flat].add(f_on_j_flat)
        return f_total

    atom_mask = jnp.ones(r.shape[0], dtype=bool)
    forces = tile_reduction_nl(r, nb_idx, atom_mask, f_tile_grad, jnp.zeros_like(r), tile_size)
    return (-g * forces * 0.5, jnp.zeros_like(charges), None)

chunked_coulomb_energy_nl.defvjp(_chunked_coulomb_nl_fwd, _chunked_coulomb_nl_bwd)
