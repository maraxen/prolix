"""Smooth Particle Mesh Ewald (SPME) electrostatics.

Custom implementation using jnp.fft.rfftn/irfftn for ~50% FFT speedup
over the full complex FFT. Analytical forces via jax.custom_vjp to avoid
checkpointing massive 3D grids through jax.grad.

Key design decisions (see implementation plan Phase 2):
- R2C FFTs (rfftn) halve the Z-dimension → ~50% memory + compute savings
- Grid dims factorize into {2,3,5,7} to prevent Bluestein fallback
- jax.custom_vjp wraps forward energy + analytical force for gradient
- B-spline order 4 (cubic) matches AMBER/GROMACS defaults

References:
- Essmann et al., J. Chem. Phys. 103(19), 8577-8593, 1995
- Darden, York, Pedersen, J. Chem. Phys. 98(12), 10089-10092, 1993
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from proxide.physics.constants import COULOMB_CONSTANT

# ===========================================================================
# Configuration
# ===========================================================================

class SPMEParams(NamedTuple):
    """Smooth PME configuration parameters."""
    alpha: float = 0.34        # Ewald splitting parameter (1/Å)
    grid_spacing: float = 1.0  # Target grid spacing (Å), actual snapped
    order: int = 4             # B-spline interpolation order


# ===========================================================================
# B-spline machinery
# ===========================================================================

def _bspline4(u: jnp.ndarray) -> jnp.ndarray:
    """Order-4 (cubic) B-spline values and derivatives."""
    w0 = (1.0 - u) ** 3 / 6.0
    w1 = (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0
    w2 = (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0
    w3 = u**3 / 6.0
    return jnp.stack([w0, w1, w2, w3], axis=0)

def _bspline4_deriv(u: jnp.ndarray) -> jnp.ndarray:
    """Order-4 B-spline derivative dM_4/du."""
    dw0 = -0.5 * (1.0 - u) ** 2
    dw1 = (9.0 * u**2 - 12.0 * u) / 6.0
    dw2 = (-9.0 * u**2 + 6.0 * u + 3.0) / 6.0
    dw3 = 0.5 * u**2
    return jnp.stack([dw0, dw1, dw2, dw3], axis=0)


# ===========================================================================
# Grid dimension selection
# ===========================================================================

def _factorizable(n: int, factors: tuple[int, ...] = (2, 3, 5, 7)) -> bool:
    if n <= 0: return False
    for f in factors:
        while n % f == 0: n //= f
    return n == 1

def compute_pme_grid_dims(box_size: jnp.ndarray, grid_spacing: float = 1.0, min_dim: int = 8) -> tuple[int, int, int]:
    dims = []
    for L in box_size:
        target = max(int(float(L) / grid_spacing + 0.5), min_dim)
        n = target
        while not _factorizable(n): n += 1
        dims.append(n)
    return tuple(dims)


# ===========================================================================
# SPME core
# ===========================================================================

def spread_charges(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray, box_size: jnp.ndarray, grid_dims: tuple[int, int, int], order: int = 4) -> jnp.ndarray:
    Kx, Ky, Kz = grid_dims
    frac = (positions / box_size) * jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    wx = _bspline4(u[:, 0]); wy = _bspline4(u[:, 1]); wz = _bspline4(u[:, 2])
    Q = jnp.zeros((Kx, Ky, Kz), dtype=jnp.float32)
    q = (charges * atom_mask.astype(jnp.float32)).astype(Q.dtype)

    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz
                w = (wx[dx] * wy[dy] * wz[dz] * q).astype(Q.dtype)
                Q = Q.at[ix, iy, iz].add(w)
    return Q


def influence_function(grid_dims: tuple[int, int, int], box_size: jnp.ndarray, alpha: float, order: int = 4) -> jnp.ndarray:
    Kx, Ky, Kz = grid_dims
    V = jnp.prod(box_size)

    mx = jnp.fft.fftfreq(Kx) * Kx / box_size[0]
    my = jnp.fft.fftfreq(Ky) * Ky / box_size[1]
    mz = jnp.fft.rfftfreq(Kz) * Kz / box_size[2]
    m_sq = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2

    gauss = jnp.exp(-jnp.pi**2 * m_sq / alpha**2)
    m_sq_safe = jnp.where(m_sq > 0, m_sq, jnp.float32(1.0))
    denom = jnp.pi * m_sq_safe * V

    bx = _bspline_modulation(jnp.fft.fftfreq(Kx), Kx, order)
    by = _bspline_modulation(jnp.fft.fftfreq(Ky), Ky, order)
    bz = _bspline_modulation(jnp.fft.rfftfreq(Kz), Kz, order)
    b_sq = (bx[:, None, None] * by[None, :, None] * bz[None, None, :]) ** 2
    b_sq_safe = jnp.maximum(b_sq, jnp.float32(1e-10))

    G = gauss / (denom * b_sq_safe)
    G = G.at[0, 0, 0].set(0.0)
    return G


def _bspline_modulation(freq_frac: jnp.ndarray, K: int, order: int) -> jnp.ndarray:
    if order == 4: m_vals = jnp.array([1.0/6.0, 4.0/6.0, 1.0/6.0, 0.0])
    else: raise NotImplementedError()
    k_indices = jnp.arange(order, dtype=jnp.float32)
    phases = jnp.exp(2j * jnp.pi * freq_frac[:, None] * k_indices[None, :])
    return jnp.maximum(jnp.abs(jnp.sum(m_vals[None, :] * phases, axis=-1)), jnp.float32(1e-6))


def spme_reciprocal_energy(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray, box_size: jnp.ndarray, grid_dims: tuple[int, int, int], alpha: float = 0.34, order: int = 4) -> jnp.ndarray:
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims)
    
    # Normalization Fix: JAX irfftn divides by N_grid.
    # Parseval's theorem for discrete sum requires multiplying by N_grid 
    # if we want the sum over reciprocal space S(m)^2 G(m) to match.
    N_grid = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta) * N_grid


def spme_self_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float = 0.34) -> jnp.ndarray:
    q_masked = charges * atom_mask.astype(jnp.float32)
    return -alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * jnp.sum(q_masked ** 2)


def spme_background_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float, box_size: jnp.ndarray) -> jnp.ndarray:
    Q = jnp.sum(charges * atom_mask.astype(jnp.float32))
    V = jnp.prod(box_size)
    return -jnp.pi * Q**2 / (2.0 * alpha**2 * V) * COULOMB_CONSTANT


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def spme_energy_with_forces(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray, box_size: jnp.ndarray, grid_dims: tuple[int, int, int], alpha: float, order: int) -> jnp.ndarray:
    return spme_reciprocal_energy(positions, charges, atom_mask, box_size, grid_dims, alpha, order) + spme_self_energy(charges, atom_mask, alpha)


def _spme_fwd(positions, charges, atom_mask, box_size, grid_dims, alpha, order):
    e_recip = spme_reciprocal_energy(positions, charges, atom_mask, box_size, grid_dims, alpha, order)
    e_self = spme_self_energy(charges, atom_mask, alpha)
    
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims)
    # Store theta with the N_grid factor for force calculation consistency
    N_grid = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return e_recip + e_self, (positions, charges, atom_mask, box_size, theta * N_grid)


def _spme_bwd(grid_dims, alpha, order, residuals, g):
    positions, charges, atom_mask, box_size, theta_norm = residuals
    Kx, Ky, Kz = grid_dims
    K_arr = jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    frac = positions / box_size * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    wx = _bspline4(u[:, 0]); wy = _bspline4(u[:, 1]); wz = _bspline4(u[:, 2])
    dwx = _bspline4_deriv(u[:, 0]); dwy = _bspline4_deriv(u[:, 1]); dwz = _bspline4_deriv(u[:, 2])

    q_masked = charges * atom_mask.astype(jnp.float32)
    forces = jnp.zeros((positions.shape[0], 3), dtype=jnp.float32)

    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz
                t_val = theta_norm[ix, iy, iz]
                fx = q_masked * (K_arr[0] / box_size[0]) * dwx[dx] * wy[dy] * wz[dz] * t_val
                fy = q_masked * (K_arr[1] / box_size[1]) * wx[dx] * dwy[dy] * wz[dz] * t_val
                fz = q_masked * (K_arr[2] / box_size[2]) * wx[dx] * wy[dy] * dwz[dz] * t_val
                forces = forces + jnp.stack([fx, fy, fz], axis=-1)

    # Return gradient w.r.t positions = dE/dr.
    # E = 0.5 * sum(q_i * phi_i). dE/dr_i = q_i * dphi/dr_i.
    # forces variable computes q_i * dphi/dr_i.
    return COULOMB_CONSTANT * forces * g, jnp.zeros_like(charges), jnp.zeros_like(atom_mask, dtype=jnp.float32), jnp.zeros_like(box_size, dtype=jnp.float32)

spme_energy_with_forces.defvjp(_spme_fwd, _spme_bwd)

def make_spme_energy_fn(box_size: jnp.ndarray, alpha: float = 0.34, grid_spacing: float = 1.0, order: int = 4):
    grid_dims = compute_pme_grid_dims(box_size, grid_spacing)
    return lambda positions, charges, atom_mask: spme_energy_with_forces(positions, charges, atom_mask, box_size, grid_dims, alpha, order)

def make_pme_energy_fn(charges: jnp.ndarray, box_size: jnp.ndarray, *, grid_points: int = 64, alpha: float = 0.34, order: int = 4):
    charges, box_size = jnp.asarray(charges), jnp.asarray(box_size)
    mask = jnp.ones(charges.shape[0], dtype=bool)
    grid_spacing = float(jnp.mean(box_size.astype(jnp.float64))) / float(max(grid_points, 1))
    inner = make_spme_energy_fn(box_size, alpha=alpha, grid_spacing=grid_spacing, order=order)
    return lambda positions: inner(positions, charges, mask) + spme_background_energy(charges, mask, alpha, box_size)
