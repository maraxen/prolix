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
    # Derived grid dims set at build time


# ===========================================================================
# B-spline machinery
# ===========================================================================

def _bspline_coeffs(order: int, u: jnp.ndarray) -> jnp.ndarray:
    """Cardinal B-spline values M_n(u) for fractional coordinates u.

    Uses the recurrence relation:
        M_1(u) = 1 if 0 <= u < 1, else 0
        M_n(u) = u/(n-1) * M_{n-1}(u) + (n-u)/(n-1) * M_{n-1}(u-1)

    Args:
        order: B-spline order (typically 4 for cubic)
        u: Fractional coordinate offset in [0, order)

    Returns:
        Array of B-spline values at u, shape same as u.
    """
    # Initialize M_1
    # For order=4, we need M_4 at u, u-1, u-2, u-3
    # Use the recurrence iteratively
    coeffs = jnp.zeros((order,) + u.shape, dtype=u.dtype)

    # M_1(u - k) for k in [0, order)
    w = jnp.ones_like(u)

    # Build via recurrence
    for n in range(2, order + 1):
        # M_n(u) computed from M_{n-1}
        pass

    # Direct formula for order 4 (most common, avoids recurrence overhead):
    if order == 4:
        return _bspline4(u)
    elif order == 6:
        return _bspline6(u)
    else:
        raise ValueError(f"B-spline order {order} not implemented. Use 4 or 6.")


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


def _bspline6(u: jnp.ndarray) -> jnp.ndarray:
    """Order-6 B-spline (for higher accuracy PME)."""
    # Order 6 has 6 pieces. Less common but used in some codes.
    raise NotImplementedError("Order 6 B-spline not yet implemented")


# ===========================================================================
# Grid dimension selection
# ===========================================================================

def _factorizable(n: int, factors: tuple[int, ...] = (2, 3, 5, 7)) -> bool:
    """Check if n factors completely into the given prime set."""
    if n <= 0:
        return False
    for f in factors:
        while n % f == 0:
            n //= f
    return n == 1


def compute_pme_grid_dims(
    box_size: jnp.ndarray,
    grid_spacing: float = 1.0,
    min_dim: int = 8,
) -> tuple[int, int, int]:
    """Compute PME grid dimensions that factorize into {2,3,5,7}."""
    dims = []
    for L in box_size:
        target = max(int(float(L) / grid_spacing + 0.5), min_dim)
        # Search upward for factorizable dimension
        n = target
        while not _factorizable(n):
            n += 1
        dims.append(n)
    return tuple(dims)


# ===========================================================================
# SPME core: charge spreading + influence function + interpolation
# ===========================================================================

def spread_charges(
    positions: jnp.ndarray,  # (N, 3)
    charges: jnp.ndarray,    # (N,)
    atom_mask: jnp.ndarray,  # (N,)
    box_size: jnp.ndarray,   # (3,)
    grid_dims: tuple[int, int, int],
    order: int = 4,
) -> jnp.ndarray:
    """Spread atomic charges onto PME grid using B-spline interpolation."""
    Kx, Ky, Kz = grid_dims
    frac = (positions / box_size) * jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    wx = _bspline4(u[:, 0])
    wy = _bspline4(u[:, 1])
    wz = _bspline4(u[:, 2])

    q = charges * atom_mask.astype(jnp.float32)
    Q = jnp.zeros((Kx, Ky, Kz), dtype=jnp.float32)

    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz
                w = wx[dx] * wy[dy] * wz[dz] * q
                Q = Q.at[ix, iy, iz].add(w)
    return Q


def influence_function(
    grid_dims: tuple[int, int, int],
    box_size: jnp.ndarray,
    alpha: float,
    order: int = 4,
) -> jnp.ndarray:
    Kx, Ky, Kz = grid_dims
    Kz_half = Kz // 2 + 1
    V = jnp.prod(box_size)

    mx = jnp.fft.fftfreq(Kx) * Kx / box_size[0]
    my = jnp.fft.fftfreq(Ky) * Ky / box_size[1]
    mz = jnp.fft.rfftfreq(Kz) * Kz / box_size[2]

    mx2 = mx[:, None, None] ** 2
    my2 = my[None, :, None] ** 2
    mz2 = mz[None, None, :] ** 2
    m_sq = mx2 + my2 + mz2

    gauss = jnp.exp(-jnp.pi**2 * m_sq / alpha**2)
    m_sq_safe = jnp.where(m_sq > 0, m_sq, jnp.float32(1.0))
    denom = jnp.pi * m_sq_safe * V

    bx_freq = jnp.fft.fftfreq(Kx)
    by_freq = jnp.fft.fftfreq(Ky)
    bz_freq = jnp.fft.rfftfreq(Kz)
    bx = _bspline_modulation(bx_freq, Kx, order)
    by = _bspline_modulation(by_freq, Ky, order)
    bz = _bspline_modulation(bz_freq, Kz, order)

    b_sq = (bx[:, None, None] * by[None, :, None] * bz[None, None, :]) ** 2
    b_sq_safe = jnp.maximum(b_sq, jnp.float32(1e-10))

    G = gauss / (denom * b_sq_safe)
    G = G.at[0, 0, 0].set(0.0)
    return G


def _bspline_modulation(freq_frac: jnp.ndarray, K: int, order: int) -> jnp.ndarray:
    if order == 4:
        m_vals = jnp.array([1.0/6.0, 4.0/6.0, 1.0/6.0, 0.0])
    else:
        raise NotImplementedError(f"B-spline order {order} modulation not implemented")

    k_indices = jnp.arange(order, dtype=jnp.float32)
    phases = jnp.exp(2j * jnp.pi * freq_frac[:, None] * k_indices[None, :])
    b_complex = jnp.sum(m_vals[None, :] * phases, axis=-1)
    return jnp.maximum(jnp.abs(b_complex), jnp.float32(1e-6))


def spme_reciprocal_energy(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    alpha: float = 0.34,
    order: int = 4,
) -> jnp.ndarray:
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta_hat = Q_hat * G
    theta = jnp.fft.irfftn(theta_hat, s=grid_dims)
    N_grid = grid_dims[0] * grid_dims[1] * grid_dims[2]
    energy = 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta) * N_grid
    return energy


def spme_self_energy(
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    alpha: float = 0.34,
) -> jnp.ndarray:
    q_masked = charges * atom_mask.astype(jnp.float32)
    q_sq_sum = jnp.sum(q_masked ** 2)
    val = -alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * q_sq_sum
    # print(f"DEBUG PME SELF: {val}")
    return val


def spme_background_energy(
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    alpha: float,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    q_masked = charges * atom_mask.astype(jnp.float32)
    Q = jnp.sum(q_masked)
    V = jnp.prod(box_size)
    return -jnp.pi * Q**2 / (2.0 * alpha**2 * V) * COULOMB_CONSTANT


def excluded_pme_correction(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    idx_12_13: jnp.ndarray,
    idx_14: jnp.ndarray,
    scale_14: float,
    alpha: float,
    displacement_fn,
    skip_self: bool = True,
) -> jnp.ndarray:
    """Subtract reciprocal contribution of excluded and scaled pairs.

    Reciprocal part contribution is erf(alpha * r) / r for all pairs (i, j).
    We must subtract this for pairs that are either fully excluded (1-2, 1-3)
    or scaled (1-4). 
    
    Self-interactions (i, i) are already handled by spme_self_energy.
    If skip_self=True, we avoid double-subtracting them here.
    """
    def _pair_energy(idx, scale):
        if idx.shape[0] == 0:
            return 0.0
        
        p1 = idx[:, 0]
        p2 = idx[:, 1]
        
        # Only correct pairs of DISTINCT atoms by default
        if skip_self:
            mask_active = p1 != p2
        else:
            mask_active = jnp.ones(p1.shape, dtype=bool)
        
        r_i = positions[p1]
        r_j = positions[p2]
        dr = jax.vmap(displacement_fn)(r_i, r_j)
        dist = jnp.linalg.norm(dr, axis=-1)
        
        # Safe distance for distinct pairs
        dist_safe = jnp.where(dist > 1e-6, dist, 1.0)
        
        q_i = charges[p1]
        q_j = charges[p2]
        
        e_pair = jax.scipy.special.erf(alpha * dist_safe) / dist_safe
        
        # Zero out corrections for inactive pairs
        e_pair = jnp.where(mask_active, e_pair, 0.0)
        
        return COULOMB_CONSTANT * jnp.sum(q_i * q_j * e_pair * (1.0 - scale))

    e_1213 = _pair_energy(idx_12_13, 0.0)
    e_14 = _pair_energy(idx_14, scale_14)
    return - (e_1213 + e_14)


# ===========================================================================
# Custom VJP wrapper for analytical forces
# ===========================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def spme_energy_with_forces(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    alpha: float,
    order: int,
) -> jnp.ndarray:
    e_recip = spme_reciprocal_energy(
        positions, charges, atom_mask, box_size, grid_dims, alpha, order
    )
    e_self = spme_self_energy(charges, atom_mask, alpha)
    return e_recip + e_self


def _spme_fwd(positions, charges, atom_mask, box_size, grid_dims, alpha, order):
    e_recip = spme_reciprocal_energy(
        positions, charges, atom_mask, box_size, grid_dims, alpha, order
    )
    e_self = spme_self_energy(charges, atom_mask, alpha)
    energy = e_recip + e_self

    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta_hat = Q_hat * G
    theta = jnp.fft.irfftn(theta_hat, s=grid_dims)
    N_grid = grid_dims[0] * grid_dims[1] * grid_dims[2]
    theta = theta * N_grid
    residuals = (positions, charges, atom_mask, box_size, theta)
    return energy, residuals


def _spme_bwd(grid_dims, alpha, order, residuals, g):
    positions, charges, atom_mask, box_size, theta = residuals
    Kx, Ky, Kz = grid_dims
    K_arr = jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    frac = positions / box_size * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    wx = _bspline4(u[:, 0])
    wy = _bspline4(u[:, 1])
    wz = _bspline4(u[:, 2])
    dwx = _bspline4_deriv(u[:, 0])
    dwy = _bspline4_deriv(u[:, 1])
    dwz = _bspline4_deriv(u[:, 2])

    q_masked = charges * atom_mask.astype(jnp.float32)
    N = positions.shape[0]
    forces = jnp.zeros((N, 3), dtype=jnp.float32)

    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz
                theta_val = theta[ix, iy, iz]
                fx = q_masked * (K_arr[0] / box_size[0]) * dwx[dx] * wy[dy] * wz[dz] * theta_val
                fy = q_masked * (K_arr[1] / box_size[1]) * wx[dx] * dwy[dy] * wz[dz] * theta_val
                fz = q_masked * (K_arr[2] / box_size[2]) * wx[dx] * wy[dy] * dwz[dz] * theta_val
                forces = forces + jnp.stack([fx, fy, fz], axis=-1)

    grad_positions = COULOMB_CONSTANT * forces * g
    return grad_positions, jnp.zeros_like(charges), jnp.zeros_like(atom_mask, dtype=jnp.float32), jnp.zeros_like(box_size, dtype=jnp.float32)


spme_energy_with_forces.defvjp(_spme_fwd, _spme_bwd)


# ===========================================================================
# Public API
# ===========================================================================

def make_spme_energy_fn(
    box_size: jnp.ndarray,
    alpha: float = 0.34,
    grid_spacing: float = 1.0,
    order: int = 4,
):
    grid_dims = compute_pme_grid_dims(box_size, grid_spacing)
    def energy_fn(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray) -> jnp.ndarray:
        return spme_energy_with_forces(positions, charges, atom_mask, box_size, grid_dims, alpha, order)
    return energy_fn


def make_pme_energy_fn(
    charges: jnp.ndarray,
    box_size: jnp.ndarray,
    *,
    grid_points: int = 64,
    alpha: float = 0.34,
    order: int = 4,
):
    charges = jnp.asarray(charges)
    box_size = jnp.asarray(box_size)
    n = int(charges.shape[0])
    mask = jnp.ones((n,), dtype=bool)
    mean_l = jnp.mean(box_size.astype(jnp.float64))
    grid_spacing = float(mean_l) / float(max(grid_points, 1))
    inner = make_spme_energy_fn(box_size, alpha=alpha, grid_spacing=grid_spacing, order=order)

    def energy_fn(positions: jnp.ndarray) -> jnp.ndarray:
        e = inner(positions, charges, mask)
        e = e + spme_background_energy(charges, mask, alpha, box_size)
        return e
    return energy_fn
