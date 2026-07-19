"""PME (Particle Mesh Ewald) implementation in JAX."""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# kcal*A/(mol*e^2)
COULOMB_CONSTANT = 332.0637

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
    """Evaluate 4th order B-spline basis at fractional offsets u in [0, 1)."""
    u2 = u * u
    u3 = u2 * u
    
    w0 = (1.0 - 3.0*u + 3.0*u2 - u3) / 6.0
    w1 = (4.0 - 6.0*u2 + 3.0*u3) / 6.0
    w2 = (1.0 + 3.0*u + 3.0*u2 - 3.0*u3) / 6.0
    w3 = u3 / 6.0
    return jnp.stack([w0, w1, w2, w3])


def _bspline4_deriv(u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate derivative of 4th order B-spline basis."""
    u2 = u * u
    
    dw0 = (-3.0 + 6.0*u - 3.0*u2) / 6.0
    dw1 = (-12.0*u + 9.0*u2) / 6.0
    dw2 = (3.0 + 6.0*u - 9.0*u2) / 6.0
    dw3 = (3.0*u2) / 6.0
    return jnp.stack([dw0, dw1, dw2, dw3])


def _factorizable(n: int) -> bool:
    """True if n is a product of small primes {2, 3, 5, 7} for efficient FFT."""
    if n <= 1: return True
    for p in [2, 3, 5, 7]:
        while n % p == 0:
            n //= p
    return n == 1


def compute_pme_grid_dims(
    box_size: jnp.ndarray,
    grid_spacing: float = 1.0,
    min_dim: int = 8
) -> tuple[int, int, int]:
    """Determine FFT grid dimensions based on box size and target spacing."""
    # grid_dims must be static Python ints (used as literal array shapes
    # downstream), so this must never touch jnp: float()/int() on a jax.Array
    # raises ConcretizationTypeError whenever any JAX trace is active,
    # regardless of whether the array is actually concrete (debt 770).
    box_size_np = np.asarray(box_size)
    dims = np.maximum(np.ceil(box_size_np / grid_spacing).astype(np.int32), min_dim)
    # Ensure even/optimal FFT dimensions? For now just cast.
    return (int(dims[0]), int(dims[1]), int(dims[2]))


# ===========================================================================
# SPME Kernels
# ===========================================================================

def spread_charges(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray, box_size: jnp.ndarray, grid_dims: tuple[int, int, int], order: int = 4) -> jnp.ndarray:
    """Spread charges onto FFT grid using B-spline interpolation."""
    Kx, Ky, Kz = grid_dims
    K_arr = jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    
    # Fractional coordinates in [0, K)
    frac = (positions / box_size) * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    # w shape: (order, N_atoms)
    wx = _bspline4(u[:, 0]); wy = _bspline4(u[:, 1]); wz = _bspline4(u[:, 2])
    
    Q = jnp.zeros(grid_dims, dtype=jnp.float32)
    q = charges * atom_mask.astype(jnp.float32)

    with jax.named_scope("pme_charge_spread"):
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
    with jax.named_scope("pme_greens_setup_standalone"):
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
    else: raise NotImplementedError
    k_indices = jnp.arange(order, dtype=jnp.float32)
    phases = jnp.exp(2j * jnp.pi * freq_frac[:, None] * k_indices[None, :])
    return jnp.maximum(jnp.abs(jnp.sum(m_vals[None, :] * phases, axis=-1)), jnp.float32(1e-6))


def spme_reciprocal_energy(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray, box_size: jnp.ndarray, grid_dims: tuple[int, int, int], alpha: float = 0.34, order: int = 4) -> jnp.ndarray:
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims)
    N_grid = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
    return 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta) * N_grid


def spme_self_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float = 0.34) -> jnp.ndarray:
    q_masked = charges * atom_mask.astype(jnp.float32)
    return -alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * jnp.sum(q_masked ** 2)


def spme_background_energy(charges: jnp.ndarray, atom_mask: jnp.ndarray, alpha: float, box_size: jnp.ndarray) -> jnp.ndarray:
    Q = jnp.sum(charges * atom_mask.astype(jnp.float32))
    V = jnp.prod(box_size)
    return -jnp.pi * Q**2 / (2.0 * alpha**2 * V) * COULOMB_CONSTANT


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 6))
def spme_energy_with_forces(positions: jnp.ndarray, charges: jnp.ndarray, atom_mask: jnp.ndarray, box_size: jnp.ndarray, grid_dims: tuple[int, int, int], alpha: float, order: int) -> jnp.ndarray:
    return spme_reciprocal_energy(positions, charges, atom_mask, box_size, grid_dims, alpha, order) + spme_self_energy(charges, atom_mask, alpha)


def _spme_fwd(positions, charges, atom_mask, box_size, grid_dims, alpha, order):
    Kx, Ky, Kz = grid_dims
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    with jax.named_scope("pme_fft_forward"):
        Q_hat = jnp.fft.rfftn(Q)

    # Precompute G and m_sq for both energy and alpha-gradient
    with jax.named_scope("pme_greens_setup"):
        mx = jnp.fft.fftfreq(Kx) * Kx / box_size[0]
        my = jnp.fft.fftfreq(Ky) * Ky / box_size[1]
        mz = jnp.fft.rfftfreq(Kz) * Kz / box_size[2]
        m_sq = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2

        V = jnp.prod(box_size)
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

    Q_hat_sq = jnp.abs(Q_hat) ** 2

    # irfftn normalization
    with jax.named_scope("pme_fft_inverse"):
        theta = jnp.fft.irfftn(Q_hat * G, s=grid_dims)
    N_grid = float(Kx * Ky * Kz)
    theta_norm = theta * N_grid
    
    e_recip = 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta_norm)
    e_self = spme_self_energy(charges, atom_mask, alpha)
    
    return e_recip + e_self, (positions, charges, atom_mask, box_size, theta_norm, m_sq, Q_hat_sq, G, alpha, mx, my, mz, Kx, Ky, Kz)


def _spme_bwd(grid_dims, order, residuals, g):
    positions, charges, atom_mask, box_size, theta_norm, m_sq, Q_hat_sq, G, alpha, mx, my, mz, Kx, Ky, Kz = residuals
    K_arr = jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    frac = positions / box_size * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    wx = _bspline4(u[:, 0]); wy = _bspline4(u[:, 1]); wz = _bspline4(u[:, 2])
    dwx = _bspline4_deriv(u[:, 0]); dwy = _bspline4_deriv(u[:, 1]); dwz = _bspline4_deriv(u[:, 2])

    q_masked = charges * atom_mask.astype(jnp.float32)
    forces = jnp.zeros((positions.shape[0], 3), dtype=jnp.float32)
    potentials = jnp.zeros(positions.shape[0], dtype=jnp.float32)

    with jax.named_scope("pme_bwd_gather"):
        for dx in range(order):
            for dy in range(order):
                for dz in range(order):
                    ix = (grid_idx[:, 0] + dx - 1) % Kx
                    iy = (grid_idx[:, 1] + dy - 1) % Ky
                    iz = (grid_idx[:, 2] + dz - 1) % Kz
                    t_val = theta_norm[ix, iy, iz]

                    # Potential for charge gradient
                    w = wx[dx] * wy[dy] * wz[dz]
                    potentials = potentials + w * t_val

                    # Forces for position gradient
                    fx = q_masked * (K_arr[0] / box_size[0]) * dwx[dx] * wy[dy] * wz[dz] * t_val
                    fy = q_masked * (K_arr[1] / box_size[1]) * wx[dx] * dwy[dy] * wz[dz] * t_val
                    fz = q_masked * (K_arr[2] / box_size[2]) * wx[dx] * wy[dy] * dwz[dz] * t_val
                    forces = forces + jnp.stack([fx, fy, fz], axis=-1)

    # 1. Position gradient
    dE_dpos = COULOMB_CONSTANT * forces * g

    # 2. Charge gradient
    dE_dq = COULOMB_CONSTANT * potentials
    dE_dq += -2.0 * alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * charges
    dE_dq = dE_dq * atom_mask.astype(jnp.float32) * g

    # 3. Alpha gradient
    dG_dalpha = G * (2.0 * jnp.pi**2 * m_sq / alpha**3)
    weights = jnp.ones_like(G) * 2.0
    weights = weights.at[:, :, 0].set(1.0)
    if Kz % 2 == 0:
        weights = weights.at[:, :, -1].set(1.0)
    
    dE_dalpha_recip = 0.5 * COULOMB_CONSTANT * jnp.sum(Q_hat_sq * dG_dalpha * weights)
    dE_dalpha_self = -1.0 / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * jnp.sum(q_masked**2)
    dE_dalpha = (dE_dalpha_recip + dE_dalpha_self) * g

    # 4. Box size gradient
    virial_term = -1.0 / box_size * jnp.sum(positions * dE_dpos, axis=0)
    Kz_rfft = Kz // 2 + 1
    m_x_sq = mx[:, None, None]**2 * jnp.ones((1, Ky, Kz_rfft), dtype=jnp.float32)
    m_y_sq = my[None, :, None]**2 * jnp.ones((Kx, 1, Kz_rfft), dtype=jnp.float32)
    m_z_sq = mz[None, None, :]**2 * jnp.ones((Kx, Ky, 1), dtype=jnp.float32)
    m_i_sq = jnp.stack([m_x_sq, m_y_sq, m_z_sq], axis=-1)
    
    term_alpha = 2.0 * jnp.pi**2 * m_i_sq / alpha**2
    term_msq = 2.0 * m_i_sq / jnp.where(m_sq[:, :, :, None] > 0, m_sq[:, :, :, None], 1.0)
    dG_dL = G[:, :, :, None] * (term_alpha + term_msq - 1.0) / box_size
    dE_dL_recip = 0.5 * COULOMB_CONSTANT * jnp.sum(Q_hat_sq[:, :, :, None] * dG_dL * weights[:, :, :, None], axis=(0, 1, 2))
    dE_dL = (virial_term + dE_dL_recip) * g

    return dE_dpos, dE_dq, jnp.zeros_like(atom_mask, dtype=jnp.float32), dE_dL, dE_dalpha

spme_energy_with_forces.defvjp(_spme_fwd, _spme_bwd)

def make_spme_energy_fn(box_size: jnp.ndarray, alpha: float = 0.34, grid_spacing: float = 1.0, order: int = 4):
    grid_dims = compute_pme_grid_dims(box_size, grid_spacing)
    return lambda positions, charges, atom_mask: spme_energy_with_forces(positions, charges, atom_mask, box_size, grid_dims, alpha, order)

def make_pme_energy_fn(charges: jnp.ndarray, box_size: jnp.ndarray, *, grid_points: int = 64, alpha: float = 0.34, order: int = 4):
    charges, box_size = jnp.asarray(charges), jnp.asarray(box_size)
    mask = jnp.ones(charges.shape[0], dtype=bool)
    # numpy, not jnp: see compute_pme_grid_dims above (debt 770).
    grid_spacing = float(np.mean(np.asarray(box_size))) / float(max(grid_points, 1))
    inner = make_spme_energy_fn(box_size, alpha=alpha, grid_spacing=grid_spacing, order=order)
    return lambda positions: inner(positions, charges, mask) + spme_background_energy(charges, mask, alpha, box_size)
