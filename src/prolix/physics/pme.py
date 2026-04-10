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
    """Order-4 (cubic) B-spline values and derivatives.

    Returns shape (4,) + u.shape with M_4(u), M_4(u-1), M_4(u-2), M_4(u-3).

    For fractional coord f, the 4 grid points affected are:
        grid[floor(f)-1], grid[floor(f)], grid[floor(f)+1], grid[floor(f)+2]
    and u = f - floor(f) + 1 (shifted so u ∈ [1, 2))

    Standard cubic B-spline M_4(t) for t ∈ [0, 4]:
        M_4(t) = t³/6                           for t ∈ [0,1)
        M_4(t) = (-3t³ + 12t² - 12t + 4)/6      for t ∈ [1,2)
        M_4(t) = (3t³ - 24t² + 60t - 44)/6       for t ∈ [2,3)
        M_4(t) = (4-t)³/6                        for t ∈ [3,4)

    Args:
        u: Fractional part, range [0, 1). Shape (...,)

    Returns:
        (4, ...) array of B-spline weights.
    """
    # The 4 evaluation points for grid indices [k-1, k, k+1, k+2]
    # where k = floor(frac_coord)
    t0 = u + 1.0  # ∈ [1, 2)  → M_4 piece 2
    t1 = u        # ∈ [0, 1)  → M_4 piece 1
    t2 = 1.0 - u  # ∈ (0, 1]  → M_4 piece 1 (reflected)
    t3 = 2.0 - u  # ∈ (1, 2]  → M_4 piece 2 (reflected)

    # Wait — let me use the standard formulation more carefully.
    # For a fractional coordinate w ∈ [0,1), the B-spline weights on
    # the 4 surrounding grid points are:
    w0 = (1.0 - u) ** 3 / 6.0
    w1 = (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0
    w2 = (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0
    w3 = u**3 / 6.0

    return jnp.stack([w0, w1, w2, w3], axis=0)


def _bspline4_deriv(u: jnp.ndarray) -> jnp.ndarray:
    """Order-4 B-spline derivative dM_4/du.

    Args:
        u: Fractional part, range [0, 1). Shape (...,)

    Returns:
        (4, ...) array of B-spline derivative weights.
    """
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
    """Compute PME grid dimensions that factorize into {2,3,5,7}.

    This prevents XLA from falling back to Bluestein's algorithm
    for the FFT, which would be much slower.

    Args:
        box_size: (3,) box dimensions in Å
        grid_spacing: Target grid spacing (Å). Smaller = more accurate.
        min_dim: Minimum grid dimension per axis.

    Returns:
        (Kx, Ky, Kz) grid dimensions.
    """
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
    """Spread atomic charges onto PME grid using B-spline interpolation.

    Args:
        positions: Atomic positions in Å
        charges: Atomic partial charges
        atom_mask: Boolean mask for real atoms
        box_size: Simulation box dimensions
        grid_dims: (Kx, Ky, Kz) grid dimensions
        order: B-spline order (4 = cubic)

    Returns:
        (Kx, Ky, Kz) charge grid Q.
    """
    Kx, Ky, Kz = grid_dims

    # Fractional coordinates [0, K)
    frac = positions / box_size * jnp.array([Kx, Ky, Kz], dtype=jnp.float32)

    # Grid indices and fractional offsets
    grid_idx = jnp.floor(frac).astype(jnp.int32)  # (N, 3)
    u = frac - grid_idx.astype(jnp.float32)         # (N, 3) in [0,1)

    # B-spline weights: (4, N) for each dimension
    wx = _bspline4(u[:, 0])  # (4, N)
    wy = _bspline4(u[:, 1])  # (4, N)
    wz = _bspline4(u[:, 2])  # (4, N)

    # Masked charges
    q = charges * atom_mask.astype(jnp.float32)  # (N,)

    # Accumulate onto grid using scatter_add
    # For each atom, spread to 4³ = 64 grid points
    Q = jnp.zeros((Kx, Ky, Kz), dtype=jnp.float32)

    # Vectorized spreading over the 4x4x4 stencil
    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                # Grid positions with PBC wrapping
                ix = (grid_idx[:, 0] + dx - 1) % Kx  # (N,)
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz

                # Weight = product of 1D B-spline weights
                w = wx[dx] * wy[dy] * wz[dz] * q  # (N,)

                # Scatter add
                Q = Q.at[ix, iy, iz].add(w)

    return Q


def influence_function(
    grid_dims: tuple[int, int, int],
    box_size: jnp.ndarray,
    alpha: float,
    order: int = 4,
) -> jnp.ndarray:
    """Compute the influence function (Green's function) on the reciprocal grid.

    G(m) = exp(-π²|m|²/α²) / (π|m|² · V) × B(m)⁻²

    where B(m) is the B-spline modulation factor.

    Uses the R2C FFT convention: only half of the Z-axis is stored.

    Args:
        grid_dims: (Kx, Ky, Kz) grid dimensions
        box_size: (3,) box dimensions
        alpha: Ewald splitting parameter
        order: B-spline order

    Returns:
        (Kx, Ky, Kz//2+1) influence function array.
    """
    Kx, Ky, Kz = grid_dims
    Kz_half = Kz // 2 + 1
    V = jnp.prod(box_size)

    # Reciprocal lattice vectors: m_i / L_i
    mx = jnp.fft.fftfreq(Kx) * Kx / box_size[0]
    my = jnp.fft.fftfreq(Ky) * Ky / box_size[1]
    mz = jnp.fft.rfftfreq(Kz) * Kz / box_size[2]  # Only half-Z

    # |m|² on 3D grid
    mx2 = mx[:, None, None] ** 2
    my2 = my[None, :, None] ** 2
    mz2 = mz[None, None, :] ** 2
    m_sq = mx2 + my2 + mz2  # (Kx, Ky, Kz_half)

    # Gaussian: exp(-π²|m|²/α²)
    gauss = jnp.exp(-jnp.pi**2 * m_sq / alpha**2)

    # Denominator: π|m|²V
    # Avoid division by zero at m=(0,0,0)
    m_sq_safe = jnp.where(m_sq > 0, m_sq, jnp.float32(1.0))
    denom = jnp.pi * m_sq_safe * V

    # B-spline modulation factor B(m)
    # _bspline_modulation expects fractional frequencies ν/K = fftfreq(K)
    bx_freq = jnp.fft.fftfreq(Kx)      # fractional freq for x
    by_freq = jnp.fft.fftfreq(Ky)      # fractional freq for y
    bz_freq = jnp.fft.rfftfreq(Kz)     # fractional freq for z (half)
    bx = _bspline_modulation(bx_freq, Kx, order)
    by = _bspline_modulation(by_freq, Ky, order)
    bz = _bspline_modulation(bz_freq, Kz, order)

    b_sq = (bx[:, None, None] * by[None, :, None] * bz[None, None, :]) ** 2
    b_sq_safe = jnp.maximum(b_sq, jnp.float32(1e-10))

    G = gauss / (denom * b_sq_safe)

    # Zero out the DC component (m=0)
    G = G.at[0, 0, 0].set(0.0)

    return G


def _bspline_modulation(freq_frac: jnp.ndarray, K: int, order: int) -> jnp.ndarray:
    """B-spline structure factor |b(ν/K)|.

    b(ν/K) = ∑_{k=0}^{order-1} M_order(k+1) * exp(2πi·k·ν/K)

    Args:
        freq_frac: Fractional frequencies ν/K, as returned by jnp.fft.fftfreq(K)
                   or jnp.fft.rfftfreq(K). Shape (K,) or (K//2+1,).
        K: Grid dimension (not used in computation, kept for API clarity)
        order: B-spline order

    Returns:
        |b(ν/K)| array, same shape as freq_frac. All values >= some positive min.
    """
    if order == 4:
        # M_4 at integer knots k+1 for k=0..3:
        # M_4(1) = 1/6, M_4(2) = 4/6, M_4(3) = 1/6, M_4(4) = 0
        m_vals = jnp.array([1.0/6.0, 4.0/6.0, 1.0/6.0, 0.0])
    else:
        raise NotImplementedError(f"B-spline order {order} modulation not implemented")

    # Compute b(f) = ∑_{k=0}^{order-1} M(k+1) * exp(2πi·k·f)
    # where f = freq_frac (= ν/K)
    k_indices = jnp.arange(order, dtype=jnp.float32)  # [0, 1, 2, 3]

    # Phase factors: exp(2πi·k·f) for each (f, k) pair
    # freq_frac shape: (F,), k_indices shape: (4,)
    phases = jnp.exp(
        2j * jnp.pi * freq_frac[:, None] * k_indices[None, :]
    )  # (F, 4) complex

    # b(f) = sum_k M(k+1) * exp(2πi·k·f)
    b_complex = jnp.sum(m_vals[None, :] * phases, axis=-1)  # (F,) complex

    # Return |b(f)|, clamped to avoid division by zero
    return jnp.maximum(jnp.abs(b_complex), jnp.float32(1e-6))


def spme_reciprocal_energy(
    positions: jnp.ndarray,   # (N, 3)
    charges: jnp.ndarray,     # (N,)
    atom_mask: jnp.ndarray,   # (N,) bool
    box_size: jnp.ndarray,    # (3,)
    grid_dims: tuple[int, int, int],
    alpha: float = 0.34,
    order: int = 4,
) -> jnp.ndarray:
    """Compute SPME reciprocal space energy.

    1. Spread charges onto 3D grid via B-spline interpolation
    2. FFT(Q) → Q̂ (using rfftn for R2C efficiency)
    3. Multiply by influence function: Q̂·G
    4. IFFT back: θ = IFFT(Q̂·G)
    5. Energy = 0.5 * sum(Q·θ) — convolution theorem

    Args:
        positions: (N, 3) atomic positions
        charges: (N,) partial charges
        atom_mask: (N,) boolean mask for real atoms
        box_size: (3,) simulation box
        grid_dims: (Kx, Ky, Kz) PME grid dimensions
        alpha: Ewald splitting parameter
        order: B-spline order

    Returns:
        Scalar reciprocal-space energy in kcal/mol.
    """
    # Step 1: Spread charges
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)

    # Step 2: R2C FFT (halves Z dimension → ~50% speedup)
    Q_hat = jnp.fft.rfftn(Q)

    # Step 3: Influence function (precomputable, but recompute for JIT simplicity)
    G = influence_function(grid_dims, box_size, alpha, order)

    # Step 4: Convolution in reciprocal space
    theta_hat = Q_hat * G

    # Step 5: Inverse FFT
    theta = jnp.fft.irfftn(theta_hat, s=grid_dims)

    # Step 6: Energy via convolution theorem
    # E_recip = 0.5 * COULOMB_CONSTANT * sum(Q * θ)
    # Factor N_grid (Kx*Ky*Kz) is required because jnp.fft.irfftn scales by 1/N,
    # but the discrete convolution sum requires the physical potential (unscaled).
    N_grid = grid_dims[0] * grid_dims[1] * grid_dims[2]
    energy = 0.5 * COULOMB_CONSTANT * jnp.sum(Q * theta) * N_grid

    return energy


def spme_self_energy(
    charges: jnp.ndarray,     # (N,)
    atom_mask: jnp.ndarray,   # (N,) bool
    alpha: float = 0.34,
) -> jnp.ndarray:
    """Compute the SPME self-energy correction.

    E_self = -α / √π * COULOMB_CONSTANT * ∑_i q_i²

    This must be subtracted from the reciprocal energy.

    Args:
        charges: (N,) partial charges
        atom_mask: (N,) boolean mask
        alpha: Ewald splitting parameter

    Returns:
        Scalar self-energy (negative value).
    """
    q_masked = charges * atom_mask.astype(jnp.float32)
    q_sq_sum = jnp.sum(q_masked ** 2)
    return -alpha / jnp.sqrt(jnp.pi) * COULOMB_CONSTANT * q_sq_sum


def spme_background_energy(
    charges: jnp.ndarray,     # (N,)
    atom_mask: jnp.ndarray,   # (N,) bool
    alpha: float,
    box_size: jnp.ndarray,    # (3,)
) -> jnp.ndarray:
    """Neutralizing background correction: E_bg = -π Q² / (2α² V).

    Required for systems with a net charge Q != 0 to prevent divergent
    reciprocal space term at k=0. Standard Ewald convention.

    Args:
        charges: (N,) partial charges
        atom_mask: (N,) boolean mask
        alpha: Ewald splitting parameter (1/Å)
        box_size: (3,) simulation box dimensions (Å)

    Returns:
        Scalar neutralizing background energy correction.
    """
    q_masked = charges * atom_mask.astype(jnp.float32)
    Q = jnp.sum(q_masked)
    V = jnp.prod(box_size)
    return -jnp.pi * Q**2 / (2.0 * alpha**2 * V) * COULOMB_CONSTANT


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
    """SPME energy + self-correction, with analytical gradient via custom_vjp.

    This avoids jax.grad checkpointing the 3D FFT grids, which would
    require O(K³) memory per gradient evaluation.

    Args:
        positions: (N, 3) atomic positions
        charges: (N,) partial charges
        atom_mask: (N,) boolean mask
        box_size: (3,) simulation box
        grid_dims: PME grid dimensions
        alpha: Ewald splitting parameter
        order: B-spline order

    Returns:
        Scalar total reciprocal energy (recip + self correction).
    """
    # NOTE: We call spme_reciprocal_energy directly here to avoid
    # infinite recursion with the custom_vjp decoration.
    e_recip = spme_reciprocal_energy(
        positions, charges, atom_mask, box_size, grid_dims, alpha, order
    )
    e_self = spme_self_energy(charges, atom_mask, alpha)
    return e_recip + e_self


def _spme_fwd(positions, charges, atom_mask, box_size, grid_dims, alpha, order):
    """Forward pass: compute energy and save residuals for backward."""
    # We call the functions directly to avoid recursion with the decorated 
    # spme_energy_with_forces.
    e_recip = spme_reciprocal_energy(
        positions, charges, atom_mask, box_size, grid_dims, alpha, order
    )
    e_self = spme_self_energy(charges, atom_mask, alpha)
    energy = e_recip + e_self

    # Recompute charge grid and potential for residuals
    Q = spread_charges(positions, charges, atom_mask, box_size, grid_dims, order)
    Q_hat = jnp.fft.rfftn(Q)
    G = influence_function(grid_dims, box_size, alpha, order)
    theta_hat = Q_hat * G
    theta = jnp.fft.irfftn(theta_hat, s=grid_dims)

    # Scale potential by grid size so it is physical potential (not scaled by 1/N_grid)
    N_grid = grid_dims[0] * grid_dims[1] * grid_dims[2]
    theta = theta * N_grid
    
    residuals = (positions, charges, atom_mask, box_size, theta)
    return energy, residuals


def _spme_bwd(grid_dims, alpha, order, residuals, g):
    """Backward pass: analytical forces from the potential grid.

    Force on atom i = -q_i * ∇θ(r_i) * COULOMB_CONSTANT

    where θ is the potential grid, and ∇θ is computed via B-spline
    derivative interpolation from the grid.
    """
    positions, charges, atom_mask, box_size, theta = residuals
    Kx, Ky, Kz = grid_dims

    # Fractional coordinates
    K_arr = jnp.array([Kx, Ky, Kz], dtype=jnp.float32)
    frac = positions / box_size * K_arr

    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(jnp.float32)

    # B-spline weights and derivatives for each dimension
    wx = _bspline4(u[:, 0])      # (4, N) values
    wy = _bspline4(u[:, 1])
    wz = _bspline4(u[:, 2])
    dwx = _bspline4_deriv(u[:, 0])  # (4, N) derivatives
    dwy = _bspline4_deriv(u[:, 1])
    dwz = _bspline4_deriv(u[:, 2])

    q_masked = charges * atom_mask.astype(jnp.float32)
    N = positions.shape[0]

    # Force accumulation via grid interpolation of ∇θ
    # F_i = -q_i * COULOMB * [dθ/dx, dθ/dy, dθ/dz]
    # dθ/dx_i = K_x/L_x * sum_{abc} dwx_a * wy_b * wz_c * θ(ix+a, iy+b, iz+c)
    # (chain rule: d/dx = (K/L) * d/du since u = x*K/L)

    forces = jnp.zeros((N, 3), dtype=jnp.float32)

    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz

                # Potential value at this grid point for each atom
                theta_val = theta[ix, iy, iz]  # (N,)

                # Force contributions via chain rule
                # dE/dx = q * K_x/L_x * dwx * wy * wz * θ
                fx = q_masked * (K_arr[0] / box_size[0]) * dwx[dx] * wy[dy] * wz[dz] * theta_val
                fy = q_masked * (K_arr[1] / box_size[1]) * wx[dx] * dwy[dy] * wz[dz] * theta_val
                fz = q_masked * (K_arr[2] / box_size[2]) * wx[dx] * wy[dy] * dwz[dz] * theta_val

                forces = forces + jnp.stack([fx, fy, fz], axis=-1)

    # Scale by COULOMB_CONSTANT and upstream gradient
    grad_positions = COULOMB_CONSTANT * forces * g

    # Gradient w.r.t. mask, box, and charges (zero)
    grad_charges = jnp.zeros_like(charges)
    grad_mask = jnp.zeros_like(atom_mask, dtype=jnp.float32)
    grad_box = jnp.zeros_like(box_size, dtype=jnp.float32)

    return grad_positions, grad_charges, grad_mask, grad_box


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
    """Create an SPME reciprocal energy function.

    Args:
        box_size: (3,) simulation box in Å
        alpha: Ewald splitting parameter
        grid_spacing: Target grid spacing in Å
        order: B-spline order (4 = cubic)

    Returns:
        Callable (positions, charges, atom_mask) → energy
    """
    grid_dims = compute_pme_grid_dims(box_size, grid_spacing)

    def energy_fn(
        positions: jnp.ndarray,
        charges: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        return spme_energy_with_forces(
            positions, charges, atom_mask, box_size,
            grid_dims, alpha, order,
        )

    return energy_fn


def make_pme_energy_fn(
    charges: jnp.ndarray,
    box_size: jnp.ndarray,
    *,
    grid_points: int = 64,
    alpha: float = 0.34,
    order: int = 4,
):
    """Bind charges and box; return ``energy(positions)`` for SPME + background.

    Reciprocal + self term uses :func:`spme_energy_with_forces` (kcal/mol).
    Adds neutralizing background :func:`spme_background_energy` for net charge.

    This is the API expected by ``tests/physics/test_explicit_parity`` and
    callers that only pass positions each evaluation.

    Args:
        charges: (N,) partial charges (fixed for the lifetime of the closure).
        box_size: (3,) orthorhombic box lengths in Å.
        grid_points: Target mesh size per axis (spacing = mean(box)/grid_points).
        alpha: Ewald splitting parameter (1/Å).
        order: B-spline order (4 = cubic).

    Returns:
        Callable ``energy_fn(positions) -> scalar`` in kcal/mol.
    """
    charges = jnp.asarray(charges)
    box_size = jnp.asarray(box_size)
    n = int(charges.shape[0])
    mask = jnp.ones((n,), dtype=bool)
    mean_l = jnp.mean(box_size.astype(jnp.float64))
    grid_spacing = float(mean_l) / float(max(grid_points, 1))
    inner = make_spme_energy_fn(
        box_size, alpha=alpha, grid_spacing=grid_spacing, order=order
    )

    def energy_fn(positions: jnp.ndarray) -> jnp.ndarray:
        e = inner(positions, charges, mask)
        e = e + spme_background_energy(charges, mask, alpha, box_size)
        return e

    return energy_fn
