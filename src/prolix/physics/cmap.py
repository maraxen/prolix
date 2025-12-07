# File: src/prolix.physics/cmap.py
"""CMAP torsion energy using OpenMM-compatible periodic cubic spline interpolation.

This module implements CMAP energy calculation using the same algorithm as OpenMM's
CMAPTorsionForceImpl::calcMapDerivatives(). The key difference from simple finite
differences is the use of periodic cubic spline fitting to compute derivatives at
grid points, which provides much better accuracy for angles between grid points.

References:
    - OpenMM source: openmmapi/src/CMAPTorsionForceImpl.cpp
    - OpenMM source: platforms/reference/src/SimTKReference/ReferenceCMAPTorsionIxn.cpp
"""
import jax
import jax.numpy as jnp
from typing import Tuple

# OpenMM weight matrix for computing bicubic spline coefficients from corner values
# This converts [f, df/dx, df/dy, d²f/dxdy] at 4 corners → 16 polynomial coefficients
# From OpenMM CMAPTorsionForceImpl.cpp
WT = jnp.array([
    [1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4],
    [0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4],
    [0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2],
    [0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2],
    [0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2],
    [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2],
    [0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1],
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1]
], dtype=jnp.float64)


def create_periodic_spline(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Fit a periodic natural cubic spline through (x, y) data points.
    
    This matches OpenMM's SplineFitter::createPeriodicSpline().
    For a periodic spline: f(x[0]) = f(x[n]), f'(x[0]) = f'(x[n]), f''(x[0]) = f''(x[n])
    
    Args:
        x: x-coordinates (n+1,) where x[n] = x[0] + period (for periodicity)
        y: y-values (n+1,) where y[n] = y[0] (for periodicity)
        
    Returns:
        deriv: Second derivatives at each point (n+1,)
    """
    n = len(x) - 1
    
    # Build tridiagonal system for periodic spline
    # Using Sherman-Morrison formula for periodic boundary
    
    h = jnp.diff(x)  # h[i] = x[i+1] - x[i]
    
    # Build right-hand side
    # b[i] = 6 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    dy = jnp.diff(y)
    slopes = dy / h
    
    # For periodic: index -1 wraps to n-1
    b = jnp.zeros(n)
    b = b.at[1:].set(6 * (slopes[1:] - slopes[:-1]))
    b = b.at[0].set(6 * (slopes[0] - slopes[n-1]))
    
    # Build tridiagonal matrix coefficients
    # Diagonal: 2*(h[i-1] + h[i])
    # Off-diagonal: h[i]
    
    diag = 2 * (jnp.roll(h, 1) + h)
    
    # Solve using Thomas algorithm with periodic correction
    # For periodic spline, we use the Sherman-Morrison formula
    # M * z = b where M is almost tridiagonal but has corner elements
    
    # Standard tridiagonal part
    lower = h.copy()  # sub-diagonal
    upper = jnp.roll(h, -1)  # super-diagonal (shifted)
    
    # Solve Az = b where A is the periodic tridiagonal matrix
    # Use cyclic reduction / Sherman-Morrison
    
    # For simplicity, use direct solve with periodic BCs
    # Build full matrix (small for typical CMAP grids of 24x24)
    A = jnp.diag(diag) + jnp.diag(lower[1:], -1) + jnp.diag(upper[:-1], 1)
    # Add periodic corner elements
    A = A.at[0, n-1].set(h[n-1])
    A = A.at[n-1, 0].set(h[n-1])
    
    # Solve for second derivatives (M'')
    deriv_interior = jnp.linalg.solve(A, b)
    
    # Extend to include endpoint (periodic: deriv[n] = deriv[0])
    deriv = jnp.concatenate([deriv_interior, deriv_interior[0:1]])
    
    return deriv


def evaluate_spline_derivative(
    x: jnp.ndarray, y: jnp.ndarray, deriv: jnp.ndarray, t: float
) -> float:
    """Evaluate the first derivative of a cubic spline at point t.
    
    This matches OpenMM's SplineFitter::evaluateSplineDerivative().
    
    Args:
        x: x-coordinates (n+1,)
        y: y-values (n+1,)
        deriv: Second derivatives from create_periodic_spline (n+1,)
        t: Point at which to evaluate the derivative
        
    Returns:
        First derivative f'(t)
    """
    n = len(x) - 1
    
    # Find interval containing t
    # For periodic, t should be in [x[0], x[n])
    idx = jnp.searchsorted(x[1:], t, side='right')
    idx = jnp.clip(idx, 0, n - 1)
    
    h = x[idx + 1] - x[idx]
    a = (x[idx + 1] - t) / h
    b = (t - x[idx]) / h
    
    # Derivative of cubic spline:
    # f(t) = a*y[i] + b*y[i+1] + ((a³-a)*M[i] + (b³-b)*M[i+1]) * h²/6
    # f'(t) = (y[i+1] - y[i])/h + h/6 * ((1-3a²)*M[i] + (3b²-1)*M[i+1])
    
    dy = y[idx + 1] - y[idx]
    dfdx = dy / h + h / 6.0 * ((1 - 3 * a**2) * deriv[idx] + (3 * b**2 - 1) * deriv[idx + 1])
    
    return dfdx


def compute_map_derivatives(
    energy: jnp.ndarray, size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute CMAP derivatives using periodic cubic spline fitting.
    
    This matches OpenMM's CMAPTorsionForceImpl::calcMapDerivatives().
    
    Grid convention: energy[i, j] where i indexes first angle (phi), j indexes second angle (psi)
    - Index i corresponds to angle i * 2π/size
    - Index j corresponds to angle j * 2π/size
    
    Args:
        energy: (size, size) energy grid values where energy[i,j] = E(phi_i, psi_j)
        size: Grid size (typically 24)
        
    Returns:
        d1: Derivatives with respect to first angle (dE/dphi) at each grid point
        d2: Derivatives with respect to second angle (dE/dpsi) at each grid point
        d12: Cross derivatives (d²E/dphi dpsi)
    """
    # Create x-coordinates for [0, 2π] range (size+1 points for periodicity)
    x = jnp.linspace(0, 2 * jnp.pi, size + 1)
    
    # Compute d1: derivatives with respect to first angle (phi, axis 0)
    # For fixed psi (column j), compute dE/dphi by fitting spline along phi (rows)
    def compute_d1_col(j):
        # Extract column: energy[:, j] = E(phi_0..phi_n-1, psi_j)
        col = energy[:, j]
        y = jnp.concatenate([col, col[0:1]])  # Make periodic
        deriv = create_periodic_spline(x, y)
        # Evaluate first derivative at each phi grid point
        return jax.vmap(lambda i: evaluate_spline_derivative(x, y, deriv, x[i]))(
            jnp.arange(size)
        )
    
    # d1[i, j] = dE/dphi at (phi_i, psi_j)
    d1 = jax.vmap(compute_d1_col)(jnp.arange(size)).T  # Transpose: (size, size)
    
    # Compute d2: derivatives with respect to second angle (psi, axis 1)
    # For fixed phi (row i), compute dE/dpsi by fitting spline along psi (columns)
    def compute_d2_row(i):
        # Extract row: energy[i, :] = E(phi_i, psi_0..psi_n-1)
        row = energy[i, :]
        y = jnp.concatenate([row, row[0:1]])  # Make periodic
        deriv = create_periodic_spline(x, y)
        # Evaluate first derivative at each psi grid point
        return jax.vmap(lambda j: evaluate_spline_derivative(x, y, deriv, x[j]))(
            jnp.arange(size)
        )
    
    # d2[i, j] = dE/dpsi at (phi_i, psi_j)
    d2 = jax.vmap(compute_d2_row)(jnp.arange(size))  # Shape: (size, size)
    
    # Compute d12: cross derivative d/dphi(dE/dpsi) = d²E/(dphi dpsi)
    # Use d2 values and compute derivative along phi direction (columns)
    def compute_d12_col(j):
        # d2[:, j] = dE/dpsi at (phi_0..phi_n-1, psi_j)
        col = d2[:, j]
        y = jnp.concatenate([col, col[0:1]])
        deriv = create_periodic_spline(x, y)
        return jax.vmap(lambda i: evaluate_spline_derivative(x, y, deriv, x[i]))(
            jnp.arange(size)
        )
    
    d12 = jax.vmap(compute_d12_col)(jnp.arange(size)).T
    
    return d1, d2, d12

    d12 = jax.vmap(compute_d12_col)(jnp.arange(size)).T
    
    return d1, d2, d12


def precompute_cmap_coefficients(cmap_grids: jnp.ndarray) -> jnp.ndarray:
    """Precompute bicubic spline coefficients for a batch of CMAP grids.
    
    Args:
        cmap_grids: (N_maps, Grid, Grid) raw energy grids
        
    Returns:
        coeffs: (N_maps, Grid, Grid, 16) precomputed coefficients
    """
    if cmap_grids.ndim != 3:
        raise ValueError(f"cmap_grids must be 3D (N, Grid, Grid), got {cmap_grids.shape}")
        
    n_maps = cmap_grids.shape[0]
    grid_size = cmap_grids.shape[1]
    
    # Transpose grids: OpenMM stores energy[i+size*j] (column-major) but XML
    # parsing with Python default reshape uses C-order, transposing phi/psi axes.
    # We transpose each grid to match OpenMM convention: grid[phi_idx, psi_idx]
    cmap_grids_transposed = jnp.transpose(cmap_grids, (0, 2, 1))
    
    # Compute coefficients for each map
    def compute_map_coeffs(m):
        return compute_bicubic_coefficients(cmap_grids_transposed[m], grid_size)
    
    # We use vmap to compute efficienty
    coeffs = jax.vmap(compute_map_coeffs)(jnp.arange(n_maps))
    return coeffs



def compute_bicubic_coefficients(
    energy: jnp.ndarray, size: int
) -> jnp.ndarray:
    """Compute bicubic spline coefficients for each patch.
    
    This matches OpenMM's coefficient computation in calcMapDerivatives().
    
    Args:
        energy: (size, size) energy grid
        size: Grid size
        
    Returns:
        coeffs: (size*size, 16) bicubic coefficients for each patch
    """
    d1, d2, d12 = compute_map_derivatives(energy, size)
    
    delta = 2 * jnp.pi / size
    
    def compute_patch_coeffs(idx):
        i = idx // size
        j = idx % size
        nexti = (i + 1) % size
        nextj = (j + 1) % size
        
        # Corner values: energy
        e = jnp.array([
            energy[i, j], energy[nexti, j], 
            energy[nexti, nextj], energy[i, nextj]
        ])
        # First derivatives wrt phi
        e1 = jnp.array([d1[i, j], d1[nexti, j], d1[nexti, nextj], d1[i, nextj]])
        # First derivatives wrt psi
        e2 = jnp.array([d2[i, j], d2[nexti, j], d2[nexti, nextj], d2[i, nextj]])
        # Cross derivatives
        e12 = jnp.array([d12[i, j], d12[nexti, j], d12[nexti, nextj], d12[i, nextj]])
        
        # Build RHS vector (scaled by delta as in OpenMM)
        rhs = jnp.concatenate([e, e1 * delta, e2 * delta, e12 * delta * delta])
        
        # Apply weight matrix (transposed because OpenMM stores wt in column-major order)
        coeffs = jnp.dot(WT.T, rhs)
        return coeffs
    
    coeffs = jax.vmap(compute_patch_coeffs)(jnp.arange(size * size))
    return coeffs.reshape(size, size, 16)


def eval_bicubic_patch(coeffs: jnp.ndarray, da: float, db: float) -> float:
    """Evaluate bicubic polynomial at local coordinates (da, db).
    
    This matches OpenMM's ReferenceCMAPTorsionIxn evaluation.
    Coefficients are stored as c[i*4+j] where polynomial is sum c[i*4+j] * da^i * db^j.
    
    Args:
        coeffs: (16,) polynomial coefficients for this patch
        da: Local coordinate in [0, 1) for first angle
        db: Local coordinate in [0, 1) for second angle
        
    Returns:
        Interpolated energy value
    """
    # Vectorized: sum_{i,j} c[i*4+j] * da^i * db^j
    # Build power vectors [1, x, x², x³]
    da_pow = jnp.array([1.0, da, da**2, da**3])
    db_pow = jnp.array([1.0, db, db**2, db**3])
    
    # Coefficients are stored as c[i*4+j] -> reshape to (4, 4) with c[i, j]
    c_matrix = coeffs.reshape(4, 4)
    
    # Compute: sum_i sum_j c[i,j] * da^i * db^j = da_pow @ c_matrix @ db_pow
    return jnp.dot(da_pow, jnp.dot(c_matrix, db_pow))


def compute_cmap_energy(
    phi_angles: jnp.ndarray,
    psi_angles: jnp.ndarray,
    map_indices: jnp.ndarray,
    cmap_coeffs: jnp.ndarray,
) -> float:
    """Compute CMAP energy using OpenMM-compatible bicubic spline interpolation.
    
    Args:
        phi_angles: (N_torsions,) phi angles in radians [-π, π]
        psi_angles: (N_torsions,) psi angles in radians [-π, π]
        map_indices: (N_torsions,) index of map to use for each torsion
        cmap_coeffs: Either:
            - (N_maps, Grid, Grid) raw energy grids (will compute coefficients)
            - (N_maps, Grid, Grid, 16) precomputed bicubic coefficients
            
    Returns:
        Total CMAP energy (scalar) in kcal/mol (same units as input grid)
    """
    # Determine if we have raw energy or precomputed coefficients
    if cmap_coeffs.ndim == 3:
        # Raw energy grids - compute bicubic coefficients
        n_maps = cmap_coeffs.shape[0]
        grid_size = cmap_coeffs.shape[1]
        
        # Transpose grids: OpenMM stores energy[i+size*j] (column-major) but XML
        # parsing with Python default reshape uses C-order, transposing phi/psi axes.
        # We transpose each grid to match OpenMM convention: grid[phi_idx, psi_idx]
        cmap_grids_transposed = jnp.transpose(cmap_coeffs, (0, 2, 1))
        
        # Compute coefficients for each map
        def compute_map_coeffs(m):
            return compute_bicubic_coefficients(cmap_grids_transposed[m], grid_size)
        
        coeffs = jax.vmap(compute_map_coeffs)(jnp.arange(n_maps))
        # Shape: (N_maps, Grid, Grid, 16)
    else:
        coeffs = cmap_coeffs
        grid_size = coeffs.shape[1]
    
    # Normalize angles to [0, 2π) - equivalent to OpenMM's fmod(angle + 2*M_PI, 2*M_PI)
    # For angles in [-π, π], mod(angle + 2π, 2π) is equivalent to fmod(angle + 2π, 2π)
    phi_norm = jnp.mod(phi_angles + 2 * jnp.pi, 2 * jnp.pi)
    psi_norm = jnp.mod(psi_angles + 2 * jnp.pi, 2 * jnp.pi)
    
    # Convert to grid coordinates
    delta = 2 * jnp.pi / grid_size
    
    def sample_one(m_idx, phi, psi):
        # Grid indices
        i = jnp.floor(phi / delta).astype(jnp.int32)
        j = jnp.floor(psi / delta).astype(jnp.int32)
        i = jnp.clip(i, 0, grid_size - 1)
        j = jnp.clip(j, 0, grid_size - 1)
        
        # Local coordinates [0, 1)
        da = phi / delta - i
        db = psi / delta - j
        
        # Get patch coefficients
        patch_coeffs = coeffs[m_idx, i, j]
        
        # Evaluate
        return eval_bicubic_patch(patch_coeffs, da, db)
    
    energies = jax.vmap(sample_one)(map_indices, phi_norm, psi_norm)
    
    return jnp.sum(energies)

