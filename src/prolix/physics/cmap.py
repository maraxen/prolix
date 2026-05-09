# File: src/prolix/physics/cmap.py
"""CMAP torsion energy using OpenMM-compatible periodic cubic spline interpolation.

This module implements CMAP energy calculation using the same algorithm as OpenMM's
CMAPTorsionForceImpl::calcMapDerivatives(). The key difference from simple finite
differences is the use of periodic cubic spline fitting to compute derivatives at
grid points, which provides much better accuracy for angles between grid points.

References:
    - OpenMM source: openmmapi/src/CMAPTorsionForceImpl.cpp
    - OpenMM source: platforms/reference/src/SimTKReference/ReferenceCMAPTorsionIxn.cpp

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Float

if TYPE_CHECKING:
  from prolix.types import (
    ArrayLike,
    CmapCoeffs,
    CmapEnergyGrids,
    CmapGrid,
    CmapPoints,
    ScalarFloat,
    TorsionAngles,
    TorsionIndices,
  )


# OpenMM weight matrix for computing bicubic spline coefficients from corner values
# This converts [f, df/dx, df/dy, d²f/dxdy] at 4 corners → 16 polynomial coefficients
# From OpenMM CMAPTorsionForceImpl.cpp
WT = jnp.array(
  [
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
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1],
  ],
  # TODO(mar): float64 may be needed for explicit solvent CMAP accuracy.
  # For implicit solvent (f32 GPU), JAX silently downcasts this when x64=False.
  dtype=jnp.float64,
)


def create_periodic_spline(
  x: CmapPoints,
  y: CmapPoints,
) -> CmapPoints:
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
  b = b.at[0].set(6 * (slopes[0] - slopes[n - 1]))

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
  A = A.at[0, n - 1].set(h[n - 1])
  A = A.at[n - 1, 0].set(h[n - 1])

  # Solve for second derivatives (M'')
  deriv_interior = jnp.linalg.solve(A, b)

  # Extend to include endpoint (periodic: deriv[n] = deriv[0])
  return jnp.concatenate([deriv_interior, deriv_interior[0:1]])


def evaluate_spline_derivative(
  x: CmapPoints,
  y: CmapPoints,
  deriv: CmapPoints,
  t: ScalarFloat,
) -> ScalarFloat:
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
  idx = jnp.searchsorted(x[1:], t, side="right")
  idx = jnp.clip(idx, 0, n - 1)

  h = x[idx + 1] - x[idx]
  a = (x[idx + 1] - t) / h
  b = (t - x[idx]) / h

  # Derivative of cubic spline:
  # f(t) = a*y[i] + b*y[i+1] + ((a³-a)*M[i] + (b³-b)*M[i+1]) * h²/6
  # f'(t) = (y[i+1] - y[i])/h + h/6 * ((1-3a²)*M[i] + (3b²-1)*M[i+1])

  dy = y[idx + 1] - y[idx]
  return dy / h + h / 6.0 * ((1 - 3 * a**2) * deriv[idx] + (3 * b**2 - 1) * deriv[idx + 1])


def compute_map_derivatives(energy: CmapGrid, size: int) -> tuple[CmapGrid, CmapGrid, CmapGrid]:
  """Compute CMAP derivatives using OpenMM's periodic cubic spline method.

  Matches OpenMM's CMAPTorsionForceImpl::calcMapDerivatives() exactly.
  Uses SplineFitter::createPeriodicSpline and evaluateSplineDerivative,
  NOT finite differences.

  Args:
      energy: (size, size) energy grid where energy[i,j] = E(phi_i, psi_j),
              i.e. axis 0 = phi, axis 1 = psi (caller must transpose from proxide layout)
      size: Grid size (typically 24)

  Returns:
      d1: phi derivatives at each grid point (kcal/mol/rad), shape (size, size)
      d2: psi derivatives at each grid point (kcal/mol/rad), shape (size, size)
      d12: cross derivatives d²E/dphi/dpsi (kcal/mol/rad²), shape (size, size)

  """
  x = jnp.linspace(0, 2 * jnp.pi, size + 1)

  def d1_at_psi_j(j):
    # phi-derivative: spline along phi (axis 0) for fixed psi index j
    col = energy[:, j]
    y = jnp.concatenate([col, col[0:1]])
    deriv = create_periodic_spline(x, y)
    return jax.vmap(lambda i: evaluate_spline_derivative(x, y, deriv, x[i]))(jnp.arange(size))

  def d2_at_phi_i(i):
    # psi-derivative: spline along psi (axis 1) for fixed phi index i
    row = energy[i, :]
    y = jnp.concatenate([row, row[0:1]])
    deriv = create_periodic_spline(x, y)
    return jax.vmap(lambda j: evaluate_spline_derivative(x, y, deriv, x[j]))(jnp.arange(size))

  # d1: vmap over psi (j), result shape (size_j, size_i) -> transpose to (phi, psi)
  d1 = jax.vmap(d1_at_psi_j)(jnp.arange(size)).T

  # d2: vmap over phi (i), result shape (size_i, size_j) = (phi, psi)
  d2 = jax.vmap(d2_at_phi_i)(jnp.arange(size))

  def d12_at_psi_j(j):
    # cross derivative: spline d1 along phi for fixed psi index j
    col = d2[:, j]
    y = jnp.concatenate([col, col[0:1]])
    deriv = create_periodic_spline(x, y)
    return jax.vmap(lambda i: evaluate_spline_derivative(x, y, deriv, x[i]))(jnp.arange(size))

  # d12: vmap over psi (j), result shape (size_j, size_i) -> transpose to (phi, psi)
  d12 = jax.vmap(d12_at_psi_j)(jnp.arange(size)).T

  return d1, d2, d12


def precompute_cmap_coefficients(cmap_grids: CmapEnergyGrids) -> CmapCoeffs:
  """Precompute bicubic spline coefficients for a batch of CMAP grids.

  Args:
      cmap_grids: (N_maps, Grid, Grid) raw energy grids where axis 0 = phi, axis 1 = psi

  Returns:
      coeffs: (N_maps, Grid, Grid, 16) precomputed coefficients

  """
  if cmap_grids.ndim != 3:
    msg = f"cmap_grids must be 3D (N, Grid, Grid), got {cmap_grids.shape}"
    raise ValueError(msg)

  n_maps = cmap_grids.shape[0]
  grid_size = cmap_grids.shape[1]

  def compute_map_coeffs(m):
    return compute_bicubic_coefficients(cmap_grids[m], grid_size)

  return jax.vmap(compute_map_coeffs)(jnp.arange(n_maps))


def compute_bicubic_coefficients(energy: CmapGrid, size: int) -> CmapCoeffs:
  """Compute bicubic spline coefficients for each patch.

  This matches OpenMM's coefficient computation in calcMapDerivatives().

  Args:
      energy: (size, size) energy grid from proxide, stored as (psi, phi) — i.e.
              energy[psi_i, phi_j]. Transposed internally to (phi, psi) convention.
      size: Grid size

  Returns:
      coeffs: (size, size, 16) bicubic coefficients, indexed as coeffs[phi_i, psi_j]

  """
  # Proxide stores grids as (psi, phi) because OpenMM's flat F-order array
  # energy[phi + psi*size] becomes grid[psi, phi] after C-order reshape.
  # Transpose to (phi, psi) so axis 0 = phi throughout.
  energy = energy.T

  d1, d2, d12 = compute_map_derivatives(energy, size)

  # delta = grid spacing in radians; used to scale spline derivatives to [0,1] local coords
  delta = 2 * jnp.pi / size

  def compute_patch_coeffs(idx):
    # i = phi index (axis 0 after transpose), j = psi index (axis 1)
    i = idx // size
    j = idx % size
    nexti = (i + 1) % size
    nextj = (j + 1) % size

    # Corner values
    e = jnp.array([energy[i, j], energy[nexti, j], energy[nexti, nextj], energy[i, nextj]])
    e1 = jnp.array([d1[i, j], d1[nexti, j], d1[nexti, nextj], d1[i, nextj]])
    e2 = jnp.array([d2[i, j], d2[nexti, j], d2[nexti, nextj], d2[i, nextj]])
    e12 = jnp.array([d12[i, j], d12[nexti, j], d12[nexti, nextj], d12[i, nextj]])

    # Scale derivatives by delta to match OpenMM's rhs convention:
    # rhs[k+4] = e1[k]*delta, rhs[k+8] = e2[k]*delta, rhs[k+12] = e12[k]*delta^2
    rhs = jnp.concatenate([e, e1 * delta, e2 * delta, e12 * delta * delta])

    return jnp.dot(WT.T, rhs)

  coeffs = jax.vmap(compute_patch_coeffs)(jnp.arange(size * size))
  return coeffs.reshape(size, size, 16)


def eval_bicubic_patch(coeffs: Float[ArrayLike, 16], da: float, db: float) -> ScalarFloat:
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


def compute_cmap_energies(
  phi_angles: TorsionAngles,
  psi_angles: TorsionAngles,
  map_indices: TorsionIndices,
  cmap_coeffs: CmapCoeffs | CmapEnergyGrids,
) -> Array:
  """Compute CMAP energies using OpenMM-compatible bicubic spline interpolation.

  Args:
      phi_angles: (N_torsions,) phi angles in radians [-π, π]
      psi_angles: (N_torsions,) psi angles in radians [-π, π]
      map_indices: (N_torsions,) index of map to use for each torsion
      cmap_coeffs: Either:
          - (N_maps, Grid, Grid) raw energy grids (will compute coefficients)
          - (N_maps, Grid, Grid, 16) precomputed bicubic coefficients

  Returns:
      Per-torsion CMAP energies (N_torsions,) in kcal/mol (same units as input grid)

  """
  # Determine if we have raw energy or precomputed coefficients
  if cmap_coeffs.ndim == 3:
    # Raw energy grids - compute bicubic coefficients
    n_maps = cmap_coeffs.shape[0]
    grid_size = cmap_coeffs.shape[1]

    # Compute coefficients for each map
    def compute_map_coeffs(m):
      return compute_bicubic_coefficients(cmap_coeffs[m], grid_size)

    coeffs = jax.vmap(compute_map_coeffs)(jnp.arange(n_maps))
    # Shape: (N_maps, Grid, Grid, 16)
  else:
    coeffs = cmap_coeffs
    grid_size = coeffs.shape[1]

  # Map angles from [-pi, pi] to [0, 2*pi) to match OpenMM's fmod(angle+2pi, 2pi).
  phi_norm = jnp.mod(phi_angles + 2 * jnp.pi, 2 * jnp.pi)
  psi_norm = jnp.mod(psi_angles + 2 * jnp.pi, 2 * jnp.pi)

  # Convert to grid coordinates
  delta = 2 * jnp.pi / grid_size

  def sample_one(m_idx, phi, psi):
    # Grid indices
    i = jnp.floor(phi / delta).astype(jnp.int32)
    j = jnp.floor(psi / delta).astype(jnp.int32)
    i = i % grid_size
    j = j % grid_size

    # Local coordinates [0, 1)
    da = phi / delta - i
    db = psi / delta - j

    # Get patch coefficients
    patch_coeffs = coeffs[m_idx, i, j]

    # Evaluate
    return eval_bicubic_patch(patch_coeffs, da, db)

  return jax.vmap(sample_one)(map_indices, phi_norm, psi_norm)

def compute_cmap_energy(
  phi_angles: TorsionAngles,
  psi_angles: TorsionAngles,
  map_indices: TorsionIndices,
  cmap_coeffs: CmapCoeffs | CmapEnergyGrids,
) -> ScalarFloat:
  """Compute total CMAP energy (summed). Kept for backward compatibility.
  
  See compute_cmap_energies for arguments.
  """
  energies = compute_cmap_energies(phi_angles, psi_angles, map_indices, cmap_coeffs)
  return jnp.sum(energies)
