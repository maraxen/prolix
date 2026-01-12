"""Analysis tools for MD trajectories."""

from __future__ import annotations

import jax.numpy as jnp
from jax_md import util

Array = util.Array


def compute_rmsd(positions: Array, reference: Array, mask: Array | None = None) -> Array:
  """Compute Root Mean Square Deviation (RMSD) from reference structure.

  Args:
      positions: (N, 3) or (B, N, 3) coordinates.
      reference: (N, 3) reference coordinates.
      mask: (N,) boolean mask of atoms to include (e.g. backbone).

  Returns:
      Scalar RMSD value or (B,) array of RMSD values.

  """
  if mask is not None:
    p = positions[..., mask, :]
    r = reference[mask, :]
    jnp.sum(mask)
  else:
    p = positions
    r = reference
    r.shape[0]

  # Kabsch alignment? Usually RMSD implies superposition.
  # For now, we assume pre-aligned or we perform alignment.
  # Implementing Kabsch here is complex in pure JAX without external imports like helper geometry.
  # However, simple RMSD without superposition is (P - R)^2.
  # Usually "RMSD" implies minimal RMSD (after rotation).
  # We'll calculate simple RMSD for now (assuming alignment is done externally or by the user).
  # Or strict RMSD: sqrt(sum((p - r)**2) / N)

  diff = p - r
  sq_dist = jnp.sum(diff**2, axis=-1)  # (N,) or (B, N)
  mean_sq_dist = jnp.mean(sq_dist, axis=-1)  # Scalar or (B,)
  return jnp.sqrt(mean_sq_dist)


def compute_radius_of_gyration(positions: Array, masses: Array | None = None) -> Array:
  """Compute Radius of Gyration.

  Args:
      positions: (N, 3)
      masses: (N,) or None

  Returns:
      Scalar Rg

  """
  if masses is None:
    masses = jnp.ones(positions.shape[-2])

  masses = masses / jnp.sum(masses)  # Normalize weights

  # Center of mass
  # Handle batching carefully: masses is (N,), positions (..., N, 3)
  # weights: (N, 1) broadcastable against (..., N, 3)
  w = masses[..., None]

  com = jnp.sum(positions * w, axis=-2, keepdims=True)

  # Distance from COM
  diff = positions - com
  sq_dist = jnp.sum(diff**2, axis=-1)  # (..., N)

  # Weighted mean square distance
  rg_sq = jnp.sum(sq_dist * masses, axis=-1)

  return jnp.sqrt(rg_sq)


def compute_contact_map(
  positions: Array, threshold_angstrom: float = 8.0, mask: Array | None = None
) -> Array:
  """Compute contact map (boolean or probability).

  Args:
      positions: (N, 3)
      threshold_angstrom: Cutoff for contact
      mask: Optional (N,) mask to filter atoms

  Returns:
      (N, N) boolean adjacency matrix (1 if < threshold, 0 otherwise)

  """
  pos = positions[mask] if mask is not None else positions

  # Distance matrix
  # vmap displacement? Or broadcasting
  # r_i - r_j

  # Naive n^2 broadcasting is fine for small proteins (Trp-cage ~138 atoms)
  # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
  diff = pos[:, None, :] - pos[None, :, :]
  dist = jnp.linalg.norm(diff, axis=-1)

  return (dist < threshold_angstrom).astype(jnp.float32)


def compute_fraction_native_contacts(
  positions: Array, reference: Array, threshold_angstrom: float = 8.0, mask: Array | None = None
) -> Array:
  """Compute Q (fraction of native contacts)."""
  native_map = compute_contact_map(reference, threshold_angstrom, mask)
  current_map = compute_contact_map(
    positions, threshold_angstrom * 1.2, mask
  )  # Often slightly looser

  # Mask out diagonal and neighbors (assume |i-j| > 3 for protein contacts usually)
  # But simple version:
  n_contacts = jnp.sum(native_map)
  shared = jnp.sum(native_map * current_map)

  return shared / (n_contacts + 1e-8)


def compute_dihedrals(positions: Array, indices: Array) -> Array:
  """Compute dihedral angles for given quadruple indices.

  Args:
      positions: (N, 3)
      indices: (M, 4) indices (i, j, k, l)

  Returns:
      (M,) angles in radians [-pi, pi]

  """
  # Reuse logic from bonded.py

  r_i = positions[..., indices[:, 0], :]
  r_j = positions[..., indices[:, 1], :]
  r_k = positions[..., indices[:, 2], :]
  r_l = positions[..., indices[:, 3], :]

  # Vectors
  # b0: i -> j
  # b1: j -> k
  # b2: k -> l
  # using simple difference (no PBC/displacement_fn for standard analysis usually)
  b0 = r_j - r_i
  b1 = r_k - r_j
  b2 = r_l - r_k

  # Normalize b1
  b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
  b1_unit = b1 / b1_norm

  # Projections onto plane perpendicular to b1
  # v = b0 - (b0 . b1_unit) * b1_unit
  # w = b2 - (b2 . b1_unit) * b1_unit
  v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
  w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

  # Angle calculation
  # x = v . w
  # y = (b1_unit x v) . w
  x = jnp.sum(v * w, axis=-1)
  y = jnp.sum(jnp.cross(b1_unit, v, axis=-1) * w, axis=-1)

  return jnp.arctan2(y, x)


def compute_free_energy_surface(
  metric: Array,
  temperature: float = 300.0,
  bins: int = 50,
  range: tuple[float, float] | None = None,
) -> tuple[Array, Array]:
  """Compute 1D Free Energy Surface (FES) from histogram of a metric (e.g. RMSD).

  F(x) = -kT * ln(P(x))

  Args:
      metric: (T,) array of reaction coordinate values
      temperature: Kelvin
      bins: Number of histogram bins
      range: (min, max) tuple

  Returns:
      (centers, free_energy)

  """
  # Histogram
  hist, edges = jnp.histogram(metric, bins=bins, range=range)

  # Bin centers
  centers = (edges[:-1] + edges[1:]) / 2.0

  # Probability
  prob = hist / jnp.sum(hist)

  # Free Energy
  kT = 0.0019872041 * temperature  # kcal/mol

  # Clip prob to avoid log(0)
  prob_safe = jnp.maximum(prob, 1e-10)

  fes = -kT * jnp.log(prob_safe)

  # Shift minimum to 0
  fes = fes - jnp.min(fes)

  return centers, fes
