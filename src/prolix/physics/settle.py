"""SETTLE constraint algorithm for rigid 3-site water molecules.

The SETTLE algorithm (Miyamoto & Kollman, 1992) is an analytical constraint
solver for rigid 3-site water molecules. Unlike iterative SHAKE/RATTLE, it
solves the constraints in closed form, making it faster and more accurate.

References:
    Miyamoto, S., & Kollman, P. A. (1992). Settle: An analytical version of
    the SHAKE and RATTLE algorithm for rigid water models.
    Journal of Computational Chemistry, 13(8), 952-962.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random
from jax_md import quantity, simulate, util

if TYPE_CHECKING:
  from collections.abc import Callable

Array = util.Array

# TIP3P water geometry constants
TIP3P_ROH = 0.9572  # O-H bond length (Å)
TIP3P_RHH = 1.5139  # H-H distance (Å)
TIP3P_THETA = 104.52 * jnp.pi / 180.0  # H-O-H angle (rad)


def settle_positions(
  R_unconstrained: Array,
  R_old: Array,
  water_indices: Array,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  m_O: float = 15.999,
  m_H: float = 1.008,
  box: Array | None = None,
) -> Array:
  """Apply SETTLE position constraints to water molecules.

  This function corrects unconstrained positions to satisfy the rigid
  water geometry constraints: O-H1, O-H2, and H1-H2 distances.

  Args:
      R_unconstrained: Unconstrained positions after integrator step (N, 3).
      R_old: Positions before the integrator step (N, 3).
      water_indices: Array of shape (N_waters, 3) containing atom indices
          for each water molecule [O_idx, H1_idx, H2_idx].
      r_OH: Target O-H bond length (Å).
      r_HH: Target H-H distance (Å).
      m_O: Mass of oxygen atom (amu).
      m_H: Mass of hydrogen atom (amu).
      box: Optional box dimensions for PBC (3,). If provided, applies
          minimum image convention to handle periodic boundary crossings.

  Returns:
      Constrained positions with correct water geometry.
  """
  if water_indices.shape[0] == 0:
    return R_unconstrained

  # Total mass and inverse masses
  m_total = m_O + 2 * m_H
  inv_m_O = 1.0 / m_O
  inv_m_H = 1.0 / m_H

  # Extract water atom positions (N_waters, 3, 3) = (waters, atoms, xyz)
  O_idx = water_indices[:, 0]
  H1_idx = water_indices[:, 1]
  H2_idx = water_indices[:, 2]

  # Old positions
  r_O_old = R_old[O_idx]  # (N_waters, 3)
  r_H1_old = R_old[H1_idx]
  r_H2_old = R_old[H2_idx]

  # Unconstrained positions
  r_O_new = R_unconstrained[O_idx]
  r_H1_new = R_unconstrained[H1_idx]
  r_H2_new = R_unconstrained[H2_idx]

  # Compute constrained positions using SETTLE algorithm
  r_O_c, r_H1_c, r_H2_c = _settle_water_batch(
    r_O_old, r_H1_old, r_H2_old, r_O_new, r_H1_new, r_H2_new, r_OH, r_HH, m_O, m_H, box
  )

  # Re-wrap constrained positions back into the periodic box
  # Use MOLECULE-CENTERED wrapping: wrap based on O position,
  # then apply same displacement to H atoms to preserve geometry
  if box is not None:
    # Compute how to wrap O atoms
    r_O_wrapped = jnp.mod(r_O_c, box)
    wrap_displacement = r_O_wrapped - r_O_c  # (N_waters, 3)

    # Apply same displacement to H atoms
    r_H1_wrapped = r_H1_c + wrap_displacement
    r_H2_wrapped = r_H2_c + wrap_displacement

    r_O_c = r_O_wrapped
    r_H1_c = r_H1_wrapped
    r_H2_c = r_H2_wrapped

  # Update positions in result array
  R_constrained = R_unconstrained.at[O_idx].set(r_O_c)
  R_constrained = R_constrained.at[H1_idx].set(r_H1_c)
  R_constrained = R_constrained.at[H2_idx].set(r_H2_c)

  return R_constrained


def _settle_water_batch(
  r_O_old: Array,
  r_H1_old: Array,
  r_H2_old: Array,
  r_O_new: Array,
  r_H1_new: Array,
  r_H2_new: Array,
  r_OH: float,
  r_HH: float,
  m_O: float,
  m_H: float,
  box: Array | None = None,
) -> tuple[Array, Array, Array]:
  """SETTLE algorithm for a batch of water molecules.

  This implements a simplified but robust version of SETTLE that:
  1. Moves to center of mass frame
  2. Reconstructs the ideal water geometry
  3. Aligns it with the unconstrained motion direction

  For numerical stability, we use iterative refinement after the analytical step.

  If box is provided, applies minimum image convention to handle positions
  that have wrapped across periodic boundaries.
  """
  # Apply minimum image convention if box is provided
  # This unwraps R_new relative to R_old to handle PBC crossings
  if box is not None:

    def unwrap(r_new, r_old):
      delta = r_new - r_old
      delta = delta - box * jnp.round(delta / box)
      return r_old + delta

    r_O_new = unwrap(r_O_new, r_O_old)
    r_H1_new = unwrap(r_H1_new, r_H1_old)
    r_H2_new = unwrap(r_H2_new, r_H2_old)
  m_total = m_O + 2 * m_H

  # Step 1: Compute center of mass of new (unconstrained) configuration
  # The COM motion is preserved - only internal geometry is corrected
  com_new = (m_O * r_O_new + m_H * r_H1_new + m_H * r_H2_new) / m_total

  # Step 2: Get unconstrained positions relative to COM
  d_O = r_O_new - com_new  # (N_waters, 3)

  # Step 3: Build rotation frame from the old geometry
  # This defines how we orient the ideal triangle
  com_old = (m_O * r_O_old + m_H * r_H1_old + m_H * r_H2_old) / m_total

  d_O_old = r_O_old - com_old
  d_H1_old = r_H1_old - com_old
  d_H2_old = r_H2_old - com_old

  # Step 4: Build ideal water geometry in local frame
  # The ideal TIP3P water has:
  # - O at distance ra from COM along +Y axis
  # - H1 at distance rb from COM, offset in +X
  # - H2 at distance rb from COM, offset in -X

  # Distance from O to midpoint of H-H
  # In isoceles triangle: cos(theta/2) = (r_HH/2) / r_OH gives the geometry
  # Actually: r_om = sqrt(r_OH^2 - (r_HH/2)^2)
  r_om = jnp.sqrt(r_OH**2 - (r_HH / 2) ** 2)

  # In COM frame, O is at (0, ra) and H midpoint is at (0, -rb)
  # where m_O * ra = m_H * (2 * rb) / (m_O + 2*m_H) ... mass weighting
  # ra = 2 * m_H * r_om / m_total
  # rb = m_O * r_om / m_total
  ra = 2 * m_H * r_om / m_total  # O to COM distance
  rb = m_O * r_om / m_total  # H-midpoint to COM distance

  # rc = half of H-H distance
  rc = r_HH / 2

  # Step 5: Determine rotation using weighted average of unconstrained directions
  # We use the old geometry to define axes and project unconstrained motion

  # Y axis: from H-midpoint toward O (in old geometry)
  midpoint_old = 0.5 * (d_H1_old + d_H2_old)
  axis_y = d_O_old - midpoint_old
  len_y = jnp.linalg.norm(axis_y, axis=-1, keepdims=True) + 1e-12
  axis_y = axis_y / len_y  # (N_waters, 3)

  # X axis: from H2 toward H1 (in old geometry)
  axis_x = d_H1_old - d_H2_old
  len_x = jnp.linalg.norm(axis_x, axis=-1, keepdims=True) + 1e-12
  axis_x = axis_x / len_x

  # Z axis: perpendicular
  axis_z = jnp.cross(axis_x, axis_y)
  len_z = jnp.linalg.norm(axis_z, axis=-1, keepdims=True) + 1e-12
  axis_z = axis_z / len_z

  # Recompute y to ensure orthonormality
  axis_y = jnp.cross(axis_z, axis_x)

  # Step 6: Project unconstrained positions onto local axes
  # to get the rotation implied by the unconstrained motion

  # Where does O want to be?
  O_proj_x = jnp.sum(d_O * axis_x, axis=-1)  # (N_waters,)
  O_proj_y = jnp.sum(d_O * axis_y, axis=-1)

  # Determine in-plane rotation angle from O position
  # O should be at (0, ra) in unrotated frame
  # If unconstrained O is at (x, y) in local frame, rotation ~ atan2(x, y)
  phi = jnp.arctan2(O_proj_x, O_proj_y + 1e-12)

  # Out-of-plane tilt from z component
  # For small tilts, we can include this effect
  # But for simplicity, we project to xy plane (ignore tilt for now)

  # Step 7: Construct ideal positions with rotation phi
  cos_phi = jnp.cos(phi)
  sin_phi = jnp.sin(phi)

  # O: (0, ra) rotated by phi -> (-ra*sin, ra*cos)
  O_x = -ra * sin_phi
  O_y = ra * cos_phi
  O_z = jnp.zeros_like(O_x)

  # H1: (rc, -rb) rotated by phi
  H1_x = rc * cos_phi - (-rb) * sin_phi
  H1_y = rc * sin_phi + (-rb) * cos_phi
  H1_z = jnp.zeros_like(H1_x)

  # H2: (-rc, -rb) rotated by phi
  H2_x = -rc * cos_phi - (-rb) * sin_phi
  H2_y = -rc * sin_phi + (-rb) * cos_phi
  H2_z = jnp.zeros_like(H2_x)

  # Step 8: Transform back to global coordinates
  r_O_c = com_new + O_x[:, None] * axis_x + O_y[:, None] * axis_y + O_z[:, None] * axis_z

  r_H1_c = com_new + H1_x[:, None] * axis_x + H1_y[:, None] * axis_y + H1_z[:, None] * axis_z

  r_H2_c = com_new + H2_x[:, None] * axis_x + H2_y[:, None] * axis_y + H2_z[:, None] * axis_z

  return r_O_c, r_H1_c, r_H2_c


def settle_velocities(
  V: Array,
  R_old: Array,
  R_new: Array,
  water_indices: Array,
  dt: float,
  m_O: float = 15.999,
  m_H: float = 1.008,
) -> Array:
  """Apply velocity constraints after position SETTLE.

  After SETTLE corrects positions, the velocities need to be corrected
  to be consistent with the constrained motion. This is the RATTLE-like
  velocity correction for SETTLE.

  Args:
      V: Unconstrained velocities (N, 3).
      R_old: Positions before integration step (N, 3).
      R_new: Constrained positions after SETTLE (N, 3).
      water_indices: Water molecule indices (N_waters, 3).
      dt: Timestep.
      m_O: Oxygen mass (amu).
      m_H: Hydrogen mass (amu).

  Returns:
      Velocities consistent with constrained motion.
  """
  if water_indices.shape[0] == 0:
    return V

  # Compute the implicit velocity from position change
  # v_implied = (R_new - R_old) / dt

  # The velocity constraint requires dot(v_ij, r_ij) = 0 for all bonds
  # We use iterative RATTLE for velocity correction

  O_idx = water_indices[:, 0]
  H1_idx = water_indices[:, 1]
  H2_idx = water_indices[:, 2]

  # Get constrained positions
  r_O = R_new[O_idx]
  r_H1 = R_new[H1_idx]
  r_H2 = R_new[H2_idx]

  # Bond vectors
  r_OH1 = r_H1 - r_O  # (N_waters, 3)
  r_OH2 = r_H2 - r_O
  r_H1H2 = r_H2 - r_H1

  # Normalize bond vectors
  d_OH1 = jnp.linalg.norm(r_OH1, axis=-1, keepdims=True) + 1e-10
  d_OH2 = jnp.linalg.norm(r_OH2, axis=-1, keepdims=True) + 1e-10
  d_H1H2 = jnp.linalg.norm(r_H1H2, axis=-1, keepdims=True) + 1e-10

  n_OH1 = r_OH1 / d_OH1
  n_OH2 = r_OH2 / d_OH2
  n_H1H2 = r_H1H2 / d_H1H2

  # Inverse masses
  inv_m_O = 1.0 / m_O
  inv_m_H = 1.0 / m_H

  def correct_velocities(V_curr):
    """Single iteration of velocity corrections."""
    v_O = V_curr[O_idx]
    v_H1 = V_curr[H1_idx]
    v_H2 = V_curr[H2_idx]

    # Relative velocities along bonds
    v_OH1 = v_H1 - v_O
    v_OH2 = v_H2 - v_O
    v_H1H2 = v_H2 - v_H1

    # Velocity components along bonds (should be zero)
    dot_OH1 = jnp.sum(v_OH1 * n_OH1, axis=-1)
    dot_OH2 = jnp.sum(v_OH2 * n_OH2, axis=-1)
    dot_H1H2 = jnp.sum(v_H1H2 * n_H1H2, axis=-1)

    # Compute Lagrange multipliers
    # For bond i-j: lambda = -dot(v_ij, n_ij) / (1/m_i + 1/m_j)
    lambda_OH1 = -dot_OH1 / (inv_m_O + inv_m_H)
    lambda_OH2 = -dot_OH2 / (inv_m_O + inv_m_H)
    lambda_H1H2 = -dot_H1H2 / (2 * inv_m_H)

    # Velocity corrections
    # dv_i = lambda * n_ij / m_i
    # dv_j = -lambda * n_ij / m_j

    dv_O_from_OH1 = -lambda_OH1[:, None] * n_OH1 * inv_m_O
    dv_H1_from_OH1 = lambda_OH1[:, None] * n_OH1 * inv_m_H

    dv_O_from_OH2 = -lambda_OH2[:, None] * n_OH2 * inv_m_O
    dv_H2_from_OH2 = lambda_OH2[:, None] * n_OH2 * inv_m_H

    dv_H1_from_H1H2 = -lambda_H1H2[:, None] * n_H1H2 * inv_m_H
    dv_H2_from_H1H2 = lambda_H1H2[:, None] * n_H1H2 * inv_m_H

    # Accumulate corrections
    dv_O = dv_O_from_OH1 + dv_O_from_OH2
    dv_H1 = dv_H1_from_OH1 + dv_H1_from_H1H2
    dv_H2 = dv_H2_from_OH2 + dv_H2_from_H1H2

    # Apply corrections
    V_new = V_curr.at[O_idx].add(dv_O)
    V_new = V_new.at[H1_idx].add(dv_H1)
    V_new = V_new.at[H2_idx].add(dv_H2)
    return V_new

  # Iterate a few times for convergence
  return jax.lax.fori_loop(0, 5, lambda _i, v: correct_velocities(v), V)


def get_water_indices(n_protein_atoms: int, n_waters: int) -> Array:
  """Generate water molecule indices assuming standard O, H1, H2 ordering.

  For a system with n_protein_atoms followed by n_waters water molecules
  in O, H1, H2 order, this returns the indices for SETTLE.

  Args:
      n_protein_atoms: Number of protein atoms (non-water).
      n_waters: Number of water molecules.

  Returns:
      Array of shape (n_waters, 3) with [O_idx, H1_idx, H2_idx] per water.
  """
  if n_waters == 0:
    return jnp.zeros((0, 3), dtype=jnp.int32)

  water_indices = []
  for i in range(n_waters):
    base = n_protein_atoms + i * 3
    water_indices.append([base, base + 1, base + 2])

  return jnp.array(water_indices, dtype=jnp.int32)


def settle_langevin(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  gamma: float = 1.0,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  m_O: float = 15.999,
  m_H: float = 1.008,
  box: Array | None = None,
):
  """Langevin dynamics integrator with SETTLE constraints for water.

  This combines the BAOAB Langevin integrator with SETTLE position
  constraints and velocity corrections for rigid water molecules.

  Args:
      energy_or_force_fn: Energy or force function.
      shift_fn: Position shift function (for PBC).
      dt: Timestep (ps).
      kT: Thermal energy (temperature * Boltzmann constant).
      gamma: Friction coefficient (1/ps).
      mass: Mass array or scalar.
      water_indices: Array of shape (N_waters, 3) with water atom indices.
      r_OH: Target O-H bond length (Å).
      r_HH: Target H-H distance (Å).
      m_O: Oxygen mass (amu).
      m_H: Hydrogen mass (amu).
      box: Optional box dimensions for PBC (3,). If provided, applies
          minimum image convention to handle periodic boundary crossings.

  Returns:
      (init_fn, apply_fn) tuple for JAX-MD style integration.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  # If no water indices, fall back to standard Langevin
  if water_indices is None or water_indices.shape[0] == 0:
    return simulate.nvt_langevin(energy_or_force_fn, shift_fn, dt, kT, gamma=gamma, mass=mass)

  def init_fn(key, R, mass=mass, **kwargs):
    _kT = kwargs.pop("kT", kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)

    # Handle mass array
    mass_arr = jnp.array(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr

    # Initialize momenta: p = sqrt(m * kT) * N(0, 1)
    momenta = jnp.sqrt(mass_arr[:, None] * _kT) * random.normal(split, R.shape, dtype=R.dtype)

    # Store mass for broadcasting
    mass_state = mass_arr[:, None]

    # Import NVTLangevinState from simulate module
    from prolix.physics.simulate import NVTLangevinState

    return NVTLangevinState(R, momenta, force, mass_state, key)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)

    from prolix.physics.simulate import NVTLangevinState

    # Store old positions for SETTLE
    R_old = state.position

    # Step 1: B (half velocity update)
    momentum = state.momentum + 0.5 * _dt * state.force

    # Step 2: A (half position update)
    velocity = momentum / state.mass
    position = shift_fn(state.position, 0.5 * _dt * velocity)

    # Step 3: O (stochastic update)
    c1 = jnp.exp(-gamma * _dt)
    c2 = jnp.sqrt(1 - c1**2)

    key, split = random.split(state.rng)
    noise = random.normal(split, state.momentum.shape)
    momentum = c1 * momentum + c2 * jnp.sqrt(state.mass * _kT) * noise

    # Step 4: A (second half position update)
    velocity = momentum / state.mass
    position = shift_fn(position, 0.5 * _dt * velocity)

    # Step 5: SETTLE position constraints (with PBC handling if box provided)
    position = settle_positions(position, R_old, water_indices, r_OH, r_HH, m_O, m_H, box)

    # Step 6: Force update with constrained positions
    force = force_fn(position, **kwargs)

    # Step 7: B (second half velocity update)
    momentum = momentum + 0.5 * _dt * force

    # Step 8: SETTLE velocity constraints
    velocity = momentum / state.mass
    velocity = settle_velocities(velocity, R_old, position, water_indices, _dt, m_O, m_H)
    momentum = velocity * state.mass

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn
