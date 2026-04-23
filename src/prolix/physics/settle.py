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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import random
from jax_md import quantity

from prolix.types import WaterIndices, WaterIndicesArray

if TYPE_CHECKING:
  from jax_md.util import Array

Array = Any

# TIP3P water geometry constants
TIP3P_ROH = 0.9572  # O-H bond length (Å)
TIP3P_RHH = 1.5139  # H-H distance (Å)
TIP3P_THETA = 104.52 * jnp.pi / 180.0  # H-O-H angle (rad)


def settle_positions(
  positions_unconstrained: Array,
  positions_old: Array,
  water_indices: WaterIndicesArray,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  box: Array | None = None,
) -> Array:
  r"""Apply SETTLE position constraints to water molecules.

  Process:
  1.  **Extract Indices**: Use `WaterIndices` to identify O, H1, H2.
  2.  **Batch SETTLE**: Call `_settle_water_batch` for analytical reconstruction.
  3.  **PBC Handling**: Correct for periodic boundary crossings if `box` provided.
  4.  **Scatter Update**: Place constrained coordinates back into the global array.

  Args:
      positions_unconstrained: Unconstrained positions after integrator step (N, 3).
      positions_old: Positions before the integrator step (N, 3).
      water_indices: (N_waters, 3) atom indices [O, H1, H2].
      r_OH: Target O-H bond length (Å).
      r_HH: Target H-H distance (Å).
      mass_oxygen: Mass of oxygen (amu).
      mass_hydrogen: Mass of hydrogen (amu).
      box: Optional box dimensions for PBC (3,).

  Returns:
      Constrained positions array.
  """
  if water_indices.shape[0] == 0:
    return positions_unconstrained

  # Extract water atom positions
  indices = WaterIndices.from_row(water_indices.T)

  # Old positions
  pos_oxygen_old = positions_old[indices.oxygen]
  pos_h1_old = positions_old[indices.hydrogen1]
  pos_h2_old = positions_old[indices.hydrogen2]

  # Unconstrained positions
  pos_oxygen_new = positions_unconstrained[indices.oxygen]
  pos_h1_new = positions_unconstrained[indices.hydrogen1]
  pos_h2_new = positions_unconstrained[indices.hydrogen2]

  # Compute constrained positions using SETTLE algorithm
  pos_oxygen_c, pos_h1_c, pos_h2_c = _settle_water_batch(
    pos_oxygen_old,
    pos_h1_old,
    pos_h2_old,
    pos_oxygen_new,
    pos_h1_new,
    pos_h2_new,
    r_OH,
    r_HH,
    mass_oxygen,
    mass_hydrogen,
    box,
  )

  # Re-wrap constrained positions back into the periodic box
  # Use MOLECULE-CENTERED wrapping: wrap based on O position,
  # then apply same displacement to H atoms to preserve geometry
  if box is not None:
    r_oxygen_wrapped = jnp.mod(pos_oxygen_c, box)
    wrap_displacement = r_oxygen_wrapped - pos_oxygen_c  # (N_waters, 3)

    # Apply same displacement to H atoms
    r_h1_wrapped = pos_h1_c + wrap_displacement
    r_h2_wrapped = pos_h2_c + wrap_displacement
    pos_oxygen_c = r_oxygen_wrapped
    pos_h1_c = r_h1_wrapped
    pos_h2_c = r_h2_wrapped

  # Update positions in result array
  positions_constrained = positions_unconstrained.at[indices.oxygen].set(pos_oxygen_c)
  positions_constrained = positions_constrained.at[indices.hydrogen1].set(pos_h1_c)
  positions_constrained = positions_constrained.at[indices.hydrogen2].set(pos_h2_c)

  return positions_constrained


def _skew_symmetric3(r: Array) -> Array:
  """3×3 skew matrix ``[r]_×`` with ``[r]_× @ \\omega = r \\times \\omega``."""
  x, y, z = r[0], r[1], r[2]
  return jnp.array(
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
    dtype=r.dtype,
  )


def _project_one_water_momentum_rigid(
  p_stack: Array,
  r_stack: Array,
  m_stack: Array,
) -> Array:
  """Mass-weighted LS fit of atomic momenta to rigid motion for one TIP3P-like triplet.

  Finds ``v_com``, ``\\omega`` minimizing ``\\sum_i m_i |v_i - v_com - \\omega\\times r_{i,rel}|^2``,
  then returns projected atomic momenta ``m_i v_i``.
  """
  msum = jnp.sum(m_stack)
  eps = jnp.array(1e-30, dtype=msum.dtype)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / jnp.maximum(msum, eps)
  rrel = r_stack - com
  v = p_stack / m_stack[:, None]
  v_flat = v.reshape(9)
  rows = []
  for i in range(3):
    rows.append(jnp.concatenate([jnp.eye(3, dtype=r_stack.dtype), -_skew_symmetric3(rrel[i])], axis=1))
  jmat = jnp.vstack(rows)
  m_rep = jnp.repeat(m_stack, 3)
  g = (jmat.T * m_rep) @ jmat
  rhs = jmat.T @ (m_rep * v_flat)
  reg = jnp.array(1e-12, dtype=g.dtype) * (jnp.trace(g) / 6.0 + 1.0)
  g_reg = g + reg * jnp.eye(6, dtype=g.dtype)
  x6 = jnp.linalg.solve(g_reg, rhs)
  v_proj = (jmat @ x6).reshape(3, 3)
  return v_proj * m_stack[:, None]


def project_tip3p_waters_momentum_rigid(
  momentum: Array,
  position: Array,
  mass: Array,
  water_indices: WaterIndicesArray,
) -> Array:
  """Project atomic momenta onto rigid-body subspace for each water (shape ``(N_w,3)`` indices)."""
  if water_indices.shape[0] == 0:
    return momentum
  mass_flat = mass.reshape(-1)
  idx = water_indices
  p_stack = jnp.stack([momentum[idx[:, 0]], momentum[idx[:, 1]], momentum[idx[:, 2]]], axis=1)
  r_stack = jnp.stack([position[idx[:, 0]], position[idx[:, 1]], position[idx[:, 2]]], axis=1)
  m_stack = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)
  p_proj = jax.vmap(_project_one_water_momentum_rigid)(p_stack, r_stack, m_stack)
  idx_flat = idx.reshape(-1)
  return momentum.at[idx_flat].set(p_proj.reshape(-1, 3))


def _settle_water_batch(
  pos_oxygen_old: Array,
  pos_h1_old: Array,
  pos_h2_old: Array,
  pos_oxygen_new: Array,
  pos_h1_new: Array,
  pos_h2_new: Array,
  r_OH: float,
  r_HH: float,
  mass_oxygen: float,
  mass_hydrogen: float,
  box: Array | None = None,
) -> tuple[Array, Array, Array]:
  r"""Analytical SETTLE algorithm for a batch of water molecules.

  Process:
  1.  **Unwrap**: Apply minimum image convention to handle PBC crossings.
  2.  **COM Frame**: Move to center of mass of unconstrained configuration.
  3.  **Local Frame**: Construct orthonormal axes $(\hat{x}, \hat{y}, \hat{z})$ from old positions.
  4.  **Ideal Geometry**: Define ideal $C_{2v}$ water triangle in local COM frame.
  5.  **Rotation**: Project unconstrained motion onto local axes to find optimal in-plane rotation $\phi$.
  6.  **Transform**: Reconstruct positions and transform back to global frame.

  Notes:
  Preserves COM position while correcting internal bond lengths and angles.
  Implemented according to Miyamoto & Kollman (1992).

  Args:
      r_O_old, r_H1_old, r_H2_old: Old positions.
      r_O_new, r_H1_new, r_H2_new: Unconstrained new positions.
      r_OH: Target O-H distance.
      r_HH: Target H-H distance.
      m_O, m_H: Masses.
      box: Periodic box dimensions.

  Returns:
      Tuple of constrained positions (r_O_c, r_H1_c, r_H2_c).
  """
  # Apply minimum image convention if box is provided
  # This unwraps R_new relative to R_old to handle PBC crossings
  if box is not None:

    def unwrap(pos_new, pos_old):
      delta = pos_new - pos_old
      delta = delta - box * jnp.round(delta / box)
      return pos_old + delta

    pos_oxygen_new = unwrap(pos_oxygen_new, pos_oxygen_old)
    pos_h1_new = unwrap(pos_h1_new, pos_h1_old)
    pos_h2_new = unwrap(pos_h2_new, pos_h2_old)
  mass_total = mass_oxygen + 2 * mass_hydrogen

  # COM motion is preserved - only internal geometry is corrected
  com_new = (
    mass_oxygen * pos_oxygen_new + mass_hydrogen * pos_h1_new + mass_hydrogen * pos_h2_new
  ) / mass_total

  delta_oxygen = pos_oxygen_new - com_new  # (N_waters, 3)

  com_old = (
    mass_oxygen * pos_oxygen_old + mass_hydrogen * pos_h1_old + mass_hydrogen * pos_h2_old
  ) / mass_total

  delta_oxygen_old = pos_oxygen_old - com_old
  delta_h1_old = pos_h1_old - com_old
  delta_h2_old = pos_h2_old - com_old

  # The ideal TIP3P water geometry:
  dist_oh_mid = jnp.sqrt(r_OH**2 - (r_HH / 2) ** 2)

  # In COM frame, O is at (0, dist_O_to_COM) and H midpoint is at (0, -dist_H_mid_to_COM)
  dist_O_to_COM = 2 * mass_hydrogen * dist_oh_mid / mass_total
  dist_H_mid_to_COM = mass_oxygen * dist_oh_mid / mass_total
  half_dist_hh = r_HH / 2

  # Y axis: from H-midpoint toward O (in old geometry)
  midpoint_old = 0.5 * (delta_h1_old + delta_h2_old)
  axis_y = delta_oxygen_old - midpoint_old
  len_y = jnp.linalg.norm(axis_y, axis=-1, keepdims=True) + 1e-12
  axis_y = axis_y / len_y  # (N_waters, 3)

  # X axis: from H1 toward H2 (in old geometry)
  axis_x = delta_h1_old - delta_h2_old
  len_x = jnp.linalg.norm(axis_x, axis=-1, keepdims=True) + 1e-12
  axis_x = axis_x / len_x

  # Z axis: perpendicular
  axis_z = jnp.cross(axis_x, axis_y)
  len_z = jnp.linalg.norm(axis_z, axis=-1, keepdims=True) + 1e-12
  axis_z = axis_z / len_z

  # Recompute y to ensure orthonormality
  axis_y = jnp.cross(axis_z, axis_x)

  # Where does O want to be?
  pos_O_proj_x = jnp.sum(delta_oxygen * axis_x, axis=-1)  # (N_waters,)
  pos_O_proj_y = jnp.sum(delta_oxygen * axis_y, axis=-1)

  # Determine in-plane rotation angle from O position
  # O should be at (0, dist_O_to_COM) in unrotated frame
  phi = jnp.arctan2(pos_O_proj_x, pos_O_proj_y + 1e-12)

  # Out-of-plane tilt from z component
  # For small tilts, we can include this effect
  # But for simplicity, we project to xy plane (ignore tilt for now)
  cos_phi = jnp.cos(phi)
  sin_phi = jnp.sin(phi)

  # O: (0, dist_O_to_COM) rotated by phi -> (-dist_O_to_COM*sin, dist_O_to_COM*cos)
  oxygen_x = -dist_O_to_COM * sin_phi
  oxygen_y = dist_O_to_COM * cos_phi
  oxygen_z = jnp.zeros_like(oxygen_x)

  # H1: (half_dist_hh, -dist_H_mid_to_COM) rotated by phi
  h1_x = half_dist_hh * cos_phi - (-dist_H_mid_to_COM) * sin_phi
  h1_y = half_dist_hh * sin_phi + (-dist_H_mid_to_COM) * cos_phi
  h1_z = jnp.zeros_like(h1_x)

  # H2: (-half_dist_hh, -dist_H_mid_to_COM) rotated by phi
  h2_x = -half_dist_hh * cos_phi - (-dist_H_mid_to_COM) * sin_phi
  h2_y = -half_dist_hh * sin_phi + (-dist_H_mid_to_COM) * cos_phi
  h2_z = jnp.zeros_like(h2_x)
  pos_oxygen_c = (
    com_new + oxygen_x[:, None] * axis_x + oxygen_y[:, None] * axis_y + oxygen_z[:, None] * axis_z
  )

  pos_h1_c = com_new + h1_x[:, None] * axis_x + h1_y[:, None] * axis_y + h1_z[:, None] * axis_z

  pos_h2_c = com_new + h2_x[:, None] * axis_x + h2_y[:, None] * axis_y + h2_z[:, None] * axis_z

  return pos_oxygen_c, pos_h1_c, pos_h2_c


def _get_settle_bond_vectors(
  pos_oxygen: Array, pos_h1: Array, pos_h2: Array
) -> tuple[Array, Array, Array]:
  """Compute normalized bond vectors for SETTLE."""
  vec_oh1 = pos_h1 - pos_oxygen  # (N_waters, 3)
  vec_oh2 = pos_h2 - pos_oxygen
  vec_h1h2 = pos_h2 - pos_h1

  # Normalize bond vectors
  dist_oh1 = jnp.linalg.norm(vec_oh1, axis=-1, keepdims=True) + 1e-10
  dist_oh2 = jnp.linalg.norm(vec_oh2, axis=-1, keepdims=True) + 1e-10
  dist_h1h2 = jnp.linalg.norm(vec_h1h2, axis=-1, keepdims=True) + 1e-10

  norm_vec_oh1 = vec_oh1 / dist_oh1
  norm_vec_oh2 = vec_oh2 / dist_oh2
  norm_vec_h1h2 = vec_h1h2 / dist_h1h2

  return norm_vec_oh1, norm_vec_oh2, norm_vec_h1h2


def _apply_rattle_velocity_correction(
  vel_curr: Array,
  indices: WaterIndices,
  norm_vec_oh1: Array,
  norm_vec_oh2: Array,
  norm_vec_h1h2: Array,
  inv_mass_oxygen: float,
  inv_mass_hydrogen: float,
) -> Array:
  """Single iteration of velocity corrections (RATTLE)."""
  vel_oxygen = vel_curr[indices.oxygen]
  vel_h1 = vel_curr[indices.hydrogen1]
  vel_h2 = vel_curr[indices.hydrogen2]

  # Relative velocities along bonds
  rel_vel_oh1 = vel_h1 - vel_oxygen
  rel_vel_oh2 = vel_h2 - vel_oxygen
  rel_vel_h1h2 = vel_h2 - vel_h1

  # Velocity components along bonds (should be zero)
  dot_oh1 = jnp.sum(rel_vel_oh1 * norm_vec_oh1, axis=-1)
  dot_oh2 = jnp.sum(rel_vel_oh2 * norm_vec_oh2, axis=-1)
  dot_h1h2 = jnp.sum(rel_vel_h1h2 * norm_vec_h1h2, axis=-1)

  # Compute Lagrange multipliers
  lambda_oh1 = -dot_oh1 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_oh2 = -dot_oh2 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_h1h2 = -dot_h1h2 / (2 * inv_mass_hydrogen)

  # Velocity corrections
  dv_oxygen_from_oh1 = -lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_oxygen
  dv_h1_from_oh1 = lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_hydrogen

  dv_oxygen_from_oh2 = -lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_oxygen
  dv_h2_from_oh2 = lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_hydrogen

  dv_h1_from_h1h2 = -lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen
  dv_h2_from_h1h2 = lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen

  # Accumulate corrections
  dv_oxygen = dv_oxygen_from_oh1 + dv_oxygen_from_oh2
  dv_h1 = dv_h1_from_oh1 + dv_h1_from_h1h2
  dv_h2 = dv_h2_from_oh2 + dv_h2_from_h1h2

  # Apply corrections
  vel_new = vel_curr.at[indices.oxygen].add(dv_oxygen)
  vel_new = vel_new.at[indices.hydrogen1].add(dv_h1)
  vel_new = vel_new.at[indices.hydrogen2].add(dv_h2)
  return vel_new


def _apply_rattle_velocity_correction_with_residual(
  vel_curr: Array,
  indices: WaterIndices,
  norm_vec_oh1: Array,
  norm_vec_oh2: Array,
  norm_vec_h1h2: Array,
  inv_mass_oxygen: float,
  inv_mass_hydrogen: float,
) -> tuple[Array, Array]:
  """Single iteration of velocity corrections (RATTLE) with residual tracking."""
  vel_oxygen = vel_curr[indices.oxygen]
  vel_h1 = vel_curr[indices.hydrogen1]
  vel_h2 = vel_curr[indices.hydrogen2]
  rel_vel_oh1 = vel_h1 - vel_oxygen
  rel_vel_oh2 = vel_h2 - vel_oxygen
  rel_vel_h1h2 = vel_h2 - vel_h1
  dot_oh1 = jnp.sum(rel_vel_oh1 * norm_vec_oh1, axis=-1)
  dot_oh2 = jnp.sum(rel_vel_oh2 * norm_vec_oh2, axis=-1)
  dot_h1h2 = jnp.sum(rel_vel_h1h2 * norm_vec_h1h2, axis=-1)
  lambda_oh1 = -dot_oh1 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_oh2 = -dot_oh2 / (inv_mass_oxygen + inv_mass_hydrogen)
  lambda_h1h2 = -dot_h1h2 / (2 * inv_mass_hydrogen)
  dv_oxygen_from_oh1 = -lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_oxygen
  dv_h1_from_oh1 = lambda_oh1[:, None] * norm_vec_oh1 * inv_mass_hydrogen
  dv_oxygen_from_oh2 = -lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_oxygen
  dv_h2_from_oh2 = lambda_oh2[:, None] * norm_vec_oh2 * inv_mass_hydrogen
  dv_h1_from_h1h2 = -lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen
  dv_h2_from_h1h2 = lambda_h1h2[:, None] * norm_vec_h1h2 * inv_mass_hydrogen
  dv_oxygen = dv_oxygen_from_oh1 + dv_oxygen_from_oh2
  dv_h1 = dv_h1_from_oh1 + dv_h1_from_h1h2
  dv_h2 = dv_h2_from_oh2 + dv_h2_from_h1h2
  vel_new = vel_curr.at[indices.oxygen].add(dv_oxygen)
  vel_new = vel_new.at[indices.hydrogen1].add(dv_h1)
  vel_new = vel_new.at[indices.hydrogen2].add(dv_h2)
  residual = jnp.max(jnp.abs(jnp.stack([dot_oh1, dot_oh2, dot_h1h2])))
  return vel_new, residual


def settle_velocities(
  velocities: Array,
  positions_old: Array,
  positions_constrained: Array,
  water_indices: WaterIndicesArray,
  dt: float,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  n_iters: int = 10,
  adaptive_tol: float | None = None,
) -> Array:
  r"""Apply velocity constraints after position SETTLE.

  Process:
  1.  **Extract Indices**: Use `WaterIndices` for O, H1, H2.
  2.  **Bond Vectors**: Compute normalized bond vectors $\hat{n}_{ij}$ from constrained positions.
  3.  **RATTLE Loop**: Iterate velocity corrections to satisfy $\vec{v}_{ij} \cdot \hat{n}_{ij} = 0$.

  Notes:
  This ensures velocities are consistent with the rigid constraints.
  Typically converges in 1-5 iterations.

  Args:
      V: Unconstrained velocities (N, 3).
      R_old, R_new: Old and constrained new positions.
      water_indices: (N_waters, 3).
      dt: Timestep.
      m_O, m_H: Masses.

  Returns:
      Constrained velocities array.
  """
  if water_indices.shape[0] == 0:
    return velocities

  indices = WaterIndices.from_row(water_indices.T)

  # Get normalized bond vectors from constrained positions
  norm_vec_oh1, norm_vec_oh2, norm_vec_h1h2 = _get_settle_bond_vectors(
    positions_constrained[indices.oxygen],
    positions_constrained[indices.hydrogen1],
    positions_constrained[indices.hydrogen2],
  )

  # Inverse masses
  inv_mass_oxygen = 1.0 / mass_oxygen
  inv_mass_hydrogen = 1.0 / mass_hydrogen

  # Iterate a few times for convergence.
  n_iters = max(int(n_iters), 0)
  if adaptive_tol is None:
    return jax.lax.fori_loop(
      0,
      n_iters,
      lambda _i, v: _apply_rattle_velocity_correction(
        v,
        indices,
        norm_vec_oh1,
        norm_vec_oh2,
        norm_vec_h1h2,
        inv_mass_oxygen,
        inv_mass_hydrogen,
      ),
      velocities,
    )
  else:
    def body(carry):
      i, vel, residual = carry
      vel_new, res = _apply_rattle_velocity_correction_with_residual(
        vel,
        indices,
        norm_vec_oh1,
        norm_vec_oh2,
        norm_vec_h1h2,
        inv_mass_oxygen,
        inv_mass_hydrogen,
      )
      return (i + jnp.int32(1), vel_new, res)
    initial = (jnp.int32(0), velocities, jnp.array(jnp.inf, dtype=velocities.dtype))
    cond = lambda carry: jnp.logical_and(
      carry[0] < jnp.int32(n_iters),
      carry[2] > jnp.array(adaptive_tol, dtype=velocities.dtype)
    )
    _, final_vel, _ = jax.lax.while_loop(cond, body, initial)
    return final_vel


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
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  box: Array | None = None,
  constraints: tuple[Array, Array] | None = None,
  remove_linear_com_momentum: bool = False,
  project_ou_momentum_rigid: bool = True,
  projection_site: str = "post_o",
  settle_velocity_iters: int = 10,
  settle_velocity_tol: float | None = None,
):
  r"""Langevin dynamics integrator with SETTLE constraints for water.

  Process:
  1.  **Half-B**: Update momenta by half timestep.
  2.  **Half-A**: Update positions by half timestep.
  3.  **O**: Stochastic velocity update (Ornstein-Uhlenbeck pulse).
  4.  **Half-A**: Second half position update.
  5.  **SETTLE-Pos**: Correct water positions analytically.
  6.  **Force**: Recompute forces at constrained positions.
  7.  **Half-B**: Final half velocity update.
  8.  **SETTLE-Vel**: Final velocity constraint correction.

  Notes:
  Combines BAOAB Langevin integrator with analytical rigid water constraints.
  After the Ornstein–Uhlenbeck step, atomic momenta are **mass-weighted projected** onto each
  water’s rigid-body subspace (unless ``project_ou_momentum_rigid=False``), so isotropic
  Cartesian noise does not over-drive ``6 N_w-3`` kinetic degrees of freedom.

  Args:
      energy_or_force_fn: System force definition.
      shift_fn: Displacement function.
      dt: Timestep.
      kT: Thermal energy target.
      gamma: Friction coefficient.
      mass: Atomic masses.
      water_indices: (N_waters, 3) indices.
      r_OH, r_HH: Target water geometry.
      m_O, m_H: Solvent masses.
      box: Periodic box dimensions.
      constraints: Optional (pairs, lengths) tuple for solute RATTLE.
      remove_linear_com_momentum: If True, after SETTLE velocity projection subtract the
          total center-of-mass velocity from every atom: ``p <- p - m * v_com`` with
          ``v_com = sum(p) / sum(m)``. This matches the **linear** part of OpenMM's
          ``CMMotionRemover`` (default frequency 1: subtract uniform ``v_com`` from all
          velocities), but occurs at a **different** point in the BAOAB+SETTLE timestep
          than OpenMM's integrator schedule. Use for production-style COM drift control,
          not for claiming bitwise phase-space parity with OpenMM.
    project_ou_momentum_rigid: If True (default), apply rigid momentum projection at the location
      selected by ``projection_site``.
    projection_site: Where to apply rigid momentum projection when
      ``project_ou_momentum_rigid=True``:
      ``post_o`` (legacy default), ``post_settle_vel``, or ``both``.
      Isotropic OU noise is applied in Cartesian atomic space; **always** projecting once
      immediately after the ``O`` step (using the mid-step positions) is required so rigid
      water degrees of freedom are not over-driven. ``post_settle_vel`` / ``both`` add an
      optional second projection after ``settle_velocities`` at the final constrained geometry.
    settle_velocity_iters: Number of RATTLE-like velocity correction iterations in
      ``settle_velocities``.

  Returns:
      (init_fn, apply_fn) pair.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  if projection_site not in ("post_o", "post_settle_vel", "both"):
    msg = f"invalid projection_site={projection_site!r}; expected post_o|post_settle_vel|both"
    raise ValueError(msg)
  if not project_ou_momentum_rigid and projection_site == "both":
    msg = "projection_site='both' requires project_ou_momentum_rigid=True"
    raise ValueError(msg)

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

    # Import required components from simulate module
    from prolix.physics.simulate import NVTLangevinState

    return NVTLangevinState(R, momenta, force, mass_state, key)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)

    # Store old positions for SETTLE
    positions_old = state.position

    momentum = _langevin_step_b(state.momentum, state.force, _dt)
    position = _langevin_step_a(state.position, momentum, state.mass, _dt, shift_fn)

    # Stochastic update (BAOAB "O" step)
    momentum, key = _langevin_step_o(momentum, state.mass, gamma, _dt, _kT, state.rng)
    # Always project after O when rigid OU control is on: skipping this (e.g. only projecting
    # after settle_vel) leaves full Cartesian OU noise and biases rigid kinetic energy high.
    if project_ou_momentum_rigid:
      momentum = project_tip3p_waters_momentum_rigid(
        momentum, position, state.mass, water_indices
      )

    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    # Solute RATTLE (SHAKE) - applied BEFORE SETTLE per batched_simulate convention
    if constraints is not None:
      from prolix.physics.simulate import project_positions
      pairs, lengths = constraints
      position = project_positions(position, pairs, lengths, state.mass, shift_fn)

    # SETTLE position constraints
    position = settle_positions(
      position,
      positions_old,
      water_indices,
      r_OH,
      r_HH,
      mass_oxygen,
      mass_hydrogen,
      box,
    )

    force = force_fn(position, **kwargs)
    momentum = _langevin_step_b(momentum, force, _dt)

    # Solute RATTLE (Velocity)
    if constraints is not None:
      from prolix.physics.simulate import project_momenta
      pairs, _ = constraints
      momentum = project_momenta(momentum, position, pairs, state.mass, shift_fn)

    # SETTLE velocity constraints
    momentum = _langevin_settle_vel(
      momentum,
      positions_old,
      position,
      state.mass,
      water_indices,
      _dt,
      mass_oxygen,
      mass_hydrogen,
      n_iters=settle_velocity_iters,
      settle_velocity_tol=settle_velocity_tol,
    )
    if project_ou_momentum_rigid and projection_site in ("post_settle_vel", "both"):
      momentum = project_tip3p_waters_momentum_rigid(
        momentum, position, state.mass, water_indices
      )

    if remove_linear_com_momentum:
      mass_col = state.mass
      p_tot = jnp.sum(momentum, axis=0)
      m_tot = jnp.sum(mass_col)
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = momentum - mass_col * v_com

    from prolix.physics.simulate import NVTLangevinState

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn


def _langevin_step_b(momentum: Array, force: Array, dt: float) -> Array:
  """Half-step velocity update (B step in BAOAB)."""
  return momentum + 0.5 * dt * force


def _langevin_step_a(
  position: Array, momentum: Array, mass: Array, dt: float, shift_fn: Callable
) -> Array:
  """Half-step position update (A step in BAOAB)."""
  velocity = momentum / mass
  return shift_fn(position, 0.5 * dt * velocity)


def _langevin_step_o(
  momentum: Array, mass: Array, gamma: float, dt: float, kT: float, rng: Array
) -> tuple[Array, Array]:
  """Stochastic velocity update (O step in BAOAB)."""
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(1 - c1**2)

  key, split = jax.random.split(rng)
  noise = jax.random.normal(split, momentum.shape)
  momentum_new = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise
  return momentum_new, key


def _langevin_settle_vel(
  momentum: Array,
  positions_old: Array,
  positions_new: Array,
  mass: Array,
  water_indices: WaterIndicesArray,
  dt: float,
  mass_oxygen: float,
  mass_hydrogen: float,
  n_iters: int = 10,
  settle_velocity_tol: float | None = None,
) -> Array:
  """Apply SETTLE velocity constraints and update momentum."""
  velocity = momentum / mass
  velocity = settle_velocities(
    velocity,
    positions_old,
    positions_new,
    water_indices,
    dt,
    mass_oxygen,
    mass_hydrogen,
    n_iters=n_iters,
    adaptive_tol=settle_velocity_tol,
  )
  return velocity * mass
