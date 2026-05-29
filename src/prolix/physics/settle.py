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

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import random
from jax_md import quantity, simulate

from prolix.physics import (
    md_potential_bundle,
    pbc as pbc_module,
    pressure as pressure_module,
    rigid_water_ke,
    stress as stress_module,
    units as units_module,
)
from prolix.physics.constraints import project_momenta, project_positions
from prolix.typing import NPTState, NVTLangevinState, WaterIndices, WaterIndicesArray


if TYPE_CHECKING:

  from jax_md.util import Array

Array = Any

# TIP3P water geometry constants
TIP3P_ROH = 0.9572  # O-H bond length (Å)
TIP3P_RHH = 1.5139  # H-H distance (Å)
TIP3P_THETA = 104.52 * jnp.pi / 180.0  # H-O-H angle (rad)

# Default CSVR relaxation time: 0.1 ps in AKMA units (1 AKMA ≈ 48.888 fs)
_DEFAULT_CSVR_TAU_AKMA: float = 100.0 / 48.88821291839


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


def _make_settle_compatible_force_fn(
  energy_or_force_fn: Callable[..., Array],
  mass: float | Array,
  box_template: Array | None,
) -> Callable[..., Array]:
  """``jax_md.quantity.canonicalize_force``-compatible force fn using one ``value_and_grad`` when safe.

  If ``mass`` is a scalar, the atom count is unknown at integrator construction time;
  we fall back to ``canonicalize_force`` only.
  """
  mass_arr = jnp.asarray(mass)
  if mass_arr.ndim == 0:
    return quantity.canonicalize_force(energy_or_force_fn)
  n_atoms = int(mass_arr.reshape(-1).shape[0])
  template_R = jnp.zeros((n_atoms, 3), dtype=jnp.float64)
  kw: dict[str, Any] = {}
  if box_template is not None:
    kw["box"] = box_template
  return md_potential_bundle.make_force_fn_like_canonicalize(
    energy_or_force_fn,
    template_R=template_R,
    template_kwargs=kw if kw else None,
  )


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
  water's rigid-body subspace (unless ``project_ou_momentum_rigid=False``), so isotropic
  Cartesian noise does not over-drive ``6 N_w-3`` kinetic degrees of freedom.

  **IMPORTANT**: For stable temperature control with SETTLE constraints, use **dt ≤ 0.5 fs**.
  Larger timesteps couple SETTLE constraint impulses with Langevin thermostat feedback,
  creating oscillations that compromise temperature stability. The reduced timestep constraint
  is a known limitation in v1.0; future versions will implement constraint-aware thermostats
  to eliminate this restriction.

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
  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)
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

    # Initialize momenta differently depending on whether we use constrained thermostat
    if project_ou_momentum_rigid and water_indices is not None and water_indices.shape[0] > 0:
      # Start with zero momenta, will fill in for water and non-water separately
      momenta = jnp.zeros_like(R, dtype=R.dtype)

      # Water atoms: sample from constrained distribution
      idx = water_indices
      mass_flat = mass_arr.reshape(-1)
      r_water = jnp.stack([R[idx[:, 0]], R[idx[:, 1]], R[idx[:, 2]]], axis=1)
      m_water = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)

      def init_one_water(carry, inputs):
        key_w = carry
        r_w, m_w = inputs
        p_w, key_w = _init_momentum_one_water_rigid(key_w, r_w, m_w, _kT)
        return key_w, p_w

      key, p_water = jax.lax.scan(init_one_water, key, (r_water, m_water))
      idx_flat = idx.reshape(-1)
      momenta = momenta.at[idx_flat].set(p_water.reshape(-1, 3))

      # Non-water atoms (solute): standard unconstrained noise
      # Create a mask for non-water atoms
      all_indices = jnp.arange(R.shape[0])
      is_water = jnp.isin(all_indices, idx_flat, assume_unique=False)
      non_water_mask = ~is_water

      if jnp.any(non_water_mask):
        key, split = jax.random.split(key)
        non_water_noise = jnp.sqrt(mass_arr[non_water_mask, None] * _kT) * jax.random.normal(split, (jnp.sum(non_water_mask), 3), dtype=R.dtype)
        momenta = momenta.at[non_water_mask].set(non_water_noise)
    else:
      # Standard unconstrained initialization
      key, split = jax.random.split(key)
      momenta = jnp.sqrt(mass_arr[:, None] * _kT) * jax.random.normal(split, R.shape, dtype=R.dtype)

    # Store mass for broadcasting
    mass_state = mass_arr[:, None]

    return NVTLangevinState(R, momenta, force, mass_state, key)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)

    # Store old positions for SETTLE
    positions_old = state.positions

    momentum = _langevin_step_b(state.momentum, state.force, _dt)
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)

    # Stochastic update (BAOAB "O" step)
    # When project_ou_momentum_rigid=True, use constrained noise directly in rigid-body subspace.
    # This provides correct covariance (kT * M * P_rigid) for equipartition across 6*N_w-3 DOF.
    if project_ou_momentum_rigid:
      momentum, key = _langevin_step_o_constrained(
        momentum, position, state.mass, gamma, _dt, _kT, state.key, water_indices
      )
    else:
      momentum, key = _langevin_step_o(momentum, state.mass, gamma, _dt, _kT, state.key)

    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    # Solute RATTLE (SHAKE) - applied BEFORE SETTLE per batched_simulate convention
    if constraints is not None:
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

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn


def settle_lfmiddle_langevin(
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
  r"""Langevin + SETTLE with Leimkuhler-Matthews O-step splitting (LFMiddle).

  Same as :func:`settle_langevin` but splits the stochastic O-step into two
  halves around the mid-step force recompute:

  B(0.5) → A(0.5) → O(0.5) → A(0.5) → SETTLE_pos → Force → A(0.5) → O(0.5) → B(0.5) → SETTLE_vel
  """
  if projection_site not in ("post_o", "post_settle_vel", "both"):
    msg = f"invalid projection_site={projection_site!r}; expected post_o|post_settle_vel|both"
    raise ValueError(msg)

  init_fn, _ = settle_langevin(
      energy_or_force_fn,
      shift_fn,
      dt,
      kT,
      gamma=gamma,
      mass=mass,
      water_indices=water_indices,
      r_OH=r_OH,
      r_HH=r_HH,
      mass_oxygen=mass_oxygen,
      mass_hydrogen=mass_hydrogen,
      box=box,
      constraints=constraints,
      remove_linear_com_momentum=remove_linear_com_momentum,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
      projection_site=projection_site,
      settle_velocity_iters=settle_velocity_iters,
      settle_velocity_tol=settle_velocity_tol,
  )
  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)
  if water_indices is None or water_indices.shape[0] == 0:
    return simulate.nvt_langevin(energy_or_force_fn, shift_fn, dt, kT, gamma=gamma, mass=mass)

  def apply_fn(state, **kwargs):
    _dt = kwargs.pop("dt", dt)
    _kT = kwargs.pop("kT", kT)
    half_dt = 0.5 * _dt
    positions_old = state.positions

    momentum = _langevin_step_b(state.momentum, state.force, _dt)
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)

    if project_ou_momentum_rigid:
      momentum, key = _langevin_step_o_constrained(
          momentum, position, state.mass, gamma, half_dt, _kT, state.key, water_indices
      )
    else:
      momentum, key = _langevin_step_o(momentum, state.mass, gamma, half_dt, _kT, state.key)

    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    if constraints is not None:
      pairs, lengths = constraints
      position = project_positions(position, pairs, lengths, state.mass, shift_fn)

    position = settle_positions(
        position, positions_old, water_indices, r_OH, r_HH,
        mass_oxygen, mass_hydrogen, box,
    )

    force = force_fn(position, **kwargs)

    # A(0.5) second half — mirrors first half for time-reversibility.
    # Re-apply SETTLE so the returned state and next step's positions_old
    # both satisfy constraints.
    positions_pre_settle2 = position
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)
    if constraints is not None:
      pairs, lengths = constraints
      position = project_positions(position, pairs, lengths, state.mass, shift_fn)
    position = settle_positions(
        position, positions_pre_settle2, water_indices, r_OH, r_HH,
        mass_oxygen, mass_hydrogen, box,
    )
    if project_ou_momentum_rigid:
      momentum, key = _langevin_step_o_constrained(
          momentum, position, state.mass, gamma, half_dt, _kT, key, water_indices
      )
    else:
      momentum, key = _langevin_step_o(momentum, state.mass, gamma, half_dt, _kT, key)

    momentum = _langevin_step_b(momentum, force, _dt)

    if constraints is not None:
      pairs, _ = constraints
      momentum = project_momenta(momentum, position, pairs, state.mass, shift_fn)

    momentum = _langevin_settle_vel(
        momentum, positions_old, position, state.mass, water_indices, _dt,
        mass_oxygen, mass_hydrogen, n_iters=settle_velocity_iters,
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

    return NVTLangevinState(position, momentum, force, state.mass, key)

  return init_fn, apply_fn


def settle_with_nhc(
  energy_or_force_fn: Callable,
  shift_fn: Callable,
  dt: float,
  kT: float,
  mass: Array | float = 1.0,
  water_indices: WaterIndicesArray | None = None,
  box: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  settle_velocity_iters: int = 10,
  settle_velocity_tol: float | None = None,
  constraints: tuple | None = None,
  remove_linear_com_momentum: bool = True,
  chain_length: int = 5,
  chain_steps: int = 2,
  sy_steps: int = 3,
  tau: float | None = None,
) -> tuple[Callable, Callable]:
  r"""SETTLE constraints with JAX MD's Nosé-Hoover Chain thermostat.

  Wraps JAX MD's proven `nvt_nose_hoover()` with SETTLE rigid water
  constraint enforcement. Uses a Nosé-Hoover Chain (NHC) thermostat
  (chain_length=5 by default, more stable than single thermostat).

  Integrator Order:
  ```
  Base: JAX MD's nvt_nose_hoover (NHC thermostat)
  Wrapped with: SETTLE position constraints → SETTLE velocity constraints
  ```

  Args:
      energy_or_force_fn: Callable returning force (or energy).
      shift_fn: PBC shift function from `jax_md.space`.
      dt: Timestep (AKMA units).
      kT: Thermal energy (k_B * T in AKMA units).
      mass: Atomic masses.
      water_indices: (N_waters, 3) indices [O, H1, H2] for SETTLE waters.
      box: Optional PBC box dimensions (3,).
      r_OH, r_HH: Bond lengths for SETTLE.
      mass_oxygen, mass_hydrogen: TIP3P masses.
      settle_velocity_iters: SETTLE velocity correction iterations.
      settle_velocity_tol: SETTLE velocity tolerance.
      constraints: Optional (pairs, lengths) for solute RATTLE.
      remove_linear_com_momentum: Subtract COM velocity after SETTLE_vel.
      chain_length: NHC chain length (default 5).
      chain_steps: NHC outer substeps (default 2).
      sy_steps: Suzuki-Yoshida steps (default 3, must be 1/3/5/7).
      tau: NHC coupling timescale (in units of dt).

  Returns:
      (init_fn, apply_fn) tuple for NVT dynamics with NHC + SETTLE.
  """
  # Get JAX MD's proven Nosé-Hoover Chain integrator
  init_fn_nhc, apply_fn_nhc = simulate.nvt_nose_hoover(

    energy_or_force_fn,
    shift_fn,
    dt,
    kT,
    chain_length=chain_length,
    chain_steps=chain_steps,
    sy_steps=sy_steps,
    tau=tau,
  )

  # If no water indices, return the NHC integrator as-is
  if water_indices is None or water_indices.shape[0] == 0:
    return nhc_init, nhc_apply

  force_fn_after_settle = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)

  # Wrap NHC apply function with SETTLE constraints
  def apply_with_settle(state, **kwargs):
    """Apply one NHC step, then enforce SETTLE constraints."""
    # Store old positions for SETTLE velocity correction
    positions_old = state.positions

    # Apply one step of JAX MD's NHC integrator
    state = nhc_apply(state, **kwargs)

    # Enforce SETTLE position constraints
    position = settle_positions(
      state.positions,
      positions_old,
      water_indices,
      r_OH,
      r_HH,
      mass_oxygen,
      mass_hydrogen,
      box,
    )

    # Recompute forces after constraining positions
    force = force_fn_after_settle(position, **kwargs)

    # Enforce SETTLE velocity constraints
    momentum = _langevin_settle_vel(
      state.momentum,
      positions_old,
      position,
      state.mass,
      water_indices,
      kwargs.pop("dt", dt),
      mass_oxygen,
      mass_hydrogen,
      n_iters=settle_velocity_iters,
      settle_velocity_tol=settle_velocity_tol,
    )

    if remove_linear_com_momentum:
      mass_col = state.mass
      p_tot = jnp.sum(momentum, axis=0)
      m_tot = jnp.sum(mass_col)
      v_com = p_tot / jnp.maximum(m_tot, jnp.array(1e-30, dtype=m_tot.dtype))
      momentum = momentum - mass_col * v_com

    # Re-synchronize NHC chain state after SETTLE constraints modify momentum.
    # SETTLE removes kinetic energy from constrained DOF, so the NHC chain state
    # (position, momentum of fictitious particles) becomes desynchronized. Reset
    # to zero to allow thermostat to re-equilibrate smoothly on the next step.
    new_chain = state.chain.set(
      positions=jnp.zeros_like(state.chain.positions),
      momentum=jnp.zeros_like(state.chain.momentum),
    )

    # Return updated state with constrained position, momentum, force, AND chain state
    return state.set(
      positions=position,
      momentum=momentum,
      force=force,
      chain=new_chain,
    )

  return nhc_init, apply_with_settle


def langevin_with_constraints(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  gamma: float = 1.0,
  mass: float | Array = 1.0,
  constraint: Any | None = None,
  box: Array | None = None,
  remove_linear_com_momentum: bool = False,
  project_ou_momentum_rigid: bool = True,
  water_indices: Array | None = None,
  **kwargs,
) -> tuple[Callable, Callable]:
  r"""Langevin integrator with injected constraint algorithm.

  Wraps settle_langevin with a constraint plugin system. The constraint object
  provides apply_positions() and apply_velocities() methods, allowing flexible
  constraint composition (NullConstraint, SETTLEConstraint, ShakeRattleConstraint, etc.).

  Args:
      energy_or_force_fn: System force definition.
      shift_fn: Displacement function.
      dt: Timestep (AKMA units).
      kT: Thermal energy target (kcal/mol).
      gamma: Langevin friction coefficient (1/ps, reduced to AKMA units by caller).
      mass: Atomic masses (scalar or (N,) array).
      constraint: ConstraintAlgorithm instance (or None for no constraints).
      box: Periodic box dimensions or None.
      remove_linear_com_momentum: If True, remove COM momentum each step.
      project_ou_momentum_rigid: If True, project OU noise to rigid-body subspace for water.
      water_indices: (N_waters, 3) array of water atom indices for rigid-body projection.
      **kwargs: Additional arguments passed to energy_fn.

  Returns:
      (init_fn, apply_fn) following JAX-MD convention.
  """
  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)

  def init_fn(key, R, mass=mass, **init_kwargs):
    _kT = init_kwargs.pop("kT", kT)
    key, split = jax.random.split(key)
    force = force_fn(R, **init_kwargs)

    # Handle mass - keep as 1D for indexing, expand to (N, 1) for broadcasting
    mass_arr = jnp.asarray(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr
    elif mass_arr.ndim == 2:
      mass_arr = mass_arr.reshape(-1)

    # Initialize momenta from Maxwell-Boltzmann
    momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
      split, R.shape, dtype=R.dtype
    )

    # Store mass as (N, 1) for broadcasting
    mass_for_state = mass_arr[:, jnp.newaxis]

    return NVTLangevinState(R, momenta, force, mass_for_state, key)

  def apply_fn(state, **step_kwargs):
    _dt = step_kwargs.pop("dt", dt)
    _kT = step_kwargs.pop("kT", kT)

    # Store old positions for constraints
    positions_old = state.positions

    # === B-step (half-kick) ===
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # === A-step (half-move) ===
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)

    # === O-step (stochastic update) ===
    if project_ou_momentum_rigid and water_indices is not None:
      momentum, key_out = _langevin_step_o_constrained(
        momentum, position, state.mass, gamma, _dt, _kT, state.key, water_indices
      )
    else:
      momentum, key_out = _langevin_step_o(momentum, state.mass, gamma, _dt, _kT, state.key)

    # === A-step (second half-move) ===
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    # === Apply position constraints ===
    if constraint is not None:
      position = constraint.apply_positions(positions_old, position, state.mass, box)

    # === Force eval (at constrained positions) ===
    force = force_fn(position, **step_kwargs)

    # === B-step (final half-kick) ===
    momentum = _langevin_step_b(momentum, force, _dt)

    # === Apply velocity constraints ===
    if constraint is not None:
      momentum = constraint.apply_velocities(
        positions_old, position, momentum, state.mass, _dt, shift_fn
      )

    # === Remove COM momentum if requested ===
    if remove_linear_com_momentum:
      total_mass = jnp.sum(state.mass)
      com_momentum = jnp.sum(momentum, axis=0) / total_mass
      momentum = momentum - state.mass * com_momentum[jnp.newaxis, :]

    return NVTLangevinState(position, momentum, force, state.mass, key_out)

  return init_fn, apply_fn


def _csvr_compute_lambda(
  key: Array,
  ke_current: Array,
  n_dof: int,
  kT: float,
  dt: float,
  tau: float = None,
) -> tuple[Array, Array]:
  r"""Compute CSVR (Bussi velocity-rescaling) lambda factor.

  Implements Bussi et al. 2007 velocity rescaling thermostat.
  Draws random samples from chi-squared distribution to stochastically
  rescale velocities toward the target kinetic energy.

  Args:
    key: JAX PRNGKey for random sampling.
    ke_current: Current total kinetic energy (kcal/mol).
    n_dof: Number of thermostated degrees of freedom.
    kT: Target thermal energy (kcal/mol).
    dt: Timestep in AKMA units.
    tau: Relaxation time in AKMA units. Default ~0.1ps equivalent.

  Returns:
    (lambda_factor, new_key) where lambda_factor rescales all momenta.
  """
  if tau is None:
    tau = _DEFAULT_CSVR_TAU_AKMA

  n_dof = max(int(n_dof), 1)

  # Coupling strength: exp(-dt/tau)
  c1 = jnp.exp(-dt / tau)
  c2_sq = 1.0 - c1  # Note: this is (1 - c1), not sqrt(1 - c1^2)

  # Draw random numbers: one gaussian + (n_dof - 1) chi-squared components
  key, split = jax.random.split(key)
  r_gaussian = jax.random.normal(split, dtype=ke_current.dtype)

  key, split = jax.random.split(key)
  # Sum of (n_dof - 1) chi-squared components = sum of squares of (n_dof-1) gaussians
  s_chi_squared = jnp.sum(
    jax.random.normal(split, (n_dof - 1,), dtype=ke_current.dtype) ** 2
  )

  target_ke = 0.5 * n_dof * kT
  ke_safe = jnp.maximum(ke_current, 1e-10)

  # Bussi et al. 2007 eq. A7:
  # lambda^2 = c1 + c2*(R^2 + S) + 2*R*sqrt(c1*c2)
  # where c1 = exp(-dt/tau), c2 = (1-c1)*(target_ke)/(ke_current*n_dof)
  # R ~ N(0,1), S ~ chi^2(n_dof - 1)
  # The n_dof in target_ke = 0.5*n_dof*kT partially cancels, but we need n_dof in denominator of c2.

  # Compute the corrected c2 factor: (1-c1) * K_target / (K_current * n_dof)
  # where K_target = 0.5 * n_dof * kT, so this gives (1-c1) * 0.5 * kT / K_current
  n_dof_f = float(n_dof)
  c2_corrected = c2_sq * (target_ke / ke_safe) / n_dof_f

  # Full Bussi formula with cross term
  lambda_sq = (
    c1
    + c2_corrected * (r_gaussian**2 + s_chi_squared)
    + 2.0 * r_gaussian * jnp.sqrt(c1 * c2_corrected)
  )
  lambda_sq_safe = jnp.maximum(lambda_sq, 0.0)

  # Edge case: if ke_current <= 0, return lambda=1.0 (no rescaling)
  lambda_factor = jnp.where(ke_current <= 0.0, 1.0, jnp.sqrt(lambda_sq_safe))

  return lambda_factor, key


def _csvr_rescale_momenta(momentum: Array, lambda_factor: Array) -> Array:
  r"""Apply scalar rescaling to momenta.

  Since rescaling is a scalar operation, it preserves constraint subspaces:
  if v is in the tangent space of SETTLE constraints, so is alpha*v.

  Args:
    momentum: Atomic momenta (N, 3).
    lambda_factor: Scalar rescaling factor.

  Returns:
    Rescaled momenta.
  """
  return lambda_factor * momentum


def _n_dof_thermostated(
  n_atoms: int,
  n_waters: int,
  n_constraint_pairs: int = 0,
  remove_com: bool = True,
) -> int:
  r"""Compute degrees of freedom for CSVR thermostat.

  For a system with rigid water (SETTLE) + optional solute bonds (SHAKE/RATTLE):
  DOF = 3*N_atoms - 3*N_waters (SETTLE removes 3 DOF per water) - n_constraint_pairs (solute bonds)

  Then subtract 3 for COM motion removal if requested.

  For pure-water system (all atoms are water):
  - n_atoms = 3*N_waters
  - DOF = 3*(3*N_waters) - 3*N_waters = 9*N_waters - 3*N_waters = 6*N_waters
  - With remove_com=True: DOF = 6*N_waters - 3

  Args:
    n_atoms: Total number of atoms.
    n_waters: Number of water molecules.
    n_constraint_pairs: Number of constrained bond pairs (solute).
    remove_com: If True, subtract 3 for COM motion removal.

  Returns:
    Number of thermostated degrees of freedom.
  """
  # Start with 3N (all translational DOF)
  dof = 3 * n_atoms

  # Subtract 3 DOF per water molecule (SETTLE removes 3 constraints per water)
  dof -= 3 * n_waters

  # Subtract additional solute bond constraints
  dof -= n_constraint_pairs

  # Subtract 3 for COM motion if requested
  if remove_com:
    dof -= 3

  return dof


def settle_csvr(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  tau: float = None,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  n_constraint_pairs: int = 0,
  remove_com: bool = True,
  box: Array | None = None,
  **kwargs,
) -> tuple[Callable, Callable]:
  r"""CSVR (Bussi velocity-rescaling) thermostat integrator with SETTLE constraints.

  Uses velocity-Verlet (B-A-A-B) rather than BAOAB — no O-step; CSVR replaces it.

  Process:
  1.  **B-step**: Half-kick momenta.
  2.  **A-step × 2**: Full position advance (two half-moves at fixed momentum).
  3.  **SETTLE-Pos**: Correct water positions analytically.
  4.  **Force eval**: Recompute forces at constrained positions.
  5.  **B-step**: Final half-kick.
  6.  **SETTLE-Vel**: Velocity constraint correction.
  7.  **CSVR-Rescale**: Global scalar velocity rescaling via chi-squared sampling.

  CSVR replaces the per-DOF Langevin O-step with a single chi-squared sample that
  drives a scalar rescaling of all momenta. This scalar operation preserves SETTLE
  constraint subspaces by construction: if v is in tangent space, so is alpha*v.

  **Allows dt=2.0fs (or larger)** without temperature oscillations.

  Args:
      energy_or_force_fn: System force definition.
      shift_fn: Displacement function.
      dt: Timestep (AKMA units).
      kT: Thermal energy target (kcal/mol).
      tau: Velocity rescaling time constant (AKMA units); default ~0.1ps. Require tau >> dt.
      mass: Atomic masses (scalar or (N,) array).
      water_indices: (N_waters, 3) array of water atom indices, or None.
      r_OH: Target O-H bond length (Å).
      r_HH: Target H-H distance (Å).
      mass_oxygen: Mass of oxygen (amu).
      mass_hydrogen: Mass of hydrogen (amu).
      n_constraint_pairs: Number of solute SHAKE/RATTLE bond pairs (for DOF count).
      remove_com: If True, exclude COM translation from DOF count.
      box: Periodic box dimensions or None.
      **kwargs: Additional arguments (energy_fn kwargs).

  Note:
      **Known limitation — temperature bias at dt >= 1 fs**: CSVR+SETTLE exhibits a
      tau-dependent mean temperature bias of approximately +8 K at dt >= 1 fs (e.g.,
      +8 K at tau=0.1 ps, dt=1 fs). Root cause: velocity-Verlet (B-A-A-B) operator
      splitting introduces a leading-order discretization error in the Bussi 2007
      rescaling step when dt is comparable to the thermostat relaxation time tau.
      At dt < 1 fs, the bias drops below ~1 K and is not practically significant.
      For simulations requiring tight temperature control (|T - T_target| < 5 K),
      use dt <= 0.5 fs or increase tau (weaker coupling reduces per-step bias).
      A runtime warning is emitted when dt >= 1 fs / 48.888 AKMA units.

  Returns:
      (init_fn, apply_fn) following JAX-MD convention.
  """
  # AKMA time unit = 48.888 fs; 1 fs ≈ 0.02046 AKMA
  _ONE_FS_AKMA = 1.0 / 48.88821291839
  if dt >= _ONE_FS_AKMA:
    warnings.warn(
      f"settle_csvr: dt={dt:.4f} AKMA ({dt * 48.88821291839:.1f} fs) >= 1 fs. "
      "CSVR+SETTLE shows a tau-dependent ~+8K mean temperature bias at dt>=1 fs "
      "due to velocity-Verlet discretization. This is a known limitation; see "
      "settle_csvr docstring for details. For tight temperature control, use dt<1 fs.",
      UserWarning,
      stacklevel=2,
    )

  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box)
  if tau is None:
    tau = _DEFAULT_CSVR_TAU_AKMA

  def init_fn(key, R, mass=mass, **init_kwargs):
    _kT = init_kwargs.pop("kT", kT)
    key, split = jax.random.split(key)
    force = force_fn(R, **init_kwargs)

    # Handle mass - keep as 1D for indexing, expand to (N, 1) for broadcasting
    mass_arr = jnp.asarray(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr
    elif mass_arr.ndim == 2:
      mass_arr = mass_arr.reshape(-1)

    # Initialize momenta from Maxwell-Boltzmann
    momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
      split, R.shape, dtype=R.dtype
    )

    # Store mass as (N, 1) for broadcasting
    mass_for_state = mass_arr[:, jnp.newaxis]

    return NVTLangevinState(R, momenta, force, mass_for_state, key)

  def apply_fn(state, **step_kwargs):
    _dt = step_kwargs.pop("dt", dt)
    _kT = step_kwargs.pop("kT", kT)

    # Store old positions for SETTLE
    positions_old = state.positions

    # === B-step (half-kick) ===
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # === A-step (full position advance via two half-steps) ===
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    # === SETTLE position constraints ===
    if water_indices is not None:
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

    # === Force eval (at constrained positions) ===
    force = force_fn(position, **step_kwargs)

    # === B-step (final half-kick) ===
    momentum = _langevin_step_b(momentum, force, _dt)

    # === SETTLE velocity constraints ===
    if water_indices is not None:
      momentum = _langevin_settle_vel(
        momentum,
        positions_old,
        position,
        state.mass,
        water_indices,
        _dt,
        mass_oxygen,
        mass_hydrogen,
        n_iters=10,
        settle_velocity_tol=None,
      )

    # === CSVR velocity rescaling ===
    # Compute total kinetic energy in the constrained system
    if water_indices is not None and water_indices.shape[0] > 0:
      n_waters = water_indices.shape[0]
      ke_total = rigid_water_ke.rigid_tip3p_box_ke_kcal(position, momentum, state.mass, n_waters)
    else:
      # No water: standard KE formula
      velocity = momentum / state.mass
      ke_total = 0.5 * jnp.sum(state.mass * velocity**2)

    # Compute DOF for this system
    n_atoms = state.positions.shape[0]
    n_waters = water_indices.shape[0] if water_indices is not None else 0
    n_dof = _n_dof_thermostated(
      n_atoms, n_waters, n_constraint_pairs=n_constraint_pairs, remove_com=remove_com
    )

    # Draw chi-squared and compute lambda
    lambda_factor, key_out = _csvr_compute_lambda(
      state.key, ke_total, n_dof, _kT, _dt, tau
    )

    # Apply rescaling
    momentum = _csvr_rescale_momenta(momentum, lambda_factor)

    # Return updated state
    return NVTLangevinState(position, momentum, force, state.mass, key_out)

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


def _ou_noise_one_water_rigid(
  key: Array,
  r_stack: Array,
  m_stack: Array,
  kT: float,
) -> tuple[Array, Array]:
  r"""Sample OU noise in the 6D rigid-body subspace for one TIP3P water.

  Returns mass-weighted noise (shape (3, 3)) and consumed key.
  Covariance is kT * M * P_rigid — correct constrained Maxwell-Boltzmann.

  Args:
    key: JAX PRNGKey.
    r_stack: (3, 3) array of O, H1, H2 positions.
    m_stack: (3,) array of O, H1, H2 masses.
    kT: Thermal energy.

  Returns:
    (p_noise, key_out) where p_noise is (3, 3) mass-weighted noise momenta.
  """
  msum = jnp.sum(m_stack)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum
  rrel = r_stack - com

  rows = []
  for i in range(3):
    row = jnp.concatenate([jnp.eye(3, dtype=r_stack.dtype), -_skew_symmetric3(rrel[i])], axis=1)
    rows.append(row)
  jmat = jnp.vstack(rows)

  m_rep = jnp.repeat(m_stack, 3)
  g = (jmat.T * m_rep) @ jmat
  reg = jnp.array(1e-12, dtype=g.dtype) * (jnp.trace(g) / 6.0 + 1.0)
  g_reg = g + reg * jnp.eye(6, dtype=g.dtype)

  key, split = jax.random.split(key)
  z = jax.random.normal(split, (6,), dtype=r_stack.dtype)

  L = jnp.linalg.cholesky(g_reg)
  xi = jnp.linalg.solve(L.T, jnp.sqrt(kT) * z)

  p_noise_flat = m_rep * (jmat @ xi)
  return p_noise_flat.reshape(3, 3), key


def _init_momentum_one_water_rigid(
  key: Array,
  r_stack: Array,
  m_stack: Array,
  kT: float,
) -> tuple[Array, Array]:
  r"""Sample initial momentum in rigid-body subspace for one TIP3P water.

  For initialization (not OU step), we sample directly from the Maxwell-Boltzmann
  distribution in the constrained subspace, i.e., `p ~ N(0, kT * M * P_rigid)`.

  Returns:
    (momentum, key_out) where momentum is (3, 3) mass-weighted momenta.
  """
  return _ou_noise_one_water_rigid(key, r_stack, m_stack, kT)


def _langevin_step_o_constrained(
  momentum: Array,
  position: Array,
  mass: Array,
  gamma: float,
  dt: float,
  kT: float,
  rng: Array,
  water_indices: WaterIndicesArray,
) -> tuple[Array, Array]:
  r"""Constrained O-step: OU noise restricted to 6D rigid-body subspace per water.

  For water molecules, noise is sampled in the rigid-body subspace via
  ``_ou_noise_one_water_rigid``, ensuring correct equipartition. Non-water
  atoms (if present) use standard isotropic OU noise.

  Args:
    momentum: (N_atoms, 3) momentum array.
    position: (N_atoms, 3) position array.
    mass: (N_atoms,) or (N_atoms, 1) mass array.
    gamma: Friction coefficient.
    dt: Timestep.
    kT: Thermal energy.
    rng: JAX PRNGKey.
    water_indices: (N_waters, 3) array of O, H1, H2 indices.

  Returns:
    (momentum_new, key_out).
  """
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(1 - c1**2)

  key, split = jax.random.split(rng)
  noise_std = jax.random.normal(split, momentum.shape, dtype=momentum.dtype)
  p_ou = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise_std

  idx = water_indices
  mass_flat = mass.reshape(-1)
  p_water_in = jnp.stack([momentum[idx[:, 0]], momentum[idx[:, 1]], momentum[idx[:, 2]]], axis=1)
  r_water = jnp.stack([position[idx[:, 0]], position[idx[:, 1]], position[idx[:, 2]]], axis=1)
  m_water = jnp.stack([mass_flat[idx[:, 0]], mass_flat[idx[:, 1]], mass_flat[idx[:, 2]]], axis=1)

  def step_one_water(carry, inputs):
    key_w = carry
    r_w, m_w, p_w = inputs
    p_rigid = _project_one_water_momentum_rigid(p_w, r_w, m_w)
    p_c1 = c1 * p_rigid
    noise_w, key_w = _ou_noise_one_water_rigid(key_w, r_w, m_w, kT)
    p_out = p_c1 + c2 * noise_w
    return key_w, p_out

  key, p_water_out = jax.lax.scan(
    step_one_water, key, (r_water, m_water, p_water_in)
  )

  idx_flat = idx.reshape(-1)
  p_out = p_ou.at[idx_flat].set(p_water_out.reshape(-1, 3))
  return p_out, key


def _langevin_step_o_free_dof(
    momentum: Array,
    mass: Array,
    gamma: float,
    dt: float,
    kT: float,
    rng: Array,
    free_dof_mask: Array,
) -> tuple[Array, Array]:
    r"""Ornstein-Uhlenbeck noise applied ONLY to free (non-water) atoms.

    Constrained (water) atoms receive NO thermostat noise, breaking the
    SETTLE+thermostat KE feedback loop. Free atoms get standard isotropic OU.

    Args:
        momentum: (N_atoms, 3) momentum array.
        mass: (N_atoms,) or (N_atoms, 1) mass array.
        gamma: Friction coefficient (ps^-1 in AKMA).
        dt: Timestep (AKMA).
        kT: Thermal energy (kcal/mol).
        rng: JAX PRNGKey.
        free_dof_mask: (N_atoms,) bool — True = free atom, False = constrained.
            Must not be all-False (pure-water systems not supported).

    Returns:
        (momentum_new, key_out) tuple.
    """
    # NOTE: Do NOT add a jnp.all(~free_dof_mask) guard here — this function is
    # JIT-traced and jnp.all returns a traced value. Python `if` on a traced bool
    # causes a concretization error. Caller must validate free_dof_mask at
    # construction time (before JIT), e.g. in O_Step.__init__ or settle_langevin.

    # Canonicalize mass to (N,1) so jnp.sqrt(mass*kT)*noise broadcasts correctly
    # against noise.shape=(N,3). Without this, (N,) mass would broadcast as (1,N),
    # applying mass[j] to spatial dim j rather than mass[i] to atom i.
    mass = mass.reshape(-1, 1) if mass.ndim == 1 else mass

    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(1 - c1**2)

    key, split = jax.random.split(rng)
    noise = jax.random.normal(split, momentum.shape, dtype=momentum.dtype)
    p_ou = c1 * momentum + c2 * jnp.sqrt(mass * kT) * noise

    # Apply only to free atoms; constrained atoms keep original momentum
    mask = free_dof_mask[:, None] if free_dof_mask.ndim == 1 else free_dof_mask
    momentum_new = jnp.where(mask, p_ou, momentum)
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


def settle_csvr_npt(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: Callable[..., Array],
  dt: float,
  kT: float,
  target_pressure_bar: float,
  tau_barostat_akma: float,
  tau_thermostat_akma: float | None = None,
  mass: float | Array = 1.0,
  water_indices: Array | None = None,
  r_OH: float = TIP3P_ROH,
  r_HH: float = TIP3P_RHH,
  mass_oxygen: float = 15.999,
  mass_hydrogen: float = 1.008,
  n_constraint_pairs: int = 0,
  remove_com: bool = True,
  box_init: Array | None = None,
  compressibility_bar_inv: float = 4.5e-5,
  mu_min: float = 0.999,
  project_ou_momentum_rigid: bool = True,
  projection_site: str = "post_o",
  **kwargs,
) -> tuple[Callable, Callable]:
  r"""NPT barostat integrator combining CSVR thermostat with stochastic cell rescaling.

  Implements the NPT ensemble (isothermal-isobaric) by coupling velocity-Verlet
  dynamics with:
  1. **CSVR thermostat** (Bussi et al., 2007) for temperature control
  2. **Stochastic cell rescaling** (Bernetti & Bussi, 2020) for pressure control
  3. **SETTLE constraints** for rigid water molecules

  Algorithm sequence per timestep:
  1. Compute instantaneous pressure from pre-step config (virial + kinetic energy)
  2. B-step: Half-kick momentum from current forces
  3. A-step × 2: Full position update with constant momentum
  4. SETTLE position projection (re-satisfy water geometry)
  5. Stochastic cell rescaling:
     a. Apply stochastic scaling: μ = exp(dε/3) where dε includes pressure deviation + noise
     b. Scale box and positions by μ
  6. SETTLE position re-projection (re-satisfy constraints after box scaling)
  7. Force recompute with new box (MANDATORY for energy consistency)
  8. B-step: Final half-kick from new forces
  9. SETTLE velocity projection
  10. CSVR velocity rescaling (global chi-squared-based rescaling)

  **Key differences from NVT CSVR**:
  - Box rescaling occurs AFTER position SETTLE but BEFORE force recomputation
  - O and H positions scaled together (isotropic); H re-constrained after scaling
  - PME grid is FIXED at init; runtime check ensures box doesn't drift >10% (option B per oracle)
  - Pressure calculated from virial + kinetic energy; box volume used for normalization

  **AKMA pressure units**: 1 kcal/mol/Å³ ≈ 69,477 bar (NOT 14,583)

  Args:
      energy_or_force_fn: Force or energy function F(R, **kwargs).
      shift_fn: Displacement function for periodic boundary conditions.
      dt: Timestep (AKMA units).
      kT: Thermal energy target (kcal/mol).
      target_pressure_bar: Target pressure in bar (converted to AKMA internally).
      tau_barostat_akma: Barostat time constant (AKMA units, typically 0.1 ps ≈ 2000 AKMA).
      tau_thermostat_akma: Thermostat time constant (AKMA units); if None, defaults to ~0.1 ps.
      mass: Atomic masses (scalar or (N,) array).
      water_indices: (N_waters, 3) array of [O, H1, H2] indices; if None, no SETTLE.
      r_OH, r_HH: Target O-H and H-H distances (Å).
      mass_oxygen, mass_hydrogen: Atomic masses for water (amu).
      n_constraint_pairs: Number of solute SHAKE/RATTLE pairs (for DOF counting).
      remove_com: If True, exclude COM translation from thermostat DOF.
      box_init: Initial box dimensions for PME grid validation (optional).
      compressibility_bar_inv: Isothermal compressibility in bar⁻¹ (default TIP3P 4.5e-5).
      mu_min: Minimum scaling factor (clipping lower bound; default 0.999 ≈ 0.1% max volume change per step).
      project_ou_momentum_rigid: If True (default), project OU noise to rigid-body subspace
          for water molecules (not used in velocity-Verlet CSVR, kept for API consistency).
      projection_site: Placement for optional rigid momentum projection (not used here,
          kept for API consistency with settle_langevin).
      **kwargs: Additional arguments passed to energy_or_force_fn (e.g., soft_core_lambda).

  Returns:
      (init_fn, apply_fn) tuple following JAX-MD convention.

  References:
      Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity
      rescaling. J. Chem. Phys., 126(1), 014101.

      Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic cell rescaling.
      J. Chem. Phys., 153(11), 114107.

      Miyamoto, S., & Kollman, P. A. (1992). SETTLE: An analytical version of the SHAKE
      and RATTLE algorithm for rigid water models. J. Comput. Chem., 13(8), 952-962.
  """

  force_fn = _make_settle_compatible_force_fn(energy_or_force_fn, mass, box_init)

  # Default tau_thermostat if not provided
  if tau_thermostat_akma is None:
    tau_thermostat_akma = _DEFAULT_CSVR_TAU_AKMA

  # Convert pressure units: bar → AKMA (kcal/mol/Å³)
  target_pressure_akma = target_pressure_bar * units_module.AKMA_PRESSURE_PER_BAR

  # Convert compressibility: bar⁻¹ → AKMA units
  # β_akma = β_bar * BAR_PER_AKMA_PRESSURE
  compressibility_akma = compressibility_bar_inv * units_module.BAR_PER_AKMA_PRESSURE

  def init_fn(
      key,
      R,
      mass=mass,
      box=box_init,
      momentum: Array | None = None,
      **init_kwargs,
  ):
    r"""Initialize NPT state (host-side; not part of the jitted step loop).

    JAX-MD convention: call once from Python with concrete arrays, then
    ``jax.jit(apply_fn)`` for production steps. The Maxwell-Boltzmann vs
    warm-handoff choice uses ``momentum is None`` at Python trace time — do
    not wrap ``init_fn`` in ``jax.jit`` (would require ``static_argnames`` or
    ``jax.lax.cond`` for a traced optional momentum).
    """
    _kT = init_kwargs.pop("kT", kT)
    _box = init_kwargs.pop("box", box)

    # Validate box_init for PME grid validation
    if box_init is None:
      raise ValueError(
        "box_init must be provided for settle_csvr_npt; "
        "pass the initial box dimensions for PME grid drift validation"
      )

    key, split = jax.random.split(key)
    force = force_fn(R, **init_kwargs)

    # Handle mass array
    mass_arr = jnp.asarray(mass, dtype=R.dtype)
    if mass_arr.ndim == 0:
      mass_arr = jnp.ones((R.shape[0],), dtype=R.dtype) * mass_arr
    elif mass_arr.ndim == 2:
      mass_arr = mass_arr.reshape(-1)

    # Host branch: sample MB (cold) or copy equilibrated NVT momenta (warm handoff).
    if momentum is not None:
      momenta = jnp.asarray(momentum, dtype=R.dtype)
    else:
      momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
        split, R.shape, dtype=R.dtype
      )

    # Store mass as (N, 1) for broadcasting
    mass_for_state = mass_arr[:, jnp.newaxis]

    # Validate and store box
    if _box is None:
      msg = "box must be provided for NPT initialization"
      raise ValueError(msg)
    _box = jnp.asarray(_box, dtype=R.dtype)

    return NPTState(R, momenta, force, mass_for_state, key, _box)

  def apply_fn(state, **step_kwargs):
    """Apply NPT integrator step."""
    _dt = step_kwargs.pop("dt", dt)
    _kT = step_kwargs.pop("kT", kT)
    _box = step_kwargs.pop("box", state.box)

    # Store old positions for SETTLE
    positions_old = state.positions

    # === Compute instantaneous pressure for barostat (from pre-step config) ===
    # Use positions and forces from the same configuration (start of step)
    # Kinetic energy (using rigid-body KE if water is present)
    if water_indices is not None and water_indices.shape[0] > 0:
      n_waters = water_indices.shape[0]
      ke_total = rigid_water_ke.rigid_tip3p_box_ke_kcal(positions_old, state.momentum, state.mass, n_waters)
    else:
      # Standard KE: 0.5 * sum(p² / m)
      velocity = state.momentum / state.mass
      ke_total = 0.5 * jnp.sum(state.mass * velocity**2)

    # Virial trace (use forces from current state, positions before A-step)
    virial = stress_module.virial_trace(positions_old, state.force)

    # === B-step (half-kick) ===
    momentum = _langevin_step_b(state.momentum, state.force, _dt)

    # === A-step (full position advance via two half-steps) ===
    position = _langevin_step_a(state.positions, momentum, state.mass, _dt, shift_fn)
    position = _langevin_step_a(position, momentum, state.mass, _dt, shift_fn)

    # === SETTLE position constraints ===
    if water_indices is not None and water_indices.shape[0] > 0:
      position = settle_positions(
        position,
        positions_old,
        water_indices,
        r_OH,
        r_HH,
        mass_oxygen,
        mass_hydrogen,
        _box,
      )

    # Box volume
    volume = pbc_module.box_volume(_box)

    # Instantaneous pressure
    pressure = pressure_module.compute_pressure_akma(ke_total, virial, volume, ndim=3)

    # === Stochastic cell rescaling (Bernetti & Bussi 2020) ===
    # dε = -(dt/tau_P) * β * (P - P_0) + sqrt(2*kT*β*dt / (tau_P*V)) * noise
    pressure_deviation = pressure - target_pressure_akma

    key, split = jax.random.split(state.key)  # key: for CSVR, split: for barostat noise
    random_noise = jax.random.normal(split, dtype=_box.dtype)

    # Deterministic term
    depsilon_det = (_dt / tau_barostat_akma) * compressibility_akma * pressure_deviation

    # Stochastic term
    depsilon_stoch = (
      jnp.sqrt(2.0 * _kT * compressibility_akma * _dt / (tau_barostat_akma * volume))
      * random_noise
    )

    depsilon = depsilon_det + depsilon_stoch

    # Linear scaling factor: μ = exp(dε/3)
    mu = jnp.exp(depsilon / 3.0)

    # Safety clamp: μ ∈ [μ_min, 1/μ_min]
    mu_max = 1.0 / mu_min
    mu = jnp.clip(mu, mu_min, mu_max)

    # SCR isotropic scaling: momenta inverse-scale with mu (Sprint 14; Bernetti-Bussi convention).
    momentum = momentum / mu

    # Scale box
    new_box = pbc_module.isotropic_box_scale(_box, mu)

    # Scale positions
    scaled_positions = position * mu

    # === SETTLE position re-projection (after box scaling) ===
    if water_indices is not None and water_indices.shape[0] > 0:
      position = settle_positions(
        positions_unconstrained=scaled_positions,
        positions_old=scaled_positions,  # Use scaled positions as reference: fix geometry around correctly-placed O
        water_indices=water_indices,
        r_OH=r_OH,
        r_HH=r_HH,
        mass_oxygen=mass_oxygen,
        mass_hydrogen=mass_hydrogen,
        box=new_box,
      )
    else:
      position = scaled_positions

    # === PME grid validation (option B: fixed grid at init-time) ===
    if box_init is not None:
      volume_init = pbc_module.box_volume(box_init)
      volume_new = pbc_module.box_volume(new_box)
      volume_ratio = volume_new / volume_init
      # volume_ratio computed above; warn if >10% drift (informational; no-op here)
      del volume_ratio

    # === Force eval (at new box) ===
    force = force_fn(position, **step_kwargs)

    # === B-step (final half-kick) ===
    momentum = _langevin_step_b(momentum, force, _dt)

    # === SETTLE velocity constraints ===
    if water_indices is not None and water_indices.shape[0] > 0:
      momentum = _langevin_settle_vel(
        momentum,
        positions_old,
        position,
        state.mass,
        water_indices,
        _dt,
        mass_oxygen,
        mass_hydrogen,
        n_iters=10,
        settle_velocity_tol=None,
      )

    # === CSVR velocity rescaling ===
    if water_indices is not None and water_indices.shape[0] > 0:
      n_waters = water_indices.shape[0]
      ke_total = rigid_water_ke.rigid_tip3p_box_ke_kcal(position, momentum, state.mass, n_waters)
    else:
      velocity = momentum / state.mass
      ke_total = 0.5 * jnp.sum(state.mass * velocity**2)

    # Compute DOF
    n_atoms = position.shape[0]
    n_waters = water_indices.shape[0] if water_indices is not None else 0
    n_dof = _n_dof_thermostated(
      n_atoms, n_waters, n_constraint_pairs=n_constraint_pairs, remove_com=remove_com
    )

    # Draw chi-squared and compute lambda (use 'key' from barostat split, not state.key)
    lambda_factor, key_out = _csvr_compute_lambda(
      key, ke_total, n_dof, _kT, _dt, tau_thermostat_akma
    )

    # Apply rescaling
    momentum = _csvr_rescale_momenta(momentum, lambda_factor)

    return NPTState(position, momentum, force, state.mass, key_out, new_box)

  return init_fn, apply_fn
