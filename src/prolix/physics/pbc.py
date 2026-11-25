"""Periodic boundary conditions for JAX MD."""

from __future__ import annotations

import jax.numpy as jnp
from jax_md import space, util

Array = util.Array


def create_periodic_space(
  box: Array,
) -> tuple[space.DisplacementFn, space.ShiftFn]:
  """Creates displacement and shift functions for a periodic box.

  Args:
      box: (3,) or (3, 3) array defining the periodic box.
           If (3,), assumes an orthogonal box with side lengths.
           If (3, 3), assumes lower-triangular triclinic box vectors (rows are vectors).

  Returns:
      (displacement_fn, shift_fn)

  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    # Orthogonal box
    return space.periodic(box)
  if box.ndim == 2:
    # Triclinic box (OpenMM-convention lower-triangular row-vector lattice A)
    def displacement_fn(
      Ra: Array, Rb: Array, perturbation: Array | None = None, **kwargs
    ) -> Array:
      b = kwargs.get("box", kwargs.get("new_box", box))
      dR = fractional_minimum_image_displacement(Ra, Rb, b)
      if perturbation is not None:
        dR = space.raw_transform(perturbation, dR)
      return dR

    def shift_fn(R: Array, dR: Array, **kwargs) -> Array:
      b = kwargs.get("box", kwargs.get("new_box", box))
      return wrap_positions(R + dR, b)

    return displacement_fn, shift_fn
  msg = f"Box must be 1D or 2D, got shape {box.shape}"
  raise ValueError(msg)


def _round_half_away_from_zero(x: Array) -> Array:
  """Round with tie-breaking: round-half-away-from-zero (deterministic, standard MD convention).

  jnp.round uses round-half-to-even (banker's rounding), which is non-deterministic for
  ties (e.g. 0.5, 1.5). This function implements the standard MD convention where ties
  are broken away from zero: 0.5→1, -0.5→-1. This resolves ambiguity at PBC boundary
  (backlog #4232) when a pair separation is exactly box/2.
  """
  return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def invert_box_lower_triangular(box: Array) -> Array:
  """Compute closed-form reciprocal matrix B = A^{-1} for a (3,) or lower-triangular (3, 3) box.

  Strictly avoids non-portable LAPACK custom calls for StableHLO export compatibility.

  Args:
      box: (3,) orthogonal box lengths or (3, 3) lower-triangular row-vector lattice matrix A.

  Returns:
      (3, 3) lower-triangular inverse matrix B = A^{-1}.
  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    b00 = 1.0 / box[0]
    b11 = 1.0 / box[1]
    b22 = 1.0 / box[2]
    zero = jnp.zeros_like(b00)
    row0 = jnp.stack([b00, zero, zero])
    row1 = jnp.stack([zero, b11, zero])
    row2 = jnp.stack([zero, zero, b22])
    return jnp.stack([row0, row1, row2])
  if box.ndim == 2:
    ax = box[0, 0]
    bx = box[1, 0]
    by = box[1, 1]
    cx = box[2, 0]
    cy = box[2, 1]
    cz = box[2, 2]
    b00 = 1.0 / ax
    b11 = 1.0 / by
    b22 = 1.0 / cz
    b10 = -bx / (ax * by)
    b21 = -cy / (by * cz)
    b20 = (bx * cy - by * cx) / (ax * by * cz)
    zero = jnp.zeros_like(ax)
    row0 = jnp.stack([b00, zero, zero])
    row1 = jnp.stack([b10, b11, zero])
    row2 = jnp.stack([b20, b21, b22])
    return jnp.stack([row0, row1, row2])
  msg = f"box must be 1D or 2D, got shape {box.shape}"
  raise ValueError(msg)


reciprocal_box = invert_box_lower_triangular


def reduce_triclinic_box_vectors(box: Array) -> Array:
  """Reduce a lower-triangular (3, 3) lattice matrix to satisfy OpenMM reduced-basis invariants.

  Enforces:
      a_x > 0, b_y > 0, c_z > 0
      |b_x| <= 0.5 * a_x, |c_x| <= 0.5 * a_x, |c_y| <= 0.5 * b_y
      a_y = a_z = b_z = 0
  """
  box = jnp.asarray(box)
  if box.ndim != 2 or box.shape != (3, 3):
    msg = f"reduce_triclinic_box_vectors expects (3, 3) box, got shape {box.shape}"
    raise ValueError(msg)
  a = box[0]
  b = box[1]
  c = box[2]
  # Flip row signs if any diagonal is negative so a_x > 0, b_y > 0, c_z > 0
  a = jnp.where(a[0] < 0, -a, a)
  b = jnp.where(b[1] < 0, -b, b)
  c = jnp.where(c[2] < 0, -c, c)
  # Sequential reduction: first c by b, then c by a, then b by a
  c = c - b * jnp.round(c[1] / b[1])
  c = c - a * jnp.round(c[0] / a[0])
  b = b - a * jnp.round(b[0] / a[0])
  zero = jnp.zeros_like(a[0])
  row0 = jnp.stack([a[0], zero, zero])
  row1 = jnp.stack([b[0], b[1], zero])
  row2 = jnp.stack([c[0], c[1], c[2]])
  return jnp.stack([row0, row1, row2])


def triclinic_box_from_lengths_angles(
  a: float | Array,
  b: float | Array,
  c: float | Array,
  alpha_deg: float | Array,
  beta_deg: float | Array,
  gamma_deg: float | Array,
  *,
  reduce: bool = True,
  dtype=jnp.float32,
) -> Array:
  """Construct an OpenMM-reduced lower-triangular (3, 3) triclinic box from lengths and angles.

  Supports rhombic dodecahedron (a=b=c=d, alpha=60, beta=60, gamma=60 or 90) and
  truncated octahedron (a=b=c=d, alpha=arccos(-1/3), beta=arccos(-1/3), gamma=arccos(-1/3)
  or alpha=gamma=arccos(1/3), beta=arccos(-1/3)).

  Args:
      a, b, c: Lattice vector lengths in Angstroms.
      alpha_deg, beta_deg, gamma_deg: Inter-vector angles in degrees.
      reduce: Whether to apply OpenMM lattice reduction (default True).
      dtype: Output JAX dtype (default float32).

  Returns:
      (3, 3) lower-triangular lattice matrix A with rows [a_vec, b_vec, c_vec].
  """
  a_val = jnp.asarray(a, dtype=dtype)
  b_val = jnp.asarray(b, dtype=dtype)
  c_val = jnp.asarray(c, dtype=dtype)
  alpha = jnp.deg2rad(jnp.asarray(alpha_deg, dtype=dtype))
  beta = jnp.deg2rad(jnp.asarray(beta_deg, dtype=dtype))
  gamma = jnp.deg2rad(jnp.asarray(gamma_deg, dtype=dtype))

  cos_a = jnp.cos(alpha)
  cos_b = jnp.cos(beta)
  cos_g = jnp.cos(gamma)
  sin_g = jnp.sin(gamma)

  ax = a_val
  bx = b_val * cos_g
  by = b_val * sin_g
  cx = c_val * cos_b
  cy = c_val * (cos_a - cos_b * cos_g) / sin_g
  cz = jnp.sqrt(jnp.maximum(c_val * c_val - cx * cx - cy * cy, jnp.zeros_like(a_val)))

  zero = jnp.zeros_like(ax)
  box = jnp.stack([
    jnp.stack([ax, zero, zero]),
    jnp.stack([bx, by, zero]),
    jnp.stack([cx, cy, cz]),
  ])
  if reduce:
    return reduce_triclinic_box_vectors(box)
  return box


def fractional_minimum_image_displacement(r1: Array, r2: Array, box: Array) -> Array:
  """Compute minimum-image displacement r1 - r2 for a lower-triangular (3, 3) triclinic box.

  Uses OpenMM's 3-step sequential triangular reduction (z -> y -> x):
      d0 = r1 - r2
      sz = round(d0_z / c_z),  d1 = d0 - sz * c
      sy = round(d1_y / b_y),  d2 = d1 - sy * b
      sx = round(d2_x / a_x),  d  = d2 - sx * a
  """
  d0 = r1 - r2
  a = box[0]
  b = box[1]
  c = box[2]
  sz = jnp.round(d0[..., 2:3] / box[2, 2])
  d1 = d0 - sz * c
  sy = jnp.round(d1[..., 1:2] / box[1, 1])
  d2 = d1 - sy * b
  sx = jnp.round(d2[..., 0:1] / box[0, 0])
  return d2 - sx * a


def minimum_image_displacement(r1: Array, r2: Array, box: Array) -> Array:
  """Computes displacement vector r1 - r2 using minimum image convention.

  Args:
      r1: (..., 3) positions
      r2: (..., 3) positions
      box: (3,) orthogonal box dimensions or (3, 3) lower-triangular triclinic box matrix.

  Returns:
      Minimum-image displacement (..., 3).
  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    dr = r1 - r2
    # backlog #4232: use deterministic tie-breaking (round-half-away-from-zero) instead of banker's rounding
    return dr - _round_half_away_from_zero(dr / box) * box
  if box.ndim == 2:
    return fractional_minimum_image_displacement(r1, r2, box)
  msg = f"box must be 1D or 2D, got shape {box.shape}"
  raise ValueError(msg)


def minimum_image_distance(r1: Array, r2: Array, box: Array) -> Array:
  """Computes distance between particles using minimum image convention.

  Args:
      r1: (N, 3) or (3,) positions
      r2: (N, 3) or (3,) positions
      box: (3,) array of box dimensions or (3, 3) lower-triangular triclinic box matrix.

  Returns:
      Distances (N,) or scalar

  """
  dr = minimum_image_displacement(r1, r2, box)
  return jnp.linalg.norm(dr, axis=-1)


def wrap_positions(positions: Array, box: Array) -> Array:
  """Wraps positions into the periodic box.

  Args:
      positions: (..., 3) positions
      box: (3,) box dimensions or (3, 3) lower-triangular triclinic box matrix.

  Returns:
      Wrapped positions (..., 3)

  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    return positions % box
  if box.ndim == 2:
    B = invert_box_lower_triangular(box)
    s = positions @ B
    s_wrap = s - jnp.floor(s)
    return s_wrap @ box
  msg = f"box must be 1D or 2D, got shape {box.shape}"
  raise ValueError(msg)


def box_volume(box: Array) -> Array:
  """Compute volume of simulation box using closed-form scalar arithmetic.

  Args:
      box: (3,) array (orthogonal box with side lengths) or (3, 3) array (lower-triangular triclinic box vectors).

  Returns:
      Volume as scalar in Å³.
      For orthogonal box: box[0] * box[1] * box[2].
      For lower-triangular triclinic box: box[0, 0] * box[1, 1] * box[2, 2].
  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    # Orthogonal box: volume = Lx * Ly * Lz
    return box[0] * box[1] * box[2]
  if box.ndim == 2:
    # Lower-triangular triclinic box: volume = a_x * b_y * c_z
    return box[0, 0] * box[1, 1] * box[2, 2]
  msg = f"box must be 1D or 2D, got shape {box.shape}"
  raise ValueError(msg)


def isotropic_box_scale(box: Array, scaling_factor: Array) -> Array:
  """Scale box isotropically by a uniform factor.

  Args:
      box: (3,) or (3, 3) array representing box dimensions.
      scaling_factor: Scalar or array to multiply box by (typically μ in barostat).

  Returns:
      Scaled box with same shape as input.
  """
  return box * scaling_factor
