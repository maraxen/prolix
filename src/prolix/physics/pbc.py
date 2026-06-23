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
           If (3, 3), assumes triclinic box vectors (rows are vectors).

  Returns:
      (displacement_fn, shift_fn)

  """
  if box.ndim == 1:
    # Orthogonal box
    return space.periodic(box)
  if box.ndim == 2:
    # Triclinic box
    return space.periodic_general(box)
  msg = f"Box must be 1D or 2D, got shape {box.shape}"
  raise ValueError(msg)


def minimum_image_distance(r1: Array, r2: Array, box: Array) -> Array:
  """Computes distance between particles using minimum image convention.

  Args:
      r1: (N, 3) or (3,) positions
      r2: (N, 3) or (3,) positions
      box: (3,) array of box dimensions (orthogonal only for now)

  Returns:
      Distances (N,) or scalar

  """
  # Simple implementation for orthogonal boxes
  # For triclinic, need to use space.periodic_general displacement
  dr = r1 - r2
  dr = dr - jnp.round(dr / box) * box
  return jnp.linalg.norm(dr, axis=-1)


def wrap_positions(positions: Array, box: Array) -> Array:
  """Wraps positions into the periodic box [0, box].

  Args:
      positions: (N, 3) positions
      box: (3,) box dimensions

  Returns:
      Wrapped positions (N, 3)

  """
  return positions % box


def box_volume(box: Array) -> Array:
  """Compute volume of simulation box.

  Args:
      box: (3,) array (orthogonal box with side lengths) or (3, 3) array (triclinic box vectors).

  Returns:
      Volume as scalar in Å³.
      For orthogonal box: product of side lengths.
      For triclinic box: absolute value of determinant of box vectors.
  """
  box = jnp.asarray(box)
  if box.ndim == 1:
    # Orthogonal box: volume = Lx * Ly * Lz
    return jnp.prod(box)
  if box.ndim == 2:
    # Triclinic box: volume = |det(box vectors)|
    return jnp.abs(jnp.linalg.det(box))
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
