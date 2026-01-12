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
