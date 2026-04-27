"""Virial stress computation for pressure calculations.

Implements virial-based pressure computation used in barostat integrators.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax_md import util

Array = util.Array


def virial_trace(positions: Array, forces: Array) -> Array:
  r"""Compute the virial trace for pressure calculation.

  The virial W is defined as:
    W = Σᵢ rᵢ · Fᵢ

  where rᵢ are atomic positions and Fᵢ are forces (Fᵢ = -∂U/∂rᵢ).

  This is used in the pressure calculation:
    P = (2K + W) / (3V)

  where K is kinetic energy, V is volume, and d=3 for isotropic systems.

  Args:
      positions: Atomic positions (N, 3) in Å.
      forces: Atomic forces (N, 3) in kcal/mol/Å.

  Returns:
      Scalar virial W in kcal/mol (sum of position-force dot products).
  """
  # W = Σᵢ rᵢ · Fᵢ (forces are -∂U/∂r, so W < 0 for bound systems)
  return jnp.sum(positions * forces)
