"""Pressure calculation from kinetic energy and virial terms.

Implements instantaneous pressure computation used in barostat algorithms.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax_md import util

Array = util.Array


def instantaneous_pressure_akma(
  kinetic_energy: Array, virial: Array, volume: Array, ndim: int = 3
) -> Array:
  r"""Compute instantaneous pressure from kinetic energy and virial.

  Uses the pressure equation:
    P = (2K + W) / (d·V)

  where:
    K = kinetic energy (kcal/mol)
    W = virial trace = -Σᵢ rᵢ · Fᵢ (kcal/mol)
    V = volume (Å³)
    d = number of dimensions (3 for isotropic systems)

  Returns pressure in AKMA units (kcal/mol/Å³).

  For an ideal gas with no interactions (W=0), this reduces to:
    P = 2K / (3V) = NkT / V  (where K = 3NkT/2 for N particles)

  Args:
      kinetic_energy: Total kinetic energy (kcal/mol).
      virial: Virial trace W = -Σᵢ rᵢ · Fᵢ (kcal/mol).
      volume: Simulation box volume (Å³).
      ndim: Number of spatial dimensions (default 3).

  Returns:
      Instantaneous pressure in kcal/mol/Å³ (AKMA pressure units).
  """
  return (2.0 * kinetic_energy + virial) / (ndim * volume)
