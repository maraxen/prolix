"""Pressure calculation from kinetic energy and virial terms.

Implements instantaneous pressure computation used in barostat algorithms.
"""

from __future__ import annotations

from typing import Any
import jax.numpy as jnp
from jax_md import util
from prolix.physics import explicit_corrections

Array = util.Array


def instantaneous_pressure_akma(
  kinetic_energy: Array,
  virial: Array,
  physics_system: Any,
  params: Any,
  cutoff_distance: float = 9.0,
  ndim: int = 3,
) -> Array:
  r"""Compute instantaneous pressure from kinetic energy, virial, and tail corrections.

  Uses the pressure equation:
    P = (2K + W + W_tail) / (d·V)

  where:
    K = kinetic energy (kcal/mol)
    W = virial trace = -Σᵢ rᵢ · Fᵢ (kcal/mol)
    V = volume (Å³)
    W_tail = P_tail * V * d
    d = number of dimensions (3 for isotropic systems)

  Args:
      kinetic_energy: Total kinetic energy (kcal/mol).
      virial: Virial trace W = -Σᵢ rᵢ · Fᵢ (kcal/mol).
      physics_system: The PhysicsSystem object containing box_size, atom_mask.
      params: The parameters containing sigmas and epsilons.
      cutoff_distance: The cutoff distance used for LJ.
      ndim: Number of spatial dimensions (default 3).

  Returns:
      Instantaneous pressure in kcal/mol/Å³ (AKMA pressure units).
  """
  volume = physics_system.box_size[0] * physics_system.box_size[1] * physics_system.box_size[2]
  
  p_tail = explicit_corrections.lj_dispersion_tail_pressure(
      physics_system.box_size,
      jnp.maximum(params.params['sigmas'], 1e-6),
      params.params['epsilons'],
      cutoff_distance,
      physics_system.atom_mask,
  )
  p_imp = explicit_corrections.lj_dispersion_tail_impulsive_pressure(
      physics_system.box_size,
      jnp.maximum(params.params['sigmas'], 1e-6),
      params.params['epsilons'],
      cutoff_distance,
      physics_system.atom_mask,
  )
  
  # Total virial includes tail contribution
  total_virial = virial + (p_tail + p_imp) * volume * ndim
  
  return (2.0 * kinetic_energy + total_virial) / (ndim * volume)
