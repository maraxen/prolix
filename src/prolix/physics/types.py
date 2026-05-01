"""Standardized PyTree types for Prolix physics.

These types ensure that all parameters passed to the MD engine are JAX-compatible
and follow a strictly static signature, unblocking StableHLO/WASM export.
"""

from typing import Any, Optional

import equinox as eqx
import jax
from jaxtyping import Array as ArrayType


class EnergyParams(eqx.Module):
    """Container for energy function parameters.
    
    Attributes:
        params: Arbitrary PyTree of parameters (e.g., force field parameters).
    """
    params: Any


class IntegratorParams(eqx.Module):
    """Parameters for MD integration steps.
    
    Attributes:
        dt: Timestep (in AKMA units).
        kT: Thermal energy (Boltzmann constant * Temperature).
        gamma: Friction coefficient for Langevin dynamics.
        energy_params: Parameters for the energy/force function.
        water_indices: Indices of water molecules for SETTLE.
        constraint_dofs: Mask for degrees of freedom restricted by constraints.
        box: Periodic boundary condition box dimensions (3,).
    """
    dt: float | ArrayType
    kT: float | ArrayType
    gamma: float | ArrayType
    energy_params: EnergyParams
    water_indices: Optional[ArrayType] = None
    constraint_dofs: Optional[ArrayType] = None
    box: Optional[ArrayType] = None
    positions_old: Optional[ArrayType] = None
    n_dof: Optional[float | ArrayType] = None
