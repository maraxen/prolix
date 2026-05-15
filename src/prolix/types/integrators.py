"""Integrator state types and configuration for prolix MD engine.

Provides jax.export-compatible integrator state hierarchy with no Optional fields,
plus frozen configuration dataclass for static parameters.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class IntegratorState(eqx.Module):
    """Base integrator state: no Optional fields, jax.export-compatible.

    All integrator states must include positions, momenta, forces, PRNG key,
    and the box for potential periodic boundary condition support.
    """

    positions: Float[Array, "N 3"]
    momenta:   Float[Array, "N 3"]
    forces:    Float[Array, "N 3"]
    key:       PRNGKeyArray
    box:       Float[Array, "3 3"]  # zero array when has_pbc=False


class LangevinState(IntegratorState):
    """Langevin thermostat integrator state.

    No additional fields beyond base IntegratorState.
    All state needed is in the parent class.
    """
    pass


class CSVRState(IntegratorState):
    """CSVR (Colored Stochastic Velocity Rescaling) thermostat state.

    Adds accumulated half-step kinetic energy for CSVR thermostat.
    """
    csvr_ke_half: Float[Array, ""]


class NHCState(IntegratorState):
    """Nosé-Hoover Chain thermostat state.

    Adds M-length chain degree-of-freedom vectors for extended-system
    thermostat control.
    """
    nhc_xi:  Float[Array, "M"]   # chain positions
    nhc_vxi: Float[Array, "M"]   # chain velocities


@dataclass(frozen=True)
class IntegratorConfig:
    """Static integrator configuration — not part of the JIT-traced state.

    Immutable (frozen) dataclass holding parameters that do not change during
    a simulation trajectory. Separate from IntegratorState to support clean
    jax.export signatures without Optional fields.

    Attributes:
        thermostat: Thermostat mode ('langevin', 'csvr', or 'nhc')
        has_pbc: Whether periodic boundary conditions are active
        dt: Timestep in AKMA units (0.5 fs)
        kT: Boltzmann constant × temperature (energy units)
        gamma: Langevin friction coefficient (ps^-1)
        n_nhc_chains: Number of NHC chain stages (default 0, ignored for non-nhc)
    """

    thermostat: Literal["langevin", "csvr", "nhc"]
    has_pbc: bool
    dt: float
    kT: float
    gamma: float
    n_nhc_chains: int = 0
