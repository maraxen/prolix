"""Prolix API: top-level interfaces for simulation planning and execution.

Exports:
  - EnsemblePlan: Orchestrate batch MD simulations (v1.1 stub, run() deferred to Sprint 38)
  - Observable: @runtime_checkable Protocol for observable quantities
  - Trajectory: eqx.Module output type from simulation runs
  - Temperature: Example Observable for kinetic temperature
  - Energy: Observable for total potential energy
  - KineticEnergy: Observable for total kinetic energy
  - RMSD: Observable for root-mean-square displacement vs reference
  - Pressure: Observable for instantaneous pressure (ideal gas)
"""

from prolix.api.ensemble_plan import EnsemblePlan
from prolix.api.observables import (
    Observable,
    Trajectory,
    Temperature,
    Energy,
    KineticEnergy,
    RMSD,
    Pressure,
)

__all__ = [
    "EnsemblePlan",
    "Observable",
    "Trajectory",
    "Temperature",
    "Energy",
    "KineticEnergy",
    "RMSD",
    "Pressure",
]
