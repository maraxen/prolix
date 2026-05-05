"""Physics-based feature calculations for protein structure."""

from proxide.physics.constants import (
  ANGSTROM_TO_NM,
  COULOMB_CONSTANT,
  COULOMB_CONSTANT_ATOMIC,
  COULOMB_CONSTANT_KCAL,
  DEFAULT_EPSILON,
  DEFAULT_SIGMA,
  KCAL_TO_KJ,
  KJ_TO_KCAL,
  MIN_DISTANCE,
  NM_TO_ANGSTROM,
)
from proxide.physics.electrostatics import (
  compute_coulomb_forces,
  compute_coulomb_forces_at_backbone,
  compute_pairwise_displacements,
)
from proxide.physics.force_fields import (
  FullForceField,
  list_available_force_fields,
  load_force_field,
)
from proxide.physics.projections import (
  compute_backbone_frame,
  project_forces_onto_backbone,
  project_forces_onto_backbone_per_atom,
)
from proxide.physics.vdw import (
  combine_lj_parameters,
  compute_lj_energy_at_backbone,
  compute_lj_energy_at_positions,
  compute_lj_energy_pairwise,
  compute_lj_force_magnitude_pairwise,
  compute_lj_forces,
  compute_lj_forces_at_backbone,
)

from .noising import (
  compute_thermal_sigma,
  thermal_noise_fn,
)

# NPT integrators and supporting functions (Sprint 6)
from .settle import settle_csvr_npt
from .simulate import NPTState
from .pressure import instantaneous_pressure_akma
from .stress import virial_trace
from .pbc import box_volume, isotropic_box_scale
from .units import (
  BAR_PER_AKMA_PRESSURE,
  AKMA_PRESSURE_PER_BAR,
  WATER_COMPRESSIBILITY_300K_BAR_INV,
  WATER_COMPRESSIBILITY_300K_AKMA_INV,
  AKMA_TIME_UNIT_FS,
)

# Explicit-params energy API for jax.export / StableHLO (v1.1 Item 1)
from .system import DifferentiableParams, make_energy_fn_pure

# kUPS cross-validation unit conversion adapter (v1.1 Item 3)
from prolix.physics import kups_adapter

# Modular integrator builder (Phase 2.1, ADR-005; Phase 4 batching)
from .integrator_builder import make_integrator, make_integrator_batched
from .step_system import (
  IntegratorState,
  Step,
  StepSequence,
  step_sequences,
  make_step,
  make_sequence,
)

__all__ = [
  "ANGSTROM_TO_NM",
  # Constants
  "COULOMB_CONSTANT",
  "COULOMB_CONSTANT_ATOMIC",
  "COULOMB_CONSTANT_KCAL",
  "DEFAULT_EPSILON",
  "DEFAULT_SIGMA",
  "KCAL_TO_KJ",
  "KJ_TO_KCAL",
  "MIN_DISTANCE",
  "NM_TO_ANGSTROM",
  # Force fields
  "FullForceField",
  # Van der Waals
  "combine_lj_parameters",
  # Projections
  "compute_backbone_frame",
  "compute_coulomb_forces",
  "compute_coulomb_forces_at_backbone",
  "compute_lj_energy_at_backbone",
  "compute_lj_energy_at_positions",
  "compute_lj_energy_pairwise",
  "compute_lj_force_magnitude_pairwise",
  "compute_lj_forces",
  "compute_lj_forces_at_backbone",
  "compute_thermal_sigma",
  # Electrostatics
  "compute_pairwise_displacements",
  "list_available_force_fields",
  "load_force_field",
  "project_forces_onto_backbone",
  "project_forces_onto_backbone_per_atom",
  "thermal_noise_fn",
  # NPT ensemble (Sprint 6)
  "settle_csvr_npt",
  "NPTState",
  "instantaneous_pressure_akma",
  "virial_trace",
  "box_volume",
  "isotropic_box_scale",
  "BAR_PER_AKMA_PRESSURE",
  "AKMA_PRESSURE_PER_BAR",
  "WATER_COMPRESSIBILITY_300K_BAR_INV",
  "WATER_COMPRESSIBILITY_300K_AKMA_INV",
  "AKMA_TIME_UNIT_FS",
  # Explicit-params energy API (v1.1 Item 1)
  "DifferentiableParams",
  "make_energy_fn_pure",
  # kUPS adapter (v1.1 Item 3)
  "kups_adapter",
  # Modular integrator builder (Phase 2.1, Phase 4 batching)
  "make_integrator",
  "make_integrator_batched",
  "IntegratorState",
  "Step",
  "StepSequence",
  "step_sequences",
  "make_step",
  "make_sequence",
]
