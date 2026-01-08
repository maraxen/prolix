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

from prolix.physics.noising import (
  compute_thermal_sigma,
  thermal_noise_fn,
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
]
