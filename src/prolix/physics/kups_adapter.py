"""Unit conversion adapter for kUPS cross-validation.

kUPS uses eV/Å/amu/fs units. Prolix uses AKMA (kcal/mol/Å/amu, time in AKMA).
This module provides conversion constants and thin wrapper functions so tests
can call kUPS reference functions without ad-hoc unit math inline.

Reference:
  - 1 eV = 23.0605 kcal/mol (NIST 2018 CODATA)
  - 1 AKMA time unit ≈ 48.888 fs
"""

from __future__ import annotations

from prolix.simulate import AKMA_TIME_UNIT_FS

# 1 eV = 23.0605 kcal/mol (NIST 2018 CODATA)
EV_TO_KCAL_MOL: float = 23.060549
KCAL_MOL_TO_EV: float = 1.0 / EV_TO_KCAL_MOL

# kUPS internal constants (eV / Å / amu unit system):
#   - 1 kUPS time unit = 1 Å * sqrt(1 amu / 1 eV) ≈ 10.18050571 fs
#   - FEMTO_SECOND in kUPS = 1 fs / (10.18050571 fs) ≈ 0.09822694788464063
#   - BOLTZMANN_CONSTANT in kUPS = 8.617333262e-5 eV/K
KUPS_FEMTOSECOND: float = 1.0 / (AKMA_TIME_UNIT_FS / (EV_TO_KCAL_MOL ** 0.5))
KUPS_TIME_UNIT_FS: float = 1.0 / KUPS_FEMTOSECOND
KUPS_BOLTZMANN_EV_PER_K: float = 8.617333262e-5
VELOCITY_KUPS_TO_AKMA: float = EV_TO_KCAL_MOL ** 0.5
VELOCITY_AKMA_TO_KUPS: float = 1.0 / VELOCITY_KUPS_TO_AKMA


def dt_fs_to_akma(dt_fs: float) -> float:
  """Convert timestep from femtoseconds to AKMA units."""
  return dt_fs / AKMA_TIME_UNIT_FS


def dt_akma_to_fs(dt_akma: float) -> float:
  """Convert timestep from AKMA units to femtoseconds."""
  return dt_akma * AKMA_TIME_UNIT_FS


def dt_fs_to_kups(dt_fs: float) -> float:
  """Convert timestep from femtoseconds to kUPS internal time units (Å·sqrt(amu/eV))."""
  return dt_fs * KUPS_FEMTOSECOND


def dt_kups_to_fs(dt_kups: float) -> float:
  """Convert timestep from kUPS internal time units to femtoseconds."""
  return dt_kups / KUPS_FEMTOSECOND


def gamma_ps_to_akma(gamma_ps: float) -> float:
  """Convert Langevin friction coefficient from ps⁻¹ to AKMA⁻¹."""
  return gamma_ps * AKMA_TIME_UNIT_FS / 1000.0


def gamma_akma_to_ps(gamma_akma: float) -> float:
  """Convert Langevin friction coefficient from AKMA⁻¹ to ps⁻¹."""
  return gamma_akma * 1000.0 / AKMA_TIME_UNIT_FS


def gamma_ps_to_kups(gamma_ps: float) -> float:
  """Convert Langevin friction coefficient from ps⁻¹ to kUPS time⁻¹."""
  return (gamma_ps / 1000.0) / KUPS_FEMTOSECOND


def tau_ps_to_akma(tau_ps: float) -> float:
  """Convert thermostat time constant from ps to AKMA units."""
  return tau_ps * 1000.0 / AKMA_TIME_UNIT_FS


def tau_akma_to_ps(tau_akma: float) -> float:
  """Convert thermostat time constant from AKMA to ps."""
  return tau_akma * AKMA_TIME_UNIT_FS / 1000.0


def tau_ps_to_kups(tau_ps: float) -> float:
  """Convert thermostat time constant from ps to kUPS time units."""
  return (tau_ps * 1000.0) * KUPS_FEMTOSECOND


def spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev: float) -> float:
  """Convert spring constant from eV/Å² to kcal/mol/Å²."""
  return k_ev * EV_TO_KCAL_MOL


def spring_constant_kcal_mol_to_ev_per_angstrom_sq(k_kcal: float) -> float:
  """Convert spring constant from kcal/mol/Å² to eV/Å²."""
  return k_kcal * KCAL_MOL_TO_EV


def energy_ev_to_kcal_mol(energy_ev):
  """Convert energy from eV to kcal/mol (scalar or array)."""
  return energy_ev * EV_TO_KCAL_MOL


def energy_kcal_mol_to_ev(energy_kcal):
  """Convert energy from kcal/mol to eV (scalar or array)."""
  return energy_kcal * KCAL_MOL_TO_EV


def force_ev_per_angstrom_to_kcal_mol(force_ev):
  """Convert force from eV/Å to kcal/(mol·Å) (scalar or array)."""
  return force_ev * EV_TO_KCAL_MOL


def force_kcal_mol_to_ev_per_angstrom(force_kcal):
  """Convert force from kcal/(mol·Å) to eV/Å (scalar or array)."""
  return force_kcal * KCAL_MOL_TO_EV


def momentum_kups_to_akma(momentum_kups):
  """Convert momentum from kUPS units (amu·Å/t_kups) to AKMA (amu·Å/t_akma)."""
  return momentum_kups * VELOCITY_KUPS_TO_AKMA


def momentum_akma_to_kups(momentum_akma):
  """Convert momentum from AKMA units to kUPS units."""
  return momentum_akma * VELOCITY_AKMA_TO_KUPS
