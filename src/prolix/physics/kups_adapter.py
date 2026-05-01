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


def dt_fs_to_akma(dt_fs: float) -> float:
  """Convert timestep from femtoseconds to AKMA units."""
  return dt_fs / AKMA_TIME_UNIT_FS


def dt_akma_to_fs(dt_akma: float) -> float:
  """Convert timestep from AKMA units to femtoseconds."""
  return dt_akma * AKMA_TIME_UNIT_FS


def gamma_ps_to_akma(gamma_ps: float) -> float:
  """Convert Langevin friction coefficient from ps⁻¹ to AKMA⁻¹."""
  return gamma_ps * AKMA_TIME_UNIT_FS / 1000.0


def gamma_akma_to_ps(gamma_akma: float) -> float:
  """Convert Langevin friction coefficient from AKMA⁻¹ to ps⁻¹."""
  return gamma_akma * 1000.0 / AKMA_TIME_UNIT_FS


def tau_ps_to_akma(tau_ps: float) -> float:
  """Convert thermostat time constant from ps to AKMA units."""
  return tau_ps * 1000.0 / AKMA_TIME_UNIT_FS


def tau_akma_to_ps(tau_akma: float) -> float:
  """Convert thermostat time constant from AKMA to ps."""
  return tau_akma * AKMA_TIME_UNIT_FS / 1000.0


def spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev: float) -> float:
  """Convert spring constant from eV/Å² to kcal/mol/Å²."""
  return k_ev * EV_TO_KCAL_MOL


def spring_constant_kcal_mol_to_ev_per_angstrom_sq(k_kcal: float) -> float:
  """Convert spring constant from kcal/mol/Å² to eV/Å²."""
  return k_kcal * KCAL_MOL_TO_EV
