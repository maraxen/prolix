"""AKMA unit system constants and conversions for molecular dynamics.

AKMA unit system (AMBER/CHARMM convention):
  Distance: Ångströms (Å)
  Energy: kcal/mol
  Mass: g/mol (Daltons, amu)
  Time: derived unit τ = sqrt(amu·Å²/(kcal/mol)) ≈ 48.888 fs

This module provides conversion factors for external units (bar, ps) to AKMA units.
"""

from __future__ import annotations

# ============================================================================
# Time Constants
# ============================================================================

# 1 AKMA time unit in femtoseconds
AKMA_TIME_UNIT_FS: float = 48.88821291839


# ============================================================================
# Pressure Constants
# ============================================================================

# Pressure conversion: 1 kcal/mol/Å³ = 6.9477e4 bar
# Derivation:
#   1 kcal/mol = 6.9477e-21 J
#   1 Å³ = 1e-30 m³
#   1 kcal/mol/Å³ = 6.9477e-21 J / 1e-30 m³ = 6.9477e9 Pa = 6.9477e4 bar

BAR_PER_AKMA_PRESSURE: float = 69477.0  # bar per (kcal/mol/Å³)
AKMA_PRESSURE_PER_BAR: float = 1.0 / BAR_PER_AKMA_PRESSURE  # (kcal/mol/Å³) per bar


# ============================================================================
# Water Properties (TIP3P at 300K, 1 atm)
# ============================================================================

# Isothermal compressibility of water: β ≈ 4.5e-5 bar⁻¹
# Reference: Jorgensen et al., J. Chem. Phys. 1983, 79(2), 926-935
# Experimental value at 300K: 4.57e-5 bar⁻¹
# TIP3P predicts: ~4.5e-5 bar⁻¹ (good agreement with experiment)

WATER_COMPRESSIBILITY_300K_BAR_INV: float = 4.5e-5  # bar⁻¹

# Convert to AKMA pressure units
# β_akma = β_bar * BAR_PER_AKMA_PRESSURE (volume change per unit pressure in AKMA units)
WATER_COMPRESSIBILITY_300K_AKMA_INV: float = (
    WATER_COMPRESSIBILITY_300K_BAR_INV * BAR_PER_AKMA_PRESSURE
)
