"""Bidirectional unit conversion between OpenMM native units and Prolix AKMA units.

This module provides conversion constants and helper functions to translate between:
- OpenMM units: nm, ps, kJ/mol, nm/ps, kJ/(mol·nm), bar, kJ/(mol·nm²), kJ/(mol·rad²), openmm.unit.Quantity
- Prolix AKMA internal units: Å, τ_AKMA (48.88821291839 fs), kcal/mol, Å/τ_AKMA, kcal/(mol·Å), kcal/(mol·Å³)

Architectural constraint:
This module is a strict 0-dependency leaf module within `prolix`. It does NOT import
any other `prolix.*` modules, strictly adhering to `tests/test_import_architecture.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import openmm.unit as u

if TYPE_CHECKING:
    from jaxtyping import Array
    from openmm.unit import Quantity

__all__ = [
    # Length
    "NM_TO_ANGSTROM",
    "ANGSTROM_TO_NM",
    # Time
    "TAU_AKMA_FS",
    "TAU_AKMA_PS",
    "PS_TO_TAU_AKMA",
    "TAU_AKMA_TO_PS",
    # Energy
    "KJ_PER_MOL_TO_KCAL_PER_MOL",
    "KCAL_PER_MOL_TO_KJ_PER_MOL",
    # Velocity
    "VELOCITY_OMM_TO_AKMA",
    "VELOCITY_AKMA_TO_OMM",
    # Force
    "FORCE_OMM_TO_AKMA",
    "FORCE_AKMA_TO_OMM",
    # Pressure
    "BAR_PER_AKMA_PRESSURE",
    "AKMA_PRESSURE_PER_BAR",
    # Harmonic Force Constants
    "BOND_K_OMM_TO_AKMA",
    "BOND_K_AKMA_TO_OMM",
    "ANGLE_K_OMM_TO_AKMA",
    "ANGLE_K_AKMA_TO_OMM",
    "TORSION_K_OMM_TO_AKMA",
    "TORSION_K_AKMA_TO_OMM",
    "IMPROPER_K_OMM_TO_AKMA",
    "IMPROPER_K_AKMA_TO_OMM",
    # Friction
    "FRICTION_OMM_TO_AKMA",
    "FRICTION_AKMA_TO_OMM",
    # Backward compatibility / Convenience aliases
    "KJ_TO_KCAL",
    "KCAL_TO_KJ",
    "KJ_NM2_TO_KCAL_A2",
    "KCAL_A2_TO_KJ_NM2",
    # Conversion Functions
    "to_akma",
    "from_akma",
]

# ============================================================================
# Exact Conversion Factors & Constants
# ============================================================================

# Length: nanometers <-> Ångströms (1 nm = 10 Å)
NM_TO_ANGSTROM: float = 10.0
ANGSTROM_TO_NM: float = 0.1

# Time: picoseconds <-> AKMA time units
# 1 AKMA time unit τ_AKMA = 48.88821291839 fs = 0.04888821291839 ps
TAU_AKMA_FS: float = 48.88821291839
TAU_AKMA_PS: float = 0.04888821291839
PS_TO_TAU_AKMA: float = 1000.0 / 48.88821291839
TAU_AKMA_TO_PS: float = 0.04888821291839

# Energy: kJ/mol <-> kcal/mol (exact thermochemical definition: 1 cal = 4.184 J)
KJ_PER_MOL_TO_KCAL_PER_MOL: float = 1.0 / 4.184
KCAL_PER_MOL_TO_KJ_PER_MOL: float = 4.184

# Velocity: nm/ps <-> Å/τ_AKMA
# Derivation: 1 nm/ps = (10 Å) / ((1000 / 48.88821291839) τ) = 0.4888821291839 Å/τ
VELOCITY_OMM_TO_AKMA: float = 0.4888821291839
VELOCITY_AKMA_TO_OMM: float = 1.0 / 0.4888821291839

# Force: kJ/(mol·nm) <-> kcal/(mol·Å)
# Derivation: (1 / 4.184 kcal/mol) / (10 Å) = 1 / 41.84 kcal/(mol·Å)
FORCE_OMM_TO_AKMA: float = 1.0 / 41.84
FORCE_AKMA_TO_OMM: float = 41.84

# Pressure: bar <-> kcal/(mol·Å³)
# Reference: 1 kcal/(mol·Å³) = 69477.0 bar
BAR_PER_AKMA_PRESSURE: float = 69477.0
AKMA_PRESSURE_PER_BAR: float = 1.0 / 69477.0

# Harmonic Bond Force Constant: kJ/(mol·nm²) <-> kcal/(mol·Å²)
# Both OpenMM and Prolix use E = 0.5 * k * (r - r0)^2
BOND_K_OMM_TO_AKMA: float = 1.0 / 418.4
BOND_K_AKMA_TO_OMM: float = 418.4

# Harmonic Angle Force Constant: kJ/(mol·rad²) <-> kcal/(mol·rad²)
# Both OpenMM and Prolix use E = 0.5 * k * (theta - theta0)^2
ANGLE_K_OMM_TO_AKMA: float = 1.0 / 4.184
ANGLE_K_AKMA_TO_OMM: float = 4.184

# Periodic Torsion Force Constant: kJ/mol <-> kcal/mol
# Both OpenMM and Prolix use E = k * (1 + cos(n * phi - phase))
TORSION_K_OMM_TO_AKMA: float = 1.0 / 4.184
TORSION_K_AKMA_TO_OMM: float = 4.184

# Harmonic Improper Force Constant:
# OpenMM convention: E = 0.5 * k_omm * (psi - psi0)^2 (kJ/mol)
# Prolix convention: E = 1.0 * k_prolix * (psi - psi0)^2 (kcal/mol)
# Conversion accounts for 0.5 vs 1.0 factor: k_prolix = 0.5 * k_omm / 4.184 = k_omm / 8.368
IMPROPER_K_OMM_TO_AKMA: float = 0.5 / 4.184
IMPROPER_K_AKMA_TO_OMM: float = 2.0 * 4.184

# Langevin Friction / Collision Frequency: ps^-1 <-> τ_AKMA^-1
FRICTION_OMM_TO_AKMA: float = 0.04888821291839
FRICTION_AKMA_TO_OMM: float = 1000.0 / 48.88821291839

# Convenience / Backward-Compatibility Aliases
KJ_TO_KCAL: float = KJ_PER_MOL_TO_KCAL_PER_MOL
KCAL_TO_KJ: float = KCAL_PER_MOL_TO_KJ_PER_MOL
KJ_NM2_TO_KCAL_A2: float = BOND_K_OMM_TO_AKMA
KCAL_A2_TO_KJ_NM2: float = BOND_K_AKMA_TO_OMM


# ============================================================================
# Unit Registry & Specification
# ============================================================================

_UNIT_SPECS: dict[str, dict[str, Any]] = {
    "length": {
        "to_akma": NM_TO_ANGSTROM,
        "from_akma": ANGSTROM_TO_NM,
        "omm_unit": u.nanometer,
        "akma_desc": "angstrom (Å)",
    },
    "time": {
        "to_akma": PS_TO_TAU_AKMA,
        "from_akma": TAU_AKMA_TO_PS,
        "omm_unit": u.picosecond,
        "akma_desc": "tau_akma (48.88821291839 fs)",
    },
    "energy": {
        "to_akma": KJ_PER_MOL_TO_KCAL_PER_MOL,
        "from_akma": KCAL_PER_MOL_TO_KJ_PER_MOL,
        "omm_unit": u.kilojoule_per_mole,
        "akma_desc": "kcal/mol",
    },
    "velocity": {
        "to_akma": VELOCITY_OMM_TO_AKMA,
        "from_akma": VELOCITY_AKMA_TO_OMM,
        "omm_unit": u.nanometer / u.picosecond,
        "akma_desc": "Å/tau_akma",
    },
    "force": {
        "to_akma": FORCE_OMM_TO_AKMA,
        "from_akma": FORCE_AKMA_TO_OMM,
        "omm_unit": u.kilojoule_per_mole / u.nanometer,
        "akma_desc": "kcal/(mol·Å)",
    },
    "pressure": {
        "to_akma": AKMA_PRESSURE_PER_BAR,
        "from_akma": BAR_PER_AKMA_PRESSURE,
        "omm_unit": u.bar,
        "akma_desc": "kcal/(mol·Å³)",
    },
    "bond_k": {
        "to_akma": BOND_K_OMM_TO_AKMA,
        "from_akma": BOND_K_AKMA_TO_OMM,
        "omm_unit": u.kilojoule_per_mole / (u.nanometer**2),
        "akma_desc": "kcal/(mol·Å²)",
    },
    "angle_k": {
        "to_akma": ANGLE_K_OMM_TO_AKMA,
        "from_akma": ANGLE_K_AKMA_TO_OMM,
        "omm_unit": u.kilojoule_per_mole / (u.radian**2),
        "akma_desc": "kcal/(mol·rad²)",
    },
    "torsion_k": {
        "to_akma": TORSION_K_OMM_TO_AKMA,
        "from_akma": TORSION_K_AKMA_TO_OMM,
        "omm_unit": u.kilojoule_per_mole,
        "akma_desc": "kcal/mol",
    },
    "improper_k": {
        "to_akma": IMPROPER_K_OMM_TO_AKMA,
        "from_akma": IMPROPER_K_AKMA_TO_OMM,
        "omm_unit": u.kilojoule_per_mole / (u.radian**2),
        "akma_desc": "kcal/(mol·rad²)",
    },
    "friction": {
        "to_akma": FRICTION_OMM_TO_AKMA,
        "from_akma": FRICTION_AKMA_TO_OMM,
        "omm_unit": u.picosecond**-1,
        "akma_desc": "1/tau_akma",
    },
}

_ALIAS_MAP: dict[str, str] = {
    # Length
    "length": "length",
    "distance": "length",
    "position": "length",
    "positions": "length",
    "coord": "length",
    "coords": "length",
    "coordinates": "length",
    "box": "length",
    "box_vectors": "length",
    "nm": "length",
    "nanometer": "length",
    "nanometers": "length",
    "angstrom": "length",
    "angstroms": "length",
    # Time
    "time": "time",
    "dt": "time",
    "timestep": "time",
    "ps": "time",
    "picosecond": "time",
    "picoseconds": "time",
    # Energy
    "energy": "energy",
    "potential_energy": "energy",
    "kinetic_energy": "energy",
    "kj/mol": "energy",
    "kj_per_mol": "energy",
    "kilojoule_per_mole": "energy",
    "kilojoules_per_mole": "energy",
    # Velocity
    "velocity": "velocity",
    "velocities": "velocity",
    "vel": "velocity",
    "nm/ps": "velocity",
    # Force
    "force": "force",
    "forces": "force",
    "kj/(mol*nm)": "force",
    "kj/(mol·nm)": "force",
    "kj_per_mol_per_nm": "force",
    # Pressure
    "pressure": "pressure",
    "bar": "pressure",
    "atm": "pressure",
    "atmosphere": "pressure",
    # Bond force constant
    "bond_k": "bond_k",
    "bond": "bond_k",
    "bonds": "bond_k",
    "harmonic_bond": "bond_k",
    "bond_force_constant": "bond_k",
    "kj/(mol*nm^2)": "bond_k",
    "kj/(mol·nm²)": "bond_k",
    "kj_per_mol_per_nm2": "bond_k",
    # Angle force constant
    "angle_k": "angle_k",
    "angle": "angle_k",
    "angles": "angle_k",
    "harmonic_angle": "angle_k",
    "angle_force_constant": "angle_k",
    "kj/(mol*rad^2)": "angle_k",
    "kj/(mol·rad²)": "angle_k",
    "kj_per_mol_per_rad2": "angle_k",
    # Periodic torsion force constant
    "torsion_k": "torsion_k",
    "torsion": "torsion_k",
    "torsions": "torsion_k",
    "periodic_torsion": "torsion_k",
    "dihedral": "torsion_k",
    "dihedrals": "torsion_k",
    "dihedral_force_constant": "torsion_k",
    # Improper force constant
    "improper_k": "improper_k",
    "improper": "improper_k",
    "impropers": "improper_k",
    "harmonic_improper": "improper_k",
    "improper_force_constant": "improper_k",
    "improper_torsion": "improper_k",
    # Friction
    "friction": "friction",
    "gamma": "friction",
    "collision_rate": "friction",
    "collision_frequency": "friction",
    "ps^-1": "friction",
    "1/ps": "friction",
    "ps**-1": "friction",
}


def _resolve_spec(unit_type: str) -> dict[str, Any]:
    """Resolve a unit_type string or alias to its specification dictionary."""
    if not isinstance(unit_type, str):
        raise TypeError(f"unit_type must be a string, got {type(unit_type).__name__}")
    norm = unit_type.strip().lower()
    if norm not in _ALIAS_MAP:
        valid_keys = sorted(set(_UNIT_SPECS.keys()))
        raise ValueError(
            f"Unsupported unit_type: {unit_type!r}. Supported unit types: {valid_keys}"
        )
    canonical = _ALIAS_MAP[norm]
    return _UNIT_SPECS[canonical]


def _scale(val: Any, factor: float) -> Any:
    """Recursively scale scalars, lists, tuples, NumPy arrays, or JAX arrays."""
    if val is None or isinstance(val, (str, bytes)):
        raise TypeError(f"Expected numeric value or array, got {type(val).__name__}")
    if isinstance(val, list):
        return [_scale(item, factor) for item in val]
    if isinstance(val, tuple):
        return tuple(_scale(item, factor) for item in val)
    return val * factor


def to_akma(val_or_quantity: Any, unit_type: str) -> float | Array | Any:
    """Convert an OpenMM quantity or raw OpenMM-unit value to Prolix AKMA internal units.

    Args:
        val_or_quantity: A scalar, list, NumPy array, JAX array, or `openmm.unit.Quantity`.
            If a `Quantity` is passed, its unit is checked for dimensional compatibility
            with the expected OpenMM unit for `unit_type`.
            If raw numeric data is passed, it is assumed to already be in canonical OpenMM units.
        unit_type: The physical quantity type (e.g. 'length', 'time', 'energy', 'velocity',
            'force', 'pressure', 'bond_k', 'angle_k', 'torsion_k', 'improper_k', 'friction').

    Returns:
        The numerical value converted to Prolix AKMA units, preserving container type
        (float, list, tuple, NumPy ndarray, or JAX Array).

    Raises:
        TypeError: If `val_or_quantity` is a `Quantity` whose unit is incompatible with `unit_type`,
            or if `val_or_quantity` is non-numeric.
        ValueError: If `unit_type` is unrecognized.
    """
    if val_or_quantity is None or isinstance(val_or_quantity, (str, bytes)):
        raise TypeError(
            f"Expected numeric value, array, or Quantity, got {type(val_or_quantity).__name__}"
        )

    spec = _resolve_spec(unit_type)
    if isinstance(val_or_quantity, u.Quantity):
        omm_unit = spec["omm_unit"]
        if not val_or_quantity.unit.is_compatible(omm_unit):
            raise TypeError(
                f"Unit mismatch for unit_type={unit_type!r}: Quantity has unit "
                f"{val_or_quantity.unit}, which is incompatible with expected unit {omm_unit}."
            )
        raw_val = val_or_quantity.value_in_unit(omm_unit)
    else:
        raw_val = val_or_quantity

    return _scale(raw_val, spec["to_akma"])


def from_akma(
    val: Any,
    unit_type: str,
    as_quantity: bool = True,
) -> Quantity | float | Array | Any:
    """Convert a Prolix AKMA internal unit value to OpenMM representation.

    Args:
        val: A scalar, list, tuple, NumPy array, or JAX array in Prolix AKMA units.
        unit_type: The physical quantity type (e.g. 'length', 'time', 'energy', 'velocity',
            'force', 'pressure', 'bond_k', 'angle_k', 'torsion_k', 'improper_k', 'friction').
        as_quantity: If True, wraps the result in an `openmm.unit.Quantity` with the
            canonical OpenMM unit. If False, returns the raw numerical value in OpenMM units.

    Returns:
        An `openmm.unit.Quantity` if `as_quantity` is True, else the raw value in OpenMM units.

    Raises:
        TypeError: If `val` is non-numeric.
        ValueError: If `unit_type` is unrecognized.
    """
    if val is None or isinstance(val, (str, bytes)):
        raise TypeError(
            f"Expected numeric value, array, or Quantity, got {type(val).__name__}"
        )

    spec = _resolve_spec(unit_type)
    omm_unit = spec["omm_unit"]

    if isinstance(val, u.Quantity):
        if val.unit.is_compatible(omm_unit):
            scaled_omm = val.value_in_unit(omm_unit)
            if as_quantity:
                return scaled_omm * omm_unit
            return scaled_omm
        raise TypeError(
            f"Quantity has unit {val.unit}, which is incompatible with expected unit {omm_unit}."
        )

    scaled = _scale(val, spec["from_akma"])
    if as_quantity:
        return scaled * omm_unit
    return scaled
