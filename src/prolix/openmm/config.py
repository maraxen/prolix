"""Prolix OpenMM configuration, platform property parsing, and platform shim.

Provides:
  - ProlixOpenMMConfig: Typed, validated immutable dataclass for all execution axes.
  - parse_platform_properties: Parses OpenMM string dictionaries into ProlixOpenMMConfig.
  - ProlixPlatform: OpenMM Platform subclass and registration shim.

Architectural constraint:
This module is a strict 0-dependency leaf module within `prolix`. It does NOT import
any other `prolix.*` modules, strictly adhering to `tests/test_import_architecture.py`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import openmm

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ProlixOpenMMConfig",
    "parse_platform_properties",
    "ProlixPlatform",
]


# ============================================================================
# ProlixOpenMMConfig Dataclass
# ============================================================================

PrecisionType = Literal["single", "mixed", "double"]
NonbondedEngineType = Literal["auto", "neighbor_list", "flash", "dense"]
CustomForceModeType = Literal["auto", "lepton_only", "pattern_only"]
IntegratorSchemeType = Literal[
    "auto", "baoab", "leapfrog", "langevin", "verlet", "brownian", "nose_hoover"
]
BarostatModeType = Literal["auto", "mc", "scr"]
WaterConstraintModeType = Literal["settle", "none"]
SoluteConstraintModeType = Literal["shake", "none"]
ExecutionModeType = Literal["inference", "trajectory"]
MinimizerBackendType = Literal["lbfgs", "fire", "openmm_cpu"]


@dataclass(frozen=True)
class ProlixOpenMMConfig:
    """Configuration for Prolix OpenMM Context, Simulation, and Rollout runtime.

    Units:
      - Lengths: Ångströms (Å)
      - Energies: kcal/mol
      - Time: AKMA time / step count
    """

    # --- Precision ---
    precision: PrecisionType = "single"

    # --- Nonbonded Engine ---
    nonbonded_engine: NonbondedEngineType = "auto"

    # --- Pallas Kernels ---
    use_pallas: bool = False
    use_pallas_pme: bool = False
    pallas_interpret: bool = False

    # --- PME & LJ Parameters (Lengths in Å) ---
    pme_grid_spacing: float = 1.0
    ewald_error_tolerance: float = 5e-4
    lj_switch_width: float = 1.5
    use_dispersion_correction: bool = True

    # --- Neighbor List Parameters (Lengths in Å) ---
    nl_skin: float = 2.0
    nl_capacity_multiplier: float = 1.5
    nl_update_every: int = 10
    disable_cell_list: bool = False

    # --- Custom Force Compilation ---
    custom_force_mode: CustomForceModeType = "auto"
    custom_nonbonded_chunk_size: int = 128

    # --- Integrator & Thermostat Routing ---
    integrator_scheme: IntegratorSchemeType = "auto"
    thermostat_override: str | None = None
    remove_com_motion: bool = True
    deterministic_seed: int = 42

    # --- Barostat Routing ---
    barostat_mode: BarostatModeType = "auto"
    barostat_frequency_override: int | None = None

    # --- Constraints & Virtual Sites ---
    water_constraint_mode: WaterConstraintModeType = "settle"
    solute_constraint_mode: SoluteConstraintModeType = "shake"
    shake_iterations: int = 25
    constraint_tolerance: float = 1e-5
    settle_velocity_tol: float = 1e-6

    # --- Compilation & Bucketing ---
    use_size_buckets: bool = True
    max_steps_per_scan: int = 1000
    execution_mode: ExecutionModeType = "inference"

    # --- Minimizer Backend ---
    minimizer_backend: MinimizerBackendType = "lbfgs"

    def __post_init__(self) -> None:
        """Validate all configuration fields and bounds."""
        # Precision validation
        if self.precision not in ("single", "mixed", "double"):
            raise ValueError(
                f"Invalid precision: {self.precision!r}. Allowed: 'single', 'mixed', 'double'."
            )

        # Nonbonded engine validation
        if self.nonbonded_engine not in ("auto", "neighbor_list", "flash", "dense"):
            raise ValueError(
                f"Invalid nonbonded_engine: {self.nonbonded_engine!r}. "
                "Allowed: 'auto', 'neighbor_list', 'flash', 'dense'."
            )

        # PME & LJ validation
        if (
            not isinstance(self.pme_grid_spacing, (int, float))
            or not math.isfinite(self.pme_grid_spacing)
            or self.pme_grid_spacing <= 0.0
        ):
            raise ValueError(
                f"pme_grid_spacing must be positive and finite, got {self.pme_grid_spacing} Å."
            )
        if (
            not isinstance(self.ewald_error_tolerance, (int, float))
            or not math.isfinite(self.ewald_error_tolerance)
            or not (0.0 < self.ewald_error_tolerance < 1.0)
        ):
            raise ValueError(
                f"ewald_error_tolerance must be in (0, 1), got {self.ewald_error_tolerance}."
            )
        if (
            not isinstance(self.lj_switch_width, (int, float))
            or not math.isfinite(self.lj_switch_width)
            or self.lj_switch_width < 0.0
        ):
            raise ValueError(
                f"lj_switch_width must be non-negative and finite, got {self.lj_switch_width} Å."
            )

        # Neighbor list validation
        if (
            not isinstance(self.nl_skin, (int, float))
            or not math.isfinite(self.nl_skin)
            or self.nl_skin <= 0.0
        ):
            raise ValueError(f"nl_skin must be positive and finite, got {self.nl_skin} Å.")
        if (
            not isinstance(self.nl_capacity_multiplier, (int, float))
            or not math.isfinite(self.nl_capacity_multiplier)
            or self.nl_capacity_multiplier <= 1.0
        ):
            raise ValueError(
                f"nl_capacity_multiplier must be > 1.0 and finite, got {self.nl_capacity_multiplier}."
            )
        if not isinstance(self.nl_update_every, int) or self.nl_update_every < 1:
            raise ValueError(
                f"nl_update_every must be >= 1 step, got {self.nl_update_every}."
            )

        # Custom force validation
        if self.custom_force_mode not in ("auto", "lepton_only", "pattern_only"):
            raise ValueError(
                f"Invalid custom_force_mode: {self.custom_force_mode!r}. "
                "Allowed: 'auto', 'lepton_only', 'pattern_only'."
            )
        if not isinstance(self.custom_nonbonded_chunk_size, int) or self.custom_nonbonded_chunk_size < 1:
            raise ValueError(
                f"custom_nonbonded_chunk_size must be >= 1, got {self.custom_nonbonded_chunk_size}."
            )

        # Integrator scheme validation
        valid_integrators = (
            "auto",
            "baoab",
            "leapfrog",
            "langevin",
            "verlet",
            "brownian",
            "nose_hoover",
            "lfmiddle",
        )
        if self.integrator_scheme not in valid_integrators:
            raise ValueError(
                f"Invalid integrator_scheme: {self.integrator_scheme!r}. "
                f"Allowed: {valid_integrators}."
            )

        if not isinstance(self.deterministic_seed, int) or self.deterministic_seed < 0:
            raise ValueError(
                f"deterministic_seed must be non-negative, got {self.deterministic_seed}."
            )

        # Barostat validation
        if self.barostat_mode not in ("auto", "mc", "scr"):
            raise ValueError(
                f"Invalid barostat_mode: {self.barostat_mode!r}. Allowed: 'auto', 'mc', 'scr'."
            )
        if self.barostat_frequency_override is not None and (
            not isinstance(self.barostat_frequency_override, int)
            or self.barostat_frequency_override < 1
        ):
            raise ValueError(
                f"barostat_frequency_override must be >= 1, got {self.barostat_frequency_override}."
            )

        # Constraints validation
        if self.water_constraint_mode not in ("settle", "none"):
            raise ValueError(
                f"Invalid water_constraint_mode: {self.water_constraint_mode!r}. "
                "Allowed: 'settle', 'none'."
            )
        if self.solute_constraint_mode not in ("shake", "none"):
            raise ValueError(
                f"Invalid solute_constraint_mode: {self.solute_constraint_mode!r}. "
                "Allowed: 'shake', 'none'."
            )
        if not isinstance(self.shake_iterations, int) or self.shake_iterations < 1:
            raise ValueError(
                f"shake_iterations must be >= 1, got {self.shake_iterations}."
            )
        if (
            not isinstance(self.constraint_tolerance, (int, float))
            or not math.isfinite(self.constraint_tolerance)
            or self.constraint_tolerance <= 0.0
        ):
            raise ValueError(
                f"constraint_tolerance must be positive and finite, got {self.constraint_tolerance}."
            )
        if (
            not isinstance(self.settle_velocity_tol, (int, float))
            or not math.isfinite(self.settle_velocity_tol)
            or self.settle_velocity_tol <= 0.0
        ):
            raise ValueError(
                f"settle_velocity_tol must be positive and finite, got {self.settle_velocity_tol}."
            )

        # Bucketing & execution validation
        if not isinstance(self.max_steps_per_scan, int) or self.max_steps_per_scan < 1:
            raise ValueError(
                f"max_steps_per_scan must be >= 1, got {self.max_steps_per_scan}."
            )
        if self.execution_mode not in ("inference", "trajectory"):
            raise ValueError(
                f"Invalid execution_mode: {self.execution_mode!r}. Allowed: 'inference', 'trajectory'."
            )

        # Minimizer validation
        if self.minimizer_backend not in ("lbfgs", "fire", "openmm_cpu"):
            raise ValueError(
                f"Invalid minimizer_backend: {self.minimizer_backend!r}. "
                "Allowed: 'lbfgs', 'fire', 'openmm_cpu'."
            )


# ============================================================================
# Property Parsing & Type Coercion Helpers
# ============================================================================

def _normalize_key(key: str) -> str:
    """Normalize property key by removing non-alphanumeric chars and lowercasing."""
    return re.sub(r"[^a-zA-Z0-9]", "", key).lower()


def _parse_bool(val: Any, key: str) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "t", "on"):
        return True
    if s in ("false", "0", "no", "f", "off"):
        return False
    raise ValueError(f"Invalid boolean value for property {key!r}: {val!r}")


def _parse_int(val: Any, key: str) -> int:
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    try:
        s = str(val).strip()
        # Disallow floats formatted as strings like "12.5"
        if "." in s:
            raise ValueError(f"Invalid integer value for property {key!r}: {val!r}")
        return int(s)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid integer value for property {key!r}: {val!r}") from err


def _parse_opt_int(val: Any, key: str) -> int | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("none", "null", ""):
        return None
    return _parse_int(val, key)


def _parse_float(val: Any, key: str) -> float:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
    else:
        try:
            f = float(str(val).strip())
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid float value for property {key!r}: {val!r}") from err
    if not math.isfinite(f):
        raise ValueError(f"Float value for property {key!r} cannot be NaN or Inf: {val!r}")
    return f


def _parse_str(val: Any, key: str) -> str:
    return str(val).strip().lower()


def _parse_opt_str(val: Any, key: str) -> str | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("none", "null", ""):
        return None
    return s


# Parsers per field
_FIELD_PARSERS: dict[str, Any] = {
    "precision": _parse_str,
    "nonbonded_engine": _parse_str,
    "use_pallas": _parse_bool,
    "use_pallas_pme": _parse_bool,
    "pallas_interpret": _parse_bool,
    "pme_grid_spacing": _parse_float,
    "ewald_error_tolerance": _parse_float,
    "lj_switch_width": _parse_float,
    "use_dispersion_correction": _parse_bool,
    "nl_skin": _parse_float,
    "nl_capacity_multiplier": _parse_float,
    "nl_update_every": _parse_int,
    "disable_cell_list": _parse_bool,
    "custom_force_mode": _parse_str,
    "custom_nonbonded_chunk_size": _parse_int,
    "integrator_scheme": _parse_str,
    "thermostat_override": _parse_opt_str,
    "remove_com_motion": _parse_bool,
    "deterministic_seed": _parse_int,
    "barostat_mode": _parse_str,
    "barostat_frequency_override": _parse_opt_int,
    "water_constraint_mode": _parse_str,
    "solute_constraint_mode": _parse_str,
    "shake_iterations": _parse_int,
    "constraint_tolerance": _parse_float,
    "settle_velocity_tol": _parse_float,
    "use_size_buckets": _parse_bool,
    "max_steps_per_scan": _parse_int,
    "execution_mode": _parse_str,
    "minimizer_backend": _parse_str,
}

# Canonical property names in standard OpenMM PascalCase
_CANONICAL_NAMES: dict[str, str] = {
    "precision": "Precision",
    "nonbonded_engine": "NonbondedEngine",
    "use_pallas": "UsePallas",
    "use_pallas_pme": "UsePallasPME",
    "pallas_interpret": "PallasInterpret",
    "pme_grid_spacing": "PmeGridSpacing",
    "ewald_error_tolerance": "EwaldErrorTolerance",
    "lj_switch_width": "LjSwitchWidth",
    "use_dispersion_correction": "UseDispersionCorrection",
    "nl_skin": "NeighborListSkin",
    "nl_capacity_multiplier": "NeighborListCapacityMultiplier",
    "nl_update_every": "NeighborListUpdateEvery",
    "disable_cell_list": "DisableCellList",
    "custom_force_mode": "CustomForceMode",
    "custom_nonbonded_chunk_size": "CustomNonbondedChunkSize",
    "integrator_scheme": "IntegratorScheme",
    "thermostat_override": "ThermostatOverride",
    "remove_com_motion": "RemoveComMotion",
    "deterministic_seed": "DeterministicSeed",
    "barostat_mode": "BarostatMode",
    "barostat_frequency_override": "BarostatFrequencyOverride",
    "water_constraint_mode": "WaterConstraintMode",
    "solute_constraint_mode": "SoluteConstraintMode",
    "shake_iterations": "ShakeIterations",
    "constraint_tolerance": "ConstraintTolerance",
    "settle_velocity_tol": "SettleVelocityTol",
    "use_size_buckets": "UseSizeBuckets",
    "max_steps_per_scan": "MaxStepsPerScan",
    "execution_mode": "ExecutionMode",
    "minimizer_backend": "MinimizerBackend",
}

# Alias mapping from normalized string to canonical field name
_KEY_MAP: dict[str, str] = {
    _normalize_key(field): field for field in _FIELD_PARSERS
}

# Explicit aliases for common OpenMM PascalCase and short forms
_EXTRA_ALIASES: dict[str, str] = {
    "neighborlistskin": "nl_skin",
    "nlskin": "nl_skin",
    "skin": "nl_skin",
    "neighborlistcapacitymultiplier": "nl_capacity_multiplier",
    "nlcapacitymultiplier": "nl_capacity_multiplier",
    "capacitymultiplier": "nl_capacity_multiplier",
    "neighborlistupdateevery": "nl_update_every",
    "nlupdateevery": "nl_update_every",
    "updateevery": "nl_update_every",
    "pmegridspacing": "pme_grid_spacing",
    "gridspacing": "pme_grid_spacing",
    "ewalderrortolerance": "ewald_error_tolerance",
    "ljswitchwidth": "lj_switch_width",
    "switchwidth": "lj_switch_width",
    "usedispersioncorrection": "use_dispersion_correction",
    "dispersioncorrection": "use_dispersion_correction",
    "disablecelllist": "disable_cell_list",
    "customforcemode": "custom_force_mode",
    "customnonbondedchunksize": "custom_nonbonded_chunk_size",
    "chunksize": "custom_nonbonded_chunk_size",
    "integratorscheme": "integrator_scheme",
    "thermostatoverride": "thermostat_override",
    "removecommotion": "remove_com_motion",
    "removecmmotion": "remove_com_motion",
    "deterministicseed": "deterministic_seed",
    "seed": "deterministic_seed",
    "randomseed": "deterministic_seed",
    "barostatmode": "barostat_mode",
    "barostatfrequencyoverride": "barostat_frequency_override",
    "barostatfrequency": "barostat_frequency_override",
    "waterconstraintmode": "water_constraint_mode",
    "soluteconstraintmode": "solute_constraint_mode",
    "shakeiterations": "shake_iterations",
    "constrainttolerance": "constraint_tolerance",
    "settlevelocitytol": "settle_velocity_tol",
    "usesizebuckets": "use_size_buckets",
    "maxstepsperscan": "max_steps_per_scan",
    "executionmode": "execution_mode",
    "minimizerbackend": "minimizer_backend",
}
for _alias, _field in _EXTRA_ALIASES.items():
    _KEY_MAP[_normalize_key(_alias)] = _field


def parse_platform_properties(
    properties: dict[str, str] | None,
) -> ProlixOpenMMConfig:
    """Parse an OpenMM-style platformProperties string dictionary into a ProlixOpenMMConfig.

    Parameters
    ----------
    properties : dict[str, str] | None
        Dictionary of string key-value pairs (e.g. ``{"Precision": "double", "UsePallas": "true"}``).
        Matching is case-insensitive and ignores underscores/hyphens.

    Returns
    -------
    ProlixOpenMMConfig
        Validated, immutable configuration object.
    """
    if not properties:
        return ProlixOpenMMConfig()

    parsed_kwargs: dict[str, Any] = {}
    for raw_key, raw_val in properties.items():
        norm_key = _normalize_key(raw_key)
        if norm_key not in _KEY_MAP:
            raise ValueError(f"Unknown platform property: {raw_key!r}")

        field = _KEY_MAP[norm_key]
        parser = _FIELD_PARSERS[field]
        parsed_kwargs[field] = parser(raw_val, raw_key)

    return ProlixOpenMMConfig(**parsed_kwargs)


# ============================================================================
# ProlixPlatform OpenMM Platform Shim
# ============================================================================

class ProlixPlatform(openmm.Platform):
    """OpenMM Platform shim for the Prolix JAX execution backend.

    Registers the 'Prolix' platform in OpenMM and exposes standard platform
    property getters and setters.
    """

    _instance: ProlixPlatform | None = None
    _is_registered: bool = False
    _orig_methods: ClassVar[dict[str, Any]] = {}

    def __init__(self) -> None:
        # Do not invoke super().__init__() because openmm.Platform is an abstract C++ class
        # in OpenMM SWIG and has no Python-accessible constructor.
        self._default_properties: dict[str, str] = {}
        default_cfg = ProlixOpenMMConfig()
        for field, canonical in _CANONICAL_NAMES.items():
            val = getattr(default_cfg, field)
            if val is None:
                str_val = "none"
            elif isinstance(val, bool):
                str_val = "true" if val else "false"
            else:
                str_val = str(val)
            self._default_properties[canonical] = str_val

    def getName(self) -> str:  # noqa: N802
        """Return the platform name ('Prolix')."""
        return "Prolix"

    def getSpeed(self) -> float:  # noqa: N802
        """Return the relative speed rating for platform selection (100.0)."""
        return 100.0

    def supportsDoublePrecision(self) -> bool:  # noqa: N802
        """Return whether the platform supports double precision (True)."""
        return True

    def supportsKernels(  # noqa: N802
        self, kernelNames: Sequence[str] | None = None  # noqa: ARG002
    ) -> bool:
        """Return whether the platform supports the specified OpenMM kernels."""
        return True

    def getPropertyNames(self) -> tuple[str, ...]:  # noqa: N802
        """Return all supported property names in canonical PascalCase."""
        return tuple(self._default_properties.keys())

    def getPropertyDefaultValue(self, name: str) -> str:  # noqa: N802
        """Get default value for a platform property (case-insensitive)."""
        norm = _normalize_key(name)
        if norm not in _KEY_MAP:
            raise openmm.OpenMMException(
                f"getPropertyDefaultValue: Illegal property name '{name}'"
            )
        field = _KEY_MAP[norm]
        canonical = _CANONICAL_NAMES[field]
        return self._default_properties[canonical]

    def setPropertyDefaultValue(self, name: str, value: str) -> None:  # noqa: N802
        """Set default value for a platform property (case-insensitive)."""
        norm = _normalize_key(name)
        if norm not in _KEY_MAP:
            raise openmm.OpenMMException(
                f"setPropertyDefaultValue: Illegal property name '{name}'"
            )
        field = _KEY_MAP[norm]
        parser = _FIELD_PARSERS[field]
        # Validate value
        parser(value, name)
        canonical = _CANONICAL_NAMES[field]
        self._default_properties[canonical] = str(value)

    def getPropertyValue(self, context: Any, name: str) -> str:  # noqa: N802
        """Get property value for a specific context, falling back to default."""
        norm = _normalize_key(name)
        if norm not in _KEY_MAP:
            raise openmm.OpenMMException(
                f"getPropertyValue: Illegal property name '{name}'"
            )
        field = _KEY_MAP[norm]
        canonical = _CANONICAL_NAMES[field]

        if context is not None:
            if hasattr(context, "get_platform_property"):
                return str(context.get_platform_property(canonical))
            if hasattr(context, "_platform_properties"):
                for k, v in context._platform_properties.items():
                    if _normalize_key(k) == norm:
                        return str(v)
            if hasattr(context, "_config") and isinstance(context._config, ProlixOpenMMConfig):
                val = getattr(context._config, field)
                if val is None:
                    return "none"
                if isinstance(val, bool):
                    return "true" if val else "false"
                return str(val)

        return self.getPropertyDefaultValue(name)

    def setPropertyValue(self, context: Any, name: str, value: str) -> None:  # noqa: N802
        """Set property value on a context, or default if context is None."""
        norm = _normalize_key(name)
        if norm not in _KEY_MAP:
            raise openmm.OpenMMException(
                f"setPropertyValue: Illegal property name '{name}'"
            )
        field = _KEY_MAP[norm]
        canonical = _CANONICAL_NAMES[field]

        if context is not None:
            if hasattr(context, "set_platform_property"):
                context.set_platform_property(canonical, value)
                return
            if hasattr(context, "_platform_properties"):
                context._platform_properties[canonical] = str(value)
                return

        self.setPropertyDefaultValue(name, value)

    def create_config(
        self, properties: dict[str, str] | None = None
    ) -> ProlixOpenMMConfig:
        """Create a ProlixOpenMMConfig merged with platform default properties."""
        merged = dict(self._default_properties)
        if properties:
            merged.update(properties)
        return parse_platform_properties(merged)

    @classmethod
    def register(cls) -> ProlixPlatform:
        """Register 'Prolix' platform in the OpenMM platform registry."""
        if cls._instance is None:
            cls._instance = cls()

        if not cls._is_registered:
            cls._orig_methods = {
                "getPlatformByName": openmm.Platform.getPlatformByName,
                "getNumPlatforms": openmm.Platform.getNumPlatforms,
                "getPlatform": openmm.Platform.getPlatform,
                "registerPlatform": openmm.Platform.registerPlatform,
            }
            orig_get_by_name = cls._orig_methods["getPlatformByName"]
            orig_get_num = cls._orig_methods["getNumPlatforms"]
            orig_get_plat = cls._orig_methods["getPlatform"]
            orig_register = cls._orig_methods["registerPlatform"]

            def getPlatformByName(name: str):  # noqa: N802
                if name.lower() == "prolix":
                    return cls._instance
                return orig_get_by_name(name)

            def getNumPlatforms() -> int:  # noqa: N802
                return orig_get_num() + 1

            def getPlatform(index_or_name: int | str):  # noqa: N802
                if isinstance(index_or_name, str):
                    return getPlatformByName(index_or_name)
                orig_num = orig_get_num()
                if index_or_name in (orig_num, -1):
                    return cls._instance
                return orig_get_plat(index_or_name)

            def registerPlatform(plat):  # noqa: N802
                if hasattr(plat, "getName") and plat.getName().lower() == "prolix":
                    cls._instance = plat
                    return
                orig_register(plat)

            openmm.Platform.getPlatformByName = staticmethod(getPlatformByName)
            openmm.Platform.getNumPlatforms = staticmethod(getNumPlatforms)
            openmm.Platform.getPlatform = staticmethod(getPlatform)
            openmm.Platform.registerPlatform = staticmethod(registerPlatform)
            cls._is_registered = True

        return cls._instance

    @classmethod
    def unregister(cls) -> None:
        """Unregister 'Prolix' platform and restore original OpenMM methods."""
        if cls._is_registered:
            openmm.Platform.getPlatformByName = staticmethod(
                cls._orig_methods["getPlatformByName"]
            )
            openmm.Platform.getNumPlatforms = staticmethod(
                cls._orig_methods["getNumPlatforms"]
            )
            openmm.Platform.getPlatform = staticmethod(
                cls._orig_methods["getPlatform"]
            )
            openmm.Platform.registerPlatform = staticmethod(
                cls._orig_methods["registerPlatform"]
            )
            cls._is_registered = False
            cls._instance = None


# Ensure registered at module import
ProlixPlatform.register()
