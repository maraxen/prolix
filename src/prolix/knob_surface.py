"""Canonical zero-cycle control-knob surface for Prolix molecular dynamics.

This module is a strict leaf in the Prolix import graph (importing only Python
standard library modules and ``prolix.typing`` under ``TYPE_CHECKING``). All
simulation, physics, nonbonded, implicit-solvent, constraint, cosolvent,
softcore/restraint, collective-variable bias, and Pallas kernel dispatch knobs
converge onto :class:`KnobSurface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from prolix.typing import ArrayLike


@dataclass(frozen=True)
class ThermostatControls:
    """Integrator and thermostat control knobs (AKMA / fs / K / ps^-1)."""

    integrator_method: str = "settle_langevin"
    temperature_k: float = 300.0
    gamma_inv_ps: float = 1.0
    timestep_fs: float = 2.0
    n_steps: int = 1000
    save_interval: int = 500
    seed: int = 0


@dataclass(frozen=True)
class BarostatControls:
    """Isothermal-isobaric (NPT) barostat control knobs."""

    pressure_atm: float | None = None
    barostat_method: str = "mc"
    barostat_interval: int = 25
    anisotropic: bool = False
    compressibility_inv_bar: float = 4.5e-5
    tau_p_ps: float = 1.0


@dataclass(frozen=True)
class NonbondedControls:
    """Direct-space and reciprocal-space nonbonded interaction control knobs."""

    nonbonded_method: str = "pme"
    cutoff_angstrom: float = 9.0
    switch_width_angstrom: float = 1.0
    ewald_error_tolerance: float = 5e-4
    pme_alpha: float = 0.34
    pme_grid_dims: tuple[int, int, int] | None = None
    dispersion_correction: bool = True
    reaction_field_dielectric: float = 78.3
    dsf_alpha: float | None = None
    coulomb_constant: float = 332.0637
    scale_14_lj: float = 0.5
    scale_14_elec: float = 0.8333333333333334


@dataclass(frozen=True)
class ImplicitSolventControls:
    """Generalized Born (OBC-II) and nonpolar SA implicit-solvent control knobs."""

    solvent_model: str | None = None
    solvent_dielectric: float = 78.5
    solute_dielectric: float = 1.0
    obc_alpha: float = 1.0
    obc_beta: float = 0.8
    obc_gamma: float = 4.85
    surface_tension_kcal_mol_a2: float = 0.005
    probe_radius_angstrom: float = 1.4
    dielectric_offset_angstrom: float = 0.09
    salt_concentration_m: float = 0.0
    kappa_inv_angstrom: float = 0.0


@dataclass(frozen=True)
class ConstraintControls:
    """Holonomic bond/angle (SETTLE / SHAKE / RATTLE) constraint control knobs."""

    constraint_mode: str = "hbonds"
    rigid_water: bool = True
    r_oh_angstrom: float = 0.9572
    r_hh_angstrom: float = 1.5139
    shake_tolerance: float = 1e-5
    rattle_tolerance: float = 1e-5
    max_shake_iterations: int = 50


@dataclass(frozen=True)
class CosolventControls:
    """Multi-component cosolvent packing and inter-cosolvent repulsion knobs."""

    cosolvent_entries: tuple[Any, ...] = ()
    protein_radius_angstrom: float = 5.5
    cosolvent_placement_radius_angstrom: float = 2.5
    repulsive_eps_kcal: float = 0.01
    repulsive_r0_angstrom: float = 10.0
    repulsive_cutoff_angstrom: float = 9.0
    deduplicate_interaction_groups: bool = False


@dataclass(frozen=True)
class SoftcoreRestraintControls:
    """Beutler soft-core alchemical sweep and harmonic positional restraint knobs."""

    lambda_schedule: tuple[Any, ...] = ()
    softcore_alpha: float = 0.5
    softcore_exponent: int = 2
    soft_coulomb: bool = False
    restraint_selection: str = "none"
    restraint_k_kcal_mol_a2: float = 0.0
    flat_bottom_radius_angstrom: float = 0.0
    per_stage_max_iterations: int = 500
    tolerance_kcal_mol_a: float = 2.390057


@dataclass(frozen=True)
class CollectiveVariableBiasControls:
    """Native differentiable collective-variable and enhanced-sampling bias knobs."""

    cv_specs: tuple[Any, ...] = ()
    bias_mode: str = "none"
    umbrella_k: float = 0.0
    umbrella_center: float = 0.0
    smd_velocity: float = 0.0
    metad_height: float = 0.0
    metad_sigma: float = 0.05
    metad_bias_factor: float = 10.0
    metad_pace: int = 500
    plumed_min_beta_nm: float = 20.0


@dataclass(frozen=True)
class PallasControls:
    """Full-potential Pallas GPU kernel dispatch control knobs."""

    use_pallas: bool | None = None
    use_pallas_pme: bool | None = None
    gb_use_neighbor_list: bool = False


SUB_SURFACE_TYPES: tuple[type, ...] = (
    ThermostatControls,
    BarostatControls,
    NonbondedControls,
    ImplicitSolventControls,
    ConstraintControls,
    CosolventControls,
    SoftcoreRestraintControls,
    CollectiveVariableBiasControls,
    PallasControls,
)


@dataclass(frozen=True)
class KnobSurface:
    """Canonical composite control-knob surface across all 9 physics sub-surfaces."""

    thermostat: ThermostatControls = field(default_factory=ThermostatControls)
    barostat: BarostatControls = field(default_factory=BarostatControls)
    nonbonded: NonbondedControls = field(default_factory=NonbondedControls)
    implicit_solvent: ImplicitSolventControls = field(
        default_factory=ImplicitSolventControls
    )
    constraints: ConstraintControls = field(default_factory=ConstraintControls)
    cosolvent: CosolventControls = field(default_factory=CosolventControls)
    softcore_restraints: SoftcoreRestraintControls = field(
        default_factory=SoftcoreRestraintControls
    )
    cv_bias: CollectiveVariableBiasControls = field(
        default_factory=CollectiveVariableBiasControls
    )
    pallas: PallasControls = field(default_factory=PallasControls)

    @classmethod
    def sub_surface_classes(cls) -> tuple[type, ...]:
        """Return the ordered tuple of the 9 sub-surface dataclass types."""
        return SUB_SURFACE_TYPES

    @classmethod
    def leaf_field_names(cls) -> set[str]:
        """Return the set of all leaf knob field names across all 9 sub-surfaces."""
        names: set[str] = set()
        for sub_cls in SUB_SURFACE_TYPES:
            for f in fields(sub_cls):
                names.add(f.name)
        return names

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten all 9 sub-surfaces into a single ``{leaf_field: value}`` dict."""
        flat: dict[str, Any] = {}
        for top_field in fields(self):
            sub_obj = getattr(self, top_field.name)
            for leaf_field in fields(sub_obj):
                flat[leaf_field.name] = getattr(sub_obj, leaf_field.name)
        return flat

    @classmethod
    def from_flat_dict(cls, values: Mapping[str, Any]) -> KnobSurface:
        """Construct a :class:`KnobSurface` from flat leaf knob key-value pairs."""
        unknown = set(values.keys()) - cls.leaf_field_names()
        if unknown:
            raise ValueError(f"Unknown KnobSurface leaf fields: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for top_field in fields(cls):
            sub_cls = top_field.default_factory  # type: ignore[assignment]
            if isinstance(sub_cls, type):
                sub_names = {f.name for f in fields(sub_cls)}
                sub_kwargs = {k: v for k, v in values.items() if k in sub_names}
                kwargs[top_field.name] = sub_cls(**sub_kwargs)
        return cls(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate leaf knob attribute access to the owning sub-surface."""
        for top_field in fields(type(self)):
            sub_obj = object.__getattribute__(self, top_field.name)
            if hasattr(type(sub_obj), "__dataclass_fields__") and (
                name in type(sub_obj).__dataclass_fields__
            ):
                return getattr(sub_obj, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


ProlixKnobSurface = KnobSurface

__all__ = [
    "BarostatControls",
    "CollectiveVariableBiasControls",
    "ConstraintControls",
    "CosolventControls",
    "ImplicitSolventControls",
    "KnobSurface",
    "NonbondedControls",
    "PallasControls",
    "ProlixKnobSurface",
    "SUB_SURFACE_TYPES",
    "SoftcoreRestraintControls",
    "ThermostatControls",
]
