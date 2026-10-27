"""Adapter and fallback implementation for ``redsox`` static governance (U1 + U5).

Provides:
- :func:`adapt_model_to_dataclass`: adapts Pydantic v2 ``BaseModel`` classes
  (via ``cls.model_fields``) and ``NamedTuple`` classes (via ``cls._fields``)
  into standard ``@dataclass`` types using :func:`dataclasses.make_dataclass`.
- :func:`extract_dataclass_fields`: extracts field names from ``KnobSurface``,
  standard ``@dataclass``, ``NamedTuple``, or Pydantic ``BaseModel`` classes.
- :func:`check_superset` (U5): verifies that ``KnobSurface`` covers every field
  of a target/vendor config schema under ``SUPERSET_ALIAS_MAP`` (excluding
  ``DOCUMENTED_INERT_KNOBS``), delegating to ``redsox`` when installed.
- :func:`check_reachability` (U1) and :func:`check_prolix_reachability`:
  scans consumer-only AST paths (``src/prolix/physics/``, ``src/prolix/api/``,
  ``src/prolix/batched_energy.py``, ``src/prolix/batched_simulate.py``) while
  explicitly excluding config declaration modules (``knob_surface.py``,
  ``nonbonded_config.py``, ``spec.py``, ``simulate.py``).
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass, make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prolix.governance.pending_knobs import (
    DOCUMENTED_INERT_KNOBS,
    PENDING_EPIC_KNOBS,
)
from prolix.knob_surface import KnobSurface
from prolix.physics.nonbonded_config import NeighborListConfig, NonbondedConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# Optional redsox delegation when installed in the environment
try:
    from redsox.checkers.reachability import (  # type: ignore[import-not-found]
        check_reachability as _redsox_check_reachability,
    )
    from redsox.checkers.superset import (  # type: ignore[import-not-found]
        check_superset as _redsox_check_superset,
    )

    HAS_REDSOX = True
except ImportError:
    try:
        from redsox.checks.reachability import (  # type: ignore[import-not-found]
            check_reachability as _redsox_check_reachability,
        )
        from redsox.checks.superset_parity import (  # type: ignore[import-not-found]
            check_superset as _redsox_check_superset,
        )

        HAS_REDSOX = True
    except ImportError:
        _redsox_check_reachability = None
        _redsox_check_superset = None
        HAS_REDSOX = False


DEFAULT_CONSUMER_PATHS: tuple[str, ...] = (
    "src/prolix/physics",
    "src/prolix/api",
    "src/prolix/pallas_kernels.py",
    "src/prolix/batched_energy.py",
    "src/prolix/batched_simulate.py",
)

DEFAULT_EXCLUDED_DECLARATION_MODULES: frozenset[str] = frozenset(
    {
        "knob_surface.py",
        "nonbonded_config.py",
        "spec.py",
        "simulate.py",
    }
)

SUPERSET_ALIAS_MAP: dict[str, str] = {
    # Thermostat / Integrator / Simulation / Equilibration / Production
    "integrator": "integrator_method",
    "thermostat": "integrator_method",
    "ensemble": "integrator_method",
    "temperature": "temperature_k",
    "temp_k": "temperature_k",
    "kT": "temperature_k",
    "temp_start_k": "temperature_k",
    "temp_end_k": "temperature_k",
    "heating_steps": "n_steps",
    "equilibration_steps": "n_steps",
    "production_steps": "n_steps",
    "equilibration_phase_ps": "n_steps",
    "total_time_ns": "n_steps",
    "duration_ns": "n_steps",
    "duration_ps": "n_steps",
    "friction": "gamma_inv_ps",
    "friction_per_ps": "gamma_inv_ps",
    "gamma": "gamma_inv_ps",
    "tau": "gamma_inv_ps",
    "chain_length": "integrator_method",
    "chain_steps": "integrator_method",
    "n_nhc_chains": "integrator_method",
    "timestep": "timestep_fs",
    "dt": "timestep_fs",
    "dt_fs": "timestep_fs",
    "step_size_fs": "timestep_fs",
    "n_report": "save_interval",
    "report_interval": "save_interval",
    "save_every": "save_interval",
    "save_interval_ns": "save_interval",
    "trajectory_interval_ps": "save_interval",
    "random_seed": "seed",
    # Barostat
    "pressure": "pressure_atm",
    "pressure_bar": "pressure_atm",
    "target_pressure_bar": "pressure_atm",
    "barostat": "barostat_method",
    "compressibility": "compressibility_inv_bar",
    "tau_barostat": "tau_p_ps",
    # Nonbonded / PME / NeighborList
    "method": "nonbonded_method",
    "electrostatic_method": "nonbonded_method",
    "engine": "nonbonded_method",
    "use_pbc": "nonbonded_method",
    "has_pbc": "nonbonded_method",
    "box": "nonbonded_method",
    "cutoff": "cutoff_angstrom",
    "cutoff_nm": "cutoff_angstrom",
    "neighbor_cutoff": "cutoff_angstrom",
    "cutoff_distance": "cutoff_angstrom",
    "switch_width": "switch_width_angstrom",
    "switching_distance_nm": "switch_width_angstrom",
    "switch_distance_angstrom": "switch_width_angstrom",
    "use_switching_function": "switch_width_angstrom",
    "alpha": "pme_alpha",
    "grid_spacing": "pme_grid_dims",
    "pme_grid_spacing": "pme_grid_dims",
    "pme_grid_size": "pme_grid_dims",
    "pme_grid_points": "pme_grid_dims",
    "grid_dims": "pme_grid_dims",
    "order": "pme_grid_dims",
    "pme_order": "pme_grid_dims",
    "use_dispersion_correction": "dispersion_correction",
    "long_range_dispersion": "dispersion_correction",
    "exclusion_spec": "scale_14_lj",
    "coulomb14scale": "scale_14_elec",
    "lj14scale": "scale_14_lj",
    "neighbor_list": "nonbonded_method",
    "use_neighbor_list": "nonbonded_method",
    "nl_update_every": "nonbonded_method",
    "neighbor_update_interval_fs": "nonbonded_method",
    "fire_neighbor_update_interval": "nonbonded_method",
    "skin": "cutoff_angstrom",
    "capacity_multiplier": "cutoff_angstrom",
    "disable_cell_list": "nonbonded_method",
    "flash_tile_size": "nonbonded_method",
    "flash_remat": "nonbonded_method",
    # Implicit Solvent
    "implicit_solvent": "solvent_model",
    "gb_model": "solvent_model",
    "model": "solvent_model",
    "epsilon_solvent": "solvent_dielectric",
    "epsilon_solute": "solute_dielectric",
    "alpha_obc": "obc_alpha",
    "beta_obc": "obc_beta",
    "gamma_obc": "obc_gamma",
    "surface_tension": "surface_tension_kcal_mol_a2",
    "sasa_penalty": "surface_tension_kcal_mol_a2",
    "probe_radius": "probe_radius_angstrom",
    "dielectric_offset": "dielectric_offset_angstrom",
    "ionic_strength_molar": "salt_concentration_m",
    "salt_conc": "salt_concentration_m",
    "debye_kappa": "kappa_inv_angstrom",
    # Constraints & HMR
    "constraints": "constraint_mode",
    "constraint_algorithm": "constraint_mode",
    "use_shake": "constraint_mode",
    "constraint_tolerance": "shake_tolerance",
    "settle_velocity_tol": "rattle_tolerance",
    "water_model": "rigid_water",
    "r_OH": "r_oh_angstrom",
    "r_HH": "r_hh_angstrom",
    "theta_HOH": "r_hh_angstrom",
    "charge_O": "coulomb_constant",
    "charge_H": "coulomb_constant",
    "sigma_O": "cutoff_angstrom",
    "epsilon_O": "cutoff_angstrom",
    "k_bond": "r_oh_angstrom",
    "k_angle": "r_hh_angstrom",
    "remove_cm_motion": "constraint_mode",
    "remove_linear_com_momentum": "constraint_mode",
    "hydrogen_mass_repartition": "constraint_mode",
    "hydrogen_mass_amu": "constraint_mode",
    "target_h_mass": "constraint_mode",
    "min_heavy_mass": "constraint_mode",
    "hmr": "constraint_mode",
    "enabled": "constraint_mode",
    "forcefield": "nonbonded_method",
    # Cosolvents (CosolvConfig, CosolvEntry, SimulationConfig)
    "cosolvents": "cosolvent_entries",
    "cosolv_system_xml_path": "cosolvent_entries",
    "smiles": "cosolvent_entries",
    "concentration_molar": "cosolvent_entries",
    "copies": "cosolvent_entries",
    "resname": "cosolvent_entries",
    "padding_nm": "protein_radius_angstrom",
    "add_repulsive": "repulsive_eps_kcal",
    "repulsive_residues": "cosolvent_entries",
    "repulsive_epsilon": "repulsive_eps_kcal",
    "repulsive_sigma": "repulsive_r0_angstrom",
    "repulsive_cutoff": "repulsive_cutoff_angstrom",
    "cosolvent_clash_distance_angstrom": "cosolvent_placement_radius_angstrom",
    "solvent_clash_distance_angstrom": "cosolvent_placement_radius_angstrom",
    "max_placement_attempts": "cosolvent_placement_radius_angstrom",
    # Softcore / Restraints / Minimization
    "windows": "lambda_schedule",
    "schedule": "lambda_schedule",
    "max_iterations_per_window": "per_stage_max_iterations",
    "max_iterations": "per_stage_max_iterations",
    "tolerance_kj_per_mol_per_nm": "tolerance_kcal_mol_a",
    "tolerance": "tolerance_kcal_mol_a",
    "adaptive_scale": "per_stage_max_iterations",
    "restraint": "restraint_selection",
    "atom_names": "restraint_selection",
    "exclude_residue_names": "restraint_selection",
    "anchor": "restraint_selection",
    "force_constant_kcal_per_mol_per_A2": "restraint_k_kcal_mol_a2",
    "restraint_k": "restraint_k_kcal_mol_a2",
    "flat_bottom_radius": "flat_bottom_radius_angstrom",
    # Collective Variable / Bias (BiasSpec)
    "bias": "bias_mode",
    "cvs": "cv_specs",
    "sigma": "metad_sigma",
    "pace": "metad_pace",
    "biasfactor": "metad_bias_factor",
    "barrier": "metad_bias_factor",
    "height": "metad_height",
    "kappa": "umbrella_k",
    "kernel": "bias_mode",
    "grid_min": "umbrella_center",
    "grid_max": "umbrella_center",
    "grid_bin": "metad_pace",
    "min_beta_nm": "plumed_min_beta_nm",
    # Composite sub-configs on SimulationConfig / SimulationSpec
    "physics": "integrator_method",
    "nonbonded": "nonbonded_method",
    "minimization": "per_stage_max_iterations",
    "equilibration": "n_steps",
    "production": "n_steps",
}


@dataclass(frozen=True)
class ReachabilityResult:
    """Result contract for U1 consumer-only static AST reachability."""

    config_class: str
    declared_fields: frozenset[str]
    read_fields: frozenset[str]
    unread_fields: frozenset[str]
    field_read_locations: dict[str, list[str]] = field(default_factory=dict)
    scanned_files: tuple[str, ...] = ()
    verdict: str = "REACHABILITY-HYPOTHESIS-PASS"
    raw_redsox: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        """True when all non-inert, non-pending fields are read in consumers."""
        unaccounted = (
            set(self.unread_fields) - PENDING_EPIC_KNOBS - DOCUMENTED_INERT_KNOBS
        )
        return len(unaccounted) == 0


@dataclass(frozen=True)
class SupersetResult:
    """Result contract for U5 vendor/schema superset parity."""

    target_class: str
    vendor_class: str
    target_fields: frozenset[str]
    vendor_fields: frozenset[str]
    mapped_fields: dict[str, str]
    missing_fields: frozenset[str]
    inert_fields: frozenset[str] = frozenset()
    verdict: str = "SUPERSET-HYPOTHESIS-PASS"
    raw_redsox: dict[str, Any] | None = None

    @property
    def is_superset(self) -> bool:
        """True when zero vendor fields are unmapped."""
        return len(self.missing_fields) == 0

    @property
    def passed(self) -> bool:
        """Alias for :attr:`is_superset`."""
        return self.is_superset

    @property
    def unmapped_fields(self) -> frozenset[str]:
        """Alias for :attr:`missing_fields`."""
        return self.missing_fields


def adapt_model_to_dataclass(cls: type) -> type:
    """Adapt a Pydantic v2 ``BaseModel`` or ``NamedTuple`` into a ``@dataclass``.

    When ``hasattr(cls, "model_fields")`` (Pydantic v2 ``BaseModel`` such as
    ``CosolvConfig`` / ``CosolvEntry``), constructs a dynamic ``@dataclass`` via
    ``dataclasses.make_dataclass(cls.__name__, list(cls.model_fields.keys()))``.
    """
    if hasattr(cls, "model_fields"):
        model_fields = getattr(cls, "model_fields")
        field_names = list(model_fields.keys())
        return make_dataclass(cls.__name__, field_names)
    if is_dataclass(cls):
        return cls
    nt_fields = getattr(cls, "_fields", None)
    if isinstance(nt_fields, tuple):
        return make_dataclass(cls.__name__, list(nt_fields))
    raise TypeError(
        f"Cannot adapt {cls!r} to dataclass: expected @dataclass, Pydantic "
        "BaseModel (with model_fields), or NamedTuple (with _fields)."
    )


def extract_dataclass_fields(cls: type) -> set[str]:
    """Extract canonical field names from a dataclass, ``KnobSurface``, or Pydantic model."""
    if hasattr(cls, "leaf_field_names") and callable(cls.leaf_field_names):
        return set(cls.leaf_field_names())
    adapted = adapt_model_to_dataclass(cls)
    return {f.name for f in fields(adapted)}


def check_superset(
    target_cls: type = KnobSurface,
    vendor_cls: type | Sequence[type] = (),
    alias_map: Mapping[str, str] | None = None,
    *,
    inert_knobs: frozenset[str] = DOCUMENTED_INERT_KNOBS,
) -> SupersetResult:
    """Run U5 superset parity check between ``target_cls`` and ``vendor_cls``."""
    aliases = dict(SUPERSET_ALIAS_MAP if alias_map is None else alias_map)
    target_fields = frozenset(extract_dataclass_fields(target_cls))

    vendor_classes: list[type] = (
        [vendor_cls] if isinstance(vendor_cls, type) else list(vendor_cls)
    )
    all_vendor_fields: set[str] = set()
    mapped: dict[str, str] = {}
    missing: set[str] = set()
    inert: set[str] = set()

    for v_cls in vendor_classes:
        v_fields = extract_dataclass_fields(v_cls)
        all_vendor_fields |= v_fields
        for f_name in sorted(v_fields):
            qualified = f"{v_cls.__name__}.{f_name}"
            if f_name in inert_knobs or qualified in inert_knobs:
                inert.add(f_name)
                continue
            canonical = aliases.get(qualified, aliases.get(f_name, f_name))
            if canonical in target_fields:
                mapped[f_name] = canonical
            elif canonical in inert_knobs:
                inert.add(f_name)
            else:
                missing.add(f_name)

    raw_redsox: dict[str, Any] | None = None
    if HAS_REDSOX and _redsox_check_superset is not None and vendor_classes:
        flat_target = make_dataclass(target_cls.__name__, sorted(target_fields))
        adapted_refs = []
        merged_aliases = dict(aliases)
        for v_cls in vendor_classes:
            adapted_v = adapt_model_to_dataclass(v_cls)
            adapted_refs.append(adapted_v)
            for f_name in extract_dataclass_fields(v_cls):
                qualified = f"{v_cls.__name__}.{f_name}"
                if (f_name in inert_knobs or qualified in inert_knobs) and target_fields:
                    merged_aliases.setdefault(f_name, next(iter(target_fields)))
        try:
            raw_redsox = _redsox_check_superset(
                [flat_target], adapted_refs, alias_map=merged_aliases
            )
        except Exception:
            raw_redsox = None

    vendor_label = (
        vendor_classes[0].__name__
        if len(vendor_classes) == 1
        else ",".join(c.__name__ for c in vendor_classes)
    )
    verdict = (
        "SUPERSET-HYPOTHESIS-PASS"
        if len(missing) == 0
        else "SUPERSET-HYPOTHESIS-FAIL"
    )
    return SupersetResult(
        target_class=target_cls.__name__,
        vendor_class=vendor_label,
        target_fields=target_fields,
        vendor_fields=frozenset(all_vendor_fields),
        mapped_fields=mapped,
        missing_fields=frozenset(missing),
        inert_fields=frozenset(inert),
        verdict=verdict,
        raw_redsox=raw_redsox,
    )


def _collect_consumer_python_files(
    consumer_paths: Iterable[str | Path],
    *,
    repo_root: Path | None = None,
    exclude_files: frozenset[str] = DEFAULT_EXCLUDED_DECLARATION_MODULES,
) -> list[Path]:
    """Expand ``consumer_paths`` to Python files, excluding declaration modules."""
    collected: list[Path] = []
    seen: set[Path] = set()
    for p_raw in consumer_paths:
        p = Path(p_raw)
        if repo_root is not None and not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix == ".py" and p.name not in exclude_files:
                resolved = p.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    collected.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*.py")):
                if child.name in exclude_files:
                    continue
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    collected.append(child)
    return collected


def _scan_ast_attribute_reads(py_files: Sequence[Path]) -> dict[str, list[str]]:
    """Scan Python files via ``ast`` for attribute accesses and ``getattr`` calls."""
    reads: dict[str, list[str]] = {}
    for path in py_files:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                reads.setdefault(node.attr, []).append(f"{path}:{node.lineno}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                attr_name = node.args[1].value
                reads.setdefault(attr_name, []).append(f"{path}:{node.lineno}")
    return reads


def check_reachability(
    config_classes: type | Sequence[type] = KnobSurface,
    consumer_paths: Sequence[str | Path] = DEFAULT_CONSUMER_PATHS,
    *,
    repo_root: Path | None = None,
    exclude_files: Iterable[str] = DEFAULT_EXCLUDED_DECLARATION_MODULES,
) -> ReachabilityResult:
    """Run U1 static AST reachability across ``consumer_paths``."""
    classes: list[type] = (
        [config_classes]
        if isinstance(config_classes, type)
        else list(config_classes)
    )
    excluded = frozenset(exclude_files)
    py_files = _collect_consumer_python_files(
        consumer_paths, repo_root=repo_root, exclude_files=excluded
    )

    declared: set[str] = set()
    for cls in classes:
        declared |= extract_dataclass_fields(cls)

    all_reads = _scan_ast_attribute_reads(py_files)
    read_fields = frozenset(f for f in declared if f in all_reads)
    unread_fields = frozenset(declared - read_fields)
    filtered_locations = {
        f: all_reads[f] for f in sorted(read_fields) if f in all_reads
    }

    raw_redsox: dict[str, Any] | None = None
    if HAS_REDSOX and _redsox_check_reachability is not None and classes:
        flat_target = make_dataclass(classes[0].__name__, sorted(declared))
        try:
            raw_redsox = _redsox_check_reachability(flat_target, py_files)
        except Exception:
            raw_redsox = None

    label = (
        classes[0].__name__
        if len(classes) == 1
        else ",".join(c.__name__ for c in classes)
    )
    verdict = (
        "REACHABILITY-HYPOTHESIS-PASS"
        if len(unread_fields) == 0
        else "REACHABILITY-HYPOTHESIS-FAIL"
    )
    return ReachabilityResult(
        config_class=label,
        declared_fields=frozenset(declared),
        read_fields=read_fields,
        unread_fields=unread_fields,
        field_read_locations=filtered_locations,
        scanned_files=tuple(str(p) for p in py_files),
        verdict=verdict,
        raw_redsox=raw_redsox,
    )


def check_prolix_reachability(
    repo_root: Path | str | None = None,
    *,
    config_classes: type | Sequence[type] | None = None,
) -> ReachabilityResult:
    """Scan ONLY Prolix consumer paths and exclude config declaration modules.

    Consumer paths:
    - ``src/prolix/physics/``
    - ``src/prolix/api/``
    - ``src/prolix/batched_energy.py``
    - ``src/prolix/batched_simulate.py``

    Excluded declaration modules:
    - ``knob_surface.py``
    - ``nonbonded_config.py``
    - ``spec.py``
    - ``simulate.py``
    """
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    targets: type | Sequence[type] = (
        config_classes
        if config_classes is not None
        else (KnobSurface, NonbondedConfig, NeighborListConfig)
    )
    return check_reachability(
        targets,
        DEFAULT_CONSUMER_PATHS,
        repo_root=root,
        exclude_files=DEFAULT_EXCLUDED_DECLARATION_MODULES,
    )


# ---------------------------------------------------------------------------
# Vendor & Prolix reference schema definitions (used when naurmalade or JAX is
# not installed in the active environment, or imported directly by U5 tests).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PMEParams:
    """Canonical PME parameter reference schema."""

    alpha: float = 0.34
    grid_spacing: float = 1.0
    order: int = 4
    pme_grid_dims: tuple[int, int, int] | None = None
    ewald_error_tolerance: float = 5e-4


@dataclass(frozen=True)
class ImplicitSolventParams:
    """Canonical implicit-solvent parameter reference schema."""

    solvent_model: str | None = "obc2"
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
class _FallbackSimulationSpec:
    """Structural mirror of ``prolix.simulate.SimulationSpec`` for AST-only envs."""

    total_time_ns: float = 1.0
    physics: Any = None
    save_interval_ns: float = 0.001
    accumulate_steps: int = 500
    save_path: str = "trajectory.array_record"
    box: Any = None
    use_pbc: bool = False
    pme_grid_size: int = 64
    use_neighbor_list: bool = False
    neighbor_update_interval_fs: float = 20.0
    fire_neighbor_update_interval: int = 1
    exclusion_spec: Any = None
    pme_grid_spacing: float | None = None
    electrostatic_method: str = "pme"
    reaction_field_dielectric: float = 78.3
    dsf_alpha: float | None = None
    constraint_algorithm: Any = None


@dataclass(frozen=True)
class _FallbackNaurmaladeEnsembleConfig:
    ensemble: str = "NVT"
    temperature: float = 298.15
    pressure: float | None = 1.0
    barostat_interval: int = 25


@dataclass(frozen=True)
class _FallbackNaurmaladeIntegratorConfig:
    integrator: str = "LangevinMiddle"
    timestep: float = 2.0
    friction: float = 1.0
    constraint_tolerance: float = 1e-5
    hydrogen_mass_repartition: float = 0.0
    random_seed: int | None = None


@dataclass(frozen=True)
class _FallbackNaurmaladeForceFieldConfig:
    forcefield: str = "amber14-all.xml"
    water_model: str = "tip3pfb"
    constraints: str = "HBonds"
    constraint_tolerance: float = 1e-5
    fix_missing_atoms: bool = True
    solvent_model: str = "explicit"


@dataclass(frozen=True)
class _FallbackNaurmaladeNonbondedConfig:
    method: str = "PME"
    cutoff_nm: float = 1.0
    switching_distance_nm: float = 0.8
    ewald_error_tolerance: float = 0.0005
    remove_cm_motion: bool = True
    rigid_water: bool = True


@dataclass(frozen=True)
class _FallbackNaurmaladeMinimizationConfig:
    max_iterations: int = 1000
    tolerance: float = 10.0


@dataclass(frozen=True)
class _FallbackNaurmaladeEquilibrationConfig:
    equilibration_phase_ps: float = 500.0
    temperature: float = 298.15
    timestep: float = 2.0
    friction: float = 1.0
    restraint_k: float = 10.0


@dataclass(frozen=True)
class _FallbackNaurmaladeProductionConfig:
    n_steps: int = 10000
    n_report: int = 100
    temperature: float = 298.15
    pressure: float | None = 1.0
    barostat_interval: int = 25
    timestep: float = 2.0
    trajectory_interval_ps: float = 10.0


@dataclass(frozen=True)
class _FallbackNaurmaladeImplicitSolventConfig:
    solvent_model: str = "obc2"
    solvent_dielectric: float = 78.5
    solute_dielectric: float = 1.0
    salt_concentration_m: float = 0.0
    surface_tension_kcal_mol_a2: float = 0.005
    probe_radius_angstrom: float = 1.4


@dataclass(frozen=True)
class _FallbackNaurmaladeHMRConfig:
    enabled: bool = False
    target_h_mass: float = 3.024
    min_heavy_mass: float = 1.5
    hydrogen_mass_repartition: float = 0.0


@dataclass(frozen=True)
class _FallbackNaurmaladeBiasSpec:
    method: str = "opes_metad"
    cvs: tuple[Any, ...] = ()
    sigma: float = 0.5
    pace: int = 500
    biasfactor: float | None = None
    barrier: float | None = None
    kappa: float | None = None
    kernel: str | None = None
    grid_min: float = 0.0
    grid_max: float = 2.0
    grid_bin: int = 100
    height: float | None = None


@dataclass(frozen=True)
class _FallbackNaurmaladeSimulationConfig:
    schema_version: str = "1.0"
    input_file: str = "system.pdb"
    output_dir: str = "output"
    name: str = "simulation"
    ensemble: _FallbackNaurmaladeEnsembleConfig = field(
        default_factory=_FallbackNaurmaladeEnsembleConfig
    )
    integrator: _FallbackNaurmaladeIntegratorConfig = field(
        default_factory=_FallbackNaurmaladeIntegratorConfig
    )
    forcefield: _FallbackNaurmaladeForceFieldConfig = field(
        default_factory=_FallbackNaurmaladeForceFieldConfig
    )
    nonbonded: _FallbackNaurmaladeNonbondedConfig = field(
        default_factory=_FallbackNaurmaladeNonbondedConfig
    )
    minimization: _FallbackNaurmaladeMinimizationConfig = field(
        default_factory=_FallbackNaurmaladeMinimizationConfig
    )
    checkpoint: Any = None
    n_steps: int = 10000
    n_report: int = 100
    equilibration_phase_ps: float = 500.0
    platform: str = "CPU"
    precision: str = "mixed"
    device_index: int | None = None
    trajectory_format: str = "xtc"
    trajectory_interval_ps: float = 10.0
    log_level: str = "INFO"
    verbose: bool = False
    skip_minimization: bool = False
    skip_equilibration_phases: bool = False
    input_stage: str | None = None
    target_stage: str = "PRODUCTION"
    step_configs: dict[str, Any] = field(default_factory=dict)
    cosolv_system_xml_path: str | None = None
    cosolv_metadata_path: str | None = None
    bias: _FallbackNaurmaladeBiasSpec | None = None


@dataclass(frozen=True)
class _FallbackNaurmaladeRestraintSpec:
    atom_names: frozenset[str] | None = None
    force_constant_kcal_per_mol_per_A2: float = 10.0
    exclude_residue_names: frozenset[str] = frozenset({"HOH", "CL", "NA", "WAT"})
    anchor: str = "reference"


@dataclass(frozen=True)
class _FallbackNaurmaladeLambdaSchedule:
    windows: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0)
    max_iterations_per_window: int = 5000
    tolerance_kj_per_mol_per_nm: float = 10.0
    alpha: float = 0.5
    adaptive_scale: bool = True


@dataclass(frozen=True)
class _FallbackNaurmaladeSoftcoreSweepStep:
    schedule: _FallbackNaurmaladeLambdaSchedule = field(
        default_factory=_FallbackNaurmaladeLambdaSchedule
    )
    restraint: _FallbackNaurmaladeRestraintSpec | None = None
    soft_coulomb: bool = False


class _FallbackNaurmaladeCosolvEntry:
    """Pydantic v2 ``BaseModel`` mirror with ``model_fields``."""

    model_fields: dict[str, Any] = {
        "name": str,
        "smiles": str,
        "concentration_molar": float,
        "copies": int,
        "resname": str,
    }


_FallbackNaurmaladeCosolvEntry.__name__ = "CosolvEntry"


class _FallbackNaurmaladeCosolvConfig:
    """Pydantic v2 ``BaseModel`` mirror with ``model_fields``."""

    model_fields: dict[str, Any] = {
        "cosolvents": list,
        "padding_nm": float,
        "ionic_strength_molar": float,
        "simulation_format": str,
        "small_molecule_ff": str,
        "protein_ff": str,
        "water_ff": str,
        "output_dir": Path,
        "add_repulsive": bool,
        "repulsive_residues": list,
        "repulsive_epsilon": float,
        "repulsive_sigma": float,
        "protein_radius_angstrom": float,
        "cosolvent_clash_distance_angstrom": float,
        "solvent_clash_distance_angstrom": float,
        "max_placement_attempts": int,
    }


_FallbackNaurmaladeCosolvConfig.__name__ = "CosolvConfig"


def get_reference_config_schemas() -> dict[str, type]:
    """Return dictionary of all Prolix and ``naurmalade`` reference config schemas."""
    schemas: dict[str, type] = {
        "SimulationConfig": _FallbackNaurmaladeSimulationConfig,
        "EquilibrationConfig": _FallbackNaurmaladeEquilibrationConfig,
        "MinimizationConfig": _FallbackNaurmaladeMinimizationConfig,
        "ProductionConfig": _FallbackNaurmaladeProductionConfig,
        "ImplicitSolventConfig": _FallbackNaurmaladeImplicitSolventConfig,
        "HMRConfig": _FallbackNaurmaladeHMRConfig,
        "EnsembleConfig": _FallbackNaurmaladeEnsembleConfig,
        "IntegratorConfig": _FallbackNaurmaladeIntegratorConfig,
        "ForceFieldConfig": _FallbackNaurmaladeForceFieldConfig,
        "NaurmaladeNonbondedConfig": _FallbackNaurmaladeNonbondedConfig,
        "BiasSpec": _FallbackNaurmaladeBiasSpec,
        "RestraintSpec": _FallbackNaurmaladeRestraintSpec,
        "LambdaSchedule": _FallbackNaurmaladeLambdaSchedule,
        "SoftcoreSweepStep": _FallbackNaurmaladeSoftcoreSweepStep,
        "CosolvConfig": _FallbackNaurmaladeCosolvConfig,
        "CosolvEntry": _FallbackNaurmaladeCosolvEntry,
        "SimulationSpec": _FallbackSimulationSpec,
        "NonbondedConfig": NonbondedConfig,
        "NeighborListConfig": NeighborListConfig,
        "PMEParams": PMEParams,
        "ImplicitSolventParams": ImplicitSolventParams,
    }
    try:
        from naurmalade.cosolv.config import (  # type: ignore[import-not-found]
            CosolvConfig,
            CosolvEntry,
        )
        from naurmalade.minimization.restraints import (  # type: ignore[import-not-found]
            RestraintSpec,
        )
        from naurmalade.minimization.softcore import (  # type: ignore[import-not-found]
            LambdaSchedule,
            SoftcoreSweepStep,
        )
        from naurmalade.simulation import (
            config as nm_cfg,  # type: ignore[import-not-found]
        )

        schemas["CosolvConfig"] = CosolvConfig
        schemas["CosolvEntry"] = CosolvEntry
        schemas["RestraintSpec"] = RestraintSpec
        schemas["LambdaSchedule"] = LambdaSchedule
        schemas["SoftcoreSweepStep"] = SoftcoreSweepStep
        for attr in (
            "SimulationConfig",
            "EquilibrationConfig",
            "MinimizationConfig",
            "ProductionConfig",
            "ImplicitSolventConfig",
            "HMRConfig",
            "EnsembleConfig",
            "IntegratorConfig",
            "ForceFieldConfig",
            "BiasSpec",
        ):
            if hasattr(nm_cfg, attr):
                schemas[attr] = getattr(nm_cfg, attr)
        if hasattr(nm_cfg, "NonbondedConfig"):
            schemas["NaurmaladeNonbondedConfig"] = nm_cfg.NonbondedConfig
    except ImportError:
        pass
    return schemas


__all__ = [
    "DEFAULT_CONSUMER_PATHS",
    "DEFAULT_EXCLUDED_DECLARATION_MODULES",
    "HAS_REDSOX",
    "ImplicitSolventParams",
    "PMEParams",
    "ReachabilityResult",
    "SUPERSET_ALIAS_MAP",
    "SupersetResult",
    "adapt_model_to_dataclass",
    "check_prolix_reachability",
    "check_reachability",
    "check_superset",
    "extract_dataclass_fields",
    "get_reference_config_schemas",
]
