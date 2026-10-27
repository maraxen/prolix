"""Governance tests for FR-1: canonical KnobSurface & redsox U1/U5 static checks."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
import importlib.util
from pathlib import Path
import sys
import tomllib
import types

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"


def _ensure_importable() -> None:
    """Allow running under system python3 when JAX is not installed in sandbox."""
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    if "prolix" not in sys.modules:
        try:
            import prolix  # noqa: F401
        except ImportError:
            pkg = types.ModuleType("prolix")
            pkg.__path__ = [str(_SRC_ROOT / "prolix")]
            sys.modules["prolix"] = pkg
            phys_pkg = types.ModuleType("prolix.physics")
            phys_pkg.__path__ = [str(_SRC_ROOT / "prolix" / "physics")]
            sys.modules["prolix.physics"] = phys_pkg


_ensure_importable()

from prolix.governance import (  # noqa: E402
    DEFAULT_CONSUMER_PATHS,
    DEFAULT_EXCLUDED_DECLARATION_MODULES,
    DOCUMENTED_INERT_KNOBS,
    PENDING_EPIC_KNOBS,
    PENDING_EPIC_KNOBS_BY_ITEM,
    SUPERSET_ALIAS_MAP,
    ImplicitSolventParams,
    PMEParams,
    adapt_model_to_dataclass,
    check_prolix_reachability,
    check_reachability,
    check_superset,
    extract_dataclass_fields,
    get_reference_config_schemas,
)
from prolix.knob_surface import (  # noqa: E402
    BarostatControls,
    CollectiveVariableBiasControls,
    ConstraintControls,
    CosolventControls,
    ImplicitSolventControls,
    KnobSurface,
    NonbondedControls,
    PallasControls,
    SoftcoreRestraintControls,
    ThermostatControls,
)
from prolix.physics.nonbonded_config import (  # noqa: E402
    NeighborListConfig,
    NonbondedConfig,
)


EXPECTED_SUB_SURFACES: dict[type, set[str]] = {
    ThermostatControls: {
        "integrator_method",
        "temperature_k",
        "gamma_inv_ps",
        "timestep_fs",
        "n_steps",
        "save_interval",
        "seed",
    },
    BarostatControls: {
        "pressure_atm",
        "barostat_method",
        "barostat_interval",
        "anisotropic",
        "compressibility_inv_bar",
        "tau_p_ps",
    },
    NonbondedControls: {
        "nonbonded_method",
        "cutoff_angstrom",
        "switch_width_angstrom",
        "ewald_error_tolerance",
        "pme_alpha",
        "pme_grid_dims",
        "dispersion_correction",
        "reaction_field_dielectric",
        "dsf_alpha",
        "coulomb_constant",
        "scale_14_lj",
        "scale_14_elec",
    },
    ImplicitSolventControls: {
        "solvent_model",
        "solvent_dielectric",
        "solute_dielectric",
        "obc_alpha",
        "obc_beta",
        "obc_gamma",
        "surface_tension_kcal_mol_a2",
        "probe_radius_angstrom",
        "dielectric_offset_angstrom",
        "salt_concentration_m",
        "kappa_inv_angstrom",
    },
    ConstraintControls: {
        "constraint_mode",
        "rigid_water",
        "r_oh_angstrom",
        "r_hh_angstrom",
        "shake_tolerance",
        "rattle_tolerance",
        "max_shake_iterations",
    },
    CosolventControls: {
        "cosolvent_entries",
        "protein_radius_angstrom",
        "cosolvent_placement_radius_angstrom",
        "repulsive_eps_kcal",
        "repulsive_r0_angstrom",
        "repulsive_cutoff_angstrom",
        "deduplicate_interaction_groups",
    },
    SoftcoreRestraintControls: {
        "lambda_schedule",
        "softcore_alpha",
        "softcore_exponent",
        "soft_coulomb",
        "restraint_selection",
        "restraint_k_kcal_mol_a2",
        "flat_bottom_radius_angstrom",
        "per_stage_max_iterations",
        "tolerance_kcal_mol_a",
    },
    CollectiveVariableBiasControls: {
        "cv_specs",
        "bias_mode",
        "umbrella_k",
        "umbrella_center",
        "smd_velocity",
        "metad_height",
        "metad_sigma",
        "metad_bias_factor",
        "metad_pace",
        "plumed_min_beta_nm",
    },
    PallasControls: {
        "use_pallas",
        "use_pallas_pme",
        "gb_use_neighbor_list",
    },
}


def test_knob_surface_structure_and_flat_roundtrip() -> None:
    """Verify all 9 sub-surfaces and root KnobSurface leaf fields and helpers."""
    assert len(KnobSurface.sub_surface_classes()) == 9
    for sub_cls, expected_names in EXPECTED_SUB_SURFACES.items():
        actual_names = {f.name for f in fields(sub_cls)}
        assert actual_names == expected_names, (
            f"{sub_cls.__name__} mismatch: {actual_names ^ expected_names}"
        )

    all_expected_leaves = set().union(*EXPECTED_SUB_SURFACES.values())
    assert KnobSurface.leaf_field_names() == all_expected_leaves

    surface = KnobSurface()
    assert surface.leaf_field_names() == all_expected_leaves

    flat = surface.to_flat_dict()
    assert set(flat.keys()) == all_expected_leaves
    assert flat["temperature_k"] == 300.0
    assert flat["barostat_interval"] == 25
    assert flat["cutoff_angstrom"] == 9.0
    assert flat["use_pallas"] is None
    assert flat["use_pallas_pme"] is None
    assert flat["gb_use_neighbor_list"] is False

    # Attribute delegation from root KnobSurface to sub-surfaces
    assert surface.temperature_k == 300.0
    assert surface.plumed_min_beta_nm == 20.0
    assert surface.gb_use_neighbor_list is False

    reconstructed = KnobSurface.from_flat_dict(
        {"temperature_k": 310.0, "use_pallas": True, "barostat_interval": 10}
    )
    assert reconstructed.thermostat.temperature_k == 310.0
    assert reconstructed.pallas.use_pallas is True
    assert reconstructed.barostat.barostat_interval == 10


def test_knob_surface_zero_cycle_leaf_import_purity() -> None:
    """Verify knob_surface.py is a zero-cycle leaf and import graph remains clean."""
    ks_path = _SRC_ROOT / "prolix" / "knob_surface.py"
    tree = ast.parse(ks_path.read_text(encoding="utf-8"), filename=str(ks_path))

    allowed_imports = {
        "__future__",
        "dataclasses",
        "typing",
        "collections.abc",
        "prolix.typing",
    }
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    disallowed = imported_modules - allowed_imports
    assert not disallowed, f"knob_surface.py imported disallowed modules: {disallowed}"

    # Verify repo-wide import architecture guardrails (cycles, dodges, lazy imports)
    script_path = _REPO_ROOT / "scripts" / "ci" / "import_graph.py"
    spec = importlib.util.spec_from_file_location("prolix_import_graph", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    graph = mod.build(_SRC_ROOT / "prolix")

    assert graph.cycles == [], f"Import cycle(s) introduced: {graph.cycles}"
    assert graph.top.get("prolix.knob_surface", set()) <= {"prolix.typing"}
    assert "prolix.knob_surface" not in graph.lazy
    assert "prolix.governance" not in graph.lazy
    assert "prolix.governance.pending_knobs" not in graph.lazy
    assert "prolix.governance.redsox_adapter" not in graph.lazy


def test_pydantic_base_model_adapter() -> None:
    """Verify adapt_model_to_dataclass converts Pydantic v2 model_fields classes."""
    schemas = get_reference_config_schemas()
    for name in ("CosolvConfig", "CosolvEntry"):
        cls = schemas[name]
        adapted = adapt_model_to_dataclass(cls)
        assert hasattr(adapted, "__dataclass_fields__")
        extracted = extract_dataclass_fields(cls)
        assert len(extracted) > 0
        if hasattr(cls, "model_fields"):
            assert extracted == set(cls.model_fields.keys())


def test_u5_superset_parity_across_all_reference_schemas() -> None:
    """Run U5 check_superset against all Prolix and naurmalade config schemas."""
    schemas = get_reference_config_schemas()
    expected_schema_keys = {
        "SimulationConfig",
        "EquilibrationConfig",
        "MinimizationConfig",
        "ProductionConfig",
        "ImplicitSolventConfig",
        "HMRConfig",
        "RestraintSpec",
        "LambdaSchedule",
        "SoftcoreSweepStep",
        "CosolvConfig",
        "CosolvEntry",
        "SimulationSpec",
        "NonbondedConfig",
        "NeighborListConfig",
        "PMEParams",
        "ImplicitSolventParams",
    }
    assert expected_schema_keys <= set(schemas.keys())

    for name, schema_cls in schemas.items():
        res = check_superset(KnobSurface, schema_cls, SUPERSET_ALIAS_MAP)
        assert res.is_superset, (
            f"U5 superset parity failed for {name}: missing={sorted(res.missing_fields)}"
        )

    # Combined multi-class check
    combined = check_superset(
        KnobSurface, list(schemas.values()), SUPERSET_ALIAS_MAP
    )
    assert combined.is_superset
    assert combined.missing_fields == frozenset()

    # Negative control: an unmapped vendor field must fail check_superset
    @dataclass(frozen=True)
    class _UnmappedVendorConfig:
        temperature: float = 300.0
        unmapped_alien_knob_xyz: int = 42

    neg = check_superset(KnobSurface, _UnmappedVendorConfig, SUPERSET_ALIAS_MAP)
    assert not neg.is_superset
    assert neg.missing_fields == frozenset({"unmapped_alien_knob_xyz"})


def test_redsox_toml_configuration_parity() -> None:
    """Verify redsox.toml defines reachability, superset_parity, and alias map."""
    toml_path = _REPO_ROOT / "redsox.toml"
    assert toml_path.exists()
    cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))

    assert "reachability" in cfg
    assert "superset_parity" in cfg
    assert "superset_alias_map" in cfg

    reach = cfg["reachability"]
    assert list(reach["consumer_paths"]) == list(DEFAULT_CONSUMER_PATHS)
    assert set(reach["exclude_files"]) == set(DEFAULT_EXCLUDED_DECLARATION_MODULES)

    toml_aliases = dict(cfg["superset_alias_map"])
    assert toml_aliases == SUPERSET_ALIAS_MAP
    leaf_names = KnobSurface.leaf_field_names()
    for vendor_field, canonical_leaf in toml_aliases.items():
        assert canonical_leaf in leaf_names, (
            f"Alias {vendor_field!r} -> {canonical_leaf!r} not in KnobSurface leaves"
        )


def test_u1_consumer_only_reachability() -> None:
    """Verify U1 consumer-only reachability excludes declaration files and bounds unread knobs."""
    result = check_prolix_reachability(_REPO_ROOT)

    scanned_names = {Path(p).name for p in result.scanned_files}
    assert len(result.scanned_files) > 0
    assert scanned_names.isdisjoint(DEFAULT_EXCLUDED_DECLARATION_MODULES), (
        f"Declaration modules leaked into consumer scan: "
        f"{scanned_names & DEFAULT_EXCLUDED_DECLARATION_MODULES}"
    )

    unaccounted = set(result.unread_fields) - PENDING_EPIC_KNOBS
    assert unaccounted <= DOCUMENTED_INERT_KNOBS, (
        f"Untracked unread knobs outside PENDING_EPIC_KNOBS and DOCUMENTED_INERT_KNOBS: "
        f"{sorted(unaccounted - DOCUMENTED_INERT_KNOBS)}"
    )
    assert result.passed

    # Also verify NeighborListConfig specifically exposes its 3 documented inert knobs
    nl_res = check_reachability(
        NeighborListConfig,
        DEFAULT_CONSUMER_PATHS,
        repo_root=_REPO_ROOT,
        exclude_files=DEFAULT_EXCLUDED_DECLARATION_MODULES,
    )
    assert (
        set(nl_res.unread_fields) - PENDING_EPIC_KNOBS
        == {"safety_factor", "max_exclusions", "on_overflow"}
    )
    assert set(nl_res.unread_fields) - PENDING_EPIC_KNOBS <= DOCUMENTED_INERT_KNOBS

    # Negative control: synthetic dead knob not in PENDING_EPIC_KNOBS or DOCUMENTED_INERT_KNOBS
    @dataclass(frozen=True)
    class _SyntheticDeadConfig:
        n_steps: int = 100
        dead_unwired_knob_xyz: float = 1.0

    neg_u1 = check_reachability(
        _SyntheticDeadConfig,
        DEFAULT_CONSUMER_PATHS,
        repo_root=_REPO_ROOT,
        exclude_files=DEFAULT_EXCLUDED_DECLARATION_MODULES,
    )
    assert "n_steps" in neg_u1.read_fields
    assert "dead_unwired_knob_xyz" in neg_u1.unread_fields
    assert not neg_u1.passed
    assert len(PENDING_EPIC_KNOBS_BY_ITEM) > 0
    assert PENDING_EPIC_KNOBS == frozenset(), (
        f"Tier 3 requires PENDING_EPIC_KNOBS == frozenset(), got {sorted(PENDING_EPIC_KNOBS)}"
    )


if __name__ == "__main__":
    test_knob_surface_structure_and_flat_roundtrip()
    test_knob_surface_zero_cycle_leaf_import_purity()
    test_pydantic_base_model_adapter()
    test_u5_superset_parity_across_all_reference_schemas()
    test_redsox_toml_configuration_parity()
    test_u1_consumer_only_reachability()
    print("All governance U1/U5 and KnobSurface tests passed.")
