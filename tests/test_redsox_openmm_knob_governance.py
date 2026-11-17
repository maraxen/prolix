"""CI governance suite for Track 2: redsox U1 + U5 OpenMM knob parity and reachability.

Enforces:
1. U5 Superset Parity (`check_superset`):
   - `KnobSurface` covers 100% of `ProlixOpenMMConfig` (39 fields),
     `OpenMMPlatformPropertiesSchema`, `OpenMMSystemKnobsSchema`, and
     `OpenMMIntegratorKnobsSchema` with 0 missing knobs.
   - `KnobSurface` covers all registered reference config schemas in
     `get_reference_config_schemas()` with 0 missing knobs.
2. U1 Consumer AST Reachability (`check_prolix_reachability`):
   - 100% of `KnobSurface` leaf fields (including promoted `precision`,
     `nb_k_splits`, `tp_group`, `gb_use_neighbor_list`, `deterministic_forces`,
     `disable_pme_stream`, `device_index`, `cmm_frequency`,
     `anisotropic_barostat`, `virtual_site_mode`) are read in consumer ASTs
     (`src/prolix/openmm/`, `src/prolix/physics/`, `src/prolix/api/`, etc.)
     while declaration modules (`config.py`, `knob_surface.py`, etc.) are excluded.
   - `"precision"` is no longer in `DOCUMENTED_INERT_KNOBS` and
     `PENDING_EPIC_KNOBS == frozenset()`.
3. Physical Constant & Platform Property Round-Tripping:
   - `KnobSurface().nonbonded.coulomb_constant` equals NIST CODATA 2018
     `332.06371329919216` kcal*A/(mol*e^2).
   - Every canonical OpenMM PascalCase property and extra alias (`Precision`,
     `CudaPrecision`, `DeterministicForces`, `DisablePmeStream`, `DeviceIndex`,
     `NbKSplits`, `TpGroup`, `GbUseNeighborList`, `CmmFrequency`,
     `AnisotropicBarostat`, `VirtualSiteMode`) round-trips through
     `ProlixPlatform` and `parse_platform_properties`.
"""

from __future__ import annotations

import dataclasses

from prolix.governance.pending_knobs import DOCUMENTED_INERT_KNOBS, PENDING_EPIC_KNOBS
from prolix.governance.redsox_adapter import (
    DEFAULT_CONSUMER_PATHS,
    DEFAULT_EXCLUDED_DECLARATION_MODULES,
    SUPERSET_ALIAS_MAP,
    OpenMMIntegratorKnobsSchema,
    OpenMMPlatformPropertiesSchema,
    OpenMMSystemKnobsSchema,
    check_prolix_reachability,
    check_superset,
    get_reference_config_schemas,
)
from prolix.knob_surface import KnobSurface
from prolix.openmm.config import (
    _CANONICAL_NAMES,
    _EXTRA_ALIASES,
    ProlixOpenMMConfig,
    ProlixPlatform,
    parse_platform_properties,
)


def test_precision_promoted_from_documented_inert() -> None:
    """`precision` must be an active governed knob, not an inert exemption."""
    assert "precision" not in DOCUMENTED_INERT_KNOBS
    assert "precision" in KnobSurface.leaf_field_names()
    assert PENDING_EPIC_KNOBS == frozenset()


def test_coulomb_constant_matches_nist_codata_2018() -> None:
    """`KnobSurface.nonbonded.coulomb_constant` must match OpenMM / CODATA 2018."""
    ks = KnobSurface()
    assert abs(ks.nonbonded.coulomb_constant - 332.06371329919216) < 1e-12


def test_consumer_paths_include_openmm_and_exclude_config_declarations() -> None:
    """`src/prolix/openmm` is scanned as a consumer while `config.py` is excluded."""
    assert "src/prolix/openmm" in DEFAULT_CONSUMER_PATHS
    assert "config.py" in DEFAULT_EXCLUDED_DECLARATION_MODULES
    assert "knob_surface.py" in DEFAULT_EXCLUDED_DECLARATION_MODULES


def test_u5_superset_openmm_schemas_zero_missing() -> None:
    """U5: `KnobSurface` must cover all OpenMM config, platform, system, and integrator schemas."""
    targets = {
        "ProlixOpenMMConfig": ProlixOpenMMConfig,
        "OpenMMPlatformPropertiesSchema": OpenMMPlatformPropertiesSchema,
        "OpenMMSystemKnobsSchema": OpenMMSystemKnobsSchema,
        "OpenMMIntegratorKnobsSchema": OpenMMIntegratorKnobsSchema,
    }
    for name, schema in targets.items():
        res = check_superset(
            KnobSurface,
            schema,
            alias_map=SUPERSET_ALIAS_MAP,
            inert_knobs=DOCUMENTED_INERT_KNOBS,
        )
        assert res.passed, f"U5 failed for {name}: missing={res.missing_fields}"
        assert res.missing_fields == frozenset()


def test_u5_superset_all_reference_schemas() -> None:
    """U5: `KnobSurface` must cover every registered reference config schema."""
    schemas = get_reference_config_schemas()
    assert "ProlixOpenMMConfig" in schemas
    assert "OpenMMPlatformPropertiesSchema" in schemas
    assert "OpenMMSystemKnobsSchema" in schemas
    assert "OpenMMIntegratorKnobsSchema" in schemas

    for name, schema in schemas.items():
        res = check_superset(
            KnobSurface,
            schema,
            alias_map=SUPERSET_ALIAS_MAP,
            inert_knobs=DOCUMENTED_INERT_KNOBS,
        )
        assert res.passed, f"U5 failed for {name}: missing={res.missing_fields}"


def test_u1_consumer_reachability_100_percent() -> None:
    """U1: Every leaf of `KnobSurface` and `ProlixOpenMMConfig` must be AST-reachable in consumers."""
    res = check_prolix_reachability(KnobSurface)
    assert res.passed, f"U1 failed for KnobSurface with unread leaves: {sorted(res.unread_fields)}"
    assert res.unread_fields == frozenset()

    required_active_surface = {
        "precision",
        "nb_k_splits",
        "tp_group",
        "gb_use_neighbor_list",
        "deterministic_forces",
        "disable_pme_stream",
        "cmm_frequency",
        "anisotropic",
        "virtual_site_mode",
        "ewald_error_tolerance",
    }
    assert required_active_surface.issubset(res.read_fields)

    res_omm = check_prolix_reachability(ProlixOpenMMConfig)
    assert res_omm.passed, (
        f"U1 failed for ProlixOpenMMConfig with unread leaves: {sorted(res_omm.unread_fields)}"
    )
    assert res_omm.unread_fields == frozenset()
    assert {"device_index", "anisotropic_barostat"}.issubset(res_omm.read_fields)


def test_openmm_platform_property_roundtrip() -> None:
    """All 39 `ProlixOpenMMConfig` fields and OpenMM platform aliases round-trip cleanly."""
    field_names = [f.name for f in dataclasses.fields(ProlixOpenMMConfig)]
    assert len(field_names) == 39
    assert set(field_names) == set(_CANONICAL_NAMES.keys())

    plat = ProlixPlatform()
    prop_names = plat.getPropertyNames()
    assert len(prop_names) == 39
    assert set(prop_names) == set(_CANONICAL_NAMES.values())

    roundtrip_props = {
        "Precision": "mixed",
        "NonbondedEngine": "morton_tile_pairs",
        "UsePallas": "true",
        "UsePallasPME": "true",
        "NbKSplits": "4",
        "TpGroup": "4",
        "GbUseNeighborList": "false",
        "DeterministicForces": "true",
        "DisablePmeStream": "true",
        "DeviceIndex": "0",
        "EwaldErrorTolerance": "1e-4",
        "CmmFrequency": "50",
        "AnisotropicBarostat": "true",
        "VirtualSiteMode": "auto",
    }
    for k, v in roundtrip_props.items():
        plat.setPropertyDefaultValue(k, v)
        assert plat.getPropertyDefaultValue(k) == v

    cfg = plat.create_config()
    assert cfg.precision == "mixed"
    assert cfg.nonbonded_engine == "morton_tile_pairs"
    assert cfg.use_pallas is True
    assert cfg.use_pallas_pme is True
    assert cfg.nb_k_splits == 4
    assert cfg.tp_group == 4
    assert cfg.gb_use_neighbor_list is False
    assert cfg.deterministic_forces is True
    assert cfg.disable_pme_stream is True
    assert cfg.device_index == "0"
    assert abs(cfg.ewald_error_tolerance - 1e-4) < 1e-12
    assert cfg.cmm_frequency == 50
    assert cfg.anisotropic_barostat is True
    assert cfg.virtual_site_mode == "auto"

    # Also verify CUDA/OpenCL platform alias compatibility
    alias_cfg = parse_platform_properties(
        {
            "CudaPrecision": "mixed",
            "CudaDeviceIndex": "0,1",
            "DeterministicForces": "true",
            "DisablePmeStream": "false",
        }
    )
    assert alias_cfg.precision == "mixed"
    assert alias_cfg.device_index == "0,1"
    assert alias_cfg.deterministic_forces is True
    assert alias_cfg.disable_pme_stream is False
    assert "cudaprecision" in _EXTRA_ALIASES


if __name__ == "__main__":
    test_precision_promoted_from_documented_inert()
    test_coulomb_constant_matches_nist_codata_2018()
    test_consumer_paths_include_openmm_and_exclude_config_declarations()
    test_u5_superset_openmm_schemas_zero_missing()
    test_u5_superset_all_reference_schemas()
    test_u1_consumer_reachability_100_percent()
    test_openmm_platform_property_roundtrip()
    print("ALL test_redsox_openmm_knob_governance checks PASSED.")
