"""Tests for Prolix OpenMM configuration and platform shim (src/prolix/openmm/config.py).

Verifies:
1. Default instantiation of ProlixOpenMMConfig matches PROJECT.md / GEMINI.md standards.
2. Frozen dataclass immutability, equality, and hashability.
3. Strict __post_init__ validation across all 30 configuration parameters.
4. parse_platform_properties string parsing:
   - Case-insensitive key matching ("Precision", "precision", "PRECISION").
   - Boolean coercion ("true", "false", "1", "0", "yes", "no").
   - Integer and float conversions with invalid string rejection.
   - Unknown property detection.
5. ProlixPlatform OpenMM platform shim:
   - Platform registration in OpenMM.
   - getPropertyNames(), getPropertyDefaultValue(), getPropertyValue(), setPropertyValue().
6. Zero circular imports and adherence to leaf architecture.
"""

from __future__ import annotations

import dataclasses
import openmm
import pytest

from prolix.openmm import config
from prolix.openmm.config import (
    ProlixOpenMMConfig,
    ProlixPlatform,
    parse_platform_properties,
)


class TestDefaultConfig:
    """Verify default instantiation and immutability of ProlixOpenMMConfig."""

    def test_all_default_field_values(self):
        cfg = ProlixOpenMMConfig()

        # Precision
        assert cfg.precision == "single"

        # Nonbonded Engine & Pallas
        assert cfg.nonbonded_engine == "auto"
        assert cfg.use_pallas is False
        assert cfg.use_pallas_pme is False
        assert cfg.pallas_interpret is False

        # PME & LJ
        assert cfg.pme_grid_spacing == 1.0
        assert cfg.ewald_error_tolerance == 5e-4
        assert cfg.lj_switch_width == 1.5
        assert cfg.use_dispersion_correction is True

        # Neighbor List
        assert cfg.nl_skin == 2.0
        assert cfg.nl_capacity_multiplier == 1.5
        assert cfg.nl_update_every == 10
        assert cfg.disable_cell_list is False

        # Custom Forces
        assert cfg.custom_force_mode == "auto"
        assert cfg.custom_nonbonded_chunk_size == 128

        # Integrator & Thermostat
        assert cfg.integrator_scheme == "auto"
        assert cfg.thermostat_override is None
        assert cfg.remove_com_motion is True
        assert cfg.deterministic_seed == 42

        # Barostat
        assert cfg.barostat_mode == "auto"
        assert cfg.barostat_frequency_override is None

        # Constraints
        assert cfg.water_constraint_mode == "settle"
        assert cfg.solute_constraint_mode == "shake"
        assert cfg.shake_iterations == 25
        assert cfg.constraint_tolerance == 1e-5
        assert cfg.settle_velocity_tol == 1e-6

        # Compilation & Minimizer
        assert cfg.use_size_buckets is True
        assert cfg.max_steps_per_scan == 1000
        assert cfg.execution_mode == "inference"
        assert cfg.minimizer_backend == "lbfgs"

    def test_frozen_immutability(self):
        cfg = ProlixOpenMMConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.precision = "double"  # type: ignore[misc]

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.use_pallas = True  # type: ignore[misc]

    def test_equality_and_hashability(self):
        cfg1 = ProlixOpenMMConfig()
        cfg2 = ProlixOpenMMConfig()
        cfg3 = ProlixOpenMMConfig(precision="double")

        assert cfg1 == cfg2
        assert cfg1 != cfg3
        assert hash(cfg1) == hash(cfg2)
        # Can be used in sets or dict keys
        s = {cfg1, cfg3}
        assert len(s) == 2

    def test_custom_valid_construction(self):
        cfg = ProlixOpenMMConfig(
            precision="mixed",
            nonbonded_engine="flash",
            use_pallas=True,
            nl_skin=2.5,
            execution_mode="trajectory",
            max_steps_per_scan=500,
        )
        assert cfg.precision == "mixed"
        assert cfg.nonbonded_engine == "flash"
        assert cfg.use_pallas is True
        assert cfg.nl_skin == 2.5
        assert cfg.execution_mode == "trajectory"
        assert cfg.max_steps_per_scan == 500


class TestConfigValidation:
    """Verify strict __post_init__ validation across all axes."""

    @pytest.mark.parametrize("bad_val", ["half", "quad", "fp16", "", 123])
    def test_invalid_precision(self, bad_val):
        with pytest.raises(ValueError, match="precision"):
            ProlixOpenMMConfig(precision=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", ["fast", "opencl", "cuda", "cpu", None])
    def test_invalid_nonbonded_engine(self, bad_val):
        with pytest.raises(ValueError, match="nonbonded_engine"):
            ProlixOpenMMConfig(nonbonded_engine=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", [0.0, -0.5, -10.0])
    def test_invalid_pme_grid_spacing(self, bad_val):
        with pytest.raises(ValueError, match="pme_grid_spacing"):
            ProlixOpenMMConfig(pme_grid_spacing=bad_val)

    @pytest.mark.parametrize("bad_val", [0.0, -1e-4, 1.0, 1.5])
    def test_invalid_ewald_error_tolerance(self, bad_val):
        with pytest.raises(ValueError, match="ewald_error_tolerance"):
            ProlixOpenMMConfig(ewald_error_tolerance=bad_val)

    @pytest.mark.parametrize("bad_val", [-0.1, -1.0])
    def test_invalid_lj_switch_width(self, bad_val):
        with pytest.raises(ValueError, match="lj_switch_width"):
            ProlixOpenMMConfig(lj_switch_width=bad_val)

    @pytest.mark.parametrize("bad_val", [0.0, -1.0, -0.001])
    def test_invalid_nl_skin(self, bad_val):
        with pytest.raises(ValueError, match="nl_skin"):
            ProlixOpenMMConfig(nl_skin=bad_val)

    @pytest.mark.parametrize("bad_val", [1.0, 0.9, 0.0, -1.5])
    def test_invalid_nl_capacity_multiplier(self, bad_val):
        with pytest.raises(ValueError, match="nl_capacity_multiplier"):
            ProlixOpenMMConfig(nl_capacity_multiplier=bad_val)

    @pytest.mark.parametrize("bad_val", [0, -1, -10])
    def test_invalid_nl_update_every(self, bad_val):
        with pytest.raises(ValueError, match="nl_update_every"):
            ProlixOpenMMConfig(nl_update_every=bad_val)

    @pytest.mark.parametrize("bad_val", ["native", "inline", "all", "none"])
    def test_invalid_custom_force_mode(self, bad_val):
        with pytest.raises(ValueError, match="custom_force_mode"):
            ProlixOpenMMConfig(custom_force_mode=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", [0, -1, -64])
    def test_invalid_custom_nonbonded_chunk_size(self, bad_val):
        with pytest.raises(ValueError, match="custom_nonbonded_chunk_size"):
            ProlixOpenMMConfig(custom_nonbonded_chunk_size=bad_val)

    @pytest.mark.parametrize("bad_val", ["verlet", "euler", "rk4", "none"])
    def test_invalid_integrator_scheme(self, bad_val):
        with pytest.raises(ValueError, match="integrator_scheme"):
            ProlixOpenMMConfig(integrator_scheme=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", ["berendsen", "andersen", "parrinello-rahman"])
    def test_invalid_barostat_mode(self, bad_val):
        with pytest.raises(ValueError, match="barostat_mode"):
            ProlixOpenMMConfig(barostat_mode=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", [0, -1, -10])
    def test_invalid_barostat_frequency_override(self, bad_val):
        with pytest.raises(ValueError, match="barostat_frequency_override"):
            ProlixOpenMMConfig(barostat_frequency_override=bad_val)

    @pytest.mark.parametrize("bad_val", ["shake", "rattle", "all"])
    def test_invalid_water_constraint_mode(self, bad_val):
        with pytest.raises(ValueError, match="water_constraint_mode"):
            ProlixOpenMMConfig(water_constraint_mode=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", ["settle", "rattle", "all"])
    def test_invalid_solute_constraint_mode(self, bad_val):
        with pytest.raises(ValueError, match="solute_constraint_mode"):
            ProlixOpenMMConfig(solute_constraint_mode=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", [0, -1, -25])
    def test_invalid_shake_iterations(self, bad_val):
        with pytest.raises(ValueError, match="shake_iterations"):
            ProlixOpenMMConfig(shake_iterations=bad_val)

    @pytest.mark.parametrize("bad_val", [0.0, -1e-5])
    def test_invalid_constraint_tolerance(self, bad_val):
        with pytest.raises(ValueError, match="constraint_tolerance"):
            ProlixOpenMMConfig(constraint_tolerance=bad_val)

    @pytest.mark.parametrize("bad_val", [0.0, -1e-6])
    def test_invalid_settle_velocity_tol(self, bad_val):
        with pytest.raises(ValueError, match="settle_velocity_tol"):
            ProlixOpenMMConfig(settle_velocity_tol=bad_val)

    @pytest.mark.parametrize("bad_val", [0, -10, -1000])
    def test_invalid_max_steps_per_scan(self, bad_val):
        with pytest.raises(ValueError, match="max_steps_per_scan"):
            ProlixOpenMMConfig(max_steps_per_scan=bad_val)

    @pytest.mark.parametrize("bad_val", ["interactive", "batch", "scan", ""])
    def test_invalid_execution_mode(self, bad_val):
        with pytest.raises(ValueError, match="execution_mode"):
            ProlixOpenMMConfig(execution_mode=bad_val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_val", ["scipy", "bfgs", "cg", "gradient_descent"])
    def test_invalid_minimizer_backend(self, bad_val):
        with pytest.raises(ValueError, match="minimizer_backend"):
            ProlixOpenMMConfig(minimizer_backend=bad_val)  # type: ignore[arg-type]


class TestParsePlatformProperties:
    """Verify OpenMM string platformProperties parsing."""

    def test_none_or_empty_dict(self):
        assert parse_platform_properties(None) == ProlixOpenMMConfig()
        assert parse_platform_properties({}) == ProlixOpenMMConfig()

    @pytest.mark.parametrize("key", ["Precision", "precision", "PRECISION", "PrEcIsIoN"])
    def test_case_insensitive_key_parsing(self, key):
        cfg = parse_platform_properties({key: "double"})
        assert cfg.precision == "double"

    @pytest.mark.parametrize(
        ("val_str", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("Yes", True),
            ("YES", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("No", False),
            ("NO", False),
        ],
    )
    def test_boolean_coercion(self, val_str, expected):
        cfg = parse_platform_properties({"UsePallas": val_str})
        assert cfg.use_pallas is expected

    @pytest.mark.parametrize("bad_bool", ["maybe", "enabled", "disabled", "2", "foo"])
    def test_invalid_boolean_string(self, bad_bool):
        with pytest.raises(ValueError, match="boolean"):
            parse_platform_properties({"UsePallas": bad_bool})

    def test_integer_coercion(self):
        cfg = parse_platform_properties({
            "nl_update_every": "25",
            "shake_iterations": "50",
            "deterministic_seed": "12345",
        })
        assert cfg.nl_update_every == 25
        assert cfg.shake_iterations == 50
        assert cfg.deterministic_seed == 12345

    @pytest.mark.parametrize("bad_int", ["abc", "12.5", ""])
    def test_invalid_integer_string(self, bad_int):
        with pytest.raises(ValueError):
            parse_platform_properties({"shake_iterations": bad_int})

    def test_float_coercion(self):
        cfg = parse_platform_properties({
            "pme_grid_spacing": "1.25",
            "ewald_error_tolerance": "1e-5",
            "nl_skin": "2.8",
        })
        assert cfg.pme_grid_spacing == 1.25
        assert cfg.ewald_error_tolerance == 1e-5
        assert cfg.nl_skin == 2.8

    @pytest.mark.parametrize("bad_float", ["fast", "none", "1.2.3"])
    def test_invalid_float_string(self, bad_float):
        with pytest.raises(ValueError):
            parse_platform_properties({"pme_grid_spacing": bad_float})

    def test_unknown_property_rejection(self):
        with pytest.raises(ValueError, match="Unknown platform property"):
            parse_platform_properties({"NonExistentProperty": "val"})

    def test_comprehensive_mixed_properties_dict(self):
        props = {
            "Precision": "mixed",
            "nonbonded_engine": "neighbor_list",
            "USE_PALLAS": "true",
            "PmeGridSpacing": "0.8",
            "NL_SKIN": "2.2",
            "nl_capacity_multiplier": "1.75",
            "shake_iterations": "30",
            "deterministic_seed": "1337",
            "execution_mode": "trajectory",
        }
        cfg = parse_platform_properties(props)
        assert cfg.precision == "mixed"
        assert cfg.nonbonded_engine == "neighbor_list"
        assert cfg.use_pallas is True
        assert cfg.pme_grid_spacing == 0.8
        assert cfg.nl_skin == 2.2
        assert cfg.nl_capacity_multiplier == 1.75
        assert cfg.shake_iterations == 30
        assert cfg.deterministic_seed == 1337
        assert cfg.execution_mode == "trajectory"


class TestProlixPlatform:
    """Verify ProlixPlatform shim and registration."""

    def test_platform_registered_in_openmm(self):
        # Platform should be registered in openmm
        all_platforms = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
        assert "Prolix" in all_platforms

        platform = openmm.Platform.getPlatformByName("Prolix")
        assert platform.getName() == "Prolix"
        assert platform.getSpeed() > 0.0
        assert platform.supportsDoublePrecision() is True
        assert platform.supportsKernels() is True

    def test_get_property_names(self):
        platform = openmm.Platform.getPlatformByName("Prolix")
        names = platform.getPropertyNames()
        assert len(names) > 0
        # Check core expected names exist
        expected_keys = {
            "Precision",
            "NonbondedEngine",
            "UsePallas",
            "PmeGridSpacing",
            "NeighborListSkin",
        }
        assert expected_keys.issubset(set(names))

    def test_get_property_default_values(self):
        platform = openmm.Platform.getPlatformByName("Prolix")
        assert platform.getPropertyDefaultValue("Precision") == "single"
        assert platform.getPropertyDefaultValue("NonbondedEngine") == "auto"
        assert platform.getPropertyDefaultValue("UsePallas") in ("false", "False", "0")

    def test_get_property_default_value_unknown(self):
        platform = openmm.Platform.getPlatformByName("Prolix")
        with pytest.raises((openmm.OpenMMException, KeyError, ValueError)):
            platform.getPropertyDefaultValue("UnknownNonExistentProperty")

    def test_property_get_and_set(self):
        platform = openmm.Platform.getPlatformByName("Prolix")
        platform.setPropertyDefaultValue("Precision", "double")
        assert platform.getPropertyDefaultValue("Precision") == "double"
        # Reset back
        platform.setPropertyDefaultValue("Precision", "single")
        assert platform.getPropertyDefaultValue("Precision") == "single"

    def test_create_config(self):
        platform = openmm.Platform.getPlatformByName("Prolix")
        cfg = platform.create_config({"Precision": "double", "UsePallas": "true"})
        assert cfg.precision == "double"
        assert cfg.use_pallas is True


class TestArchitectureGuardrails:
    """Verify architectural boundaries: no circular imports and units.py is a leaf."""

    def test_units_is_leaf(self):
        import prolix.openmm.units as u_mod

        # units module should not import higher-level openmm modules
        assert not hasattr(u_mod, "ProlixOpenMMConfig")
        assert not hasattr(u_mod, "ProlixContext")
        assert not hasattr(u_mod, "from_openmm_system")

    def test_config_clean_import(self):
        import prolix.openmm.config as c_mod

        # config module should not import higher-level openmm modules
        assert not hasattr(c_mod, "ProlixContext")
        assert not hasattr(c_mod, "from_openmm_system")
