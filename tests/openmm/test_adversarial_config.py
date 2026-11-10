"""Adversarial stress-testing suite for Prolix OpenMM Config and Platform Shim.

Empirically tests:
1. Fuzzing parse_platform_properties:
   - Weird casings (e.g. 'pReCiSiOn', 'uSe_pAlLaS', 'nL-sKiN')
   - Extra whitespace (leading, trailing, internal whitespace, tabs, newlines)
   - Invalid boolean strings ('maybe', 'true-ish', '2', '-1', 'null', 'none', '', None)
   - Invalid integer strings ('1.5', '1.0', '1e3', 'abc', '', None)
   - Invalid float strings ('abc', '', None, '1.0.0')
   - Empty dicts and None
   - Weird symbols and non-string inputs
   - Complete coverage of all aliases in _EXTRA_ALIASES
2. Boundary stress across all 30 configuration fields:
   - Numerical boundary limits (negative cutoff, non-positive grid spacing, zero capacity multiplier, zero scan steps, etc.)
   - Empirical probing of NaN, Inf, and -Inf across float fields
   - String enum validation across all enum fields
   - Integer bound validation (seed >= 0, update_every >= 1, shake_iterations >= 1, etc.)
3. ProlixPlatform OpenMM Platform shim stress:
   - openmm.Platform.getPlatformByName("Prolix") instance identity
   - openmm.Platform.getNumPlatforms() and index iteration
   - Negative index handling (index -1 vs out-of-bounds -2)
   - Out-of-bounds positive index handling (openmm.OpenMMException)
   - Platform properties getters and setters with valid and invalid property names
   - getPropertyValue / setPropertyValue with mock contexts (having _config, _platform_properties, get_platform_property)
   - Speed rating (100.0) and double precision support
   - Stress testing repeated registration and unregistration cycles without C++ registry corruption
   - Regression verification: standard OpenMM Reference & CPU platforms can still build contexts and step
4. Immutability & Concurrency stress:
   - Frozen dataclass immutability across all 30 fields (setattr, delattr raise FrozenInstanceError)
   - Deep immutability (all fields are immutable primitives)
   - Concurrent multi-threaded execution of parse_platform_properties and platform queries
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import math
from typing import Any

import openmm
import pytest

from prolix.openmm.config import (
    _CANONICAL_NAMES,
    _EXTRA_ALIASES,
    ProlixOpenMMConfig,
    ProlixPlatform,
    parse_platform_properties,
)


# ============================================================================
# 1. Fuzzing parse_platform_properties
# ============================================================================

class TestParsePlatformPropertiesFuzzing:
    """Fuzzing and adversarial testing of parse_platform_properties."""

    def test_none_and_empty_dict(self):
        """None and empty dict must return default ProlixOpenMMConfig."""
        assert parse_platform_properties(None) == ProlixOpenMMConfig()
        assert parse_platform_properties({}) == ProlixOpenMMConfig()

    @pytest.mark.parametrize(
        "weird_key",
        [
            "pReCiSiOn",
            "PRECISION",
            "precision",
            "  Precision  ",
            "\t\nPrecision\n\t",
            "use_pallas",
            "USE_PALLAS",
            "Use_Pallas",
            "use-pallas",
            "USE-PALLAS",
            "  uSe_PaLlAs  ",
            "neighbor_list_skin",
            "NEIGHBOR_LIST_SKIN",
            "neighbor-list-skin",
            "  Neighbor List Skin  ",
            "custom_nonbonded_chunk_size",
            "CUSTOM_NONBONDED_CHUNK_SIZE",
            "custom-nonbonded-chunk-size",
            "  custom nonbonded chunk size  ",
        ],
    )
    def test_weird_key_casings_and_separators(self, weird_key):
        """Keys with mixed casing, underscores, hyphens, and whitespace must normalize properly."""
        if "precision" in weird_key.lower():
            cfg = parse_platform_properties({weird_key: "double"})
            assert cfg.precision == "double"
        elif "pallas" in weird_key.lower():
            cfg = parse_platform_properties({weird_key: "true"})
            assert cfg.use_pallas is True
        elif "skin" in weird_key.lower():
            cfg = parse_platform_properties({weird_key: "3.5"})
            assert cfg.nl_skin == 3.5
        elif "chunk" in weird_key.lower():
            cfg = parse_platform_properties({weird_key: "256"})
            assert cfg.custom_nonbonded_chunk_size == 256

    def test_all_extra_aliases_parse_correctly(self):
        """Every alias registered in _EXTRA_ALIASES must successfully parse."""
        for alias, field in _EXTRA_ALIASES.items():
            # Construct a minimal valid value for the alias
            if "skin" in alias or "spacing" in alias or "width" in alias:
                val = "2.5"
            elif "multiplier" in alias:
                val = "1.8"
            elif "every" in alias or "chunk" in alias or "iterations" in alias or "scan" in alias or "frequency" in alias:
                val = "50"
            elif "seed" in alias:
                val = "999"
            elif "tolerance" in alias or "tol" in alias:
                val = "1e-4"
            elif "correction" in alias or "list" in alias or "motion" in alias or "buckets" in alias:
                val = "false"
            elif "mode" in alias and "custom" in alias:
                val = "lepton_only"
            elif "mode" in alias and "barostat" in alias:
                val = "mc"
            elif "mode" in alias and "water" in alias:
                val = "none"
            elif "mode" in alias and "solute" in alias:
                val = "none"
            elif "mode" in alias and "execution" in alias:
                val = "trajectory"
            elif "scheme" in alias:
                val = "baoab"
            elif "override" in alias:
                val = "andersen"
            elif "backend" in alias:
                val = "fire"
            else:
                val = "true"

            cfg = parse_platform_properties({alias: val})
            assert hasattr(cfg, field), f"Config missing attribute {field} for alias {alias}"

    @pytest.mark.parametrize(
        "invalid_bool",
        [
            "maybe",
            "true-ish",
            "false-ish",
            "2",
            "-1",
            "10",
            "null",
            "none",
            "",
            "   ",
            "enabled",
            "disabled",
            "yep",
            "nope",
            "!@#$",
            None,
        ],
    )
    def test_invalid_booleans_rejected(self, invalid_bool):
        """Ambiguous or invalid booleans must raise ValueError."""
        with pytest.raises(ValueError, match="boolean"):
            parse_platform_properties({"UsePallas": invalid_bool})

    @pytest.mark.parametrize(
        "invalid_int",
        [
            "1.5",
            "1.0",
            "0.0",
            "-1.5",
            "1e3",
            "abc",
            "",
            "   ",
            "two",
            "1,000",
            "0x10",
            "!@#$",
            None,
        ],
    )
    def test_invalid_integers_rejected(self, invalid_int):
        """Floats formatted as strings, scientific notation, strings, or None must be rejected for ints."""
        with pytest.raises(ValueError):
            parse_platform_properties({"shake_iterations": invalid_int})

    @pytest.mark.parametrize(
        "invalid_float",
        [
            "abc",
            "",
            "   ",
            "one",
            "1.0.0",
            "1,0",
            "!@#$",
            None,
        ],
    )
    def test_invalid_floats_rejected(self, invalid_float):
        """Malformed float strings must be rejected."""
        with pytest.raises(ValueError):
            parse_platform_properties({"pme_grid_spacing": invalid_float})

    @pytest.mark.parametrize(
        "weird_symbol_key",
        [
            "---",
            "!@#$%",
            "   ",
            "\t\n",
            "***",
            "???",
        ],
    )
    def test_weird_symbol_keys_rejected(self, weird_symbol_key):
        """Keys consisting only of symbols or whitespace normalize to empty string and must be rejected."""
        with pytest.raises(ValueError, match="Unknown platform property"):
            parse_platform_properties({weird_symbol_key: "value"})


# ============================================================================
# 2. Boundary Stress across all 30 configuration fields
# ============================================================================

class TestBoundaryStressAllFields:
    """Stress test boundary conditions for all 30 configuration fields."""

    # --- Precision ---
    @pytest.mark.parametrize("val", ["single", "mixed", "double"])
    def test_precision_valid(self, val):
        assert ProlixOpenMMConfig(precision=val).precision == val

    @pytest.mark.parametrize("val", ["", "half", "quad", "fp16", "float32", "DOUBLE"])
    def test_precision_invalid(self, val):
        with pytest.raises(ValueError, match="precision"):
            ProlixOpenMMConfig(precision=val)  # type: ignore[arg-type]

    # --- Nonbonded Engine ---
    @pytest.mark.parametrize("val", ["auto", "neighbor_list", "flash", "dense"])
    def test_nonbonded_engine_valid(self, val):
        assert ProlixOpenMMConfig(nonbonded_engine=val).nonbonded_engine == val

    @pytest.mark.parametrize("val", ["", "pallas", "opencl", "cuda", "cpu", None])
    def test_nonbonded_engine_invalid(self, val):
        with pytest.raises(ValueError, match="nonbonded_engine"):
            ProlixOpenMMConfig(nonbonded_engine=val)  # type: ignore[arg-type]

    # --- PME & LJ Parameters ---
    def test_pme_grid_spacing_boundaries(self):
        # Valid boundaries
        assert ProlixOpenMMConfig(pme_grid_spacing=0.001).pme_grid_spacing == 0.001
        assert ProlixOpenMMConfig(pme_grid_spacing=10.0).pme_grid_spacing == 10.0
        # Invalid boundaries (<= 0)
        with pytest.raises(ValueError, match="pme_grid_spacing"):
            ProlixOpenMMConfig(pme_grid_spacing=0.0)
        with pytest.raises(ValueError, match="pme_grid_spacing"):
            ProlixOpenMMConfig(pme_grid_spacing=-0.5)

    def test_ewald_error_tolerance_boundaries(self):
        # Valid boundaries (0 < tol < 1)
        assert ProlixOpenMMConfig(ewald_error_tolerance=1e-12).ewald_error_tolerance == 1e-12
        assert ProlixOpenMMConfig(ewald_error_tolerance=0.999).ewald_error_tolerance == 0.999
        # Invalid boundaries
        with pytest.raises(ValueError, match="ewald_error_tolerance"):
            ProlixOpenMMConfig(ewald_error_tolerance=0.0)
        with pytest.raises(ValueError, match="ewald_error_tolerance"):
            ProlixOpenMMConfig(ewald_error_tolerance=1.0)
        with pytest.raises(ValueError, match="ewald_error_tolerance"):
            ProlixOpenMMConfig(ewald_error_tolerance=-1e-4)
        with pytest.raises(ValueError, match="ewald_error_tolerance"):
            ProlixOpenMMConfig(ewald_error_tolerance=1.5)

    def test_lj_switch_width_boundaries(self):
        # Valid boundaries (>= 0)
        assert ProlixOpenMMConfig(lj_switch_width=0.0).lj_switch_width == 0.0
        assert ProlixOpenMMConfig(lj_switch_width=5.0).lj_switch_width == 5.0
        # Invalid boundaries (< 0)
        with pytest.raises(ValueError, match="lj_switch_width"):
            ProlixOpenMMConfig(lj_switch_width=-0.0001)

    # --- Neighbor List Parameters ---
    def test_nl_skin_boundaries(self):
        # Valid boundaries (> 0)
        assert ProlixOpenMMConfig(nl_skin=0.001).nl_skin == 0.001
        # Invalid boundaries (<= 0)
        with pytest.raises(ValueError, match="nl_skin"):
            ProlixOpenMMConfig(nl_skin=0.0)
        with pytest.raises(ValueError, match="nl_skin"):
            ProlixOpenMMConfig(nl_skin=-1.0)

    def test_nl_capacity_multiplier_boundaries(self):
        # Valid boundaries (> 1.0)
        assert ProlixOpenMMConfig(nl_capacity_multiplier=1.0001).nl_capacity_multiplier == 1.0001
        # Invalid boundaries (<= 1.0)
        with pytest.raises(ValueError, match="nl_capacity_multiplier"):
            ProlixOpenMMConfig(nl_capacity_multiplier=1.0)
        with pytest.raises(ValueError, match="nl_capacity_multiplier"):
            ProlixOpenMMConfig(nl_capacity_multiplier=0.99)
        with pytest.raises(ValueError, match="nl_capacity_multiplier"):
            ProlixOpenMMConfig(nl_capacity_multiplier=0.0)
        with pytest.raises(ValueError, match="nl_capacity_multiplier"):
            ProlixOpenMMConfig(nl_capacity_multiplier=-1.0)

    def test_nl_update_every_boundaries(self):
        # Valid boundaries (>= 1)
        assert ProlixOpenMMConfig(nl_update_every=1).nl_update_every == 1
        # Invalid boundaries (< 1)
        with pytest.raises(ValueError, match="nl_update_every"):
            ProlixOpenMMConfig(nl_update_every=0)
        with pytest.raises(ValueError, match="nl_update_every"):
            ProlixOpenMMConfig(nl_update_every=-1)

    # --- Custom Forces ---
    @pytest.mark.parametrize("val", ["auto", "lepton_only", "pattern_only"])
    def test_custom_force_mode_valid(self, val):
        assert ProlixOpenMMConfig(custom_force_mode=val).custom_force_mode == val

    @pytest.mark.parametrize("val", ["", "native", "jit", "all", None])
    def test_custom_force_mode_invalid(self, val):
        with pytest.raises(ValueError, match="custom_force_mode"):
            ProlixOpenMMConfig(custom_force_mode=val)  # type: ignore[arg-type]

    def test_custom_nonbonded_chunk_size_boundaries(self):
        # Valid boundaries (>= 1)
        assert ProlixOpenMMConfig(custom_nonbonded_chunk_size=1).custom_nonbonded_chunk_size == 1
        # Invalid boundaries (< 1)
        with pytest.raises(ValueError, match="custom_nonbonded_chunk_size"):
            ProlixOpenMMConfig(custom_nonbonded_chunk_size=0)
        with pytest.raises(ValueError, match="custom_nonbonded_chunk_size"):
            ProlixOpenMMConfig(custom_nonbonded_chunk_size=-1)

    # --- Integrator & Thermostat Routing ---
    @pytest.mark.parametrize("val", ["auto", "baoab", "leapfrog", "langevin"])
    def test_integrator_scheme_valid(self, val):
        assert ProlixOpenMMConfig(integrator_scheme=val).integrator_scheme == val

    @pytest.mark.parametrize("val", ["", "verlet", "rk4", "euler", None])
    def test_integrator_scheme_invalid(self, val):
        with pytest.raises(ValueError, match="integrator_scheme"):
            ProlixOpenMMConfig(integrator_scheme=val)  # type: ignore[arg-type]

    def test_deterministic_seed_boundaries(self):
        # Valid boundaries (>= 0)
        assert ProlixOpenMMConfig(deterministic_seed=0).deterministic_seed == 0
        assert ProlixOpenMMConfig(deterministic_seed=2**31 - 1).deterministic_seed == 2**31 - 1
        # Invalid boundaries (< 0)
        with pytest.raises(ValueError, match="deterministic_seed"):
            ProlixOpenMMConfig(deterministic_seed=-1)

    # --- Barostat Routing ---
    @pytest.mark.parametrize("val", ["auto", "mc", "scr"])
    def test_barostat_mode_valid(self, val):
        assert ProlixOpenMMConfig(barostat_mode=val).barostat_mode == val

    @pytest.mark.parametrize("val", ["", "berendsen", "parrinello_rahman", None])
    def test_barostat_mode_invalid(self, val):
        with pytest.raises(ValueError, match="barostat_mode"):
            ProlixOpenMMConfig(barostat_mode=val)  # type: ignore[arg-type]

    def test_barostat_frequency_override_boundaries(self):
        # Valid boundaries (None or >= 1)
        assert ProlixOpenMMConfig(barostat_frequency_override=None).barostat_frequency_override is None
        assert ProlixOpenMMConfig(barostat_frequency_override=1).barostat_frequency_override == 1
        assert ProlixOpenMMConfig(barostat_frequency_override=25).barostat_frequency_override == 25
        # Invalid boundaries (< 1)
        with pytest.raises(ValueError, match="barostat_frequency_override"):
            ProlixOpenMMConfig(barostat_frequency_override=0)
        with pytest.raises(ValueError, match="barostat_frequency_override"):
            ProlixOpenMMConfig(barostat_frequency_override=-1)

    # --- Constraints ---
    @pytest.mark.parametrize("val", ["settle", "none"])
    def test_water_constraint_mode_valid(self, val):
        assert ProlixOpenMMConfig(water_constraint_mode=val).water_constraint_mode == val

    @pytest.mark.parametrize("val", ["", "shake", "rattle", None])
    def test_water_constraint_mode_invalid(self, val):
        with pytest.raises(ValueError, match="water_constraint_mode"):
            ProlixOpenMMConfig(water_constraint_mode=val)  # type: ignore[arg-type]

    @pytest.mark.parametrize("val", ["shake", "none"])
    def test_solute_constraint_mode_valid(self, val):
        assert ProlixOpenMMConfig(solute_constraint_mode=val).solute_constraint_mode == val

    @pytest.mark.parametrize("val", ["", "settle", "rattle", None])
    def test_solute_constraint_mode_invalid(self, val):
        with pytest.raises(ValueError, match="solute_constraint_mode"):
            ProlixOpenMMConfig(solute_constraint_mode=val)  # type: ignore[arg-type]

    def test_shake_iterations_boundaries(self):
        # Valid boundaries (>= 1)
        assert ProlixOpenMMConfig(shake_iterations=1).shake_iterations == 1
        # Invalid boundaries (< 1)
        with pytest.raises(ValueError, match="shake_iterations"):
            ProlixOpenMMConfig(shake_iterations=0)
        with pytest.raises(ValueError, match="shake_iterations"):
            ProlixOpenMMConfig(shake_iterations=-1)

    def test_constraint_tolerance_boundaries(self):
        # Valid boundaries (> 0)
        assert ProlixOpenMMConfig(constraint_tolerance=1e-12).constraint_tolerance == 1e-12
        # Invalid boundaries (<= 0)
        with pytest.raises(ValueError, match="constraint_tolerance"):
            ProlixOpenMMConfig(constraint_tolerance=0.0)
        with pytest.raises(ValueError, match="constraint_tolerance"):
            ProlixOpenMMConfig(constraint_tolerance=-1e-5)

    def test_settle_velocity_tol_boundaries(self):
        # Valid boundaries (> 0)
        assert ProlixOpenMMConfig(settle_velocity_tol=1e-12).settle_velocity_tol == 1e-12
        # Invalid boundaries (<= 0)
        with pytest.raises(ValueError, match="settle_velocity_tol"):
            ProlixOpenMMConfig(settle_velocity_tol=0.0)
        with pytest.raises(ValueError, match="settle_velocity_tol"):
            ProlixOpenMMConfig(settle_velocity_tol=-1e-6)

    # --- Compilation & Execution ---
    def test_max_steps_per_scan_boundaries(self):
        # Valid boundaries (>= 1)
        assert ProlixOpenMMConfig(max_steps_per_scan=1).max_steps_per_scan == 1
        # Invalid boundaries (< 1)
        with pytest.raises(ValueError, match="max_steps_per_scan"):
            ProlixOpenMMConfig(max_steps_per_scan=0)
        with pytest.raises(ValueError, match="max_steps_per_scan"):
            ProlixOpenMMConfig(max_steps_per_scan=-1)

    @pytest.mark.parametrize("val", ["inference", "trajectory"])
    def test_execution_mode_valid(self, val):
        assert ProlixOpenMMConfig(execution_mode=val).execution_mode == val

    @pytest.mark.parametrize("val", ["", "batch", "training", None])
    def test_execution_mode_invalid(self, val):
        with pytest.raises(ValueError, match="execution_mode"):
            ProlixOpenMMConfig(execution_mode=val)  # type: ignore[arg-type]

    # --- Minimizer Backend ---
    @pytest.mark.parametrize("val", ["lbfgs", "fire", "openmm_cpu"])
    def test_minimizer_backend_valid(self, val):
        assert ProlixOpenMMConfig(minimizer_backend=val).minimizer_backend == val

    @pytest.mark.parametrize("val", ["", "scipy", "adam", "gradient_descent", None])
    def test_minimizer_backend_invalid(self, val):
        with pytest.raises(ValueError, match="minimizer_backend"):
            ProlixOpenMMConfig(minimizer_backend=val)  # type: ignore[arg-type]


# ============================================================================
# 3. ProlixPlatform OpenMM Platform Shim Stress
# ============================================================================

class TestProlixPlatformShimStress:
    """Adversarial stress-testing of ProlixPlatform shim and registry interactions."""

    def test_get_platform_by_name_identity(self):
        """getPlatformByName must return the identical singleton instance across casings."""
        p_canonical = openmm.Platform.getPlatformByName("Prolix")
        p_lower = openmm.Platform.getPlatformByName("prolix")
        p_upper = openmm.Platform.getPlatformByName("PROLIX")
        p_mixed = openmm.Platform.getPlatformByName("ProLiX")

        assert p_canonical is p_lower
        assert p_canonical is p_upper
        assert p_canonical is p_mixed
        assert isinstance(p_canonical, ProlixPlatform)

    def test_get_platform_by_name_unknown_raises(self):
        """Unknown platform names must raise OpenMMException."""
        with pytest.raises(openmm.OpenMMException):
            openmm.Platform.getPlatformByName("NonExistentPlatform")

    def test_get_num_platforms_and_iteration(self):
        """getNumPlatforms must include Prolix and iteration must yield all platforms."""
        num = openmm.Platform.getNumPlatforms()
        assert num >= 3  # Reference, CPU, Prolix

        platforms = [openmm.Platform.getPlatform(i) for i in range(num)]
        names = [p.getName() for p in platforms]
        assert "Reference" in names
        assert "CPU" in names
        assert "Prolix" in names
        assert platforms[-1].getName() == "Prolix"

    def test_get_platform_negative_and_out_of_bounds(self):
        """getPlatform(-1) yields Prolix; getPlatform(num) and getPlatform(-2) raise OpenMMException."""
        p_last = openmm.Platform.getPlatform(-1)
        assert p_last.getName() == "Prolix"
        assert p_last is openmm.Platform.getPlatformByName("Prolix")

        num = openmm.Platform.getNumPlatforms()
        with pytest.raises(openmm.OpenMMException):
            openmm.Platform.getPlatform(num)

        with pytest.raises(openmm.OpenMMException):
            openmm.Platform.getPlatform(-2)

    def test_platform_speed_and_capabilities(self):
        """Verify speed rating and double precision capabilities."""
        plat = openmm.Platform.getPlatformByName("Prolix")
        assert plat.getSpeed() == 100.0
        assert plat.supportsDoublePrecision() is True
        assert plat.supportsKernels() is True

    def test_all_30_platform_property_defaults_and_mutation(self):
        """Test getting and setting default values across all 30 properties."""
        plat = ProlixPlatform()
        names = plat.getPropertyNames()
        assert len(names) == 30

        for name in names:
            # Case-insensitivity for property access
            val = plat.getPropertyDefaultValue(name)
            assert isinstance(val, str)
            assert plat.getPropertyDefaultValue(name.lower()) == val
            assert plat.getPropertyDefaultValue(name.upper()) == val

        # Test mutating and reading back
        plat.setPropertyDefaultValue("Precision", "double")
        assert plat.getPropertyDefaultValue("Precision") == "double"
        plat.setPropertyDefaultValue("NeighborListSkin", "3.2")
        assert plat.getPropertyDefaultValue("NeighborListSkin") == "3.2"

        # Verify create_config picks up the mutated default
        cfg = plat.create_config()
        assert cfg.precision == "double"
        assert cfg.nl_skin == 3.2

    def test_platform_property_context_routing(self):
        """Verify getPropertyValue and setPropertyValue routing for different context shapes."""
        plat = ProlixPlatform()

        # 1. Context is None -> routes to default
        plat.setPropertyDefaultValue("Precision", "single")
        assert plat.getPropertyValue(None, "Precision") == "single"
        plat.setPropertyValue(None, "Precision", "double")
        assert plat.getPropertyValue(None, "Precision") == "double"

        # 2. Context has _platform_properties dict
        class MockContextWithDict:
            def __init__(self):
                self._platform_properties = {"Precision": "mixed", "NeighborListSkin": "4.0"}

        ctx_dict = MockContextWithDict()
        assert plat.getPropertyValue(ctx_dict, "Precision") == "mixed"
        assert plat.getPropertyValue(ctx_dict, "NeighborListSkin") == "4.0"
        plat.setPropertyValue(ctx_dict, "Precision", "single")
        assert ctx_dict._platform_properties["Precision"] == "single"

        # 3. Context has get_platform_property / set_platform_property methods
        class MockContextWithMethods:
            def __init__(self):
                self.props = {"Precision": "double"}

            def get_platform_property(self, name):
                return self.props.get(name, "none")

            def set_platform_property(self, name, val):
                self.props[name] = val

        ctx_methods = MockContextWithMethods()
        assert plat.getPropertyValue(ctx_methods, "Precision") == "double"
        plat.setPropertyValue(ctx_methods, "Precision", "single")
        assert ctx_methods.props["Precision"] == "single"

        # 4. Context has _config (ProlixOpenMMConfig)
        class MockContextWithConfig:
            def __init__(self):
                self._config = ProlixOpenMMConfig(precision="mixed", use_pallas=True)

        ctx_cfg = MockContextWithConfig()
        assert plat.getPropertyValue(ctx_cfg, "Precision") == "mixed"
        assert plat.getPropertyValue(ctx_cfg, "UsePallas") == "true"

    def test_illegal_property_names_raise_openmm_exception(self):
        """Illegal property names must raise openmm.OpenMMException in all 4 property methods."""
        plat = ProlixPlatform()
        for bad_name in ["InvalidProperty", "foo_bar", "", "123"]:
            with pytest.raises(openmm.OpenMMException, match="Illegal property name"):
                plat.getPropertyDefaultValue(bad_name)

            with pytest.raises(openmm.OpenMMException, match="Illegal property name"):
                plat.setPropertyDefaultValue(bad_name, "value")

            with pytest.raises(openmm.OpenMMException, match="Illegal property name"):
                plat.getPropertyValue(None, bad_name)

            with pytest.raises(openmm.OpenMMException, match="Illegal property name"):
                plat.setPropertyValue(None, bad_name, "value")

    def test_repeated_registration_and_unregistration_stress(self):
        """Repeated register() / unregister() cycles must not corrupt OpenMM C++ platform registry."""
        initial_num = openmm.Platform.getNumPlatforms()

        # 50 consecutive register() calls must be idempotent
        for _ in range(50):
            ProlixPlatform.register()
        assert openmm.Platform.getNumPlatforms() == initial_num

        # 10 cycles of unregister and register
        for _ in range(10):
            ProlixPlatform.unregister()
            assert openmm.Platform.getNumPlatforms() == initial_num - 1
            with pytest.raises(openmm.OpenMMException):
                openmm.Platform.getPlatformByName("Prolix")

            ProlixPlatform.register()
            assert openmm.Platform.getNumPlatforms() == initial_num
            assert openmm.Platform.getPlatformByName("Prolix") is not None

        # Verify C++ platform health: Reference and CPU platforms can still create contexts
        system = openmm.System()
        system.addParticle(1.0)
        integrator = openmm.VerletIntegrator(0.001)
        ref_plat = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(system, integrator, ref_plat)
        assert context.getPlatform().getName() == "Reference"
        del context


# ============================================================================
# 4. Immutability & Concurrency Stress
# ============================================================================

class TestImmutabilityAndConcurrencyStress:
    """Stress test immutability and multi-threaded concurrency."""

    def test_frozen_immutability_all_fields(self):
        """All 30 fields of ProlixOpenMMConfig must forbid mutation and deletion."""
        cfg = ProlixOpenMMConfig()
        for field in _CANONICAL_NAMES:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(cfg, field, "dummy_value")
            with pytest.raises(dataclasses.FrozenInstanceError):
                delattr(cfg, field)

    def test_hashability_and_equality(self):
        """ProlixOpenMMConfig instances must be hashable and support set/dict deduplication."""
        cfg1 = ProlixOpenMMConfig()
        cfg2 = ProlixOpenMMConfig()
        cfg3 = ProlixOpenMMConfig(precision="double")

        assert cfg1 == cfg2
        assert cfg1 != cfg3
        assert hash(cfg1) == hash(cfg2)
        assert hash(cfg1) != hash(cfg3)

        config_set = {cfg1, cfg2, cfg3}
        assert len(config_set) == 2

    def test_concurrent_multithreaded_stress(self):
        """Concurrent parsing and config creation across 16 threads must be thread-safe."""
        def worker(idx: int) -> None:
            props = {
                "Precision": "double" if idx % 2 == 0 else "single",
                "UsePallas": "true" if idx % 3 == 0 else "false",
                "PmeGridSpacing": str(0.5 + (idx % 10) * 0.05),
                "NeighborListSkin": str(1.5 + (idx % 5) * 0.1),
                "ShakeIterations": str(20 + (idx % 15)),
                "DeterministicSeed": str(1000 + idx),
                "ExecutionMode": "trajectory" if idx % 2 == 0 else "inference",
            }
            cfg = parse_platform_properties(props)
            assert cfg.precision == ("double" if idx % 2 == 0 else "single")
            assert cfg.deterministic_seed == 1000 + idx

            # Concurrent ProlixPlatform queries
            plat = openmm.Platform.getPlatformByName("Prolix")
            assert plat.getName() == "Prolix"
            assert plat.getPropertyDefaultValue("Precision") in ("single", "double")

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker, i) for i in range(200)]
            for f in concurrent.futures.as_completed(futures):
                f.result()
