"""Adversarial stress-testing suite for prolix.openmm.units.

Empirically challenges:
1. Floating-point precision across wide magnitude ranges (10^-20 to 10^20) for round-trip drift (< 1e-12 relative).
2. JAX differentiability: jax.grad, jax.hessian, jax.jvp, jax.vjp, jax.vmap, and jax.jit.
3. Container diversity: high-dimensional tensors, empty arrays, single-element arrays,
   non-contiguous NumPy arrays, and deeply nested Python lists/tuples.
4. OpenMM Quantities: exotic and compound units across all physical dimensions and OpenMM Vec3 structures.
5. Invalidation conditions: malformed inputs, incompatible unit dimensions, bad types, and predictable exceptions.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as u
import pytest

from prolix.openmm import units
from prolix.openmm.units import (
    AKMA_PRESSURE_PER_BAR,
    ANGSTROM_TO_NM,
    BAR_PER_AKMA_PRESSURE,
    FORCE_AKMA_TO_OMM,
    FORCE_OMM_TO_AKMA,
    KCAL_PER_MOL_TO_KJ_PER_MOL,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
    PS_TO_TAU_AKMA,
    TAU_AKMA_FS,
    TAU_AKMA_PS,
    VELOCITY_AKMA_TO_OMM,
    VELOCITY_OMM_TO_AKMA,
    from_akma,
    to_akma,
)

jax.config.update("jax_enable_x64", True)

ALL_UNIT_TYPES = list(units._UNIT_SPECS.keys())


# ============================================================================
# 1. Floating-Point Precision & Extreme Range Stress Tests
# ============================================================================


class TestFloatingPointPrecisionAdversarial:
    """Stress-test numerical precision and round-trip drift across extreme ranges."""

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_wide_magnitude_round_trip_omm_to_akma(self, unit_type: str):
        """Verify round-trip drift is strictly < 1e-12 relative across 10^-20 to 10^20."""
        rng = np.random.default_rng(20261109)
        exps = rng.uniform(-20.0, 20.0, size=2000)
        signs = rng.choice([-1.0, 1.0], size=2000)
        vals = signs * (10.0**exps)

        akma = to_akma(vals, unit_type)
        back = from_akma(akma, unit_type, as_quantity=False)

        rel_err = np.abs((back - vals) / vals)
        max_rel_err = np.max(rel_err)
        assert max_rel_err < 1e-12, f"Round-trip failed for {unit_type}: max rel err = {max_rel_err}"
        # Also confirm precision matches machine epsilon ~ 2.22e-16
        assert max_rel_err < 1e-14

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_wide_magnitude_round_trip_akma_to_omm(self, unit_type: str):
        """Verify round-trip drift is strictly < 1e-12 relative for AKMA -> OMM -> AKMA."""
        rng = np.random.default_rng(99991)
        exps = rng.uniform(-20.0, 20.0, size=2000)
        signs = rng.choice([-1.0, 1.0], size=2000)
        vals = signs * (10.0**exps)

        omm = from_akma(vals, unit_type, as_quantity=False)
        back = to_akma(omm, unit_type)

        rel_err = np.abs((back - vals) / vals)
        max_rel_err = np.max(rel_err)
        assert max_rel_err < 1e-12, f"AKMA round-trip failed for {unit_type}: max rel err = {max_rel_err}"
        assert max_rel_err < 1e-14

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_special_ieee754_values(self, unit_type: str):
        """Test behavior on inf, -inf, nan, and signed zero."""
        # inf
        assert math.isinf(to_akma(float("inf"), unit_type))
        assert math.isinf(from_akma(float("inf"), unit_type, as_quantity=False))
        # -inf
        assert math.isinf(to_akma(float("-inf"), unit_type))
        assert math.isinf(from_akma(float("-inf"), unit_type, as_quantity=False))
        # nan
        assert math.isnan(to_akma(float("nan"), unit_type))
        assert math.isnan(from_akma(float("nan"), unit_type, as_quantity=False))
        # -0.0
        neg_zero = to_akma(-0.0, unit_type)
        assert neg_zero == 0.0
        assert math.copysign(1.0, neg_zero) == -1.0

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_ultra_extreme_scales(self, unit_type: str):
        """Test scale factors near floating point limits (1e-150 to 1e150)."""
        scales = [1e-150, 1e-100, 1e-50, 1e50, 1e100, 1e150]
        for s in scales:
            akma = to_akma(s, unit_type)
            assert math.isfinite(akma)
            back = from_akma(akma, unit_type, as_quantity=False)
            assert math.isfinite(back)
            assert np.abs((back - s) / s) < 1e-12


# ============================================================================
# 2. JAX Differentiability & Autodiff Suite
# ============================================================================


class TestJAXDifferentiabilityAdversarial:
    """Stress-test JAX transformations: grad, hessian, jvp, vjp, vmap, and jit."""

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_jax_grad_and_hessian_vector(self, unit_type: str):
        """Test reverse-mode 1st and 2nd derivatives across all unit types."""
        to_f = units._UNIT_SPECS[unit_type]["to_akma"]
        from_f = units._UNIT_SPECS[unit_type]["from_akma"]

        def f_to(x):
            ak = to_akma(x, unit_type)
            return jnp.sum(ak**3)

        def f_from(y):
            om = from_akma(y, unit_type, as_quantity=False)
            return jnp.sum(om**3)

        x = jnp.array([-2.5, 0.5, 3.2, 10.0], dtype=jnp.float64)
        g_to = jax.grad(f_to)(x)
        expected_g_to = 3.0 * (to_f**3) * (x**2)
        np.testing.assert_allclose(g_to, expected_g_to, rtol=1e-12)

        # 2nd derivative (Hessian diagonal)
        h_to = jax.hessian(f_to)(x)
        expected_h_to = jnp.diag(6.0 * (to_f**3) * x)
        np.testing.assert_allclose(h_to, expected_h_to, rtol=1e-12)

        # from_akma
        g_from = jax.grad(f_from)(x)
        expected_g_from = 3.0 * (from_f**3) * (x**2)
        np.testing.assert_allclose(g_from, expected_g_from, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_jax_jvp_forward_mode(self, unit_type: str):
        """Test forward-mode autodiff (JVP)."""
        to_f = units._UNIT_SPECS[unit_type]["to_akma"]
        primals = (jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),)
        tangents = (jnp.array([0.1, -0.2, 0.5], dtype=jnp.float64),)

        def f(x):
            return to_akma(x, unit_type)

        out_primals, out_tangents = jax.jvp(f, primals, tangents)
        np.testing.assert_allclose(out_primals, primals[0] * to_f, rtol=1e-12)
        np.testing.assert_allclose(out_tangents, tangents[0] * to_f, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", ALL_UNIT_TYPES)
    def test_jax_vmap_and_jit(self, unit_type: str):
        """Verify vmap and jit work together with no tracing issues."""
        @jax.jit
        def batched_convert(batch):
            return jax.vmap(lambda x: to_akma(x, unit_type))(batch)

        batch = jnp.arange(100, dtype=jnp.float64).reshape((10, 10))
        res = batched_convert(batch)
        expected = batch * units._UNIT_SPECS[unit_type]["to_akma"]
        np.testing.assert_allclose(res, expected, rtol=1e-12)


# ============================================================================
# 3. Container Diversity Stress Tests
# ============================================================================


class TestContainerDiversityAdversarial:
    """Stress-test high-dimensional, empty, non-contiguous, and nested containers."""

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
    def test_high_dimensional_numpy_and_jax(self, ndim: int):
        shape = tuple([2] * ndim)
        # NumPy
        arr_np = np.random.randn(*shape).astype(np.float64)
        ak_np = to_akma(arr_np, "length")
        assert ak_np.shape == shape
        back_np = from_akma(ak_np, "length", as_quantity=False)
        assert back_np.shape == shape
        np.testing.assert_allclose(back_np, arr_np, rtol=1e-12)

        # JAX
        arr_jax = jnp.array(arr_np)
        ak_jax = to_akma(arr_jax, "length")
        assert ak_jax.shape == shape
        back_jax = from_akma(ak_jax, "length", as_quantity=False)
        assert back_jax.shape == shape
        np.testing.assert_allclose(back_jax, arr_jax, rtol=1e-12)

    def test_empty_arrays_various_shapes(self):
        for shape in [(0,), (0, 3), (4, 0), (2, 0, 3), (0, 0, 0)]:
            arr = np.zeros(shape, dtype=np.float64)
            ak = to_akma(arr, "energy")
            assert ak.shape == shape and ak.size == 0
            back = from_akma(ak, "energy", as_quantity=False)
            assert back.shape == shape and back.size == 0

    def test_single_element_containers(self):
        # 1D single element
        s1 = np.array([123.456], dtype=np.float64)
        ak1 = to_akma(s1, "velocity")
        assert ak1.shape == (1,)
        back1 = from_akma(ak1, "velocity", as_quantity=False)
        np.testing.assert_allclose(back1, s1, rtol=1e-12)

        # 4D single element
        s4 = np.array([[[[789.0]]]], dtype=np.float64)
        ak4 = to_akma(s4, "time")
        assert ak4.shape == (1, 1, 1, 1)
        back4 = from_akma(ak4, "time", as_quantity=False)
        np.testing.assert_allclose(back4, s4, rtol=1e-12)

    def test_non_contiguous_numpy_arrays(self):
        base = np.arange(240, dtype=np.float64).reshape((12, 20))
        # Fortran contiguous
        f_arr = np.asfortranarray(base)
        assert not f_arr.flags.c_contiguous
        ak_f = to_akma(f_arr, "pressure")
        back_f = from_akma(ak_f, "pressure", as_quantity=False)
        np.testing.assert_allclose(back_f, f_arr, rtol=1e-12)

        # Strided slice
        sliced = base[::3, ::4]
        assert not sliced.flags.contiguous
        ak_s = to_akma(sliced, "force")
        back_s = from_akma(ak_s, "force", as_quantity=False)
        np.testing.assert_allclose(back_s, sliced, rtol=1e-12)

        # Negative strides
        rev = base[::-1, ::-1]
        assert not rev.flags.c_contiguous
        ak_r = to_akma(rev, "bond_k")
        back_r = from_akma(ak_r, "bond_k", as_quantity=False)
        np.testing.assert_allclose(back_r, rev, rtol=1e-12)

    def test_deeply_nested_python_structures(self):
        nested = [
            1.0,
            (2.0, [3.0, (4.0, 5.0)]),
            [[6.0, 7.0], (8.0,)],
            (),
            [],
        ]
        ak = to_akma(nested, "length")
        assert ak[0] == 10.0
        assert ak[1][0] == 20.0
        assert ak[1][1][0] == 30.0
        assert ak[1][1][1] == (40.0, 50.0)
        assert ak[2] == [[60.0, 70.0], (80.0,)]
        assert ak[3] == ()
        assert ak[4] == []

        back = from_akma(ak, "length", as_quantity=False)
        assert back == nested

    def test_input_non_mutability(self):
        """Ensure to_akma and from_akma never mutate their inputs in-place."""
        orig_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        orig_copy = orig_np.copy()
        _ = to_akma(orig_np, "length")
        np.testing.assert_array_equal(orig_np, orig_copy)

        orig_list = [10.0, 20.0, [30.0, 40.0]]
        orig_list_copy = [10.0, 20.0, [30.0, 40.0]]
        _ = to_akma(orig_list, "time")
        assert orig_list == orig_list_copy


# ============================================================================
# 4. OpenMM Exotic Quantities & Vec3 Stress Tests
# ============================================================================


class TestOpenMMExoticQuantitiesAdversarial:
    """Stress-test compound and exotic units across all physical domains."""

    def test_exotic_length_units(self):
        units_and_factors = [
            (1.0 * u.meter, 1e10),
            (1.0 * u.centimeter, 1e8),
            (1.0 * u.millimeter, 1e7),
            (1.0 * u.micrometer, 1e4),
            (1.0 * u.nanometer, 10.0),
            (1.0 * u.angstrom, 1.0),
            (1.0 * u.bohr, 0.529177210903),
            (1.0 * u.inch, 2.54e8),
        ]
        for q, expected in units_and_factors:
            val = to_akma(q, "length")
            np.testing.assert_allclose(val, expected, rtol=1e-6)

    def test_exotic_time_units(self):
        units_and_expected = [
            (1.0 * u.second, 1e15 / 48.88821291839),
            (1.0 * u.millisecond, 1e12 / 48.88821291839),
            (1.0 * u.microsecond, 1e9 / 48.88821291839),
            (1.0 * u.nanosecond, 1e6 / 48.88821291839),
            (1.0 * u.picosecond, 1e3 / 48.88821291839),
            (1.0 * u.femtosecond, 1.0 / 48.88821291839),
        ]
        for q, expected in units_and_expected:
            val = to_akma(q, "time")
            np.testing.assert_allclose(val, expected, rtol=1e-12)

    def test_exotic_energy_units(self):
        units_and_expected = [
            (1.0 * u.kilojoule / u.mole, 1.0 / 4.184),
            (1.0 * u.kilocalorie / u.mole, 1.0),
            (4184.0 * u.joule / u.mole, 1.0),
            (1000.0 * u.calorie / u.mole, 1.0),
        ]
        for q, expected in units_and_expected:
            val = to_akma(q, "energy")
            np.testing.assert_allclose(val, expected, rtol=1e-12)

    def test_exotic_pressure_units(self):
        units_and_expected = [
            (1.0 * u.bar, 1.0 / 69477.0),
            (1.0 * u.atmosphere, 1.01325 / 69477.0),
            (1e5 * u.pascal, 1.0 / 69477.0),
            (100.0 * u.kilopascal, 1.0 / 69477.0),
            (0.1 * u.megapascal, 1.0 / 69477.0),
            (760.0 * u.torr, 1.01325 / 69477.0),
        ]
        for q, expected in units_and_expected:
            val = to_akma(q, "pressure")
            np.testing.assert_allclose(val, expected, rtol=1e-5)

    def test_exotic_velocity_and_force_units(self):
        # 1000 m/s = 1 nm/ps -> 0.4888821291839 Angstrom/tau_AKMA
        q_v = 1000.0 * (u.meter / u.second)
        np.testing.assert_allclose(to_akma(q_v, "velocity"), VELOCITY_OMM_TO_AKMA, rtol=1e-10)

        # Force in kcal/(mol*A)
        q_f = 1.0 * (u.kilocalorie / (u.mole * u.angstrom))
        np.testing.assert_allclose(to_akma(q_f, "force"), 1.0, rtol=1e-10)

    def test_openmm_vec3_and_vec3_lists(self):
        """Verify handling of OpenMM's native Vec3 objects."""
        v = openmm.Vec3(1.0, 2.0, 3.0)
        ak = to_akma(v, "length")
        assert ak == (10.0, 20.0, 30.0)

        # Quantity wrapping list of Vec3 (standard OpenMM state positions)
        positions = [
            openmm.Vec3(1.0, 2.0, 3.0),
            openmm.Vec3(4.0, 5.0, 6.0),
            openmm.Vec3(7.0, 8.0, 9.0),
        ] * u.nanometers
        ak_pos = to_akma(positions, "length")
        assert ak_pos == [
            (10.0, 20.0, 30.0),
            (40.0, 50.0, 60.0),
            (70.0, 80.0, 90.0),
        ]
        back_pos = from_akma(ak_pos, "length", as_quantity=True)
        assert isinstance(back_pos, u.Quantity)
        assert back_pos.value_in_unit(u.nanometer) == [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
        ]


# ============================================================================
# 5. Invalidation Conditions & Error Handling Stress Tests
# ============================================================================


class TestInvalidationConditionsAdversarial:
    """Stress-test error boundaries, unmaskable exceptions, and malformed inputs."""

    def test_incompatible_unit_conversion_fails_fast(self):
        incompatibilities = [
            (1.0 * u.kilogram, "length"),
            (1.0 * u.second, "energy"),
            (1.0 * u.nanometer, "time"),
            (1.0 * u.kelvin, "pressure"),
            (1.0 * u.bar, "velocity"),
            (1.0 * u.radian, "bond_k"),
            (1.0 * u.mole, "force"),
            (1.0 * u.second, "friction"),
        ]
        for q, ut in incompatibilities:
            with pytest.raises(TypeError, match="Unit mismatch"):
                to_akma(q, ut)

    def test_from_akma_incompatible_quantity_fails_fast(self):
        with pytest.raises(TypeError, match="incompatible with expected unit"):
            from_akma(10.0 * u.second, "length")

    def test_malformed_and_unsupported_input_types(self):
        bad_values = [
            None,
            "10.0",
            b"20.0",
            {"key": 1.0},
            {1.0, 2.0},
            object(),
        ]
        for bad in bad_values:
            with pytest.raises(TypeError):
                to_akma(bad, "length")
            with pytest.raises(TypeError):
                from_akma(bad, "length")

    def test_invalid_unit_type_strings(self):
        bad_types = [
            "",
            "   ",
            "parsec",
            "lightyear",
            "celsius",
            "random_unknown_type",
        ]
        for bt in bad_types:
            with pytest.raises(ValueError, match="Unsupported unit_type"):
                to_akma(1.0, bt)
            with pytest.raises(ValueError, match="Unsupported unit_type"):
                from_akma(1.0, bt)

    def test_non_string_unit_type_raises_type_error(self):
        for invalid_type in [None, 123, 45.6, [], {}]:
            with pytest.raises(TypeError, match="unit_type must be a string"):
                to_akma(1.0, invalid_type)
            with pytest.raises(TypeError, match="unit_type must be a string"):
                from_akma(1.0, invalid_type)
