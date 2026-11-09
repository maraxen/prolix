"""Tests for Prolix OpenMM unit adapter (src/prolix/openmm/units.py).

Verifies:
1. Exact conversion constants against analytical definitions and prolix.physics.units.
2. Bidirectional round-trips within < 1e-12 relative tolerance across all unit types.
3. Type diversity: scalars, lists, 1D/2D NumPy arrays, JAX arrays, openmm.unit.Quantity.
4. OpenMM unit normalization from alternative units (angstrom, femtosecond, atm).
5. Error handling: unit mismatch, unknown unit types, non-numeric types.
6. Boundary conditions: zero, negative values, extreme scales.
7. JAX autodiff compatibility through conversion functions.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
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
import prolix.physics.units as phys_u

# Enable float64 for JAX in this test module
jax.config.update("jax_enable_x64", True)


# All supported unit types and their canonical OpenMM units
UNIT_TYPE_MAPPING = {
    "length": u.nanometer,
    "time": u.picosecond,
    "energy": u.kilojoules_per_mole,
    "velocity": u.nanometer / u.picosecond,
    "force": u.kilojoules_per_mole / u.nanometer,
    "pressure": u.bar,
    "bond_k": u.kilojoules_per_mole / (u.nanometer**2),
    "angle_k": u.kilojoules_per_mole / (u.radian**2),
    "torsion_k": u.kilojoules_per_mole,
    "improper_k": u.kilojoules_per_mole / (u.radian**2),
    "friction": u.picosecond**-1,
}


class TestUnitConstants:
    """Verify exact values and reciprocal identities of unit conversion constants."""

    def test_length_constants(self):
        assert NM_TO_ANGSTROM == 10.0
        assert ANGSTROM_TO_NM == 0.1
        assert NM_TO_ANGSTROM * ANGSTROM_TO_NM == pytest.approx(1.0, rel=1e-15)

    def test_energy_constants(self):
        assert KCAL_PER_MOL_TO_KJ_PER_MOL == 4.184
        assert KJ_PER_MOL_TO_KCAL_PER_MOL == pytest.approx(1.0 / 4.184, rel=1e-15)
        assert KJ_PER_MOL_TO_KCAL_PER_MOL * KCAL_PER_MOL_TO_KJ_PER_MOL == pytest.approx(
            1.0, rel=1e-15
        )

    def test_time_constants(self):
        assert TAU_AKMA_FS == 48.88821291839
        assert TAU_AKMA_PS == pytest.approx(0.04888821291839, rel=1e-15)
        assert PS_TO_TAU_AKMA == pytest.approx(1000.0 / 48.88821291839, rel=1e-15)
        assert TAU_AKMA_PS * PS_TO_TAU_AKMA == pytest.approx(1.0, rel=1e-15)

    def test_velocity_constants(self):
        assert VELOCITY_OMM_TO_AKMA == pytest.approx(0.4888821291839, rel=1e-15)
        assert VELOCITY_AKMA_TO_OMM == pytest.approx(1.0 / 0.4888821291839, rel=1e-15)
        assert VELOCITY_OMM_TO_AKMA * VELOCITY_AKMA_TO_OMM == pytest.approx(1.0, rel=1e-15)

    def test_force_constants(self):
        assert FORCE_AKMA_TO_OMM == 41.84
        assert FORCE_OMM_TO_AKMA == pytest.approx(1.0 / 41.84, rel=1e-15)
        assert FORCE_OMM_TO_AKMA * FORCE_AKMA_TO_OMM == pytest.approx(1.0, rel=1e-15)

    def test_pressure_constants(self):
        assert BAR_PER_AKMA_PRESSURE == 69477.0
        assert AKMA_PRESSURE_PER_BAR == pytest.approx(1.0 / 69477.0, rel=1e-15)
        assert BAR_PER_AKMA_PRESSURE * AKMA_PRESSURE_PER_BAR == pytest.approx(1.0, rel=1e-15)

    def test_cross_module_consistency(self):
        """Ensure constants match prolix.physics.units exactly."""
        assert BAR_PER_AKMA_PRESSURE == phys_u.BAR_PER_AKMA_PRESSURE
        assert AKMA_PRESSURE_PER_BAR == phys_u.AKMA_PRESSURE_PER_BAR
        assert TAU_AKMA_FS == phys_u.AKMA_TIME_UNIT_FS


class TestBidirectionalRoundTrip:
    """Verify bidirectional round trips satisfy < 1e-12 relative tolerance."""

    @pytest.mark.parametrize("unit_type", list(UNIT_TYPE_MAPPING.keys()))
    @pytest.mark.parametrize("val", [0.001, 0.123, 1.0, 4.56, 100.0, 12345.67])
    def test_scalar_omm_to_akma_and_back(self, unit_type, val):
        akma_val = to_akma(val, unit_type)
        back_val = from_akma(akma_val, unit_type, as_quantity=False)
        assert back_val == pytest.approx(val, rel=1e-12)

    @pytest.mark.parametrize("unit_type", list(UNIT_TYPE_MAPPING.keys()))
    @pytest.mark.parametrize("val", [0.001, 0.123, 1.0, 4.56, 100.0, 12345.67])
    def test_scalar_akma_to_omm_and_back(self, unit_type, val):
        omm_val = from_akma(val, unit_type, as_quantity=False)
        back_akma = to_akma(omm_val, unit_type)
        assert back_akma == pytest.approx(val, rel=1e-12)

    @pytest.mark.parametrize("unit_type,omm_unit", list(UNIT_TYPE_MAPPING.items()))
    def test_quantity_round_trip(self, unit_type, omm_unit):
        val = 2.5
        q = val * omm_unit
        akma_val = to_akma(q, unit_type)
        q_back = from_akma(akma_val, unit_type, as_quantity=True)

        assert isinstance(q_back, u.Quantity)
        assert q_back.unit.is_compatible(omm_unit)
        assert q_back.value_in_unit(omm_unit) == pytest.approx(val, rel=1e-12)


class TestTypeDiversity:
    """Verify to_akma and from_akma support all required data containers."""

    @pytest.mark.parametrize("unit_type", ["length", "velocity", "force"])
    def test_python_list(self, unit_type):
        raw_list = [1.0, 2.5, 5.0, 10.0]
        akma = to_akma(raw_list, unit_type)
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_allclose(back, raw_list, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", ["length", "velocity", "force"])
    def test_nested_python_list(self, unit_type):
        raw_nested = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        akma = to_akma(raw_nested, unit_type)
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_allclose(back, raw_nested, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", ["length", "velocity", "force"])
    def test_numpy_1d_array(self, unit_type):
        arr = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
        akma = to_akma(arr, unit_type)
        assert isinstance(akma, np.ndarray)
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_allclose(back, arr, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", ["length", "velocity", "force"])
    def test_numpy_2d_array(self, unit_type):
        arr = np.arange(30, dtype=np.float64).reshape((10, 3)) + 1.0
        akma = to_akma(arr, unit_type)
        assert akma.shape == (10, 3)
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_allclose(back, arr, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", ["length", "velocity", "force"])
    def test_jax_array(self, unit_type):
        j_arr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64)
        akma = to_akma(j_arr, unit_type)
        assert isinstance(akma, (jax.Array, np.ndarray))
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_allclose(back, j_arr, rtol=1e-12)

    def test_quantity_wrapping_numpy_array(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        q = arr * u.nanometers
        akma = to_akma(q, "length")
        np.testing.assert_allclose(akma, arr * 10.0, rtol=1e-12)
        q_back = from_akma(akma, "length", as_quantity=True)
        np.testing.assert_allclose(q_back.value_in_unit(u.nanometers), arr, rtol=1e-12)


class TestEdgeCasesAndBoundaries:
    """Verify zero, negative values, and extreme numerical scales."""

    @pytest.mark.parametrize("unit_type", list(UNIT_TYPE_MAPPING.keys()))
    def test_zero_scalar(self, unit_type):
        akma = to_akma(0.0, unit_type)
        assert akma == 0.0
        back = from_akma(0.0, unit_type, as_quantity=False)
        assert back == 0.0

    @pytest.mark.parametrize("unit_type", list(UNIT_TYPE_MAPPING.keys()))
    def test_zero_array(self, unit_type):
        zeros = np.zeros((4, 3), dtype=np.float64)
        akma = to_akma(zeros, unit_type)
        np.testing.assert_array_equal(akma, 0.0)
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_array_equal(back, zeros)

    @pytest.mark.parametrize("unit_type", ["length", "velocity", "force", "energy"])
    def test_negative_values(self, unit_type):
        vals = np.array([-10.0, -1.5, 0.0, 2.5, -0.001], dtype=np.float64)
        akma = to_akma(vals, unit_type)
        back = from_akma(akma, unit_type, as_quantity=False)
        np.testing.assert_allclose(back, vals, rtol=1e-12)

    @pytest.mark.parametrize("unit_type", list(UNIT_TYPE_MAPPING.keys()))
    @pytest.mark.parametrize("scale", [1e-15, 1e-12, 1e-6, 1e6, 1e12])
    def test_extreme_scales(self, unit_type, scale):
        akma = to_akma(scale, unit_type)
        assert math.isfinite(akma) and akma != 0.0
        back = from_akma(akma, unit_type, as_quantity=False)
        assert back == pytest.approx(scale, rel=1e-12)


class TestAlternativeOpenMMUnits:
    """Verify OpenMM Quantities in non-canonical units convert correctly."""

    def test_length_alternative_units(self):
        # 1.0 Angstrom = 0.1 nm = 1.0 AKMA Angstrom
        q_ang = 1.0 * u.angstrom
        assert to_akma(q_ang, "length") == pytest.approx(1.0, rel=1e-12)

        # 1.0 meter = 1e10 Angstroms
        q_m = 1.0 * u.meter
        assert to_akma(q_m, "length") == pytest.approx(1e10, rel=1e-12)

    def test_time_alternative_units(self):
        # 1000 fs = 1 ps
        q_fs = 1000.0 * u.femtosecond
        q_ps = 1.0 * u.picosecond
        assert to_akma(q_fs, "time") == pytest.approx(to_akma(q_ps, "time"), rel=1e-12)

        # 1.0 ns = 1000 ps
        q_ns = 1.0 * u.nanosecond
        assert to_akma(q_ns, "time") == pytest.approx(
            to_akma(1000.0 * u.picosecond, "time"), rel=1e-12
        )

    def test_energy_alternative_units(self):
        # 1.0 kcal/mol in OpenMM Quantity
        q_kcal = 1.0 * u.kilocalories_per_mole
        assert to_akma(q_kcal, "energy") == pytest.approx(1.0, rel=1e-12)

        # 4184 J/mol = 1 kcal/mol
        q_j = 4184.0 * (u.joule / u.mole)
        assert to_akma(q_j, "energy") == pytest.approx(1.0, rel=1e-12)

    def test_pressure_alternative_units(self):
        # 1.0 atm = 1.01325 bar
        q_atm = 1.0 * u.atmosphere
        q_bar = 1.01325 * u.bar
        assert to_akma(q_atm, "pressure") == pytest.approx(to_akma(q_bar, "pressure"), rel=1e-10)


class TestUnitMismatchAndValidation:
    """Verify informative errors on unit mismatches, unsupported types, and bad types."""

    def test_incompatible_quantity_unit(self):
        # Passing time quantity to length converter
        with pytest.raises((TypeError, ValueError)):
            to_akma(10.0 * u.picoseconds, "length")

        # Passing energy quantity to time converter
        with pytest.raises((TypeError, ValueError)):
            to_akma(100.0 * u.kilojoules_per_mole, "time")

        # Passing temperature quantity to energy converter
        with pytest.raises((TypeError, ValueError)):
            to_akma(300.0 * u.kelvin, "energy")

    def test_unknown_unit_type_string(self):
        with pytest.raises(ValueError, match="Unsupported unit_type"):
            to_akma(1.0, "astronomical_unit")

        with pytest.raises(ValueError, match="Unsupported unit_type"):
            from_akma(1.0, "parsec")

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            to_akma("10.0 nm", "length")

        with pytest.raises(TypeError):
            to_akma(None, "energy")

        with pytest.raises(TypeError):
            from_akma("10.0", "length")

        with pytest.raises(TypeError):
            from_akma(None, "energy")


class TestJAXAutodiffCompatibility:
    """Verify that to_akma and from_akma differentiate cleanly in JAX."""

    def test_grad_through_to_akma(self):
        # f(x) = sum(to_akma(x, "length")**2) = sum((10*x)**2) = 100 * sum(x**2)
        # df/dx = 200 * x
        def f(x):
            akma = to_akma(x, "length")
            return jnp.sum(akma**2)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        grad_fn = jax.grad(f)
        grad = grad_fn(x)
        expected_grad = 200.0 * x
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-12)

    def test_grad_through_from_akma(self):
        # g(y) = sum(from_akma(y, "energy", as_quantity=False)**2) = sum((4.184 * y)**2)
        # dg/dy = 2 * (4.184**2) * y
        def g(y):
            omm = from_akma(y, "energy", as_quantity=False)
            return jnp.sum(omm**2)

        y = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        grad_fn = jax.grad(g)
        grad = grad_fn(y)
        expected_grad = 2.0 * (4.184**2) * y
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-12)
