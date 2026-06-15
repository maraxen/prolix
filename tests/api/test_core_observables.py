"""Tests for core observables: KineticEnergy, RMSD, Pressure."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import pytest

from prolix.api.observables import Observable, KineticEnergy, RMSD, Pressure
from prolix.typing import IntegratorState as LangevinState
from prolix.simulate import BOLTZMANN_KCAL


class TestKineticEnergy:
    """Tests for KineticEnergy observable."""

    def test_kinetic_energy_is_observable(self):
        """KineticEnergy should implement the Observable protocol."""
        ke = KineticEnergy()
        assert isinstance(ke, Observable), "KineticEnergy should implement Observable protocol"

    def test_kinetic_energy_has_compute(self):
        """KineticEnergy should have a compute method."""
        ke = KineticEnergy()
        assert hasattr(ke, "compute")
        assert callable(ke.compute)

    def test_kinetic_energy_returns_scalar(self):
        """KineticEnergy.compute() should return a scalar Array."""
        n_atoms = 3
        positions = jnp.zeros((n_atoms, 3))
        momentum = jnp.ones((n_atoms, 3))
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        ke = KineticEnergy()
        result = ke.compute(state)

        assert result.shape == (), f"Expected scalar, got shape {result.shape}"
        assert isinstance(result, jnp.ndarray), "Result should be a JAX array"

    def test_kinetic_energy_correct_calculation(self):
        """KineticEnergy should compute KE = sum(p^2 / (2*m)) correctly."""
        n_atoms = 2
        positions = jnp.zeros((n_atoms, 3))
        # p = [1, 1, 1] for atom 1, [2, 2, 2] for atom 2
        momentum = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        force = jnp.zeros((n_atoms, 3))
        # m = 1 for atom 1, m = 2 for atom 2
        mass = jnp.array([1.0, 2.0])
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        ke = KineticEnergy()
        result = ke.compute(state)

        # KE_1 = sum(1^2 / (2*1)) = 3 / 2 = 1.5
        # KE_2 = sum(2^2 / (2*2)) = 12 / 4 = 3.0
        # Total KE = 1.5 + 3.0 = 4.5
        expected = 4.5
        assert jnp.allclose(result, expected, rtol=1e-6), \
            f"Expected KE={expected}, got {result}"

    def test_kinetic_energy_positive(self):
        """KineticEnergy should always be non-negative."""
        n_atoms = 1
        positions = jnp.zeros((n_atoms, 3))
        momentum = jnp.array([[3.0, 4.0, 5.0]])  # Non-zero momentum
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.array([12.0])
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        ke = KineticEnergy()
        result = ke.compute(state)

        assert result >= 0.0, "KineticEnergy should be non-negative"
        assert not jnp.isnan(result), "KineticEnergy should not be NaN"

    def test_kinetic_energy_zero_momentum(self):
        """KineticEnergy should be zero for zero momentum."""
        n_atoms = 3
        positions = jnp.zeros((n_atoms, 3))
        momentum = jnp.zeros((n_atoms, 3))
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        ke = KineticEnergy()
        result = ke.compute(state)

        assert jnp.allclose(result, 0.0, rtol=1e-10), \
            f"Expected KE=0 for zero momentum, got {result}"


class TestRMSD:
    """Tests for RMSD observable."""

    def test_rmsd_is_observable(self):
        """RMSD should implement the Observable protocol."""
        ref = jnp.zeros((3, 3))
        rmsd = RMSD(reference=ref)
        assert isinstance(rmsd, Observable), "RMSD should implement Observable protocol"

    def test_rmsd_has_compute(self):
        """RMSD should have a compute method."""
        ref = jnp.zeros((3, 3))
        rmsd = RMSD(reference=ref)
        assert hasattr(rmsd, "compute")
        assert callable(rmsd.compute)

    def test_rmsd_returns_scalar(self):
        """RMSD.compute() should return a scalar Array."""
        ref = jnp.zeros((3, 3))
        rmsd = RMSD(reference=ref)

        positions = jnp.ones((3, 3))
        force = jnp.zeros((3, 3))
        momentum = jnp.zeros((3, 3))
        mass = jnp.ones(3)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        result = rmsd.compute(state)

        assert result.shape == (), f"Expected scalar, got shape {result.shape}"
        assert isinstance(result, jnp.ndarray), "Result should be a JAX array"

    def test_rmsd_at_reference_is_zero(self):
        """RMSD should be zero when positions equal reference."""
        ref = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        rmsd = RMSD(reference=ref)

        # State with positions equal to reference
        positions = ref.copy()
        force = jnp.zeros((3, 3))
        momentum = jnp.zeros((3, 3))
        mass = jnp.ones(3)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        result = rmsd.compute(state)

        assert jnp.allclose(result, 0.0, atol=1e-10), \
            f"Expected RMSD=0 at reference, got {result}"

    def test_rmsd_simple_displacement(self):
        """RMSD should compute correctly for known displacement."""
        # Reference: origin
        ref = jnp.zeros((2, 3))
        rmsd = RMSD(reference=ref)

        # All atoms displaced by 1 Angstrom in x, y, z
        # Each atom moves to (1, 1, 1)
        positions = jnp.ones((2, 3))
        force = jnp.zeros((2, 3))
        momentum = jnp.zeros((2, 3))
        mass = jnp.ones(2)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        result = rmsd.compute(state)

        # Each atom: diff^2 = 1^2 + 1^2 + 1^2 = 3
        # Mean over 2 atoms: (3 + 3) / 2 = 3
        # sqrt(3) ≈ 1.732
        expected = jnp.sqrt(3.0)
        assert jnp.allclose(result, expected, rtol=1e-6), \
            f"Expected RMSD={expected}, got {result}"

    def test_rmsd_is_equinox_module(self):
        """RMSD should be an eqx.Module."""
        ref = jnp.zeros((3, 3))
        rmsd = RMSD(reference=ref)
        assert isinstance(rmsd, eqx.Module), "RMSD should be an eqx.Module"


class TestPressure:
    """Tests for Pressure observable."""

    def test_pressure_is_observable(self):
        """Pressure should implement the Observable protocol."""
        press = Pressure(n_atoms=3, volume_angstrom3=1000.0)
        assert isinstance(press, Observable), "Pressure should implement Observable protocol"

    def test_pressure_has_compute(self):
        """Pressure should have a compute method."""
        press = Pressure(n_atoms=3, volume_angstrom3=1000.0)
        assert hasattr(press, "compute")
        assert callable(press.compute)

    def test_pressure_returns_scalar(self):
        """Pressure.compute() should return a scalar Array."""
        press = Pressure(n_atoms=3, volume_angstrom3=1000.0)

        positions = jnp.zeros((3, 3))
        momentum = jnp.ones((3, 3))
        force = jnp.zeros((3, 3))
        mass = jnp.ones(3)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        result = press.compute(state)

        assert result.shape == (), f"Expected scalar, got shape {result.shape}"
        assert isinstance(result, jnp.ndarray), "Result should be a JAX array"

    def test_pressure_positive_for_warm_state(self):
        """Pressure should be positive for a warm state."""
        n_atoms = 5
        press = Pressure(n_atoms=n_atoms, volume_angstrom3=500.0)

        positions = jnp.zeros((n_atoms, 3))
        # Non-zero momentum gives positive temperature, so positive pressure
        momentum = jnp.ones((n_atoms, 3))
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        result = press.compute(state)

        assert result > 0.0, "Pressure should be positive for warm state"
        assert not jnp.isnan(result), "Pressure should not be NaN"
        assert not jnp.isinf(result), "Pressure should be finite"

    def test_pressure_is_equinox_module(self):
        """Pressure should be an eqx.Module."""
        press = Pressure(n_atoms=3, volume_angstrom3=1000.0)
        assert isinstance(press, eqx.Module), "Pressure should be an eqx.Module"

    def test_pressure_formula_ideal_gas(self):
        """Pressure.compute() should follow ideal gas law (P = N*k_B*T/V)."""
        # Using the ideal-gas formula: P = 2*KE / (3*V)
        # where KE = sum(p^2 / (2*m))

        n_atoms = 10
        volume = 1000.0  # Angstrom^3
        press = Pressure(n_atoms=n_atoms, volume_angstrom3=volume)

        positions = jnp.zeros((n_atoms, 3))
        # All atoms have unit mass and unit momentum
        momentum = jnp.ones((n_atoms, 3))
        force = jnp.zeros((n_atoms, 3))
        mass = jnp.ones(n_atoms)
        key = jnp.zeros(2, dtype=jnp.uint32)
        box = jnp.zeros((3, 3))

        state = LangevinState(
            positions=positions,
            momentum=momentum,
            force=force,
            mass=mass,
            rng=key,
            box=box,
        )

        result = press.compute(state)

        # KE = sum(1^2 / (2*1)) = 30 * (1 / 2) = 15 kcal/mol
        # P_kcal = (2 * 15) / (3 * 1000) = 30 / 3000 = 0.01 kcal/mol/A^3
        # P_bar = 0.01 * 68568 = 685.68 bar
        KCAL_MOL_PER_A3_TO_BAR = 68568.0
        total_ke = 15.0  # sum(p^2 / (2*m)) for 10 atoms with p=1, m=1
        expected_press_kcal = (2.0 * total_ke) / (3.0 * volume)
        expected_press_bar = expected_press_kcal * KCAL_MOL_PER_A3_TO_BAR

        assert jnp.allclose(result, expected_press_bar, rtol=1e-5), \
            f"Expected pressure={expected_press_bar} bar, got {result} bar"


class TestAllObservablesProtocol:
    """Integration tests for all observables."""

    def test_all_implement_protocol(self):
        """KineticEnergy, RMSD, and Pressure should all implement Observable."""
        ke = KineticEnergy()
        rmsd = RMSD(reference=jnp.zeros((3, 3)))
        press = Pressure(n_atoms=3, volume_angstrom3=1000.0)

        assert isinstance(ke, Observable)
        assert isinstance(rmsd, Observable)
        assert isinstance(press, Observable)

    def test_all_importable_from_api(self):
        """All observables should be importable from prolix.api."""
        from prolix.api import KineticEnergy as KE_API
        from prolix.api import RMSD as RMSD_API
        from prolix.api import Pressure as Pressure_API

        assert KE_API is not None
        assert RMSD_API is not None
        assert Pressure_API is not None
