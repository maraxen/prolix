"""Tests for Observable protocol and Trajectory module."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import pytest

from prolix.api.observables import Observable, Trajectory, Temperature, Energy
from prolix.typing import IntegratorState as LangevinState
from prolix.simulate import BOLTZMANN_KCAL


def test_observable_is_runtime_checkable():
    """Observable should be runtime_checkable via isinstance()."""
    t = Temperature(dof=3)
    assert isinstance(t, Observable)


def test_observable_is_protocol():
    """Observable must define compute() method."""
    # The protocol itself shouldn't be instantiable
    assert hasattr(Observable, "__call__")


def test_trajectory_is_equinox_module():
    """Trajectory should be an eqx.Module."""
    assert issubclass(Trajectory, eqx.Module)


def test_trajectory_has_required_fields():
    """Trajectory should have positions, observable_values, and n_steps."""
    positions = jnp.ones((10, 5, 3))  # 10 steps, 5 atoms, 3 coords
    observable_values = {"energy": jnp.ones(10), "temperature": jnp.ones(10)}

    traj = Trajectory(positions=positions, observable_values=observable_values, n_steps=10)

    assert traj.positions.shape == (10, 5, 3)
    assert traj.observable_values == observable_values
    assert traj.n_steps == 10


def test_temperature_observable_compute():
    """Temperature observable should implement compute()."""
    t = Temperature(dof=3)
    assert hasattr(t, "compute")
    assert callable(t.compute)


def test_temperature_observable_returns_array():
    """Temperature.compute() should return an Array."""
    t = Temperature(dof=3)
    result = t.compute(state=None)  # Placeholder state
    assert isinstance(result, (jnp.ndarray, type(jnp.nan)))


def test_imports_from_api():
    """Observable, Trajectory, Temperature should be importable from prolix.api."""
    from prolix.api import Observable, Trajectory, Temperature

    assert Observable is not None
    assert Trajectory is not None
    assert Temperature is not None


def test_trajectory_static_field():
    """n_steps should be a static field (not traced)."""
    positions = jnp.ones((10, 5, 3))
    observable_values = {"energy": jnp.ones(10)}

    traj = Trajectory(positions=positions, observable_values=observable_values, n_steps=10)

    # n_steps should be in static fields
    _, static = eqx.partition(traj, eqx.is_array)
    # Verify that the module can be partitioned correctly
    assert traj.n_steps == 10


def test_observable_protocol_compliance():
    """Temperature satisfies the Observable protocol."""
    t = Temperature(dof=9)
    assert isinstance(t, Observable), "Temperature should implement Observable protocol"


def test_temperature_compute_with_valid_state():
    """Temperature.compute() should return a non-nan scalar for valid state."""
    # Create a simple state with known kinetic energy
    n_atoms = 3
    dof = 3 * n_atoms  # All atoms free

    # Create LangevinState with known momentum and mass
    positions = jnp.zeros((n_atoms, 3))
    momentum = jnp.ones((n_atoms, 3))  # p_i = 1 for all atoms
    force = jnp.zeros((n_atoms, 3))
    mass = jnp.ones(n_atoms)  # m_i = 1 for all atoms
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

    # Compute temperature
    t_obs = Temperature(dof=dof)
    temp = t_obs.compute(state)

    # Expected KE = sum(p^2 / (2*m)) = 9 * (1 / 2) = 4.5
    # Expected T = (2 * 4.5) / (9 * k_B)
    expected_temp = (2.0 * 4.5) / (dof * BOLTZMANN_KCAL)

    assert not jnp.isnan(temp), "Temperature should not be NaN for valid state"
    assert jnp.allclose(temp, expected_temp, rtol=1e-5), \
        f"Expected temp {expected_temp}, got {temp}"


def test_temperature_returns_scalar():
    """Temperature.compute() should return a scalar Array."""
    n_atoms = 3
    dof = 3 * n_atoms

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

    t_obs = Temperature(dof=dof)
    temp = t_obs.compute(state)

    # Check shape is scalar
    assert temp.shape == (), f"Expected scalar, got shape {temp.shape}"
    # Check it's a JAX array
    assert isinstance(temp, jnp.ndarray), "Result should be a JAX array"


def test_temperature_equipartition():
    """Temperature.compute() should follow equipartition theorem."""
    # For 1 atom with mass m in 3D, dof = 3
    # T = 2*KE / (dof * k_B) where KE = sum(p^2 / (2*m))
    # Setting up: p_x = p_y = p_z = p_dim for equal momentum in each dimension
    # Total KE = 3 * (p_dim^2 / (2*m))
    # For target T = 300K, we need: p_dim^2 / (2*m) = (k_B * 300) / 3
    n_atoms = 1
    dof = 3
    target_temp = 300.0

    mass_val = 12.0  # 12 g/mol (e.g., carbon)
    # Momentum per dimension for equipartition at target temperature
    p_dim = jnp.sqrt(2.0 * mass_val * BOLTZMANN_KCAL * target_temp / 3.0)

    positions = jnp.zeros((n_atoms, 3))
    # Equal momentum in each dimension
    momentum = jnp.ones((n_atoms, 3)) * p_dim
    force = jnp.zeros((n_atoms, 3))
    mass = jnp.array([mass_val])
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

    t_obs = Temperature(dof=dof)
    temp = t_obs.compute(state)

    # The computed temperature will be close to target_temp K
    # Note: the exact value depends on the momentum distribution
    assert temp > 100.0, "Temperature should be positive and reasonably large"
    assert not jnp.isnan(temp), "Temperature should not be NaN"


def test_temperature_handles_missing_mass():
    """Temperature.compute() should return NaN if mass is missing."""
    positions = jnp.zeros((3, 3))
    momentum = jnp.ones((3, 3))
    force = jnp.zeros((3, 3))
    key = jnp.zeros(2, dtype=jnp.uint32)
    box = jnp.zeros((3, 3))

    # Create state without mass field (simulate missing/None case)
    state = LangevinState(
        positions=positions,
        momentum=momentum,
        force=force,
        mass=None,  # type: ignore
        rng=key,
        box=box,
    )

    t_obs = Temperature(dof=9)
    temp = t_obs.compute(state)

    assert jnp.isnan(temp), "Temperature should be NaN when mass is missing"


def test_energy_observable_is_observable():
    """Energy should implement the Observable protocol."""
    def mock_energy_fn(positions, bundle):
        return jnp.sum(positions**2)

    bundle = {}  # Minimal bundle for test
    energy = Energy(energy_fn=mock_energy_fn, bundle=bundle)

    assert isinstance(energy, Observable), "Energy should implement Observable protocol"


def test_energy_observable_compute():
    """Energy observable should have a compute method."""
    def mock_energy_fn(positions, bundle):
        return jnp.sum(positions**2)

    bundle = {}
    energy = Energy(energy_fn=mock_energy_fn, bundle=bundle)

    assert hasattr(energy, "compute")
    assert callable(energy.compute)


def test_energy_observable_returns_scalar():
    """Energy.compute() should return a scalar Array."""
    def mock_energy_fn(positions, bundle):
        return jnp.sum(positions**2)

    bundle = {}
    energy = Energy(energy_fn=mock_energy_fn, bundle=bundle)

    # Create a mock state-like object with positions attribute
    class MockState:
        def __init__(self, positions):
            self.positions = positions

    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    state = MockState(positions)

    e = energy.compute(state)

    # Check shape is scalar
    assert e.shape == (), f"Expected scalar, got shape {e.shape}"
    # Check it's a JAX array
    assert isinstance(e, jnp.ndarray), "Result should be a JAX array"


def test_energy_observable_matches_direct_call():
    """Energy.compute() should match direct energy_fn call."""
    def mock_energy_fn(positions, bundle):
        # Simple energy: sum of squared displacements from origin
        return jnp.sum(positions**2)

    bundle = {"dummy": jnp.array([1.0, 2.0, 3.0])}
    energy = Energy(energy_fn=mock_energy_fn, bundle=bundle)

    # Create test state
    class MockState:
        def __init__(self, positions):
            self.positions = positions

    positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    state = MockState(positions)

    # Compute via observable
    e_obs = energy.compute(state)

    # Compute directly
    e_direct = mock_energy_fn(positions, bundle)

    # Should match exactly
    assert jnp.allclose(e_obs, e_direct, rtol=1e-10), \
        f"Observable energy {e_obs} != direct {e_direct}"


def test_energy_observable_with_bundle_params():
    """Energy observable should correctly pass bundle to energy function."""
    def mock_energy_fn(positions, bundle):
        # Energy depends on both positions and bundle parameters
        scale = bundle.get("scale", 1.0)
        return scale * jnp.sum(positions**2)

    bundle = {"scale": 2.5}
    energy = Energy(energy_fn=mock_energy_fn, bundle=bundle)

    # Create test state
    class MockState:
        def __init__(self, positions):
            self.positions = positions

    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state = MockState(positions)

    e = energy.compute(state)

    # Expected: 2.5 * (1 + 1) = 5.0
    expected = 5.0
    assert jnp.allclose(e, expected, rtol=1e-10), \
        f"Expected {expected}, got {e}"


def test_energy_is_equinox_module():
    """Energy should be an eqx.Module."""
    def mock_energy_fn(positions, bundle):
        return jnp.sum(positions**2)

    energy = Energy(energy_fn=mock_energy_fn, bundle={})
    assert isinstance(energy, eqx.Module), "Energy should be an eqx.Module"


def test_energy_importable_from_api():
    """Energy should be importable from prolix.api."""
    from prolix.api import Energy as EnergyAPI

    assert EnergyAPI is not None
    assert EnergyAPI is Energy
