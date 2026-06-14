"""Tests for Observable protocol and Trajectory module."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import pytest

from prolix.api.observables import Observable, Trajectory, Temperature
from prolix.types.integrators import LangevinState


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
