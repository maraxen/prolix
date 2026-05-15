import dataclasses
import pytest
import jax
import jax.numpy as jnp
from jax import random
from prolix.types.integrators import (
    IntegratorState, LangevinState, CSVRState, NHCState, IntegratorConfig
)


def _make_base_kwargs(n_atoms=10):
    key = random.PRNGKey(0)
    return dict(
        positions=jnp.zeros((n_atoms, 3)),
        momenta=jnp.zeros((n_atoms, 3)),
        forces=jnp.zeros((n_atoms, 3)),
        key=key,
        box=jnp.zeros((3, 3)),
    )


def test_langevin_state_no_optional_fields():
    state = LangevinState(**_make_base_kwargs())
    assert state.positions is not None
    assert state.momenta is not None
    assert state.forces is not None
    assert state.box is not None


def test_csvr_state_adds_ke_half():
    state = CSVRState(**_make_base_kwargs(), csvr_ke_half=jnp.array(0.0))
    assert state.csvr_ke_half.shape == ()


def test_nhc_state_adds_chains():
    M = 4
    state = NHCState(
        **_make_base_kwargs(),
        nhc_xi=jnp.zeros(M),
        nhc_vxi=jnp.zeros(M),
    )
    assert state.nhc_xi.shape == (M,)
    assert state.nhc_vxi.shape == (M,)


def test_integrator_config_is_frozen():
    config = IntegratorConfig(
        thermostat="langevin", has_pbc=True, dt=0.5, kT=1.0, gamma=1.0
    )
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        config.dt = 1.0  # type: ignore[misc]


def test_langevin_state_vmap():
    B = 4
    keys = random.split(random.PRNGKey(0), B)
    states = jax.vmap(lambda k: LangevinState(
        positions=jnp.zeros((10, 3)),
        momenta=jnp.zeros((10, 3)),
        forces=jnp.zeros((10, 3)),
        key=k,
        box=jnp.zeros((3, 3)),
    ))(keys)
    assert states.positions.shape == (B, 10, 3)


def test_langevin_state_jit_passthrough():
    state = LangevinState(**_make_base_kwargs())

    @jax.jit
    def identity(s):
        return s

    out = identity(state)
    assert out.positions.shape == state.positions.shape
