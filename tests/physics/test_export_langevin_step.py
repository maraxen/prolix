"""Test export_langevin_step removal of NotImplementedError and jax.export compatibility.

Tests verify:
1. export_langevin_step no longer raises NotImplementedError
2. Exported step returns LangevinState with correct shape
3. Exported step is JIT-compatible
4. Exported step is vmap-compatible
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

import pytest

# XA-CI: heavy parity/compile — deselect from GitHub-faithful suite.
pytestmark = pytest.mark.slow

from prolix.export import export_langevin_step
from prolix.types.integrators import IntegratorConfig, LangevinState


def _state(n: int = 10) -> LangevinState:
    """Create a simple LangevinState for testing."""
    return LangevinState(
        positions=jnp.zeros((n, 3)),
        momenta=jnp.zeros((n, 3)),
        forces=jnp.ones((n, 3)) * -1.0,
        key=random.PRNGKey(0),
        box=jnp.zeros((3, 3)),
    )


def _config() -> IntegratorConfig:
    """Create a test IntegratorConfig."""
    return IntegratorConfig(
        thermostat="langevin",
        has_pbc=False,
        dt=0.5,
        kT=1.0,
        gamma=1.0,
    )


def test_export_langevin_step_no_not_implemented():
    """export_langevin_step must not raise NotImplementedError."""
    def step(state: LangevinState) -> LangevinState:
        return state

    exported = export_langevin_step(step, _config())
    assert callable(exported)


def test_exported_step_returns_langevin_state():
    """Exported step must return LangevinState with correct shape."""
    def step(state: LangevinState) -> LangevinState:
        return LangevinState(
            positions=state.positions + state.momenta * 0.5,
            momenta=state.momenta,
            forces=state.forces,
            key=state.key,
            box=state.box,
        )

    exported = export_langevin_step(step, _config())
    result = exported(_state())
    assert isinstance(result, LangevinState)
    assert result.positions.shape == (10, 3)
    assert result.momenta.shape == (10, 3)
    assert result.forces.shape == (10, 3)


def test_exported_step_is_jit_compatible():
    """Exported step must be JIT-compatible."""
    def step(s: LangevinState) -> LangevinState:
        return s

    exported = export_langevin_step(step, _config())
    jitted = jax.jit(exported)
    result = jitted(_state())
    assert result.positions.shape == (10, 3)


def test_exported_step_is_vmap_compatible():
    """Exported step must be vmap-compatible with batched state."""
    B = 4

    def step(s: LangevinState) -> LangevinState:
        return s

    exported = export_langevin_step(step, _config())
    batch_state = LangevinState(
        positions=jnp.zeros((B, 10, 3)),
        momenta=jnp.zeros((B, 10, 3)),
        forces=jnp.zeros((B, 10, 3)),
        key=random.split(random.PRNGKey(0), B),
        box=jnp.zeros((B, 3, 3)),
    )
    result = jax.vmap(exported)(batch_state)
    assert result.positions.shape == (B, 10, 3)
    assert result.momenta.shape == (B, 10, 3)
