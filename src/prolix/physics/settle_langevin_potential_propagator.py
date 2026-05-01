"""SETTLE+Langevin step with a single cached potential evaluation.

One ``jax.value_and_grad`` (via :func:`value_energy_and_forces`) supplies both
scalar energy and forces written into :class:`~prolix.physics.simulate.NVTLangevinState`
for the outer ``jax.jit`` / ``jax.lax.scan`` boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from prolix.physics import settle
from prolix.physics.md_potential_bundle import value_energy_and_forces
from prolix.physics.simulate import NVTLangevinState


@dataclass(frozen=True)
class SettleLangevinPotentialStepMetrics:
  """Per-step diagnostics from the propagator (same AD site as cached forces)."""

  settle_impulse: Any
  potential_energy: Any


def settle_langevin_potential_cached_step(
  state: NVTLangevinState,
  *,
  energy_fn: Callable[..., Any],
  shift_fn: Callable[..., Any],
  mass_col: Any,
  water_indices: Any,
  box_vec: Any,
  dt_akma: float,
  gamma_reduced: float,
  kT: float,
  project_ou_momentum_rigid: bool,
) -> tuple[NVTLangevinState, SettleLangevinPotentialStepMetrics]:
  """One BAOAB-style SETTLE+Langevin step; forces and logged energy share one ``value_and_grad``."""
  positions_old = state.position
  momentum = settle._langevin_step_b(state.momentum, state.force, dt_akma)
  position = settle._langevin_step_a(state.position, momentum, state.mass, dt_akma, shift_fn)

  if project_ou_momentum_rigid:
    momentum, key = settle._langevin_step_o_constrained(
      momentum, position, state.mass, gamma_reduced, dt_akma, kT, state.rng, water_indices
    )
  else:
    momentum, key = settle._langevin_step_o(momentum, state.mass, gamma_reduced, dt_akma, kT, state.rng)

  position = settle._langevin_step_a(position, momentum, state.mass, dt_akma, shift_fn)
  position = settle.settle_positions(
    position,
    positions_old,
    water_indices,
    settle.TIP3P_ROH,
    settle.TIP3P_RHH,
    15.999,
    1.008,
    box_vec,
  )
  bundle = value_energy_and_forces(energy_fn, position)
  force = bundle.forces
  momentum = settle._langevin_step_b(momentum, force, dt_akma)

  momentum_pre_settle = momentum
  momentum = settle._langevin_settle_vel(
    momentum,
    positions_old,
    position,
    mass_col,
    water_indices,
    dt_akma,
    15.999,
    1.008,
    n_iters=10,
    settle_velocity_tol=None,
  )
  settle_impulse = jnp.linalg.norm((momentum - momentum_pre_settle).reshape(-1))

  next_state = NVTLangevinState(position, momentum, force, mass_col, key)
  metrics = SettleLangevinPotentialStepMetrics(
    settle_impulse=settle_impulse,
    potential_energy=bundle.energy,
  )
  return next_state, metrics
