"""Tests for prolix.physics.step_result."""

from __future__ import annotations

import jax.numpy as jnp

from prolix.physics.step_result import StepResult, compose_steps


def test_step_result_all_ok_empty():
  r = StepResult(value=1, predicates=())
  assert bool(r.all_ok)


def test_compose_steps_threads_value_and_predicates():
  def add_one(x: int) -> StepResult[int]:
    return StepResult(x + 1, (jnp.array(x >= 0),))

  def double(x: int) -> StepResult[int]:
    return StepResult(x * 2, (jnp.array(x < 100),))

  out = compose_steps(add_one, double)(3)
  assert out.value == 8
  assert out.predicates == (jnp.array(True), jnp.array(True))
  assert bool(out.all_ok)
