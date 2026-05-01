"""Propagator step outcome with host-checkable boolean predicates.

Lets compiled MD steps attach compact predicate arrays (e.g. finiteness masks)
that the host can ``bool()`` after ``block_until_ready``, instead of relying on
``jax.debug.print`` for sanity checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax import Array

T = TypeVar("T")


@dataclass(frozen=True)
class StepResult(Generic[T]):
  """Outcome of a propagator step plus zero or more boolean predicates."""

  value: T
  """Primary return (often next ``State``)."""

  predicates: tuple[Array, ...] = ()
  """Each entry should be a scalar boolean JAX array (checked outside ``jit``)."""

  @property
  def all_ok(self) -> Array:
    """Scalar ``True`` iff every predicate is True (empty → True)."""
    if not self.predicates:
      return jnp.array(True)
    return jnp.all(jnp.stack(self.predicates))


def compose_steps(*steps: Callable[[Any], StepResult[Any]]) -> Callable[[Any], StepResult[Any]]:
  """Run ``StepResult``-returning callables left-to-right, threading ``.value``.

  Assertions are concatenated in order. If any step's ``all_ok`` is False,
  later steps still run (caller may branch); use short-circuit wrappers if needed.
  """

  if not steps:
    raise ValueError("compose_steps requires at least one step")

  def composed(x: Any) -> StepResult[Any]:
    preds: list[Array] = []
    cur = x
    for fn in steps:
      out = fn(cur)
      preds.extend(out.predicates)
      cur = out.value
    return StepResult(cur, tuple(preds))

  return composed
