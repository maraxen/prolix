"""Abstract ``Step`` base class for integrator steps (leaf module).

Split out of ``step_system`` so ``barostat`` and ``virtual_sites_step`` can subclass
``Step`` without importing ``step_system`` -- which removes the lazy imports that
``step_system.make_step`` used to dodge that cycle (Phase 1.2).
``prolix.physics.step_system`` re-exports ``Step`` for backward compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx

from prolix.typing import IntegratorParams, IntegratorState

__all__ = ["Step"]


class Step(eqx.Module, ABC):
  """Abstract base class for integrator steps.

  All steps are Equinox Modules that implement a pure `apply` method.
  This ensures they are valid JAX PyTrees and can capture static configuration.

  Purity Guarantee:
  - No side effects (no global state mutation)
  - No Python control flow (use jax.lax.cond, jax.lax.scan, etc.)
  - Output deterministic given input state
  """

  @abstractmethod
  def apply(
      self,
      state: IntegratorState,
      params: IntegratorParams,
  ) -> IntegratorState:
    """Apply one step of the integrator.

    Args:
        state: Current integrator state.
        params: Standardized integrator parameters (dt, kT, etc.).

    Returns:
        Updated state (including updated PRNGKey if stochastic).
    """
    raise NotImplementedError
