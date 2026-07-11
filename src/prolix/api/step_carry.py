"""MD step-axis CarrySpec planning (XR-CARRY)."""

from __future__ import annotations

import dataclasses

from xtrax.tiling.carry import CarrySpec

from prolix.tiling.axes import N_STEPS
from prolix.tiling.planner import AxisDecision, BatchPlan, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

__all__ = [
    "plan_n_steps_with_carry",
    "stub_step_transition",
]


def stub_step_transition(carry, x):
    """Placeholder transition for planning-time CarrySpec (execute builds the real one)."""
    return carry, x


def plan_n_steps_with_carry(
    n_steps: int,
    *,
    budget_bytes: float | int = 1 << 40,
) -> BatchPlan:
    """Plan the ``n_steps`` axis with a CarrySpec so xtrax pre-demotes to Scan.

    The CarrySpec uses a stub transition/init; execution wires the real
    integrator step through ``dispatch_n_steps``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    axis = dataclasses.replace(N_STEPS, cardinality=int(n_steps), heterogeneous=False)
    cs = CarrySpec(
        axis_name=N_STEPS.name,
        init=None,
        transition=stub_step_transition,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(
            decisions, base_shape_bytes=8.0, activation_multiplier=1.0
        )

    return plan_axes_with_xtrax(
        axes=[axis],
        budget_bytes=budget_bytes,
        estimate_memory=estimate,
        carry_specs=[cs],
    )
