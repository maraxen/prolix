"""XR-CARRY: CarrySpec → Scan planning contracts."""

from __future__ import annotations

import dataclasses

import pytest

from prolix.api.step_carry import plan_n_steps_with_carry
from prolix.tiling.axes import N_STEPS
from prolix.tiling.planner import AxisDecision, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax
from xtrax.tiling.carry import CarrySpec


def _stub_transition(carry, x):
    return carry, x


def test_plan_n_steps_with_carry_is_carry_bearing_scan():
    plan = plan_n_steps_with_carry(n_steps=16)
    d = plan.decision_for("n_steps")
    assert "carry-bearing scan" in d.reasoning
    assert d.batch_size >= 1


def test_carry_spec_on_heterogeneous_axis_raises():
    steps = dataclasses.replace(N_STEPS, cardinality=8, heterogeneous=True)
    cs = CarrySpec(
        axis_name=N_STEPS.name,
        init=None,
        transition=_stub_transition,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, 1024.0, 1.0)

    with pytest.raises(ValueError, match="[Hh]eterogeneous|static carry"):
        plan_axes_with_xtrax(
            axes=[steps],
            budget_bytes=1e18,
            estimate_memory=estimate,
            carry_specs=[cs],
        )
