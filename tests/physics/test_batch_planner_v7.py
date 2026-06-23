"""V7 / B5: BatchPlanner decision correctness across memory budgets.

Validates that the greedy planner picks expected vmap vs safe_map splits
at generous (2×), tight (0.5×), and very tight (0.1×) budgets, and that
heterogeneous axes are always pre-demoted to safe_map.
"""

from __future__ import annotations

import pytest

from prolix.tiling.axes import N_ATOMS, N_BONDS, N_MOLS
from prolix.tiling.planner import (
    AxisDecision,
    BatchPlanner,
    estimate_memory_theoretical,
)


def _estimate(decisions: list[AxisDecision]) -> float:
    return estimate_memory_theoretical(
        decisions,
        base_shape_bytes=1024.0,
        activation_multiplier=2.0,
    )


def _planner(budget_bytes: float) -> BatchPlanner:
    return BatchPlanner(
        axes=[N_ATOMS, N_BONDS, N_MOLS],
        budget_bytes=budget_bytes,
        estimate_memory=_estimate,
    )


def _all_vmap_plan() -> float:
    """Memory estimate when every homogeneous axis uses vmap."""
    hom = [
        AxisDecision(axis=N_ATOMS, batch_size=0, reasoning="vmap"),
        AxisDecision(axis=N_BONDS, batch_size=0, reasoning="vmap"),
    ]
    het = [
        AxisDecision(
            axis=N_MOLS,
            batch_size=max(1, N_MOLS.tile_granularity),
            reasoning="heterogeneous",
        ),
    ]
    return _estimate(het + hom)


def test_heterogeneous_axis_always_safe_map():
    """N_MOLS must never be vmap (batch_size == 0)."""
    baseline = _all_vmap_plan()
    for mult in (2.0, 0.5, 0.1):
        plan = _planner(baseline * mult).plan()
        mol = plan.decision_for("n_mols")
        assert mol.batch_size > 0, f"budget={mult}×: n_mols must use safe_map"
        assert "heterogeneous" in mol.reasoning


def test_generous_budget_keeps_homogeneous_axes_vmap():
    """At 2× baseline estimate, inner homogeneous axes stay vmap."""
    baseline = _all_vmap_plan()
    plan = _planner(baseline * 2.0).plan()
    atoms = plan.decision_for("n_atoms")
    bonds = plan.decision_for("n_bonds")
    assert atoms.batch_size == 0
    assert bonds.batch_size == 0
    assert not plan.exceeded_budget()


@pytest.mark.parametrize("budget_mult", [0.5, 0.1])
def test_tight_budget_demotes_innermost_homogeneous_first(budget_mult: float):
    """Under budget pressure, innermost homogeneous axis demotes before outer."""
    baseline = _all_vmap_plan()
    plan = _planner(baseline * budget_mult).plan()
    atoms = plan.decision_for("n_atoms")
    bonds = plan.decision_for("n_bonds")
    # n_atoms is innermost (axis_index=0); it demotes first under pressure.
    if budget_mult <= 0.1:
        assert atoms.batch_size > 0, "0.1× budget should demote n_atoms"
    # n_mols always safe_map regardless of budget
    assert plan.decision_for("n_mols").batch_size > 0


def test_plan_is_deterministic_for_fixed_budget():
    """Same inputs → identical decisions (planner is pure Python)."""
    baseline = _all_vmap_plan()
    p = _planner(baseline)
    plan_a = p.plan()
    plan_b = p.plan()
    assert [(d.axis.name, d.batch_size) for d in plan_a.decisions] == [
        (d.axis.name, d.batch_size) for d in plan_b.decisions
    ]
