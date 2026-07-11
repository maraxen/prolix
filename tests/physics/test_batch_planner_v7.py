"""V7 / B5: BatchPlanner joint-budget decision contracts (XR-KILL-FORK).

Validates xtrax MemoryBudget semantics via the thin ``BatchPlanner.plan()``
facade: hetero SafeMap, generous Vmap retention, tight demotion, infeasible
raise, and determinism. Does not encode the deleted prolix greedy oracle.
"""

from __future__ import annotations

import dataclasses

import pytest

from prolix.tiling.axes import N_ATOMS, N_BONDS, N_MOLS
from prolix.tiling.planner import (
    AxisDecision,
    AxisSpec,
    BatchPlanner,
    estimate_memory_theoretical,
)
from prolix.tiling.xtrax_adapter import BudgetInfeasibleError


def _estimate(decisions: list[AxisDecision]) -> float:
    return estimate_memory_theoretical(
        decisions,
        base_shape_bytes=1024.0,
        activation_multiplier=2.0,
    )


def _planner(budget_bytes: float, axes: list[AxisSpec] | None = None) -> BatchPlanner:
    return BatchPlanner(
        axes=axes if axes is not None else [N_ATOMS, N_BONDS, N_MOLS],
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
    """N_MOLS must never be vmap (batch_size == 0) at feasible budgets."""
    baseline = _all_vmap_plan()
    for mult in (2.0, 1.0):
        plan = _planner(baseline * mult).plan()
        mol = plan.decision_for("n_mols")
        assert mol.batch_size > 0, f"budget={mult}×: n_mols must use safe_map"
        assert "heterogeneous" in mol.reasoning


def test_generous_budget_keeps_homogeneous_axes_vmap():
    """At 2× baseline estimate, homogeneous axes stay vmap."""
    baseline = _all_vmap_plan()
    plan = _planner(baseline * 2.0).plan()
    atoms = plan.decision_for("n_atoms")
    bonds = plan.decision_for("n_bonds")
    assert atoms.batch_size == 0
    assert bonds.batch_size == 0
    assert not plan.exceeded_budget()


def test_tight_budget_joint_demotion_to_tile():
    """Homogeneous axis with cardinality > default_batch_size demotes under budget."""
    atoms = dataclasses.replace(
        N_ATOMS,
        cardinality=64,
        default_batch_size=8,
        tile_granularity=8,
        heterogeneous=False,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(
            decisions, base_shape_bytes=10.0, activation_multiplier=1.0
        )

    # Vmap product = 64*10 = 640; SafeMap tile=8 → 80. Budget between them.
    planner = BatchPlanner(
        axes=[atoms],
        budget_bytes=200,
        estimate_memory=estimate,
    )
    plan = planner.plan()
    assert plan.decision_for("n_atoms").batch_size == 8
    assert "joint-budget" in plan.decision_for("n_atoms").reasoning
    assert plan.budget_exceeded is False


def test_infeasible_budget_raises():
    """Budget below minimum tile estimate → BudgetInfeasibleError (fail-loud)."""
    atoms = dataclasses.replace(
        N_ATOMS,
        cardinality=64,
        default_batch_size=8,
        tile_granularity=8,
        heterogeneous=False,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(
            decisions, base_shape_bytes=10.0, activation_multiplier=1.0
        )

    with pytest.raises(BudgetInfeasibleError):
        BatchPlanner(
            axes=[atoms],
            budget_bytes=1,
            estimate_memory=estimate,
        ).plan()


def test_plan_is_deterministic_for_fixed_budget():
    """Same inputs → identical decisions."""
    baseline = _all_vmap_plan()
    p = _planner(baseline)
    plan_a = p.plan()
    plan_b = p.plan()
    assert [(d.axis.name, d.batch_size) for d in plan_a.decisions] == [
        (d.axis.name, d.batch_size) for d in plan_b.decisions
    ]


def test_plan_aliases_plan_with_xtrax():
    """Facade: plan() and plan_with_xtrax() agree (single authority)."""
    baseline = _all_vmap_plan()
    p = _planner(baseline * 2.0)
    via_plan = p.plan()
    via_xtrax = p.plan_with_xtrax()
    assert [(d.axis.name, d.batch_size) for d in via_plan.decisions] == [
        (d.axis.name, d.batch_size) for d in via_xtrax.decisions
    ]
