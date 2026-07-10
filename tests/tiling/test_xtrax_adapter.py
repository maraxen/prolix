"""Tests for xtrax.tiling adapter (#1842 / XR-BUDGET)."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from prolix.tiling.axes import N_ATOMS, N_MOLS
from prolix.tiling.planner import (
    AxisDecision,
    BatchPlanner,
    estimate_memory_theoretical,
)
from prolix.tiling.xtrax_adapter import (
    BudgetInfeasibleError,
    plan_axes_with_xtrax,
    prolix_axis_to_xtrax,
)
import prolix.tiling.xtrax_adapter as xtrax_adapter_mod


def test_xtrax_batch_planner_importable():
    from xtrax.tiling.plan import BatchPlanner as XtraxBatchPlanner

    assert XtraxBatchPlanner is not None


def test_prolix_vmap_sentinel_maps_to_xtrax_vmap_threshold():
    xax = prolix_axis_to_xtrax(N_ATOMS)
    assert xax.default_batch_size >= N_ATOMS.cardinality
    assert xax.name == "n_atoms"


def test_heterogeneous_axis_maps_to_positive_default_batch_size():
    xax = prolix_axis_to_xtrax(N_MOLS)
    assert xax.default_batch_size >= 1
    assert xax.heterogeneous is True


def test_no_secondary_prolix_budget_demotion_in_adapter_source():
    source = inspect.getsource(xtrax_adapter_mod)
    assert "prolix budget demotion" not in source
    assert "while estimate_memory" not in source


def test_plan_axes_with_xtrax_uses_memory_budget_not_per_axis_estimator(monkeypatch):
    captured: dict = {}

    class FakePlanner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def plan(self, specs):
            from xtrax.tiling.plan import AxisDecision as XD
            from xtrax.tiling.strategy import Vmap

            return type(
                "P",
                (),
                {
                    "decisions": tuple(
                        XD(
                            spec=s,
                            batch_size=s.default_batch_size,
                            reasoning="joint-budget: Vmap retained (test)",
                            strategy=Vmap(),
                        )
                        for s in specs
                    )
                },
            )()

    monkeypatch.setattr(xtrax_adapter_mod, "XtraxBatchPlanner", FakePlanner)

    def estimate(decisions: list[AxisDecision]) -> float:
        return 1.0

    atoms = dataclasses.replace(N_ATOMS, cardinality=8, heterogeneous=False)
    plan = plan_axes_with_xtrax(
        axes=[atoms],
        budget_bytes=1_000_000,
        estimate_memory=estimate,
    )
    assert "budget" in captured
    assert captured.get("memory_estimator") is None
    assert captured["budget"].bytes == 1_000_000
    assert plan.decision_for("n_atoms").batch_size == 0
    assert plan.budget_exceeded is False


def test_plan_axes_with_xtrax_heterogeneous_n_mols_safe_map():
    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, 1024.0, 2.0)

    plan = plan_axes_with_xtrax(
        axes=[N_ATOMS, N_MOLS],
        budget_bytes=1e18,
        estimate_memory=estimate,
    )
    mol = plan.decision_for("n_mols")
    assert mol.batch_size > 0


def test_plan_axes_with_xtrax_joint_demotion_under_tight_budget():
    """Homogeneous axis with cardinality > default_batch_size demotes under budget."""
    atoms = dataclasses.replace(
        N_ATOMS,
        cardinality=64,
        default_batch_size=8,
        tile_granularity=8,
        heterogeneous=False,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, base_shape_bytes=10.0, activation_multiplier=1.0)

    # Vmap product = 64*10 = 640; SafeMap tile=8 → 80. Budget between them.
    plan = plan_axes_with_xtrax(
        axes=[atoms],
        budget_bytes=200,
        estimate_memory=estimate,
    )
    assert plan.decision_for("n_atoms").batch_size == 8
    assert "joint-budget" in plan.decision_for("n_atoms").reasoning
    assert plan.budget_exceeded is False


def test_plan_axes_with_xtrax_infeasible_raises():
    atoms = dataclasses.replace(
        N_ATOMS,
        cardinality=64,
        default_batch_size=8,
        tile_granularity=8,
        heterogeneous=False,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, base_shape_bytes=10.0, activation_multiplier=1.0)

    with pytest.raises(BudgetInfeasibleError):
        plan_axes_with_xtrax(
            axes=[atoms],
            budget_bytes=1,
            estimate_memory=estimate,
        )


def test_hetero_only_over_budget_raises():
    mols = dataclasses.replace(N_MOLS, cardinality=4, heterogeneous=True)

    def estimate(decisions: list[AxisDecision]) -> float:
        return 1e12

    with pytest.raises(BudgetInfeasibleError):
        plan_axes_with_xtrax(
            axes=[mols],
            budget_bytes=100,
            estimate_memory=estimate,
        )


def test_batch_planner_plan_with_xtrax_preserves_prolix_api():
    """plan() aliases plan_with_xtrax() — single xtrax authority (XR-KILL-FORK)."""
    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, 1024.0, 2.0)

    planner = BatchPlanner(
        axes=[N_ATOMS, N_MOLS],
        budget_bytes=1e18,
        estimate_memory=estimate,
    )
    via_plan = planner.plan()
    via_xtrax = planner.plan_with_xtrax()
    assert via_plan.decision_for("n_mols").batch_size > 0
    assert via_xtrax.decision_for("n_mols").batch_size > 0
    assert [(d.axis.name, d.batch_size) for d in via_plan.decisions] == [
        (d.axis.name, d.batch_size) for d in via_xtrax.decisions
    ]
    assert hasattr(via_xtrax, "exceeded_budget")
    assert via_plan.budget_exceeded is False
