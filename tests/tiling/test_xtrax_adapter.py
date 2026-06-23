"""Tests for xtrax.tiling adapter (#1842)."""

from __future__ import annotations

from prolix.tiling.axes import N_ATOMS, N_MOLS
from prolix.tiling.planner import (
    AxisDecision,
    BatchPlanner,
    estimate_memory_theoretical,
)
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax, prolix_axis_to_xtrax


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


def test_batch_planner_plan_with_xtrax_preserves_prolix_api():
    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, 1024.0, 2.0)

    planner = BatchPlanner(
        axes=[N_ATOMS, N_MOLS],
        budget_bytes=1e18,
        estimate_memory=estimate,
    )
    greedy = planner.plan()
    xtrax_backed = planner.plan_with_xtrax()
    assert greedy.decision_for("n_mols").batch_size > 0
    assert xtrax_backed.decision_for("n_mols").batch_size > 0
    assert hasattr(xtrax_backed, "exceeded_budget")
