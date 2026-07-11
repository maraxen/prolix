"""Adapter between prolix.tiling.planner and xtrax.tiling.plan (#1842 / XR-BUDGET).

Converts AxisSpec / BatchPlan between the two representations and delegates
strategy selection to xtrax joint ``MemoryBudget`` planning. Heterogeneous
axes are fixed to SafeMap before the joint plan; homogeneous axes are planned
under ``MemoryBudget``. No secondary host demotion loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.plan import AxisDecision as XtraxAxisDecision
from xtrax.tiling.plan import AxisSpec as XtraxAxisSpec
from xtrax.tiling.plan import BatchPlanner as XtraxBatchPlanner
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap

from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan

__all__ = [
    "BudgetInfeasibleError",
    "CarrySpec",
    "DedupSpec",
    "MemoryBudget",
    "plan_axes_with_xtrax",
    "prolix_axis_to_xtrax",
    "xtrax_decision_to_prolix",
]


def prolix_axis_to_xtrax(axis: AxisSpec) -> XtraxAxisSpec:
    """Map prolix AxisSpec to xtrax AxisSpec.

    Prolix uses default_batch_size=0 as the vmap sentinel; xtrax uses
    cardinality <= default_batch_size for Vmap selection.
    """
    if axis.heterogeneous:
        default_batch_size = max(1, axis.default_batch_size or axis.tile_granularity)
    elif axis.default_batch_size == 0:
        default_batch_size = max(1, axis.cardinality)
    else:
        default_batch_size = max(1, axis.default_batch_size)

    return XtraxAxisSpec(
        name=axis.name,
        cardinality=max(1, axis.cardinality),
        default_batch_size=default_batch_size,
        tile_granularity=max(1, axis.tile_granularity),
        heterogeneous=axis.heterogeneous,
    )


def xtrax_decision_to_prolix(
    decision: XtraxAxisDecision,
    axis: AxisSpec,
) -> AxisDecision:
    """Convert one xtrax AxisDecision to prolix batch_size semantics."""
    batch_size = _strategy_to_batch_size(decision, axis)
    return AxisDecision(
        axis=axis,
        batch_size=batch_size,
        reasoning=decision.reasoning,
    )


def _strategy_to_batch_size(
    decision: XtraxAxisDecision,
    axis: AxisSpec,
) -> int:
    """Map xtrax strategy objects to prolix batch_size (0 = vmap)."""
    strategy = decision.strategy
    if isinstance(strategy, Vmap):
        return 0
    if isinstance(strategy, SafeMap):
        return max(1, strategy.batch_size)
    if isinstance(strategy, Bucket):
        return max(1, decision.batch_size or axis.tile_granularity)
    if isinstance(strategy, Scan):
        return max(1, decision.batch_size or 1)
    if isinstance(strategy, DedupGather):
        return max(1, getattr(strategy, "k_bucket", None) or decision.batch_size or 1)
    if axis.heterogeneous:
        return max(1, decision.batch_size or axis.tile_granularity)
    return max(1, decision.batch_size)


def _hetero_safe_map_decision(axis: AxisSpec) -> AxisDecision:
    tile = max(1, axis.tile_granularity)
    return AxisDecision(
        axis=axis,
        batch_size=tile,
        reasoning="heterogeneous axis: element shapes vary; safe_map required",
    )


def plan_axes_with_xtrax(
    axes: list[AxisSpec],
    budget_bytes: float | int,
    estimate_memory: Callable[[list[AxisDecision]], float],
    carry_specs: list[CarrySpec] | None = None,
    dedup_specs: list[DedupSpec] | None = None,
) -> BatchPlan:
    """Plan using xtrax ``BatchPlanner(budget=MemoryBudget(...))``.

    Heterogeneous axes are fixed to SafeMap and excluded from joint demotion
    candidates. Homogeneous axes are planned under the joint budget. Optional
    ``carry_specs`` pre-demote named axes to ``Scan`` (XR-CARRY). Optional
    ``dedup_specs`` pre-demote named axes to ``DedupGather`` (XR-DEDUP). The
    estimate callable bridges xtrax ``AxisDecision`` sequences back to the
    caller-supplied prolix estimator (typically ``estimate_memory_theoretical``).

    Raises:
        BudgetInfeasibleError: If the joint estimate cannot fit ``budget_bytes``
            after demotion (or for hetero-only plans that already exceed budget).
        ValueError: If a CarrySpec targets a heterogeneous axis (xtrax contract).
    """
    budget_int = max(1, int(budget_bytes))
    sorted_axes = sorted(axes, key=lambda a: a.axis_index)
    axis_by_name = {ax.name: ax for ax in sorted_axes}
    carry_list = list(carry_specs or [])
    carry_names = {cs.axis_name for cs in carry_list}
    dedup_list = list(dedup_specs or [])
    dedup_names = {ds.axis_name for ds in dedup_list}

    # CarrySpec / DedupSpec axes must reach xtrax Phase-0/0b even if marked
    # heterogeneous so the planner can fail loud rather than silent SafeMap.
    het_decisions: list[AxisDecision] = []
    plan_axes: list[AxisSpec] = []
    for ax in sorted_axes:
        if ax.heterogeneous and ax.name not in carry_names and ax.name not in dedup_names:
            het_decisions.append(_hetero_safe_map_decision(ax))
        else:
            plan_axes.append(ax)

    def _finalize(all_decisions: list[AxisDecision]) -> BatchPlan:
        final_estimate = float(estimate_memory(all_decisions))
        if final_estimate > budget_int:
            state_desc = ", ".join(
                f"{d.axis.name}=batch_size={d.batch_size}" for d in all_decisions
            )
            raise BudgetInfeasibleError(
                f"plan cannot fit MemoryBudget: estimate {int(final_estimate)} B > "
                f"budget {budget_int} B; final decisions: {state_desc}"
            )
        return BatchPlan(
            decisions=all_decisions,
            total_memory_estimate=final_estimate,
            axes_by_index={ax.axis_index: ax for ax in sorted_axes},
            budget_exceeded=False,
        )

    if not plan_axes:
        return _finalize(het_decisions)

    xspecs = [prolix_axis_to_xtrax(ax) for ax in plan_axes]
    heterogeneous_axes = {ax.name for ax in sorted_axes if ax.heterogeneous}

    def joint_estimate(xdecisions: Sequence[XtraxAxisDecision]) -> int:
        prolix_hom = [
            xtrax_decision_to_prolix(xd, axis_by_name[xd.spec.name]) for xd in xdecisions
        ]
        return int(estimate_memory(het_decisions + prolix_hom))

    budget = MemoryBudget(bytes=budget_int, estimate=joint_estimate)
    xplan = XtraxBatchPlanner(
        budget=budget,
        heterogeneous_axes=heterogeneous_axes,
        carry_specs=carry_list,
        dedup_specs=dedup_list,
    ).plan(xspecs)

    planned = [
        xtrax_decision_to_prolix(xd, axis_by_name[xd.spec.name]) for xd in xplan.decisions
    ]
    planned = sorted(planned, key=lambda d: d.axis.axis_index)
    return _finalize(het_decisions + planned)
