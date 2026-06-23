"""Adapter between prolix.tiling.planner and xtrax.tiling.plan (#1842).

Converts AxisSpec / BatchPlan between the two representations and delegates
strategy selection to xtrax.tiling.BatchPlanner. A secondary prolix greedy
demotion loop runs when the converted plan still exceeds budget_bytes.
"""

from __future__ import annotations

from collections.abc import Callable

from xtrax.tiling.plan import AxisDecision as XtraxAxisDecision
from xtrax.tiling.plan import AxisSpec as XtraxAxisSpec
from xtrax.tiling.plan import BatchPlanner as XtraxBatchPlanner
from xtrax.tiling.strategy import Bucket, SafeMap, Scan, Vmap

from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan


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
    if axis.heterogeneous:
        return max(1, decision.batch_size or axis.tile_granularity)
    return max(1, decision.batch_size)


def plan_axes_with_xtrax(
    axes: list[AxisSpec],
    budget_bytes: float,
    estimate_memory: Callable[[list[AxisDecision]], float],
) -> BatchPlan:
    """Plan using xtrax BatchPlanner, then apply prolix budget demotion if needed."""
    sorted_axes = sorted(axes, key=lambda a: a.axis_index)
    axis_by_name = {ax.name: ax for ax in sorted_axes}

    het_decisions: list[AxisDecision] = []
    hom_axes: list[AxisSpec] = []
    for ax in sorted_axes:
        if ax.heterogeneous:
            tile = max(1, ax.tile_granularity)
            het_decisions.append(
                AxisDecision(
                    axis=ax,
                    batch_size=tile,
                    reasoning="heterogeneous axis: element shapes vary; safe_map required",
                )
            )
        else:
            hom_axes.append(ax)

    if not hom_axes:
        final_estimate = estimate_memory(het_decisions)
        return BatchPlan(
            decisions=het_decisions,
            total_memory_estimate=final_estimate,
            axes_by_index={ax.axis_index: ax for ax in sorted_axes},
            budget_exceeded=final_estimate > budget_bytes,
        )

    xspecs = [prolix_axis_to_xtrax(ax) for ax in hom_axes]
    heterogeneous_axes = {ax.name for ax in sorted_axes if ax.heterogeneous}

    def memory_estimator(xspec: XtraxAxisSpec) -> int:
        prolix_axis = axis_by_name[xspec.name]
        probe = [
            AxisDecision(
                axis=prolix_axis,
                batch_size=0,
                reasoning="xtrax memory probe (vmap)",
            )
        ]
        return int(estimate_memory(het_decisions + probe))

    xplan = XtraxBatchPlanner(
        memory_estimator=memory_estimator,
        heterogeneous_axes=heterogeneous_axes,
    ).plan(xspecs)

    hom_decisions: list[AxisDecision] = []
    for xd in xplan.decisions:
        prolix_axis = axis_by_name[xd.spec.name]
        hom_decisions.append(xtrax_decision_to_prolix(xd, prolix_axis))

    hom_decisions = sorted(hom_decisions, key=lambda d: d.axis.axis_index)

    while estimate_memory(het_decisions + hom_decisions) > budget_bytes:
        demoted = False
        for i, decision in enumerate(hom_decisions):
            if decision.batch_size == 0:
                tile = max(1, decision.axis.tile_granularity)
                hom_decisions[i] = AxisDecision(
                    axis=decision.axis,
                    batch_size=tile,
                    reasoning=(
                        f"prolix budget demotion after xtrax plan (tile={tile})"
                    ),
                )
                demoted = True
                break
        if not demoted:
            break

    all_decisions = het_decisions + hom_decisions
    final_estimate = estimate_memory(all_decisions)
    return BatchPlan(
        decisions=all_decisions,
        total_memory_estimate=final_estimate,
        axes_by_index={ax.axis_index: ax for ax in sorted_axes},
        budget_exceeded=final_estimate > budget_bytes,
    )
