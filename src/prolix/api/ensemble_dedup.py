"""Duplicate-topology DedupSpec planning + execute (XR-DEDUP).

``make_axis_dispatch`` rejects ``DedupGather``; execute uses xtrax
``axis_dispatch`` (dedup → map → gather), not silent vmap.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.dispatch import axis_dispatch

from prolix.tiling.axes import N_MOLS
from prolix.tiling.planner import AxisDecision, BatchPlan, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

__all__ = [
    "dispatch_n_mols_dedup",
    "plan_n_mols_with_dedup",
]


def plan_n_mols_with_dedup(
    n_mols: int,
    dedup_spec: DedupSpec,
    *,
    budget_bytes: float | int = 1 << 40,
) -> BatchPlan:
    """Plan ``n_mols`` with a DedupSpec so xtrax pre-demotes to DedupGather.

    The axis is forced homogeneous: duplicate topologies share shapes.
    """
    if n_mols < 1:
        raise ValueError(f"n_mols must be >= 1, got {n_mols}")
    if dedup_spec.axis_name != N_MOLS.name:
        raise ValueError(
            f"DedupSpec.axis_name must be '{N_MOLS.name}', got {dedup_spec.axis_name!r}"
        )
    if len(dedup_spec.index_map) != n_mols:
        raise ValueError(
            f"DedupSpec.index_map length {len(dedup_spec.index_map)} != n_mols={n_mols}"
        )
    axis = dataclasses.replace(N_MOLS, cardinality=int(n_mols), heterogeneous=False)

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(
            decisions, base_shape_bytes=8.0, activation_multiplier=1.0
        )

    return plan_axes_with_xtrax(
        axes=[axis],
        budget_bytes=budget_bytes,
        estimate_memory=estimate,
        dedup_specs=[dedup_spec],
    )


def dispatch_n_mols_dedup(
    dedup_spec: DedupSpec,
    fn: Callable[[Any], Any],
    xs: Any,
) -> Any:
    """Execute DedupGather via xtrax ``axis_dispatch`` (not ``make_axis_dispatch``).

    Phases: select K unique → map ``fn`` → scatter to N via ``index_map``.
    """
    strategy = dedup_spec.to_dedup_gather()
    return axis_dispatch(strategy, fn, xs)
