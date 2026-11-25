"""Batch-plan domain types and the theoretical memory estimator (leaf module).

``AxisSpec``, ``AxisDecision``, ``BatchPlan``, ``ceil_to_granularity`` and
``estimate_memory_theoretical`` live here, with no dependency on
``prolix.tiling.xtrax_adapter``, so the adapter and ``BatchPlanner`` no longer form a
cycle that needed a lazy import in ``BatchPlanner.plan_with_xtrax`` (Phase 1.2).
``prolix.tiling.planner`` re-exports every name for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

__all__ = [
    "AxisDecision",
    "AxisSpec",
    "BatchPlan",
    "ceil_to_granularity",
    "estimate_memory_theoretical",
]

# Keep the historical logger name so existing log filters / caplog tests still match.
logger = logging.getLogger("prolix.tiling.planner")

# TODO: introduce bucketing management in here

def ceil_to_granularity(n: int, g: int) -> int:
    """Round n up to the nearest multiple of g. If g <= 1 or n == 0, returns n unchanged."""
    if g <= 1 or n == 0:
        return n
    return ((n + g - 1) // g) * g


def estimate_memory_theoretical(
    decisions: list[AxisDecision],
    base_shape_bytes: float,
    activation_multiplier: float,
) -> float:
    """Estimate peak memory for a set of axis decisions.

    Axes with batch_size=0 (vmap) contribute their full cardinality to the
    memory product. Axes with batch_size>0 (safe_map) contribute only their
    tile size — one tile is live at a time.

    activation_multiplier must be supplied by the caller (no default).
    """
    product = 1
    for d in decisions:
        if d.batch_size == 0:
            product *= d.axis.cardinality
        else:
            product *= d.batch_size
    return base_shape_bytes * product * activation_multiplier


@dataclass(frozen=True)
class AxisSpec:
    """Describes one mappable axis for the batch planner."""
    name: str
    axis_index: int        # lower = innermost; demotion order follows ascending index
    cardinality: int       # typical/max size of this axis
    default_batch_size: int  # 0 = vmap; positive = safe_map tile size
    tile_granularity: int  # safe_map tile sizes are rounded up to multiples of this
    heterogeneous: bool    # if True, element shapes may vary; vmap invalid; always safe_map
    doc: str


@dataclass(frozen=True)
class AxisDecision:
    """Planner output for one axis."""
    axis: AxisSpec
    batch_size: int        # 0 = vmap; positive = safe_map tile size
    reasoning: str


@dataclass(frozen=True)
class BatchPlan:
    """Planner output for a full set of axes."""
    decisions: list[AxisDecision]
    total_memory_estimate: float
    axes_by_index: dict[int, AxisSpec]
    budget_exceeded: bool

    def exceeded_budget(self) -> bool:
        """True when even minimum-tile decisions exceed the budget. Never raises."""
        return self.budget_exceeded

    def decision_for(self, name: str) -> AxisDecision:
        for d in self.decisions:
            if d.axis.name == name:
                return d
        raise KeyError(name)

    def log_summary(self) -> None:
        parts = [f"{d.axis.name}=bs:{d.batch_size}" for d in self.decisions]
        status = "EXCEEDED" if self.budget_exceeded else "ok"
        logger.debug(
            "BatchPlan [%s]: %s | estimate=%.1f bytes",
            status, ", ".join(parts), self.total_memory_estimate,
        )

