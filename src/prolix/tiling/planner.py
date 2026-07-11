"""Batch tiling planner: thin facade over xtrax joint MemoryBudget.

Domain types (``AxisSpec``, ``AxisDecision``, ``BatchPlan``) and the
``BatchPlanner`` constructor stay in prolix for a stable public API.
Strategy selection is exclusively via ``plan_with_xtrax`` →
``prolix.tiling.xtrax_adapter.plan_axes_with_xtrax`` (XR-KILL-FORK).

See ``prolix.tiling.xtrax_adapter`` and
``.praxia/docs/specs/260709_xr-kill-fork.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class BatchPlanner:
    """Host-side planner facade: decides vmap vs safe_map tile size per axis.

    Authority is xtrax joint ``MemoryBudget`` via ``plan_with_xtrax``.
    ``plan()`` is a thin alias for call-site stability (XR-KILL-FORK).
    Optional ``carry_specs`` pre-demote named axes to Scan (XR-CARRY).
    Optional ``dedup_specs`` pre-demote named axes to DedupGather (XR-DEDUP).

    estimate_memory is injected so the theoretical estimator can be swapped
    for an HLO-backed empirical model without changing this class.
    """
    axes: list[AxisSpec]
    budget_bytes: float
    estimate_memory: Callable[..., float]
    carry_specs: list | None = None
    dedup_specs: list | None = None

    def plan(self) -> BatchPlan:
        """Generate a BatchPlan via xtrax joint MemoryBudget (alias of plan_with_xtrax)."""
        return self.plan_with_xtrax()

    def plan_with_xtrax(self) -> BatchPlan:
        """Plan using xtrax joint MemoryBudget via the adapter (XR-BUDGET / XR-KILL-FORK)."""
        from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

        return plan_axes_with_xtrax(
            axes=self.axes,
            budget_bytes=self.budget_bytes,
            estimate_memory=lambda decisions: self.estimate_memory(decisions),
            carry_specs=self.carry_specs,
            dedup_specs=self.dedup_specs,
        )
