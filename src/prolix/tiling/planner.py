"""Batch tiling planner: thin facade over xtrax joint MemoryBudget.

Domain types (``AxisSpec``, ``AxisDecision``, ``BatchPlan``) and the
``BatchPlanner`` constructor stay in prolix for a stable public API.
Strategy selection is exclusively via ``plan_with_xtrax`` →
``prolix.tiling.xtrax_adapter.plan_axes_with_xtrax`` (XR-KILL-FORK).

See ``prolix.tiling.xtrax_adapter`` and
``.praxia/docs/specs/260709_xr-kill-fork.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from prolix.tiling.plan_types import (
    AxisDecision,
    AxisSpec,
    BatchPlan,
    ceil_to_granularity,
    estimate_memory_theoretical,
)
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "AxisDecision",
    "AxisSpec",
    "BatchPlan",
    "BatchPlanner",
    "ceil_to_granularity",
    "estimate_memory_theoretical",
]


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
        return plan_axes_with_xtrax(
            axes=self.axes,
            budget_bytes=self.budget_bytes,
            estimate_memory=lambda decisions: self.estimate_memory(decisions),
            carry_specs=self.carry_specs,
            dedup_specs=self.dedup_specs,
        )
