"""Batch planner for EnsemblePlan multi-bundle MD dispatch (#1842 / XR-BUDGET)."""

from __future__ import annotations

import dataclasses
from typing import Any

from xtrax.tiling.estimators import device_memory_budget

from prolix.tiling.axes import N_ATOMS, N_MOLS
from prolix.tiling.planner import BatchPlan, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import BudgetInfeasibleError, plan_axes_with_xtrax

from prolix.api.bundle_stack import can_jit_vmap_n_mols

# CI / CPU backends without memory_stats: match historical _device_budget_bytes fallback.
_CI_DEVICE_LIMIT_BYTES = 4 << 30

__all__ = [
    "BudgetInfeasibleError",
    "EnsembleMDPlanner",
    "resolve_device_budget_bytes",
]


def resolve_device_budget_bytes(headroom: float, param_bytes: float = 0.0) -> int:
    """Resolve an int MemoryBudget byte count (1A estimator policy).

    GPU/prod: ``device_memory_budget(fraction=headroom)`` when the device
    reports ``bytes_limit``. CI CPU: fixed ``4<<30`` device limit × headroom.
    ``param_bytes`` are subtracted in both cases (floored at 1).
    """
    try:
        device_budget = device_memory_budget(fraction=headroom)
    except (RuntimeError, ValueError, AttributeError, IndexError, TypeError, KeyError):
        device_budget = int(_CI_DEVICE_LIMIT_BYTES * headroom)
    return max(1, int(device_budget - param_bytes))


@dataclasses.dataclass(frozen=True)
class EnsembleMDPlanner:
    """xtrax-backed planner for EnsemblePlan.run() multi-bundle dispatch.

    Builds a BatchPlan over N_MOLS (and N_ATOMS cardinality) from the bundle
    list passed to plan(bundles).
    """

    headroom: float = 0.80
    param_bytes: float = 0.0
    base_shape_bytes: float = 1024.0
    activation_multiplier: float = 2.5

    def plan(self, bundles: list[Any]) -> BatchPlan:
        """Return a BatchPlan for the given bundle batch."""
        n_systems = max(1, len(bundles))
        max_atoms = max((int(b.n_atoms) for b in bundles), default=1)
        homo_n_atoms = len(bundles) > 1 and can_jit_vmap_n_mols(bundles)

        axes = [
            dataclasses.replace(N_ATOMS, cardinality=max_atoms),
            dataclasses.replace(
                N_MOLS,
                cardinality=n_systems,
                heterogeneous=not homo_n_atoms,
            ),
        ]
        budget = resolve_device_budget_bytes(self.headroom, self.param_bytes)

        def estimate_memory(decisions: list) -> float:
            return estimate_memory_theoretical(
                decisions,
                self.base_shape_bytes,
                self.activation_multiplier,
            )

        return plan_axes_with_xtrax(
            axes=axes,
            budget_bytes=budget,
            estimate_memory=estimate_memory,
        )
