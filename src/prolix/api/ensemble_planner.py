"""Batch planner for EnsemblePlan multi-bundle MD dispatch (#1842)."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax

from prolix.tiling.axes import N_ATOMS, N_MOLS
from prolix.tiling.planner import BatchPlan, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

from prolix.api.bundle_stack import can_jit_vmap_n_mols, can_stack_molecular_bundles


def _device_budget_bytes(headroom: float, param_bytes: float) -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            raise AttributeError("memory_stats returned None")
        device_limit = stats["bytes_limit"]
    except (KeyError, IndexError, AttributeError, TypeError):
        device_limit = 4 * 1024**3
    return device_limit * headroom - param_bytes


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
        stackable = len(bundles) > 1 and can_stack_molecular_bundles(bundles)
        homo_n_atoms = (
            len(bundles) > 1
            and can_jit_vmap_n_mols(bundles)
        )

        axes = [
            dataclasses.replace(N_ATOMS, cardinality=max_atoms),
            dataclasses.replace(
                N_MOLS,
                cardinality=n_systems,
                heterogeneous=not homo_n_atoms,
            ),
        ]
        budget = _device_budget_bytes(self.headroom, self.param_bytes)

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
