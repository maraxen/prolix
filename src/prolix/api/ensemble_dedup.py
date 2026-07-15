"""Duplicate-topology DedupSpec planning + execute (XR-DEDUP).

``make_axis_dispatch`` rejects ``DedupGather``; execute uses xtrax
``axis_dispatch`` (dedup → map → gather), not silent vmap.

Scope (B1-XTRAX-WIRE / using-xtrax):
  - ``DedupGather`` / ``dispatch_n_mols_dedup``: **topology-keyed** bodies only
    (energy, compile probes, identical-input duplicates).
  - Seeded Langevin MD: **never** DedupGather — distinct seeds ⇒ Vmap/SafeMap
    within a ``shape_spec`` class after host ``partition_bundles_by_shape``.
  - ``partition_bundles_by_shape`` is the host Bucket-analogue for variable
    topology classes (K≪N Python groups); it is not device DedupGather.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.dispatch import axis_dispatch

from prolix.tiling.axes import N_MOLS
from prolix.tiling.planner import AxisDecision, BatchPlan, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax

__all__ = [
    "build_dedup_spec_by_shape",
    "dispatch_n_mols_dedup",
    "partition_bundles_by_shape",
    "plan_n_mols_with_dedup",
]


def partition_bundles_by_shape(bundles: list[Any]) -> list[list[int]]:
    """Stable host partition of bundle indices by ``shape_spec`` equality.

    Use this to form homogeneous N_MOLS groups before Vmap/SafeMap. Distinct
    shape classes cannot share one device map; Python over K classes is
    intentional (K≪N). This is **not** DedupGather execute.
    """
    groups: dict[Any, list[int]] = {}
    order: list[Any] = []
    for i, b in enumerate(bundles):
        key = b.shape_spec
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    return [groups[k] for k in order]


def build_dedup_spec_by_shape(bundles: list[Any]) -> DedupSpec:
    """Host DedupSpec: first index per distinct ``shape_spec`` is unique.

    For topology-keyed bodies only. Callers that need distinct Langevin seeds
    must vmap over replicas — do **not** ``dispatch_n_mols_dedup`` the MD step
    (that scatters one trajectory onto all slots).
    """
    if not bundles:
        raise ValueError("bundles must be non-empty")
    unique_indices: list[int] = []
    index_map: list[int] = []
    key_to_u: dict[Any, int] = {}
    for i, b in enumerate(bundles):
        key = b.shape_spec
        if key not in key_to_u:
            key_to_u[key] = len(unique_indices)
            unique_indices.append(i)
        index_map.append(key_to_u[key])
    return DedupSpec(
        axis_name=N_MOLS.name,
        unique_indices=np.asarray(unique_indices, dtype=np.int32),
        index_map=np.asarray(index_map, dtype=np.int32),
        k=len(unique_indices),
    )


def plan_n_mols_with_dedup(
    n_mols: int,
    dedup_spec: DedupSpec,
    *,
    budget_bytes: float | int = 1 << 40,
) -> BatchPlan:
    """Plan ``n_mols`` with a DedupSpec so xtrax pre-demotes to DedupGather.

    The axis is forced homogeneous: duplicate topologies share shapes.
    Use only with topology-keyed execute via ``dispatch_n_mols_dedup``.
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

    **Not for seeded MD.** ``fn`` must be topology-/input-identical across
    duplicate slots; otherwise scatter silently copies the wrong result.
    """
    strategy = dedup_spec.to_dedup_gather()
    return axis_dispatch(strategy, fn, xs)
