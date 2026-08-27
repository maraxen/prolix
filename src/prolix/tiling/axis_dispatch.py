"""JIT-safe xtrax Vmap/SafeMap on N_WATERS.

Imported from physics (settle) at module scope — not inside a jitted body.
Does not import ``prolix.api`` (physics must not depend on the API layer).
"""

from __future__ import annotations

from typing import Any, Callable

from xtrax.tiling.dispatch import DispatchRejected, make_axis_dispatch
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap

from prolix.tiling.axes import N_WATERS
from prolix.tiling.planner import BatchPlan

__all__ = [
    "dispatch_n_waters",
    "n_waters_strategy",
]


def _vmap_or_safemap(
    plan: BatchPlan | None,
    axis_name: str,
    n: int,
) -> Vmap | SafeMap:
    if plan is None or n <= 1:
        return Vmap()
    try:
        decision = plan.decision_for(axis_name)
    except KeyError:
        return Vmap()
    batch_size = decision.batch_size
    if batch_size == 0 or batch_size >= n:
        return Vmap()
    return SafeMap(batch_size=batch_size)


def n_waters_strategy(plan: BatchPlan | None, n_waters: int) -> Vmap | SafeMap:
    return _vmap_or_safemap(plan, N_WATERS.name, n_waters)


def dispatch_n_waters(
    plan: BatchPlan | None,
    n_waters: int,
    fn: Callable[[Any], Any],
    stacked_waters: Any,
) -> Any:
    """Dispatch ``fn(water_slice)`` over the leading N_WATERS axis."""
    strategy = n_waters_strategy(plan, n_waters)
    if isinstance(strategy, (Bucket, DedupGather, Scan)):
        raise DispatchRejected(
            f"{N_WATERS.name} dispatch rejects {type(strategy).__name__}; "
            "only Vmap/SafeMap are supported."
        )
    if not isinstance(strategy, (Vmap, SafeMap)):
        raise TypeError(
            f"{N_WATERS.name} dispatch unsupported strategy type: {type(strategy)!r}"
        )
    iterator = make_axis_dispatch(strategy, axis=N_WATERS.name)
    if isinstance(strategy, SafeMap) and n_waters % strategy.batch_size != 0:
        raise ValueError(
            f"N_WATERS SafeMap requires n_waters % batch_size == 0, "
            f"got n_waters={n_waters} batch_size={strategy.batch_size}. "
            "Pad water_indices to a multiple of the tile or keep plan=None (Vmap)."
        )
    return iterator(fn, stacked_waters, in_axes=0)
