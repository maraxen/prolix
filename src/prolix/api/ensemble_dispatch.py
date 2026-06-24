"""N_MOLS axis dispatch via xtrax make_axis_dispatch (#2645)."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.strategy import SafeMap, Vmap
from xtrax.transforms.map import safe_map

from prolix.tiling.axes import N_MOLS
from prolix.tiling.planner import BatchPlan


def n_mols_strategy(plan: BatchPlan | None, n_systems: int) -> Vmap | SafeMap:
    """Map prolix BatchPlan N_MOLS decision to an xtrax strategy."""
    if plan is None or n_systems <= 1:
        return Vmap()
    try:
        decision = plan.decision_for(N_MOLS.name)
    except KeyError:
        return Vmap()
    batch_size = decision.batch_size
    if batch_size == 0 or batch_size >= n_systems:
        return Vmap()
    return SafeMap(batch_size=batch_size)


def dispatch_n_mols(
    plan: BatchPlan | None,
    n_systems: int,
    fn: Callable[[Any, jnp.ndarray], Any],
    stacked_bundle: Any,
    seeds: jnp.ndarray,
) -> Any:
    """Dispatch ``fn(bundle, seed)`` over stacked bundles on N_MOLS."""
    strategy = n_mols_strategy(plan, n_systems)
    # Validate strategy and select iterator kind (VmapIterator / SafeMapIterator).
    make_axis_dispatch(strategy, axis=N_MOLS.name)
    if isinstance(strategy, Vmap):
        return jax.vmap(fn, in_axes=(0, 0))(stacked_bundle, seeds)
    return safe_map(fn, (stacked_bundle, seeds), batch_size=strategy.batch_size)
