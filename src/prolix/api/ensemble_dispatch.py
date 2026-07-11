"""Axis dispatch via xtrax make_axis_dispatch (#2645 / XR-DISPATCH / XR-DISPATCH-MULTI / XR-CARRY).

N_MOLS (MVP 2A): ``Vmap`` / ``SafeMap`` only via ``dispatch_n_mols``.
N_ATOMS (XR-DISPATCH-MULTI): same Vmap/SafeMap policy via ``dispatch_n_atoms``.
``Bucket``, ``DedupGather``, and ``Scan`` on mapped axes raise — Scan belongs on
the step axis (XR-CARRY); ``DedupGather`` executes via
``prolix.api.ensemble_dedup.dispatch_n_mols_dedup``.

N_STEPS (XR-CARRY): ``dispatch_n_steps`` applies ``JaxScanIterator`` from
``make_axis_dispatch(Scan(), ...)``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from xtrax.tiling.dispatch import DispatchRejected, make_axis_dispatch
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap

from prolix.tiling.axes import N_ATOMS, N_MOLS, N_STEPS
from prolix.tiling.planner import BatchPlan

__all__ = [
    "DispatchRejected",
    "dispatch_n_atoms",
    "dispatch_n_mols",
    "dispatch_n_steps",
    "dispatch_vmap_safemap",
    "n_atoms_strategy",
    "n_mols_strategy",
]


def _vmap_or_safemap(
    plan: BatchPlan | None,
    axis_name: str,
    n: int,
) -> Vmap | SafeMap:
    """Map a BatchPlan decision to Vmap (full) or SafeMap (chunked)."""
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


def n_mols_strategy(plan: BatchPlan | None, n_systems: int) -> Vmap | SafeMap:
    """Map prolix BatchPlan N_MOLS decision to an xtrax strategy."""
    return _vmap_or_safemap(plan, N_MOLS.name, n_systems)


def n_atoms_strategy(plan: BatchPlan | None, n_atoms: int) -> Vmap | SafeMap:
    """Map prolix BatchPlan N_ATOMS decision to an xtrax strategy (XR-DISPATCH-MULTI)."""
    return _vmap_or_safemap(plan, N_ATOMS.name, n_atoms)


def dispatch_vmap_safemap(
    axis_name: str,
    strategy: Vmap | SafeMap,
    fn: Callable[..., Any],
    args: Any,
    *,
    in_axes: Any = 0,
) -> Any:
    """Apply ``make_axis_dispatch`` Vmap/SafeMap iterator on ``axis_name``.

    Rejects Bucket / DedupGather / Scan rather than silent-vmap.
    """
    if isinstance(strategy, (Bucket, DedupGather, Scan)):
        raise DispatchRejected(
            f"{axis_name} dispatch rejects {type(strategy).__name__}; "
            "only Vmap/SafeMap are supported "
            "(Scan → XR-CARRY; DedupGather → ensemble_dedup; Bucket → XR-BUCKET)."
        )
    if not isinstance(strategy, (Vmap, SafeMap)):
        raise TypeError(
            f"{axis_name} dispatch unsupported strategy type: {type(strategy)!r}"
        )

    iterator = make_axis_dispatch(strategy, axis=axis_name)
    return iterator(fn, args, in_axes=in_axes)


def dispatch_n_mols(
    plan: BatchPlan | None,
    n_systems: int,
    fn: Callable[[Any, jnp.ndarray], Any],
    stacked_bundle: Any,
    seeds: jnp.ndarray,
) -> Any:
    """Dispatch ``fn(bundle, seed)`` over stacked bundles on N_MOLS.

    Applies the iterator returned by ``make_axis_dispatch`` (VmapIterator /
    SafeMapIterator). Unsupported strategies raise rather than silent-vmap.
    """
    strategy = n_mols_strategy(plan, n_systems)

    def _mapped(pair: tuple[Any, jnp.ndarray]) -> Any:
        bundle, seed = pair
        return fn(bundle, seed)

    return dispatch_vmap_safemap(
        N_MOLS.name,
        strategy,
        _mapped,
        (stacked_bundle, seeds),
        in_axes=0,
    )


def dispatch_n_atoms(
    plan: BatchPlan | None,
    n_atoms: int,
    fn: Callable[[Any], Any],
    stacked_atoms: Any,
) -> Any:
    """Dispatch ``fn(atom_slice)`` over a leading atom/batch axis (XR-DISPATCH-MULTI).

    ``stacked_atoms`` is a pytree whose leaves share leading dimension ``n_atoms``.
    """
    strategy = n_atoms_strategy(plan, n_atoms)
    return dispatch_vmap_safemap(
        N_ATOMS.name,
        strategy,
        fn,
        stacked_atoms,
        in_axes=0,
    )


def dispatch_n_steps(
    step_fn: Callable[[Any, Any], tuple[Any, Any]],
    init_state: Any,
    n_steps: int,
) -> tuple[Any, Any]:
    """Dispatch a carry-bearing MD step via ``JaxScanIterator`` (XR-CARRY).

    ``step_fn(carry, x) -> (carry, y)`` matches ``jax.lax.scan``. ``xs`` is
    ``arange(n_steps)`` (values unused when the step ignores ``x``).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    iterator = make_axis_dispatch(Scan(), axis=N_STEPS.name)
    xs = jnp.arange(int(n_steps))
    return iterator(step_fn, init_state, xs)
