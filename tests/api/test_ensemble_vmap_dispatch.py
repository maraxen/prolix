"""Tests for stacked-bundle N_MOLS dispatch (#2645 / XR-DISPATCH)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import jax.numpy as jnp
import pytest

from prolix.api import EnsemblePlan
from prolix.api.bundle_stack import can_stack_molecular_bundles, stack_molecular_bundles
from prolix.api.ensemble_dispatch import DispatchRejected, dispatch_n_mols, n_mols_strategy
from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import Bucket, DedupGather, SafeMap, Scan, Vmap

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle


def test_can_stack_hetero_sizes_same_bucket():
    bundles = [_make_bundle(n, seed=100 + i) for i, n in enumerate((5, 10, 20, 35))]
    assert can_stack_molecular_bundles(bundles)
    stacked = stack_molecular_bundles(bundles)
    assert stacked.positions.shape[0] == 4


def test_n_mols_strategy_vmap_when_stackable():
    bundles = [_make_bundle(10, seed=1), _make_bundle(10, seed=2)]
    plan = EnsemblePlan.from_bundles(bundles).batch_plan
    assert plan is not None
    strategy = n_mols_strategy(plan, len(bundles))
    assert isinstance(strategy, Vmap)


def test_n_mols_strategy_safemap_when_not_stackable(monkeypatch):
    bundles = [_make_bundle(10, seed=1), _make_bundle(10, seed=2)]
    monkeypatch.setattr(
        "prolix.api.ensemble_planner.can_stack_molecular_bundles",
        lambda _: False,
    )
    plan = EnsemblePlan.from_bundles(bundles).batch_plan
    assert plan is not None
    strategy = n_mols_strategy(plan, len(bundles))
    assert isinstance(strategy, SafeMap)


def test_dispatch_n_mols_applies_make_axis_dispatch_iterator(monkeypatch):
    """Kill test: iterator from make_axis_dispatch is applied, not discarded."""
    calls: list = []

    class FakeIterator:
        def __call__(self, fn, xs, *, in_axes=0):
            calls.append({"fn": fn, "xs": xs, "in_axes": in_axes})
            bundle, seeds = xs
            n = int(seeds.shape[0])
            return jnp.arange(n)

    def fake_make(strategy, *, axis="", heterogeneous_axes=None):
        calls.append({"strategy": strategy, "axis": axis})
        return FakeIterator()

    monkeypatch.setattr(
        "prolix.api.ensemble_dispatch.make_axis_dispatch",
        fake_make,
    )

    axis = AxisSpec(
        name="n_mols",
        axis_index=5,
        cardinality=2,
        default_batch_size=0,
        tile_granularity=1,
        heterogeneous=False,
        doc="test",
    )
    plan = BatchPlan(
        decisions=[AxisDecision(axis=axis, batch_size=0, reasoning="vmap")],
        total_memory_estimate=1.0,
        axes_by_index={5: axis},
        budget_exceeded=False,
    )
    stacked = jnp.zeros((2, 3))
    seeds = jnp.arange(2, dtype=jnp.int32)
    out = dispatch_n_mols(plan, 2, lambda b, s: b, stacked, seeds)
    assert list(out) == [0, 1]
    assert any("strategy" in c for c in calls)
    assert any("fn" in c for c in calls), "iterator must be invoked"
    assert calls[-1]["in_axes"] == 0


def test_dispatch_n_mols_rejects_bucket():
    with pytest.raises((DispatchRejected, TypeError)):
        # Force unsupported strategy past n_mols_strategy by monkeypatching.
        import prolix.api.ensemble_dispatch as mod

        real = mod.n_mols_strategy

        def _bucket(*_a, **_k):
            return Bucket(boundaries=(8, 16))

        mod.n_mols_strategy = _bucket  # type: ignore[assignment]
        try:
            dispatch_n_mols(None, 2, lambda b, s: b, jnp.zeros((2, 1)), jnp.arange(2))
        finally:
            mod.n_mols_strategy = real


def test_dispatch_n_mols_rejects_scan(monkeypatch):
    monkeypatch.setattr(
        "prolix.api.ensemble_dispatch.n_mols_strategy",
        lambda *_a, **_k: Scan(),
    )
    with pytest.raises(DispatchRejected):
        dispatch_n_mols(None, 2, lambda b, s: b, jnp.zeros((2, 1)), jnp.arange(2))


def test_dispatch_n_mols_rejects_dedup(monkeypatch):
    from xtrax.tiling.strategy import _default_dedup_fn, _default_gather_fn

    monkeypatch.setattr(
        "prolix.api.ensemble_dispatch.n_mols_strategy",
        lambda *_a, **_k: DedupGather(
            unique_indices=jnp.array([0]),
            index_map=jnp.array([0, 0]),
            k=1,
            k_bucket=1,
            dedup_fn=_default_dedup_fn,
            gather_fn=_default_gather_fn,
        ),
    )
    with pytest.raises(DispatchRejected):
        dispatch_n_mols(None, 2, lambda b, s: b, jnp.zeros((2, 1)), jnp.arange(2))
