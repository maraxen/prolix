"""XR-DISPATCH-MULTI: N_ATOMS + generic dispatch_vmap_safemap."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from prolix.api.ensemble_dispatch import (
    DispatchRejected,
    dispatch_n_atoms,
    dispatch_vmap_safemap,
    n_atoms_strategy,
)
from prolix.tiling.axes import N_ATOMS
from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import Bucket, SafeMap, Scan, Vmap


def test_n_atoms_strategy_vmap_default():
    assert isinstance(n_atoms_strategy(None, 8), Vmap)


def test_n_atoms_strategy_safemap_when_chunked():
    axis = AxisSpec(
        name=N_ATOMS.name,
        axis_index=0,
        cardinality=16,
        default_batch_size=4,
        tile_granularity=1,
        heterogeneous=False,
        doc="test",
    )
    plan = BatchPlan(
        decisions=[AxisDecision(axis=axis, batch_size=4, reasoning="safemap")],
        total_memory_estimate=1.0,
        axes_by_index={0: axis},
        budget_exceeded=False,
    )
    assert isinstance(n_atoms_strategy(plan, 16), SafeMap)


def test_dispatch_n_atoms_applies_make_axis_dispatch_iterator(monkeypatch):
    calls: list = []

    class FakeIterator:
        def __call__(self, fn, xs, *, in_axes=0):
            calls.append({"fn": fn, "xs": xs, "in_axes": in_axes})
            return fn(xs) if False else jnp.sum(xs, axis=-1)

    def fake_make(strategy, *, axis="", heterogeneous_axes=None):
        calls.append({"strategy": strategy, "axis": axis})
        return FakeIterator()

    monkeypatch.setattr(
        "prolix.api.ensemble_dispatch.make_axis_dispatch",
        fake_make,
    )
    xs = jnp.arange(6.0).reshape(3, 2)
    out = dispatch_n_atoms(None, 3, lambda row: jnp.sum(row), xs)
    assert out.shape == (3,)
    assert any(c.get("axis") == N_ATOMS.name for c in calls)
    assert any("fn" in c for c in calls)


def test_dispatch_n_atoms_vmap_parity():
    xs = jnp.arange(8.0).reshape(4, 2)
    out = dispatch_n_atoms(None, 4, lambda row: jnp.sum(row), xs)
    expected = jnp.sum(xs, axis=-1)
    assert jnp.allclose(out, expected)


def test_dispatch_vmap_safemap_rejects_bucket():
    with pytest.raises(DispatchRejected):
        dispatch_vmap_safemap(
            N_ATOMS.name,
            Bucket(boundaries=(8, 16)),
            lambda x: x,
            jnp.zeros((2, 1)),
        )


def test_dispatch_n_atoms_rejects_scan(monkeypatch):
    monkeypatch.setattr(
        "prolix.api.ensemble_dispatch.n_atoms_strategy",
        lambda *_a, **_k: Scan(),
    )
    with pytest.raises(DispatchRejected):
        dispatch_n_atoms(None, 2, lambda x: x, jnp.zeros((2, 1)))
