"""N_WATERS xtrax dispatch for constrained SETTLE / rigid OU."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from prolix.api.ensemble_dispatch import (
    DispatchRejected,
    dispatch_n_waters,
    n_waters_strategy,
)
from prolix.tiling.axes import N_WATERS
from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan
from xtrax.tiling.strategy import SafeMap, Scan, Vmap


def test_n_waters_strategy_vmap_default():
    assert isinstance(n_waters_strategy(None, 8), Vmap)


def _n_waters_plan(*, n: int, batch_size: int) -> BatchPlan:
    axis = AxisSpec(
        name=N_WATERS.name,
        axis_index=7,
        cardinality=n,
        default_batch_size=batch_size,
        tile_granularity=1,
        heterogeneous=False,
        doc="test",
    )
    return BatchPlan(
        decisions=[AxisDecision(axis=axis, batch_size=batch_size, reasoning="safemap")],
        total_memory_estimate=1.0,
        axes_by_index={7: axis},
        budget_exceeded=False,
    )


def test_n_waters_strategy_safemap_when_chunked():
    plan = _n_waters_plan(n=64, batch_size=8)
    assert isinstance(n_waters_strategy(plan, 64), SafeMap)


def test_dispatch_n_waters_vmap_parity():
    keys = jnp.arange(4.0)
    r = jnp.arange(12.0).reshape(4, 3)
    stacked = (keys, r)

    def body(pack):
        k, row = pack
        return k + jnp.sum(row)

    out = dispatch_n_waters(None, 4, body, stacked)
    expected = keys + jnp.sum(r, axis=-1)
    assert jnp.allclose(out, expected)


def test_dispatch_n_waters_rejects_scan(monkeypatch):
    monkeypatch.setattr(
        "prolix.tiling.axis_dispatch.n_waters_strategy",
        lambda *_a, **_k: Scan(),
    )
    with pytest.raises(DispatchRejected):
        dispatch_n_waters(None, 2, lambda x: x, jnp.zeros((2, 1)))


def test_dispatch_n_waters_safemap_matches_vmap_when_divisible():
    n, bs = 16, 8
    keys = jnp.arange(n, dtype=jnp.float32)
    r = jnp.arange(n * 3, dtype=jnp.float32).reshape(n, 3)
    stacked = (keys, r)

    def body(pack):
        k, row = pack
        return k + jnp.sum(row)

    vmap_out = dispatch_n_waters(None, n, body, stacked)
    chunk_out = dispatch_n_waters(_n_waters_plan(n=n, batch_size=bs), n, body, stacked)
    assert jnp.allclose(chunk_out, vmap_out)


def test_dispatch_n_waters_safemap_rejects_remainder():
    """xtrax safe_map forbids n % batch_size != 0 (1vii ~455 waters, tile 32)."""
    n, bs = 455, 32
    xs = jnp.arange(n, dtype=jnp.float32)
    with pytest.raises(ValueError, match="n_waters=455"):
        dispatch_n_waters(_n_waters_plan(n=n, batch_size=bs), n, lambda x: x, xs)
