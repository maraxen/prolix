"""XR-DEDUP: DedupSpec → DedupGather planning + host execute contracts."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.tiling.dedup import DedupSpec, get_k_bucket

from prolix.api.ensemble_dedup import (
    dispatch_n_mols_dedup,
    plan_n_mols_with_dedup,
)
from prolix.tiling.axes import N_MOLS
from prolix.tiling.planner import AxisDecision, estimate_memory_theoretical
from prolix.tiling.xtrax_adapter import plan_axes_with_xtrax


def _dup_topology_spec(n: int = 4, k: int = 2) -> DedupSpec:
    """N=4 batch with two unique topologies: slots [0,1] unique, map [0,1,0,1]."""
    assert n == 4 and k == 2
    return DedupSpec(
        axis_name=N_MOLS.name,
        unique_indices=np.array([0, 1], dtype=np.int32),
        index_map=np.array([0, 1, 0, 1], dtype=np.int32),
        k=k,
    )


def test_plan_n_mols_with_dedup_is_dedup_gather():
    ds = _dup_topology_spec()
    plan = plan_n_mols_with_dedup(n_mols=4, dedup_spec=ds)
    d = plan.decision_for(N_MOLS.name)
    assert "dedup-gather" in d.reasoning
    assert d.batch_size == get_k_bucket(ds.k)


def test_dispatch_n_mols_dedup_scatters_bit_exact_vs_full_map():
    ds = _dup_topology_spec()
    # Values: positions 0,1 unique; 2,3 duplicates of 0,1
    xs = jnp.array([10.0, 20.0, 10.0, 20.0])

    def body(x):
        return x * 3.0

    got = dispatch_n_mols_dedup(ds, body, xs)
    ref = jax_vmap_ref(body, xs)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # Unique slots only: body(10), body(20) → scatter
    np.testing.assert_array_equal(np.asarray(got), np.array([30.0, 60.0, 30.0, 60.0]))


def jax_vmap_ref(fn, xs):
    import jax

    return jax.vmap(fn)(xs)


def test_dedup_spec_k_bucket_padding_in_gather():
    ds = _dup_topology_spec(k=2)
    dg = ds.to_dedup_gather()
    assert dg.k == 2
    assert dg.k_bucket == get_k_bucket(2)
    assert len(dg.unique_indices) == dg.k_bucket


def test_plan_axes_with_xtrax_forwards_dedup_specs():
    axis = dataclasses.replace(N_MOLS, cardinality=4, heterogeneous=False)
    ds = DedupSpec(
        axis_name=N_MOLS.name,
        unique_indices=np.array([0, 2], dtype=np.int32),
        index_map=np.array([0, 0, 1, 1], dtype=np.int32),
        k=2,
    )

    def estimate(decisions: list[AxisDecision]) -> float:
        return estimate_memory_theoretical(decisions, 8.0, 1.0)

    plan = plan_axes_with_xtrax(
        axes=[axis],
        budget_bytes=1e18,
        estimate_memory=estimate,
        dedup_specs=[ds],
    )
    assert "dedup-gather" in plan.decision_for(N_MOLS.name).reasoning
