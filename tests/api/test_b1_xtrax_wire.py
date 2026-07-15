"""B1-XTRAX-WIRE: same-bucket vmap, host partition, no Dedup of seeded MD."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.api.bundle_md import masses_with_prefix
from prolix.api.bundle_stack import (
    can_jit_vmap_n_mols,
    integration_prefix_for_bundles,
)
from prolix.api.ensemble_dedup import partition_bundles_by_shape
from prolix.api.ensemble_plan import EnsemblePlan
from prolix.types.bundles import ATOM_BUCKETS

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle

_ep_path = Path(__file__).resolve().parent / "test_ensemble_plan.py"
_espec = importlib.util.spec_from_file_location("_ep_helper", _ep_path)
assert _espec and _espec.loader
_ep = importlib.util.module_from_spec(_espec)
_espec.loader.exec_module(_ep)
_make_minimal_bundle = _ep._make_minimal_bundle


def test_can_jit_vmap_same_bucket_different_n_atoms():
    """Claim-1 substrate: equal shape_spec, unequal n_atoms ⇒ stackable."""
    b_small = _make_bundle(5, seed=1)
    b_large = _make_bundle(20, seed=2)
    assert b_small.shape_spec == b_large.shape_spec
    assert int(b_small.n_atoms) != int(b_large.n_atoms)
    assert can_jit_vmap_n_mols([b_small, b_large])
    prefix = integration_prefix_for_bundles([b_small, b_large])
    assert prefix == ATOM_BUCKETS[b_small.shape_spec.atom_bucket_idx]
    assert prefix >= int(b_large.n_atoms)


def test_masses_with_prefix_not_unit():
    bundle = _make_minimal_bundle(n_atoms=3)
    masses = jnp.asarray(bundle.masses).at[:3].set(
        jnp.array([12.0, 1.0, 16.0], dtype=bundle.masses.dtype)
    )
    bundle = eqx.tree_at(lambda b: b.masses, bundle, masses)
    prefix = integration_prefix_for_bundles([bundle])
    got = masses_with_prefix(bundle, prefix)
    assert float(got[0]) == 12.0
    assert float(got[1]) == 1.0
    assert not bool(jnp.allclose(got[:3], jnp.ones(3)))


def test_partition_by_shape_counts_classes():
    a = _make_bundle(5, seed=1)
    b = _make_bundle(5, seed=2)
    c = _make_bundle(20, seed=3)
    assert a.shape_spec == c.shape_spec
    groups = partition_bundles_by_shape([a, b, c])
    assert len(groups) == 1
    assert groups[0] == [0, 1, 2]


def test_grouped_run_uses_stacked_not_per_system(monkeypatch):
    """Within one shape class, EnsemblePlan must call stacked dispatch once."""
    bundles = [_make_bundle(8, seed=10 + i) for i in range(4)]
    plan = EnsemblePlan.from_bundles(bundles)
    calls = {"stacked": 0, "single": 0}

    orig_stacked = plan._run_stacked_dispatch
    orig_single = plan._run_single

    def spy_stacked(*args, **kwargs):
        calls["stacked"] += 1
        return orig_stacked(*args, **kwargs)

    def spy_single(*args, **kwargs):
        calls["single"] += 1
        return orig_single(*args, **kwargs)

    monkeypatch.setattr(plan, "_run_stacked_dispatch", spy_stacked)
    monkeypatch.setattr(plan, "_run_single", spy_single)

    trajs = plan.run(
        n_steps=2,
        dt=0.5,
        kT=0.596,
        seed=0,
        gamma=10.0,
        run_mode="trajectory",
    )
    assert len(trajs) == 4
    assert calls["stacked"] == 1
    # `_run_single` may still be entered from inside stacked `run_one` (vmap);
    # the ablation we forbid is a Python for-loop of singles instead of stack.


def test_distinct_seeds_within_class_diverge():
    """Guard: seeded MD must not DedupGather-scatter one traj onto replicas."""
    bundles = [_make_minimal_bundle(n_atoms=3) for _ in range(3)]
    trajs = EnsemblePlan.from_bundles(bundles).run(
        n_steps=8,
        dt=0.5,
        kT=0.596,
        seed=0,
        gamma=10.0,
        run_mode="inference",
    )
    finals = [np.asarray(t.positions[-1]) for t in trajs]
    diffs = [
        not np.allclose(finals[i], finals[j], atol=1e-5)
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    assert any(diffs), "replicas collapsed to identical frames (accidental Dedup?)"


def test_same_bucket_stacked_inference_finite():
    bundles = [
        _make_bundle(5, seed=1),
        _make_bundle(20, seed=2),
    ]
    trajs = EnsemblePlan.from_bundles(bundles).run(
        n_steps=3,
        dt=0.5,
        kT=0.596,
        seed=1,
        gamma=10.0,
        run_mode="inference",
    )
    assert len(trajs) == 2
    assert trajs[0].positions.shape[0] == 1
    assert trajs[0].positions.shape[1] == 5
    assert trajs[1].positions.shape[1] == 20
    for t in trajs:
        assert bool(np.all(np.isfinite(t.positions)))
