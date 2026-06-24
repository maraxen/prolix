"""Tests for stacked-bundle N_MOLS dispatch (#2645)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from prolix.api import EnsemblePlan
from prolix.api.bundle_stack import can_stack_molecular_bundles, stack_molecular_bundles
from prolix.api.ensemble_dispatch import n_mols_strategy
from xtrax.tiling.strategy import SafeMap, Vmap

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
        "prolix.api.ensemble_planner.can_jit_vmap_n_mols",
        lambda _: False,
    )
    plan = EnsemblePlan.from_bundles(bundles).batch_plan
    assert plan is not None
    strategy = n_mols_strategy(plan, len(bundles))
    assert isinstance(strategy, SafeMap)
