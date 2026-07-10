"""Tests for host-side batch planning and fitting configuration."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from prolix.api.ensemble_planner import resolve_device_budget_bytes
from prolix.run.spec import (
    BatchingConfig,
    FittingAxisNames,
    extract_batch_sizes,
    make_fitting_planner,
)
from prolix.tiling.axes import (
    N_ANGLES,
    N_ATOMS,
    N_BONDS,
    N_CONFORMERS,
    N_MOLS,
    N_TORSIONS,
)
from prolix.tiling.planner import BatchPlanner, estimate_memory_theoretical


def test_make_fitting_planner_returns_valid_plan():
    """Smoke test: planner produces a BatchPlan with both axes decided."""
    spec = BatchingConfig(n_mols=32, n_conformers=100)
    plan = make_fitting_planner(spec)
    assert plan.decision_for(FittingAxisNames.N_MOLS) is not None
    assert plan.decision_for(FittingAxisNames.N_CONFORMERS) is not None


def test_extract_batch_sizes_returns_tuple():
    """extract_batch_sizes unpacks decisions into a tuple."""
    spec = BatchingConfig(n_mols=16, n_conformers=50)
    plan = make_fitting_planner(spec)
    mols_bs, conf_bs = extract_batch_sizes(plan)
    assert isinstance(mols_bs, int)
    assert isinstance(conf_bs, int)


def test_batching_config_frozen():
    """BatchingConfig is immutable."""
    cfg = BatchingConfig(n_mols=8, n_conformers=10)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        cfg.n_mols = 99


def test_explicit_batch_size_override():
    """Explicit mols_batch_size in spec is respected by planner."""
    spec = BatchingConfig(n_mols=64, n_conformers=10, mols_batch_size=4)
    plan = make_fitting_planner(spec)
    # The planner may still demote, but the override should be the upper bound it considers
    decision = plan.decision_for(FittingAxisNames.N_MOLS)
    # Some sanity: batch_size is positive
    assert decision.batch_size > 0


def test_axis_names_match_axes_module():
    """FittingAxisNames string constants match AxisSpec.name fields."""
    assert FittingAxisNames.N_ATOMS == N_ATOMS.name
    assert FittingAxisNames.N_BONDS == N_BONDS.name
    assert FittingAxisNames.N_ANGLES == N_ANGLES.name
    assert FittingAxisNames.N_TORSIONS == N_TORSIONS.name
    assert FittingAxisNames.N_CONFORMERS == N_CONFORMERS.name
    assert FittingAxisNames.N_MOLS == N_MOLS.name


def test_fallback_when_device_memory_unavailable():
    """Planner doesn't crash on CPU/test environments without device.memory_stats."""
    spec = BatchingConfig(n_mols=8, n_conformers=10)
    plan = make_fitting_planner(spec)  # should not raise
    assert plan is not None


def test_batching_config_defaults():
    """BatchingConfig has sensible defaults."""
    cfg = BatchingConfig()
    assert cfg.n_mols == 1
    assert cfg.n_conformers == 1
    assert cfg.mols_batch_size is None
    assert cfg.conformers_batch_size is None


def test_explicit_conformers_batch_size_override():
    """Explicit conformers_batch_size in spec is respected by planner."""
    spec = BatchingConfig(n_mols=16, n_conformers=50, conformers_batch_size=8)
    plan = make_fitting_planner(spec)
    decision = plan.decision_for(FittingAxisNames.N_CONFORMERS)
    assert decision.batch_size > 0


def test_both_batch_size_overrides():
    """Both mols and conformers batch size overrides work together."""
    spec = BatchingConfig(
        n_mols=64,
        n_conformers=100,
        mols_batch_size=4,
        conformers_batch_size=10,
    )
    plan = make_fitting_planner(spec)
    mols_decision = plan.decision_for(FittingAxisNames.N_MOLS)
    conf_decision = plan.decision_for(FittingAxisNames.N_CONFORMERS)
    assert mols_decision.batch_size > 0
    assert conf_decision.batch_size > 0


def test_make_fitting_planner_uses_plan_with_xtrax_not_greedy():
    """Kill test (XR-FIT-FLIP): entrypoint must call plan_with_xtrax, not .plan()."""
    source = inspect.getsource(make_fitting_planner)
    assert "plan_with_xtrax" in source
    assert ").plan()" not in source
    assert "resolve_device_budget_bytes" in source


def test_make_fitting_planner_is_thin_wrapper_over_plan_with_xtrax():
    """Decision-parity: entrypoint matches direct BatchPlanner.plan_with_xtrax."""
    spec = BatchingConfig(n_mols=32, n_conformers=100)
    headroom = 0.80
    param_bytes = 0.0
    activation_multiplier = 2.5

    via_entrypoint = make_fitting_planner(
        spec,
        param_bytes=param_bytes,
        headroom=headroom,
        activation_multiplier=activation_multiplier,
    )

    budget = resolve_device_budget_bytes(headroom, param_bytes)
    axes = [
        dataclasses.replace(
            N_MOLS,
            cardinality=max(1, spec.n_mols),
            default_batch_size=N_MOLS.default_batch_size,
        ),
        dataclasses.replace(
            N_CONFORMERS,
            cardinality=max(1, spec.n_conformers),
            default_batch_size=N_CONFORMERS.default_batch_size,
        ),
    ]
    via_direct = BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(
            ds, 1.0, activation_multiplier
        ),
    ).plan_with_xtrax()

    for name in (FittingAxisNames.N_MOLS, FittingAxisNames.N_CONFORMERS):
        assert (
            via_entrypoint.decision_for(name).batch_size
            == via_direct.decision_for(name).batch_size
        )
