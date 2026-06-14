"""Tests for EnsemblePlan stub."""

import pytest
from prolix.api import EnsemblePlan


class FakePlanner:
    """Stub planner for testing constructor."""

    def plan(self, bundles):
        """Return None (deferred behavior)."""
        return None


def test_ensemble_plan_construction():
    """Test EnsemblePlan can be constructed with empty bundle list."""
    ep = EnsemblePlan([])
    assert hasattr(ep, "bundles")
    assert ep.bundles == []


def test_ensemble_plan_run_raises_not_implemented():
    """Test EnsemblePlan.run() raises NotImplementedError."""
    ep = EnsemblePlan([])
    with pytest.raises(NotImplementedError) as excinfo:
        ep.run(n_steps=10, dt=0.5, kT=2.479e-3)

    # Verify error message references #1842
    assert "#1842" in str(excinfo.value)
    assert "Sprint 38" in str(excinfo.value)


def test_ensemble_plan_with_planner():
    """Test EnsemblePlan stores batch_plan when planner provided."""
    fake_planner = FakePlanner()
    ep = EnsemblePlan([], planner=fake_planner)
    assert ep.batch_plan is None


def test_ensemble_plan_with_none_planner():
    """Test EnsemblePlan sets batch_plan to None when planner is None."""
    ep = EnsemblePlan([], planner=None)
    assert ep.batch_plan is None
