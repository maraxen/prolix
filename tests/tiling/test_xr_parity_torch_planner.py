"""XR-PARITY-TORCH: BatchPlanner.plan aliases plan_with_xtrax (structural)."""

from __future__ import annotations

import pytest

import inspect

from prolix.tiling.planner import BatchPlanner



# XA-CI: heavy parity/compile — deselect from GitHub-faithful suite.
pytestmark = pytest.mark.slow

def test_batch_planner_plan_aliases_plan_with_xtrax():
    body = inspect.getsource(BatchPlanner.plan)
    assert "plan_with_xtrax" in body
    assert BatchPlanner.plan is not BatchPlanner.plan_with_xtrax
