"""XR-KILL-FORK structural gates: no competing greedy planner loop."""

from __future__ import annotations

import inspect
from pathlib import Path

from prolix.tiling.planner import BatchPlanner
import prolix.tiling.planner as planner_mod
import prolix.tiling.xtrax_adapter as xtrax_adapter_mod


def test_batch_planner_plan_is_thin_alias_of_plan_with_xtrax():
    source = inspect.getsource(BatchPlanner.plan)
    assert "plan_with_xtrax" in source
    assert "hom_decisions" not in source
    assert "demoted to safe_map" not in source
    assert "for i, ax in enumerate(homogeneous)" not in source


def test_planner_module_has_no_phase2_greedy_loop():
    source = inspect.getsource(planner_mod)
    assert "Phase 2: Greedy budget loop" not in source
    assert "proven prolix greedy loop" not in source
    # Competing demotion body must be gone
    assert "hom_decisions[i] = AxisDecision" not in source


def test_adapter_still_has_no_secondary_prolix_budget_demotion():
    source = inspect.getsource(xtrax_adapter_mod)
    assert "prolix budget demotion" not in source
    assert "while estimate_memory" not in source


def test_planner_py_file_has_no_greedy_demotion_loop_text():
    path = Path(planner_mod.__file__)
    text = path.read_text(encoding="utf-8")
    assert "demoted to safe_map tile=" not in text
    assert "Greedy budget loop for homogeneous" not in text
