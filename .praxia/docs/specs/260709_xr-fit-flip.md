---
title: "XR-FIT-FLIP — fitting onto shared planner"
backlog_id: XR-FIT-FLIP
praxia_id: 3277
epic: 260709_xtrax_rewire
depends_on: [XR-SHADOW]
priority: P1
difficulty: quick
status: near_ready
challenge_verdict: near_ready
challenge_summary: "Call shape locked: .plan() → .plan_with_xtrax(); budget via resolve_device_budget_bytes; kill + thin-wrapper parity tests."
---

# XR-FIT-FLIP

## Goal
Fitting entrypoints use the shared xtrax-backed planner path; no `BatchPlanner.plan()` greedy call.

## Locked decisions

| Topic | Lock |
|-------|------|
| Replacement | `.plan()` → `.plan_with_xtrax()` on the existing `BatchPlanner` construction |
| Budget | `resolve_device_budget_bytes(headroom, param_bytes)` (reuse MD helper; drop duplicated `memory_stats` try/except in `spec.py`) |
| Axes | Unchanged: `N_MOLS` + `N_CONFORMERS` from `BatchingConfig` (still heterogeneous defaults) |
| Infeasible | `BudgetInfeasibleError` may propagate (adapter fail-loud); no silent `budget_exceeded` success |
| Kill test | Assert `make_fitting_planner` source/call path uses `plan_with_xtrax`, not `.plan()` |
| Parity | XR-SHADOW suite still green after flip; thin-wrapper parity vs direct `plan_with_xtrax` |
| Out of scope | Deleting greedy `BatchPlanner.plan` (KILL-FORK); fitting runtime → `make_axis_dispatch` |

## Scope
- `src/prolix/run/spec.py` `make_fitting_planner`
- Tests: `tests/run/test_spec.py`; keep shadow + batchplan_dispatch green

## Non-goals
Deleting the class (XR-KILL-FORK); OpenMM parity; fitting step/evaluate → `make_axis_dispatch`.

## Acceptance Criteria
1. Given `make_fitting_planner` is invoked, when a plan is produced, then it uses `plan_with_xtrax` / shared adapter path — not prolix greedy `.plan()`.
2. Given XR-SHADOW fixture, when run after flip, then fitting↔MD equality still passes.
3. Given `tests/run/test_spec.py` and `tests/fitting/test_batchplan_dispatch.py`, when updated, then they expect shared-path behavior (kill + thin-wrapper parity in `test_spec`).

## Rollback
Restore `.plan()` call site.

## References
- XR-SHADOW (prerequisite); XR-BUDGET 1A budget helper
