---
title: "XR-BUDGET — MemoryBudget; drop secondary demotion"
backlog_id: XR-BUDGET
praxia_id: 3274
epic: 260709_xtrax_rewire
depends_on: [XR-PIN]
priority: P1
difficulty: standard
status: near_ready
challenge_verdict: near_ready
challenge_summary: "1A estimator + MemoryBudget API + fail-loud infeasible locked; secondary demotion deleted."
---

# XR-BUDGET

## Goal
Adapter uses xtrax joint `MemoryBudget` with a bridged estimate callable; no host secondary demotion loop.

## Locked decisions (1A)

| Topic | Lock |
|-------|------|
| API | Construct `xtrax.tiling.MemoryBudget(bytes=int, estimate=fn)` inside the adapter; call `XtraxBatchPlanner(budget=...)`. Mutually exclusive with per-axis `memory_estimator`. Thin shim keeps `budget_bytes` / `estimate_memory` call-site signatures. |
| Secondary demotion | Delete the `while estimate_memory…` loop in `xtrax_adapter.py` entirely. |
| Estimator 1A | **GPU/prod:** prefer `device_memory_budget(fraction=headroom)` when `memory_stats` has `bytes_limit`. **CI CPU:** on `RuntimeError` / missing stats, fixed `4<<30` device limit (same as prior `_device_budget_bytes` fallback) × headroom − param_bytes. **Estimate:** bridge prolix `estimate_memory_theoretical` (or caller-supplied `estimate_memory`) over the full xtrax `AxisDecision` sequence for MVP. AOT `lowered_memory_estimate` is optional follow-up. |
| Infeasible | Propagate `BudgetInfeasibleError`. Hetero-only / over-budget paths must raise — no silent `budget_exceeded=True` success. |
| Call sites | Adapter + `EnsembleMDPlanner.plan` + `BatchPlanner.plan_with_xtrax`. Fitting greedy `.plan()` stays until XR-FIT-FLIP. |
| Kill test | Assert no `prolix budget demotion` loop; planner constructed with `budget=`; infeasible fixture raises. |

## Scope
- `src/prolix/tiling/xtrax_adapter.py`
- `src/prolix/api/ensemble_planner.py` (int budget + error propagation as needed)
- `tests/tiling/test_xtrax_adapter.py`

## Non-goals
Deleting prolix `BatchPlanner` class (XR-KILL-FORK); flipping fitting (XR-FIT-FLIP); replacing theoretical estimate with full AOT lowered estimate.

## Acceptance Criteria
1. Given `plan_axes_with_xtrax` is called, when a plan is produced under budget, then no secondary host demotion loop runs (source contains no `prolix budget demotion` reasoning string / while-demotion loop).
2. Given CI CPU, when `device_memory_budget` fails, then budget bytes use the documented `4<<30` fallback path; estimate is the bridged joint callable over the full decision sequence.
3. Given budget is infeasible, when planning, then `BudgetInfeasibleError` propagates — no silent over-budget plan with `budget_exceeded=True` masking.
4. Given existing adapter tests, when updated, then they assert joint-budget behavior (Vmap retain / demotion / infeasible raise) and kill-assert `budget=` construction.

## Rollback
Restore previous adapter commit.

## References
- xtrax `tiling/budget.py`, `tiling/estimators.py`
- Epic AC2, pre-mortem estimator skew
