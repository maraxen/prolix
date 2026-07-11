---
title: "XR-KILL-FORK — delete prolix greedy BatchPlanner"
backlog_id: XR-KILL-FORK
praxia_id: 3279
epic: 260709_xtrax_rewire
depends_on: [XR-PARITY-OMM-WATER]
priority: P1
difficulty: standard
status: completed
challenge_verdict: pass
challenge_summary: "Greedy loop deleted; plan() aliases plan_with_xtrax; V7 rewritten to joint-budget; structural kill gates green. Paper/B1-full AC8 gate lifted."
---

# XR-KILL-FORK

## Goal
Remove competing prolix `BatchPlanner.plan` greedy implementation; leave thin re-exports / domain axis registry only; rewrite V7 to joint-budget semantics.

## Locked decisions (post-kill)

| Topic | Lock |
|-------|------|
| Public API | Export `AxisSpec`, `AxisDecision`, `BatchPlan`, `BatchPlanner`, `estimate_memory_theoretical` from `prolix.tiling`. Do **not** re-export raw xtrax `BatchPlanner` as the prolix name. |
| `BatchPlanner.plan()` | Thin alias of `plan_with_xtrax()` → `plan_axes_with_xtrax` / `MemoryBudget`. |
| Infeasible | `BudgetInfeasibleError` (fail-loud); no silent `budget_exceeded=True` success. |
| Grep / kill gate | Structural: `plan` body only calls `plan_with_xtrax`; no `hom_decisions` demotion loop in `planner.py`; adapter has no `prolix budget demotion`. |

## V7 joint-budget contracts
1. Hetero `n_mols.batch_size > 0` at feasible budgets.
2. Generous (2×): homogeneous axes stay `batch_size == 0`.
3. Tight: cardinality > default_batch_size demotes to tile; `"joint-budget"` in reasoning.
4. Infeasible → `BudgetInfeasibleError`.
5. Determinism: identical inputs → identical `(name, batch_size)`.

## Acceptance Criteria
1. Given post-change tree, when structural kill tests run, then `BatchPlanner.plan` is a thin alias and `planner.py` has no Phase-2 greedy loop.
2. Given V7 tests rewritten, when pytest runs them, then they encode xtrax joint-budget demotion order/contracts — not deleted.
3. Given fitting and MD callers, when planning, then both resolve to the same xtrax-backed authority (shadow still green).
4. **Gate lifted (AC8):** paper / B1-full checklist may proceed past XR-KILL-FORK (this leaf completed 2026-07-09).

## Implementation notes (2026-07-09)
- [`src/prolix/tiling/planner.py`](../../src/prolix/tiling/planner.py): greedy body removed; `plan()` → `plan_with_xtrax()`.
- [`tests/physics/test_batch_planner_v7.py`](../../tests/physics/test_batch_planner_v7.py): joint-budget rewrite.
- [`tests/tiling/test_kill_fork.py`](../../tests/tiling/test_kill_fork.py): structural gates.

## Rollback
Revert delete commit; restore greedy from git.

## References
- Epic AC6–AC8
