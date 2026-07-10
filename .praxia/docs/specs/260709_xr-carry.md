---
title: "XR-CARRY"
backlog_id: XR-CARRY
praxia_id: 3280
epic: 260709_xtrax_rewire
depends_on: [XR-DISPATCH]
priority: P1
status: completed
challenge_verdict: pass
challenge_summary: "N_STEPS + CarrySpec→Scan planning; dispatch_n_steps via JaxScanIterator; EnsemblePlan._run_single wired. Hetero CarrySpec fails loud."
completed_2026_07_09: true
---

# XR-CARRY

## Goal
Declare xtrax `CarrySpec` for the MD step axis so `BatchPlanner` pre-demotes to `Scan`; execute via `make_axis_dispatch` iterators instead of ad-hoc `lax.scan` outside the plan.

## Locked decisions (2026-07-09)

| Topic | Lock |
|-------|------|
| Step axis | `n_steps` (`N_STEPS`, axis_index=6) |
| Replica CarrySpec | Out of scope this leaf |
| Execution wiring | `EnsemblePlan._run_single` only (`export_run` / browser_demo remain raw `lax.scan`) |
| JaxScanIterator `xs` | `jnp.arange(n_steps)` |
| Hetero CarrySpec | `ValueError` from xtrax planner |
| Plan-time CarrySpec | stub `init=None` / `stub_step_transition`; real step built at execute |

## Acceptance Criteria
1. Given `plan_n_steps_with_carry`, when planned, then `n_steps` reasoning contains `carry-bearing scan`.
2. Given `EnsemblePlan._run_single`, when the step loop executes, then it uses `dispatch_n_steps` → `JaxScanIterator` (not `jax.lax.scan` in that method).
3. Given a heterogeneous `n_steps` + CarrySpec, when planning, then `ValueError` (static carry shape).

## Implementation
- [`src/prolix/tiling/axes.py`](../../src/prolix/tiling/axes.py) — `N_STEPS`
- [`src/prolix/tiling/xtrax_adapter.py`](../../src/prolix/tiling/xtrax_adapter.py) — `carry_specs=` on `XtraxBatchPlanner`
- [`src/prolix/api/step_carry.py`](../../src/prolix/api/step_carry.py) — `plan_n_steps_with_carry`
- [`src/prolix/api/ensemble_dispatch.py`](../../src/prolix/api/ensemble_dispatch.py) — `dispatch_n_steps`
- [`src/prolix/api/ensemble_plan.py`](../../src/prolix/api/ensemble_plan.py) — `_run_single` uses `dispatch_n_steps`

## Non-goals
StageBundle; DedupGather (XR-DEDUP); traj sinks (XR-SINK-XTC); replica CarrySpec; `export_run` Scan wire.

## References
- `xtrax.tiling.carry.CarrySpec`
- Epic revised sink/carry/dedup section
