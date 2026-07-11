---
title: "XR-DEDUP"
backlog_id: XR-DEDUP
praxia_id: 3281
epic: 260709_xtrax_rewire
depends_on: [XR-DISPATCH, XR-SHADOW]
priority: P1
status: completed
challenge_verdict: pass
challenge_summary: "DedupSpec→DedupGather planning; execute via axis_dispatch (not make_axis_dispatch); N_MOLS forced homogeneous; dispatch_n_mols still rejects DedupGather."
completed_2026_07_09: true
promoted_reason: "User override 2026-07-09 — was incorrectly deferred in brainstorm"
---

# XR-DEDUP

## Goal
Use `DedupSpec` / `DedupGather` for ensemble axes where many elements share topology (multi-seed, multi-temp, conformer batches) so the body runs K≪N unique times and scatters back.

## Locked decisions (2026-07-09)

| Topic | Lock |
|-------|------|
| Axis | `n_mols` (`N_MOLS`); forced `heterogeneous=False` when DedupSpec present (shared topology ⇒ shared shapes) |
| Planning | `dedup_specs=` on `plan_axes_with_xtrax` / `BatchPlanner` → xtrax Phase 0b → `DedupGather` |
| K bucket | `get_k_bucket(k)` power-of-2; prolix `batch_size` = `k_bucket` |
| Execution | `dispatch_n_mols_dedup` → xtrax `axis_dispatch(DedupGather, …)` (dedup→map→gather) |
| `make_axis_dispatch` | Still rejects `DedupGather` — `dispatch_n_mols` raises `DispatchRejected` (no silent vmap) |
| EnsemblePlan wire | Out of scope this leaf (helper + adapter only; callers opt in) |
| Topology keying | Caller supplies `unique_indices` / `index_map` / `k` (no auto-hash in this leaf) |

## Acceptance Criteria
1. Given a fixture batch with duplicate topologies, when planned with `DedupSpec`, then the decision is `DedupGather` with K = unique count (bucketed per xtrax `get_k_bucket`).
2. Given that plan, when the body executes, then unique slots run once and results scatter to N positions with bit-exact equality on duplicates vs a no-dedup reference.
3. Given `make_axis_dispatch` rejects DedupGather, when this leaf lands, then host-side dedup prep + documented execute path is used (not silent vmap).

## Implementation
- [`src/prolix/tiling/xtrax_adapter.py`](../../src/prolix/tiling/xtrax_adapter.py) — `dedup_specs=` on `XtraxBatchPlanner`; `DedupGather` → `batch_size=k_bucket`
- [`src/prolix/tiling/planner.py`](../../src/prolix/tiling/planner.py) — `BatchPlanner.dedup_specs`
- [`src/prolix/api/ensemble_dedup.py`](../../src/prolix/api/ensemble_dedup.py) — `plan_n_mols_with_dedup`, `dispatch_n_mols_dedup`
- [`tests/tiling/test_xr_dedup.py`](../../tests/tiling/test_xr_dedup.py)

## Non-goals
Host length-bucketing (#746 / XR-BUCKET); Zarr sinks; auto topology hashing; EnsemblePlan production wire.

## References
- `xtrax.tiling.dedup.DedupSpec`, `xtrax.tiling.strategy.DedupGather`
- `xtrax.tiling.dispatch.axis_dispatch` (eager DedupGather path)
