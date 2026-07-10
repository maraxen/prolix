---
title: "XR-DISPATCH-MULTI — make_axis_dispatch beyond N_MOLS"
backlog_id: XR-DISPATCH-MULTI
praxia_id: 3287
epic: 260709_xtrax_rewire
depends_on: [XR-DISPATCH]
priority: P3
difficulty: standard
status: completed
challenge_verdict: pass
challenge_summary: "dispatch_vmap_safemap + dispatch_n_atoms; dispatch_n_mols refactored; 13 tests green."
---

# XR-DISPATCH-MULTI

## Goal
Expand EnsemblePlan / tiling dispatch so `make_axis_dispatch` iterators apply on axes beyond N_MOLS (debt from XR-DISPATCH 2A).

## Locked decisions

| Topic | Lock |
|-------|------|
| Generic helper | `dispatch_vmap_safemap(axis_name, strategy, fn, args, in_axes=0)` applies `make_axis_dispatch` iterator (not discarded) |
| New axis | **N_ATOMS** via `dispatch_n_atoms` + `n_atoms_strategy(plan, n)` (Vmap if batch_size 0/`>=n`, else SafeMap) |
| Refactor | `dispatch_n_mols` routes through the generic helper |
| Unsupported on mapped axes | `Bucket` / `DedupGather` / `Scan` → `DispatchRejected` / `TypeError` (unchanged policy) |
| Non-goals | Host Bucket pad (#746 / XR-BUCKET); DedupGather execute; Scan (already XR-CARRY on N_STEPS); Claim-1 |

## Acceptance Criteria
1. Given Vmap/SafeMap strategy, when `dispatch_vmap_safemap` / `dispatch_n_atoms` runs, then the `make_axis_dispatch` iterator is **applied**.
2. Given `dispatch_n_mols`, when grepped, then it uses the generic helper (no duplicated iterator apply).
3. Given Bucket/Scan/Dedup on N_ATOMS path, when dispatched, then raise — not silent-vmap.
4. Given existing N_MOLS V-dispatch tests, when run, then they stay green.

## References
- `src/prolix/api/ensemble_dispatch.py`
- Parent: `.praxia/docs/specs/260709_xr-dispatch.md`
- Idea: `.praxia/ideas.jsonl` `xr_dispatch_multi_2026_07_09`
