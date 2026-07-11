---
title: "XR-DISPATCH — make_axis_dispatch EnsemblePlan"
backlog_id: XR-DISPATCH
praxia_id: 3275
epic: 260709_xtrax_rewire
depends_on: [XR-PIN]
priority: P1
difficulty: standard
status: near_ready
challenge_verdict: near_ready
challenge_summary: "2A N_MOLS-only; iterators applied (not discarded); Bucket/Dedup/Scan rejected; multi-axis debt filed."
---

# XR-DISPATCH

## Goal
EnsemblePlan N_MOLS execution applies `xtrax.tiling.dispatch.make_axis_dispatch` iterators; prolix `batched_simulate.safe_map` is not used on that path.

## Locked decisions (2A)

| Topic | Lock |
|-------|------|
| Axes | **N_MOLS only** via `dispatch_n_mols`. |
| Apply iterators | `iterator = make_axis_dispatch(strategy, axis=N_MOLS.name); return iterator(fn, (stacked_bundle, seeds), in_axes=0)` (or documented equivalent). **No** discarded `make_axis_dispatch` + hand-rolled `jax.vmap` / `safe_map`. |
| Unsupported | `Bucket` / `DedupGather` / `Scan` on this path: raise `DispatchRejected` or `TypeError` — do not silent-vmap. Scan deferred to XR-CARRY. |
| V-gates | V1/V3/V4/V5 + V8 (`tests/batching/test_safe_map_varying_shape_spec.py`); V6 out of this leaf. |
| Debt | Expand `make_axis_dispatch` to other EnsemblePlan axes (e.g. N_ATOMS) post-MVP — `XR-DISPATCH-MULTI`. |

## Scope
- `src/prolix/api/ensemble_dispatch.py` and callers
- Vmap / SafeMap only on N_MOLS
- Keep V1/V3/V4/V5/V8 green

## Non-goals
Host Bucket pad pipeline (XR-BUCKET); DedupGather; Scan/Carry (XR-CARRY); multi-axis dispatch (debt); legacy `batched_simulate` callers outside EnsemblePlan.

## Acceptance Criteria
1. Given an EnsemblePlan multi-bundle stacked run, when N_MOLS executes, then dispatch **applies** the iterator returned by `make_axis_dispatch` for Vmap/SafeMap (not a dead validate-only call).
2. Given V-series tests V1/V3/V4/V5 and V8, when run with `-m "not slow"`, then they pass after the change.
3. Given EnsemblePlan path, when grepping imports, then that path does not call `prolix.batched_simulate.safe_map`.
4. Given `DedupGather`, `Bucket`, or `Scan` strategy appears on this path, when dispatch is requested, then raise `DispatchRejected` or `TypeError` — not silently vmap'd.

## Rollback
Revert dispatch module commit.

## References
- `ensemble_dispatch.py`, xtrax `make_axis_dispatch`
- Debt: XR-DISPATCH-MULTI
