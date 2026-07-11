---
title: "XR-SINK-XTC"
backlog_id: XR-SINK-XTC
praxia_id: 3282
epic: 260709_xtrax_rewire
depends_on: [XR-DISPATCH]
priority: P1
status: completed
challenge_verdict: pass
challenge_summary: "Proxide XtcWriter is a Rust stub; lock mdtraj XTCTrajectoryFile behind XtcWriterBackend; XtcFrameSink implements xtrax Sink; EnsemblePlan.run(xtc_path=) single-bundle flush; Zarr rejected."
completed_2026_07_09: true
promoted_reason: "User override 2026-07-09 — Zarr rejected for MD traj; compose XTC via proxide"
---

# XR-SINK-XTC

## Goal
Compose **XTC writing** with the xtrax sink / `AxisBoundary` Tap–Sink protocol for EnsemblePlan trajectories.

## Explicit reject
Do **not** adopt `xtrax.run.ZarrStagingSink` as the production MD trajectory sink. Zarr may remain available for non-MD tensor dumps elsewhere; it is the wrong format for MD traj interchange.

## Locked decisions (2026-07-09)

| Topic | Lock |
|-------|------|
| Writer | `XtcWriterBackend` Protocol; default `MdtrajXtcWriter` (mdtraj `XTCTrajectoryFile`) |
| Proxide | Rust `XtcWriter` is a **stub** (`create`/`write_frame` return not-implemented). Readback via `proxide.parse_xtc` (Å). Swap backend when proxide ships a real Py binding. |
| Sink | `XtcFrameSink` — `ordered=True`, `__call__(positions)` → frames; host-side (wrap with `io_callback` under JAX trace) |
| EnsemblePlan | Optional `run(..., xtc_path=)` single-bundle only; post-run flush via `write_positions_xtc` |
| Units | Prolix positions Å → mdtraj writes nm; proxide parse returns Å |
| Zarr | Never recommended for MD traj |

## Acceptance Criteria
1. Given an EnsemblePlan run with traj output enabled, when frames flush, then an `.xtc` file is written (via `XtcWriterBackend` / mdtraj interim).
2. Given that `.xtc`, when read back with proxide and/or MDTraj, then coordinates match the in-memory trajectory within XTC single-precision tolerance.
3. Given the implementation, when inspected, then it composes with xtrax `Sink` protocol (`XtcFrameSink`) — not a fork that bypasses the sink protocol entirely.
4. Given CI/docs, when describing MD traj output, then ZarrStagingSink is not recommended as the MD path.

## Implementation
- [`src/prolix/api/xtc_sink.py`](../../src/prolix/api/xtc_sink.py) — `XtcFrameSink`, `MdtrajXtcWriter`, `write_positions_xtc`
- [`src/prolix/api/ensemble_plan.py`](../../src/prolix/api/ensemble_plan.py) — `run(xtc_path=)`
- [`tests/api/test_xr_sink_xtc.py`](../../tests/api/test_xr_sink_xtc.py)

## Non-goals
Claim-1 throughput; StageBundle; replacing all legacy ArrayRecord writers; implementing proxide Rust XTC encode; multi-bundle `xtc_path`.

## References
- proxide Rust stub: `crates/proxide-io/src/formats/xtc.rs` (`XtcWriter`)
- xtrax `Sink` / `AxisBoundary`; `make_sink` today only zarr/none — XTC lives in prolix composition
- mdtraj `XTCTrajectoryFile`
