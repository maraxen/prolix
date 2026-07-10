# XR rewire — spec_challenge rollup (2026-07-09; refreshed 2026-07-10 XA-SYNC)

**Promotion:** 13 items → `.praxia/backlog.jsonl` (staging DB insert failed).

**Epic status (2026-07-10):** all XR leaves **completed** / challenge `pass` (see table). Stale “all 8 = not_ready” header retired.

**XA-SYNC (2026-07-10):** `bth sync engaging --pull` failed (rsync 23); fallback catalog rsync OK. Local bathos row `cbea8064…` tags `xr-parity-omm-water` still `outcome=unknown`. Campaign `5ffe2644` pass remains **UNVERIFIED locally** (cited only via challenge summary / Titanix claim). Paper must not treat local catalog as confirmatory.

| Leaf | Verdict | Blocking | Worst gap |
|------|---------|----------|-----------|
| XR-PIN | pass (2026-07-09) | 0 | PyPI `xtrax>=0.4.0a5,<0.5`; check_xtrax_pin.py + CI; path/editable rejected. |
| XR-BUDGET | near_ready (implemented 2026-07-09) | 0 | 1A locked; MemoryBudget live; secondary demotion deleted. |
| XR-DISPATCH | near_ready (implemented 2026-07-09) | 0 | 2A N_MOLS-only; iterators applied; XR-DISPATCH-MULTI #3287 debt. |
| XR-SHADOW | near_ready (implemented 2026-07-09) | 0 | Option 1: shared xtrax path in test; SHADOW_AXES; strategy-type + dispatch asserts. |
| XR-FIT-FLIP | near_ready (implemented 2026-07-09) | 0 | .plan()→plan_with_xtrax; resolve_device_budget_bytes; kill+thin-wrapper parity. |
| XR-PARITY-OMM-WATER | pass (2026-07-09) | 0 | Campaign 5ffe2644 pass: |ΔE|=0.040, force_rmse=0.011, mean_T=303.6 K; XR-KILL-FORK unblocked. |
| XR-KILL-FORK | pass (2026-07-09) | 0 | Greedy loop deleted; plan()→plan_with_xtrax; V7 joint-budget; AC8 paper gate lifted. |
| XR-CARRY | pass (2026-07-09) | 0 | N_STEPS CarrySpec→Scan; dispatch_n_steps JaxScanIterator; EnsemblePlan._run_single wired. |
| XR-DEDUP | pass (2026-07-09) | 0 | DedupSpec→DedupGather; axis_dispatch execute; dispatch_n_mols still rejects DedupGather. |
| XR-SINK-XTC | pass (2026-07-09) | 0 | XtcFrameSink + mdtraj interim (proxide XtcWriter stub); EnsemblePlan xtc_path; Zarr rejected. |
| XR-A2A3 | pass (2026-07-09) | 0 | A1+A2 done; A3 C3 via XR-VACUUM-DT (unit+gamma policy delivered). |
| XR-VACUUM-DT | pass (2026-07-09) | 0 | dt fs→AKMA; gamma ps→AKMA; vacuum γ≥50 at dt=0.5 (or dt≤0.1 @ γ=10). |
| XR-PARITY-OMM-PROTEIN | pass (2026-07-10) | 0 | exception_* in energy_fn_from_bundle; 2GB1 ΔE≈0.001, force_rmse≈2e-6. |
| XR-PARITY-KUPS | pass (2026-07-10) | 0 | bathos sidecar+adapter probe pass (campaign 991b1851); crossval optional w/ kups. |
| XR-PARITY-TORCH | pass (2026-07-10) | 0 | prolix Scope A rewire regression; bathos pass (29c5f27f); planner→xtrax. |
| XR-DISPATCH-MULTI | pass (2026-07-10) | 0 | dispatch_vmap_safemap + N_ATOMS; N_MOLS refactored; Bucket/Scan/Dedup reject. |
| XR-BUCKET | pass (2026-07-10) | 0 | #746 compute_dense_tiling_dims + tile_reduction assert; dense LJ/Coulomb routed. |

**Epic:** all XR leaves closed 2026-07-10 (XR-EPIC complete).

## Critical probes to resolve first

1. **XR-PIN:** ~~floor + PyPI pin + CI~~ done (2026-07-09).
2. **XR-BUDGET:** CI/GPU estimator decision table; adapter signature → `MemoryBudget`; hetero infeasible semantics.
3. **XR-DISPATCH:** require applying `make_axis_dispatch` *iterator*, not dead validation call; name axes in scope.
4. **XR-SHADOW:** name post-plan object that still carries strategy types; shared AxisSpec fixture; fitting path before FIT-FLIP.
5. **XR-PARITY-OMM-WATER:** ~~pin T2 + claim~~ done (campaign 5ffe2644 pass).
6. **XR-KILL-FORK:** ~~post-kill API + V7 + structural gate~~ done (2026-07-09); paper/B1-full AC8 lifted.
7. **XR-A2A3:** numeric |grad| bound; V4/V5 in/out; A3 trajectory vs docs; exclusion representation.

Secondary leaves (PROTEIN/KUPS/TORCH/BUCKET) challenge deferred until critical path revises.
