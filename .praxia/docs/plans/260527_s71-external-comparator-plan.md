---
task_id: 260527_sprint4_p2b_close_plus_71
sprint: 4
date: 260527
status: pending_C1
owner: planner
---

# §7.1 External Comparator Figure — Implementation Plan

## Scope (locked)

§7.1 compares **prolix v1.2** against **DMFF / TorchMD / espaloma** on the **bonded-energy Scope A primitive** (forward + backward across N mols of mixed sizes, one Adam step). The three-property substrate claim is the load-bearing message:

1. **Portability** (StableHLO export → GPU / TPU / WASM)
2. **Heterogeneous batching** (mixed-size molecules in one graph)
3. **Composability** (Bundle + BatchPlan auto-budgeting)

Throughput is supporting evidence, not the load-bearing claim.

ForceBalance is deferred to Scope C (v1.3). PyTorch-from-scratch is removed (was a strawman; TorchMD is the real citable baseline).

## What was wrong before

**2026-05-21 → 2026-05-22:** An internal looped-vs-batched sweep (prolix N={16..512}, 99.89× at N=512 f32) was mistakenly positioned as the §7.1 paper figure — conflating "prolix beats its own loop" with "prolix beats external tools." User directive on 2026-05-22 reset the framing: §7.1 is external-only; internal speedup is release-notes context.

**Footprint check:** Scaffolding (`bench_*.py` + sidecars + slurm wrapper + campaign `edbd0b84`) is sound. The failure was framing, not code. Only prolix has run (job 14263023, 14.9µs/mol-step f32 N=64 Blackwell).

## User decisions (locked 260527)

| Decision | Value | Rationale |
|---|---|---|
| Comparator system | ANI-1x subset (16 base mols, tiled to N) | Tests heterogeneous batching claim directly |
| Hardware policy | Multi-GPU; **"what doesn't run IS data"** | Tool×hardware compatibility is a first-class §7.1 finding, reported as a coverage matrix — not a footnote caveat |
| Pre-reg threshold | Conservative: `per_mol_step_seconds(prolix) < 0.5 × min(dmff, torchmd)` | Strong-but-achievable; 2× faster than fastest external |

## What "reasonable" looks like

**Figure shape:** Two artifacts together —

1. **Performance plot:** X=N ∈ {16, 32, 64, 128, 256, 512}; Y=per_mol_step_seconds (log); facets by precision (f32 | f64); tools colored (prolix / DMFF / TorchMD / espaloma); error bars = min/max over 3 trials.
2. **Compatibility matrix:** Rows=tool, Cols=hardware (Blackwell RTX Pro 6000, A100, CPU-fallback). Cells: ran / build-failed / runtime-failed / OOM. **This is the portability finding made legible.**

**Definition of done:** All four tools attempted on Blackwell + A100; runs that complete land in bath catalog; failures land in the compatibility matrix with the precise error class; figure + caption committed.

## Decomposition (C1–C7)

### C1 — Audit scaffolding (~2h, planner/auditor)

Verify each `bench_*.py` and `.bth.toml` does what its docstring claims:
- Loads ANI-1x base set + tiles to N
- One Adam step per trial; timing excludes compile
- Output JSON matches `result_schema` in sidecar
- Sidecars have exactly one `is_residual=true` outcome and valid DuckDB conditions
- `_generate_xml.py` unit-converts AKMA → OpenMM correctly

Per-tool venv check (`.venv-dmff`, `.venv-torchmd`, `.venv-espaloma`).

**Gate:** No showstoppers, OR each blocker named with severity.

### C2 — Local CPU smoke per tool (~3h, fixer×4 in parallel)

Each tool at N=16, n-trials=1, float32, CPU only, <30s. Verify JSON validity, non-zero loss (XLA-DCE check), non-zero timing. Fix locally before any cluster dispatch.

**Gate:** 4/4 tools produce valid JSON locally.

### C3 — Cluster smoke matrix (~1d calendar, cluster ops)

Per decision 2 ("what doesn't run IS data"), C3 is a **matrix smoke**, not a single-hardware smoke:

For each (tool, hardware) ∈ {prolix, dmff, torchmd, espaloma} × {Blackwell, A100}:
- N=64, f32, 1 trial
- Submit via `bth run` wrapper + slurm
- Record: ran (per_mol_step_seconds) | build-failed (which dep) | runtime-failed (which error) | OOM

**Gate:** All 8 cells have a recorded outcome (success OR classified failure).

### C4 — Anchor thresholds in `campaign_design.toml` (~1h, planner; user review)

Replace TODO placeholders with empirical baselines from C3 ran-cells. Use 0.5× of `min(dmff_ran, torchmd_ran)` per decision 3. Validate via `bth check`. Commit before C6.

**Gate:** Sidecar passes bath validation; threshold cites C3 git hash.

### C5 — Implement `scripts/analysis/s71_external_comparator.py` (~3–4h, fixer)

Reads bath catalog filtered by campaign `edbd0b84`. Produces two artifacts:
- Performance plot (matplotlib; spec: log Y, faceted by precision, colored by tool, error bars=min/max)
- Compatibility matrix (markdown table or PNG heatmap)

Idempotent; runnable via `uv run python scripts/analysis/s71_external_comparator.py --campaign edbd0b84`.

**Gate:** Script runs end-to-end on C3 smoke data; output files produced.

### C6 — Full confirmation sweep (~1 week calendar, cluster ops)

For each cell that **ran** in C3: full grid N×precision×trials = 6×2×3 = 36 trials. Skip cells that failed in C3 (their failure is the data point).

Total: up to 4 tools × 2 hardware × 36 = 288 trials, parallelized.

**Gate:** All scheduled cells complete or land in catalog with classified failure; `bth campaign review edbd0b84` passes.

### C7 — Final figure + caption + release notes (~2h, planner)

Run C5 script on C6 data. Caption emphasizes the three-property claim and reads the compatibility matrix as evidence of portability (not as a defect list). Release-notes paragraph aligned with outcome (pass / marginal / fail).

**Gate:** Figure + caption + release-notes committed; campaign tagged `s71-final`.

## Risk table

| Risk | Likelihood | Mitigation | Residual |
|---|---|---|---|
| DMFF/espaloma CUDA-12.1-locked, won't run on Blackwell | High | C3 matrix smoke records this as data, not a blocker | None — it's the data |
| Tool venv install fails | Medium | C1 audit catches; C2 local smoke catches | Low |
| Bench script emits wrong-schema JSON | Medium | sidecar `result_schema` validation in C2 | Low |
| XLA dead-code elimination (loss→0) | Low | sidecar gate asserts final_loss > 0 | Low |
| Threshold too tight post-C3 | Low–Medium | 0.5× is conservative; if marginal, that's a finding, not a fail | Medium — pre-reg integrity intact |
| Cluster partition overload | Low | Stagger submissions; %M ≤ 10 | Low |
| Hardware split confuses readers | Medium | Compatibility matrix is explicit; caption frames it as portability finding | Low |

## Remaining decision points (defer to relevant phase)

- **C5 figure aesthetics:** log vs linear Y, bar vs line, library (matplotlib vs plotly). Decide after seeing C3 data.
- **C6 trial count if cluster time is tight:** can drop to 2 trials per cell.
- **C7 release-notes framing:** depends on whether prolix passes 0.5× threshold or lands marginal.

## Files touched (cumulative through C7)

| Path | Action | Phase |
|---|---|---|
| `scripts/benchmarks/external_baseline/bench_prolix.py` | finalize | C1 |
| `scripts/benchmarks/external_baseline/bench_dmff.py` | finalize | C1 |
| `scripts/benchmarks/external_baseline/bench_torchmd.py` | finalize | C1 |
| `scripts/benchmarks/external_baseline/bench_espaloma.py` | finalize | C1 |
| `scripts/benchmarks/external_baseline/_generate_xml.py` | verify | C1 |
| `scripts/benchmarks/external_baseline/campaign_design.toml` | anchor thresholds | C4 |
| `scripts/analysis/s71_external_comparator.py` | create | C5 |
| `outputs/analysis/s71_external_comparator.png` | output | C7 |
| `outputs/analysis/s71_external_comparator_caption.md` | create | C7 |

**Do NOT touch:** `scripts/analysis/s71_scaling_curve.py` (internal looped-vs-batched, unrelated). No ForceBalance, no Scope C, no PME work — all out of scope.

## Next step

C1 audit — dispatch when ready. Recommended as a single auditor agent reading all four bench scripts + sidecars + campaign_design + `_generate_xml`; output a checklist with anchored issues.
