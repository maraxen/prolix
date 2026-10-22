---
title: 'Stage-2 profiling sweep synthesis: full-md-path-matrix, p7-stage2, full-md-production-defaults'
description: 'Synthesizes 3 open bathos campaigns (#4326): NL vs flash vs dense harness quality + preliminary ranking, a sidecar-schema-mismatch bug that made every p7-stage2 flash-vs-dense sweep run gate-fail regardless of content, and confirmation that full-md-production-defaults was pre-registered but never executed.'
status: draft
task_id: 260831_sprint114_bathos_synthesis
date: '260831'
confidence: 'exploration-mode, single-sample-per-cell -- see caveats in each section'
sources: '~/.bth/catalog/runs/prolix/run_*.parquet; profiling/full_md_path_matrix/*/summary.json; profiling/stage2/*_summary.json; profiling/stage2_fresh_h100/bottleneck_report.md; scripts/experiments/prof_ensemble_step.bth.toml; scripts/experiments/profile_b1_flash_vs_autodiff_forces.bth.toml'
---
# Stage-2 profiling sweep synthesis: full-md-path-matrix, p7-stage2, full-md-production-defaults

Closes backlog #4326: synthesize the 3 open bathos campaigns that predate the
#4330 profiler-instrumentation fixes (PR #15). All three turned out to need a
diagnostic pass before any real synthesis was possible -- see each section.

## 1. `full-md-path-matrix` (campaign `bd1f47e8`, exploration mode)

**Question:** "On real `EnsemblePlan.run` inference steps, what are P_flash /
P_nl / P_dense costs, and does NL survive init+step?"

**Harness design (from the sidecar):** this campaign's own `.bth.toml`
explicitly states its gate is **harness-quality-only**
(`r_squared > 0.8` on the fixed-cost/per-step regression fit, or an expected
`status=oom` for the dense path) -- it deliberately does *not* encode
"NL faster than flash" as a per-run pass/fail condition, leaving the kernel
ranking to `bath campaign review` plus manual inspection. That design is
sound and was followed correctly.

**19 catalog rows, single protein (1vii, n_real=1963):**

| config (bucket_remat_dtype) | nonbonded | n_padded | per_step_s | r² | gate |
|---|---|---|---|---|---|
| `bon_ron_f32` | nl | 5000 | 0.1081 | 0.924 | PASS (reliable) |
| `bon_ron_f32` | flash | 5000 | 0.0725 | 0.991 | PASS (reliable) |
| `bon_ron_f32` | dense | 5000 | 0.0507 | 0.304 | FAIL (bad fit, not a real ranking signal) |
| `bon_ron_f64` | flash | 5000 | crash | -- | ERROR |
| `boff_ron_f32` | nl | 1963 | 0.0398 | 0.578 | FAIL (bad fit) |
| `bon_roff_f32` | flash | 5000 | 0.0741 | 0.555 | FAIL (bad fit) |
| `boff_ron_f32` | flash | 1963 | 0.0973 | 0.987 | PASS (reliable) |

(7 additional rows are early submission-attempt `error`s, most retried
successfully at the timestamps above; not separately tabulated.)

**Reliable comparison (the only two cells with `r² > 0.8` at matching
settings):** at production bucket size (`use_size_buckets=True`,
`n_padded=5000`), **flash beats NL** on 1vii: 0.0725s/step vs 0.1081s/step
(flash ~1.49x faster). NL does **not crash** -- it "survives init+step"
structurally -- but it is currently slower than flash at this bucket size.
Likely cause (not directly measured here): NL's neighbor-list build/rebuild
cost scales with the *padded* atom count, not the sparse `k_over_n≈0.18`
working set, so bucketing to a fixed 5000-atom pad (needed for cross-protein
compile sharing, the core B1 thesis) erodes NL's sparsity advantage.

**Suggestive but unreliable (r² ≤ 0.8, treat as directional only):**
unbucketed (`n_padded=1963`, exact system size) NL (0.0398s) looks much
faster than unbucketed flash (0.0973s) -- consistent with the above
hypothesis (NL should win once it isn't forced to build over a padded box)
-- but both fits are poor (r²=0.578, r²=0.987 respectively; only the flash
one clears the bar) so this specific number should not be cited without a
re-run. Dense (bon_ron_f32, r²=0.304) similarly cannot be trusted as
"dense is fastest" -- the regression that produced 0.0507s/step was a bad
fit, most likely from too few timing samples or GPU-contention noise in that
run, not evidence dense genuinely undercuts flash/NL.

**The f64 flash crash is now fixed.** `l40s_1vii_flash_bon_ron_f64` failed
with `TypeError: lax.add requires arguments to have the same dtypes, got
float32, float64` -- this is exactly the hardcoded-`.astype(jnp.float32)`
bug class (debt 832) fixed in `explicit_corrections.py`/`flash_nonbonded.py`
by PR #14 (2026-08-31, merged to `b1/full`). **Recommend a re-run of this one
cell** to get a genuine f64 flash data point; the crash recorded here no
longer reflects current `main`.

**DHFR was in-scope but never run.** The sidecar's own hypothesis names both
"1vii and DHFR"; only 1vii appears anywhere in the catalog or
`profiling/full_md_path_matrix/`. DHFR coverage remains open.

**Conclusion registered:** `campaign_conclude` was called with
`outcome_label="inconclusive-harness-noise"` (see below) -- the harness
mechanics validated correctly (the two `r²>0.8` cells show flash beating NL
at production bucket size, and NL does not crash), but 3 of 7 cells have an
unreliable fit and the f64 cell's failure is stale/already-fixed, so this
campaign cannot yet support a confident kernel-ranking claim. Re-run
recommended (more timing samples per cell to fix the `r²` issue; add DHFR;
re-test the f64 cell now that debt 832 is fixed) before this question is
closed for real.

## 2. `p7-stage2-h200` (campaign `cfa03731`, exploration mode) -- REAL BUG FOUND, not just orphaned data

**Two distinct problems, not one.** Initial triage (before this doc) found
`campaign_review`/campaign_id-filtered SQL return zero runs for this
campaign -- the classic "ran without `--campaign <id>`" anti-pattern this
project's own CLAUDE.md already documents (the runs are tagged `p7-stage2`
etc. but `campaign_id` is an empty string on every one of them). That's real,
but it's the *smaller* problem. The bigger one:

**Every run in the most recent full sweep (2026-08-25, 12 runs, protein ×
{1vii, 2gb1} × mode × {dense, flash} × pme × grid) has `outcome=fail`, with
`exit_code=0`/`status=completed` -- i.e. the script ran successfully and
produced real numbers, but the bathos *gate* marked every single one a
failure.** Root cause, confirmed by reading `sidecar_path` off the catalog
rows: these runs are gated by
`scripts/experiments/profile_b1_flash_vs_autodiff_forces.bth.toml` -- a
**different, older experiment's sidecar** (debt-761, campaign `32d6574e`,
tags `b1/phase5/profiling/connection-work-scoping/flash-forces/debt-761`),
whose `[outcomes.pass]` condition is `flash_speedup >= 1.2` and whose
`[outcomes.fail]` condition is `flash_speedup IS NULL OR flash_speedup < 0.9`.

That schema assumes one run measures **both** `autodiff_forces_ms` and
`flash_forces_ms` in the same invocation and computes their ratio. But the
actual p7-stage2 array-task design splits `mode:dense` and `mode:flash` into
**separate** array tasks/runs -- so every dense-mode run has
`flash_forces_ms=null` and every flash-mode run has
`autodiff_forces_ms=null`, meaning `flash_speedup` is **structurally always
null**, which the borrowed sidecar's `fail` condition catches unconditionally.
Every run in this sweep was doomed to gate-fail regardless of its actual
timing content -- this is a false negative from a sidecar/schema mismatch,
not a real research finding. (Read directly from
`profiling/stage2/stage2_*_summary.json`: every file has
`"flash_speedup": null` next to a perfectly good real timing number in
either `autodiff_forces_ms` or `flash_forces_ms`.)

**The real finding, recovered by manually pairing dense/flash runs on
matching protein+pme+grid (n=1 per cell, exploration-mode caveats apply):**

| protein | pme | grid | dense/autodiff (ms) | flash (ms) | flash_speedup (dense/flash) |
|---|---|---|---|---|---|
| 1vii | off | na | 0.8006 | 0.9277 | 0.863 |
| 1vii | on | 1.0 | 0.8649 | 1.0762 | 0.804 |
| 1vii | on | 1.2 | 0.8662 | 1.0960 | 0.790 |
| 2gb1 | off | na | 0.7930 | 0.9288 | 0.854 |
| 2gb1 | on | 1.0 | 0.8531 | 1.0810 | 0.789 |
| 2gb1 | on | 1.2 | 0.8481 | 1.0869 | 0.780 |

All 6 pairs land at `flash_speedup` 0.78-0.86 -- consistently, comfortably
inside even the *borrowed* sidecar's real `fail` threshold (`< 0.9`), and
nowhere near its `pass` threshold (`>= 1.2`). **At this system scale
(~1963-1987 atoms, n_padded=5000) flash forces are reliably 15-22% *slower*
than dense/autodiff forces**, consistently across both proteins and every
PME/grid-spacing setting tested. This is a genuine, if exploration-mode,
negative result for flash forces at this scale -- not an artifact of the
sidecar bug (the sidecar bug only prevented the *catalog* from recording this
conclusion; the underlying numbers are real and consistent).

**Flag for reconciliation, not resolved here:** project memory
(`project_b1_flash_forces_debt761_verdict_260719.md`) records debt 761 being
reversed **FAIL → PASS** on 2026-07-19 after fixing 3 bugs (tile-padding,
dispersion-tail atom_mask, missing PBC minimum-image) in
`flash_explicit.py`. It's unclear from that record alone whether "PASS"
there meant "flash is now measured *correctly*" (a correctness claim) or
"flash is now measurably *faster*" (the performance claim this sidecar's own
gate encodes). This new p7-stage2 sweep, run over a month later
(2026-08-18 through 2026-08-25) against presumably-current code, shows the
performance claim does **not** currently hold -- flash is slower, not
faster, at this scale. Someone with context on the July verdict should
reconcile whether this is a real regression since then or whether "PASS" in
July was never a performance claim in the first place.

**Also mislabeled hardware.** `bottleneck_report.md` for these exact
`stage2_{1vii,2gb1}_flash_pmeon_*` configs (device metadata column) shows
`device_kind: h100_80gb_hbm3` -- these are H100 runs, not H200 as the
campaign name claims. Another stale/inaccurate label, not touched further
here.

**Not concluded via `campaign_conclude`:** the campaign object genuinely has
zero linked runs (confirmed by both `campaign_review` and a `campaign_id`-
filtered SQL query), so there is nothing for `campaign_conclude` to attach a
verdict to. This synthesis is filed here as the closest available substitute.
Filing this properly requires either (a) re-running the sweep with
`--campaign cfa03731-db54-42d3-a890-8b5968eb7181` explicitly passed to
`bth run` and a corrected sidecar (see backlog recommendation below), or (b)
a one-time backfill of `campaign_id` on the existing rows if bathos exposes
that (not attempted here -- out of scope for a synthesis-only pass, and
mutating historical catalog rows deserves its own explicit decision).

## 3. `full-md-production-defaults` (campaign `d2e3a396`, **confirmation mode**)

**Confirmed genuinely empty**, not just orphaned. Checked three ways: (a)
`campaign_review`/`campaign_show` by campaign_id -- no runs; (b) `run_sql`
filtered on `campaign_id = 'd2e3a396-...'` -- zero rows; (c) a broad
`array_to_string(tags, ',') LIKE '%production%' OR '%defaults%'` scan across
the *entire* catalog, all timestamps -- zero rows under any tag. There is no
evidence this campaign was ever executed against, under any tag or linkage.

Per this project's bathos discipline (confirmation-mode campaigns require an
anchored threshold decision derived from real data), **no outcome can be
fabricated here** -- there is nothing to anchor against. This campaign was
pre-registered (created 2026-08-21) and never run.

**Recommendation:** given this sprint's own findings above -- the
`full-md-path-matrix` NL-vs-flash comparison is itself still inconclusive
(harness noise) and the `p7-stage2-h200` flash-vs-dense comparison is
undermined by a sidecar bug even though its recovered numbers are directionally
clear -- I do not think the underlying "production defaults" question is moot
or superseded; if anything it depends on both of the above being resolved
first. **Recommend leaving this campaign open** rather than closing it as
abandoned, but sequencing its actual execution *after* items 1 and 2 above
get a clean re-run, so the "production defaults" confirmation run is anchored
against reliable exploration-mode data rather than repeating the same
harness-noise/sidecar-mismatch problems into a confirmation-mode result
(which would be worse, since confirmation mode is meant to be the trustworthy
final word).

## 4. Correcting the record: "flash bonded-off is slower than bonded-on" is not a real anomaly

An earlier reading of the profiling matrix suspected an anomaly along these
lines. It does not exist as stated. Confirmed two ways in this pass:

1. Every `full-md-path-matrix` summary.json's schema is
   `{system, nonbonded, use_size_buckets, remat, dtype, ...}` -- there is no
   bonded/nonbonded-terms toggle anywhere in this experiment. The `b`/`r`
   token pair in directory names like `l40s_1vii_flash_bon_ron_f32` encodes
   `use_size_buckets` (`bon`/`boff`) and `remat` (`ron`/`roff`) --
   confirmed directly from the JSON field values matching the filename
   tokens.
2. `profiling/stage2_fresh_h100/bottleneck_report.md` independently confirms
   `flash_bonded` and `flash_lj_only` are real named-scope labels inside
   `flash_explicit.py`, not a "toggle" -- and (as PR #15, backlog #4330,
   already established this session) the production flash force path
   ( `flash_explicit_forces` → `_total_energy_fn` ) never includes bonded
   terms at all, under any bucket/remat setting. There is no configuration
   in which "flash bonded-off" and "flash bonded-on" are two different
   things being compared.

No further action needed; this note exists so the phantom anomaly does not
resurface in a future session.

## 5. PME cost (informational only -- no optimization work done or planned in this pass)

`profiling/stage2_fresh_h100/bottleneck_report.md` (1vii/2gb1, flash+PME-on,
grid 1.0/1.2, H100) confirms PME reciprocal-space work dominates measured
per-step time: `pme_charge_spread` ~11-15%, `pme_fft_inverse` ~4.4-8.7%,
`pme_fft_forward` ~2.9-4.6% of total exclusive time across the 4 probe
configs present. `pme_bwd_gather` and `pme_greens_setup` are present at
~1.4-2.9% and ~1.4-2.2% respectively in this report (predating PR #15's
#4330 fixes -- `pme_greens_setup_standalone`'s "absent" rows in this same
report are now explained rather than mysterious, per that PR).
**No PME optimization work was started or is recommended in this pass** --
noting the cost breakdown for whoever picks up PME work next is the only
purpose of this section.

## Recommended backlog items (not filed by this doc -- flagging for the
## sprint owner to file or decline)

- **Sidecar/schema-mismatch bug**: any future `p7-stage2`-style sweep that
  splits a comparison across separate array-task runs must not reuse a
  sidecar whose `[result_schema]`/`[outcomes]` assume both sides of the
  comparison land in one row (like `profile_b1_flash_vs_autodiff_forces.bth.toml`'s
  `flash_speedup` field). Either write a per-array-task-aware sidecar (pass
  on `status=ok` alone, defer the ratio computation to `campaign review`/a
  dedicated aggregation step, mirroring `full-md-path-matrix`'s own
  harness-quality-only pattern) or restructure the experiment to measure
  both modes in one run.
- **Campaign-linkage discipline**: `p7-stage2-h200`'s runs need
  `--campaign cfa03731-db54-42d3-a890-8b5968eb7181` (or whatever the current
  `bth run` flag is) passed explicitly on the next re-run; tag-string
  matching alone does not link to the campaign object.
- **Debt 761 reconciliation**: the July 2026-07-19 FAIL→PASS verdict and
  this sweep's 15-22%-slower finding need someone with context on both to
  reconcile whether this is a real regression or a correctness-vs-performance
  conflation in the July record.
- **DHFR coverage**: `full-md-path-matrix`'s stated scope includes DHFR;
  only 1vii was ever run.
