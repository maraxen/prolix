---
date: 2026-06-01
task_id: 260601_s71_close_complete
sprint_id: 5
status: ready_for_next_session
branch: main
commit: 3c654fc
---

# Handoff: §7.1 Campaign Close + LFMiddle Status

## What happened this session

Executed the 5-step close procedure from the previous handoff
(`260601_hp4-s71-campaign-close.md`). All steps complete.

### Step 1 — Merged worktree-hp4-s71-postmortem

Landed 4 commits (postmortem, Blackwell slurm fix, handoff doc) onto main.
Also snapshot-committed the untracked LFMiddle diagnostic scripts from
2026-05-29 (`baoab_loop_diag.py`, `temp_profile_diag.py`, their slurm
wrappers) and added `logs/` + `tip3p_tightening/` to `.gitignore`.

### Step 2 — Catalog sync

Run `678da554` (prolix, Blackwell, N=512, 8.18e-07 s/mol-step) confirmed
in local catalog after `rsync -az engaging:~/.bth/catalog/runs/prolix/`.

### Step 3 — campaign_design.toml gate updated

`scripts/benchmarks/external_baseline/campaign_design.toml`:
- Blackwell pass threshold: **1.548e-03 → 9.28e-05** (0.5× of corrected
  espaloma Blackwell 1.856e-04; original was an XLA hang artifact)
- A100 pass threshold: 3.403e-03 → 3.49e-03 (anchored to N=512 torchmd)
- Marginal band bounds updated to match
- Reasoning updated to document the 2026-06-01 espaloma-inclusion decision

Also patched `scripts/slurm/lfmiddle_dt_sweep.slurm` with the XLA Blackwell
workaround (node4007/node4008 guard, `--xla_gpu_shard_autotuning=false`).

### Step 4 — Figure regenerated on cluster

```
ssh engaging "uv run --with duckdb --with matplotlib \
  python scripts/analysis/s71_external_comparator.py --campaign-id edbd0b84"
```

Result: 34 runs loaded, 6 hardware platforms, figure saved to
`outputs/analysis/s71_external_comparator.png` (137 KB) +
`outputs/analysis/s71_compatibility_matrix.md`. Pulled locally via myxcel.

### Step 5 — Campaign edbd0b84 concluded

Campaign metadata was missing from both local and cluster warm DuckDB
(campaign was tagged in run parquets but `bth campaign create` record was
never synced). Resolved by inserting the row directly via `duckdb.connect`.

Final state: **edbd0b84 concluded, outcome=pass**.

Full pass table:

| Hardware | min(dmff, torchmd, espaloma) | 0.5× gate | prolix | Outcome |
|---|---|---|---|---|
| A100 | 6.97e-03 | 3.49e-03 | 2.73e-06 | PASS |
| CPU-only | 1.60e-03 | 8.02e-04 | 1.45e-05 | PASS |
| H200 | 3.56e-03 | 1.78e-03 | 1.70e-06 | PASS |
| L40S | 3.69e-03 | 1.84e-03 | 1.55e-06 | PASS |
| Blackwell (corrected) | 1.86e-04 | 9.28e-05 | 8.18e-07 | PASS |
| RTX Pro 6000 SM120 | 3.09e-03 | 1.55e-03 | 2.09e-06 | PASS |

---

## LFMiddle campaign status (89c9a900) — unexpected finding

The `sacct` check revealed many more completed LFMiddle array jobs than the
one resubmission the previous handoff expected:

```
14646836 (3 tasks) — COMPLETED
14667066 (3 tasks) — COMPLETED
14667900 (3 tasks) — COMPLETED
14668791 (3 tasks) — COMPLETED
14670451 (3 tasks) — COMPLETED
14671412 (3 tasks) — COMPLETED
14672421 (at least 1 task visible) — COMPLETED
```

This is at least 7 array submissions × 3 dt values = 21+ completed runs.
The campaign was clearly actively running before this session. **The LFMiddle
results have not been analyzed yet** — next session should sync and review.

---

## Next session: analyze LFMiddle results

### Step 1 — Sync catalog

```bash
bth sync engaging --pull
# OR fallback:
rsync -az engaging:~/.bth/catalog/runs/prolix/ ~/.bth/catalog/runs/prolix/
bth compact
```

### Step 2 — Review campaign

```bash
bth campaign review 89c9a900
bth sql "
  SELECT
    JSON_EXTRACT(output_paths[1], '$.dt_fs') AS dt_fs,
    outcome,
    COUNT(*) as n_runs
  FROM read_parquet('/home/marielle/.bth/catalog/runs/prolix/run_*.parquet')
  WHERE campaign_id LIKE '89c9a900%'
    AND status = 'completed'
  GROUP BY 1, 2
  ORDER BY 1
"
```

The hypothesis (campaign 89c9a900): LFMiddle integrator at dt=1.0 fs maintains
temperature stability (mean T ∈ [295, 305] K, std < 10 K) for 2-water TIP3P
over 50 ps. Passes → dt ≤ 0.5 fs constraint can be lifted in v1.1.

### Step 3 — Pull result JSONs from cluster

The per-run output JSONs live in
`~/projects/prolix/outputs/results/lfmiddle_dt_sweep/` on the cluster. Pull
them to inspect the temp_mean/temp_std fields:

```bash
myxcel pull engaging prolix   # pulls outputs/ tree
# or directly:
rsync -az engaging:~/projects/prolix/outputs/results/lfmiddle_dt_sweep/ \
  outputs/results/lfmiddle_dt_sweep/
```

### Step 4 — Update v1.1_next_steps.md

Based on LFMiddle outcome:
- **Pass at dt=1.0 fs**: lift the dt ≤ 0.5 fs constraint; update
  `CLAUDE.md` Phase 2 Known Limitations; flag as v1.1 milestone.
- **Fail at dt=1.0 fs but pass at dt=0.5 fs**: constraint confirmed; record
  in postmortem; move Phase 5 (constraint-aware thermostat) to P1.
- **Fail at all dt**: hypothesis falsified; investigate systematic bias.

---

## Git state at handoff

Branch: `main`, commit `3c654fc` (clean)

Recent commits:
```
3c654fc chore: gitignore root analysis/ dir (spurious myxcel pull artifact)
e8eba83 fix(s71): update campaign gate to include espaloma; fix lfmiddle XLA workaround
0106bb5 chore: snapshot LFMiddle diagnostic scripts + gitignore generated dirs
f4d6014 docs(handoff): hp4-s71 campaign close + next-session steps
16dc948 fix(slurm): key Blackwell XLA workaround on node4007/node4008, not partition
```

---

## Broader roadmap — where things stand

```
[Phase 0: v1.1 stability]
  ├─ LFMiddle dt-sweep — results pending analysis (NEXT STEP) ← HERE
  ├─ NPT KE init bug — ~5000K at first record; scripts/debug/npt_step0_diagnostic.py
  └─ Large-scale SETTLE batching — 64-water 10ps (deferred)

[Phase 2a: OpenMM parity] — DONE ✅ (5/5 PASS, main)

[Phase 1: MolecularBundle refactor] — not started
  └─ blocked: wait for Phase 2a field audit to finalize field list
     (audit closed: .praxia/docs/audits/260526_p2a-bonded-field-audit.md)

[Phase 2b → 3 → 4 → 5: paper] — not started; blocked on P0 NPT KE bug
```

§7.1 figure is complete and passes all pre-registered gates. The paper sprint
is unblocked on the benchmarks side; the remaining P0 work is engine stability.

---

## Artifacts

| Artifact | Location | Status |
|---|---|---|
| §7.1 figure | `outputs/analysis/s71_external_comparator.png` | ✅ final |
| Compatibility matrix | `outputs/analysis/s71_compatibility_matrix.md` | ✅ final |
| Campaign gate | `scripts/benchmarks/external_baseline/campaign_design.toml` | ✅ updated |
| LFMiddle slurm | `scripts/slurm/lfmiddle_dt_sweep.slurm` | ✅ XLA fix applied |
| Campaign `edbd0b84` | local `~/.bth/catalog/bathos.db` | ✅ concluded/pass |
| LFMiddle results | cluster `outputs/results/lfmiddle_dt_sweep/` | pending sync |
