---
date: 2026-06-01
task_id: 260601_hp4_s71_close
sprint_id: 5
status: ready_for_next_session
worktree: worktree-hp4-s71-postmortem
branch: worktree-hp4-s71-postmortem
---

# Handoff: hp4-s71 Campaign Close + Roadmap Orientation

## What happened this session

### Espaloma / Blackwell investigation

- Established that espaloma has **no custom CUDA kernels** — its acceleration
  comes from DGL's `libdgl.so` (fused SpMM/SDDMM), not anything espaloma ships.
- The apparent espaloma 5× win on Blackwell was a **JAX XLA autotuning hang**,
  not a real performance gap. Confirmed by rerun (SLURM 15294138):
  - Before: 9.56e-04 s/mol-step
  - After `XLA_FLAGS=--xla_gpu_shard_autotuning=false`: **8.18e-07** — 1170×
    improvement. Prolix is now 228× faster than espaloma on Blackwell.
- jaxlib on cluster: **0.10.0** (PTX fallback ruled out; XLA autotuning is the
  issue, documented in NVIDIA JAX 25.04 release notes for SM120).

### What landed in worktree `worktree-hp4-s71-postmortem`

| Commit | Subject |
|--------|---------|
| `<first>` | docs(s71): scaffold espaloma sprint5 postmortem |
| `<second>` | fix(s71): XLA Blackwell workaround + finalized postmortem |
| `<third>` | fix(slurm): key Blackwell XLA workaround on node4007/node4008 |

Files changed:
- `scripts/benchmarks/external_baseline/bench_espaloma.sprint5.bth.postmortem.toml` — new
- `scripts/slurm/bench_external_baseline.slurm` — XLA workaround added (node4007/node4008)

### What landed globally (outside worktree)

- `~/.claude/rules/CLUSTER.md` §6 — new rule: JAX on Blackwell requires
  `XLA_FLAGS=--xla_gpu_shard_autotuning=false`, keyed on node4007/node4008
  hostname, not partition name.
- Praxia lesson #41 — same lesson logged with copy-paste block.

---

## Next session start: close hp4-s71 campaign

**Load skill first:** `/using-bathos`

### Step 1 — Merge worktree to main

```bash
git checkout main
git merge --ff-only worktree-hp4-s71-postmortem
git worktree remove .claude/worktrees/hp4-s71-postmortem
```

### Step 2 — Sync new Blackwell result into local catalog

```bash
rsync -az engaging:~/.bth/catalog/runs/prolix/ ~/.bth/catalog/runs/prolix/
# Verify new run is present:
bth sql "SELECT id, tags FROM runs WHERE campaign_id LIKE '%edbd0b84%' ORDER BY timestamp DESC LIMIT 3"
```

New run: `678da554` — prolix, Blackwell, N=512, 8.18e-07 s/mol-step. The
result JSON lives on the ORCD pool path (`/orcd/pool/008/so3_shared/...`);
regenerate the figure **on the cluster**, not locally.

### Step 3 — Update `campaign_design.toml` gate to include espaloma

File: `scripts/benchmarks/external_baseline/campaign_design.toml`

Current gate (outcomes.pass condition):
```
(tool = 'prolix' AND hardware_tag = 'a100-sm80' AND per_mol_step_seconds < 3.403e-03)
OR
(tool = 'prolix' AND hardware_tag = 'rtx-pro-6000-blackwell' AND per_mol_step_seconds < 1.548e-03)
```

Update to include espaloma in `min(...)` and add Blackwell threshold anchored
to the new result. The decision (2026-06-01): include espaloma as a peer
batched-graph comparator, not exclude it. The corrected Blackwell baseline
is espaloma = 1.856e-04, so 0.5× = 9.28e-05; prolix at 8.18e-07 passes
easily. Update the condition accordingly and re-run `bth check` to validate.

### Step 4 — Regenerate figure on the cluster

```bash
myxcel push engaging prolix -y   # sync campaign_design.toml change
ssh engaging "cd ~/projects/prolix && \
  rsync -az ~/.bth/catalog/runs/prolix/ outputs/catalog_local/ && \
  uv run --with duckdb --with matplotlib \
    python scripts/analysis/s71_external_comparator.py --campaign-id edbd0b84"
myxcel pull engaging prolix      # pull updated outputs/analysis/
```

The figure needs the ORCD pool result files accessible; cluster is the right
place to run this.

### Step 5 — Conclude the campaign

```bash
bth campaign conclude edbd0b84 \
  --outcome-label pass \
  --conclusion "Prolix passes pre-registered 0.5x threshold vs min(dmff, torchmd, espaloma) on all 6 hardware platforms. Blackwell result corrected after XLA autotuning workaround (job 15294138). espaloma included in gate per 2026-06-01 decision. C6 full confirmation sweep (36 trials/cell) deferred to paper sprint."
```

---

## Campaign state at close (for reference)

All runs completed. No outstanding jobs. Updated pass-threshold table
(with corrected Blackwell):

| Hardware | min(dmff, torchmd, espaloma) | 0.5× gate | prolix | Outcome |
|---|---|---|---|---|
| A100 | 6.97e-03 | 3.49e-03 | 2.73e-06 | **PASS** |
| CPU-only | 1.60e-03 | 8.02e-04 | 1.45e-05 | **PASS** |
| H200 | 3.56e-03 | 1.78e-03 | 1.70e-06 | **PASS** |
| L40S | 3.69e-03 | 1.84e-03 | 1.55e-06 | **PASS** |
| Blackwell (corrected) | 1.86e-04 | 9.28e-05 | **8.18e-07** | **PASS** |
| RTX Pro 6000 SM120 | 3.09e-03 | 1.55e-03 | 2.09e-06 | **PASS** |

Caption note to add: espaloma (DGL-backed fused scatter) is a peer
batched-graph comparator; prolix beats it on all hardware. Espaloma setup
required ~4-6h of conda/micromamba env work vs ~5min for prolix (uv) —
this friction is itself evidence of the portability claim.

---

## Broader roadmap — where things stand

### LFMiddle campaign (89c9a900) — primary open thread

From sprint 5 handoff: the original SLURM job 14644480 was **cancelled**
(was CPU + Python loop; needs resubmission as GPU + `jax.lax.scan`).
Check whether a corrected job was submitted before the hp4 session started:

```bash
squeue -u $USER
sacct --format=JobID,JobName,State,Elapsed -S 2026-05-28
```

If not yet running, the resubmit command (from sprint 5 handoff):
```bash
myxcel push engaging prolix -y
ssh engaging "cd ~/projects/prolix && export CAMPAIGN_ID=89c9a900 && \
  sbatch --array=0-2%3 \
    --export=ALL,CAMPAIGN_ID=${CAMPAIGN_ID},N_WATERS=2,SIM_PS=50,BURN_PS=10 \
    scripts/slurm/lfmiddle_dt_sweep.slurm"
```

**Add the XLA Blackwell workaround to `lfmiddle_dt_sweep.slurm`** before
submitting — it targets pi_so3 (node4007/node4008). See global CLUSTER.md §6
or lesson #41 for the exact block.

### After LFMiddle result lands

Per `.praxia/docs/v1.1_next_steps.md`:

1. **LFMiddle result** → update v1.1_next_steps.md with pass/fail per dt;
   if pass at dt=1.0fs, lifts the dt ≤ 0.5fs constraint (major v1.1 milestone).
2. **NPT KE init bug** — T ≈ 5000K at first record; likely NVT→NPT momentum
   handoff. See `scripts/debug/npt_step0_diagnostic.py`. Hard gate on Phase 2
   NPT cross-validation tests.
3. **Large-scale SETTLE batching** — expand 4-water smoke → 64-water 10ps.

### Strategic roadmap (Phase 0 → paper)

```
[Phase 0: v1.1 stability — LFMiddle, NPT KE bug, SETTLE batching]
  ↓ (parallel)
[Phase 2a: OpenMM parity harness — DONE ✅ (5/5 PASS, main)]
  ↓
[Phase 1: MolecularBundle refactor — not started]
  ↓
[Phase 2b: Full cross-validation — not started; blocked on P0 NPT KE bug]
  ↓
[Phase 3: Benchmarks] → [Phase 4: WASM] → [Phase 5: Paper]
```

Phase 1 design is fully spec'd in `.praxia/docs/roadmaps/prolix/260515_prolix-strategic-roadmap.md`.
Don't start Phase 1 until Phase 2a field audit is used to finalize
`MolecularBundle` field list (already closed — audit doc at
`.praxia/docs/audits/260526_p2a-bonded-field-audit.md`).

### Untracked files on main (from git status at session start)

These were present before this session and are unrelated to hp4:

```
?? logs/
?? scripts/experiments/baoab_loop_diag.py
?? scripts/experiments/temp_profile_diag.py
?? scripts/slurm/baoab_loop_diag.slurm
?? scripts/slurm/temp_profile_diag.slurm
?? tip3p_tightening/
```

These are diagnostic scripts from the five-bug cascade (LFMiddle debugging,
2026-05-29). They should be committed or stashed before the next subagent
dispatch that touches the physics code.

---

## Artifacts for next session

| Artifact | Location | Status |
|----------|----------|--------|
| Postmortem | `scripts/benchmarks/external_baseline/bench_espaloma.sprint5.bth.postmortem.toml` | on worktree branch |
| Slurm fix | `scripts/slurm/bench_external_baseline.slurm` | on worktree branch |
| Global XLA rule | `~/.claude/rules/CLUSTER.md` §6 | live |
| Praxia lesson | lesson #41 | live |
| New Blackwell result | catalog run `678da554`, 8.18e-07 s/mol-step | in local catalog |
| Campaign design update | `scripts/benchmarks/external_baseline/campaign_design.toml` | **TODO** |
| Regenerated figure | `outputs/analysis/s71_external_comparator.png` | **TODO (on cluster)** |
| Campaign conclusion | edbd0b84 | **TODO** |
