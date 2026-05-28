---
task_id: 260528_s5_paper_gate
sprint_id: 5
date: 2026-05-28
status: closed_local
cluster_status: jobs_submitted_2026-05-28
sync_tool: myxcel
---

# Sprint 5 Close: Paper Gate + v1.1 Tracks

**Source of truth for cluster coordination.** Update this file when jobs complete or campaigns change.

## Active SLURM jobs (2026-05-28)

| Job ID | Script | Notes |
|--------|--------|-------|
| **14644480** | `lfmiddle_dt_sweep.slurm` | Array 0–2 (dt 0.25 / 0.5 / 1.0 fs), campaign `89c9a900` |
| **14644483** | `validate_npt_20ps.slurm` | Diagnostic; test still xfail locally |

**Sync:** `myxcel push engaging prolix -y` (code pushed; `.git` excluded — cluster `git log` may lag).  
**Queue:** `ssh engaging 'squeue -u $USER -n plx-lfmiddle-dt,plx-npt-20ps'`  
**Pull logs/results:** `myxcel pull engaging prolix` then `bth sync engaging --pull`

## Local deliverables (committed)

| Track | Status | Artifact |
|-------|--------|----------|
| §7.1 slurm JSON paths | Done | `scripts/slurm/bench_external_baseline.slurm` (`PROLIX_ROOT` + absolute `OUT`) |
| §7.1 figure | Done (local catalog) | `outputs/analysis/s71_external_comparator.png`, `s71_compatibility_matrix.md` |
| B1 pre-registration | Done (doc only) | `.praxia/docs/specs/260528_b1-preregistration.md` |
| NPT 20 ps gate | **Not lifted** | `test_npt_20ps_liquid_water` strict xfail; handoff + `/mu` SCR parity shipped |
| LFMiddle exploration | **GO** (implementation) | `settle_lfmiddle_langevin`, `Force_Step`, bath experiment |

**Git:** use `myxcel push engaging prolix -y` before submitting (rsync; `.git` not synced).

## Bathos campaigns

| Campaign | ID | Mode | Purpose |
|----------|-----|------|---------|
| hp4-s71-external-baseline | `edbd0b84-c0e2-4d2d-a89f-5f9fe8ae3aff` | exploration | §7.1 external comparator (7/8 cells; figure regenerated locally) |
| lfmiddle-dt-sweep | `89c9a900` | exploration | LFMiddle falsification dt ∈ {0.25, 0.5, 1.0} fs |
| b1-full | *(not created)* | confirmation | Blocked until B1-full cluster spend approved |

**After cluster runs:** `bth sync engaging --pull` then `bth campaign review <id>`.

## Cluster jobs to run

### 1. LFMiddle dt-sweep (priority — full falsification)

```bash
# On engaging, from synced workspace:
cd ~/projects/prolix
export CAMPAIGN_ID=89c9a900
sbatch --array=0-2%1 \
  --export=ALL,CAMPAIGN_ID=${CAMPAIGN_ID},N_WATERS=2,SIM_PS=50,BURN_PS=10 \
  scripts/slurm/lfmiddle_dt_sweep.slurm
```

Array index → `dt_fs`: 0=0.25, 1=0.5, 2=1.0. Expect **fail** outcome on dt=1.0 if hypothesis holds; pass band ±5 K on 0.25/0.5.

**Wall time:** ~2 h/task (CPU, float64). Logs: `outputs/logs/slurm/lfmiddle_dt_sweep_<job>_<task>.out`.

### 2. NPT 20 ps gate (diagnostic — test is xfail)

Re-validates Sprint 14/15 fixes on cluster CPU. Will **xfail** until CSVR+SETTLE coupling fixed; still useful for log capture.

```bash
sbatch scripts/slurm/validate_npt_20ps.slurm
```

Logs: `outputs/logs/engaging/<date>/app/npt_20ps_<jobid>.log`.

### 3. External baseline (optional — fill missing cells only)

Only if catalog gaps after path fix. Example Blackwell prolix smoke:

```bash
sbatch --export=ALL,TOOL=prolix,N_MOLS=64,HARDWARE_TAG=rtx-pro-6000-blackwell \
  scripts/slurm/bench_external_baseline.slurm
```

JSON lands under `${PROLIX_ROOT}/outputs/results/external_baseline/` (durable).

## Post-run checklist

- [x] `myxcel push engaging prolix -y` (2026-05-28)
- [x] Submit LFMiddle array + NPT diagnostic (jobs 14644480, 14644483)
- [ ] `myxcel pull engaging prolix` when jobs complete
- [ ] `bth sync engaging --pull`
- [ ] `bth campaign review 89c9a900` — record pass/fail per `dt_fs` in this file
- [ ] Re-run figure if new external cells: `uv run python scripts/analysis/s71_external_comparator.py --campaign-id edbd0b84`
- [ ] Update `.praxia/docs/v1.1_next_steps.md` LFMiddle row with cluster verdict
- [ ] NPT: if pytest unexpectedly passes, remove xfail and open `npt-20ps-gate` bath campaign

## Known blockers

1. **NPT:** Step-1 T spike ~7×10³ K (cold or warm `momentum=` handoff). Escalate CSVR+SETTLE+KE — see `scripts/debug/npt_step0_diagnostic.py`, `.praxia/docs/npt_ke_bug_diagnosis.md`.
2. **LFMiddle:** Long CPU tests slow locally; cluster CPU + 50 ps required for bath gate. Short smokes fail ±5 K (insufficient burn).
3. **B1-full:** Pre-reg only; no cluster spend until separate sprint approval.

## Phase logs (Praxia)

- `260528_s5_paper_gate_audit.jsonl`
- `260528_npt_long_traj_plan.jsonl` / `_audit.jsonl`
- `260528_lfmiddle_hypothesis_plan.jsonl` / `_audit.jsonl`
- Recon: `260528_s5_npt_lfmiddle_recon_recon.jsonl`

## Last updated

2026-05-28 — myxcel push + SLURM 14644480 (lfmiddle array), 14644483 (npt diagnostic).
