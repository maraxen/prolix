# Postmortem — Titanix OMM-WATER XA-SYNC + audit mid-point

**date:** 2026-07-10  
**run_id:** `dfa001bf-c376-4bfb-a1ff-cdc68d35da86`  
**bathos postmortem:** `scripts/experiments/xr_parity_omm_tip3p.py.dfa001bf-….bth.postmortem.toml` (validated)  
**next steps:** `.praxia/docs/audits/260710_xtrax-rewire_next_steps.md`

## Verdict

**Hypothesis confirmed.** Xtrax 0.4 rewire preserves OpenMM TIP3P ΔE/ΔF/T within T2 gates on Titanix.

## What worked

- Detached `screen` + free GPU + `PYTHONUNBUFFERED` for ~10 ps NVT.
- Citing JSON `gate_pass` when bathos `outcome` is stuck at `unknown`.
- Atomic commits on `audit/xtrax-rewire-xa` for hygiene without premature push.

## What failed (ops)

| Failure | Fix applied |
|---------|-------------|
| SSH-attached run SIGTERM | screen/tmux |
| nohup GPU0 SIGKILL | GPU1 + screen |
| myxcel wrong workspace + interactive push | path `/home/solab/projects` + `-y` |
| engaging `bth sync` rsync 23 | Titanix catalog rsync |
| EnsemblePlan `float(dt)` concretization | jnp conversion (landed on branch) |

## Open debts

- `debt_bathos_outcome_unknown_gate_pass`
- `debt_praxia_backlog_db_insert`
- XA-CI still blocked (default suite)

## Next action

Unblock **XA-CI** (see next-steps doc), then XA-DRIFT → XA-CLOSEOUT → TRIAGE.
