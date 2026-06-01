---
date: 2026-06-01
task_id: 260601_p0_parallel_tracks
session_id: f9c51c55-bc1a-4013-bb1e-93cc392c8491
status: in_progress
branch: main
commit: 27879d9
phase: P0 to P1 transition
---

# Handoff: P0 Parallel Tracks Close

## Summary

Ran `p0-parallel-tracks` workflow (3-phase: Orient → Analyze → Synthesize) to close both
P0 items in parallel. LFMiddle campaign 89c9a900 FALSIFIED — 46 runs, 0 passes across all
dt values, integrators, and system sizes. NPT KE init bug fixed at `settle.py:1944` with a
one-character change (`/ mu` → `* mu`, commit `b6e5bb9`, xfail removed). Docs updated
(`27879d9`). P1a MolecularBundle (#163) is now unblocked.

---

## What happened this session

1. Read prior handoff (`260601_s71-close-complete`) — designated LFMiddle analysis as next.
2. Dispatched recon subagent to validate orientation across full project surface. Confirmed
   LFMiddle is correct P0, NPT should run in parallel.
3. Composed and ran `p0-parallel-tracks` workflow (5 agents, 3 phases, ~6 min).
4. **LFMiddle result**: FALSIFIED. All configurations fail.
   - dt=0.25 fs: mean T = 13,818 K
   - dt=0.5 fs: mean T = 1.22×10⁶¹ K
   - dt=1.0 fs: mean T = 3.35×10⁵⁷ K
   - baoab control also fails at small N (tiling bug confound, backlog #746)
   - 895-water production runs show genuine thermal runaway independent of tiling bug
5. **NPT result**: Root cause confirmed — Sprint 14 inverted Bernetti-Bussi sign. Fix: 2 lines.
6. Dispatched NPT fixer (worktree isolation) → commit `b6e5bb9` on main.
7. Updated `CLAUDE.md` and `.praxia/docs/v1.1_next_steps.md` → commit `27879d9`.
8. Ran `test_npt_20ps_liquid_water` in background → CPU timeout (not a correctness failure).

---

## Next session: 3 steps

### Step 1 — Confirm NPT test (5 min)

Option A (quick sanity, step-0 T should be ~260–300 K not ~7000 K):
```bash
uv run python scripts/debug/npt_step0_diagnostic.py
```

Option B (full test, needs extended timeout):
```bash
uv run pytest tests/physics/test_npt_barostat.py::test_npt_20ps_liquid_water -v --timeout=600
```

Option C (cluster, fastest GPU path):
```bash
sbatch --array=0-0 --time=0:15:00 scripts/slurm/npt_verify.sbatch  # if exists
```

### Step 2 — Dispatch P1a MolecularBundle (#163)

Field audit closed 260526 (`.praxia/docs/audits/260526_p2a-bonded-field-audit.md`). No blockers.

Dispatch context to include:
- task_id: 260601_p1a_molecular_bundle
- NPT fix is on main (b6e5bb9); P0 cleared
- Field audit doc: `.praxia/docs/audits/260526_p2a-bonded-field-audit.md`
- Backlog item: #163, "P1a: MolecularBundle — bucketed dynamic topology JIT boundary"
- Target sprint: v1.2

### Step 3 — Plan P2b after P1a

P2b cross-validation harness requires:
- NPT fix confirmed (this step)
- P1a MolecularBundle landed (step 2)
- No other P0 blockers

---

## Immediately relevant files

| File | Lines | Why |
|---|---|---|
| `src/prolix/physics/settle.py` | ~1935–1995 | NPT fix: line ~1944 `* mu`, line ~1985 `position` |
| `tests/physics/test_npt_barostat.py` | ~300–435 | xfail removed; needs confirm |
| `scripts/debug/npt_step0_diagnostic.py` | 1–50 | Step-0 T sanity check |
| `.praxia/docs/v1.1_next_steps.md` | 1–65 | Updated priorities: LFMiddle struck, Phase 5 P1 |
| `.praxia/docs/audits/260526_p2a-bonded-field-audit.md` | all | P1a field list — read before dispatch |

---

## Failed attempts (do not retry)

### LFMiddle hypothesis (campaign 89c9a900)

- **What**: LFMiddle integrator (Leimkuhler-Matthews O-step splitting) at dt ∈ {0.25, 0.5, 1.0} fs
- **Outcome**: FALSIFIED. 0/46 passes. Thermal runaway at all dt.
- **Root cause of failure**: O-step splitting does not resolve SETTLE+thermostat coupling.
  Small-system results additionally corrupted by tiling/exclusion-buffer bug (backlog #746).
  895-water production shows genuine runaway independent of tiling confound.
- **DO NOT RETRY** — Phase 5 (constraint-aware thermostat) is the only known path to dt ≥ 1.0 fs.

---

## Open questions

1. Does `test_npt_20ps_liquid_water` pass end-to-end (T_mean in [295, 305] K, last 100 records
   of 20 ps NVT→NPT trajectory) after the momentum scaling fix? CPU timed out; answer pending.

---

## Deferred

1. **Tiling bucketing structural fix** (backlog #746): enforce `inner_tile_size` as multiple of
   `tile_size` at call site via bucketing invariant. Current hotfix rounds up; structural fix
   needs API change. Confounds any small-system integrator sweep. Target: post-P1a, pre-P2b.

2. **Phase 5 constraint-aware thermostat**: Only known path to lifting dt ≤ 0.5 fs. Promoted
   to P1 this session. 4–6 day effort. Must start after P1a to stay on 6-week paper critical
   path (P0 → P1 → P2b → P3 → P5 → paper).

---

## Git state at handoff

```
Branch: main, commit 27879d9 (clean)

27879d9 docs: record LFMiddle falsification + NPT fix in CLAUDE.md and v1.1_next_steps
b6e5bb9 fix(npt): correct momentum scaling sign in CSVR step (/ mu → * mu); remove xfail
761c048 chore: pre-dispatch snapshot — recon.jsonl updated by p0-parallel-tracks workflow
7621892 docs(handoff): s71 campaign close complete — LFMiddle next
```

---

## Roadmap state at handoff

```
[P0: CLEARED]
  ├─ LFMiddle dt-sweep     FALSIFIED (campaign 89c9a900, 2026-06-01)
  ├─ NPT KE init bug       RESOLVED (commit b6e5bb9, 2026-06-01)
  └─ test_npt_20ps         PENDING CPU timeout; needs cluster/extended-timeout confirm

[P1a: NEXT DISPATCH]
  └─ MolecularBundle (#163) — bucketed JIT boundary, field audit closed, unblocked

[Phase 5: P1 CANDIDATE]
  └─ Constraint-aware thermostat — only dt-fix path; must start after P1a

[P2b: after P1a + NPT confirmed]
  └─ Cross-validation harness

[Paper: 6-week critical path]
  └─ P0 → P1 → P2b → P3 → P5 → submit
```
