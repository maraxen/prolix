# Prolix Loop Priorities

**Last updated:** 2026-06-15 | **Loop task_id:** 260615_autonomous-loop | **Iteration:** 1

This file is the source of truth for the autonomous loop. Read it on every session start.
It overrides any stale context you have from prior sessions.

---

## Current State

```
phase:          SPRINT_COMPOSE / gate_pending
sprint:         Sprint 41 (.praxia/sprint_plans/260615_sprint41.toml)
gate:           sprint_approved = false  ← user must flip this to proceed
iteration:      1 (Sprint 40 complete: #250 #289 #257 closed)
run mode:       hitl_semi_autonomous (human available)
```

**Immediate action on session start:** Check `loop_state.toml`.
- If `sprint_approved = false` → present Sprint 41 summary and wait for approval.
- If `sprint_approved = true` → dispatch Track A (fixer, worktree) + Track B (librarian, 6-item research batch) in parallel.

---

## Priorities

Listed in paper-critical-path order. Always pick the highest executable item.

### Tier 0 — Paper gate (do not skip these)

| # | Item | Blocks |
|---|------|--------|
| #283 | EnsemblePlan.from_bundles + .from_bundle + .run | V1–V7 tests, #259 §7.1 figure |
| #260 | HP4 ANI-1x curation (cluster-scale, ~5GB) | #259 §7.1 figure |
| #259 | §7.1 differentiable bonded-parameter fitting figure | Paper (#172) |

### Tier 1 — Research batch (quick, parallel-safe)

| # | Item |
|---|------|
| #296 | Plan-then-execute API design patterns |
| #297 | PDB/CASP/AlphaFold-DB size distributions → bucket ladder |
| #298 | IREE-WASM ecosystem status 2026 |
| #299 | Browser MD precedent (NGL, Mol*, WebGPU) |
| #300 | Declarative scientific API surveys |
| #301 | Differentiable FF fitting prior art |
| #304 | Venue fit scan (JCP/JCTC/JCIM/JOSS/ICML/ICLR/MLSys) |
| #305 | Recent MD-engine papers — reviewer tropes |

### Tier 2 — Implementation (unblock V-series after #283)

| # | Item | Depends on |
|---|------|-----------|
| #271 | B3 memory scaling benchmark | — |
| #272 | B4 hetero-batch vs naive-pad + bucket sweep | — |
| #273 | B5 BatchPlanner decision quality | — |
| #295 | S2 DHFR throughput parity vs OpenMM | #160 |
| #294 | S1 D3 AUTOGRAD vs ANALYTICAL throughput | — |

### Tier 3 — After #283 + #260 both done (V-series gates)

V1–V7 (#263–#269), B1-full (#270), R4 CI monitor (#274), Claim 2 W-series (#275–#278)

---

## Hard Rules

### Semantic execution gates (never skip)

1. **L2 gates are human-gated.** `spec_confirmed` and `sprint_approved` in `loop_state.toml` require human sign-off. Do not self-approve. Present the TOML summary, then wait.

2. **All reviewer gates must pass before closing a backlog item.** A fixer saying "done" is not enough. Run the gate commands from the reviewer_prompt in the sprint TOML and confirm each one passes.

3. **Verify deprecation warnings empirically.** After any `__getattr__`-based DeprecationWarning change, run `python -c 'import <module> as m; m.<Name>'` and confirm the warning appears in stderr. Assert `'<Name>' not in m.__dict__`. (Lesson #141 — fixer shipped this wrong in Sprint 40.)

4. **Do not commit pre-existing dirty-tree files.** The main checkout has ~50 modified `src/prolix/` files that pre-exist this loop. These are NOT this session's changes. Stage only explicitly changed files by name.

5. **NPTState lives in `prolix.typing`, not `prolix.physics.simulate`.** Any new code or test that imports `NPTState` must use `from prolix.typing import NPTState`. (Fixed in Sprint 40 — do not regress.)

6. **Type checker is `ty`, not `mypy`.** Run `uv run ty check src/prolix/api/` (or targeted subdir). Never add mypy back. Two known false positives are suppressed in `[tool.ty.rules]` in pyproject.toml (`unresolved-import`, `empty-body`).

### Convergence guardrails

- **Stall detection:** If `loop.fingerprint.recent_tool_hashes` shows 3+ identical consecutive hashes, the loop is stuck. Stop and report to user.
- **Zero-progress limit:** `max_zero_progress = 3`. After 3 consecutive iterations with 0 items closed, escalate to user.
- **Remediate cap:** `remediate_attempt` ≥ 3 on the same item → escalate to user, do not keep retrying.
- **HP4 is cluster-scale.** Do not attempt to run ANI-1x curation locally. It requires: `bth campaign create` → cluster submission via `sbatch` → `bth sync` from engaging. Budget at least 2 sessions for it.

---

## Composition-Engine Execution Path

For each sprint track that uses the composition engine:

```
1. praxia dw compose-sprint --items <ids> --sprint-id YYMMDD_<slug>
   → writes .praxia/sprint_plans/<sprint_id>.toml

2. Edit the TOML: set workflow_template per track, fill fixer_prompt + reviewer_prompt

3. praxia dw emit --all
   → writes .claude/workflows/<sprint_id>_<track>.js

4. Run the emitted .js workflow (via /workflow or Workflow tool)
   → fan-out: fixer dispatch per track (worktree-isolated for code tracks)

5. Auditor over the diff
   → reviewer_prompt gates; verdict: advance | needs_work

6. On advance: close backlog items, update loop_state.toml, compose next sprint
```

**Currently emittable templates:** `bugfix_simple`, `spec_driven_dev`, `refactor_with_audit`

**Drift check (CI gate):** `praxia dw emit --check` — run before PR if TOML was edited.

---

## Deferred Items (do not pick up without user direction)

| Item | Why deferred | Resume condition |
|------|-------------|-----------------|
| #260 HP4 ANI-1x | Extended, cluster-scale, ~5GB download | Dedicated session; create bathos campaign first |
| #319 §9.1 code health | Backlog description references mypy (stale); ty already passes | Update backlog item description first |
| ~50 unstaged src/prolix/ modifications | Pre-existing dirty tree, unknown origin | Investigate before committing |
| #171 P4b WebGPU | P3 priority | After paper gates clear |

---

## Key File Locations

| File | Purpose |
|------|---------|
| `.praxia/loop_state.toml` | Persistent FSM state; read on every session start |
| `.praxia/loop_priorities.md` | This file — source of truth for priorities + hard rules |
| `.praxia/sprint_plans/260615_sprint41.toml` | Sprint 41 TOML (pending approval) |
| `.praxia/docs/superpowers/specs/260615_hp1-migration-policy.md` | HP1 deprecation policy spec |
| `.praxia/docs/superpowers/specs/260615_hp4-ani1x-subset.md` | HP4 ANI-1x curation spec |
| `.praxia/docs/research/260615_rr1-differentiation-evidence.md` | RR1 rebuttal (kUPS/OpenMM/GROMACS vs prolix) |
| `src/prolix/api/ensemble_plan.py` | EnsemblePlan v1.0 (Track A target) |
| `src/prolix/types/bundles.py:66-113` | MolecularShapeSpec — bucketed JIT key invariant |
| `src/prolix/tiling/planner.py:55-63` | AxisSpec.heterogeneous — heterogeneous axis demotion |
| `CHANGELOG.md` | HP1 migration table added Sprint 40 |
| `tests/api/test_hp1_migration_smoke.py` | HP1 deprecation smoke test (4 tests, all PASS) |
