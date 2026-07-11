# Prolix Loop Priorities

**Last updated:** 2026-06-24 | **Loop task_id:** 260615_autonomous-loop | **Iteration:** 59

This file is the source of truth for the autonomous loop. Read it on every session start.

---

## Current State

```
phase:          IMPLEMENT (Sprint 59)
sprint:         Sprint 59 — HP2 NB wiring + V6 gate + backlog hygiene
gate:           sprint_approved = true (user-directed 2026-06-24)
iteration:      59 (Sprints 53–58: Claim 2 W1–W4/WB2, V1/V3–V5/V4-HLO, HP2 bonded, #2645 vmap stack)
run mode:       hitl_semi_autonomous
git HEAD:       0f0e220 (Sprint 58 #263 V1 1k parity)
```

**Validation matrix (git truth):**

| Gate | Backlog | Status |
|------|---------|--------|
| V1 | #263 | ✅ smoke + 1k parity (`test_v1_ensemble_plan_parity.py`) |
| V3 | #264 | ✅ `test_v3_homogeneous_batch_parity.py` |
| V4 | #265 | ✅ `test_v4_heterogeneous_batch_parity.py` |
| V4-HLO | #266 | ✅ `test_v4_hlo_hetero_compile_once.py` |
| V5 | #267 | ✅ `test_v5_observable_parity.py` |
| V6 | #268 | 🔄 Sprint 59 — remove stale skip, wire EnsemblePlan grad test |
| V7 | — | ✅ `tests/physics/test_batch_planner_v7.py` |
| V8 | — | ✅ `tests/batching/test_safe_map_varying_shape_spec.py` |
| W1–W4, WB2 | #275–280 | ✅ committed Sprints 51–55 |
| #2645 vmap stack | #2645 | ✅ Sprints 48–49 |

**Residual:** V1 strict solvated ala-dip + full NB on bundle path (HP2 #162 follow-up, Sprint 59 Track B).

---

## Priorities (MoSCoW)

### P1 — Paper gate

| # | Item | Blocks |
|---|------|--------|
| **#259** | §7.1 differentiable bonded fitting figure | Paper (#172) |
| **#260** | HP4 ANI-1x curation (cluster) | #259 |
| **#162** | Full FF on bundle path (`energy_fn_from_bundle`) | Strict solvated V1, export paths |

### P1 — Correctness

| # | Item |
|---|------|
| **#746** | Tiling bucketing invariant (silent atom drop) |
| **#934** | Batched SETTLE vmap divergence |

### P2 — Validation / evidence

| # | Item |
|---|------|
| **#268** | V6 jax.grad finite-diff nightly |
| **#274** | R4 AOT-ratio CI monitor |
| **#295** | S2 DHFR throughput vs OpenMM |
| **#270** | B1-full paper headline (bathos) |

### P3 — Defer

P5 R-step chain (#856–862), LB2/LB3/MTT, v2.0 refactors (#325–326), #171 WebGPU.

---

## Sprint 59 scope (active)

| Track | Backlog | Deliverable |
|-------|---------|-------------|
| A | #2645 | ✅ already shipped — confirm tests green |
| B | #162 | `energy_fn_from_bundle` + wire `EnsemblePlan.run` + NB parity test |
| C | #268 | V6 EnsemblePlan jax.grad gate (remove stale xfail) |
| D | hygiene | Backlog notes on closed items; this file + `loop_state.toml` |

---

## Hard Rules

1. **L2 gates:** `spec_confirmed` / `sprint_approved` require human sign-off unless user explicitly directs sprint execution in-thread.
2. **Reviewer gates** before closing backlog items — run reviewer_prompt VERIFY commands.
3. **Deprecation warnings:** empirically verify after `__getattr__` shims.
4. **Stage only explicit files** — ignore unrelated dirty tree.
5. **`NPTState`** from `prolix.typing`.
6. **Type checker:** `uv run ty check`, not mypy.
7. **HP4:** cluster-only; bathos campaign first.
8. **Backlog MCP:** `status` field may reject updates — use `notes` for hygiene until close workflow fixed.

---

## Key Files

| Path | Role |
|------|------|
| `src/prolix/api/bundle_md.py` | `energy_fn_from_bundle` (Sprint 59) |
| `src/prolix/api/ensemble_plan.py` | Integration entry |
| `tests/api/test_s1_jaxgrad_parity.py` | V6 / #268 |
| `tests/physics/test_hp2_bundle_factory_parity.py` | HP2 bonded gate |
| `.praxia/loop_state.toml` | FSM state |
