# Oracle Critique — Cycle 2: Explicit Solvent Validation Plan

**Date:** 2026-04-16  
**Verdict:** ACCEPT WITH CONDITIONS  
**Confidence:** 88%

---

## Executive Summary

The revised comprehensive explicit solvent validation plan is substantially improved from Cycle 1 and successfully addresses all three critical concerns: (1) P1a Option A vs B fork is now explicit with decision criteria, (2) SETTLE scalar path is verified as complete (commit 6a47281) with only P1b documentation remaining, and (3) P2a metrics are specific and realistic (mean T tolerance 295–305K, 5 ps window, float64 precision). The 5–6 week critical path is credible, falling from 6–7 weeks in the draft because the solvated protein test (1UAO) already exists in the codebase and can be adapted to CI gating.

However, four conditional items must be verified before execution to maintain credibility: **(1)** confirm 1UAO.pdb CI integration is straightforward (test locally), **(2)** decouple P3 profiling from P4a start (profiling informs whether Morton is needed, not whether benchmarking automation is needed), **(3)** explicitly bound P2a CPU/GPU testing scope (GPU required for CI, CPU optional; bound acceptable variance <2%), and **(4)** inline P3 decision tree thresholds from the gate doc so executors don't need external references. The plan maintains strong alignment with prior Phase 3/4/6/8/9 and SETTLE integration plans, with no contradictions found.

---

## Strengths & Improvements (vs. Cycle 1)

1. **P1a Option A/B fork is explicit.** Plan clearly states: Option A uses existing test (1 week effort, requires 1UAO.pdb available ✓), Option B creates new minimal test (adds 1 week). Decision point is upfront: verify 1UAO.pdb in repo (already confirmed in codebase). This removes vagueness that plagued Cycle 1.

2. **SETTLE status finally correct.** Scalar path is complete (commit 6a47281, function `settle_langevin`, `rigid_water` flag in `SimulationSpec`). P1b correctly identified as documentation-only (create config file + CI gate), not implementation blocker. This resolves Cycle 1 concern #2.

3. **PME regression config already exists.** Conftest.py contains `REGRESSION_EXPLICIT_PME` fixture with canonical params (alpha=0.34, grid=32, cutoff=9.0). P1b work is straightforward: extract to YAML, load in tests, add CI gate (2–3 days). Eliminates risk of silent PME parameter drift.

4. **P2a metrics are specific & credible.** Tolerance table is concrete: 4-charge PME solvated box (~100 atoms), 10 ps total duration, mean T ≤ 5% variance (295–305K target), last 5 ps window after equilibration. Uses distribution-level comparison (mean T, variance) instead of RNG matching, avoiding complexity of trajectory matching. This resolves Cycle 1 concern #3.

5. **Timeline revised downward: 5–6 weeks (vs. 6–7).** Reuse of existing test reduces P1a to 1 week. Fallback +1 week for Option B is reasonable. Critical path (P1a → P1b → P5a → P5b) = 3–4 weeks elapsed; total is 5–6 weeks because P2a (2–3 weeks, non-blocking) runs in parallel. Feasibility improved by avoiding speculative scope.

6. **Risk assessment updated.** RNG complexity mitigated by distribution-level approach; PME drift mitigated by regression config in CI; cluster delays mitigated by smoke tests being sufficient for release.

7. **Fallback paths explicit.** P1a Option B (new minimal test, Ala10 + 500 waters, ~3k atoms). P3 can close with "current NL sufficient" if profiling shows no bottleneck. P5a can proceed with draft PME params before P1a test passes.

8. **Parallelization clearly charted.** Mermaid dependency graph, P3/P4 independent, P2 alongside P1. No hidden sync points beyond P5a blocking P5b.

---

## Concerns Requiring Clarification (Medium/Low Risk)

### C1: P1a CI Integration Path Implicit ⚠️ MEDIUM

**Issue:** Plan says "add test to CI gate: weekly run with OpenMM dep" but does not specify: (1) CI workflow type (nightly, weekly cron, PR gate?), (2) OpenMM optional dep installation in CI, (3) 1UAO.pdb commit vs dynamic fetch. Could stall if CI setup is harder than expected (e.g., OpenMM build failures on GPU runners).

**Mitigation:** Before starting P1a, create brief CI checklist: (1) test locally: `pytest tests/physics/test_explicit_solvation_parity.py::test_energy_parity -m openmm`, (2) document CI trigger (e.g., "weekly cron job" or "on-demand trigger"), (3) estimate OpenMM build time. If setup adds >1 day, revise P1a to 2 weeks.

**Status:** 1UAO.pdb already in repo ✓; test file exists ✓. Only CI gating needs verification.

---

### C2: P3 Profiling Incorrectly Gated to P4a Start ⚠️ MEDIUM

**Issue:** Timeline shows "P4a (if P3 gates optimization)" implying P4a waits for P3 profiling, but spatial_sorting_profile_gate.md clearly states profiling is a *measurement tool* to decide whether Morton is *needed*, not a *prerequisite for benchmarking automation*. This creates artificial sequencing delaying P4a.

**Mitigation:** Decouple P3 and P4a: (1) P3 runs independently weeks 2–3, produces recommendation (e.g., "current NL sufficient" or "Morton estimate 2–4 weeks"). (2) P4a starts week 3 *regardless* of P3 outcome. (3) If P3 recommends Morton, that becomes separate backlog item. Rewrite Section 6 timeline to make this explicit.

**Impact:** Without decoupling, P4a could slip into weeks 4–5 unnecessarily. With decoupling, maintains 5–6 week total.

---

### C3: P2a CPU/GPU Testing Scope Vague ⚠️ MEDIUM

**Issue:** Plan says "CPU + GPU (if feasible; GPU required for CI)" but does not specify: (1) what "if feasible" means, (2) how CPU and GPU results differ (expected variance in float64?), (3) acceptance if diverge >5%. Underspecifies validation; could cause scope creep if platform differences emerge.

**Mitigation:** Add clarification to P2a metrics table: "GPU (NVIDIA) required for CI; CPU testing optional (local sandbox). Expect <2% platform variance in float64. If diverge >5%, investigate precision or scheduler differences; document as known variance in protocol."

**Impact:** Bounds interpretation, prevents false-positive failures due to legitimate platform variance.

---

### C4: P5a Start Condition Could Be Tightened (Low Priority) ℹ️ LOW

**Issue:** Plan says P5a requires "P1a + P1b complete" but P5a could start once P1a *direction* is chosen (Option A/B), using draft PME params. No account of this minor parallelization.

**Mitigation:** Optional: Document P5a can begin once P1a direction chosen, final sign-off requires P1a passing + P1b complete. Low priority since P1a/P1b = 1–2 weeks, P5a = 1 week; overlap is small.

---

## Key Verifications Passed

| Check | Status | Evidence |
|-------|--------|----------|
| 1UAO.pdb file in repo | ✓ | Found at `data/pdb/1UAO.pdb` |
| Regression PME config centralized | ✓ | `tests/physics/conftest.py`: `REGRESSION_EXPLICIT_PME` fixture |
| Test file exists | ✓ | `tests/physics/test_explicit_solvation_parity.py` with 1UAO + 5000 waters |
| SETTLE scalar path complete | ✓ | `settle_langevin` in `src/prolix/physics/settle.py`; commit 6a47281 |
| Phase 3/4/6/8/9 plan aligned | ✓ | No contradictions; current plan supersedes prior, coherent references |
| SETTLE integration plan aligned | ✓ | Current plan correctly treats SETTLE as complete, P1b as doc-only |
| Spatial sorting gate doc exists | ✓ | Decision tree and profiling workflow documented |

---

## Timeline Credibility Assessment

**Critical Path:** P1a (1 wk) → P1b (3 days overlap) → P5a (1 wk) → P5b (1 wk) = **3–4 weeks elapsed**

**Total Project Time:** 5–6 weeks (P2a/P3 run in parallel, non-blocking)

**Effort Breakdown:**
- **P1a (1 week):** Reuse existing test; verify it runs, document tolerances, add CI gate. Fallback: new minimal test (Ala10 + 500 waters) adds 1 week.
- **P1b (2–3 days):** Config file creation + CI gate. Low risk (fixture already exists).
- **P2a (2–3 weeks):** Langevin parity test with statistical metrics. Mid-risk (RNG/platform testing could slip 1 week).
- **P3 (1 week):** Profiling execution (gate doc already written). Low risk (no implementation).
- **P4a/P4b (1–2 weeks):** SLURM wiring + benchmarking schema. Low risk (templates exist).
- **P5a/P5b (2 weeks):** Runbook + release notes. Low risk (straightforward text).

**Assessment:** REALISTIC if C1–C2 are verified upfront. Contingency: +1 week if P1a CI setup is harder, or +1 week if P2a platform testing diverges >5%.

---

## Questions for Stakeholders

1. **P1a CI Integration:** Will OpenMM be installed in CI? If optional dep, will P1a test run on every PR (strict) or weekly (relaxed)? This affects quality gating vs timeline.

2. **P3/P4a Decoupling:** Do you want to decouple P3 profiling from P4a benchmarking start, or keep them sequenced? Recommend decoupling for parallelization.

3. **P2a CPU/GPU:** Is GPU testing required in CI, or sufficient for local dev? This affects scope (1–2 weeks vs 2–3 weeks).

4. **P5a Runbook Timing:** Can runbook drafting start once P1a direction chosen, or must it wait for test passing?

---

## Recommendation

**ACCEPT WITH CONDITIONS.** Resolve C1–C4 (especially C2: decouple profiling from benchmarking). If addressed, plan is execution-ready with credible 5–6 week timeline and clear fallback paths. If not addressed, risk medium timeline slip (1–2 weeks) due to misaligned expectations around CI gating and profiling coupling.

**Next Step:** Circulate this critique to team. Collect answers to four questions above. Incorporate C1–C4 clarifications into revised plan. Then begin execution with P1a/P1b/P2a/P3 in parallel.
