# Revisions: Explicit Solvent Validation Plan (Cycle 1 → Cycle 2)

**Date:** 2026-04-16  
**Cycle 1 Oracle Verdict:** REVISE (3 concerns)  
**Cycle 2 Oracle Verdict:** ACCEPT WITH CONDITIONS (4 conditional items)

---

## Major Changes

### 1. P1a Solvated Protein Test (Oracle C1: P1a_Solvated_Protein_Test_Clarity)

**Cycle 1 Finding:** Vague; claimed existing test `test_solvated_openmm_explicit_parity.py` only has 2 waters, not a protein. Plan mentioned creating new test but wasn't clear about reuse vs. new artifact.

**Revision:**
- Clarified: Existing test `test_explicit_solvation_parity.py::test_energy_parity` uses **1UAO (real 52-residue protein) + 5000 TIP3P waters**
- Two-path approach:
  - **Option A (Preferred):** Use existing 1UAO test (1 week, if PDB file available)
  - **Option B (Fallback):** Create minimal Ala10 + 500 waters test (+1 week if PDB unavailable)
- **Pre-execution validation added:** Checklist to verify 1UAO.pdb exists (IT DOES ✅) + test runs locally before execution

### 2. SETTLE Scalar Path Status (Oracle C1: P1b_Scalar_SETTLE_Effort_Underestimated)

**Cycle 1 Finding:** Unclear whether `settle_rattle_langevin` is merged or still in progress; effort estimate (2–3 days) may be hiding scope.

**Revision:**
- **Verified:** SETTLE scalar path is complete (commit 6a47281)
  - Function is `settle_langevin` (not `settle_rattle_langevin` as draft said)
  - `rigid_water: bool` flag added to `SimulationSpec`
  - Dispatch wired in `simulate.py` `run_simulation()`
- **Clarification:** P1b is **purely documentation** (extract PME config, create CI gate)
  - No implementation work needed
  - Effort estimate 2–3 days is realistic

### 3. P2a Langevin Parity Metrics (Oracle C2: P2a_RNG_and_Tolerance_Strategy_Undefined)

**Cycle 1 Finding:** "Mean T within 5%" is undefined; missing: window length, platform coverage, system size, precision context.

**Revision:**
- **Added detailed metrics table** with:
  - System size: 4-charge PME solvated box (~100 atoms)
  - Duration: 10 ps (500 steps × 20 fs)
  - Sampling window: last 5 ps (steps 250–500) post-equilibration
  - Temperature tolerance: 295–305K (±1.67% at 300K target)
  - Precision: JAX x64 (float64)
  - Platform: GPU required for CI; CPU optional
  - Integrator: Prolix `settle_langevin` vs OpenMM `LangevinMiddleIntegrator`
  - Approach: **Distribution-level** (mean T, variance) NOT trajectory matching
- Eliminates RNG matching complexity; sufficient for validation

### 4. P3 Profiling Decision Thresholds (Oracle C3 from Cycle 2)

**Cycle 2 Finding:** Plan says "decision tree in gate doc" but doesn't inline thresholds. Oracle suggested inlining.

**Revision:**
- Added explicit decision tree in §3a:
  - IF scatter_time_ms / total > 15% for N ≥ 5000 → profile Morton sort
  - IF cell_list outperforms JAX-MD by >20% → consider cell-list switch
  - ELSE → keep current NL default
- Links to `spatial_sorting_profile_gate.md` for detailed rationale

### 5. Decoupling P3 & P4a (Oracle C2 from Cycle 2 — C2: P3_Profiling_Priority)

**Cycle 2 Finding:** Plan incorrectly showed P3 blocking P4a. Profiling informs optimization decisions, not automation.

**Revision:**
- **Decoupled in §3 Dependencies & Sequencing:**
  - P3 (profiling) outputs a decision (Morton needed? Yes/No)
  - P4a (SLURM + benchmarking) is **independent** — starts Week 3 regardless of P3 outcome
  - P3 decision may trigger future optimization work (Phase 6+), not immediate P4a
- Updated dependency graph and timeline

### 6. P1a CI Integration Path (Oracle C1 from Cycle 2)

**Cycle 2 Finding:** How to verify 1UAO test runs locally before committing to Option A?

**Revision:**
- Added **pre-execution validation checklist** (C1):
  - `ls -la data/pdb/1UAO.pdb` check
  - Local test run: `pytest tests/physics/test_explicit_solvation_parity.py::test_energy_parity -m openmm -xvs`
  - Go/No-Go decision upfront (not mid-execution)
  - **VERIFIED ✅:** File exists at `./data/pdb/1UAO.pdb`

### 7. P2a GPU/CPU Scope (Oracle C3 from Cycle 2)

**Cycle 2 Finding:** "Validate on both if feasible" is too vague for executors.

**Revision:**
- **Clarified:** GPU required for CI; CPU optional for local dev
- Variance bound: Accept <2% float64 platform differences
- Document any variance in test results

### 8. P5a Start Condition (Oracle C4 from Cycle 2)

**Cycle 2 Finding:** P5a seems to depend on P1a completion, but runbook drafting could start earlier.

**Revision:**
- **Updated:** P5a can draft once P1a **choice** is made (not completion)
- Can start Week 2 (parallel with P1a execution)
- Runbook example will reference 1UAO (if Option A) or Ala10 (if Option B)

### 9. Timeline Revision

**Cycle 1:** 6–7 weeks critical path  
**Cycle 2:** 3–4 weeks critical path (revised)

Changes:
- P1a reduced from 1–2 weeks to **1 week** (existing test verified)
- P1b remains 2–3 days
- P2/P3/P4 decoupled; don't extend critical path
- P5a can start Week 2 (parallel); ready faster

**New total:** 5–6 weeks elapsed (vs 6–7 projected)

### 10. Pre-Execution Checklist Addition

**New section:** "§6b Pre-Execution Validation Checklist"
- 4 items corresponding to Cycle 2 oracle conditions (C1–C4)
- De-risks before execution starts
- 1UAO.pdb availability verified ✅

---

## Alignment with Prior Approved Plans

**Verified no contradictions:**
- Phase 3/4/6/8/9 plan (prior approval) — this plan inherits & extends
- SETTLE integration plan (Cycle 3 APPROVED) — implementation confirmed complete
- Phase 6 profile gate — referenced & inlined decision thresholds

---

## Summary

All three Cycle 1 critical concerns addressed:
1. **P1a clarity:** Two-path approach with decision checklist
2. **SETTLE status:** Verified complete; P1b is documentation-only
3. **P2a metrics:** Detailed tolerance table with all context

Plus four Cycle 2 conditions resolved (pre-execution checklist added).

**Status for Cycle 3:** Ready for APPROVE verdict if oracle confirms:
- Pre-execution checklist is sufficient de-risking
- Timeline of 5–6 weeks is credible
- No new gaps introduced by revisions
