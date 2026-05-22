# Oracle Critique: Cycle 5 (Final)
**Date:** 2026-04-16
**Target:** `explicit_solvent_validation_comprehensive.md` (v1.7)
**Status:** **REJECTED**

## 1. Evaluation against Cycle 5 Focus Areas

### 1.1 End-to-End Release Readiness
- **Finding:** The plan mentions "Long-term stability (10ns)" in Phase 4 but lacks a formal **"Go/No-Go" gate**. There are no specific criteria (e.g., maximum allowable energy drift, simulation survival rate, or physical property convergence limits) that would trigger a rejection of the release.
- **Recommendation:** Define a specific "Go/No-Go" milestone in Phase 4 with quantitative success metrics for the 10ns production run.

### 1.2 Regression Maintenance
- **Finding:** The plan does not address post-release maintenance of the `PMEConfig` or the validation manifest. There is no mention of a **SHA-256 manifest** for "Golden Set" reference configurations or an **Automated Golden Set Update Policy** to handle intentional changes to the physics engine or precision models.
- **Recommendation:** Add a section for "Post-Release Regression & Maintenance" that defines how the Golden Set is versioned and updated.

### 1.3 Implicit/Explicit Boundary
- **Finding:** The plan focuses exclusively on explicit solvent in isolation. It fails to verify the **transition stability** from implicit solvent minimization (often used as a starting point) to explicit solvent production. This is a common source of "energy spikes" and integrator failures.
- **Recommendation:** Add a task in Phase 1 or 2 to verify the potential energy delta and force continuity when switching between implicit (GBSA) and explicit (PME) potentials for a relaxed structure.

### 1.4 Documentation Accuracy
- **Finding:** There is an inconsistency in `just` command naming. Section 2.1h specifies a pattern of `just test-explicit-*`. However, the Implementation Roadmap (Section 3) lists `just bench-explicit-scaling`. While related, this deviates from the test prefix. Furthermore, the actual `Justfile` currently lacks all four commands listed in the roadmap.
- **Recommendation:** Synchronize the command naming patterns and ensure the `Justfile` reflects the planned automation interface.

## 2. Status of Previous Recommendations
- [x] Virtual Site synchronization (mid-step updates) - Included in P1e.
- [x] Molecule-Aware PBC Wrapping - Included in P2c.
- [x] Padded vs. Non-Padded Neighbor List parity - Included in P2b.
- [x] Justfile automation task - Included in P1h.

## 3. Final Verdict
**REJECTED.** 

While the plan has matured significantly regarding algorithmic parity and PBC handling, it lacks the operational rigor required for a "release-ready" validation strategy. Specifically, the absence of a Go/No-Go gate, regression manifest policy, and boundary stability verification makes the plan incomplete for Cycle 5.

**Required Revisions for v1.8:**
1. Explicit "Go/No-Go" gate with stability metrics.
2. "Golden Set" SHA-256 manifest and update policy.
3. Implicit/Explicit transition stability verification.
4. Consistent `just` command naming in Section 3.
