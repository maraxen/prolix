# Oracle Critique #21: Explicit Solvent Validation Plan (v1.23)

**Date:** 2025-02-11  
**Verdict:** **REVISE** (Changes Requested)  
**Confidence:** High  
**Approved for Execution:** No

## Strategic Assessment
The current draft (v1.23) demonstrates strong continuity by maintaining previous mandates such as Grid-Size Snapping Efficiency and Git Submodule Hashing. However, it fails to evolve into the "Cycle 21" requirements, specifically regarding pipeline resiliency and deep physical auditing of the PME grid. The truncation of Phase 3 also prevents a comprehensive review of the statistical parity axis.

## Multi-Axis Analysis

### 1. Validation Pipeline Resiliency
*   **Status:** FAIL
*   **Issue:** No **Rollback Protocol** exists for the "Golden Set". The manifest-driven validation is only as strong as its latest commit; if a GPG-signed update introduces a corrupt reference, the system lacks a defined path to revert the Git LFS state safely.
*   **Recommendation:** Define a protocol in P1h for tag-based snapshots.

### 2. Platform-Specific Optimization (FP16/BF16)
*   **Status:** FAIL
*   **Issue:** The plan is silent on the **Mixed Precision Boundary**. With H100 migrations imminent, the project needs accepted wide-tolerances for `bfloat16` PME reciprocal sums to avoid "parity hell" where numerically valid but bit-divergent results block progress.
*   **Recommendation:** Define wide-tolerances (e.g., 5e-3) for mixed-precision reciprocal sums in P1b.

### 3. Virial/Pressure Precision
*   **Status:** FAIL
*   **Issue:** Missing the **Neutralizing Background Charge** virial contribution in P2c. This is a common source of pressure divergence in solvated protein systems that carry a net charge.
*   **Recommendation:** Explicitly add background charge virial validation to P2c.

### 4. Physicist HITL Checklist
*   **Status:** FAIL
*   **Issue:** P3f (Visual Gallery) is too vague. It needs to mandate the auditing of **Charge Density Grid Maps** (2D slices) to verify B-Spline interpolation quality.
*   **Recommendation:** Add a Physicist HITL requirement to audit grid slices.

## Itemized Concerns

| Area | Severity | Issue | Recommendation |
| :--- | :--- | :--- | :--- |
| **Resiliency** | Critical | Missing Rollback Protocol for Golden Set. | Add snapshot/revert steps to P1h. |
| **Precision** | Critical | Missing Background Charge Virial in P2c. | Add explicit background charge virial check. |
| **Optimization**| Warning | Undefined FP16/BF16 parity boundaries. | Define wide-tolerances for mixed precision. |
| **Completeness**| Warning | Phase 3 is truncated ("..."). | Complete the P3x targets definition. |
| **Audit** | Suggestion | Missing Charge Density Grid Map visual audit. | Update P3f to require 2D slice reviews. |

## Rationale for Verdict
While the plan is structurally sound for basic parity, it lacks the "rigorous multi-axis" depth required for Cycle 21. Specifically, the lack of background charge virial and rollback protocols represents significant operational risks.

---
*Oracle (Subagent)*
