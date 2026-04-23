# Oracle Critique 69: Explicit Solvent Validation Plan (v1.72)

**Verdict:** Changes Requested (REVISE)  
**Confidence:** High  
**Date:** 2026-05-16

## Strategic Assessment
The Validation Plan for explicit solvent (v1.72) has reached an impressive level of rigor, successfully maintaining the cumulative requirements of 68 prior critique cycles. The implementation of SHA-256 hashes, GPG-signed audits, and the exhaustive list of kernel-level sphericity audits demonstrates a world-class commitment to numerical stability and parity.

However, Cycle 69 introduces a new layer of specificity that is currently absent or only partially addressed in the text. To achieve full compliance for this cycle, the plan must bridge the gap between "latency stability" and "issue-rate stability," and add critical metadata regarding the *intent* behind hardware state changes.

## Itemized Concerns

### 1. Computational Resource Efficiency (Instruction-Issue-Rate)
- **Issue:** The plan specifies a "Kernel-Level Instruction-Latency-Stability Sphericity Audit" (which was likely a previous cycle requirement), but it misses the **Kernel-Level Instruction-Issue-Rate-Stability Sphericity Audit** for the PME Reciprocal sum kernel.
- **Evidence:** Section 2.4 lists separate temporal audits for issue rate, but doesn't bundle them into the 5% stability/isotropy requirement using `jax.profiler`.
- **Recommendation:** Update Section 2.4 to explicitly include the **Kernel-Level Instruction-Issue-Rate-Stability Sphericity Audit**.

### 2. Metadata Integrity (Bus-Clock Change Reasons)
- **Issue:** The manifest requirement (P1h) includes a log for status changes but lacks a **Reason Log**.
- **Evidence:** "System GPU-Memory Bus-Clock Status Change Log" is present, but "System GPU-Memory Bus-Clock Status Change Reason Log" is not.
- **Recommendation:** Add the **Reason Log** requirement to P1h to ensure benchmark noise can be correlated with cluster-level energy policies (e.g., Slurm power-capping).

### 3. PME Implementation Specificity (Velocity Direction Invariance)
- **Issue:** The plan validates velocity magnitude and general velocity invariance but omits the **Velocity Direction** bitwise check.
- **Evidence:** Section 2.1 (P1b) mentions "B-Spline Grid-Summing Invariance to Particle Velocity" but lacks the bitwise identity requirement for $+v$ vs. $-v$ (within 1e-12).
- **Recommendation:** Explicitly add **B-Spline Grid-Summing Invariance to Particle Velocity Direction** to the P1b validation targets.

### 4. Finality Evidence (10D Pressure Tensor Map)
- **Issue:** The Pressure Tensor Kurtosis map is listed but lacks the **(10D)** designation and the unified visualization description.
- **Evidence:** Section 2.4 lists the map by name, but unlike the Force Vector counterpart, it doesn't specify the 10D axes or the resonance peak stability criteria.
- **Recommendation:** Update the Validation Certificate section to explicitly label the Pressure Tensor Kurtosis map as **(10D)** and include the description of unified visualization.

## Verdict Rationale
While the plan is 95% compliant with the cumulative requirements, the missing elements from the Cycle 69 focus are critical for the level of "rigorous multi-axis" validation expected at this stage. Once these surgical additions are made, the plan will be ready for approval.

**Approval Status:** This is NOT an approval. (Changes Requested).
