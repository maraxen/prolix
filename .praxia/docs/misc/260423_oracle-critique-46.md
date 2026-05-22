# Oracle Critique 46: Explicit Solvent Validation Plan (v1.48)

**Date:** 2026-04-16  
**Verdict:** **REVISE** (Changes Requested)  
**Confidence:** High

## Strategic Assessment

The validation plan (v1.48) has matured significantly, correctly retaining the structural audits requested in prior cycles (Kernel-Level Page-Table Walk, GPU-Persistence Mode, etc.). However, it currently falls short of the rigorous multi-axis requirements for the 46th cycle. The plan demonstrates "coverage" (e.g., mentioning TLB hit rates and throttle reasons) but lacks the "depth" required for finality evidence—specifically regarding temporal evolution of force errors and hardware prefetch efficiency.

## Itemized Concerns

### 1. Computational Resource Efficiency (Critical)
*   **Issue:** Missing **Kernel-Level TLB-Prefetch Efficiency Audit**.
*   **Detail:** Section 2.4 currently specifies a *TLB-Hit Rate Audit*, but this is insufficient. We must verify that the GPU's hardware prefetcher is successfully hiding **>50% of the TLB-miss latency** for neighbor list distance-matrix accesses.
*   **Recommendation:** Add a specific mandate for a `Kernel-Level TLB-Prefetch Efficiency Audit` using `jax.profiler`.

### 2. Metadata Integrity (Critical)
*   **Issue:** Missing **System GPU-Throttling Duration** in `reference_manifest.json`.
*   **Detail:** While "System GPU-Throttle Reason State" is included (P1h), the *cumulative duration* of throttling is missing. Even short bursts of throttling can skew the distribution of wall-clock times, invalidating performance parity benchmarks.
*   **Recommendation:** Add `System GPU-Throttling Duration` to the manifest requirements in Section 2.1 (P1h).

### 3. PME Implementation Specificity (Critical)
*   **Issue:** Incomplete **B-Spline Grid-Subsampling Linearity** validation.
*   **Detail:** The plan mentions "B-Spline Grid-Summing Linearity" and "Grid-Subsampling Continuity" separately (P1b). It fails to specify the bitwise identity test for a two-particle system on a **grid N/2** compared to the sum of single-particle systems on the same grid.
*   **Recommendation:** Update Section 2.1 (P1b) to include `B-Spline Grid-Subsampling Linearity` with the explicit $N/2$ bitwise identity requirement (within $1e^{-12}$).

### 4. Finality Evidence (Critical)
*   **Issue:** Missing **Force Vector Divergence Sphericity Temporal Map** in the Validation Certificate.
*   **Detail:** The certificate (Section 2.4) includes a "Sphericity Plot" and a "Temporal Map," but not the integrated **Force Vector Divergence Sphericity Temporal Map**. This 2D histogram is critical for verifying that "Grid-Axis Biasing" does not emerge as the solvent equilibrates over the 10ns stability run.
*   **Recommendation:** Explicitly list `Force Vector Divergence Sphericity Temporal Map` as a mandatory artifact in the Validation Certificate.

## Rationale for Verdict
The plan is structurally sound but misses the specific high-order metrics required to close Cycle 46. Once these four specific items are integrated, the plan will be eligible for its first consecutive approval towards finality.

---
*Oracle Audit ID: CRIT-46-260416*
