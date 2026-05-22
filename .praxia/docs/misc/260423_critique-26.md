# Oracle Critique #26: Explicit Solvent Validation (v1.28)

**Date:** 2026-05-15
**Oracle:** Gemini CLI (Oracle Sub-agent)
**Verdict:** CHANGES REQUESTED

## Summary
The `explicit_solvent_validation_comprehensive.md` (v1.28) has successfully integrated previous requirements (B-Spline Symmetry, OS Kernel Version, Cache Optimization, and Drift Waterfall Plots). However, Cycle 26 introduces four new axes of evaluation that are currently missing or insufficiently specific.

## 1. Computational Resource Efficiency (Part 10)
- **Finding:** The plan does **not** specify a **Grid-Interpolation Symmetry Pre-computation**.
- **Requirement:** Add a validation step for pre-computing symmetry-mapped spline weights for a subset of the grid. Verify if this optimization reduces spread/interpolate kernel execution time by ~15% for stationary or semi-stationary solute atoms.
- **Action:** Integrate this into Phase 4 (Stability & Maintenance) or as a new optimization sub-point in P1b.

## 2. Metadata Integrity (Part 11)
- **Finding:** While P1h includes "SHA-256 Checksums for Reference Binaries," it lacks an explicit requirement for the **Python Interpreter Binary Hash**.
- **Requirement:** Record the SHA-256 of the `python3` binary used for validation. Different builds of the same Python version (e.g., conda vs. system vs. pyenv) can exhibit different garbage collection behaviors and memory alignments that subtly influence JIT-compiled kernel performance and numerical reproducibility.
- **Action:** Update P1h to include "Python Interpreter Binary Hash (SHA-256 of `python3`)."

## 3. PME Implementation Specificity (Part 11)
- **Finding:** The plan mentions B-Spline Symmetry and Degree Invariance but misses **B-Spline Translation Invariance**.
- **Requirement:** Specifically verify that translating the entire system by a fractional grid vector $(\Delta x, \Delta y, \Delta z)$ produces bitwise identical energy (or within $10^{-12}$) after accounting for periodic wrapping and re-snapping. This ensures the spline interpolation is truly position-independent relative to the grid origin.
- **Action:** Add this to P1b or P1c.

## 4. Finality Evidence (Part 10)
- **Finding:** The "Validation Certificate" lists several excellent artifacts (Waterfall plots, Pressure Isotropy), but lacks the **Pressure-Virial Correlation Map**.
- **Requirement:** Include a 2D histogram (Correlation Map) showing the relationship between virial components and instantaneous pressure. This is critical for verifying "Stress-Virial" consistency, especially in sheared (triclinic) boxes or NPT ensembles.
- **Action:** Append to the "Final Artifact" list in Section 2.4.

---

## Verdict: CHANGES REQUESTED
This is the 26th critique. The plan is technically sound on legacy requirements but must address the Cycle 26 focus points to achieve Approval. Once these four items are integrated, the plan will be eligible for its first of three consecutive approvals.
