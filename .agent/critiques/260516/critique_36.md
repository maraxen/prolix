# Oracle Critique #36: Explicit Solvent Validation Plan (v1.38)

**Date:** 2026-05-16
**Version:** 1.38
**Status:** 36th Rigorous Multi-Axis Critique
**Verdict:** CHANGES REQUESTED

## I. Audit of Previous Recommendations
The following items from previous cycles have been successfully integrated into v1.38:
- [x] **Kernel-Level Instruction-Parallelism Audit (ILP > 2.0):** Present in Section 2.4.
- [x] **System GPU-Firmware Version (nvidia-smi) in manifest:** Present in Section 2.1 (P1h).
- [x] **B-Spline Grid-Stride Invariance validation:** Present in Section 2.1 (P1b).
- [x] **Force Vector Magnitude Divergence Map in Validation Certificate:** Present in Section 2.4.

## II. Cycle 36: New Critique Focus Findings

### 1. Computational Resource Efficiency (Part 20)
- **Finding:** While the plan mandates an ILP audit, it lacks a **Kernel-Level Warps-in-Flight Audit**.
- **Requirement:** Integrate a requirement to use `jax.profiler` to verify that the PME Reciprocal sum kernel maintains **>32 active warps-in-flight per SM**. This is essential to ensure maximum latency-hiding capability on H100 architectures.

### 2. Metadata Integrity (Part 21)
- **Finding:** The reference manifest (P1h) includes GPU firmware via `SUPPORTED_CLOCKS` but omits the **System GPU-VBios Version**.
- **Requirement:** Add the output of `nvidia-smi -q -d VBIOS` to the `reference_manifest.json` requirements. Recording the VBIOS version is critical as power-management and thermal-throttling curves at this level can introduce non-deterministic frequency scaling, complicating performance parity audits.

### 3. PME Implementation Specificity (Part 21)
- **Finding:** Section 2.1 (P1b) addresses "Boundary Wrapping" but does not specify **B-Spline Grid-Boundary Extrapolation Invariance**.
- **Requirement:** Add a specific validation test for grid-boundary extrapolation. It must verify that a particle placed slightly outside the box (e.g., at $L+\epsilon$) correctly wraps to position $\epsilon$ and produces **bitwise identical** grid weights. This ensures the numerical robustness of the wrapping logic at the grid boundaries.

### 4. Finality Evidence (Part 20)
- **Finding:** The "Validation Certificate" (Section 2.4) lacks a **Pressure Tensor Covariance Matrix Map**.
- **Requirement:** Include a 2D color-plot of the covariance between all nine pressure tensor components. This artifact is necessary to verify that off-diagonal elements are statistically zero in an isotropic fluid, confirming the physical validity of the pressure calculation.

## III. Conclusion
The validation plan continues to mature, but the omissions in warp-level efficiency, low-level hardware metadata, and statistical finality evidence must be addressed.

**Verdict: CHANGES REQUESTED.**
