# Oracle Critique 65: Explicit Solvent Validation Plan

**Target:** explicit_solvent_validation_comprehensive.md (v1.67)
**Date:** 2026-04-17
**Oracle Identifier:** Cycle 65

## 1. Executive Summary
The Explicit Solvent Validation Plan (v1.67) has successfully integrated previous requirements (6D maps, Reason Status logs), but fails to satisfy the advanced diagnostic and metadata rigorousness of Cycle 65. The plan is insufficient for finality due to gaps in kernel-level constant-memory auditing, bitwise PME scaling invariance, and 7D orientation-isotropy visualization.

## 2. Axis-Specific Critique

### 2.1 Computational Resource Efficiency (Part 43)
- **Status:** FAIL
- **Finding:** The plan lacks a **Kernel-Level Constant-Memory-Bandwidth Sphericity Audit**.
- **Impact:** Without verifying bandwidth isotropy across SMs for the constant-cache broadcast mechanism, the PME influence function may suffer from localized SM-contention that eludes standard hit-rate audits.
- **Requirement:** Explicitly mandate `jax.profiler` verification of isotropic constant-memory bandwidth (5% tolerance) for PME influence function evaluations.

### 2.2 Metadata Integrity (Part 50)
- **Status:** FAIL
- **Finding:** The plan lacks **System GPU-Memory ECC Status Verification**.
- **Impact:** "Silent Corrections" (single-bit flips corrected by ECC) can introduce subtle timing jitters or energy fluctuations that aren't logged as errors.
- **Requirement:** Mandate `nvidia-smi -q -d ECC` differential checks (pre vs. post) in the reference manifest.

### 2.3 PME Implementation Specificity (Part 50)
- **Status:** FAIL
- **Finding:** The plan lacks **B-Spline Grid-Summing Invariance to Particle Charge Magnitude**.
- **Requirement:** Verify bitwise identity (1e-12) of grid weights for $2q$ vs. $q$ to ensure magnitude scaling occurs post-interpolation.

### 2.4 Finality Evidence (Part 46)
- **Status:** FAIL
- **Finding:** The certificate specifies a 6D map instead of the required **Force Vector Divergence Kurtosis Sphericity Temporal PSD Volume Temporal Sphericity Map** (7D).
- **Requirement:** Upgrade the visualization to 7D (adding orientation-isotropy as the 7th dimension).

## 3. Previous Recommendations Audit
- [x] Kernel-Level Instruction-Throughput Sphericity Audit (5% isotropy): **Present in P4.**
- [x] System GPU-Compute Mode Reason Status Change Log in manifest: **Present in P1h.**
- [x] B-Spline Grid-Summing Invariance to Particle Charge Sign validation: **Present in P1b.**
- [x] Force Vector Divergence Kurtosis Sphericity Temporal PSD Volume Temporal Map (6D): **Present in P4.**

## 4. Verdict
**Changes Requested.**
This plan cannot proceed to Approval 2 of 3 until the 7D map and hardware-level bandwidth/ECC audits are integrated.
