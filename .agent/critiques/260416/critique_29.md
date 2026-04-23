# Oracle Critique #29 (Cycle 29)

**Date:** 2026-04-16
**Target:** `explicit_solvent_validation_comprehensive.md` (v1.31)
**Verdict:** 🛑 **Changes Requested**

## 1. Summary of Assessment
While version 1.31 successfully integrates all previous recommendations (Instruction-Throughput Audit, Environmental Dumps, Rotational Invariance, and Autocorrelation Plots), it fails to address the new Cycle 29 focus areas regarding instruction-mix audits, metadata integrity for PRNG state, grid-boundary wrapping validation, and visual finality evidence.

## 2. Compliance with Previous Recommendations
- [x] **Kernel-Level Instruction-Throughput Audit (>70% FLOPS):** **Satisfied** in Phase 4.
- [x] **Environmental Variable Dumps (LD_LIBRARY_PATH, etc.):** **Satisfied** in P1h.
- [x] **B-Spline Rotational Invariance validation:** **Satisfied** in P1b.
- [x] **Pressure Tensor Autocorrelation Plot in Validation Certificate:** **Satisfied** in Phase 4.

## 3. New Critique Focus (Cycle 29) - Gaps Identified

### 3.1 Computational Resource Efficiency (Part 13)
- **Gap:** The plan lacks a **Kernel-Level Instruction-Mix Audit**. 
- **Required:** Specify the use of `jax.profiler` to verify that the PME spread kernel maintains a high ratio of **Fused Multiply-Add (FMA)** instructions vs. simple operations, ensuring peak ALU utilization.

### 3.2 Metadata Integrity (Part 14)
- **Gap:** **System Entropy State** is missing from the `reference_manifest.json` requirements.
- **Required:** Include the capture of `/proc/sys/kernel/random/entropy_avail` to detect potential PRNG initialization stalls or stochastic divergence in Langevin noise across different compute nodes.

### 3.3 PME Implementation Specificity (Part 14)
- **Gap:** No explicit validation for **B-Spline Grid-Boundary Wrapping**.
- **Required:** Add a requirement to verify that a particle at position $L-\epsilon$ correctly spreads its charge to grid cells $0, 1, \dots$ without any "lost density" or numerical leakage at the wrap-around boundary.

### 3.4 Finality Evidence (Part 13)
- **Gap:** The "Validation Certificate" does not include a **Force Vector Divergence Map**.
- **Required:** Mandate a 3D volume rendering showing the spatial distribution of force MAE to confirm that errors are not clustered at box boundaries or virtual site regions.

## 4. Required Action Items
1.  **Phase 4:** Add "Kernel-Level Instruction-Mix Audit (FMA ratio focus)" to the Stability & Maintenance section.
2.  **P1h:** Include "System Entropy State (`/proc/sys/kernel/random/entropy_avail`)" in the reference manifest specification.
3.  **P1b:** Add "B-Spline Grid-Boundary Wrapping (no 'lost density' at wrap-around)" to the PMEConfig & Invariance validation targets.
4.  **Phase 4 (Final Artifact):** Include "Force Vector Divergence Map (3D spatial MAE distribution)" in the Validation Certificate deliverables.
