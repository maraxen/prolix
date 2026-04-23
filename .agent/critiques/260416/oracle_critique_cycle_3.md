# Oracle Critique Report: Cycle 3
**Target:** `explicit_solvent_validation_comprehensive.md` (v1.5)
**Date:** 2026-04-16
**Status:** Changes Requested

## 1. Previous Recommendations Audit
The following items from Cycle 2 have been successfully integrated:
- [x] **Numerical Precision:** Explicitly targets `float64` for anchor parity (P1a).
- [x] **Truncated Octahedron:** Identified as a specific target for triclinic cells (P1c, P2c).
- [x] **SHA-256 Manifest:** Integrated into the validation roadmap (P1g).
- [x] **PMEConfig Integration:** Single source of truth for PME parameters established (P1b).

## 2. Cycle 3 Critical Analysis

### 2.1 Constraint Solver Rigor
- **Observation:** The plan mentions SETTLE for water models but ignores RATTLE for the solute (protein).
- **Gap:** There is no distinction between the validation paths for analytic SETTLE vs. iterative RATTLE. More critically, the **intersection** where both solvers must coexist (constrained protein in constrained water) is a high-risk integration point with no dedicated test case.
- **Requirement:** Add a target (P2d) for "Constraint Solver Intersection" validating combined SETTLE/RATTLE stability and distance convergence in solvated systems.

### 2.2 Long-Range Corrections (Virial/Pressure)
- **Observation:** Energy and force parity are well-defined, but pressure/virial parity is treated as a secondary statistical metric (P3c).
- **Gap:** Validating the reciprocal-space part of the virial in **triclinic boxes** is notoriously difficult and prone to implementation errors (e.g., coordinate wrapping artifacts). Relying on long-term pressure averages is insufficient for debugging.
- **Requirement:** Add a specific target (P2e) for "Reciprocal Virial Parity" comparing the PME virial tensor directly against OpenMM for a single configuration in a Truncated Octahedron cell.

### 2.3 Performance/Accuracy Trade-off (Grid Snapping)
- **Observation:** Standardizing grid spacing to ~1.0Å is mentioned (Risk Assessment).
- **Gap:** "Grid Snapping" (rounding grid dimensions to optimal FFT sizes like $2^a \cdot 3^b \cdot 5^c$) is not addressed. If `prolix` and OpenMM snap to different dimensions for the same box, parity will fail.
- **Requirement:** The `PMEConfig` strategy must explicitly define a "Grid Snapping Policy" to ensure both engines use identical grid dimensions across all test cases.

### 2.4 Reproducibility (Stochastic Noise & Seeds)
- **Observation:** BAOAB and KS tests are included (P3a, P3d).
- **Gap:** The plan lacks a protocol for handling stochastic noise in "Bit-wise" vs "Statistical" parity. For the KS test, it is unclear if `prolix` will use an ensemble of trajectories with fixed seeds to ensure the *test itself* is deterministic.
- **Requirement:** Define a "Deterministic Validation Protocol" (P3e) enforcing fixed PRNG seeds across both engines for dynamic tests, enabling deterministic failure modes in CI.

## 3. Verdict
**Changes Requested.**
The plan has matured significantly in its structural alignment, but requires more surgical precision regarding constraint intersection, virial validation, and grid-snapping deterministicity before it can be considered "Production Ready" for the explicit solvent milestone.
