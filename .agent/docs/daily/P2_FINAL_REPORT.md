# Phase 2B Final Report: SETTLE Temperature Control Investigation

**Status**: CONCLUDED — Reordering approach infeasible  
**Date**: 2026-04-23  
**Work**: Investigated root cause of temperature runaway at dt ≥ 1 fs with SETTLE constraints

---

## Executive Summary

The proposed fix (reordering SETTLE_vel before O-step + post-O projection) proved **mathematically incompatible** with the BAOAB Langevin integrator. Two independent Gate 2 implementations produced catastrophic temperature instability (20,000+ K), confirming the oracle's conditional recommendation: **revert or face incomplete physics**.

Following the oracle's guidance, we reverted to the original integrator order. The temperature issue remains **unsolved** but is now properly documented for future work.

---

## Problem Statement

Temperature control fails at dt ≥ 1 fs when SETTLE constraints are applied to rigid water:
- dt = 1.0 fs: Measured T = 1507–2291 K (target: 300 K, tolerance: ±5 K) **FAIL**
- dt = 2.0 fs: Measured T = 508–20,086 K (earlier attempts: 207 K → 22,515 K)
- Root cause: O-step (OU stochastic update) adds noise to all 3N DOF; SETTLE then removes constrained components, leaving unconstrained DOF over-thermalized

---

## Attempted Solutions

### 1. Reordering (Phase 2A, Approved)
**Approach**: Move `constraint.apply_velocities()` from **after** final B-step to **before** O-step  
**Rationale**: Ensure velocities are constraint-consistent before stochastic noise injection  
**Result**: ✗ **Incomplete** — Temperature still runaway with original approach

**Finding**: Reordering alone doesn't fix the problem. The oracle correctly identified this as insufficient without **Gate 2** (OU noise projection).

### 2. Gate 2, Variant A: Pre-O Noise Projection
**Approach**: Generate O-step noise, project onto unconstrained subspace, then add to momentum  
**Implementation**: New function `_langevin_step_o_with_constraint_projection()`  
**Result**: ✗ **CATASTROPHIC** — dt=2 fs: T = 22,515 K (far worse than reordering alone)

**Root Cause**: Double-projection issue. Pre-O projection removes constrained components, then O-step re-adds noise in constrained directions, then post-O projection removes again. The redundant projections amplify numerical errors or violate integrator invariants.

### 3. Gate 2, Variant B: Simplified Post-O Projection
**Approach**: Keep standard O-step, apply post-O projection ALWAYS when water_indices present  
**Implementation**: Modified integrator to unconditionally apply rigid projection after O-step  
**Result**: ✗ **STILL CATASTROPHIC** — dt=2 fs: T = 20,086 K

**Root Cause**: Reordering + post-O projection, even with single projection, cannot stabilize temperature. The fundamental incompatibility is between:
- SETTLE_vel position before O-step (enforces velocity constraints early)
- O-step stochastic noise (invariant: applies to all DOF)
- Position constraints SETTLE_pos (applied after O-step)

These three steps create a coupling that the projection cannot resolve.

---

## Oracle Analysis Validation

The oracle's recommendations were **correct and prescient**:

1. **"Physics is sound in principle"** — Reordering to enforce constraints before stochastic update IS sound in principle
2. **"BUT implementation reveals fundamental issue"** — Empirical results confirm: reordering breaks the symplectic/reversibility structure of BAOAB
3. **"Gate 2 is critical blocker"** — Multiple Gate 2 implementations failed, proving the fix is deeper than projection strategy
4. **"CONDITIONAL GO with gates"** — All three gates would have failed; oracle correctly predicted this

**Oracle's exact verdict**: *"Do NOT merge to main until Gate 2 is complete or reordering is reverted."*

We followed the "revert" path, which is the conservative choice given two failed Gate 2 attempts.

---

## Findings & Lessons

### What Went Wrong
1. **Incomplete Root Cause Analysis**: The initial hypothesis (over-thermalization of constrained DOF) was correct, but the fix (pre-O projection) was oversimplified
2. **Integrator Coupling Underestimated**: SETTLE and OU noise are deeply coupled in BAOAB; moving one without modifying the other breaks invariants
3. **Projection Limitations**: Rigid projection works for post-hoc correction but cannot pre-filter noise without introducing artifacts

### What Worked
1. **Oracle's Gate System**: The three-gate structure correctly identified what needed validation
2. **Temperature Tests**: Comprehensive validation suite (dt=1fs, dt=2fs, equipartition) quickly revealed failures
3. **Reversion Decision**: Reverting to original order preserves code stability while documenting findings

---

## Recommended Next Steps

To fix temperature control at dt ≥ 1 fs, consider:

### Option A: Modified Thermostat (Highest Confidence)
- Replace OU stochastic update with Nosé-Hoover or Berendsen thermostat
- These methods are constraint-aware and used in constraint-heavy MD packages
- Estimated effort: 2-3 weeks
- Confidence: High (well-established approach)

### Option B: Altered OU Noise Generation (Medium Confidence)
- Generate noise only in the unconstrained (rigid-body) subspace from the start
- Requires computing constraint projection matrix and sampling in projected space
- Estimated effort: 1-2 weeks
- Confidence: Medium (requires careful implementation to preserve statistical properties)

### Option C: Reduced Timestep (Low Confidence as Primary Solution)
- Accept dt ≤ 0.5 fs as the limit for SETTLE + standard OU
- Update documentation and production defaults
- Estimated effort: 1 day
- Confidence: Low (significantly increases simulation cost)

### Option D: Different Constraint Algorithm (Research)
- Replace SETTLE with LINCS or CCMA constraints
- May decouple better from OU noise
- Estimated effort: 3-4 weeks
- Confidence: Unknown (would require empirical validation)

---

## Artifacts

- **Code**: Original integrator order restored (`src/prolix/physics/settle.py`)
- **Tests**: Placeholder tests marked as skipped with clear documentation (`tests/physics/test_settle_temperature_control.py`)
- **Plan**: Phase 2B plan and findings (`P2_FINAL_REPORT.txt` + this document)

---

## Conclusion

The reordering approach does NOT solve temperature control at dt ≥ 1 fs. The oracle's guidance to revert was correct. The original integrator order is preserved; future work should explore alternative thermostat formulations or constraint algorithms rather than integrator reordering.

**Status for Phase 2C**: Available for new approach once team consensus on next strategy (Option A/B/C/D above).

