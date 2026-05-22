# Phase 2 Completion & Next Steps — Adaptive RATTLE Convergence

**Status**: ✅ MERGED to main (commits 5c23554, 78d2594)  
**Date**: 2026-04-23  
**Quality Score**: 8/10 (Auditor APPROVED FOR MERGE)

---

## What Was Completed

Phase 2 implementation of adaptive RATTLE convergence in `settle_velocities()`:

1. ✅ Added `_apply_rattle_velocity_correction_with_residual()` helper (returns `(vel, residual)` tuple)
2. ✅ Extended `settle_velocities()` with `adaptive_tol: float | None = None` parameter
3. ✅ Implemented `jax.lax.while_loop` convergence checking (exits when `residual < adaptive_tol`)
4. ✅ Threaded `settle_velocity_tol` through `_langevin_settle_vel()` and `settle_langevin()`
5. ✅ Exposed in `SimulationSpec` and `run_simulation()` for user configuration
6. ✅ All regression tests pass (2/2, geometry constraints satisfied at 0.000000 Å)
7. ✅ Backward compatibility preserved (default `adaptive_tol=None` uses original fori_loop)

---

## Recommended Next Steps

### Immediate (Before Production Use)

#### **P1: Add Adaptive RATTLE Convergence Test** [HIGH PRIORITY]

**Why**: The adaptive_tol feature path is completely untested. Default behavior (fori_loop) is validated; early-exit convergence is not.

**What to do**:
1. Create `tests/physics/test_settle_adaptive_rattle.py`
2. Write pytest `test_settle_adaptive_tol_convergence()`:
   - Initialize a 64-water system
   - Run 500 BAOAB Langevin steps at dt=1.0 fs with `settle_velocity_tol=1e-6`
   - Assert: (a) constraint residual converges, (b) iteration count < 10, (c) final velocities satisfy constraints
   - Repeat for dt=2.0 fs
3. Write pytest `test_settle_adaptive_tol_float32()`:
   - Verify 1e-6 tolerance works safely in float32 precision (floor ~1e-7)
   - Confirm no premature convergence from dtype artifacts

**Estimated effort**: 30 min  
**Owner**: Recommend fixer agent

---

#### **P2: Validate Temperature Fix (Oracle Gate)**

**Why**: The root cause (iteration count vs DOF denominator vs OU projection) is not yet empirically confirmed. The adaptive RATTLE implementation is a good fix IF iteration count is the culprit; if not, the G4 temperature gate remains failed.

**What to do**:
1. Write a standalone script `scripts/validate_adaptive_rattle_temperature.py`:
   ```python
   # Run Langevin at dt=1.0 fs and dt=2.0 fs with adaptive_tol=1e-6
   # Measure temperature using corrected DOF count:
   # T = 2·KE / (k_B · (3·N_total − 3·N_water − 3))
   # Assert: ΔT < 5 K at both timesteps (G4 threshold)
   ```
2. Run the validation script
3. **If temperature improves (ΔT < 5 K)**: Root cause confirmed (iteration count). Close the G4 gate. Document temperature fix in commit.
4. **If temperature still fails**: Root cause is NOT iteration count. Escalate to oracle for DOF denominator or OU projection investigation.

**Estimated effort**: 1-2 hours  
**Owner**: Recommend oracle + general-purpose agent for investigation

---

### Short-term (Next Sprint)

#### **P3: Documentation Improvements**

1. **Docstring for `_apply_rattle_velocity_correction_with_residual()`**:
   - Add one-line clarification: "Residual is max(|bond velocity|), not constraint error. Used to detect convergence."

2. **Comment on Python-level branching (line 470 of settle.py)**:
   - Add: "This Python-level `if` is compile-time resolved; the default fori_loop path incurs no overhead."

3. **Note on non-differentiability in `settle_langevin()` docstring**:
   - Add: "The while_loop path (adaptive_tol not None) is not reverse-mode differentiable. Use adaptive_tol=None if computing gradients."

**Estimated effort**: 15 min

---

#### **P4: Production Tuning**

1. **Default tolerance recommendation**: Should `settle_velocity_tol=1e-6` be the new default in `SimulationSpec`?
   - Current: default is `None` (fixed n_iters=10)
   - Proposal: Change default to `1e-6` if temperature validation (P2) passes
   - Trade-off: Adaptive mode prevents unrolling in XLA but speeds up short-timestep runs (dt ≤ 1.0 fs)

2. **Expose max_iters tuning**: Currently hard-capped at `n_iters=10` in the while_loop.
   - Consider adding `settle_velocity_max_iters: int = 10` field to `SimulationSpec` for users to override safety cap

**Estimated effort**: Design 30 min, implementation 1-2 hours

---

### Validation Checklist

- [ ] P1: Adaptive convergence pytest passes (iteration count < 10 at fine timesteps)
- [ ] P2: Temperature validation script confirms ΔT < 5 K at dt=1.0 fs and dt=2.0 fs with `adaptive_tol=1e-6`
- [ ] P3: Docstrings updated (residual metric, non-differentiability, compile-time branching)
- [ ] P4: Decision made on default tolerance and max_iters exposure
- [ ] G4 temperature gate closed (if P2 passes)

---

## Known Issues & Risks

### Non-Blocking (Accepted)

1. **Feature-specific test coverage**: Default path is tested (regression tests); adaptive path is untested until P1 is done.
2. **Float32 safety**: Auditor flagged 1e-6 tolerance choice for float32 machine epsilon. P1 test will validate.
3. **Root cause uncertainty**: Oracle flagged that iteration count is a plausible but unconfirmed hypothesis. P2 will determine if this is correct.

### Blocked on Oracle Decision

- **Temperature gate closure**: Requires P2 validation showing ΔT < 5 K at both dt=1.0 fs and dt=2.0 fs
- **Default mode recommendation**: Auditor recommends shipping adaptive_tol=None (current) until temperature is confirmed; oracle to recommend flipping default if speedup is worthwhile

---

## Commits Merged

```
78d2594 docs(daily): update 260415 sprint log with Phase 2 adaptive RATTLE implementation
5c23554 feat(settle): add _apply_rattle_velocity_correction_with_residual helper
```

**Test results**: 2/2 passing (test_settle_water_indices, test_settle_preserves_water_geometry)  
**Backward compat**: ✅ Preserved (default adaptive_tol=None uses original fori_loop)  
**Code review**: ✅ PASS (Auditor score 8/10)

---

## Timeline Estimate

| Step | Priority | Effort | Owner | Deadline |
|------|----------|--------|-------|----------|
| P1: Adaptive test | HIGH | 30 min | Fixer | Before next deploy |
| P2: Temperature validation | CRITICAL | 1-2 hrs | Oracle/General-purpose | Blocks G4 gate |
| P3: Documentation | MEDIUM | 15 min | Fixer | Before next PR |
| P4: Production tuning | LOW | 2 hrs | Staff/Oracle | Next sprint |

---

## Decision Points for User

1. **Proceed to P1 immediately?** (Recommended: Yes, unlocks feature confidence)
2. **Proceed to P2 immediately?** (Recommended: Yes, critical for temperature gate)
3. **Change default from None → 1e-6 after validation?** (Decision pending oracle review of speedup/overhead trade-off)

---

**Next Session**: Review temperature validation results (P2) and decide on temperature gate closure + default mode.
