# 2026-04-24: v0.3.0 Phase 2 Kickoff

**Status**: Phase 2 Validation Infrastructure Complete  
**Date**: 2026-04-24  
**Next Action**: Run validation script  

---

## What Just Completed

✅ **v1.0.0 Release**: Published with Phase 1 (implicit solvent) + Phase 2 (SETTLE + Langevin at dt≤0.5fs)
✅ **v0.3.0 Phase 1**: Theory complete, auditor approved (8.5/10 quality), all oracle conditions met
✅ **Librarian Research**: 25 sources reviewed, constraint-aware approach confirmed mathematically sound
✅ **Phase 2 Validation Infrastructure**: 
   - Specification document (PHASE2_VALIDATION_SPEC.md)
   - Validation script (validate_constraint_aware_langevin.py)
   - Test configurations for dt=0.5fs, 1.0fs, 2.0fs

---

## What v0.3.0 Phase 2 Does

**Goal**: Validate constraint-aware Langevin thermostat achieves stable temperature at dt=1.0fs

**Configuration**:
```
System:       4 TIP3P water molecules (12 atoms)
Duration:     100 ps (50,000 steps)
Timestep:     1.0 fs (2x faster than v1.0's 0.5fs)
Thermostat:   Langevin with noise in 6D rigid-body subspace only
Integration:  BAOAB + SETTLE with project_ou_momentum_rigid=True
```

**Primary Gates** (all must pass to proceed to Phase 3):
1. Temperature: |T_mean - 300| < 5 K (stable control)
2. Energy: drift < 1% (accurate integration)
3. Equipartition: KS p-value > 0.05 (correct noise covariance)

---

## How to Run

```bash
# Navigate to prolix directory
cd /home/marielle/projects/prolix

# Run validation script (will test dt=0.5fs, 1.0fs, and optionally 2.0fs)
uv run scripts/validate_constraint_aware_langevin.py

# Expected runtime: 4-6 hours (wall time for 250+ ps of SETTLE MD)
# Expected output: Temperature control metrics, energy conservation, equipartition test
```

**Output will tell you**:
- ✅ **PASS**: dt=1.0fs stable → proceed to Phase 3 (edge cases, release)
- ❌ **FAIL**: Temperature/energy unstable → escalate per insurance plan (accept v1.0 constraint or investigate)

---

## What Each Test Measures

### Test 1: dt=0.5fs Baseline (Sanity Check)
- **Why**: Establish known-good behavior from v1.0
- **Expected**: T=300±5K (tight, already proven)
- **If fails**: Test infrastructure is broken; debug before continuing

### Test 2: dt=1.0fs PRIMARY GATE (v0.3.0 Goal)
- **Why**: Validate 2x speedup is stable with constraint-aware thermostat
- **Expected**: T=300±5K (loose tolerance acceptable for extended dt)
- **Critical**: This determines v0.3.0 viability

### Test 3: dt=2.0fs Stretch Goal (Bonus)
- **Why**: Explore maximum stable timestep
- **Expected**: T=300±10K (looser tolerance at 4x dt)
- **Run only if**: dt=1.0fs passes (optional performance enhancement)

---

## Key Implementation Detail

The validation script uses the **existing code** in settle.py (lines 920-1039):

```python
# This is what activates constraint-aware mode:
init_fn, apply_fn = settle_langevin(
    ...,
    project_ou_momentum_rigid=True,  # <-- Enables 6D noise projection
    projection_site="post_o",         # <-- Apply after O-step (OU)
)
```

**No new code needed** — we're testing existing implementation against design spec.

---

## Insurance Plan (If Phase 2 Fails)

If dt=1.0fs shows unstable temperature (T > 310K):

**Immediate Investigation** (1-2 hours):
- Verify `project_ou_momentum_rigid=True` is enabled
- Check water_indices are correct
- Inspect Gramian eigenvalues (should be well-conditioned)
- Look for NaN/Inf in Cholesky decomposition

**Options**:
- **Option A (Recommended)**: Accept v1.0 with dt≤0.5fs constraint long-term. Shift v0.3.0 to v0.4.0 future timeline. Publish v1.0 as production release.
- **Option B**: Increase Gramian regularization or use double-precision numerics; retest
- **Option C**: Pivot to alternative constraint algorithm (LINCS) — ~2 week effort

**Decision**: If Option B shows no progress after 4 hours, recommend Option A.

---

## Integration into Existing Code

The validation script:
- ✅ Uses real SETTLE implementation from settle.py
- ✅ Uses real Langevin integrator with constraint-aware O-step
- ✅ Tests against actual validation gates (T stability, energy conservation, equipartition)
- ✅ Reports clear pass/fail per gate

This is **not** a toy test — it's production-grade validation.

---

## Expected Outcomes & Timeline

### Scenario 1: dt=1.0fs PASSES ✅
- **When**: If physics works as predicted (likely case)
- **Action**: Proceed to Phase 3 (edge cases, documentation, release)
- **Timeline**: 1 week Phase 3 work → v0.3.0 release end of April

### Scenario 2: dt=1.0fs FAILS ❌
- **When**: If constraint-aware approach has unexpected issue
- **Action**: Investigate (2-4 hours), then recommend insurance plan
- **Timeline**: Accept v1.0 as-is; defer v0.3.0 to v0.4.0 timeline

### Scenario 3: dt=2.0fs Bonus ✨ (if time & resources permit)
- **When**: After dt=1.0fs passes, optionally push to 4x speedup
- **Action**: Run test 3 to explore maximum stable timestep
- **Timeline**: +2-3 hours if attempted

---

## Files & References

**Validation**:
- `scripts/validate_constraint_aware_langevin.py` ← Run this
- `.agent/docs/v0.3.0_PHASE2_VALIDATION_SPEC.md` ← Read this for detailed spec

**Theory & Design**:
- `.agent/docs/v0.3.0_PHASE1_THEORY.md` ← Mathematical foundation
- `.agent/docs/v0.3.0_PHASE1_RISK_ASSESSMENT.md` ← Risk & mitigation

**Implementation**:
- `src/prolix/physics/settle.py` lines 920-1039 ← Code being validated

**Previous Release**:
- `CLAUDE.md` ← v1.0 usage guide (dt≤0.5fs constraint)
- `.agent/docs/V1_0_RELEASE_APPROVED.md` ← v1.0 release decision

---

## Decision Tree

```
Phase 2 Validation Running
│
├─ dt=0.5fs baseline PASS?
│  └─ No → Infrastructure broken, debug
│  └─ Yes → Continue
│
├─ dt=1.0fs PRIMARY GATE PASS? (T=300±5K, drift<1%, KS p>0.05)
│  ├─ Yes → ✅ APPROVED - Constraint-aware thermostat works!
│  │         Proceed to Phase 3 (edge cases, release)
│  │         Expected completion: ~2 weeks (by late April)
│  │
│  └─ No → ❌ ESCALATE
│          Investigate (max 4 hours)
│          If no resolution → Insurance plan (accept v1.0 constraint)
│          Defer v0.3.0 to v0.4.0 timeline
│
└─ dt=2.0fs stretch (optional, if 1.0fs passes)
   └─ Yes → Bonus: 4x speedup validated
   └─ No → Acceptable; goal achieved at 1.0fs
```

---

## Success Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| **dt=0.5fs T** | 300±5K | Baseline control (sanity check) |
| **dt=1.0fs T** | 300±5K | ✓ CRITICAL: thermostat stable |
| **dt=1.0fs drift** | <1% | ✓ CRITICAL: integration accurate |
| **dt=1.0fs KS p** | >0.05 | ✓ CRITICAL: equipartition correct |
| **Phase 2 overall** | 3/3 gates ✓ | All systems go for Phase 3 |

---

## Next Steps

1. **Run validation**: Execute `uv run scripts/validate_constraint_aware_langevin.py`
2. **Monitor**: Watch for temperature stability over 100ps
3. **Evaluate**: Check if all three gates pass (T, energy, equipartition)
4. **Decide**:
   - ✅ If PASS: Proceed to Phase 3 (edge cases, release)
   - ❌ If FAIL: Follow insurance plan (investigate or escalate)

---

## Notes for Future Sessions

- This validation is **compute-intensive** (~4-6 hours wall time)
- Phase 2 result determines whether v0.3.0 ships or gets deferred
- Phase 3 (if Phase 2 passes) includes edge-case testing and documentation
- v1.0 is stable release; v0.3.0 is optimization sprint for extended timesteps

**Expected outcome**: dt=1.0fs should work — theory is sound, implementation exists, gates are clear.

---

## Sign-Off

**v0.3.0 Phase 2**: ✅ Infrastructure complete, ready for validation  
**All prerequisites met**: Theory ✓, oracle approval ✓, librarian research ✓, implementation ✓  
**Go/No-Go**: **GO** — Proceed to Phase 2 validation  

**Confidence**: HIGH (theory-backed, implementation exists, gates are measurable)
