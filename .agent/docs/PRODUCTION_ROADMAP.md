# Prolix Production Release Roadmap

**Date**: 2026-04-23  
**Current Version**: 0.1.0-alpha  
**Status**: Not Production-Ready (Critical Blocker Identified)  

---

## Executive Summary

Prolix has excellent infrastructure (CI/CD, testing, documentation, release automation) but cannot ship to production due to **unresolved physics integration issues** discovered in Phase 2 validation.

**Current Situation**:
- ✅ Phase 2C (Constrained OU Thermostat) is **complete and validated**
- ❌ Phase 2 (SETTLE-Langevin integration) revealed **fundamental instability** — G4 gate FAILED
- 🛑 **Temperature control is unreliable at production timesteps** (dt ≥ 1.5 fs)

**Timeline to Production**: 3–4 weeks post-fix (contingent on resolving integration issue)

---

## Critical Blocker: Phase 2 G4 Gate Failure

### The Problem

Phase 2 attempted to integrate SETTLE water constraints with Langevin thermostat by moving SETTLE_vel before the O-step:

```
B → A → O(3N) → A → SETTLE_pos → force → B → SETTLE_vel → projection
```

**Failure Mode**:
- dt=1.0 fs: 21× temperature amplification (300K → 6,300K)
- dt=1.5 fs: unstable
- dt=2.0 fs: erratic behavior

**Root Cause**: SETTLE constraints and Langevin stochastic noise interact poorly when SETTLE_vel is moved relative to the O-step. The constraint correction can add/remove energy in ways incompatible with the thermostat's equilibration dynamics.

### Phase 2C's Contribution

Phase 2C (constrained-subspace OU) was designed to fix this by sampling noise directly in the rigid-body subspace, but testing reveals:
- ✅ Phase 2C noise generation is mathematically correct
- ✅ Individual water molecules equilibrate correctly
- ❌ **Full system still shows slow equilibration at dt=1 fs** (418.6K after 100ps instead of 300K)

**Interpretation**: Phase 2C fixes the noise covariance issue but doesn't fully solve the integration incompatibility between SETTLE and Langevin.

### Impact on Production

Without resolving this blocker:
- **Cannot run long production simulations** at standard dt (1.5–2.0 fs)
- **Cannot validate against OpenMM** (which handles this coupling correctly)
- **Cannot publish papers** using Prolix results

---

## Production Readiness Assessment

### Infrastructure: 7/10 ✅ (Ready)

| Component | Status | Details |
|-----------|--------|---------|
| **CI/CD** | ✅ Ready | GitHub Actions, pytest, ruff, type-check, publish automation |
| **Testing Framework** | ✅ Ready | pytest with markers, multiple Python versions (3.11, 3.12) |
| **Package Build** | ✅ Ready | setuptools, wheel/sdist, OIDC-based PyPI publishing |
| **Documentation** | ⚠️ Partial | Sphinx configured, API docs ready, missing CHANGELOG.md and CONTRIBUTING.md |
| **Linting/Typing** | ✅ Ready | ruff + ty pre-commit hooks, CI-gated enforcement |
| **Regression Testing** | ⚠️ Minimal | Single regression dict validation, no comprehensive suite |

### Physics Validation: 4/10 ❌ (Blocked)

| Component | Status | Details |
|-----------|--------|---------|
| **Phase 1 (PME)** | ✅ Complete | OpenMM anchor parity validated, PME implementation correct |
| **Phase 2 (SETTLE-Langevin)** | ❌ FAILED | G4 gate failure: temperature instability at dt ≥ 1.5 fs |
| **Phase 2C (Constrained OU)** | ✅ Complete | Implementation correct, but doesn't resolve Phase 2 issue fully |
| **Explicit Solvent Validation** | ❌ Incomplete | OpenMM anchor tests conditional, integration tests skipped in CI |
| **Temperature Control** | ❌ Unreliable | Slow equilibration at dt=1 fs, plateau at 418K instead of 300K |
| **GPU/CUDA Support** | ⚠️ Untested | Dependency group exists, no CI testing on GPU |

---

## Detailed Production Readiness by Phase

### Phase 1: PME Implementation ✅ APPROVED
- ✅ Anchor tests pass (parity with OpenMM)
- ✅ CI-validated
- ✅ Documentation complete
- **Status**: Can be released independently

### Phase 2: SETTLE + Langevin ❌ BLOCKED
- ❌ G4 gate failed (temperature instability)
- ❌ Integration tests skipped
- ❌ Not validated against OpenMM explicit solvent
- **Status**: Cannot release without resolving integration issue

### Phase 2C: Constrained-Subspace OU ✅ READY (but insufficient)
- ✅ Implementation complete and validated
- ✅ Noise generation mathematically correct
- ✅ No bugs found
- ⚠️ Does not fully resolve Phase 2 integration issue
- **Status**: Ready to merge, but doesn't unblock production release

---

## Path to Production Release

### Stage 1: Resolve Phase 2 Integration Blocker (CRITICAL)

**Options**:

#### Option A: Fix SETTLE-Langevin Integration ⚡ (Recommended)
**Goal**: Resolve fundamental incompatibility

**Approaches**:
1. **Return SETTLE_vel to original position** (after all forces, before B-step)
   - Risk: May lose some constraint stability
   - Benefit: Avoids SETTLE-OU coupling issues
   - Timeline: 1-2 days to test

2. **Redesign SETTLE_vel timing** to minimize thermostat interference
   - Risk: High complexity, significant refactoring
   - Benefit: Maintains constraint enforcement strength
   - Timeline: 2-3 weeks

3. **Implement custom SETTLE-aware thermostat**
   - Risk: Novel algorithm, requires validation
   - Benefit: Optimal coupling
   - Timeline: 2-4 weeks

**Recommendation**: Start with Option A (1-2 days), validate if sufficient. If not, pursue Option B.

#### Option B: Document Limitations & Restrict Use ⚠️ (Fallback)
**Goal**: Ship with known constraints

**Approach**:
- Document that dt > 1.0 fs is not supported
- Use Phase 2C for smaller timesteps
- Recommend dt=1.0 fs maximum

**Pros**: Faster to ship (1 week)
**Cons**: Severely limits applicability, not production-grade

#### Option C: Revert to Phase 1 Only ⚠️ (Conservative)
**Goal**: Ship PME-only (implicit solvent) version

**Approach**:
- Remove SETTLE and Phase 2 code
- Release as v1.0-implicit-solvent
- Plan Phase 2 for future major version

**Pros**: Safe, clean release (1 week)
**Cons**: No explicit solvent support, less ambitious

**Recommendation**: **Option A is the path forward** — try simple fix first, then escalate if needed.

---

### Stage 2: Physics Validation (SEQUENTIAL)

**After Phase 2 blocker is resolved**:

1. **Run Full Integration Test Suite** (1 week)
   - Execute: `pytest -m integration` (currently skipped)
   - Validate: PME + SETTLE against OpenMM on realistic systems
   - Gate: Parity within 0.1% on 100-step equilibration

2. **Explicit Solvent Anchor Tests** (3 days)
   - Test water box, protein-water solvation scenarios
   - Compare: Energy, forces, trajectory vs OpenMM
   - Documentation: Publish validation report

3. **Temperature Stability Validation** (2 days)
   - Run 100ps simulations with multiple seeds
   - Verify: Temperature stays within ±5% of target
   - Benchmark: Compare with OpenMM/GROMACS

4. **GPU/CUDA Testing** (1 week)
   - Add GPU CI jobs
   - Validate JAX CUDA compilation
   - Performance profile vs CPU

**Total Stage 2**: 3–4 weeks

---

### Stage 3: Release Preparation (PARALLEL)

**While physics validation runs**:

1. **Documentation** (1 week)
   - Write CHANGELOG.md (features from Phase 1-2C)
   - Create CONTRIBUTING.md
   - Document known limitations
   - Installation troubleshooting guide

2. **Code Cleanup** (1 week)
   - Remove debug/validation scripts from core
   - Consolidate Phase 2 changes
   - Remove skip decorators from core tests

3. **Version Bump & Release Notes** (2 days)
   - Change version: 0.1.0-alpha → 0.2.0
   - Create release notes
   - Set up OIDC on PyPI (one-time setup)

4. **Performance Benchmarking** (optional, 3 days)
   - Spatial sorting profile
   - PME scaling plots
   - Document in release notes

**Total Stage 3**: 1–2 weeks (in parallel with Stage 2)

---

### Stage 4: Release & Monitoring (1 DAY)

1. **Final QA** (1 day)
   - Run full test suite on clean checkout
   - Build docs
   - Test PyPI publish workflow

2. **Tag & Publish** (1 hour)
   - Create GitHub release (triggers Actions)
   - Verify PyPI package appears
   - Update README disclaimer

3. **Post-Release** (ongoing)
   - Monitor GitHub Issues
   - Track user feedback
   - Plan v0.2.1 patch releases if needed

---

## Recommended Work Plan

### Week 1: Resolve Phase 2 Blocker
```
Day 1-2: Implement Option A (revert SETTLE_vel timing)
Day 3-4: Validate on test suite
Day 5:   Decision on Option B/C escalation if needed
```

### Week 2-3: Physics Validation + Release Prep
```
Parallel:
  - Run integration tests (1 week)
  - Write documentation (1 week)
  - Performance benchmarking (3 days)
  - Code cleanup (2 days)
```

### Week 4: Release
```
Day 1-2: Final QA
Day 3:   Tag & publish to PyPI
Day 4-5: Monitoring & issues
```

---

## Risk Assessment

### High Risk
- **Phase 2 integration fix not working** → Requires escalation to Option B/C (ship with limitations or revert)
- **Integration tests fail against OpenMM** → Indicates physics regression, delays release

### Medium Risk
- **GPU testing delays** → Can ship CPU-only v0.2.0, add GPU in v0.2.1
- **Documentation incomplete** → Can ship with minimal docs, improve in v0.2.1
- **Coverage below 70%** → Can lower bar to 60% for initial release

### Low Risk
- **Version bumping** → Straightforward
- **PyPI publishing** → Already automated
- **Linting/typing** → Already enforced

---

## Go/No-Go Decision Criteria

### GO (Proceed to Production)
✅ Phase 2 blocker resolved AND temperature stable (±5%)  
✅ Integration tests pass against OpenMM  
✅ Documentation complete (CHANGELOG, README, API docs)  
✅ All CI checks green (lint, type, tests)  
✅ Coverage ≥ 60%  

### NO-GO (Hold Release)
❌ Phase 2 integration issue unresolved  
❌ Temperature instability persists  
❌ Integration tests fail  
❌ Critical documentation missing  

### CONDITIONAL GO (Ship with Limitations)
⚠️ Phase 2 fixed but limitations document (max dt=1.0 fs)  
⚠️ GPU testing incomplete (CPU-only release)  
⚠️ OpenMM validation incomplete (validate in v0.2.1)  

---

## Parallel Work Items

### Immediate (This Week)
- [ ] Investigate Phase 2 SETTLE-OU integration issue
- [ ] Attempt Option A fix (revert SETTLE_vel timing)
- [ ] Begin writing CHANGELOG.md

### Short Term (Week 2)
- [ ] Run full integration test suite
- [ ] Complete documentation (CONTRIBUTING.md)
- [ ] Validate against OpenMM explicit solvent
- [ ] GPU/CUDA CI setup (if time permits)

### Before Release
- [ ] Final QA on clean checkout
- [ ] Update README (remove alpha disclaimer)
- [ ] Create GitHub release
- [ ] PyPI publish (automated)

---

## Resource & Time Estimates

| Task | Owner | Duration | Dependencies |
|------|-------|----------|--------------|
| Phase 2 fix | Research/Dev | 3–7 days | None (blocker) |
| Integration testing | Test/QA | 5 days | Phase 2 fix |
| Documentation | Writing | 5 days | Phase 2 fix |
| GPU/CUDA CI | DevOps | 5 days | Optional |
| Release/publish | DevOps | 1 day | All above |

**Total Critical Path**: 3–4 weeks (Phase 2 fix + validation + release)

---

## Success Metrics for v0.2.0 Release

| Metric | Target | Status |
|--------|--------|--------|
| Phase 1 (PME) validated | ✅ Pass | ✅ Complete |
| Phase 2 (SETTLE-OU) stable | ✅ Pass | ⏳ In progress |
| Temperature control | ±5% of target | ⏳ Blocked |
| Integration test coverage | ≥ 5 main scenarios | ⏳ Skipped in CI |
| Documentation completeness | 90% | ⏳ Partial (missing CHANGELOG) |
| CI passing | All checks green | ✅ Lint/type pass; tests partial |
| PyPI package | Published | ⏳ Ready but blocked |

---

## Conclusion

**Prolix is NOT ready for production v0.2.0 release** due to unresolved Phase 2 (SETTLE-Langevin integration) issues. However, the infrastructure and Phase 2C improvements put it **within 3–4 weeks of a release** if the integration blocker can be resolved.

**Immediate action**: Investigate Phase 2 blocker and attempt Option A fix (revert SETTLE_vel timing). If successful within 2 days, proceed with full validation and release pipeline. If blocked, escalate to Option B/C.

**Recommendation**: Allocate 1–2 researchers for 1 week on Phase 2 integration fix, then parallel workstreams for validation and documentation.

---

**Document prepared by**: Claude Code  
**Analysis based on**: Phase 2C investigation + Recon agent findings  
**Last updated**: 2026-04-23
