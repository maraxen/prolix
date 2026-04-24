# v0.3.0 Phase 2: Validation Specification

**Date:** 2026-04-24  
**Status:** Validation Design Complete  
**Goal:** Validate constraint-aware Langevin thermostat achieves stable temperature at dt=1.0fs  
**Oracle Gate:** Phase 5.1 Temperature Stability Test  

---

## 1. Test Configuration

### System Setup
- **Molecules**: 4 TIP3P water molecules (12 atoms total)
- **Duration**: 100 ps simulation
- **Timesteps**: 50,000 steps at each timestep value tested
- **Temperature Target**: 300 K (Maxwell-Boltzmann initial velocities)
- **Thermostat**: Langevin with γ = 1.0 ps⁻¹
- **Force Field**: TIP3P geometry + PME electrostatics

### Initial Conditions
```python
# 4 waters arranged in roughly 2x2 grid
# Positions: random within 12 Å box
# Velocities: Maxwell-Boltzmann sampled at T=300K
# Momenta: p = m * v
```

---

## 2. Test Cases

### 2.1 Baseline Control: dt = 0.5 fs (v1.0 Standard)

**Configuration:**
```
- Timestep: 0.5 fs (known stable for v1.0)
- Duration: 100 ps (50,000 steps)
- SETTLE: enabled
- Langevin: enabled with project_ou_momentum_rigid=True
```

**Expected Results:**
- **Temperature**: 300 ± 5 K (established v1.0 performance)
- **Energy Drift**: < 0.5% (low error expected)
- **Equipartition**: KS p-value > 0.05 (correct velocity distribution)

**Purpose:**
- Verify test framework reproduces known v1.0 behavior
- Establish baseline for comparison to 1.0fs and 2.0fs tests
- Confirm water geometry is preserved (validate non-zero KE)

**Gate**: MUST PASS (if fails, test infrastructure is broken)

---

### 2.2 Primary Goal: dt = 1.0 fs (v0.3.0 Target)

**Configuration:**
```
- Timestep: 1.0 fs (2x faster than v1.0)
- Duration: 100 ps (50,000 steps)
- SETTLE: enabled
- Langevin: enabled with project_ou_momentum_rigid=True (constraint-aware mode)
```

**Expected Results:**
- **Temperature**: 300 ± 5 K (stable if constraint-aware works)
- **Energy Drift**: < 1% (slightly higher acceptable at larger dt)
- **Equipartition**: KS p-value > 0.05 (equipartition maintained)

**Purpose:**
- Validate constraint-aware Langevin removes feedback loop at 2x timestep
- Demonstrate 2x simulation speedup without temperature runaway
- Confirm Jacobian projection and noise covariance are correct

**Gate**: CRITICAL - This is the v0.3.0 deliverable
- **PASS**: T ≤ 305 K, drift < 1%, KS p > 0.05 → proceed to Phase 3
- **FAIL**: T > 310 K or drift > 1% or KS p < 0.05 → escalate, consider insurance plan

---

### 2.3 Stretch Goal: dt = 2.0 fs (Performance Optimistic)

**Configuration:**
```
- Timestep: 2.0 fs (4x faster than v1.0)
- Duration: 100 ps (25,000 steps)
- SETTLE: enabled
- Langevin: enabled with project_ou_momentum_rigid=True
```

**Expected Results (if ambitious target met):**
- **Temperature**: 300 ± 10 K (looser tolerance at 4x timestep)
- **Energy Drift**: < 2% (higher acceptable error)
- **Equipartition**: KS p-value > 0.05 (still equipartitioned)

**Purpose:**
- Explore maximum stable timestep
- Demonstrate extended capabilities for future optimization
- Understand error growth rate with timestep

**Gate**: OPTIONAL - Only run if dt=1.0fs passes
- Success: Bonus result showing extended applicability
- Failure: Acceptable; goal was achieved at 1.0fs

---

## 3. Measurements & Metrics

### 3.1 Temperature Control (Primary Gate)

**Measurement**:
```python
# Per 1-ps checkpoint (100 measurements total)
KE_rigid = compute_rigid_body_kinetic_energy(momenta, positions, masses, water_indices)
T = 2 * KE_rigid / (6 * N_waters - 3) / BOLTZMANN_KCAL_MOL

# Statistics over full 100ps
T_mean = mean(T_per_checkpoint)
T_std = std(T_per_checkpoint)
T_min = min(T_per_checkpoint)
T_max = max(T_per_checkpoint)
```

**Acceptance Criteria**:
```
dt = 0.5 fs: |T_mean - 300| < 5 K ✓ PASS (known v1.0 baseline)
dt = 1.0 fs: |T_mean - 300| < 5 K ✓ PASS (v0.3.0 target)
dt = 1.0 fs: |T_mean - 300| < 10 K ⚠ WARNING (marginal stability)
dt = 1.0 fs: |T_mean - 300| > 10 K ✗ FAIL (feedback loop persists)

dt = 2.0 fs: |T_mean - 300| < 10 K ✓ PASS (stretch goal)
dt = 2.0 fs: |T_mean - 300| > 15 K ✗ FAIL (unstable)
```

---

### 3.2 Energy Conservation (Secondary Gate)

**Measurement**:
```python
# Total energy (kinetic + potential + constraint correction)
E_initial = KE(t=0) + PE(t=0)
E_final = KE(t=100ps) + PE(t=100ps)
E_drift_percent = abs(E_final - E_initial) / abs(E_initial) * 100

# Per-checkpoint energy conservation check
dE_per_step = E(t+dt) - E(t)  # Should be small random fluctuations, no trend
```

**Acceptance Criteria**:
```
dt = 0.5 fs: E_drift < 0.5% ✓ PASS (low error expected)
dt = 1.0 fs: E_drift < 1.0% ✓ PASS (slightly higher acceptable)
dt = 1.0 fs: E_drift < 2.0% ⚠ WARNING (accumulated error)
dt = 1.0 fs: E_drift > 2.0% ✗ FAIL (indicates integration error)

dt = 2.0 fs: E_drift < 2.0% ✓ PASS
dt = 2.0 fs: E_drift > 3.0% ✗ FAIL
```

**Interpretation**:
- Large energy drift suggests Cholesky numerics are unstable (regularization inadequate)
- Or: SETTLE constraint violations accumulating
- Or: Timestep too large for integration accuracy

---

### 3.3 Equipartition Test (Tertiary Gate)

**Measurement**:
```python
# Collect 1D velocity projections (v_x, v_y, v_z for each atom)
# Normalize by expected std: sigma_i = sqrt(kT / m_i)
# Test: Do normalized velocities match N(0,1)?

velocities_normalized = [v_i / sqrt(kT / m_i) for each atom i]
ks_stat, ks_pvalue = scipy.stats.kstest(velocities_normalized, 'norm')

# Alternative: Chi-square on velocity magnitude histogram
# v_mag ~ chi(3) distribution for 3D velocities
```

**Acceptance Criteria**:
```
dt = 0.5 fs: KS p > 0.05 ✓ PASS (equipartition correct)
dt = 1.0 fs: KS p > 0.05 ✓ PASS (noise covariance correct)
dt = 1.0 fs: KS p > 0.01 ⚠ WARNING (marginal equipartition)
dt = 1.0 fs: KS p < 0.01 ✗ FAIL (velocity distribution wrong)

dt = 2.0 fs: KS p > 0.05 ✓ PASS
dt = 2.0 fs: KS p < 0.01 ✗ FAIL
```

**Interpretation**:
- p-value < 0.05 indicates null hypothesis (velocities ~ N(0,1)) is rejected
- Likely cause: Gramian regularization is off, or Jacobian projection has bug
- Or: Noise sampling (Cholesky) is producing non-Gaussian distribution

---

## 4. Decision Tree

```
┌─ Run dt=0.5fs Baseline
│  ├─ PASS (T 300±5K, drift<0.5%, KS>0.05) → Continue
│  └─ FAIL → Test infrastructure broken; debug and restart
│
├─ Run dt=1.0fs PRIMARY GATE
│  ├─ PASS (T 300±5K, drift<1%, KS>0.05) → ✓ PROCEED to Phase 3
│  ├─ MARGINAL (T 300±10K, drift<2%, KS>0.01) → Investigate, may need tuning
│  └─ FAIL (T>310K or drift>2% or KS<0.01) → ❌ ESCALATE (see 5. Insurance Plan)
│
└─ If dt=1.0fs PASS, optionally run dt=2.0fs Stretch
   ├─ PASS (T 300±10K, drift<2%, KS>0.05) → Bonus result
   └─ FAIL → Expected; acceptable
```

---

## 5. Insurance Plan (If dt=1.0fs Fails)

### 5.1 Immediate Investigation (1-2 hours)

**Check**:
1. Is `project_ou_momentum_rigid=True` in the test configuration?
2. Are water atom indices correctly identified in `water_indices`?
3. Does Gramian G eigenvalue ratio < 10^6 (well-conditioned)?
4. Are Cholesky solve operations stable (check for NaN/Inf)?

**Debug Steps**:
```python
# Add per-water debug output
for each water:
    print(f"Water {i}: G eigenvalues = {np.linalg.eigvalsh(G)}")
    print(f"  Cholesky L diagonal = {np.diag(L)}")
    print(f"  Noise covariance norm = {np.linalg.norm(noise_cov)}")
    
# Check if momentum is staying in rigid subspace
p_before = momentum[water_atoms]
p_rigid = project_one_water_momentum_rigid(p_before, ...)
projection_error = np.linalg.norm(p_before - p_rigid) / np.linalg.norm(p_before)
print(f"Projection error: {projection_error}%")
```

### 5.2 Options if Investigation Inconclusive

**Option A (Recommended): Accept v1.0 long-term**
- Keep v1.0 with dt ≤ 0.5fs constraint (proven, stable)
- Document findings: constraint-aware thermostat showed issues at dt=1.0fs
- Shift v0.3.0 to v0.4.0 future timeline
- Publish v1.0 as production release

**Option B: Continue investigation (if close to passing)**
- Increase regularization on Gramian: `reg = 1e-9 * trace(G)` instead of 1e-12
- Test double-precision (float64) for Gramian/Cholesky
- Implement explicit condition-number check before Cholesky
- Rerun dt=1.0fs with tuned numerics

**Option C: Pivot to alternative**
- Implement LINCS constraint algorithm (different constraint solver)
- May have better thermostat compatibility
- Effort: ~2 weeks
- Risk: LINCS also couples with thermostat (Thallmair et al. 2021)

**Decision Point**:
- If Option B shows improvement, continue (max 4 more hours)
- If no progress after 4 hours, recommend Option A

---

## 6. Success Criteria Summary

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| **dt=0.5fs baseline** | T 300±5K, drift<0.5% | Test framework works |
| **dt=1.0fs temperature** | T 300±5K | Thermostat is stable |
| **dt=1.0fs energy** | drift < 1% | Integration is accurate |
| **dt=1.0fs equipartition** | KS p > 0.05 | Noise covariance is correct |
| **Phase 2 PASS** | All three gates ✓ | Constraint-aware approach works |
| **Phase 2 FAIL** | Any gate ✗ | Escalate per insurance plan |

---

## 7. Output & Reporting

**Report File**: `.agent/docs/daily/260424_phase2_validation_results.md`

Expected format:
```markdown
# v0.3.0 Phase 2 Validation Results

## Summary
[PASS / FAIL / INCONCLUSIVE]

## dt=0.5fs Baseline
- T_mean = XXX.X ± Y.Y K [PASS/FAIL]
- E_drift = Z.Z% [PASS/FAIL]
- KS p-value = 0.XXX [PASS/FAIL]

## dt=1.0fs PRIMARY GATE
- T_mean = XXX.X ± Y.Y K [PASS/FAIL]
- E_drift = Z.Z% [PASS/FAIL]
- KS p-value = 0.XXX [PASS/FAIL]

## Recommendation
[Proceed to Phase 3 / Investigate / Accept v1.0 constraint]
```

---

## 8. Related Files

- **Implementation**: `src/prolix/physics/settle.py` (lines 920-1039)
- **Theory**: `.agent/docs/v0.3.0_PHASE1_THEORY.md`
- **Risk Assessment**: `.agent/docs/v0.3.0_PHASE1_RISK_ASSESSMENT.md`
- **Test Script**: `scripts/validate_constraint_aware_langevin_v03.py`
