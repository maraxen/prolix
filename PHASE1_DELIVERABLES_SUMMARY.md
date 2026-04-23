# Phase 1 Root-Cause Isolation: Deliverables Summary

**Date**: 2026-04-23  
**Status**: Phase 1 Diagnostics Suite Complete (Ready for Execution)  
**Target Completion**: 2026-04-25  
**Owner**: Oracle + Planner Agents  

---

## Executive Summary

Phase 1 diagnostics are now ready to execute. Created comprehensive ablation test suite to isolate root cause of Langevin thermostat parity failure (ΔT = 28.8 K @ dt=1.0 fs, 10.6 K @ dt=2.0 fs).

**Three hypotheses under test**:
1. **H1 (HIGH prior)**: Fixed RATTLE saturation at n_iters=10
2. **H2 (MEDIUM prior)**: Float32 precision loss in inertia tensor
3. **H3 (MEDIUM prior)**: Projection timing bias (post_o vs both)

**Deliverables** (Created):
- ✓ `test_phase1_ablation.py` - Pytest test suite with 3 ablation axes
- ✓ `phase1_diagnostic_runner.py` - Standalone runner with vmap ensemble parallelization
- ✓ `phase1_plotting_and_verdict.py` - Automated hypothesis scoring + plots
- ✓ `PHASE1_EXECUTION_GUIDE.md` - Quick-start guide for running diagnostics
- ✓ `phase1_results/README.md` - Interpretation guide for outputs
- ✓ Documentation + implementation

---

## Deliverable Details

### 1. Test Suite: `tests/physics/test_phase1_ablation.py`

**Size**: ~400 lines  
**Status**: ✓ Syntax verified, ready to execute  
**Three Test Classes**:

#### `TestPhase1AblationRATTLEIterations`
```python
def test_rattle_convergence_vs_iterations(self, dt_akma)
```
- Sweeps `n_iters ∈ [1, 3, 5, 10, 20, 50]` at each dt
- Measures temperature, residual norm, KE_rigid per step
- Produces convergence curves as evidence for H1

**Runs**:
```bash
pytest tests/physics/test_phase1_ablation.py::TestPhase1AblationRATTLEIterations \
  -v --tb=short
```

#### `TestPhase1AblationFloatPrecision`
```python
def test_float32_vs_float64_precision(self)
```
- Runs identical trajectory at float32 and float64
- Logs condition numbers of inertia tensor
- Compares KE divergence between precisions

**Evidence for H2**:
- Condition number instrumentation
- T(float32) vs T(float64) divergence curves

#### `TestPhase1AblationProjectionSite`
```python
def test_projection_site_comparison(self, dt_akma)
```
- Compares three modes: 'post_o', 'post_settle_vel', 'both'
- Measures ΔT between modes
- Quantifies projection timing bias

**Evidence for H3**:
- T_inst vs projection_site histograms
- Statistical comparison between modes

#### `TestPhase1VmapEnsemble`
```python
def test_vmap_ensemble_runner(self)
```
- Demonstrates vmap parallelization over 5+ replicas
- Validates ensemble statistics collection
- Proof-of-concept for phase1_diagnostic_runner.py

---

### 2. Diagnostic Runner: `scripts/phase1_diagnostic_runner.py`

**Size**: ~500 lines  
**Status**: ✓ Syntax verified, ready to execute  
**Purpose**: Full Phase 1 execution with automated ensemble parallelization

**Key Features**:
- Vmap parallelization over N_replicas × N_configs
- Per-step metrics: T_inst, KE_rigid, constraint_residual_norm
- CSV output for each configuration
- JSON summary of all results
- Wall-clock timing: ~30 min (5 replicas) to 3 hours (10+ replicas)

**Usage**:
```bash
# Quick: 5 replicas (est. 30 min)
python3 scripts/phase1_diagnostic_runner.py --n-replicas 5 --output ./phase1_results

# Full: 10 replicas (est. 2-3 hours on GPU)
python3 scripts/phase1_diagnostic_runner.py --n-replicas 10 --output ./phase1_results
```

**Output Structure**:
```
phase1_results/
├── summary.json
├── axis1_iterations/    (RATTLE sweep: n_iters ∈ [1,3,5,10,20,50])
├── axis2_precision/     (Float32 vs Float64)
├── axis3_projection/    (Projection site: post_o, post_settle_vel, both)
```

**Configurations Executed**:
- **Axis 1**: 18 configs (3 dt × 6 n_iters values)
- **Axis 2**: 6 configs (3 dt × 2 precisions)
- **Axis 3**: 9 configs (3 dt × 3 projection sites)
- **Total**: 33 configurations × 5-10 replicas each = 165-330 trajectories

---

### 3. Plotting & Verdict: `scripts/phase1_plotting_and_verdict.py`

**Size**: ~300 lines  
**Status**: ✓ Syntax verified, ready to execute  
**Purpose**: Hypothesis scoring + visualization

**Outputs** (Generated):
1. **Plots**:
   - `residual_convergence.png` - SETTLE residual vs n_iters (Axis 1)
   - `ke_precision_comparison.png` - Float32 vs Float64 (Axis 2)
   - `temperature_by_projection_site.png` - Projection comparison (Axis 3)

2. **PHASE1_VERDICT.md**:
   - Hypothesis support scores (0.0 to 1.0)
   - Evidence tables by configuration
   - Decision tree for Phase 2 action
   - Go/No-Go recommendation

**Hypothesis Scoring** (Automated):
```python
# H1_score = (monotonic_improvement_ratio) × (ΔT_improvement / 30 K)
# H2_score = min(1.0, max(0, precision_gap - 5 K) / 25 K)
# H3_score = min(1.0, max(0, projection_diff - 2 K) / 20 K)

if total_score > 0.5:
    verdict = "CONDITIONAL APPROVAL" → Proceed to Phase 2
else:
    verdict = "ESCALATE" → Deeper investigation
```

**Usage**:
```bash
python3 scripts/phase1_plotting_and_verdict.py \
  --input ./phase1_results \
  --output ./phase1_results
```

---

### 4. Documentation

#### `PHASE1_EXECUTION_GUIDE.md`
- **Purpose**: Quick-start guide for running diagnostics
- **Content**: 3 execution methods (pytest, standalone, interactive)
- **Examples**: Console output, expected results, troubleshooting

#### `phase1_results/README.md`
- **Purpose**: Interpretation guide for outputs
- **Content**: How to read CSV files, understand plots, interpret hypothesis scores
- **Tables**: Expected temperature ranges, hypothesis support interpretation matrix

#### `phase1_results/PHASE1_VERDICT.template.md`
- **Purpose**: Example verdict document structure
- **Content**: Template for what PHASE1_VERDICT.md will contain

---

## Expected Results (Example)

### Console Output
```
======================================================================
PHASE 1: ROOT-CAUSE ISOLATION - COMPREHENSIVE ABLATION
======================================================================

...

AXIS 1: RATTLE ITERATION SWEEP
dt=1.0 AKMA, n_iters sweep:
  n_iters= 1: T=328.5±12.3 K, residual=4.56e-03  ← MATCHES OBSERVED ERROR
  n_iters= 3: T=317.2±10.1 K, residual=2.78e-03
  n_iters=10: T=305.1±7.8 K, residual=9.23e-04
  n_iters=50: T=300.2±6.8 K, residual=1.23e-04  ← CONVERGED

...

======================================================================
Summary saved to phase1_results/summary.json
======================================================================
```

### PHASE1_VERDICT.md (Hypothetical - H1 Confirmed)

```markdown
## Hypothesis Support Scores

### Hypothesis 1: Fixed RATTLE Saturation (HIGH prior)
- **Quantitative Support Score**: 0.85
- **Interpretation**: CONFIRMED (HIGH)
- **Evidence**:
  - Monotonic improvement ratio: 0.90
  - Temperature drop (n_iters 1→50): 28.3 K
  - Final error @ n_iters=50: 0.2 K

## Decision: Phase 2 Action

IF H1 > 0.7:
  → RATTLE saturation is PRIMARY CAUSE
  → Phase 2 Action: Increase settle_velocity_iters from 10 → [20-50]
  → Test: Verify ΔT < 5 K @ dt=1.0 fs with n_iters=50
```

---

## Code Entry Points (Production Code - Unchanged)

**No production code modifications in Phase 1** (diagnostics only).

Reference implementation points for Phase 2:

| File | Function | Lines | Purpose |
|------|----------|-------|---------|
| `settle.py` | `settle_velocities` | 382-444 | RATTLE loop (modify n_iters default) |
| `settle.py` | `settle_langevin` | 471-651 | Integrator (modify settle_velocity_iters param) |
| `settle.py` | `_project_one_water_momentum_rigid` | 129-157 | Inertia projection (instrument for H2) |
| `settle.py` | `project_tip3p_waters_momentum_rigid` | 159-176 | Momentum projection (modify for projection_site='both') |

---

## Execution Workflow

### Phase 1A: Diagnostics (2-3 hours)
```bash
cd /home/marielle/projects/prolix
python3 scripts/phase1_diagnostic_runner.py --n-replicas 5-10 --output ./phase1_results
# Wait 30 min - 3 hours
```

### Phase 1B: Analysis (5 minutes)
```bash
python3 scripts/phase1_plotting_and_verdict.py --input ./phase1_results --output ./phase1_results
```

### Phase 1C: Decision (5 minutes)
```bash
cat phase1_results/PHASE1_VERDICT.md
# Read hypothesis scores, determine Phase 2 action
```

### Phase 2: Implementation (if Go decision)
```bash
# Modify production code based on verdict
# Example (if H1 > 0.7):
#   1. Change settle_velocity_iters=10 → 50 in settle.py
#   2. Re-test against OpenMM reference
#   3. Commit changes
```

---

## Validation Checklist (Before Execution)

- [x] Test suite compiles without syntax errors
- [x] Diagnostic runner compiles without syntax errors
- [x] Plotting script compiles without syntax errors
- [x] All imports are available (JAX, jax-md, numpy, etc.)
- [x] Output directories created
- [x] Documentation complete
- [ ] **(Execute in next step)** Run phase1_diagnostic_runner.py with n_replicas ≥ 5
- [ ] **(Execute in next step)** Generate plots + verdict
- [ ] **(Execute in next step)** Read PHASE1_VERDICT.md and decide Phase 2 action

---

## Critical Success Factors

1. **Hypothesis Scores Clear**: Each hypothesis should score > 0.7 or < 0.3 (avoid ambiguity)
2. **Ensemble Statistics Valid**: No NaNs/Infs in results
3. **Temperature Ranges Reasonable**: All T values within 250-350 K (physical range)
4. **Convergence Evident**: At least 2 axes show clear monotonic trends
5. **Go/No-Go Decision Actionable**: Verdict recommends specific Phase 2 fix

---

## Phase 2 Preparation (Contingent on Verdict)

### If H1 > 0.7 (RATTLE Saturation)
- **Fix**: Increase `settle_velocity_iters` from 10 → 20-50
- **Location**: `settle.py:471` (settle_langevin, parameter default)
- **Expected Improvement**: ΔT from 28 K → < 5 K @ dt=1.0 fs
- **Retest**: `tests/physics/test_openmm_explicit_anchor.py`

### If H2 > 0.7 (Float32 Precision)
- **Fix**: Force float64 in `_project_one_water_momentum_rigid`
- **Location**: `settle.py:129-157` (add dtype enforcement)
- **Expected Improvement**: ΔT from 28 K → < 5 K via condition number control
- **Retest**: `tests/physics/test_explicit_solvation_parity.py`

### If H3 > 0.7 (Projection Timing)
- **Fix**: Change default `projection_site` from 'post_o' → 'both'
- **Location**: `settle.py:487` (settle_langevin, parameter default)
- **Expected Improvement**: ΔT from 28 K → < 15 K (modest)
- **Retest**: `tests/physics/test_explicit_l3_observables.py`

### If All Scores < 0.5 (Complex Interaction)
- **Fix**: Implement combined approach (H1 + H2 + H3 together)
- **Action**: Escalate for deeper code review + condition number logging
- **Fallback**: Manual tuning of settle_velocity_iters based on dt

---

## File Manifest

| Path | Type | Size | Status | Purpose |
|------|------|------|--------|---------|
| `tests/physics/test_phase1_ablation.py` | Test | 400 L | ✓ Ready | Pytest test suite |
| `scripts/phase1_diagnostic_runner.py` | Script | 500 L | ✓ Ready | Ensemble runner |
| `scripts/phase1_plotting_and_verdict.py` | Script | 300 L | ✓ Ready | Analysis + verdict |
| `PHASE1_EXECUTION_GUIDE.md` | Doc | 300 L | ✓ Ready | Quick-start guide |
| `phase1_results/README.md` | Doc | 400 L | ✓ Ready | Interpretation guide |
| `phase1_results/PHASE1_VERDICT.template.md` | Template | 200 L | ✓ Ready | Expected output |
| `PHASE1_DELIVERABLES_SUMMARY.md` | Doc | 500 L | ✓ Ready | This file |

**Total Lines of Code/Documentation**: ~2500 lines  
**Total Time to Create Phase 1**: ~4 hours (analysis + implementation + documentation)  
**Ready for Execution**: YES ✓

---

## Next Steps

1. **Immediate (Today)**:
   - Run Phase 1 diagnostics: `python3 scripts/phase1_diagnostic_runner.py --n-replicas 5`
   - Estimated wall-clock time: 30-45 minutes

2. **Short-term (Tomorrow)**:
   - Generate plots and verdict: `python3 scripts/phase1_plotting_and_verdict.py`
   - Review PHASE1_VERDICT.md for hypothesis scores
   - Determine Phase 2 action based on verdict

3. **Medium-term (This Week)**:
   - Implement Phase 2 fixes based on verdict
   - Revalidate against OpenMM reference
   - Commit changes and close issue

---

**Created**: 2026-04-23  
**Phase 1 Target Completion**: 2026-04-25  
**Status**: READY FOR EXECUTION ✓  

**Key Contact**: Oracle Agent (strategic guidance) + Planner Agent (implementation)  
**Success Metric**: Phase 1 Verdict with clear hypothesis support scores → Go/No-Go decision
