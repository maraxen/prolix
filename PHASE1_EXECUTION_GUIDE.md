# Phase 1 Execution Guide: Langevin Thermostat Parity Diagnostics

**Quick Start**: 5 minutes to first results, 2-3 hours for full Phase 1

## Three Ways to Run Phase 1

### 1. Pytest Tests (Unit-level, Quick)

**Time**: ~5-10 minutes for single test

```bash
cd /home/marielle/projects/prolix

# Test 1: RATTLE iteration convergence @ dt=1.0 AKMA
pytest tests/physics/test_phase1_ablation.py::TestPhase1AblationRATTLEIterations::test_rattle_convergence_vs_iterations \
  -v -s -k "1.0"

# Test 2: Projection site comparison
pytest tests/physics/test_phase1_ablation.py::TestPhase1AblationProjectionSite::test_projection_site_comparison \
  -v -s

# Test 3: VMap ensemble runner
pytest tests/physics/test_phase1_ablation.py::TestPhase1VmapEnsemble::test_vmap_ensemble_runner \
  -v -s

# Run all Phase 1 tests
pytest tests/physics/test_phase1_ablation.py -v -s
```

### 2. Standalone Diagnostic Runner (Full Suite, Parallelized)

**Time**: 30 min (5 replicas) to 2-3 hours (10+ replicas)

```bash
cd /home/marielle/projects/prolix

# Quick diagnostic (5 replicas, est. 30-45 min)
python3 scripts/phase1_diagnostic_runner.py \
  --n-replicas 5 \
  --output ./phase1_results

# Full Phase 1 (10 replicas, est. 2-3 hours on GPU)
python3 scripts/phase1_diagnostic_runner.py \
  --n-replicas 10 \
  --output ./phase1_results

# Minimal debug (2 replicas, 10 min)
python3 scripts/phase1_diagnostic_runner.py \
  --n-replicas 2 \
  --output ./phase1_results_debug
```

After execution, run plotting & verdict generation:

```bash
python3 scripts/phase1_plotting_and_verdict.py \
  --input ./phase1_results \
  --output ./phase1_results
```

This generates:
- `phase1_results/residual_convergence.png`
- `phase1_results/ke_precision_comparison.png`
- `phase1_results/temperature_by_projection_site.png`
- `phase1_results/PHASE1_VERDICT.md` ← **READ THIS FIRST**

### 3. Interactive Python (Experimentation)

**Time**: As needed

```python
#!/usr/bin/env python3
import jax
jax.config.update("jax_enable_x64", True)

from tests.physics.test_phase1_ablation import (
    TestPhase1AblationRATTLEIterations,
    TestPhase1AblationProjectionSite,
    TestPhase1AblationFloatPrecision,
)

# Test individual configurations
test_rattle = TestPhase1AblationRATTLEIterations()
test_rattle.test_rattle_convergence_vs_iterations(dt_akma=1.0)

test_proj = TestPhase1AblationProjectionSite()
test_proj.test_projection_site_comparison(dt_akma=1.0)

test_prec = TestPhase1AblationFloatPrecision()
test_prec.test_float32_vs_float64_precision()
```

## Expected Outputs

### Console Output (Example)

```
======================================================================
PHASE 1: ROOT-CAUSE ISOLATION - COMPREHENSIVE ABLATION
======================================================================

----------------------------------------------------------------------
AXIS 1: RATTLE ITERATION SWEEP
----------------------------------------------------------------------

dt=0.5 AKMA, n_iters sweep:
  n_iters= 1: T=303.2±7.2 K, residual=2.34e-03
  n_iters= 3: T=301.8±6.8 K, residual=1.45e-03
  n_iters= 5: T=300.9±6.5 K, residual=8.92e-04
  n_iters=10: T=300.3±6.2 K, residual=4.21e-04
  n_iters=20: T=300.1±6.0 K, residual=1.89e-04
  n_iters=50: T=300.0±6.0 K, residual=8.34e-05

dt=1.0 AKMA, n_iters sweep:
  n_iters= 1: T=328.5±12.3 K, residual=4.56e-03  ← OBSERVED ERROR!
  n_iters= 3: T=317.2±10.1 K, residual=2.78e-03
  n_iters= 5: T=308.4±8.5 K, residual=1.67e-03
  n_iters=10: T=305.1±7.8 K, residual=9.23e-04
  n_iters=20: T=302.3±7.1 K, residual=4.12e-04
  n_iters=50: T=300.2±6.8 K, residual=1.23e-04

dt=2.0 AKMA, n_iters sweep:
  n_iters= 1: T=320.2±14.5 K, residual=5.89e-03
  n_iters= 3: T=315.6±12.8 K, residual=4.12e-03
  n_iters= 5: T=310.8±11.2 K, residual=2.89e-03
  n_iters=10: T=310.6±10.8 K, residual=2.45e-03  ← OBSERVED ERROR!
  n_iters=20: T=304.5±8.9 K, residual=1.23e-03
  n_iters=50: T=301.4±7.5 K, residual=3.45e-04

...more axes...

======================================================================
Summary saved to phase1_results/summary.json
======================================================================
```

### File Structure

```
phase1_results/
├── README.md                             ← Overview & interpretation guide
├── PHASE1_VERDICT.template.md            ← Expected output structure
├── summary.json                          ← Master data file (created)
├── PHASE1_VERDICT.md                     ← Hypothesis scores (created)
│
├── residual_convergence.png              ← AXIS 1 plots (created)
├── ke_precision_comparison.png           ← AXIS 2 plots (created)
├── temperature_by_projection_site.png    ← AXIS 3 plots (created)
│
├── axis1_iterations/                     ← Raw data by iteration count
│   ├── dt0.5_n1_pfloat64_projpost_o_ensemble.csv
│   ├── dt0.5_n3_pfloat64_projpost_o_ensemble.csv
│   ├── ...
│   └── dt2.0_n50_pfloat64_projpost_o_ensemble.csv
│
├── axis2_precision/                      ← Raw data by precision
│   ├── dt0.5_n10_pfloat32_projpost_o_ensemble.csv
│   ├── dt0.5_n10_pfloat64_projpost_o_ensemble.csv
│   ├── ...
│   └── dt2.0_n10_pfloat64_projpost_o_ensemble.csv
│
└── axis3_projection/                     ← Raw data by projection site
    ├── dt0.5_n10_pfloat64_projpost_o_ensemble.csv
    ├── dt0.5_n10_pfloat64_projpost_settle_vel_ensemble.csv
    ├── dt0.5_n10_pfloat64_projboth_ensemble.csv
    ├── ...
    └── dt2.0_n10_pfloat64_projboth_ensemble.csv
```

## PHASE1_VERDICT.md: What to Look For

The auto-generated `PHASE1_VERDICT.md` contains:

```markdown
## Hypothesis Support Scores (0.0 = no support, 1.0 = full support)

### Hypothesis 1: Fixed RATTLE Saturation (HIGH prior)
- **Quantitative Support Score**: 0.85
- **Interpretation**: CONFIRMED (HIGH) → RATTLE convergence is primary cause
- **Evidence**:
  - Monotonic improvement ratio: 0.90
  - Temperature drop (n_iters 1→50): 28.3 K
  - Final error @ n_iters=50: 0.2 K

### Hypothesis 2: Float32 Precision Degradation (MEDIUM prior)
- **Quantitative Support Score**: 0.25
- **Interpretation**: REJECTED → Float64 operations are sufficient
- **Evidence**:
  - Float32 mean T: 305.1 K
  - Float64 mean T: 302.3 K
  - Precision gap: 2.8 K (small, not significant)

### Hypothesis 3: Projection Site Timing Bias (MEDIUM prior)
- **Quantitative Support Score**: 0.45
- **Interpretation**: PARTIAL → Projection helps but not the main issue
- **Evidence**:
  - post_o mean T: 305.1 K
  - both mean T: 303.2 K
  - Projection improvement: 1.9 K (modest)

## Decision Tree & Next Steps

IF H1 > 0.7:
  → RATTLE saturation is PRIMARY CAUSE
  → Phase 2 Action: Increase settle_velocity_iters to [20-50]
  → Test: Verify ΔT < 5 K @ dt=1.0 fs with n_iters=50

...
```

**Read this to determine Phase 2 action**.

## Quick Interpretation Matrix

| Score | Interpretation | Confidence | Action |
|-------|----------------|-----------|--------|
| > 0.7 | CONFIRMED | HIGH | Implement fix in Phase 2 |
| 0.4-0.7 | PARTIAL | MEDIUM | Fix contributes but not sole cause |
| < 0.4 | REJECTED | LOW | Not a primary factor |

## Workflow: From Diagnostics to Phase 2

```
┌──────────────────────────────────────────┐
│ 1. Run Phase 1 Diagnostics               │
│    python3 scripts/phase1_diagnostic_... │
│    (Takes 30 min - 3 hours)              │
└──────────────────────────┬───────────────┘
                           │
┌──────────────────────────v───────────────┐
│ 2. Generate Plots & Verdict              │
│    python3 scripts/phase1_plotting_...   │
│    (Takes 2 minutes)                     │
└──────────────────────────┬───────────────┘
                           │
┌──────────────────────────v───────────────┐
│ 3. Read PHASE1_VERDICT.md                │
│    - Check hypothesis scores             │
│    - Identify which fix to implement     │
└──────────────────────────┬───────────────┘
                           │
┌──────────────────────────v───────────────┐
│ 4. Phase 2 Implementation                │
│    - Modify settle.py / simulate.py      │
│    - Run production tests                │
│    - Validate against OpenMM             │
└──────────────────────────┬───────────────┘
                           │
┌──────────────────────────v───────────────┐
│ 5. Acceptance (G4 bound < 5 K)           │
│    - Commit Phase 2 changes              │
│    - Update documentation                │
└──────────────────────────────────────────┘
```

## Diagnostic Code Locations

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `tests/physics/test_phase1_ablation.py` | 3 test classes (RATTLE, precision, projection) | ~400 | ✓ Created |
| `scripts/phase1_diagnostic_runner.py` | Ensemble runner with vmap | ~500 | ✓ Created |
| `scripts/phase1_plotting_and_verdict.py` | Plots + hypothesis scoring | ~300 | ✓ Created |
| `phase1_results/README.md` | How to interpret results | - | ✓ Created |
| `PHASE1_EXECUTION_GUIDE.md` | This file | - | ✓ Created |

## Important Notes

1. **No production code changes** in Phase 1 — only diagnostics
2. **Results are ensemble statistics** — multiple replicas per config
3. **Hypothesis scores are automated** — based on observed data patterns
4. **Phase 2 action is data-driven** — determined by PHASE1_VERDICT.md

## Troubleshooting

```bash
# JAX float64 not enabled?
python3 -c "import jax; jax.config.update('jax_enable_x64', True); print(jax.config.jax_enable_x64)"

# Memory issues? Reduce replicas
python3 scripts/phase1_diagnostic_runner.py --n-replicas 3 --output ./phase1_results_small

# Test one config quickly
pytest tests/physics/test_phase1_ablation.py::TestPhase1AblationRATTLEIterations::test_rattle_convergence_vs_iterations -v -k "0.5"
```

---

**Created**: 2026-04-23
**Estimated Time to Phase 1 Completion**: 2-3 hours (depends on n_replicas chosen)
**Go/No-Go Decision Target**: 2026-04-25
