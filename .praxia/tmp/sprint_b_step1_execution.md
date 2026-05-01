# Sprint B Step 1: Discriminator Test Execution Log

## Test Specification

**Objective**: Determine whether 64-water NVT temperature drift (334 K vs 300 K target) is caused by:
- **H1**: Integration/projection error (discretization bias)
- **H3**: Force-side mechanisms (PME grid coupling with box size)

**Test Configuration**:
- Timestep: dt = 0.25 fs (half of standard 0.5 fs)
- System: 64 TIP3P water molecules
- Duration: 100 ps total (400,000 steps at 0.25 fs/step)
- Burn-in: First 33 ps (133,333 steps) discarded
- Production: Last 67 ps (266,667 steps) averaged
- Thermostat: Langevin, gamma = 1.0 ps^-1, kT = 300 K
- Seed: 7 (same as failing baseline test)
- Energy function: PME (32 grid points, alpha=0.34, cutoff=9.0 Å)
- Constraints: SETTLE rigid constraints on water
- Projection: project_ou_momentum_rigid=True, projection_site="post_o"

## Test Execution Details

**Invocation**:
```python
from tests.physics.test_settle_temperature_control import _mean_rigid_t_after_burn

T_obs, T_target = _mean_rigid_t_after_burn(
    dt_fs=0.25,
    n_waters=64,
    seed=7,
    steps=400_000,
    burn=133_333
)
```

**Infrastructure**:
- Virtual environment: `/home/marielle/projects/prolix/.venv`
- Test file: `/home/marielle/projects/prolix/tests/physics/test_settle_temperature_control.py`
- Computation: CPU (CUDA not available, JAX x64 enabled)
- Expected runtime: 45-90 minutes on single CPU core

## Hypothesis Decision Tree

Based on T_obs at dt=0.25 fs vs dt=0.5 fs baseline (T_obs=334.3 K):

### Case A: T_obs ≈ 300 ± 10 K
**Interpretation**: H1 CONFIRMED (integration/projection error)
**Mechanism**: Per-step O(dt^2) bias in OU/SETTLE discretization
**Implication**: Reducing dt significantly improves temperature control
**Fix pathway**: Implement constraint-aware OU or variable-dt integration
**Recommendation**: Option A (FIX AVAILABLE)

### Case B: T_obs ≈ 320-330 K
**Interpretation**: Mixed mechanism (H1 + H3 coupled)
**Mechanism**: Both discretization error AND force-side variance contributing
**Implication**: Slight improvement from dt reduction, but not fully resolved
**Next step**: Run Step 2 PME grid scaling test to isolate force component
**Recommendation**: Option A (FIX AVAILABLE) but multi-part solution required

### Case C: T_obs ≈ 330-340 K
**Interpretation**: H3 CONFIRMED (force-side mechanism)
**Mechanism**: PME grid coupling with box size (not integration error)
**Implication**: Reducing dt does NOT improve temperature control
**Next step**: Run Step 2 PME grid scaling test to confirm; implement grid scaling fix
**Recommendation**: Option A (FIX AVAILABLE) but different fix mechanism

### Case D: T_obs > 340 K
**Interpretation**: Unmodeled mechanism (H4 or unknown)
**Mechanism**: Unknown accumulation or JAX autodiff issue
**Implication**: Smaller timestep worsens or has no effect
**Next step**: Run Step 3 sign-analysis test; revisit assumptions
**Recommendation**: Option B (ENVELOPE + Sprint 12 investigation required)

## Expected Output Format

```
Sprint B Step 1 Result:

Test Condition: 64 waters, dt=0.25 fs, 100 ps NVT
T_observed: XXX.X K (±Y K)
T_target: 300.0 K
Deviation: ±ZZ K

Comparison to dt=0.5 fs baseline:
- dt=0.5 fs: T_obs = 334.3 K
- dt=0.25 fs: T_obs = ??? K
- Delta: ??? K

Interpretation: [Case A/B/C/D]
```

## Key Files & Parameters

- Test module: `src/prolix/physics/settle.py` (settle_langevin implementation)
- Temperature measurement: `src/prolix/physics/rigid_water_ke.py` (rigid_tip3p_box_ke_kcal)
- System setup: `tests/physics/test_explicit_langevin_tip3p_parity.py` (_grid_water_positions, _prolix_params_pure_water)

## Baseline Comparison

**dt=0.5 fs baseline** (from xfail test docstring):
- Steps: 200,000 (100 ps at 0.5 fs/step)
- Burn: 66,667 steps (33 ps)
- n_waters: 8
- T_obs: 334.3 K (reported in test failure)
- Interpretation: Significant drift (+34.3 K above 300 K target)

**This test (dt=0.25 fs)**:
- Steps: 400,000 (100 ps at 0.25 fs/step)
- Burn: 133,333 steps (33 ps)
- n_waters: 64 (larger system to replicate failing scenario)
- T_obs: [PENDING EXECUTION]

## Notes on Test Design

1. **Matching wall-clock time**: Both tests run for 100 ps despite different dt values, ensuring comparable thermodynamic trajectory length.

2. **Doubling water count**: Moving from n_waters=8 (baseline test) to n_waters=64 better replicates the failing scenario size and may reveal system-size-dependent effects.

3. **Same seed and parameters**: Using seed=7, same PME grid settings, and identical thermostat parameters ensures only dt varies systematically.

4. **Temperature from rigid-body KE**: Measurement uses `rigid_tip3p_box_ke_kcal`, which decomposes KE into COM + rotational contributions, crucial for diagnosing SETTLE constraint interactions.

## Status

- Created: 2026-04-28 10:00 UTC
- Execution started: Background task bgx80qec9
- Expected completion: 2026-04-28 12:00-14:00 UTC (pending CPU load)
- Result location: This file will be updated with result when complete
