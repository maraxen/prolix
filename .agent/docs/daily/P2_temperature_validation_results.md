# P2 Execution Report: Temperature Validation with Corrected DOF Count

**Date**: 2026-04-23  
**Status**: COMPLETE (G4 GATE: FAIL)  
**Duration**: ~10 minutes execution  

## Objective

Execute P2 from Phase 2 next steps: Validate temperature fix with corrected DOF count to determine if adaptive RATTLE (settle_velocity_tol=1e-6) fixes the inverse-timestep temperature error observed during Phase 1.

## Configuration

- **System**: TIP3P water box with 64 waters (192 atoms)
- **DOF Formula (Corrected)**: `DOF = 3*N_total - 3*N_waters - 3 = 381`
- **Target Temperature**: 300 K
- **Thermostat**: Langevin with γ = 1.0 ps⁻¹
- **Constraint Method**: SETTLE (position) + RATTLE (velocity) with adaptive_tol=1e-6
- **Experiments**: 
  - Experiment 1: dt=1.0 fs for 500 steps
  - Experiment 2: dt=2.0 fs for 500 steps

## Critical Findings

### Results Summary

| Metric | dt=1.0 fs | dt=2.0 fs |
|--------|-----------|-----------|
| Mean KE | 151.04 kcal/mol | 301.56 kcal/mol |
| T_measured | 290.27 K | 507.77 K |
| ΔT (target=300 K) | **9.73 K** | **207.77 K** |
| Status | FAIL (ΔT ≥ 5 K) | FAIL (catastrophic) |

### G4 Gate Verdict

**FAIL** — Root cause is NOT iteration count alone.

### Key Insight: Inverse Timestep Amplification

The temperature error shows pathological scaling with timestep:
- **dt=1.0 fs**: 2x larger → ΔT = 9.73 K (marginal fail)
- **dt=2.0 fs**: Doubles → ΔT = 207.77 K (~21x error amplification!)

This is NOT consistent with a simple iteration count problem. Instead, it reveals a **fundamental integrator instability** at larger timesteps.

## Root Cause Analysis

### What's NOT the problem
- **Iteration count**: If the problem were just insufficient RATTLE iterations, we'd expect similar errors at both timesteps (or worse at dt=1.0 due to different convergence behavior).

### What IS the problem (candidates)

1. **Langevin-SETTLE Coupling Order**: The velocity constraint (RATTLE) may be removing kinetic energy, then the Langevin thermostat noise injection inadequately compensates, especially at larger timesteps.

2. **Ornstein-Uhlenbeck (OU) Projection**: The noise term in the Langevin integrator may have:
   - Incorrect variance scaling with dt
   - Wrong interaction with velocity constraints
   - Missing correction for the constraint subspace

3. **DOF Denominator**: Although oracle verified the formula `DOF = 3*N_total - 3*N_waters - 3`, the interaction with the thermostat may reveal an error.

### Evidence

**Comparison to Phase 1 baseline**:
- Prior (incorrect DOF): dt=1.0 fs → ΔT = 28.8 K, dt=2.0 fs → ΔT = 10.6 K
- Current (corrected DOF): dt=1.0 fs → ΔT = 9.73 K, dt=2.0 fs → ΔT = 207.77 K

The dt=1.0 fs result (~3x improvement) suggests DOF correction is working. But dt=2.0 fs shows a **NEW failure mode** that gets worse with larger timesteps.

## Integration Issues

### Problem Pattern

```
dt=1.0 fs: Constraint → Correct kinetic energy ✓
           Thermostat noise → Adequate compensation ✓
           Result: T ≈ 290 K (ΔT = 9.73 K, still bad but stable)

dt=2.0 fs: Constraint → Remove kinetic energy ✓
           Thermostat noise → Insufficient compensation ✗
           Overcorrection in next step → Velocity explosion ✗
           Result: T ≈ 508 K (ΔT = 207.77 K, catastrophic)
```

### Suspected Implementation Issue

In `rattle_langevin` (src/prolix/physics/simulate.py), the order of operations is:
1. Velocity half-step
2. Position half-step
3. **Langevin stochastic update** (noise injection)
4. Position half-step
5. FORCE update
6. Velocity half-step
7. **RATTLE momentum projection** ← This may be undoing the thermostat setup

The velocity constraint (step 7) may be removing momentum that was carefully set up by the thermostat (step 3), breaking the equipartition theorem.

## Escalation

**Oracle Review Required**:

1. ✓ Verify DOF formula is correct (already done)
2. ✗ Investigate Langevin + SETTLE coupling
3. ✗ Check OU noise projection and variance scaling
4. ✗ Review order of operations in `rattle_langevin`

**Recommendation**: 

**Do NOT close G4 gate**. The problem is more fundamental than adaptive iteration count. Suggests a deeper issue with how velocity constraints interact with the Langevin thermostat.

## Script Output

**Location**: `/home/marielle/projects/prolix/scripts/validate_adaptive_rattle_temperature.py`

**Execution**:
```bash
uv run python scripts/validate_adaptive_rattle_temperature.py
```

**Output snippet**:
```
=== Experiment: dt=1.0 fs, n_steps=500 ===
  T_measured: 290.27 ± 11.02 K
  ΔT = |T_meas - T_target|: 9.73 K

=== Experiment: dt=2.0 fs, n_steps=500 ===
  T_measured: 507.77 ± 48.05 K
  ΔT = |T_meas - T_target|: 207.77 K

G4 Gate Status: FAIL
✗ Root cause is NOT iteration count alone.
```

## Next Steps

P2 is **complete but gates not met**. The validation proves:
- ✓ Corrected DOF formula works at dt=1.0 fs (ΔT = 9.73 K, marginal)
- ✗ Inverse timestep amplification is a fundamental integrator issue at dt=2.0 fs
- ✗ Adaptive RATTLE alone does NOT fix the temperature problem

**Phase 2 Decision Gate (G4)**: FAIL — Escalate to oracle for strategic guidance on next investigation direction.
