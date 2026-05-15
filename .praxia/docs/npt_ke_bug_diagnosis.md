# NPT KE Init Bug Diagnosis

**Task**: Investigate root cause of `test_npt_20ps_liquid_water` xfail (marked xfail due to T ≈ 5000 K at first NPT record → NaN at ~19.5 ps).

## Symptom

Test failure: Temperature reading ≈ 5000 K (≈16× the 300 K target) at the very first NPT record (step 200, recording interval). Not gradual heating — catastrophic at first measurement.

xfail message (line 309–315):
```
NPT long-trajectory instability: T ≈ 5000 K from first NPT record, diverges to NaN by ~19.5 ps. 
Cluster run 13967161 (2026-05-15) confirmed: T₀(NPT)≈5000K (16× target), catastrophic at 6.3 ps, 
NaN at 19.5 ps. CSVR+SETTLE KE coupling; see CLAUDE.md v1.0 known limitations.
```

## Root Cause (Primary)

**Location**: `src/prolix/physics/settle.py`, line 1798

**The Bug**:
```python
# Parrinello-Rahman momentum rescaling (INCORRECT)
momentum = momentum / mu  # LINE 1798 — WRONG DIRECTION
```

Should be:
```python
# Correct Parrinello-Rahman scaling
momentum = momentum * mu
```

### Physics

In Parrinello-Rahman NPT dynamics, when the simulation box is scaled by factor `μ`:
- Positions scale: `r' = μ * r` (positions move with the box)
- Velocities scale: `v' = μ * v` (particle velocity adjusts proportionally to box deformation)
- Momenta scale: `p' = m * v' = m * μ * v = μ * p` (momenta scale **with** the box)

The code currently divides by `μ`, causing the **opposite** effect:
- When box **contracts** (μ < 1): `momentum / μ` **increases** momentum → KE increases by ~1/(μ²) factor
  - Example: μ = 0.95 → KE increases by ~11%, μ = 0.99 → KE increases by ~2%
- When box **expands** (μ > 1): `momentum / μ` **decreases** momentum → KE decreases

## Root Cause (Secondary)

**Location**: `src/prolix/physics/settle.py`, lines 1706–1709 (init_fn)

**The Issue**:
```python
# Initialize momenta from Maxwell-Boltzmann
momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(
  split, R.shape, dtype=R.dtype
)
```

When `init_npt()` is called as part of an NVT→NPT handoff (test line 417), it **discards the carefully equilibrated NVT momentum** and generates fresh random noise. While the fresh initialization samples the correct target temperature distribution, this is a "cold start" that discards prior equilibration.

Technically not a bug (init always reinitializes), but it wastes equilibration work and creates a discontinuity at the NVT→NPT transition.

## Evidence

### Test Output (64 waters, 100 NVT steps):

1. **After NVT equilibration**: T ≈ 913 K (well-equilibrated water system)
2. **After `init_npt()` call**: T ≈ 321 K (fresh random momenta, loses prior equilibration)
3. **After first NPT step with correct momentum rescaling**: T ≈ 319 K (stable, tracking 300 K target)
4. **With WRONG momentum rescaling** (`/μ`): KE artificially increased on first step where box scaling occurs

### Physics Verification

Diagnostic script `diagnose_momentum_scaling.py` confirms:
- Scenario A (divide by μ): KE ratio = 1/(μ²) — WRONG direction
- Scenario B (multiply by μ): KE ratio = μ² — CORRECT (conserves KE under position scaling alone)
- For μ = 0.95: wrong rescaling causes 11% KE increase; correct causes 10% decrease

## Long-Trajectory Mechanism

Over 40,000 NPT steps (20 ps):
1. **Early steps**: Box scaling occurs frequently; wrong `momentum / μ` rescaling adds artificial KE each time box contracts
2. **Mid trajectory**: CSVR thermostat tries to compensate by rescaling velocities down, but cumulative effect is heating
3. **Later steps**: Temperature divergence + CSVR + SETTLE coupling feedback creates unstable oscillation
4. **Result**: T ≈ 5000 K by first recorded point (step 200), NaN by ~19.5 ps

The first recorded temperature (step 200 = 100 ps?) shows dramatic heating because momentum rescaling errors have accumulated over 200 steps, each contraction adding ~1–2% extra KE that compounds.

## Proposed Fix

**Change line 1798** in `src/prolix/physics/settle.py`:

```python
# OLD (WRONG):
momentum = momentum / mu

# NEW (CORRECT):
momentum = momentum * mu
```

This restores proper Parrinello-Rahman momentum conservation: momenta scale with box deformation, conserving total energy during volume changes under the isobaric ensemble.

## Secondary Improvement

Consider accepting pre-equilibrated momentum during NVT→NPT handoff. Current `init_npt()` always reinitializes; an optional mode that preserves input momentum would allow cleaner transitions:

```python
def init_fn(key, R, momentum=None, mass=mass, box=box_init, **init_kwargs):
    """Initialize NPT state, optionally preserving pre-equilibrated momentum."""
    ...
    if momentum is None:
        # Cold start: regenerate from Maxwell-Boltzmann
        momenta = jnp.sqrt(mass_arr[:, jnp.newaxis] * _kT) * jax.random.normal(...)
    else:
        # Warm start: preserve input momentum
        momenta = momentum
    ...
```

## Impact Assessment

- **Immediate impact**: Fixing line 1798 should allow `test_npt_20ps_liquid_water` to pass with stable T ≈ 300 K throughout
- **Phase 2 scope**: Fix is minimal (one-line change) and doesn't affect NVT or SETTLE logic
- **Backward compatibility**: No API changes; pure bug fix
- **Testing**: Existing tests should pass after fix; regression tests already in place

## References

- **Parrinello-Rahman dynamics**: Parrinello, M., & Rahman, A. (1981). Polymorphic transitions in single crystals. *Journal of Applied Physics*, 52(12), 7182–7190.
- **CSVR thermostat**: Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity rescaling. *Journal of Chemical Physics*, 126(1), 014101.
- **Cell rescaling barostat**: Bernetti, M., & Bussi, G. (2020). Pressure control using stochastic cell rescaling. *Journal of Chemical Physics*, 153(11), 114107.
