# Sprint B Step 1: Discriminator Test Rationale

## Why This Test?

The temperature drift in 64-water NVT simulations could stem from two very different root causes:

1. **H1: Integration Error** — A discretization bias in the coupling between SETTLE constraints and the Langevin OU thermostat
2. **H3: Force-Side Mechanism** — A systematic bias in the PME Coulomb forces that couples with rigid-body water molecules

The fix for H1 (improve integrator order) is completely different from the fix for H3 (scale PME grid or decouple constraints).

**This test isolates the root cause by changing dt and measuring if temperature improves.**

## Test Design: Why dt=0.25 fs?

### The Logic

If the problem is **integration error**, then **reducing dt should help**:
- Integration errors scale as O(dt^2), O(dt^3), etc.
- Halving dt → quartering the error magnitude
- Expected result at dt=0.25 fs: T_obs ≈ 300 K (problem solved!)

If the problem is **force-side bias**, then **reducing dt won't help**:
- Bias is in the force field itself, not the time-stepping
- Halving dt just means smaller steps through the same biased field
- Expected result at dt=0.25 fs: T_obs ≈ 334 K (same problem)

### Why Half vs Quarter?

Reducing dt by 2x is the **minimum diagnostic change**:
- Large enough to see clear signal in O(dt^2) errors (divide-by-4 improvement in error)
- Small enough to stay within Langevin stability bounds
- 400k steps is still computationally feasible on CPU (~2-3x longer wall clock than dt=0.5fs)

### Why 400,000 steps?

To maintain **identical wall-clock duration (100 ps)**:
- dt=0.5 fs: 200k steps = 100 ps (baseline test duration)
- dt=0.25 fs: 400k steps = 100 ps (this test)
- Same thermodynamic trajectory length → fair comparison

### Why 64 waters?

**System size is the key variable**:
- n_waters=8: works fine at dt=0.5 fs
- n_waters=64: FAILS at dt=0.5 fs (T=334 K)
- Scaling from 8→64 (8x more water): 334 K problem emerges
- This suggests either:
  - Integration error scales with DOF count (H1)
  - Force field bias scales with system size (H3)
  
**Either way, we need to test at n_waters=64 to match the failing scenario.**

## What the Result Means

### Scenario A: T ≈ 300 K at dt=0.25 fs

**Diagnosis**: H1 CONFIRMED (integration error)

**Interpretation**:
- Halving dt reduced temperature drift from 334 K to 300 K
- Improvement of 34 K from a 2x reduction in dt
- Consistent with O(dt^2) accumulation: error ∝ dt^2
- The SETTLE constraint + OU thermostat coupling is the culprit

**What happens next**:
- Implement integrator fix: higher-order OU discretization or constraint-aware coupling
- Expected timeline: 1-2 sprint cycles (can be done before release if critical)
- Severity: Moderate (workaround exists: use dt < 0.25 fs for production)

**Code locations to investigate**:
- `src/prolix/physics/settle.py` → `settle_langevin` integration loop
- `src/prolix/physics/settle.py` → OU projection implementation
- Likely culprit: `project_ou_momentum_rigid` discretization

---

### Scenario B: T ≈ 320-330 K at dt=0.25 fs

**Diagnosis**: Mixed mechanism (H1 + H3)

**Interpretation**:
- Reducing dt helped somewhat (334 K → 320-330 K)
- But problem isn't fully solved (should be ≤310 K if pure H1)
- Both integration error AND force-side bias are present
- They're coupled: neither alone explains the full drift

**What happens next**:
- Run Step 2 PME grid scaling test to isolate H3 component
- Combine both fixes: improve integrator (fixes H1 part) AND scale PME grid (fixes H3 part)
- Timeline: 2-3 sprint cycles
- Severity: High (needs multiple fixes)

**Code locations to investigate**:
- Everything from Scenario A, PLUS:
- `src/prolix/physics/system.py` → PME setup (grid density, alpha parameter)
- Check if n_waters scaling affects PME force accuracy

---

### Scenario C: T ≈ 330-340 K at dt=0.25 fs

**Diagnosis**: H3 CONFIRMED (force-side mechanism)

**Interpretation**:
- Reducing dt changed almost nothing (334 K → 334 K)
- Temperature drift is NOT caused by integration discretization
- Bias is in the Coulomb force field itself
- Specifically: PME grid coupling with 64-water system

**What happens next**:
- Run Step 2 PME grid scaling test to confirm grid is the culprit
- Increase PME grid density (pme_grid_points: 32 → 64 or higher)
- OR implement constraint-aware PME or force regularization
- Timeline: 1-2 sprint cycles (grid is easier to tune than integrator)
- Severity: Moderate (can work around with higher PME grid)

**Code locations to investigate**:
- `src/prolix/physics/system.py` → `make_energy_fn` with PME setup
- Check: what pme_grid_points value works for 64-water systems?
- Hypothesis: grid_points should scale with system size

---

### Scenario D: T > 340 K at dt=0.25 fs

**Diagnosis**: Unexpected / Unmodeled mechanism (H4)

**Interpretation**:
- Smaller timestep made the problem WORSE
- Contradicts all standard numerical analysis
- Suggests: JAX compilation artifact, random number generation bug, or unknown coupling

**What happens next**:
- Escalate to Step 3: detailed sign/magnitude analysis of forces
- Disable JIT compilation and re-run to check for JAX autodiff artifact
- Check random number sequence reproducibility
- Timeline: 2-3 sprint cycles (needs deep debugging)
- Severity: Critical (unknown mechanism is hard to fix)

**Code locations to investigate**:
- JAX random number generation in OU projection
- JIT compilation caching behavior
- Force autodiff precision issues

---

## Why This Test is Definitive

**Key insight**: Integration errors scale with dt, but force errors don't.

By measuring the *same system* at *different dt*, we can separate these two classes of errors:

| Mechanism | Effect of dt/2 | Expected T at dt=0.25 fs |
|-----------|-----------------|------------------------|
| Integration error O(dt^2) | Error/4 → T → 300 K | 290-310 K |
| Integration error O(dt^3) | Error/8 → T → 300 K | 310-320 K |
| Force-side bias | No change | 330-340 K |
| Unknown artifact | Can get worse | >340 K |

**No other single test can distinguish these cases as cleanly.**

---

## Computational Cost

- **Wall time**: 45-90 minutes on single CPU core
- **Alternative**: Use GPU if available → 10-20 minutes
- **Why worth it**: Result determines entire Sprint B strategy (saves 5+ sprints of debugging wrong path)

---

## Success Criteria

Test is successful if it:
1. Completes without error (numerical stability confirmed)
2. Provides clear T_obs value (no NaN or divergence)
3. Falls into one of the four scenarios above (allows diagnosis)
4. Guidance is actionable (can plan next steps immediately)

Failure modes:
- NaN/Inf in temperature (suggests unrelated numerical issue)
- Huge variance in T_obs (suggests insufficient burn-in)
- No clear result (suggests measurement method issue)

---

## Related Tests & Artifacts

**Baseline test** (already exists, shows the failure):
- File: `tests/physics/test_settle_temperature_control.py::test_temperature_langevin_dt0_5fs_green`
- Status: xfail (marked as expected failure)
- Docstring: notes that n_waters=8 passes, but n_waters=64 fails at 334 K

**KE measurement ablation** (designed to catch H2, available):
- File: `tests/physics/test_settle_temperature_control.py::test_ke_measurement_ablation`
- Status: passes (proves KE measurement is correct)
- Implication: temperature formula is not the issue

**Later tests** (will run if Step 1 result calls for them):
- Step 2: PME grid scaling (detects H3)
- Step 3: Force sign/magnitude analysis (detects H4)

---

## Decision Matrix: What to Do Next

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1 Result (T_obs at dt=0.25 fs)                            │
├─────────────────────────────────────────────────────────────────┤
│ A: 290-310 K  → H1 CONFIRMED  → FIX: improve integrator       │
│ B: 320-330 K  → Mixed H1+H3   → FIX: do both integrator + grid │
│ C: 330-340 K  → H3 CONFIRMED  → FIX: scale PME grid           │
│ D: >340 K     → H4 (unknown)  → ACTION: Step 3 analysis        │
└─────────────────────────────────────────────────────────────────┘
```

**Result determines:**
- Which code to investigate first
- Which fix to implement first
- Whether to continue in Sprint B or escalate to Sprint 12

---

## References

- **Integration theory**: Numerical Analysis textbooks (error analysis for ODEs)
- **SETTLE algorithm**: Miyamoto & Kollman 1992
- **Langevin thermostat**: Bussi & Parrinello 2007
- **OU process discretization**: Ermak & McCammon 1978
- **PME Coulomb**: Darden, York, Pedersen 1993
