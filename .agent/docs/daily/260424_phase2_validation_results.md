# v0.3.0 Phase 2C Validation Results — 2026-04-24

## Summary: FAILED — Phase 2C Thermostat Rejected

Phase 2C (constrained OU noise in rigid-body subspace) was validated against both a standalone
script and authoritative pytest tests using real TIP3P forces. All tests failed.

---

## Validation Method

Two independent test paths:

### Path A: `scripts/validate_constraint_aware_langevin.py`

Standalone script with mock energy function (no real forces), float32, manual state initialization.
Bugs found and fixed during this session:
- `water_indices` passed where `n_waters: int` expected → fixed
- `apply_fn` called without `jax.jit()` → fixed (was consuming 13GB RAM, never finishing)
- `boltzmann_kcal = 1.987e-3 / 1000.0` → fixed to `1.987e-3` (temperatures were 1000× too high)

Results (no forces, float32):
| dt    | T_mean   | E_drift  | KS p    | Verdict |
|-------|----------|----------|---------|---------|
| 0.5fs | ~346 K*  | 116.12%  | 0.2140  | FAIL    |
| 1.0fs | ~343 K*  | 90.02%   | 0.0228  | FAIL    |

*Temperature values were reported as 345,942K / 343,268K due to the k_B bug;
corrected values divide by 1000.

### Path B: `tests/physics/test_settle_temperature_control.py` (authoritative)

Real TIP3P forces, PME electrostatics, PBC, float64, correct `init_fn` usage.

Results:
| dt    | T_mean  | Target    | Verdict |
|-------|---------|-----------|---------|
| 1.0fs | 418.6 K | 300 ± 15K | **FAIL** |
| 2.0fs | 488.0 K | 300 ± 5K  | **FAIL** |
| KS equipartition | p=0.0000 | p > 0.05 | **FAIL** |

---

## Root Cause Analysis

Phase 2C generates OU noise directly in the 6D rigid-body subspace via:

1. Jacobian `J ∈ ℝ⁹×⁶` for 3 translations + 3 rotations per water
2. Gramian `G = J^T M J` (6×6)
3. Sample `ξ ~ N(0, kT * G^{-1})` via Cholesky
4. Noise momentum: `p_noise = M * J * ξ` → covariance = `kT * M * P_rigid`

The temperatures running hot (418K and 488K) suggest the noise covariance is still incorrect,
or the SETTLE_vel constraint is injecting energy that the O-step cannot remove fast enough.

The energy drift in Path A (90–116%) with no forces confirms the O-step itself is not
properly dissipating energy — SETTLE constraints are adding momentum components outside
the projected subspace on every step, and the constrained OU noise does not fully cancel this.

---

## Decision

**Phase 2C is rejected.** The constrained OU thermostat does not achieve temperature control
at dt=1.0fs or dt=2.0fs even with a carefully constructed 6D noise distribution.

**Action**: Accept v1.0 constraint (dt ≤ 0.5fs) as the production limit. Defer dt ≥ 1.0fs
to a future investigation (v0.4.0 or later) with a fundamentally different approach —
e.g., coupling the thermostat directly to unconstrained DOF (residual velocity after
SETTLE projection), rather than projecting the noise pre-SETTLE.

---

## v0.3.0 Sprint Status

- Phase 1 (implementation of `_langevin_step_o_constrained`): **complete**
- Phase 2 (validation): **FAILED** — thermostat-SETTLE coupling persists
- Phase 3 (release): **cancelled**

The v0.3.0 sprint is closed without release. CLAUDE.md already documents dt ≤ 0.5fs
as the production constraint.
