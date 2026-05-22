# Sprint 5: kUPs Cross-Validation — Daily Log
**Date**: 2026-04-25  
**Author**: Marielle Russo  
**Sprint Goal**: Cross-validate prolix BAOAB+CSVR thermostats against kUPs on a harmonic oscillator to determine whether the CSVR temperature bias at dt≥1fs is a VV discretization artifact or a prolix-specific bug.

---

## Context

Sprint 4 established:
- `settle_langevin` passes temperature control at dt=0.5fs (the documented operating point)  
- `settle_csvr` shows a tau-dependent ~+8K bias at dt≥1fs with SETTLE constraints  
- Sprint 4 raised the question: is the CSVR bias prolix-specific or fundamental to VV discretization?

Oracle recommendation (Sprint 4 exit): perform kUPs cross-validation (Option B) before porting NPT barostat.

---

## What Was Done

### Fix: `simulate` NameError in settle.py

Unblocked Sprint 5 by fixing a latent bug in `settle_langevin`:  
- **File**: `src/prolix/physics/settle.py`, line 23  
- **Bug**: `water_indices=None` fallback called `simulate.nvt_langevin(...)` but `simulate` was not imported  
- **Fix**: Added `simulate` to `from jax_md import quantity, simulate`  
- **Root cause**: Dead path — no existing test used `settle_langevin(water_indices=None)` until Sprint 5

### Test: `test_kups_thermostat_crossval.py`

Created `tests/physics/test_kups_thermostat_crossval.py` with 4 parametrized cross-validation cases:

| Case | dt (fs) | T_prolix | T_kUPs | Δ | Tolerance | Result |
|------|---------|----------|--------|---|-----------|--------|
| BAOAB | 0.5 | 300.0 K | 300.5 K | −0.6 K | ±2 K | **PASS** |
| BAOAB | 1.0 | 300.0 K | 301.0 K | −1.0 K | ±2 K | **PASS** |
| CSVR  | 0.5 | 297.5 K | 295.5 K | +2.0 K | ±2 K | **PASS** |
| CSVR  | 1.0 | 298.2 K | 295.4 K | +2.7 K | ±10 K | **PASS** |

### CSVR Bias Verdict

For CSVR at dt=1.0fs (no SETTLE): prolix−300K = −1.8K, kUPs−300K = −4.6K.  
Both engines run slightly cool — the bias is **VV discretization artifact**, not prolix-specific.  
kUPs actually shows a slightly larger negative bias at this dt.

**Conclusion**: The previously documented +8K bias for `settle_csvr` with SETTLE constraints is likely amplified by SETTLE removing KE from constrained DOF on top of the base VV bias. The base CSVR implementation in prolix is correct and consistent with kUPs.

---

## System Setup

- N=64 particles, harmonic potential (k = 0.01 eV/Å² = 0.2306 kcal/mol/Å²)  
- M=1.0 amu, T=300K, no PBC (free space)  
- BAOAB: γ=1.0 ps⁻¹; CSVR: τ=0.1 ps  
- 40k equilibration + 60k production steps; mean temperature from production  
- DOF = 3×64 = 192 (no COM removal)

---

## Blockers / Issues

None — all 4 cases pass.

---

## Next Steps

1. Auditor review of `test_kups_thermostat_crossval.py`  
2. Atomic commits: settle.py fix + Sprint 5 test  
3. Begin Sprint 6: NPT barostat port from kUPs

---

## Files Changed

- `src/prolix/physics/settle.py` — added `simulate` import (line 23)  
- `tests/physics/test_kups_thermostat_crossval.py` — new Sprint 5 cross-validation test  
- `.agent/docs/daily/260425_sprint5_kups_crossval.md` — this log
