# Sprint 4: Temperature Control Validation
Date: 2026-04-25
Status: COMPLETE

## Summary

Added passing temperature validation tests for the settle_langevin production
operating point (dt=0.5fs) and documented the known CSVR bias with a runtime warning.

## Changes

### settle.py — CSVR runtime warning
- Added `import warnings`
- `settle_csvr`: emits `UserWarning` when `dt >= 1 fs` (AKMA threshold: 1/48.888)
  documenting the tau-dependent ~+8K temperature bias
- `settle_csvr` docstring: added Note section with root cause analysis (VV
  discretization artifact), practical impact, and guidance

### test_settle_temperature_control.py — new green tests + xfail updates
- `test_temperature_langevin_dt0_5fs_green` (slow): canonical validation of
  settle_langevin at dt=0.5fs — n_waters=8, 100ps total, ±10K tolerance.
  This is the first passing end-to-end temperature test at the production constraint.
- `test_equipartition_per_molecule_com_dt0_5fs` (slow): per-molecule COM velocity
  KS test at dt=0.5fs. Structurally valid under SETTLE (COM motion preserved).
  Subsamples every 2000 steps (decorrelation time ~1ps), 120 snapshots.
- Updated xfail reasons at lines 159, 177: now reference runtime warning
  in settle.py settle_csvr

## Notes

- KUSP (user's proposed JAX reference) not found in codebase or dependencies.
  Oracle recommended using OpenMM (already wired) as the reference framework.
  KUSP follow-up deferred: user to clarify what KUSP refers to.
- OpenMM T-comparison test deferred: existing parity infrastructure needs
  more refactoring than budgeted for this sprint. Can be added in follow-up.
- CSVR root cause confirmed as VV discretization; not fixable without redesign
  (v2.0 constraint-aware thermostat). Warning + docstring is the correct v1.0 response.
