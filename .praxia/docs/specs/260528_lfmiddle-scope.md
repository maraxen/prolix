# LFMiddle Scope Decision (Oracle)

**Date:** 2026-05-28  
**Task ID:** `260528_lfmiddle_hypothesis`  
**Verdict:** **GO** (exploration sprint)

## Conflict resolved

| Source | Position |
|--------|----------|
| v1.1_next_steps | LFMiddle first — falsifiable dt sweep |
| 260430 v1-1-roadmap | Dropped after Sprint B (init artifact) |
| ADR-005 v2 | dt sweep is **falsification**, not a commitment to lift dt cap |

**Resolution:** Revive LFMiddle as a **bounded exploration** (not P0 production blocker). Outcome is informative whether dt=1.0 fs passes or fails.

## Shipped in this sprint

- `settle_lfmiddle_langevin` — monolithic SETTLE + split-O path
- `lfmiddle_langevin` sequence + `Force_Step` in modular `make_integrator`
- `tests/physics/test_lfmiddle_dt_sweep.py` — module xfail removed
- `scripts/experiments/lfmiddle_dt_sweep.py` + bath sidecar

## Bathos

- Campaign `89c9a900` (`lfmiddle-dt-sweep`, exploration)
- Script: `scripts/experiments/lfmiddle_dt_sweep.py` + sidecar
- Local smoke runs registered in catalog (short trajectories fail ±5 K gate until longer burn/GPU)

## Not in scope

- Full modular SETTLE + LFMiddle parity with `baoab_csvr_npt`
- Constraint-aware thermostat (Phase 5)

## Follow-up

- Post-force A-step uses `half_dt` (2026-05-28 fix); full dt-sweep validation remains on `tests/physics/test_lfmiddle_dt_sweep.py` (slow, mark `@pytest.mark.slow`)
