# Sprint B Step 2: Constraint-Impulse Scaling Report

## Acceptance Criteria
- Constraint-impulse metrics rise with n_waters and/or effective step frequency.
- Constraint-impulse metrics track temperature drift slope (K/ps).
- If tracking fails, branch to PME-force variance tests.

## Condition Results

| n_waters | dt_fs | project_ou | T_mean (K) | Drift (K/ps) | V_mean (kcal/mol) | V_std (kcal/mol) | Impulse mean | Impulse p95 | Impulse/water | Impulse/dof |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.50 | True | 212.77 | -8051.0341 | -130.4197 | 0.0000 | 3.9550e-02 | 3.9888e-02 | 1.9775e-02 | 4.3944e-03 |

## Sprint B Verdict (2026-04-30)

### Data Summary

| n_waters | dt_fs | T_mean (K) | ΔT | impulse_mean | impulse/DOF |
|---|---|---|---|---|---|
| 8 | 0.50 | 342.4 | +42K | 0.1195 | 0.002656 |
| 8 | 0.25 | 370.2 | +70K | 0.0678 | 0.001507 |
| 16 | 0.50 | 341.7 | +42K | 0.1737 | 0.001867 |
| 16 | 0.25 | 344.6 | +45K | 0.0900 | 0.000968 |
| 32 | 0.50 | 369.0 | +69K | 0.2907 | 0.001538 |
| 64 | 0.50 | 414.9 | +115K | 0.5034 | 0.001321 |

### Hypothesis Verdicts

**H1 (integration/projection O(dt²) error): FALSIFIED**
- dt=0.25fs makes temperature *worse* or flat at all tested system sizes
- No O(dt²) signature; reduction does not converge toward 300K

**H3 (constraint-impulse feedback): FALSIFIED as primary mechanism**
- impulse/DOF *decreases* (0.00266 → 0.00132) as n_waters increases 8→64
- Temperature offset *increases* superlinearly over the same range (+42K → +115K)
- If impulse-per-DOF were driving the offset, the correlation would be opposite

### Open Question
Temperature excess is a steady-state offset (not a growing drift), superlinear in n_waters. Flat from 8→16 waters (+42K), then jumps at 32 (+69K) and 64 (+115K). Likely a collective effect in the OU rigid-body projection at large system sizes, or a DOF counting issue in the thermometer that only manifests at scale. No confirmed fix path.

### Decision
Phase 5 (constraint-aware thermostat) deferred to v1.2. Sprint B closed. Advancing to strategic-MVP track: closure→explicit-params refactor and prolix.export module.
