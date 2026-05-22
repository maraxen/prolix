# Sprint B Step 2: Constraint-Impulse Scaling Report

## Objective
Test whether SETTLE velocity-constraint impulse accumulation explains NVT temperature drift as a function of system size and timestep.

## Acceptance Criteria
- Constraint-impulse metrics increase with `n_waters` and/or effective step frequency (`dt` decrease).
- Constraint-impulse metrics track temperature drift slope (`K/ps`) across conditions.
- If tracking is weak or absent, branch to PME-force variance pathway.

## Experiment Matrix
- `n_waters`: 8, 16, 32, 64
- `dt_fs`: 0.50, 0.25
- control: `project_ou_momentum_rigid=False` at 64 waters (both `dt` values)

## Metrics
- `T_mean`, `T_std`, `drift_k_per_ps`
- `settle_impulse_mean`, `settle_impulse_rms`, `settle_impulse_p95`, `settle_impulse_p99`
- normalized impulse: `settle_impulse_per_water_mean`, `settle_impulse_per_dof_mean`

## Results Table
| n_waters | dt_fs | project_ou | T_mean (K) | Drift (K/ps) | Impulse mean | Impulse p95 | Impulse/water | Impulse/dof |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.50 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | 0.25 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | 0.50 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | 0.25 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 32 | 0.50 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 32 | 0.25 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | 0.50 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | 0.25 | True | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | 0.50 | False | TBD | TBD | TBD | TBD | TBD | TBD |
| 64 | 0.25 | False | TBD | TBD | TBD | TBD | TBD | TBD |

## Decision
- **Primary root-cause verdict**: TBD
- **Evidence**: TBD
- **Next step**: TBD (constraint-thermostat coupling fix or PME-force variance branch)
