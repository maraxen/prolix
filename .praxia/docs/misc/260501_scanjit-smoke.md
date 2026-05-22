# Sprint B Step 2: Constraint-Impulse Scaling Report

## Acceptance Criteria
- Constraint-impulse metrics rise with n_waters and/or effective step frequency.
- Constraint-impulse metrics track temperature drift slope (K/ps).
- If tracking fails, branch to PME-force variance tests.

## Condition Results

| n_waters | dt_fs | project_ou | T_mean (K) | Drift (K/ps) | Impulse mean | Impulse p95 | Impulse/water | Impulse/dof |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.50 | True | 296.26 | -67.2378 | 5.1586e-02 | 5.2481e-02 | 6.4482e-03 | 1.1463e-03 |
