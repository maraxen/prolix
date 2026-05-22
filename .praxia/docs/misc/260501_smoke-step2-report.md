# Sprint B Step 2: Constraint-Impulse Scaling Report

## Acceptance Criteria
- Constraint-impulse metrics rise with n_waters and/or effective step frequency.
- Constraint-impulse metrics track temperature drift slope (K/ps).
- If tracking fails, branch to PME-force variance tests.

## Condition Results

| n_waters | dt_fs | project_ou | T_mean (K) | Drift (K/ps) | Impulse mean | Impulse p95 | Impulse/water | Impulse/dof |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.50 | True | 297.13 | -380.0179 | 5.1684e-02 | 5.2941e-02 | 6.4605e-03 | 1.1485e-03 |
| 8 | 0.25 | True | 297.38 | 19.4054 | 2.5733e-02 | 2.5975e-02 | 3.2166e-03 | 5.7184e-04 |
| 16 | 0.50 | True | 272.27 | -1081.9577 | 7.4956e-02 | 7.5248e-02 | 4.6848e-03 | 8.0598e-04 |
| 16 | 0.25 | True | 266.27 | -1420.9761 | 3.7388e-02 | 3.7539e-02 | 2.3368e-03 | 4.0202e-04 |
| 32 | 0.50 | True | 299.94 | 954.7984 | 1.0518e-01 | 1.0618e-01 | 3.2868e-03 | 5.5650e-04 |
| 32 | 0.25 | True | 300.86 | 268.6659 | 5.2709e-02 | 5.3234e-02 | 1.6472e-03 | 2.7888e-04 |
| 64 | 0.50 | True | 314.82 | -112.5633 | 1.5101e-01 | 1.5306e-01 | 2.3596e-03 | 3.9636e-04 |
| 64 | 0.25 | True | 318.09 | 373.5539 | 7.6899e-02 | 7.8250e-02 | 1.2015e-03 | 2.0183e-04 |
| 64 | 0.50 | False | 318.55 | -157.7773 | 5.6365e-01 | 6.0835e-01 | 8.8071e-03 | 1.4794e-03 |
| 64 | 0.25 | False | 315.43 | -469.9599 | 3.5980e-01 | 3.8770e-01 | 5.6219e-03 | 9.4436e-04 |
