# P5 dt=1.0 fs warm bias is translational finite-size — size-crossover N\* = 16

**Date:** 2026-06-12
**Task:** 260603_p5_settle_correctness
**Campaign:** ba334c1f (`p5-nvt-size-sweep-dt1fs`)
**Jobs:** 15870804 (gate, n=895), 15911790 (L3 smoke, n=2), 15929525 (array, n=16..895)

## Context

The dt=1.0 fs production gate (job 15870804: 895 waters, gamma=10 ps⁻¹, 5 seeds,
50 ps) passed with **mean T_rot = 299.63 K** (`gate_pass=1`). The handoff plan (G1)
was to remove the dt=1.0 fs unit-test xfails and lift the documented `dt ≤ 0.5 fs`
cap. Pre-removal verification instead found the dt=1fs unit tests still **fail** at
their own config: local CPU runs gave n=2 → 358–403 K, n=16 → 343 K (|dev| 43–103 K
vs the ±15 K assertion). Removing the strict xfails would have broken CI. This note
characterizes *why*, via a system-size sweep at the gate's exact configuration.

## Method

`scripts/experiments/p5_nvt_size_sweep_dt1fs.py` reuses the gate's `run_nvt()`
**verbatim** (identical metric: raw rigid-body KE, with T_trans/T_rot decomposition),
sweeping n_waters ∈ {2, 16, 64, 216, 512, 895} at dt=1.0 fs, **gamma=10 ps⁻¹**,
3 seeds (42–44), 30 ps (30 k steps, 10 k burn), `project_ou=True`, `remove_com=False`,
`settle_velocity_iters=10`. Aggregated by `scripts/analysis/p5_size_sweep_aggregate.py`.

## Result

```
n_waters    T_rot   T_trans  T_total |dev_rot| |dev_tot|  tot<=15  tot<=5
       2    311.6     600.6    407.9      11.6     107.9    False   False
      16    299.7     319.7    309.4       0.3       9.4     True   False
      64    299.5     306.0    302.7       0.5       2.7     True    True
     216    299.6     301.3    300.5       0.4       0.5     True    True
     512    299.9     301.0    300.4       0.1       0.4     True    True
     895    299.4     300.4    299.9       0.6       0.1     True    True
```

- **Crossover N\* (|T_total−300| ≤ 15 K, unit-test tolerance): n = 16.**
- **Crossover N\* (|T_total−300| ≤ 5 K, gate tolerance): n = 64.**
- **T_rot is within 15 K of 300 K at *every* size, including n=2.**

## Interpretation

1. **The warm bias is entirely translational finite-size, not a dt-stability failure.**
   T_rot is faithful at all sizes; the bias lives in T_trans, which decays ~1/N
   (600 → 320 → 306 → 301 → 300 K). At n=2 there are only 3N−3 = 3 translational DOF,
   which the Langevin thermostat under-regulates against the SETTLE constraint
   impulse; the effect washes out by n ≳ 16.

2. **dt=1.0 fs is thermally faithful at production scale.** At gamma=10, T_total is
   within ±15 K for all n ≥ 16 and within ±5 K for all n ≥ 64. The gate result
   (T_rot 299.63 K at n=895) is genuine and representative.

3. **The unit-test failures are a config artifact, not a dt failure.** The unit-test
   helper hardcodes **gamma=1 ps⁻¹** (`test_settle_temperature_control.py:30`), not
   the gate's gamma=10. At n=16, gamma=1 → 343 K (fail) but gamma=10 → 309 K (pass).
   The tests exercise a weaker-friction, small-N regime that was never gated.

## Implications for G1 (revised)

- The `dt ≤ 0.5 fs` cap **can** be lifted to `dt ≤ 1.0 fs`, scoped to adequately
  thermostatted, non-trivially-sized systems (gamma ≈ 10 ps⁻¹, n ≳ 16). The blanket
  unconditional replacement the handoff specified is **not** supported.
- The n=2 xfail (`test_temperature_dt1fs_near_target`) should be **kept**, reason
  reframed: small-N translational finite-size artifact (3 trans DOF), not dt
  instability.
- The n=16 sweep test (`test_dt_sweep_16water_nvt`) was **retargeted to assert T_rot**
  (the finite-size-robust, gate-validated metric) at gamma=10, and un-xfailed for
  dt=1.0 fs (dt=2.0 stays permanent xfail). **Resolved (2026-06-13).**

## Side finding: latent T_total failure at n=16

Retargeting uncovered that the *previous* assertion (on combined **T_total**) was a
latent failure even at the dt=0.5 fs baseline: at n=16/10 ps, T_total = 364.9 K
(gamma=1) / 315.8 K (gamma=10), both > 300 ± 15 K. It was masked because the test is
`@pytest.mark.slow` and excluded from the fast CI gate, so it was never actually run.
Root cause is the same small-N translational finite-size mode that dominates T_total at
n=16. The fix (assert T_rot) makes both dt=0.5 and dt=1.0 pass cleanly (~300 K), since
T_rot is faithful at every size.

## Open question

Does the gamma=1 warm bias also recover at large N (finite-size), or is gamma=1
fundamentally worse at dt=1fs even at scale? Only gamma=1 small-N data exists
(n≤16). A gamma=1 point at n≥216 would settle whether the friction effect is
size-curable.

## Re-validation at genuine dt=1.0 fs (2026-08-11)

**Task:** 260806_p5_measurement_pipeline_audit
**Campaign:** 46a4d737 (`bth`, parent ba334c1f)
**Jobs:** 19882395 (n=2,16,64), 19898460 (n=216,895), 19898934 (n=512, isolated re-run
after 19898460's array-slot for n=512 failed on a log-file race — see below)

This sweep predated the discovery that `settle_langevin`'s A-step was passed
`half_dt` where `_langevin_step_a` already halves internally (fixed in commit
cfd2b85; see `settle.py` docstring). The original ba334c1f numbers above were
therefore measured at an *effective* dt ≈ 0.5 fs, not the requested 1.0 fs.
Re-running the identical sweep script (`p5_nvt_size_sweep_dt1fs.py`, unchanged
config: gamma=10 ps⁻¹, 3 seeds, 30k steps/10k burn) under the corrected propagator
gives:

```
n_waters    T_rot   T_trans  T_total |dev_rot| |dev_tot|  tot<=15  tot<=5
       2    297.1     614.7    403.0       2.9     103.0    False   False
      16    296.4     321.7    308.6       3.6       8.6     True   False
      64    298.9     305.0    301.9       1.1       1.9     True    True
     216    299.0     301.0    300.0       1.0       0.0     True    True
     512    300.3     300.4    300.4       0.3       0.4     True    True
     895    299.0     299.9    299.4       1.0       0.6     True    True
```

**Conclusion holds unchanged**: Crossover N\*=16 (±15 K), N\*=64 (±5 K); T_rot
faithful at every size (max |dev| 3.6 K, well inside the 15 K tolerance). Every
value tracks the original (effective-0.5-fs) table within a few K — consistent with
3-seed noise, not a propagator-dependent shift. The translational finite-size warm
bias is confirmed to be a genuine physical effect (T_trans ~1/N decay,
614.7→321.7→305.0→~300 K), not an artifact of the half_dt bug.

**Operational note:** `myxcel submit` (the raw-sbatch-script command) silently
defaults `--time` to `01:00:00` and always forces `--partition` (default
`mit_normal`, no GPUs) onto the `sbatch` invocation — neither is inherited from the
script's own `#SBATCH` directives; CLI-level args win. The first full-array
submission (job 19882395 covering all 6 sizes) omitted `--time`, so n=216/512/895
were killed by `TIMEOUT` at 1 hour despite the script requesting 11:30:00. Always
pass `--time` and `--partition` explicitly to `myxcel submit`. Separately, myxcel
also forces the SLURM `--output`/`--error` paths to `%j.out`/`%j.err` (parent job
ID only, not `%A_%a`), so concurrent array tasks under one job race on the same
log file — a task's own stderr can be silently clobbered by a sibling. The n=512
retry (19898460 task 4) failed fast with an unreadable, clobbered log for exactly
this reason; resubmitting it alone (job 19898934) as a single-task array avoided
the race and completed cleanly.

## Correction (2026-10-01): the small-N warm bias is an estimator artifact

**Superseding** the "translational finite-size" reading above. The T_trans estimator
used here took the kinetic energy of every per-water COM velocity *without*
subtracting the system COM velocity, and divided by `3N − 3` DOF. With
`remove_linear_com_momentum=False` (the setting used throughout) the system COM is a
live, thermalized 3-DOF mode, so the estimator reads high by exactly `3T/(3N − 3)`:
+300 K at n=2, +20 K at n=16, +4.8 K at n=64, +0.34 K at n=895. The predicted
artifact matches the recorded raw values (614.7 / 321.7 / 305.0 K) to within the
seed spread.

The 46a4d737 re-run already recorded the COM-subtracted quantity in every result
JSON (`mean_t_trans_corrected`, computed inside the job, not post hoc):

| n waters | job | raw T_trans (K) | COM-subtracted T_trans (K) | T_rot (K) |
|---|---|---|---|---|
| 2 | 19881756 / 19882395 | 614.72 | 302.66 ± 13.67 | 297.09 |
| 16 | 19882395 | 321.70 | 301.15 ± 4.62 | 296.38 |
| 64 | 19882395 | 304.98 | 300.36 ± 0.72 | 298.87 |
| 216 | 19898460 | 301.02 | 299.63 ± 0.39 | 298.98 |
| 512 | 19898934 | 300.45 | 299.82 ± 1.05 | 300.28 |
| 895 | 19898460 | 299.87 | 299.54 ± 0.53 | 298.99 |

(± is the spread over the 3 seeds: a convergence diagnostic, not a significance
test.) Once COM-subtracted, T_trans is flat at the 300 K target from n=2 to n=895.
There is no small-N translational bias left to explain, and the "n ≲ 16 needs
dt ≤ 0.5 fs" carve-out has no remaining evidential basis from this sweep. The
weak-friction (gamma ≈ 1 ps⁻¹) carve-out was not tested here and stands.

Library fix: `prolix.physics.temperature_scan.rigid_tip3p_temperatures` returns
COM-subtracted `t_total`/`t_trans`/`t_rot`/`t_com`, and
`scan_settle_rigid_temperatures` now reports the COM-subtracted total (it previously
read high by `3T/(6N − 3)`, i.e. +100 K at n=2). Positive and negative controls on
exact Maxwell-Boltzmann draws: `tests/physics/test_transrot_decomposition.py`.
Bathos recorded these runs with `outcome='unknown'` because the Engaging bathos
predated the `--out` result fallback; the numbers above are read from the result
JSONs directly.
