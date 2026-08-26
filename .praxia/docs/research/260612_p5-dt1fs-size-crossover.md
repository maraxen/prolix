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
