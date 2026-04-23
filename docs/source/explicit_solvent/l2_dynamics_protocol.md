# L2 dynamics protocol (explicit solvent, P2a)

This document defines the **baseline** explicit-solvent Langevin validation for
Prolix vs OpenMM: ensemble-level thermostat statistics on a tiny periodic PME
system, not trajectory matching.

## Baseline system (N=4 mock charges)

- **Geometry:** cubic cell **45 Å**, four particles at the same Å coordinates used in
  `tests/physics/test_explicit_slow_validation.py` and `tests/physics/test_explicit_langevin_parity.py`.
- **Interactions:** electrostatics-only mock parameters (`epsilons = 0`) with PME
  parameters from `REGRESSION_EXPLICIT_PME` / `regression_pme_params`
  (`tests/physics/conftest.py`).
- **Masses:** uniform **12.0 g/mol** per particle on both sides.

## Integrators

- **Prolix path (CI-gated today):** `jax_md.simulate.nvt_langevin` with Prolix `make_energy_fn` PME.
- **OpenMM path (not CI-gated on this fixture):** `openmm.openmm.LangevinMiddleIntegrator`
  does **not** currently produce a trustworthy 300 K ensemble on this n=4 PME mock
  system at JAX-MD-compatible timesteps (and becomes unstable or badly biased at
  larger timesteps). For OpenMM-only Langevin thermostat statistics on a separate
  tiny cell, see `scripts/benchmarks/openmm_langevin_temperature_stats.py`.

## Simulation protocol

| Setting | Value |
|--------|-------|
| Target temperature | 300 K |
| Timestep | **≈0.049 fs** — `dt = 1e-3` AKMA (`AKMA_TIME_UNIT_FS` in `prolix.simulate`), matched on OpenMM |
| Production length | **3000 steps** (~0.15 ps simulated time; chosen for CI + PME stability) |
| Burn-in | discard first **1000** steps |
| Analysis window | remaining **2000** instantaneous samples |
| Instantaneous T | \(T = 2K / (k_B \cdot \mathrm{DOF})\) with **DOF = 3N = 12** (no COM subtraction) |

**Why not 20 fs / 10 ps here?** `jax_md.simulate.nvt_langevin` coupled to this explicit PME energy is **unstable** at multi-femtosecond steps on the n=4 mock system (temperature explodes). The gate is therefore **distribution-level mean T**, not a long-horizon production run. A future P2a-ext water fixture may revisit larger timesteps with constraints + smaller effective fastest modes.

## Acceptance gates (baseline)

| Metric | Gate |
|--------|------|
| Window-mean \(T\) — Prolix | within **±10%** of target (270–330 K at 300 K) |
| \(T\) variance | strictly positive (sanity: thermostat is not frozen) |
| Short NVE drift (Prolix) | relative \(|\Delta E_\text{tot}| / |E_0| \le 0.01 \times \Delta t_\text{ps}\) over an **800-step** NVE segment at the same AKMA timestep |

## Artifacts

`tests/physics/test_explicit_langevin_parity.py` writes a per-step Prolix CSV:

- `step`, `time_ps`, `T_inst`, `K_kcalmol`, `U_kcalmol`, `Etot_kcalmol`

When `GITHUB_WORKSPACE` is set (GitHub Actions), files are written under
`artifacts/l2_parity/` and uploaded from the optional OpenMM workflow job (the
job also runs this **slow** test even though it does not require OpenMM).

## Stretch goal (P2a-ext, optional)

A ~100-atom TIP3P + SETTLE fixture may tighten the window-mean temperature gate
to **±1.67%** at 300 K where effective DOFs are large enough for a narrower
ensemble gate. That extension is **not** required for the baseline test above.
