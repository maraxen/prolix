#!/usr/bin/env python3
"""Sprint B Step 2: Constraint-impulse scaling discriminator.

Runs an NVT matrix over water count and timestep, then measures whether
per-step SETTLE velocity-constraint impulse scales with drift.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 before importing project modules that instantiate arrays.
jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.physics.settle_langevin_potential_propagator import settle_langevin_potential_cached_step
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.simulate import NVTLangevinState
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water


@dataclass
class ConditionResult:
  n_waters: int
  dt_fs: float
  project_ou_momentum_rigid: bool
  steps: int
  burn: int
  t_target_k: float
  t_mean_k: float
  t_std_k: float
  drift_k_per_ps: float
  settle_impulse_mean: float
  settle_impulse_rms: float
  settle_impulse_p95: float
  settle_impulse_p99: float
  settle_impulse_per_water_mean: float
  settle_impulse_per_dof_mean: float
  potential_mean_kcal: float
  potential_std_kcal: float


def _build_state(
  n_waters: int,
  dt_fs: float,
  seed: int,
  temperature_k: float = 300.0,
):
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_ps = 1.0
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

  sys_dict = _prolix_params_pure_water(n_waters)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=32,
    pme_alpha=0.34,
    cutoff_distance=9.0,
    strict_parameterization=False,
  )
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  mass_col = mass[:, None]
  water_indices = settle.get_water_indices(0, n_waters)

  init_fn, _ = settle.settle_langevin(
    energy_fn,
    shift_fn,
    dt=dt_akma,
    kT=kT,
    gamma=gamma_reduced,
    mass=mass,
    water_indices=water_indices,
    box=box_vec,
    remove_linear_com_momentum=False,
    project_ou_momentum_rigid=True,
    projection_site="post_o",
    settle_velocity_iters=10,
  )
  state = init_fn(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  return state, energy_fn, shift_fn, mass_col, water_indices, box_vec, dt_akma, gamma_reduced, kT


def _run_condition_scan_jit(
  *,
  n_waters: int,
  dt_fs: float,
  seed: int,
  sim_ps: float,
  burn_fraction: float,
  project_ou_momentum_rigid: bool,
) -> ConditionResult:
  """Driver: one ``jax.jit`` wraps ``lax.scan`` burn + production (SETTLE + Langevin)."""
  state0, energy_fn, shift_fn, mass_col, water_indices, box_vec, dt_akma, gamma_reduced, kT = _build_state(
    n_waters=n_waters,
    dt_fs=dt_fs,
    seed=seed,
  )

  t_target_k = 300.0
  dof_rigid = jnp.asarray(6 * n_waters - 3, dtype=jnp.float64)
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(1, int(steps * burn_fraction))
  burn = min(burn, max(steps - 1, 0))
  prod = max(steps - burn, 0)

  def burn_step(s, _):
    s2, _metrics = settle_langevin_potential_cached_step(
      s,
      energy_fn=energy_fn,
      shift_fn=shift_fn,
      mass_col=mass_col,
      water_indices=water_indices,
      box_vec=box_vec,
      dt_akma=dt_akma,
      gamma_reduced=gamma_reduced,
      kT=kT,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
    )
    return s2, None

  def prod_step(s, _):
    s2, metrics = settle_langevin_potential_cached_step(
      s,
      energy_fn=energy_fn,
      shift_fn=shift_fn,
      mass_col=mass_col,
      water_indices=water_indices,
      box_vec=box_vec,
      dt_akma=dt_akma,
      gamma_reduced=gamma_reduced,
      kT=kT,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
    )
    ke_r = rigid_tip3p_box_ke_kcal(s2.position, s2.momentum, s2.mass, n_waters)
    temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
    return s2, (metrics.settle_impulse, temp, metrics.potential_energy)

  @partial(jax.jit, static_argnames=("n_burn", "n_prod"))
  def run_scanned(s_init, n_burn, n_prod):
    s_mid, _ = jax.lax.scan(burn_step, s_init, None, length=n_burn)
    if n_prod == 0:
      empty = jnp.zeros((0,), dtype=jnp.float64)
      return empty, empty, empty
    _s_fin, (imps, temps, energies) = jax.lax.scan(prod_step, s_mid, None, length=n_prod)
    return imps, temps, energies

  imps, temps, energies = run_scanned(state0, burn, prod)
  imps_arr = np.asarray(imps, dtype=np.float64)
  temps_arr = np.asarray(temps, dtype=np.float64)
  energies_arr = np.asarray(energies, dtype=np.float64)
  if prod == 0:
    return ConditionResult(
      n_waters=n_waters,
      dt_fs=dt_fs,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
      steps=steps,
      burn=burn,
      t_target_k=t_target_k,
      t_mean_k=float("nan"),
      t_std_k=float("nan"),
      drift_k_per_ps=float("nan"),
      settle_impulse_mean=float("nan"),
      settle_impulse_rms=float("nan"),
      settle_impulse_p95=float("nan"),
      settle_impulse_p99=float("nan"),
      settle_impulse_per_water_mean=float("nan"),
      settle_impulse_per_dof_mean=float("nan"),
      potential_mean_kcal=float("nan"),
      potential_std_kcal=float("nan"),
    )
  # Time axis: production steps are (burn+1 .. burn+prod) in 1-based step index → ps
  step_idx_1 = jnp.arange(1, prod + 1, dtype=jnp.float64) + float(burn)
  time_ps_j = step_idx_1 * float(dt_fs) * 1e-3
  if prod >= 2:
    slope = float(jnp.polyfit(time_ps_j, temps, deg=1)[0])
  else:
    slope = float("nan")

  return ConditionResult(
    n_waters=n_waters,
    dt_fs=dt_fs,
    project_ou_momentum_rigid=project_ou_momentum_rigid,
    steps=steps,
    burn=burn,
    t_target_k=t_target_k,
    t_mean_k=float(np.mean(temps_arr)),
    t_std_k=float(np.std(temps_arr)),
    drift_k_per_ps=float(slope),
    settle_impulse_mean=float(np.mean(imps_arr)),
    settle_impulse_rms=float(np.sqrt(np.mean(imps_arr**2))),
    settle_impulse_p95=float(np.percentile(imps_arr, 95)),
    settle_impulse_p99=float(np.percentile(imps_arr, 99)),
    settle_impulse_per_water_mean=float(np.mean(imps_arr) / n_waters),
    settle_impulse_per_dof_mean=float(np.mean(imps_arr) / max(float(6 * n_waters - 3), 1.0)),
    potential_mean_kcal=float(np.mean(energies_arr)),
    potential_std_kcal=float(np.std(energies_arr)),
  )


def _run_condition_eager_loop(
  *,
  n_waters: int,
  dt_fs: float,
  seed: int,
  sim_ps: float,
  burn_fraction: float,
  project_ou_momentum_rigid: bool,
) -> ConditionResult:
  """Python loop (debug / fallback only). Forces from ``value_energy_and_forces`` (JAX-MD ``F=-∇E``)."""
  state, energy_fn, shift_fn, mass_col, water_indices, box_vec, dt_akma, gamma_reduced, kT = _build_state(
    n_waters=n_waters,
    dt_fs=dt_fs,
    seed=seed,
  )

  t_target_k = 300.0
  dof_rigid = float(6 * n_waters - 3)
  steps = int(sim_ps * 1000.0 / dt_fs)
  burn = max(1, int(steps * burn_fraction))
  burn = min(burn, max(steps - 1, 0))

  temps: list[float] = []
  impulses: list[float] = []
  energies: list[float] = []
  time_ps: list[float] = []

  for step_idx in range(steps):
    state, metrics = settle_langevin_potential_cached_step(
      state,
      energy_fn=energy_fn,
      shift_fn=shift_fn,
      mass_col=mass_col,
      water_indices=water_indices,
      box_vec=box_vec,
      dt_akma=dt_akma,
      gamma_reduced=gamma_reduced,
      kT=kT,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
    )

    if step_idx >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
      temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
      temps.append(float(temp))
      impulses.append(float(metrics.settle_impulse))
      energies.append(float(metrics.potential_energy))
      time_ps.append(float((step_idx + 1) * dt_fs * 1e-3))

  temps_arr = np.asarray(temps, dtype=np.float64)
  impulses_arr = np.asarray(impulses, dtype=np.float64)
  energies_arr = np.asarray(energies, dtype=np.float64)
  if len(temps_arr) == 0:
    return ConditionResult(
      n_waters=n_waters,
      dt_fs=dt_fs,
      project_ou_momentum_rigid=project_ou_momentum_rigid,
      steps=steps,
      burn=burn,
      t_target_k=t_target_k,
      t_mean_k=float("nan"),
      t_std_k=float("nan"),
      drift_k_per_ps=float("nan"),
      settle_impulse_mean=float("nan"),
      settle_impulse_rms=float("nan"),
      settle_impulse_p95=float("nan"),
      settle_impulse_p99=float("nan"),
      settle_impulse_per_water_mean=float("nan"),
      settle_impulse_per_dof_mean=float("nan"),
      potential_mean_kcal=float("nan"),
      potential_std_kcal=float("nan"),
    )
  if len(time_ps) >= 2:
    slope, _intercept = np.polyfit(np.asarray(time_ps, dtype=np.float64), temps_arr, deg=1)
  else:
    slope = float("nan")

  return ConditionResult(
    n_waters=n_waters,
    dt_fs=dt_fs,
    project_ou_momentum_rigid=project_ou_momentum_rigid,
    steps=steps,
    burn=burn,
    t_target_k=t_target_k,
    t_mean_k=float(np.mean(temps_arr)),
    t_std_k=float(np.std(temps_arr)),
    drift_k_per_ps=float(slope),
    settle_impulse_mean=float(np.mean(impulses_arr)),
    settle_impulse_rms=float(np.sqrt(np.mean(impulses_arr**2))),
    settle_impulse_p95=float(np.percentile(impulses_arr, 95)),
    settle_impulse_p99=float(np.percentile(impulses_arr, 99)),
    settle_impulse_per_water_mean=float(np.mean(impulses_arr) / n_waters),
    settle_impulse_per_dof_mean=float(np.mean(impulses_arr) / max(dof_rigid, 1.0)),
    potential_mean_kcal=float(np.mean(energies_arr)),
    potential_std_kcal=float(np.std(energies_arr)),
  )


def _default_conditions(include_control: bool) -> list[tuple[int, float, bool]]:
  conditions = [(n_w, dt, True) for n_w in (8, 16, 32, 64) for dt in (0.5, 0.25)]
  if include_control:
    conditions.extend([(64, 0.5, False), (64, 0.25, False)])
  return conditions


def _write_markdown_report(path: Path, rows: list[ConditionResult]) -> None:
  lines = [
    "# Sprint B Step 2: Constraint-Impulse Scaling Report",
    "",
    "## Acceptance Criteria",
    "- Constraint-impulse metrics rise with n_waters and/or effective step frequency.",
    "- Constraint-impulse metrics track temperature drift slope (K/ps).",
    "- If tracking fails, branch to PME-force variance tests.",
    "",
    "## Condition Results",
    "",
    "| n_waters | dt_fs | project_ou | T_mean (K) | Drift (K/ps) | V_mean (kcal/mol) | V_std (kcal/mol) | Impulse mean | Impulse p95 | Impulse/water | Impulse/dof |",
    "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    lines.append(
      f"| {row.n_waters} | {row.dt_fs:.2f} | {str(row.project_ou_momentum_rigid)} | "
      f"{row.t_mean_k:.2f} | {row.drift_k_per_ps:.4f} | {row.potential_mean_kcal:.4f} | {row.potential_std_kcal:.4f} | "
      f"{row.settle_impulse_mean:.4e} | {row.settle_impulse_p95:.4e} | {row.settle_impulse_per_water_mean:.4e} | "
      f"{row.settle_impulse_per_dof_mean:.4e} |"
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description="Sprint B Step 2 constraint-impulse scaling run")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--sim-ps", type=float, default=100.0)
  parser.add_argument("--burn-fraction", type=float, default=1.0 / 3.0)
  parser.add_argument("--include-control", action="store_true", help="Include OU projection off control for 64-water runs")
  parser.add_argument("--n-waters", type=int, default=None, help="Run a single condition for this water count")
  parser.add_argument("--dt-fs", type=float, default=None, help="Run a single condition for this timestep (fs)")
  parser.add_argument(
    "--project-ou-momentum-rigid",
    type=int,
    choices=(0, 1),
    default=None,
    help="Run a single condition with explicit OU rigid projection toggle (1/0)",
  )
  parser.add_argument(
    "--results-json",
    type=Path,
    default=Path(".praxia/tmp/sprint_b_step2_constraint_impulse_results.json"),
  )
  parser.add_argument(
    "--report-md",
    type=Path,
    default=Path(".praxia/tmp/sprint_b_step2_constraint_impulse_report.md"),
  )
  parser.add_argument(
    "--eager-loop",
    action="store_true",
    help="Use a Python for-loop instead of jax.jit(jax.lax.scan(...)). For debugging only.",
  )
  args = parser.parse_args()

  if args.n_waters is not None or args.dt_fs is not None or args.project_ou_momentum_rigid is not None:
    if args.n_waters is None or args.dt_fs is None:
      raise ValueError("Single-condition mode requires both --n-waters and --dt-fs.")
    project_ou = True if args.project_ou_momentum_rigid is None else bool(args.project_ou_momentum_rigid)
    conditions = [(int(args.n_waters), float(args.dt_fs), project_ou)]
  else:
    conditions = _default_conditions(include_control=args.include_control)

  rows: list[ConditionResult] = []
  for n_waters, dt_fs, project_ou in conditions:
    print(
      f"[STEP2] running n_waters={n_waters}, dt_fs={dt_fs}, "
      f"project_ou_momentum_rigid={project_ou}"
    , flush=True)
    if args.eager_loop:
      row = _run_condition_eager_loop(
        n_waters=n_waters,
        dt_fs=dt_fs,
        seed=args.seed,
        sim_ps=args.sim_ps,
        burn_fraction=args.burn_fraction,
        project_ou_momentum_rigid=project_ou,
      )
    else:
      row = _run_condition_scan_jit(
        n_waters=n_waters,
        dt_fs=dt_fs,
        seed=args.seed,
        sim_ps=args.sim_ps,
        burn_fraction=args.burn_fraction,
        project_ou_momentum_rigid=project_ou,
      )
    rows.append(row)
    print(
      "[STEP2] result "
      f"T_mean={row.t_mean_k:.2f}K, drift={row.drift_k_per_ps:.4f}K/ps, "
      f"impulse_mean={row.settle_impulse_mean:.4e}, V_mean={row.potential_mean_kcal:.2f} kcal/mol"
    , flush=True)

  args.results_json.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "experiment": "sprint_b_step2_constraint_impulse_scaling",
    "seed": args.seed,
    "sim_ps": args.sim_ps,
    "burn_fraction": args.burn_fraction,
    "conditions": [asdict(row) for row in rows],
  }
  args.results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  _write_markdown_report(args.report_md, rows)

  print(f"[STEP2] wrote results JSON: {args.results_json}", flush=True)
  print(f"[STEP2] wrote report MD: {args.report_md}", flush=True)


if __name__ == "__main__":
  main()
