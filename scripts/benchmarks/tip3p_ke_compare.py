#!/usr/bin/env python3
"""Compare KE and thermometer diagnostics for TIP3P (OpenMM vs Prolix).

Re-run examples (repo root)::

  OPENMM_INSTALL_MODE=ephemeral uv run --with openmm python scripts/benchmarks/tip3p_ke_compare.py \\
    --engine openmm --n-waters 4 --steps 3000 --burn 1200 --sample-every 20 --replicas 1 \\
    --remove-cmmotion false --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0

  uv run python scripts/benchmarks/tip3p_ke_compare.py \\
    --engine prolix --jax-x64 on --n-waters 33 --steps 30000 --burn 10000 --sample-every 10 --replicas 5 \\
    --remove-cmmotion false --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_BENCH = _REPO / "scripts" / "benchmarks"
if str(_BENCH) not in sys.path:
  sys.path.insert(0, str(_BENCH))
import tip3p_ke_profile  # noqa: E402

_PDB_PATH = Path("/tmp/tip3p_ke_compare_watbox.pdb")
_T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
  sys.path.insert(0, str(_SRC))
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME as _REGRESSION_EXPLICIT_PME  # noqa: E402
_BOLTZMANN_KCAL = 0.0019872041
# Keep in sync with `prolix.simulate.AKMA_TIME_UNIT_FS` (used for γ, Δt reduced units).
_AKMA_TIME_UNIT_FS = 48.88821291839


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--engine", choices=("openmm", "prolix", "both"), default="both")
  ap.add_argument("--jax-x64", choices=("off", "on"), default="off")
  ap.add_argument("--n-waters", type=int, default=4)
  ap.add_argument("--steps", type=int, default=220)
  ap.add_argument("--burn", type=int, default=80)
  ap.add_argument("--sample-every", type=int, default=10)
  ap.add_argument("--seed", type=int, default=4242)
  ap.add_argument("--replicas", type=int, default=1)
  ap.add_argument("--timing-mode", choices=("cold", "steady", "both"), default="both")
  ap.add_argument("--warmup-steps", type=int, default=100)
  ap.add_argument("--measure-steps", type=int, default=500)
  ap.add_argument("--dt-fs", type=float, default=2.0, help="Integrator step (fs); same for OpenMM and Prolix in this benchmark.")
  ap.add_argument("--temperature-k", type=float, default=300.0, help="Target bath temperature (K).")
  ap.add_argument("--gamma-ps", type=float, default=1.0, help="Langevin friction coefficient (1/ps), matched to Prolix gamma_reduced mapping.")
  ap.add_argument(
    "--remove-cmmotion",
    choices=("true", "false"),
    default="false",
    help="OpenMM createSystem(removeCMMotion=...). Prolix uses the same value for settle_langevin(remove_linear_com_momentum=...). Default false: diag_linear_com_off.",
  )
  ap.add_argument(
    "--openmm-integrator",
    choices=("middle", "langevin"),
    default="middle",
    help="OpenMM only: LangevinMiddleIntegrator vs legacy LangevinIntegrator (diagnostic).",
  )
  ap.add_argument(
    "--verbose-samples",
    action="store_true",
    help="Include extra per-sample series (T_atomic, COM KE, |sum m v|); larger JSON.",
  )
  ap.add_argument(
    "--projection-site",
    choices=("post_o", "post_settle_vel", "both"),
    default="post_o",
    help="Prolix only: SETTLE+Langevin ``projection_site`` (default post_o, matches prior tip3p_ke_compare).",
  )
  ap.add_argument(
    "--settle-velocity-iters",
    type=int,
    default=10,
    help="Prolix only: RATTLE-like iterations in settle_velocities (default 10).",
  )
  ap.add_argument(
    "--project-ou-momentum-rigid",
    choices=("true", "false"),
    default="true",
    help="Prolix only: rigid OU momentum projection (default true).",
  )
  return ap.parse_args()


def _configure_jax(mode: str) -> dict:
  if mode not in ("off", "on"):
    return {"requested_jax_x64": mode, "effective_jax_x64": None, "jax_backend": None}
  import jax

  jax.config.update("jax_enable_x64", mode == "on")
  effective = bool(jax.config.read("jax_enable_x64"))
  return {
    "requested_jax_x64": mode,
    "effective_jax_x64": effective,
    "jax_backend": jax.default_backend(),
  }


def _load_tip3p_module():
  path = _REPO / "scripts/benchmarks/tip3p_langevin_tightening.py"
  spec = importlib.util.spec_from_file_location("tip3p_langevin_tightening", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _tip3p_grid_positions(n_waters: int, spacing_angstrom: float) -> tuple[np.ndarray, float]:
  o = np.zeros(3, dtype=np.float64)
  r_oh = 0.9572
  theta = math.radians(104.52)
  h1 = np.array([r_oh, 0.0, 0.0], dtype=np.float64)
  h2 = np.array([r_oh * math.cos(theta), r_oh * math.sin(theta), 0.0], dtype=np.float64)
  sites: list[tuple[int, int, int]] = []
  n = int(math.ceil(n_waters ** (1.0 / 3.0))) + 3
  for ix in range(n):
    for iy in range(n):
      for iz in range(n):
        sites.append((ix, iy, iz))
        if len(sites) >= n_waters:
          break
      if len(sites) >= n_waters:
        break
    if len(sites) >= n_waters:
      break
  base = np.array([3.0, 3.0, 3.0], dtype=np.float64)
  pos: list[np.ndarray] = []
  for ix, iy, iz in sites[:n_waters]:
    wo = base + np.array([ix * spacing_angstrom, iy * spacing_angstrom, iz * spacing_angstrom], dtype=np.float64)
    pos.extend([wo + o, wo + h1, wo + h2])
  arr = np.vstack(pos)
  span = np.max(arr, axis=0) - np.min(arr, axis=0)
  box_edge = float(np.max(span) + 16.0)
  return arr, box_edge


def _write_tip3p_pdb(path: Path, positions_angstrom: np.ndarray, box_angstrom: float) -> None:
  n_atoms = positions_angstrom.shape[0]
  n_res = n_atoms // 3
  lines = [f"CRYST1{box_angstrom:9.3f}{box_angstrom:9.3f}{box_angstrom:9.3f}  90.00  90.00  90.00 P 1\n"]
  for i in range(n_atoms):
    serial = i + 1
    res_seq = i // 3 + 1
    x, y, z = positions_angstrom[i]
    if i % 3 == 0:
      lines.append(f"HETATM{serial:5d}  O   HOH A{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n")
    elif i % 3 == 1:
      lines.append(f"HETATM{serial:5d}  H1  HOH A{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n")
    else:
      lines.append(f"HETATM{serial:5d}  H2  HOH A{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n")
  for r in range(n_res):
    o = 3 * r + 1
    h1 = 3 * r + 2
    h2 = 3 * r + 3
    lines.append(f"CONECT{o:5d}{h1:5d}{h2:5d}\n")
    lines.append(f"CONECT{h1:5d}{o:5d}\n")
    lines.append(f"CONECT{h2:5d}{o:5d}\n")
  lines.append("END\n")
  path.write_text("".join(lines), encoding="utf-8")


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)


def _dof_atomic_unconstrained(n_atoms: int) -> float:
  """Per-atom Σ p²/(2m) thermometer DOF (diagnostic only; not used for G4)."""
  return float(3 * n_atoms)


def _gamma_dt_consistency(*, gamma_ps: float, dt_fs: float, akma_time_unit_fs: float) -> dict:
  """Fail-fast: dimensionless γΔt must match OpenMM γ[1/ps] × Δt[ps]."""
  dt_ps = float(dt_fs) * 1e-3
  dt_akma = float(dt_fs) / float(akma_time_unit_fs)
  gamma_reduced = float(gamma_ps) * float(akma_time_unit_fs) * 1e-3
  lhs = gamma_reduced * dt_akma
  rhs = float(gamma_ps) * dt_ps
  rel = abs(lhs - rhs) / max(abs(rhs), 1e-30)
  return {
    "gamma_ps": float(gamma_ps),
    "dt_fs": float(dt_fs),
    "dt_ps": dt_ps,
    "dt_akma": dt_akma,
    "gamma_reduced_per_tau": gamma_reduced,
    "gamma_reduced_times_dt_akma": lhs,
    "gamma_ps_times_dt_ps": rhs,
    "relative_abs_error": rel,
    "ok": bool(rel < 1e-12),
  }


def _sem(values: list[float]) -> float:
  if len(values) < 2:
    return 0.0
  return float(statistics.pstdev(values) / math.sqrt(len(values)))


def _t95(df: int) -> float:
  if df <= 0:
    return 1.96
  if df in _T95_BY_DF:
    return _T95_BY_DF[df]
  return 1.96


def _mean_stats(values: list[float], method: str) -> dict:
  if not values:
    return {"n": 0, "mean": float("nan"), "pstdev": 0.0, "sem": 0.0, "ci95_low": float("nan"), "ci95_high": float("nan"), "method": method}
  mean = float(statistics.fmean(values))
  pstdev = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
  sem = _sem(values)
  half = _t95(len(values) - 1) * sem
  return {
    "n": len(values),
    "mean": mean,
    "pstdev": pstdev,
    "sem": sem,
    "ci95_low": mean - half,
    "ci95_high": mean + half,
    "method": method,
  }


def _block_sem(values: list[float], block_size: int) -> dict:
  n = len(values)
  n_blocks = n // max(block_size, 1)
  if n_blocks < 8:
    return {"block_size": block_size, "n_blocks": n_blocks, "sem": _sem(values), "method": "iid_sem", "warning": "n_blocks<8; fallback to iid_sem"}
  arr = np.array(values[: n_blocks * block_size], dtype=np.float64).reshape(n_blocks, block_size)
  block_means = arr.mean(axis=1).tolist()
  return {"block_size": block_size, "n_blocks": n_blocks, "sem": _sem(block_means), "method": "block_sem", "warning": None}


def _openmm_run_once(
  *,
  n_waters: int,
  n_steps: int,
  burn: int,
  sample_every: int,
  seed: int,
  timing_mode: str,
  warmup_steps: int,
  measure_steps: int,
  dt_fs: float,
  temperature_k: float,
  gamma_ps: float,
  remove_cmmotion: bool,
  openmm_integrator: str,
  verbose_samples: bool,
  akma_time_unit_fs: float,
) -> dict:
  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  positions_a, box_edge = _tip3p_grid_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3
  pme = dict(_REGRESSION_EXPLICIT_PME)
  alpha = float(pme["pme_alpha_per_angstrom"])
  grid = int(pme["pme_grid_points"])
  cutoff = float(pme["cutoff_angstrom"])
  platform_name = str(pme["openmm_platform"])
  _write_tip3p_pdb(_PDB_PATH, positions_a, box_edge)
  pdb = PDBFile(str(_PDB_PATH))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=cutoff / 10.0 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
    removeCMMotion=bool(remove_cmmotion),
  )
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(alpha * 10.0, grid, grid, grid)
      f.setUseDispersionCorrection(bool(pme["use_dispersion_correction"]))
  temp_q = temperature_k * omm_unit.kelvin
  fric_q = float(gamma_ps) / omm_unit.picosecond
  dt_q = float(dt_fs) * omm_unit.femtoseconds
  if openmm_integrator == "middle":
    integrator = openmm.LangevinMiddleIntegrator(temp_q, fric_q, dt_q)
  else:
    integrator = openmm.LangevinIntegrator(temp_q, fric_q, dt_q)
  platform_obj = openmm.Platform.getPlatformByName(platform_name)
  ctx = openmm.Context(omm_system, integrator, platform_obj)
  ctx.setPositions([openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0) for i in range(n_atoms)])
  ctx.setVelocitiesToTemperature(temp_q, seed)
  cold_start = time.perf_counter()
  ratios: list[float] = []
  t_state: list[float] = []
  t_atomic: list[float] = []
  p_tot_mag: list[float] = []
  ke_com: list[float] = []
  dof = _dof_rigid_tip3p_waters(n_waters)
  dof_atomic = _dof_atomic_unconstrained(n_atoms)
  v_unit = omm_unit.nanometer / omm_unit.picosecond

  def _to_float_in_unit(q, unit):  # OpenMM may return bare floats in some builds.
    if hasattr(q, "value_in_unit"):
      return float(q.value_in_unit(unit))
    return float(q)

  for step in range(n_steps):
    integrator.step(1)
    if step >= burn and (step - burn) % sample_every == 0:
      st = ctx.getState(getEnergy=True, getVelocities=True)
      ke_state = _to_float_in_unit(st.getKineticEnergy(), omm_unit.kilocalories_per_mole)
      vels = st.getVelocities()
      ke_sum = 0.0 * omm_unit.kilojoules_per_mole
      px = py = pz = 0.0
      m_tot = 0.0
      for i in range(n_atoms):
        m = omm_system.getParticleMass(i)
        v = vels[i]
        vx = _to_float_in_unit(v.x, v_unit)
        vy = _to_float_in_unit(v.y, v_unit)
        vz = _to_float_in_unit(v.z, v_unit)
        vsq = (vx * vx + vy * vy + vz * vz) * (omm_unit.nanometer / omm_unit.picosecond) ** 2
        ke_sum = ke_sum + 0.5 * m * vsq
        mi = _to_float_in_unit(m, omm_unit.dalton)
        px += mi * vx
        py += mi * vy
        pz += mi * vz
        m_tot += mi
      if hasattr(ke_sum, "value_in_unit"):
        ke_sum_f = float(ke_sum.value_in_unit(omm_unit.kilocalories_per_mole))
      else:
        ke_sum_f = float(ke_sum)
      ratios.append(ke_state / max(ke_sum_f, 1e-12))
      t_state.append(2.0 * ke_state / (dof * _BOLTZMANN_KCAL))
      if verbose_samples:
        t_atomic.append(2.0 * ke_sum_f / (dof_atomic * _BOLTZMANN_KCAL))
        p_mag = math.sqrt(px * px + py * py + pz * pz)
        p_tot_mag.append(p_mag)
        p_sq_q = (px * px + py * py + pz * pz) * (omm_unit.dalton * omm_unit.nanometer / omm_unit.picosecond) ** 2
        m_q = m_tot * omm_unit.dalton
        if m_tot > 1e-12:
          ke_com_q = 0.5 * p_sq_q / m_q
          ke_com_kcal = _to_float_in_unit(ke_com_q, omm_unit.kilocalories_per_mole)
        else:
          ke_com_kcal = 0.0
        ke_com.append(ke_com_kcal)
  cold_elapsed = time.perf_counter() - cold_start
  steady_elapsed = None
  if timing_mode in ("steady", "both"):
    for _ in range(max(warmup_steps, 0)):
      integrator.step(1)
    t0 = time.perf_counter()
    for _ in range(max(measure_steps, 1)):
      integrator.step(1)
    steady_elapsed = time.perf_counter() - t0
  omm_ver = getattr(openmm, "__version__", None)
  if omm_ver is None:
    try:
      from importlib.metadata import version

      omm_ver = version("openmm")
    except Exception:  # noqa: BLE001
      omm_ver = "unknown"
  samples: dict = {"ratio": ratios, "temperature": t_state}
  if verbose_samples:
    samples["temperature_atomic"] = t_atomic
    samples["p_tot_mag_dalton_nm_per_ps"] = p_tot_mag
    samples["ke_com_kcal_mol"] = ke_com
  meta_run = {
    "remove_cmmotion_requested": bool(remove_cmmotion),
    "openmm_system_has_cmmotion_remover": tip3p_ke_profile.openmm_system_has_cmmotion_remover(
      omm_system, openmm
    ),
    "openmm_integrator": openmm_integrator,
    "temperature_k": float(temperature_k),
    "kT_kcalmol": float(temperature_k) * _BOLTZMANN_KCAL,
    "dt_fs": float(dt_fs),
    "gamma_ps": float(gamma_ps),
    "dof_rigid": dof,
    "dof_atomic_diagnostic": dof_atomic,
    "gamma_dt_check": _gamma_dt_consistency(gamma_ps=gamma_ps, dt_fs=dt_fs, akma_time_unit_fs=akma_time_unit_fs),
  }
  return {
    "engine": "openmm",
    "openmm_platform": platform_name,
    "openmm_version": omm_ver,
    "samples": samples,
    "n_samples": len(ratios),
    "elapsed_cold_sec": cold_elapsed if timing_mode in ("cold", "both") else None,
    "elapsed_steady_sec": steady_elapsed,
    "steps_measured": measure_steps if steady_elapsed is not None else 0,
    "run_physics_meta": meta_run,
  }


def _prolix_run_once(
  *,
  n_waters: int,
  n_steps: int,
  burn: int,
  sample_every: int,
  seed: int,
  timing_mode: str,
  warmup_steps: int,
  measure_steps: int,
  dt_fs: float,
  temperature_k: float,
  gamma_ps: float,
  remove_linear_com_momentum: bool,
  verbose_samples: bool,
  akma_time_unit_fs: float,
  projection_site: str = "post_o",
  project_ou_momentum_rigid: bool = True,
  settle_velocity_iters: int = 10,
) -> dict:
  import jax
  import jax.numpy as jnp
  from jax_md import quantity
  from prolix.physics import pbc, settle, system
  from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
  from prolix.simulate import BOLTZMANN_KCAL

  t3 = _load_tip3p_module()
  positions_a, box_edge = _tip3p_grid_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  pme = dict(_REGRESSION_EXPLICIT_PME)
  alpha = float(pme["pme_alpha_per_angstrom"])
  grid = int(pme["pme_grid_points"])
  cutoff = float(pme["cutoff_angstrom"])
  dt_akma = float(dt_fs) / float(akma_time_unit_fs)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(akma_time_unit_fs) * 1e-3
  sys_dict = t3._prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False, pme_grid_points=grid, pme_alpha=alpha, cutoff_distance=cutoff, strict_parameterization=False)
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(
    energy_fn,
    shift_fn,
    dt=dt_akma,
    kT=kT,
    gamma=gamma_reduced,
    mass=mass,
    water_indices=water_indices,
    box=box_vec,
    remove_linear_com_momentum=bool(remove_linear_com_momentum),
    project_ou_momentum_rigid=bool(project_ou_momentum_rigid),
    projection_site=projection_site,
    settle_velocity_iters=int(settle_velocity_iters),
  )
  apply_s = jax.jit(apply_s)
  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  cold_start = time.perf_counter()
  ratios: list[float] = []
  t_rigid: list[float] = []
  t_atomic: list[float] = []
  p_tot_mag: list[float] = []
  ke_com: list[float] = []
  dof = _dof_rigid_tip3p_waters(n_waters)
  dof_atomic = _dof_atomic_unconstrained(n_atoms)
  for step in range(n_steps):
    state = apply_s(state)
    if step >= burn and (step - burn) % sample_every == 0:
      ke_a = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
      ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
      ratios.append(ke_a / max(ke_r, 1e-12))
      t_rigid.append(2.0 * ke_r / (dof * BOLTZMANN_KCAL))
      if verbose_samples:
        t_atomic.append(2.0 * ke_a / (dof_atomic * BOLTZMANN_KCAL))
        p_tot = jnp.sum(state.momentum, axis=0)
        m_tot = jnp.sum(state.mass)
        p_mag = float(jnp.linalg.norm(p_tot))
        p_tot_mag.append(p_mag)
        ke_cm = float(0.5 * jnp.dot(p_tot, p_tot) / jnp.maximum(m_tot, jnp.array(1e-12, dtype=m_tot.dtype)))
        ke_com.append(ke_cm)
  cold_elapsed = time.perf_counter() - cold_start
  steady_elapsed = None
  if timing_mode in ("steady", "both"):
    for _ in range(max(warmup_steps, 0)):
      state = apply_s(state)
    t0 = time.perf_counter()
    for _ in range(max(measure_steps, 1)):
      state = apply_s(state)
    steady_elapsed = time.perf_counter() - t0
  samples: dict = {"ratio": ratios, "temperature": t_rigid}
  if verbose_samples:
    samples["temperature_atomic"] = t_atomic
    samples["p_tot_mag_momentum_units"] = p_tot_mag
    samples["ke_com_kcal_mol"] = ke_com
  meta_run = {
    "remove_cmmotion_requested": None,
    "openmm_system_has_cmmotion_remover": None,
    "remove_linear_com_momentum": bool(remove_linear_com_momentum),
    "openmm_integrator": None,
    "temperature_k": float(temperature_k),
    "kT_kcalmol": float(temperature_k) * BOLTZMANN_KCAL,
    "dt_fs": float(dt_fs),
    "gamma_ps": float(gamma_ps),
    "projection_site": str(projection_site),
    "project_ou_momentum_rigid": bool(project_ou_momentum_rigid),
    "settle_velocity_iters": int(settle_velocity_iters),
    "dof_rigid": dof,
    "dof_atomic_diagnostic": dof_atomic,
    "gamma_dt_check": _gamma_dt_consistency(gamma_ps=gamma_ps, dt_fs=dt_fs, akma_time_unit_fs=akma_time_unit_fs),
  }
  return {
    "engine": "prolix",
    "samples": samples,
    "n_samples": len(ratios),
    "elapsed_cold_sec": cold_elapsed if timing_mode in ("cold", "both") else None,
    "elapsed_steady_sec": steady_elapsed,
    "steps_measured": measure_steps if steady_elapsed is not None else 0,
    "run_physics_meta": meta_run,
  }


def _summarize_engine_runs(engine_runs: list[dict], sample_every: int, dt_fs: float) -> dict:
  ratio_means = [float(statistics.fmean(r["samples"]["ratio"])) for r in engine_runs if r["samples"]["ratio"]]
  temp_means = [float(statistics.fmean(r["samples"]["temperature"])) for r in engine_runs if r["samples"]["temperature"]]
  ratio_all = [x for r in engine_runs for x in r["samples"]["ratio"]]
  block_size = max(10, len(ratio_all) // 20) if ratio_all else 10
  cold = [float(r["elapsed_cold_sec"]) for r in engine_runs if r["elapsed_cold_sec"] is not None]
  steady = [float(r["elapsed_steady_sec"]) for r in engine_runs if r["elapsed_steady_sec"] is not None and r["elapsed_steady_sec"] > 0]
  measured_steps = max([int(r["steps_measured"]) for r in engine_runs], default=0)
  steps_per_sec = (measured_steps / statistics.fmean(steady)) if steady and measured_steps > 0 else None
  dt_ns_per_step = float(dt_fs) * 1e-6
  ns_per_day = (steps_per_sec * dt_ns_per_step * 86400.0) if steps_per_sec is not None else None
  out = {
    "n_replicas": len(engine_runs),
    "n_samples_per_replica": [int(r["n_samples"]) for r in engine_runs],
    "replicate_ratio": _mean_stats(ratio_means, method="replicate_t95"),
    "replicate_temperature": _mean_stats(temp_means, method="replicate_t95"),
    "frame_ratio_block_diagnostic": _block_sem(ratio_all, block_size),
    "frame_ratio_total_samples": len(ratio_all),
    "sample_every": sample_every,
    "timing": {
      "elapsed_cold_sec_mean": float(statistics.fmean(cold)) if cold else None,
      "elapsed_steady_sec_mean": float(statistics.fmean(steady)) if steady else None,
      "steps_per_sec_steady": steps_per_sec,
      "ns_per_day_steady": ns_per_day,
      "cache_state": "fresh_process",
      "steps_measured": measured_steps,
    },
    "warnings": [],
  }
  if len(engine_runs) < 5:
    out["warnings"].append("replicate_warning:n_replicas<5_for_decision_claim")
  if out["frame_ratio_block_diagnostic"].get("warning"):
    out["warnings"].append(f"frame_warning:{out['frame_ratio_block_diagnostic']['warning']}")
  return out


def main() -> int:
  args = _parse_args()
  if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
  jax_meta = _configure_jax(args.jax_x64) if args.engine in ("prolix", "both") else {
    "requested_jax_x64": args.jax_x64,
    "effective_jax_x64": None,
    "jax_backend": None,
  }
  remove_cm = args.remove_cmmotion == "true"
  proj_ou = args.project_ou_momentum_rigid == "true"
  if args.projection_site == "both" and not proj_ou:
    print("error: projection_site=both requires --project-ou-momentum-rigid true", file=sys.stderr)
    return 2
  gamma_dt_top = _gamma_dt_consistency(gamma_ps=args.gamma_ps, dt_fs=args.dt_fs, akma_time_unit_fs=_AKMA_TIME_UNIT_FS)
  physics_warnings: list[str] = []
  if not gamma_dt_top["ok"]:
    physics_warnings.append("gamma_dt_consistency_check_failed")
  diagnostics: list[dict] = []
  for engine in ("openmm", "prolix"):
    if args.engine not in (engine, "both"):
      continue
    per_rep: list[dict] = []
    for rid in range(args.replicas):
      seed = args.seed + rid * 100_003
      try:
        if engine == "openmm":
          per_rep.append(
            _openmm_run_once(
              n_waters=args.n_waters,
              n_steps=args.steps,
              burn=args.burn,
              sample_every=args.sample_every,
              seed=seed,
              timing_mode=args.timing_mode,
              warmup_steps=args.warmup_steps,
              measure_steps=args.measure_steps,
              dt_fs=args.dt_fs,
              temperature_k=args.temperature_k,
              gamma_ps=args.gamma_ps,
              remove_cmmotion=remove_cm,
              openmm_integrator=args.openmm_integrator,
              verbose_samples=bool(args.verbose_samples),
              akma_time_unit_fs=_AKMA_TIME_UNIT_FS,
            )
          )
        else:
          per_rep.append(
            _prolix_run_once(
              n_waters=args.n_waters,
              n_steps=args.steps,
              burn=args.burn,
              sample_every=args.sample_every,
              seed=seed,
              timing_mode=args.timing_mode,
              warmup_steps=args.warmup_steps,
              measure_steps=args.measure_steps,
              dt_fs=args.dt_fs,
              temperature_k=args.temperature_k,
              gamma_ps=args.gamma_ps,
              remove_linear_com_momentum=remove_cm,
              verbose_samples=bool(args.verbose_samples),
              akma_time_unit_fs=_AKMA_TIME_UNIT_FS,
              projection_site=str(args.projection_site),
              project_ou_momentum_rigid=proj_ou,
              settle_velocity_iters=int(args.settle_velocity_iters),
            )
          )
      except Exception as e:  # noqa: BLE001
        diagnostics.append({"engine": engine, "error": str(e), "replica_id": rid, "seed": seed})
        per_rep = []
        break
    if engine == "openmm" and per_rep:
      m0 = per_rep[0].get("run_physics_meta") or {}
      req = bool(m0.get("remove_cmmotion_requested"))
      has = bool(m0.get("openmm_system_has_cmmotion_remover"))
      if req is not has:
        physics_warnings.append("openmm_remove_cmmotion_request_vs_force_mismatch")
    if engine == "prolix" and per_rep:
      m0p = per_rep[0].get("run_physics_meta") or {}
      if bool(m0p.get("remove_linear_com_momentum")) is not remove_cm:
        physics_warnings.append("prolix_remove_linear_com_momentum_mismatch_vs_cli_remove_cmmotion")
    if per_rep:
      summary = _summarize_engine_runs(per_rep, sample_every=args.sample_every, dt_fs=args.dt_fs)
      summary["engine"] = engine
      summary["openmm_platform"] = _REGRESSION_EXPLICIT_PME["openmm_platform"] if engine == "openmm" else None
      if engine == "openmm":
        summary["openmm_version"] = per_rep[0].get("openmm_version")
      summary["replicas"] = per_rep
      summary["profile_id"] = tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_cm)
      diagnostics.append(summary)
  payload = {
    "meta": {
      "schema": "tip3p_ke_compare/v1",
      "script": "tip3p_ke_compare.py",
      "python": platform.python_version(),
      "platform": platform.platform(),
      "openmm_install_mode": os.environ.get("OPENMM_INSTALL_MODE", "default"),
      "profile_id": tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_cm),
      "benchmark_policy": {
        "g4_gate_temperature_definition": "OpenMM: 2*getKineticEnergy/(dof_rigid*kB); Prolix: 2*rigid_tip3p_box_ke_kcal/(dof_rigid*kB); dof_rigid=6*n_waters-3",
        "remove_cmmotion_openmm": remove_cm,
        "prolix_remove_linear_com_momentum": remove_cm,
        "openmm_integrator": args.openmm_integrator,
        "dt_fs": float(args.dt_fs),
        "temperature_k": float(args.temperature_k),
        "gamma_ps": float(args.gamma_ps),
        "gamma_dt_consistency": gamma_dt_top,
        "verbose_samples": bool(args.verbose_samples),
      },
      "physics_warnings": physics_warnings,
      **jax_meta,
    },
    "config": {
      "engine": args.engine,
      "n_waters": args.n_waters,
      "steps": args.steps,
      "burn": args.burn,
      "sample_every": args.sample_every,
      "seed": args.seed,
      "replicas": args.replicas,
      "timing_mode": args.timing_mode,
      "warmup_steps": args.warmup_steps,
      "measure_steps": args.measure_steps,
      "dt_fs": float(args.dt_fs),
      "temperature_k": float(args.temperature_k),
      "gamma_ps": float(args.gamma_ps),
      "remove_cmmotion": args.remove_cmmotion,
      "openmm_integrator": args.openmm_integrator,
      "verbose_samples": bool(args.verbose_samples),
      "projection_site": str(args.projection_site),
      "settle_velocity_iters": int(args.settle_velocity_iters),
      "project_ou_momentum_rigid": str(args.project_ou_momentum_rigid),
    },
    "diagnostics": diagnostics,
  }
  print(json.dumps(payload, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
