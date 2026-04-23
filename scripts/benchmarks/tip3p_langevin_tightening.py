#!/usr/bin/env python3
"""Long-run TIP3P box Langevin for P2a-B2 tightening (replicas + checkpointing).

Designed for preemptable clusters: periodically writes OpenMM binary checkpoints or Prolix
``NVTLangevinState`` NPZ plus a JSON meta file and optional temperature trace.

Prolix summaries include **two** thermometers: ``mean_T_K`` uses rigid-body DOF ``6 N_w - 3``
(pairing OpenMM ``getKineticEnergy`` + ``rigidWater``). ``mean_T_atomic_K`` uses atomic
``\\sum_i p_i^2/(2m_i)`` with DOF ``3 N_{atoms}``, which better matches the per-atom OU noise in
``settle_langevin``.

PME parameters use ``REGRESSION_EXPLICIT_PME`` from ``prolix.physics.regression_explicit_pme``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
  sys.path.insert(0, str(_BENCH))
import tip3p_ke_profile  # noqa: E402

import jax
import jax.numpy as jnp
from jax_md import quantity
import numpy as np

from prolix.physics import pbc, settle, system
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.simulate import NVTLangevinState
from prolix.physics.water_models import WaterModelType, get_water_params
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


def _configure_jax(mode: str) -> dict:
  """Match ``tip3p_ke_compare._configure_jax`` keys for JSON meta alignment."""
  if mode not in ("off", "on"):
    return {"requested_jax_x64": mode, "effective_jax_x64": None, "jax_backend": None}
  jax.config.update("jax_enable_x64", mode == "on")
  effective = bool(jax.config.read("jax_enable_x64"))
  return {
    "requested_jax_x64": mode,
    "effective_jax_x64": effective,
    "jax_backend": jax.default_backend(),
  }


def _git_sha() -> str | None:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"],
      stderr=subprocess.DEVNULL,
      text=True,
    ).strip()
  except Exception:
    return None


def _tip3p_local_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  tip = get_water_params(WaterModelType.TIP3P)
  r = float(tip.r_OH)
  theta = 104.52 * math.pi / 180.0
  o = np.zeros(3, dtype=np.float64)
  h1 = np.array([r, 0.0, 0.0])
  h2 = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
  return o, h1, h2


def _dof_atomic_cartesian(n_atoms: int) -> float:
  """Thermometer DOF for \\sum_i p_i^2/(2m_i) (atomic Cartesian, no constraint correction)."""
  return float(3 * n_atoms)


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  """Effective DOF for rigid TIP3P in a periodic box (thermometer denominator).

  Each water contributes six rigid-body DOF (three translations + three rotations); subtract
  three for overall center-of-mass motion in PBC so ``dof = 6 * n_waters - 3``.

  Pair this with OpenMM ``State.getKineticEnergy()`` (computed from constrained particle
  velocities) and with Prolix ``rigid_tip3p_box_ke_kcal`` (COM + rotational KE per molecule),
  *not* with ``jax_md.quantity.kinetic_energy`` (sum of per-atom ``p^2/(2m)``, which still
  counts internal motion removed by SETTLE).
  """
  return float(6 * n_waters - 3)


def _grid_water_positions(n_waters: int, spacing_angstrom: float) -> tuple[np.ndarray, float]:
  o0, h1l, h2l = _tip3p_local_frame()
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
  sites = sites[:n_waters]

  base = np.array([3.0, 3.0, 3.0], dtype=np.float64)
  pos: list[np.ndarray] = []
  for ix, iy, iz in sites:
    o = base + np.array(
      [ix * spacing_angstrom, iy * spacing_angstrom, iz * spacing_angstrom],
      dtype=np.float64,
    )
    pos.append(o + o0)
    pos.append(o + h1l)
    pos.append(o + h2l)
  arr = np.vstack(pos)
  span = np.max(arr, axis=0) - np.min(arr, axis=0)
  box_edge = float(np.max(span) + 16.0)
  return arr, box_edge


def _prolix_params_pure_water(n_waters: int) -> dict:
  tip = get_water_params(WaterModelType.TIP3P)
  qo, qh = float(tip.charge_O), float(tip.charge_H)
  sig_o = float(tip.sigma_O)
  eps_o = float(tip.epsilon_O)
  n = n_waters * 3
  charges: list[float] = []
  sigmas: list[float] = []
  epsilons: list[float] = []
  for _ in range(n_waters):
    charges.extend([qo, qh, qh])
    sigmas.extend([sig_o, 1.0, 1.0])
    epsilons.extend([eps_o, 0.0, 0.0])
  mask = jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64)
  for w in range(n_waters):
    b = w * 3
    for i, j in [(0, 1), (0, 2), (1, 2)]:
      a, c = b + i, b + j
      mask = mask.at[a, c].set(0.0).at[c, a].set(0.0)
  return {
    "charges": jnp.array(charges, dtype=jnp.float64),
    "sigmas": jnp.array(sigmas, dtype=jnp.float64),
    "epsilons": jnp.array(epsilons, dtype=jnp.float64),
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
    "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "exclusion_mask": mask,
  }


def _write_wat_pdb(path: Path, positions_angstrom: np.ndarray, box_angstrom: float) -> None:
  n_atoms = positions_angstrom.shape[0]
  assert n_atoms % 3 == 0
  n_res = n_atoms // 3
  lines: list[str] = [
    f"CRYST1{box_angstrom:9.3f}{box_angstrom:9.3f}{box_angstrom:9.3f}  90.00  90.00  90.00 P 1\n",
  ]
  for i in range(n_atoms):
    serial = i + 1
    res_seq = i // 3 + 1
    x, y, z = positions_angstrom[i]
    if i % 3 == 0:
      lines.append(
        f"HETATM{serial:5d}  O   HOH A{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n"
      )
    elif i % 3 == 1:
      lines.append(
        f"HETATM{serial:5d}  H1  HOH A{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n"
      )
    else:
      lines.append(
        f"HETATM{serial:5d}  H2  HOH A{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           H\n"
      )
  for r in range(n_res):
    o = 3 * r + 1
    h1 = 3 * r + 2
    h2 = 3 * r + 3
    lines.append(f"CONECT{o:5d}{h1:5d}{h2:5d}\n")
    lines.append(f"CONECT{h1:5d}{o:5d}\n")
    lines.append(f"CONECT{h2:5d}{o:5d}\n")
  lines.append("END\n")
  path.write_text("".join(lines))


def _save_meta(path: Path, payload: dict) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_meta(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _save_prolix_temps_wip(path: Path, rigid: list[float], atomic: list[float]) -> None:
  np.savez(path, rigid=np.asarray(rigid, dtype=np.float64), atomic=np.asarray(atomic, dtype=np.float64))


def _load_prolix_temps_wip(npz_path: Path, legacy_npy: Path) -> tuple[list[float], list[float]]:
  if npz_path.is_file():
    z = np.load(npz_path)
    return z["rigid"].astype(np.float64).tolist(), z["atomic"].astype(np.float64).tolist()
  if legacy_npy.is_file():
    rigid = np.load(legacy_npy).astype(np.float64).tolist()
    return rigid, [float("nan")] * len(rigid)
  return [], []


def _nan_mean_std(xs: np.ndarray) -> tuple[float, float]:
  if xs.size == 0:
    return float("nan"), 0.0
  finite = xs[np.isfinite(xs)]
  if finite.size == 0:
    return float("nan"), 0.0
  mean_t = float(np.mean(finite))
  std_t = float(statistics.pstdev(finite.tolist())) if finite.size > 1 else 0.0
  return mean_t, std_t


def _p50_p95(xs: list[float]) -> tuple[float, float]:
  arr = np.asarray(xs, dtype=np.float64)
  finite = arr[np.isfinite(arr)]
  if finite.size == 0:
    return float("nan"), float("nan")
  return float(np.percentile(finite, 50.0)), float(np.percentile(finite, 95.0))


def _bond_residual_max_abs(position: jax.Array, water_indices: jax.Array) -> float:
  if int(water_indices.shape[0]) == 0:
    return 0.0
  o = position[water_indices[:, 0]]
  h1 = position[water_indices[:, 1]]
  h2 = position[water_indices[:, 2]]
  oh1 = jnp.linalg.norm(h1 - o, axis=1)
  oh2 = jnp.linalg.norm(h2 - o, axis=1)
  hh = jnp.linalg.norm(h2 - h1, axis=1)
  e1 = jnp.max(jnp.abs(oh1 - settle.TIP3P_ROH))
  e2 = jnp.max(jnp.abs(oh2 - settle.TIP3P_ROH))
  e3 = jnp.max(jnp.abs(hh - settle.TIP3P_RHH))
  return float(jnp.maximum(jnp.maximum(e1, e2), e3))


def run_openmm(
  *,
  checkpoint_dir: Path,
  resume: bool,
  checkpoint_every: int,
  replica_id: int,
  n_waters: int,
  temperature_k: float,
  gamma_ps: float,
  dt_fs: float,
  total_steps: int,
  burn_in: int,
  sample_every: int,
  pme: dict,
  remove_cmmotion: bool = True,
  diagnostics_level: str = "off",
  diagnostics_decimation: int = 10,
  run_metadata: dict | None = None,
) -> dict:
  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  chk_path = checkpoint_dir / "state.chk"
  meta_path = checkpoint_dir / "meta.json"
  temps_path = checkpoint_dir / "temps_wip.npy"
  summary_path = checkpoint_dir / "summary.json"
  profile_id = tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_cmmotion)

  if resume and meta_path.is_file():
    done = _load_meta(meta_path)
    prev_rm = done.get("remove_cmmotion_requested")
    if prev_rm is not None and bool(prev_rm) is not bool(remove_cmmotion):
      msg = f"resume meta remove_cmmotion_requested={prev_rm!r} != current={remove_cmmotion!r}"
      raise ValueError(msg)
    if done.get("complete") and int(done.get("next_step", 0)) >= total_steps and summary_path.is_file():
      return json.loads(summary_path.read_text())

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3
  dof_therm = _dof_rigid_tip3p_waters(n_waters)

  alpha = float(pme["pme_alpha_per_angstrom"])
  grid = int(pme["pme_grid_points"])
  cutoff = float(pme["cutoff_angstrom"])
  platform_name = os.environ.get("OPENMM_PLATFORM", str(pme["openmm_platform"]))

  pdb_path = checkpoint_dir / "watbox.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=cutoff / 10.0 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
    removeCMMotion=bool(remove_cmmotion),
  )
  has_cm = tip3p_ke_profile.openmm_system_has_cmmotion_remover(omm_system, openmm)
  if bool(remove_cmmotion) is not has_cm:
    msg = f"OpenMM removeCMMotion={remove_cmmotion!r} but CMMotionRemover present={has_cm!r}"
    raise RuntimeError(msg)
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(alpha * 10.0, grid, grid, grid)
      f.setUseDispersionCorrection(bool(pme["use_dispersion_correction"]))

  integrator = openmm.LangevinMiddleIntegrator(
    temperature_k * omm_unit.kelvin,
    gamma_ps / omm_unit.picosecond,
    dt_fs * omm_unit.femtoseconds,
  )
  platform = openmm.Platform.getPlatformByName(platform_name)
  ctx = openmm.Context(omm_system, integrator, platform)

  seed = 4242 + replica_id * 100_003
  next_step = 0
  temps: list[float] = []

  if resume and chk_path.is_file() and meta_path.is_file():
    meta = _load_meta(meta_path)
    next_step = int(meta.get("next_step", 0))
    prev_rm = meta.get("remove_cmmotion_requested")
    if prev_rm is not None and bool(prev_rm) is not bool(remove_cmmotion):
      msg = f"checkpoint meta remove_cmmotion_requested={prev_rm!r} != current={remove_cmmotion!r}"
      raise ValueError(msg)
    ctx.loadCheckpoint(chk_path.read_bytes())
    if temps_path.is_file():
      temps = np.load(temps_path).astype(np.float64).tolist()
  else:
    ctx.setPositions(
      [
        openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0)
        for i in range(n_atoms)
      ]
    )
    ctx.setVelocitiesToTemperature(temperature_k * omm_unit.kelvin, seed)

  for step in range(next_step, total_steps):
    integrator.step(1)
    if step >= burn_in and (step - burn_in) % sample_every == 0:
      # OpenMM: ``getKineticEnergy()`` uses constrained particle velocities (User Guide /
      # State::KineticEnergy) and is the appropriate scalar to pair with ``6*Nw - 3`` DOF for
      # rigid ``rigidWater`` + ``HBonds`` TIP3P.
      st = ctx.getState(getEnergy=True)
      ke = st.getKineticEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
      temps.append(2.0 * ke / (dof_therm * BOLTZMANN_KCAL))

    if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
      chk_path.write_bytes(ctx.createCheckpoint())
      _save_meta(
        meta_path,
        {
          "engine": "openmm",
          "replica_id": replica_id,
          "next_step": step + 1,
          "total_steps": total_steps,
          "burn_in": burn_in,
          "sample_every": sample_every,
          "n_waters": n_waters,
          "temperature_k": temperature_k,
          "platform": platform_name,
          "profile_id": profile_id,
          "remove_cmmotion_requested": bool(remove_cmmotion),
          "openmm_system_has_cmmotion_remover": has_cm,
          "gamma_ps": float(gamma_ps),
          "diagnostics_level": diagnostics_level,
          "diagnostics_decimation": int(diagnostics_decimation),
          "run_metadata": run_metadata,
        },
      )
      np.save(temps_path, np.array(temps, dtype=np.float64))

  chk_path.write_bytes(ctx.createCheckpoint())
  _save_meta(
    meta_path,
    {
      "engine": "openmm",
      "replica_id": replica_id,
      "next_step": total_steps,
      "total_steps": total_steps,
      "burn_in": burn_in,
      "sample_every": sample_every,
      "n_waters": n_waters,
      "temperature_k": temperature_k,
      "platform": platform_name,
      "complete": True,
      "profile_id": profile_id,
      "remove_cmmotion_requested": bool(remove_cmmotion),
      "openmm_system_has_cmmotion_remover": has_cm,
      "gamma_ps": float(gamma_ps),
      "diagnostics_level": diagnostics_level,
      "diagnostics_decimation": int(diagnostics_decimation),
      "run_metadata": run_metadata,
    },
  )
  np.save(temps_path, np.array(temps, dtype=np.float64))

  prod = np.array(temps, dtype=np.float64)
  mean_t = float(statistics.fmean(prod)) if len(prod) else float("nan")
  std_t = float(statistics.pstdev(prod)) if len(prod) > 1 else 0.0
  summary = {
    "engine": "openmm",
    "replica_id": replica_id,
    "n_samples": len(prod),
    "mean_T_K": mean_t,
    "std_T_K": std_t,
    "target_T_K": temperature_k,
    "total_steps": total_steps,
    "burn_in": burn_in,
    "n_waters": n_waters,
    "platform": platform_name,
    "profile_id": profile_id,
    "remove_cmmotion_requested": bool(remove_cmmotion),
    "openmm_system_has_cmmotion_remover": has_cm,
    "gamma_ps": float(gamma_ps),
    "diagnostics_level": diagnostics_level,
    "diagnostics_decimation": int(diagnostics_decimation),
    "run_metadata": run_metadata,
  }
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  return summary


def run_prolix(
  *,
  checkpoint_dir: Path,
  resume: bool,
  checkpoint_every: int,
  replica_id: int,
  n_waters: int,
  temperature_k: float,
  gamma_ps: float,
  dt_fs: float,
  total_steps: int,
  burn_in: int,
  sample_every: int,
  pme: dict,
  remove_linear_com_momentum: bool = True,
  project_ou_momentum_rigid: bool = True,
  projection_site: str = "post_o",
  settle_velocity_iters: int = 10,
  diagnostics_level: str = "off",
  diagnostics_decimation: int = 10,
  run_metadata: dict | None = None,
) -> dict:
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  state_path = checkpoint_dir / "state.npz"
  meta_path = checkpoint_dir / "meta.json"
  temps_npz_path = checkpoint_dir / "temps_wip.npz"
  legacy_temps_npy_path = checkpoint_dir / "temps_wip.npy"
  summary_path = checkpoint_dir / "summary.json"
  profile_id = tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_linear_com_momentum)

  if resume and meta_path.is_file():
    done = _load_meta(meta_path)
    prev_lin = done.get("remove_linear_com_momentum")
    if prev_lin is not None and bool(prev_lin) is not bool(remove_linear_com_momentum):
      msg = f"resume meta remove_linear_com_momentum={prev_lin!r} != current={remove_linear_com_momentum!r}"
      raise ValueError(msg)
    if done.get("complete") and int(done.get("next_step", 0)) >= total_steps and summary_path.is_file():
      return json.loads(summary_path.read_text())

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

  alpha = float(pme["pme_alpha_per_angstrom"])
  grid = int(pme["pme_grid_points"])
  cutoff = float(pme["cutoff_angstrom"])

  dt_akma = dt_fs / float(AKMA_TIME_UNIT_FS)
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=grid,
    pme_alpha=alpha,
    cutoff_distance=cutoff,
    strict_parameterization=False,
  )

  kT = temperature_k * BOLTZMANN_KCAL
  gamma_reduced = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)
  dof_atomic = _dof_atomic_cartesian(n_atoms)

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
    settle_velocity_iters=settle_velocity_iters,
  )
  apply_s = jax.jit(apply_s)

  base_seed = 7_031 + replica_id * 100_003
  next_step = 0
  temps_rigid: list[float] = []
  temps_atomic: list[float] = []
  diag_projection_residual: list[float] = []
  diag_bond_residual_max_abs: list[float] = []
  diag_com_metric: list[float] = []

  if resume and state_path.is_file() and meta_path.is_file():
    meta = _load_meta(meta_path)
    next_step = int(meta.get("next_step", 0))
    prev_lin = meta.get("remove_linear_com_momentum")
    if prev_lin is not None and bool(prev_lin) is not bool(remove_linear_com_momentum):
      msg = f"checkpoint meta remove_linear_com_momentum={prev_lin!r} != current={remove_linear_com_momentum!r}"
      raise ValueError(msg)
    z = np.load(state_path)
    state = NVTLangevinState(
      jnp.asarray(z["position"]),
      jnp.asarray(z["momentum"]),
      jnp.asarray(z["force"]),
      jnp.asarray(z["mass"]),
      jnp.asarray(z["rng"]),
    )
    temps_rigid, temps_atomic = _load_prolix_temps_wip(temps_npz_path, legacy_temps_npy_path)
  else:
    key = jax.random.PRNGKey(base_seed)
    state = init_s(key, jnp.array(positions_a), mass=mass)

  for step in range(next_step, total_steps):
    state = apply_s(state)
    if step >= burn_in and (step - burn_in) % sample_every == 0:
      ke_r = float(
        rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters)
      )
      ke_a = float(quantity.kinetic_energy(momentum=state.momentum, mass=state.mass))
      temps_rigid.append(2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL))
      temps_atomic.append(2.0 * ke_a / (dof_atomic * BOLTZMANN_KCAL))
      if diagnostics_level != "off":
        sample_idx = len(temps_rigid) - 1
        decim = max(int(diagnostics_decimation), 1)
        if sample_idx % decim == 0:
          p_proj = settle.project_tip3p_waters_momentum_rigid(
            state.momentum, state.position, state.mass, water_indices
          )
          num = float(jnp.linalg.norm(state.momentum - p_proj))
          den = float(jnp.linalg.norm(state.momentum))
          diag_projection_residual.append(num / max(den, 1e-12))
          diag_bond_residual_max_abs.append(_bond_residual_max_abs(state.position, water_indices))
          p_tot = jnp.sum(state.momentum, axis=0)
          m_tot = jnp.sum(state.mass)
          diag_com_metric.append(float(jnp.linalg.norm(p_tot) / jnp.maximum(m_tot, 1e-12)))

    if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
      np.savez_compressed(
        state_path,
        position=np.asarray(state.position),
        momentum=np.asarray(state.momentum),
        force=np.asarray(state.force),
        mass=np.asarray(state.mass),
        rng=np.asarray(state.rng),
      )
      _save_meta(
        meta_path,
        {
          "engine": "prolix",
          "replica_id": replica_id,
          "next_step": step + 1,
          "total_steps": total_steps,
          "burn_in": burn_in,
          "sample_every": sample_every,
          "n_waters": n_waters,
          "temperature_k": temperature_k,
          "profile_id": profile_id,
          "remove_linear_com_momentum": bool(remove_linear_com_momentum),
          "project_ou_momentum_rigid": bool(project_ou_momentum_rigid),
          "projection_site": projection_site,
          "settle_velocity_iters": int(settle_velocity_iters),
          "diagnostics_level": diagnostics_level,
          "diagnostics_decimation": int(diagnostics_decimation),
          "gamma_ps": float(gamma_ps),
          "run_metadata": run_metadata,
        },
      )
      _save_prolix_temps_wip(temps_npz_path, temps_rigid, temps_atomic)

  np.savez_compressed(
    state_path,
    position=np.asarray(state.position),
    momentum=np.asarray(state.momentum),
    force=np.asarray(state.force),
    mass=np.asarray(state.mass),
    rng=np.asarray(state.rng),
  )
  _save_meta(
    meta_path,
    {
      "engine": "prolix",
      "replica_id": replica_id,
      "next_step": total_steps,
      "total_steps": total_steps,
      "burn_in": burn_in,
      "sample_every": sample_every,
      "n_waters": n_waters,
      "temperature_k": temperature_k,
      "complete": True,
      "profile_id": profile_id,
      "remove_linear_com_momentum": bool(remove_linear_com_momentum),
      "project_ou_momentum_rigid": bool(project_ou_momentum_rigid),
      "projection_site": projection_site,
      "settle_velocity_iters": int(settle_velocity_iters),
      "diagnostics_level": diagnostics_level,
      "diagnostics_decimation": int(diagnostics_decimation),
      "gamma_ps": float(gamma_ps),
      "run_metadata": run_metadata,
    },
  )
  _save_prolix_temps_wip(temps_npz_path, temps_rigid, temps_atomic)

  prod_r = np.asarray(temps_rigid, dtype=np.float64)
  prod_a = np.asarray(temps_atomic, dtype=np.float64)
  mean_t, std_t = _nan_mean_std(prod_r)
  mean_t_atomic, std_t_atomic = _nan_mean_std(prod_a)
  n_samples = int(prod_r.size)
  proj_p50, proj_p95 = _p50_p95(diag_projection_residual)
  bond_p50, bond_p95 = _p50_p95(diag_bond_residual_max_abs)
  com_p50, com_p95 = _p50_p95(diag_com_metric)
  summary = {
    "engine": "prolix",
    "replica_id": replica_id,
    "n_samples": n_samples,
    "mean_T_K": mean_t,
    "std_T_K": std_t,
    "mean_T_atomic_K": mean_t_atomic,
    "std_T_atomic_K": std_t_atomic,
    "thermometer_dof_rigid": dof_rigid,
    "thermometer_dof_atomic": dof_atomic,
    "target_T_K": temperature_k,
    "total_steps": total_steps,
    "burn_in": burn_in,
    "n_waters": n_waters,
    "profile_id": profile_id,
    "remove_linear_com_momentum": bool(remove_linear_com_momentum),
    "project_ou_momentum_rigid": bool(project_ou_momentum_rigid),
    "projection_site": projection_site,
    "settle_velocity_iters": int(settle_velocity_iters),
    "diagnostics_level": diagnostics_level,
    "diagnostics_decimation": int(diagnostics_decimation),
    "gamma_ps": float(gamma_ps),
    "diag_projection_residual_p50": proj_p50,
    "diag_projection_residual_p95": proj_p95,
    "diag_bond_residual_max_abs_p50": bond_p50,
    "diag_bond_residual_max_abs_p95": bond_p95,
    "diag_com_metric_p50": com_p50,
    "diag_com_metric_p95": com_p95,
    "run_metadata": run_metadata,
  }
  summary_path.write_text(json.dumps(summary, indent=2) + "\n")
  return summary


def main() -> int:
  parser = argparse.ArgumentParser(description="TIP3P box Langevin tightening run with checkpoints.")
  parser.add_argument("--engine", choices=("openmm", "prolix", "both"), default="openmm")
  parser.add_argument("--checkpoint-dir", type=Path, required=True)
  parser.add_argument("--checkpoint-every", type=int, default=500)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--replica-id", type=int, default=-1)
  parser.add_argument("--n-waters", type=int, default=33)
  parser.add_argument("--temperature", type=float, default=300.0)
  parser.add_argument("--gamma-ps", type=float, default=1.0, help="Langevin friction coefficient (1/ps).")
  parser.add_argument("--dt-fs", type=float, default=2.0)
  parser.add_argument("--total-steps", type=int, default=50_000)
  parser.add_argument("--burn-in", type=int, default=10_000)
  parser.add_argument("--sample-every", type=int, default=10)
  parser.add_argument(
    "--remove-cmmotion",
    choices=("true", "false"),
    default="true",
    help="OpenMM createSystem(removeCMMotion=...). Prolix settle_langevin(remove_linear_com_momentum=...) matches.",
  )
  parser.add_argument(
    "--jax-x64",
    choices=("off", "on"),
    default="off",
    help="JAX 64-bit floats (prolix/both only). Configure before JIT. Env JAX_ENABLE_X64 still applies at process start.",
  )
  parser.add_argument(
    "--project-ou-momentum-rigid",
    choices=("true", "false"),
    default="true",
    help="Apply rigid momentum projection in Prolix SETTLE-Langevin flow.",
  )
  parser.add_argument(
    "--projection-site",
    choices=("post_o", "post_settle_vel", "both"),
    default="post_o",
    help="Where to apply rigid momentum projection in Prolix integrator.",
  )
  parser.add_argument(
    "--settle-velocity-iters",
    type=int,
    default=10,
    help="RATTLE-like velocity correction iterations in SETTLE.",
  )
  parser.add_argument(
    "--diagnostics-level",
    choices=("off", "light", "full"),
    default="off",
    help="Reserved diagnostics output level for tightening runs.",
  )
  parser.add_argument(
    "--diagnostics-decimation",
    type=int,
    default=10,
    help="Sampling decimation for diagnostics collection metadata.",
  )
  args = parser.parse_args()
  remove_cm = args.remove_cmmotion == "true"
  project_ou = args.project_ou_momentum_rigid == "true"

  if args.projection_site == "both" and not project_ou:
    raise ValueError("projection_site='both' requires --project-ou-momentum-rigid true")

  jax_meta = (
    _configure_jax(args.jax_x64)
    if args.engine in ("prolix", "both")
    else {
      "requested_jax_x64": args.jax_x64,
      "effective_jax_x64": None,
      "jax_backend": None,
    }
  )

  rid = args.replica_id
  if rid < 0:
    rid = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

  pme = dict(REGRESSION_EXPLICIT_PME)
  run_metadata = {
    "git_sha": _git_sha(),
    "jax_backend": jax_meta.get("jax_backend"),
    "profile_id": tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_cm),
    "gamma_ps": float(args.gamma_ps),
    "projection_site": args.projection_site,
    "settle_velocity_iters": int(args.settle_velocity_iters),
    "diagnostics_level": args.diagnostics_level,
    "diagnostics_decimation": int(args.diagnostics_decimation),
    "effective_integrator_config": {
      "project_ou_momentum_rigid": bool(project_ou),
      "projection_site": args.projection_site,
      "settle_velocity_iters": int(args.settle_velocity_iters),
      "remove_linear_com_momentum": bool(remove_cm),
    },
  }

  engines = ("openmm", "prolix") if args.engine == "both" else (args.engine,)

  summaries = []
  for eng in engines:
    subdir = args.checkpoint_dir / eng
    if eng == "openmm":
      summaries.append(
        run_openmm(
          checkpoint_dir=subdir,
          resume=args.resume,
          checkpoint_every=args.checkpoint_every,
          replica_id=rid,
          n_waters=args.n_waters,
          temperature_k=args.temperature,
          gamma_ps=args.gamma_ps,
          dt_fs=args.dt_fs,
          total_steps=args.total_steps,
          burn_in=args.burn_in,
          sample_every=args.sample_every,
          pme=pme,
          remove_cmmotion=remove_cm,
          diagnostics_level=args.diagnostics_level,
          diagnostics_decimation=args.diagnostics_decimation,
          run_metadata=run_metadata,
        )
      )
    else:
      summaries.append(
        run_prolix(
          checkpoint_dir=subdir,
          resume=args.resume,
          checkpoint_every=args.checkpoint_every,
          replica_id=rid,
          n_waters=args.n_waters,
          temperature_k=args.temperature,
          gamma_ps=args.gamma_ps,
          dt_fs=args.dt_fs,
          total_steps=args.total_steps,
          burn_in=args.burn_in,
          sample_every=args.sample_every,
          pme=pme,
          remove_linear_com_momentum=remove_cm,
          project_ou_momentum_rigid=project_ou,
          projection_site=args.projection_site,
          settle_velocity_iters=args.settle_velocity_iters,
          diagnostics_level=args.diagnostics_level,
          diagnostics_decimation=args.diagnostics_decimation,
          run_metadata=run_metadata,
        )
      )

  print(
    json.dumps(
      {
        "replica_id": rid,
        "remove_cmmotion": args.remove_cmmotion,
        "profile_id": tip3p_ke_profile.profile_id_from_remove_cmmotion(remove_cm),
        "jax": jax_meta,
        "run_metadata": run_metadata,
        "runs": summaries,
      },
      indent=2,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
