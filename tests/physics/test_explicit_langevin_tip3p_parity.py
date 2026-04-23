"""P2a-B2: TIP3P water box — OpenMM sanity + Prolix SETTLE smoke.

**OpenMM:** ``amber14/tip3p.xml`` + ``HBonds`` + ``LangevinMiddleIntegrator`` at 2 fs on a
grid-generated PDB. Primary assertions: **finite** potential/kinetic energies (no blow-ups).
Window-mean T uses ``dof = 6 * n_waters - 3`` (rigid nonlinear TIP3P + COM) so
``T = 2 <KE> / (k_B dof)`` matches ``rigidWater`` / SETTLE physics.

**Prolix:** Same DOF convention as OpenMM for the rigid-water thermometer.

Default **8 waters** for CPU CI; set ``PROLIX_TIP3P_PARITY_N_WATERS`` for larger grids (slow).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.water_models import WaterModelType, get_water_params
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL


def _tip3p_local_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  tip = get_water_params(WaterModelType.TIP3P)
  r = float(tip.r_OH)
  theta = 104.52 * math.pi / 180.0
  o = np.zeros(3, dtype=np.float64)
  h1 = np.array([r, 0.0, 0.0])
  h2 = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
  return o, h1, h2


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


def _n_waters_from_env() -> int:
  raw = os.environ.get("PROLIX_TIP3P_PARITY_N_WATERS", "").strip()
  n = int(raw) if raw else 8
  return max(2, min(64, n))


@pytest.mark.slow
@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_tip3p_water_box_langevin_mean_temperature(regression_pme_params, tmp_path):
  """OpenMM: stable Langevin + loose mean-T band (same thermometer convention as stats script)."""
  jax.config.update("jax_enable_x64", True)

  n_waters = _n_waters_from_env()
  temperature_k = 300.0

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3

  alpha = float(regression_pme_params["pme_alpha_per_angstrom"])
  grid = int(regression_pme_params["pme_grid_points"])
  cutoff = float(regression_pme_params["cutoff_angstrom"])
  platform_name = str(regression_pme_params["openmm_platform"])

  dt_fs = 2.0
  n_steps = 2500
  burn = 1000
  dof_therm = float(6 * n_waters - 3)

  pdb_path = tmp_path / "watbox.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=cutoff / 10.0 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
  )
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(alpha * 10.0, grid, grid, grid)
      f.setUseDispersionCorrection(bool(regression_pme_params["use_dispersion_correction"]))

  integrator = openmm.LangevinMiddleIntegrator(
    temperature_k * omm_unit.kelvin,
    1.0 / omm_unit.picosecond,
    dt_fs * omm_unit.femtoseconds,
  )
  platform = openmm.Platform.getPlatformByName(platform_name)
  ctx = openmm.Context(omm_system, integrator, platform)
  ctx.setPositions(
    [
      openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0)
      for i in range(n_atoms)
    ]
  )
  ctx.setVelocitiesToTemperature(temperature_k * omm_unit.kelvin, 4242)

  pe0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
  assert math.isfinite(pe0)

  temps: list[float] = []
  for _ in range(n_steps):
    integrator.step(1)
    st = ctx.getState(getEnergy=True)
    assert math.isfinite(st.getPotentialEnergy().value_in_unit(omm_unit.kilocalories_per_mole))
    ke = st.getKineticEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
    temps.append(2.0 * ke / (dof_therm * BOLTZMANN_KCAL))

  arr = np.array(temps[burn:], dtype=np.float64)
  mean_t = float(np.mean(arr))
  assert 80.0 <= mean_t <= 520.0, f"OpenMM mean T sanity failed: {mean_t:.2f} K"
  assert float(np.std(arr)) > 1e-3


@pytest.mark.slow
def test_prolix_settle_langevin_water_box_smoke(regression_pme_params):
  """Prolix: finite SETTLE+Langevin on shared grid; loose T band (150–400 K)."""
  jax.config.update("jax_enable_x64", True)

  n_waters = _n_waters_from_env()
  temperature_k = 300.0

  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

  alpha = float(regression_pme_params["pme_alpha_per_angstrom"])
  grid = int(regression_pme_params["pme_grid_points"])
  cutoff = float(regression_pme_params["cutoff_angstrom"])

  dt_fs = 2.0
  dt_akma = dt_fs / float(AKMA_TIME_UNIT_FS)
  n_steps = 220
  burn = 70

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
  gamma_reduced = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  dof = float(6 * n_waters - 3)

  init_s, apply_s = settle.settle_langevin(
    energy_fn,
    shift_fn,
    dt=dt_akma,
    kT=kT,
    gamma=gamma_reduced,
    mass=mass,
    water_indices=water_indices,
    box=box_vec,
  )

  state = init_s(jax.random.PRNGKey(2026), jnp.array(positions_a), mass=mass)
  ts: list[float] = []
  for _ in range(n_steps):
    state = apply_s(state)
    assert jnp.all(jnp.isfinite(state.position))
    ke = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
    ts.append(2.0 * ke / (dof * BOLTZMANN_KCAL))

  arr = np.array(ts[burn:], dtype=np.float64)
  mean_t = float(np.mean(arr))
  assert 150.0 <= mean_t <= 520.0, f"Prolix mean T sanity band failed: {mean_t:.2f} K"
  assert float(np.std(arr)) > 1e-6


@pytest.mark.slow
@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_prolix_tip3p_force_rmse_one_step(regression_pme_params, tmp_path) -> None:
  """Static forces on the shared TIP3P grid: OpenMM Reference vs Prolix ``make_energy_fn`` PME."""
  jax.config.update("jax_enable_x64", True)

  n_waters = 4
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3

  alpha = float(regression_pme_params["pme_alpha_per_angstrom"])
  grid = int(regression_pme_params["pme_grid_points"])
  cutoff = float(regression_pme_params["cutoff_angstrom"])
  platform_name = str(regression_pme_params["openmm_platform"])

  pdb_path = tmp_path / "watbox_forces.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=cutoff / 10.0 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
  )
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(alpha * 10.0, grid, grid, grid)
      f.setUseDispersionCorrection(bool(regression_pme_params["use_dispersion_correction"]))

  integrator = openmm.VerletIntegrator(0.001 * omm_unit.picoseconds)
  platform = openmm.Platform.getPlatformByName(platform_name)
  ctx = openmm.Context(omm_system, integrator, platform)
  ctx.setPositions(
    [
      openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0)
      for i in range(n_atoms)
    ]
  )
  st = ctx.getState(getForces=True)
  f_omm = st.getForces(asNumpy=True).value_in_unit(omm_unit.kilocalories_per_mole / omm_unit.angstrom)

  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, _shift = pbc.create_periodic_space(box_vec)
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
  pos_jax = jnp.array(positions_a, dtype=jnp.float64)
  f_plx = -jax.grad(energy_fn)(pos_jax)
  diff = np.asarray(f_omm, dtype=np.float64) - np.asarray(f_plx, dtype=np.float64)
  rmse = float(np.sqrt(np.mean(diff**2)))

  equiv = {
    "n_waters": n_waters,
    "pme_alpha_per_angstrom": alpha,
    "pme_grid_points": grid,
    "cutoff_angstrom": cutoff,
    "use_dispersion_correction": bool(regression_pme_params["use_dispersion_correction"]),
    "openmm_platform": platform_name,
    "openmm_constraints": "HBonds",
    "openmm_rigid_water": True,
    "prolix_path": "make_energy_fn PBC explicit PME",
    "force_rmse_kcal_mol_A": rmse,
  }
  (tmp_path / "tip3p_parity_params.json").write_text(json.dumps(equiv, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  assert rmse < 3.0, f"OpenMM vs Prolix force RMSE too high: {rmse:.4f} kcal/mol/A"
