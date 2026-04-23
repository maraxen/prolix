"""Linear COM momentum removal in ``settle_langevin`` vs OpenMM-style primary profile."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import quantity

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water, _write_wat_pdb

try:
  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)


def test_prolix_settle_langevin_remove_linear_suppresses_total_momentum() -> None:
  """With ``remove_linear_com_momentum=True``, ``|sum p|/sum(m)`` stays small (``tip3p_benchmark_policy.md``)."""
  n_waters = 4
  temperature_k = 300.0
  dt_fs = 2.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  alpha = 0.34
  grid = 32
  cutoff = 9.0
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
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
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_waters * 3)
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
    remove_linear_com_momentum=True,
  )
  apply_j = jax.jit(apply_s)
  key = jax.random.PRNGKey(991)
  state = init_s(key, jnp.array(positions_a), mass=mass)
  burn = 800
  for _ in range(burn):
    state = apply_j(state)
  samples: list[float] = []
  for _ in range(400):
    state = apply_j(state)
    p_tot = jnp.sum(state.momentum, axis=0)
    m_tot = jnp.sum(state.mass)
    samples.append(float(jnp.linalg.norm(p_tot) / jnp.maximum(m_tot, 1e-12)))
  mean_ratio = float(np.mean(samples))
  assert mean_ratio < 0.02, f"expected small COM momentum metric, got {mean_ratio}"


@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_prolix_short_window_mean_t_primary_profile(tmp_path: Path) -> None:
  """Paired short slice: ``openmm_ref_linear_com_on`` tolerances in ``tip3p_benchmark_policy.md``."""
  n_waters = 2
  temperature_k = 300.0
  dt_fs = 2.0
  gamma_ps = 1.0
  dof = _dof_rigid_tip3p_waters(n_waters)
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3

  pdb_path = tmp_path / "watbox_com_pair.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=0.9 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
    removeCMMotion=True,
  )
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(3.4, 32, 32, 32)

  integrator = openmm.LangevinMiddleIntegrator(
    temperature_k * omm_unit.kelvin,
    gamma_ps / omm_unit.picosecond,
    dt_fs * omm_unit.femtoseconds,
  )
  platform = openmm.Platform.getPlatformByName("Reference")
  ctx = openmm.Context(omm_system, integrator, platform)
  ctx.setPositions(
    [
      openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0)
      for i in range(n_atoms)
    ]
  )
  seed = 4242
  ctx.setVelocitiesToTemperature(temperature_k * omm_unit.kelvin, seed)

  temps_omm: list[float] = []
  burn = 400
  n_prod = 800
  for step in range(burn + n_prod):
    integrator.step(1)
    if step >= burn:
      st = ctx.getState(getEnergy=True)
      ke = st.getKineticEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
      temps_omm.append(2.0 * ke / (dof * BOLTZMANN_KCAL))

  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
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
    remove_linear_com_momentum=True,
  )
  apply_j = jax.jit(apply_s)
  key = jax.random.PRNGKey(seed)
  state = init_s(key, jnp.array(positions_a), mass=mass)
  temps_plx: list[float] = []
  p_ratios: list[float] = []
  for step in range(burn + n_prod):
    state = apply_j(state)
    if step >= burn:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
      temps_plx.append(2.0 * ke_r / (dof * BOLTZMANN_KCAL))
      p_tot = jnp.sum(state.momentum, axis=0)
      m_tot = jnp.sum(state.mass)
      p_ratios.append(float(jnp.linalg.norm(p_tot) / jnp.maximum(m_tot, 1e-12)))

  mean_omm = float(np.mean(temps_omm))
  mean_plx = float(np.mean(temps_plx))
  assert abs(mean_omm - mean_plx) < 80.0, f"mean T gap too large: OMM={mean_omm:.2f} PLX={mean_plx:.2f}"
  assert float(np.mean(p_ratios)) < 0.02
  _ = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)  # smoke: state valid
