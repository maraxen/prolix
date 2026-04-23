"""OpenMM rigid-water kinetic energy vs thermometer (``6 Nw - 3`` DOF)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prolix.simulate import BOLTZMANN_KCAL

try:
  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False


@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_rigid_water_get_ke_temperature_near_target(tmp_path) -> None:
  """Short Reference run: ``2*KE/(k_B dof)`` with ``dof=6*Nw-3`` stays near bath temperature."""
  from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _write_wat_pdb

  n_waters = 2
  temperature_k = 300.0
  dof = float(6 * n_waters - 3)
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3

  pdb_path = tmp_path / "watbox.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=0.9 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
  )
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(3.4, 32, 32, 32)

  integrator = openmm.LangevinMiddleIntegrator(
    temperature_k * omm_unit.kelvin,
    1.0 / omm_unit.picosecond,
    2.0 * omm_unit.femtoseconds,
  )
  platform = openmm.Platform.getPlatformByName("Reference")
  ctx = openmm.Context(omm_system, integrator, platform)
  ctx.setPositions(
    [
      openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0)
      for i in range(n_atoms)
    ]
  )
  ctx.setVelocitiesToTemperature(temperature_k * omm_unit.kelvin, 991)

  temps: list[float] = []
  n_steps = 4000
  burn = 1500
  for _ in range(n_steps):
    integrator.step(1)
    st = ctx.getState(getEnergy=True)
    ke = st.getKineticEnergy().value_in_unit(omm_unit.kilocalories_per_mole)
    temps.append(2.0 * ke / (dof * BOLTZMANN_KCAL))

  mean_t = float(np.mean(temps[burn:]))
  assert 220.0 <= mean_t <= 380.0, f"OpenMM thermometer mean T unexpected: {mean_t:.2f} K"
  assert math.isfinite(mean_t)


@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_create_system_remove_cmmotion_matches_cmmotion_remover_force(tmp_path) -> None:
  """`ForceField.createSystem(removeCMMotion=...)` must match presence of CMMotionRemover (benchmark parity guard)."""
  from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _write_wat_pdb

  n_waters = 2
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  n_atoms = n_waters * 3

  pdb_path = tmp_path / "watbox_cm.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")

  for remove_cm in (True, False):
    omm_system = ff.createSystem(
      pdb.topology,
      nonbondedMethod=PME,
      nonbondedCutoff=0.9 * omm_unit.nanometer,
      constraints=HBonds,
      rigidWater=True,
      removeCMMotion=remove_cm,
    )
    has_cm = any(isinstance(omm_system.getForce(i), openmm.CMMotionRemover) for i in range(omm_system.getNumForces()))
    assert has_cm is remove_cm, f"removeCMMotion={remove_cm} but CMMotionRemover present={has_cm}"
