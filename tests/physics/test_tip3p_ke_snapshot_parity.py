"""Snapshot oracle: OpenMM ``getKineticEnergy`` vs Prolix rigid-water KE on the same velocities.

If this fails, rigid ``mean_T_K`` comparisons against OpenMM are not trustworthy until the
velocity→momentum conversion (or ``rigid_tip3p_box_ke_kcal``) is fixed. When it passes, a hot
rigid thermometer in production runs points at **dynamics** (integrator / SETTLE / thermostat),
not at the post-hoc rigid KE functional.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS

from .test_explicit_langevin_tip3p_parity import _grid_water_positions, _write_wat_pdb

try:
  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False


def _openmm_velocity_nmps_to_prolix_momentum(
  velocities_nmps: np.ndarray,
  mass_dalton_per_atom: np.ndarray,
) -> np.ndarray:
  """Map OpenMM Cartesian velocities (nm/ps) to Prolix momenta used by ``settle_langevin``.

  Prolix uses ``velocity = momentum / mass`` with positions in Å and the AKMA time unit
  (see ``AKMA_TIME_UNIT_FS``): one AKMA time step spans ``AKMA_TIME_UNIT_FS`` femtoseconds, so
  ``Å / (AKMA time) = (Å/fs) * (fs per AKMA) = v_A_per_fs * AKMA_TIME_UNIT_FS``.
  """
  v = np.asarray(velocities_nmps, dtype=np.float64)
  m = np.asarray(mass_dalton_per_atom, dtype=np.float64).reshape(-1, 1)
  v_angstrom_per_akma = v * 0.01 * float(AKMA_TIME_UNIT_FS)
  return m * v_angstrom_per_akma


@pytest.mark.openmm
@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
def test_openmm_snapshot_rigid_ke_matches_prolix_rigid_functional(tmp_path: Path) -> None:
  jax.config.update("jax_enable_x64", True)

  n_waters = 4
  n_atoms = n_waters * 3
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)

  pdb_path = tmp_path / "watbox_ke_snapshot.pdb"
  _write_wat_pdb(pdb_path, positions_a, box_edge)
  pdb = PDBFile(str(pdb_path))
  ff = ForceField("amber14/tip3p.xml")
  omm_system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=0.9 * omm_unit.nanometer,
    constraints=HBonds,
    rigidWater=True,
    removeCMMotion=False,
  )
  for fi in range(omm_system.getNumForces()):
    f = omm_system.getForce(fi)
    if isinstance(f, openmm.NonbondedForce):
      f.setPMEParameters(3.4, 32, 32, 32)

  integrator = openmm.LangevinMiddleIntegrator(
    300.0 * omm_unit.kelvin,
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
  ctx.setVelocitiesToTemperature(300.0 * omm_unit.kelvin, 4242)

  st = ctx.getState(getEnergy=True, getVelocities=True)
  ke_omm = float(st.getKineticEnergy().value_in_unit(omm_unit.kilocalories_per_mole))
  vels = st.getVelocities()
  v_unit = omm_unit.nanometer / omm_unit.picosecond
  v_np = np.zeros((n_atoms, 3), dtype=np.float64)
  for i in range(n_atoms):
    vx, vy, vz = vels[i].x, vels[i].y, vels[i].z

    def _scalar_vel(q: object) -> float:
      if hasattr(q, "value_in_unit"):
        return float(q.value_in_unit(v_unit))
      return float(q)

    v_np[i, 0] = _scalar_vel(vx)
    v_np[i, 1] = _scalar_vel(vy)
    v_np[i, 2] = _scalar_vel(vz)

  mass_np = np.array(
    [float(omm_system.getParticleMass(i).value_in_unit(omm_unit.dalton)) for i in range(n_atoms)],
    dtype=np.float64,
  )
  p_np = _openmm_velocity_nmps_to_prolix_momentum(v_np, mass_np)
  mass_j = jnp.array([[mass_np[i]] for i in range(n_atoms)], dtype=jnp.float64)
  ke_r = float(rigid_tip3p_box_ke_kcal(jnp.asarray(positions_a), jnp.asarray(p_np), mass_j, n_waters))

  rel = abs(ke_omm - ke_r) / max(abs(ke_omm), 1e-12)
  assert rel < 1e-6, f"rigid KE mismatch: openmm={ke_omm} prolix_rigid_fn={ke_r} rel_err={rel}"
