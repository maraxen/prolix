"""Verification tests for OpenMM base fixtures and infrastructure in tests/openmm/conftest.py."""

import numpy as np
import pytest

try:
  import openmm
  from openmm import unit
  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False

from tests.openmm.conftest import (
    compute_openmm_energy,
    compute_openmm_forces,
)

pytestmark = pytest.mark.openmm


def test_platforms_available(openmm_platforms):
  """Verify that OpenMM platform discovery finds Reference and CPU platforms."""
  assert "Reference" in openmm_platforms
  assert "CPU" in openmm_platforms


def test_reference_platform_fixture(reference_platform):
  """Verify Reference platform fixture."""
  assert reference_platform.getName() == "Reference"


def test_cpu_platform_fixture(cpu_platform):
  """Verify CPU platform fixture."""
  assert cpu_platform.getName() == "CPU"


def test_synthetic_harmonic_bond_system(synthetic_harmonic_bond_system):
  """Verify HarmonicBondForce fixture energy and forces."""
  sys, pos, top, box = synthetic_harmonic_bond_system
  assert sys.getNumParticles() == 2
  assert sys.getNumForces() == 1
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isclose(energy, 15.0, atol=1e-5)
  forces = compute_openmm_forces(sys, pos, "Reference")
  assert forces.shape == (2, 3)
  assert np.isclose(forces[0, 0] + forces[1, 0], 0.0, atol=1e-5)


def test_synthetic_harmonic_angle_system(synthetic_harmonic_angle_system):
  """Verify HarmonicAngleForce fixture energy."""
  sys, pos, top, box = synthetic_harmonic_angle_system
  assert sys.getNumParticles() == 3
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert energy > 0.0
  forces = compute_openmm_forces(sys, pos, "Reference")
  assert forces.shape == (3, 3)


def test_synthetic_periodic_torsion_system(synthetic_periodic_torsion_system):
  """Verify PeriodicTorsionForce fixture energy."""
  sys, pos, top, box = synthetic_periodic_torsion_system
  assert sys.getNumParticles() == 4
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)


def test_synthetic_rb_torsion_system(synthetic_rb_torsion_system):
  """Verify RBTorsionForce fixture energy."""
  sys, pos, top, box = synthetic_rb_torsion_system
  assert sys.getNumParticles() == 4
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)


def test_synthetic_cmap_system(synthetic_cmap_system):
  """Verify CMAPTorsionForce fixture energy."""
  sys, pos, top, box = synthetic_cmap_system
  assert sys.getNumParticles() == 5
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)


def test_synthetic_nonbonded_direct_system(synthetic_nonbonded_direct_system):
  """Verify direct cutoff NonbondedForce fixture energy and forces."""
  sys, pos, top, box = synthetic_nonbonded_direct_system
  assert sys.getNumParticles() == 2
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)
  forces = compute_openmm_forces(sys, pos, "Reference")
  assert forces.shape == (2, 3)
  assert np.isclose(forces[0, 0] + forces[1, 0], 0.0, atol=1e-4)


def test_synthetic_pme_system(synthetic_pme_system):
  """Verify PME NonbondedForce fixture energy in periodic box."""
  sys, pos, top, box = synthetic_pme_system
  assert sys.getNumParticles() == 8
  assert box is not None
  energy = compute_openmm_energy(sys, pos, "Reference", box_vectors=box)
  assert np.isfinite(energy)


def test_synthetic_14_exceptions_system(synthetic_14_exceptions_system):
  """Verify 1-4 exceptions fixture energy."""
  sys, pos, top, box = synthetic_14_exceptions_system
  assert sys.getNumParticles() == 4
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)


def test_synthetic_exclusions_system(synthetic_exclusions_system):
  """Verify that fully excluded system has 0.0 potential energy."""
  sys, pos, top, box = synthetic_exclusions_system
  assert sys.getNumParticles() == 3
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isclose(energy, 0.0, atol=1e-6)


def test_synthetic_gbsa_system(synthetic_gbsa_system):
  """Verify GBSAOBCForce fixture energy."""
  sys, pos, top, box = synthetic_gbsa_system
  assert sys.getNumParticles() == 3
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert energy < 0.0


def test_synthetic_barostat_system(synthetic_barostat_system):
  """Verify MonteCarloBarostat fixture presence."""
  sys, pos, top, box = synthetic_barostat_system
  assert sys.getNumParticles() == 8
  forces = [sys.getForce(i).__class__.__name__ for i in range(sys.getNumForces())]
  assert "MonteCarloBarostat" in forces


def test_synthetic_cmm_system(synthetic_cmm_system):
  """Verify CMMotionRemover fixture presence."""
  sys, pos, top, box = synthetic_cmm_system
  assert sys.getNumParticles() == 4
  forces = [sys.getForce(i).__class__.__name__ for i in range(sys.getNumForces())]
  assert "CMMotionRemover" in forces


def test_two_particle_vs_system(synthetic_two_particle_vs_system, reference_platform):
  """Verify TwoParticleAverageSite coordinates computation."""
  sys, pos, top, box = synthetic_two_particle_vs_system
  ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), reference_platform)
  ctx.setPositions(pos)
  ctx.computeVirtualSites()
  state = ctx.getState(getPositions=True)
  vs_pos = state.getPositions()[2]
  assert np.isclose(vs_pos[0].value_in_unit(unit.nanometer), 0.04, atol=1e-5)


def test_three_particle_vs_system(synthetic_three_particle_vs_system, reference_platform):
  """Verify ThreeParticleAverageSite coordinates computation."""
  sys, pos, top, box = synthetic_three_particle_vs_system
  ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), reference_platform)
  ctx.setPositions(pos)
  ctx.computeVirtualSites()
  state = ctx.getState(getPositions=True)
  vs_pos = state.getPositions()[3]
  assert np.isclose(vs_pos[0].value_in_unit(unit.nanometer), 0.025, atol=1e-5)
  assert np.isclose(vs_pos[1].value_in_unit(unit.nanometer), 0.025, atol=1e-5)


def test_tip4pew_water_system(tip4pew_water_system):
  """Verify TIP4P-Ew water system virtual site."""
  sys, pos, top, box = tip4pew_water_system
  assert sys.getNumParticles() == 4
  assert sys.isVirtualSite(3)
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)


def test_out_of_plane_vs_system(synthetic_out_of_plane_vs_system, reference_platform):
  """Verify OutOfPlaneSite coordinates computation."""
  sys, pos, top, box = synthetic_out_of_plane_vs_system
  ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), reference_platform)
  ctx.setPositions(pos)
  ctx.computeVirtualSites()
  state = ctx.getState(getPositions=True)
  vs_pos = state.getPositions()[3]
  assert np.isclose(vs_pos[0].value_in_unit(unit.nanometer), 0.05, atol=1e-5)
  assert np.isclose(vs_pos[1].value_in_unit(unit.nanometer), 0.05, atol=1e-5)
  assert np.isclose(vs_pos[2].value_in_unit(unit.nanometer), 0.001, atol=1e-5)


def test_tip5p_water_system(tip5p_water_system):
  """Verify TIP5P water system with two lone pair virtual sites."""
  sys, pos, top, box = tip5p_water_system
  assert sys.getNumParticles() == 5
  assert sys.isVirtualSite(3)
  assert sys.isVirtualSite(4)
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isfinite(energy)


def test_local_coords_vs_system(synthetic_local_coords_vs_system, reference_platform):
  """Verify LocalCoordinatesSite coordinates computation."""
  sys, pos, top, box = synthetic_local_coords_vs_system
  ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), reference_platform)
  ctx.setPositions(pos)
  ctx.computeVirtualSites()
  state = ctx.getState(getPositions=True)
  vs_pos = state.getPositions()[3]
  assert np.isclose(vs_pos[0].value_in_unit(unit.nanometer), 0.02, atol=1e-5)
  assert np.isclose(vs_pos[1].value_in_unit(unit.nanometer), 0.03, atol=1e-5)
  assert np.isclose(vs_pos[2].value_in_unit(unit.nanometer), 0.04, atol=1e-5)


def test_chignolin_1uao_system(chignolin_1uao_system):
  """Verify Chignolin 1UAO implicit solvent system."""
  sys, pos, top, box = chignolin_1uao_system
  assert sys.getNumParticles() == 138
  energy = compute_openmm_energy(sys, pos, "Reference")
  assert np.isclose(energy, -1038.62, atol=1.0)


def test_solvated_tip3p_water_box(solvated_tip3p_water_box):
  """Verify solvated TIP3P water box system."""
  sys, pos, top, box = solvated_tip3p_water_box
  assert sys.getNumParticles() == 774
  assert box is not None
  energy = compute_openmm_energy(sys, pos, "Reference", box_vectors=box)
  assert np.isfinite(energy)


def test_dhfr_benchmark_system(dhfr_benchmark_system):
  """Verify DHFR explicit solvent JAC benchmark system (23,558 particles)."""
  sys, pos, top, box = dhfr_benchmark_system
  assert sys.getNumParticles() == 23558
  assert len(pos) == 23558
  assert box is not None
  # Compute energy on CPU platform for speed
  energy = compute_openmm_energy(sys, pos, "CPU", box_vectors=box)
  assert np.isclose(energy, -305866.9, atol=10.0)
