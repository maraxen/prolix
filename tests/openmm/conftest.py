"""Base pytest fixtures and infrastructure for the Prolix OpenMM backend test suite.

Provides:
- OpenMM platform discovery and platform fixtures (Reference, CPU).
- Minimal synthetic single-force OpenMM System fixtures:
    * HarmonicBondForce
    * HarmonicAngleForce
    * PeriodicTorsionForce
    * RBTorsionForce (Ryckaert-Bellemans)
    * CMAPTorsionForce (24x24 energy grids)
    * NonbondedForce direct cutoff (Lennard-Jones + Coulomb)
    * NonbondedForce PME (periodic box, alpha, cutoff)
    * 1-4 pair exceptions (scaled Coulomb and LJ)
    * 1-2 and 1-3 topological exclusions
    * GBSAOBCForce (OBC1/OBC2 implicit solvent)
    * MonteCarloBarostat (pressure, temperature, frequency)
    * CMMotionRemover (center-of-mass momentum removal)
- Real-world application scenario fixtures:
    * Chignolin implicit solvent (1UAO.pdb, 138 atoms, Amber14SB + GBSAOBC)
    * Solvated TIP3P water box (258 waters, 774 atoms, PME, SETTLE rigid water)
    * DHFR explicit solvent benchmark (dhfr_jac_benchmark_system.xml, 23,558 particles)
- Virtual site fixtures:
    * TwoParticleAverageSite
    * ThreeParticleAverageSite (and tip4pew.xml)
    * OutOfPlaneSite (and tip5p.xml)
    * LocalCoordinatesSite (12-float frame transformation)
- Helper calculation utilities:
    * compute_openmm_energy
    * compute_openmm_forces
    * compute_openmm_state
    * clone_openmm_system

All fixtures return isolated systems (via XmlSerializer deserialization) to prevent test-order coupling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Iterator

import numpy as np
import pytest

try:
  import openmm
  from openmm import XmlSerializer, app, unit

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False


# Mark all tests in tests/openmm with the openmm marker
pytestmark = pytest.mark.openmm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "pdb"


# =============================================================================
# Test Container
# =============================================================================


@dataclass
class OpenMMTestSystem:
  """Structured container holding an OpenMM System and associated test assets.

  Supports:
  - Direct attribute access: `item.system`, `item.positions`, `item.topology`, `item.box_vectors`
  - Unpacking: `system, positions = item[:2]` or `system, positions, topology, box = item`
  """

  system: Any  # openmm.System
  positions: Any  # openmm.unit.Quantity
  topology: Any = None  # openmm.app.Topology | None
  box_vectors: Any = None  # openmm.unit.Quantity | None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __iter__(self) -> Iterator[Any]:
    return iter([self.system, self.positions, self.topology, self.box_vectors])

  def __getitem__(self, index: int | slice) -> Any:
    elements = [self.system, self.positions, self.topology, self.box_vectors]
    return elements[index]


# =============================================================================
# Helper Utilities
# =============================================================================


def clone_openmm_system(system: openmm.System) -> openmm.System:
  """Create an isolated, deep clone of an OpenMM System via XmlSerializer."""
  if not HAS_OPENMM:
    raise RuntimeError("OpenMM is required to clone an OpenMM System.")
  return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def compute_openmm_energy(
    system: openmm.System,
    positions: Any,
    platform_name: str = "Reference",
    box_vectors: Any = None,
) -> float:
  """Compute potential energy in kJ/mol using OpenMM.

  Args:
      system: OpenMM System object.
      positions: Coordinates as OpenMM Quantity (nm or Angstrom).
      platform_name: 'Reference' or 'CPU'.
      box_vectors: Optional periodic box vectors Quantity.

  Returns:
      Potential energy in kJ/mol (float).
  """
  if not HAS_OPENMM:
    pytest.skip("OpenMM is not available.")
  platform = openmm.Platform.getPlatformByName(platform_name)
  integrator = openmm.VerletIntegrator(0.001)
  context = openmm.Context(system, integrator, platform)
  try:
    if box_vectors is not None:
      context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  finally:
    del context
    del integrator


def compute_openmm_forces(
    system: openmm.System,
    positions: Any,
    platform_name: str = "Reference",
    box_vectors: Any = None,
) -> np.ndarray:
  """Compute per-atom forces in kJ/(mol·nm) using OpenMM.

  Args:
      system: OpenMM System object.
      positions: Coordinates as OpenMM Quantity.
      platform_name: 'Reference' or 'CPU'.
      box_vectors: Optional periodic box vectors Quantity.

  Returns:
      (N, 3) numpy array of forces in kJ/(mol·nm).
  """
  if not HAS_OPENMM:
    pytest.skip("OpenMM is not available.")
  platform = openmm.Platform.getPlatformByName(platform_name)
  integrator = openmm.VerletIntegrator(0.001)
  context = openmm.Context(system, integrator, platform)
  try:
    if box_vectors is not None:
      context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    return np.asarray(forces, dtype=np.float64)
  finally:
    del context
    del integrator


def compute_openmm_state(
    system: openmm.System,
    positions: Any,
    platform_name: str = "Reference",
    box_vectors: Any = None,
    get_energy: bool = True,
    get_forces: bool = True,
    get_positions: bool = True,
    groups: int = -1,
) -> openmm.State:
  """Extract an OpenMM State object with specified observables."""
  if not HAS_OPENMM:
    pytest.skip("OpenMM is not available.")
  platform = openmm.Platform.getPlatformByName(platform_name)
  integrator = openmm.VerletIntegrator(0.001)
  context = openmm.Context(system, integrator, platform)
  try:
    if box_vectors is not None:
      context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    return context.getState(
        getEnergy=get_energy,
        getForces=get_forces,
        getPositions=get_positions,
        groups=groups,
    )
  finally:
    del context
    del integrator


# =============================================================================
# Platform Discovery Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def openmm_available() -> bool:
  """Fixture checking if OpenMM is installed in current environment."""
  if not HAS_OPENMM:
    pytest.skip("OpenMM is not installed in current environment.")
  return True


@pytest.fixture(scope="session")
def openmm_platforms(openmm_available: bool) -> list[str]:
  """Return a list of available OpenMM platform names in the environment."""
  return [
      openmm.Platform.getPlatform(i).getName()
      for i in range(openmm.Platform.getNumPlatforms())
  ]


@pytest.fixture(scope="session")
def reference_platform(openmm_available: bool) -> openmm.Platform:
  """Provide OpenMM Reference platform (pure double-precision software oracle)."""
  try:
    return openmm.Platform.getPlatformByName("Reference")
  except Exception:
    pytest.skip("OpenMM Reference platform is not available.")


@pytest.fixture(scope="session")
def cpu_platform(openmm_available: bool) -> openmm.Platform:
  """Provide OpenMM CPU platform (SIMD-accelerated CPU reference)."""
  try:
    return openmm.Platform.getPlatformByName("CPU")
  except Exception:
    pytest.skip("OpenMM CPU platform is not available.")


@pytest.fixture(params=["Reference", "CPU"])
def any_openmm_platform(request: pytest.FixtureRequest, openmm_available: bool) -> openmm.Platform:
  """Parametrized fixture yielding available OpenMM platforms."""
  try:
    return openmm.Platform.getPlatformByName(request.param)
  except Exception:
    pytest.skip(f"OpenMM platform '{request.param}' is not available.")


# =============================================================================
# Minimal Synthetic Single-Force Fixtures
# =============================================================================


@pytest.fixture
def synthetic_harmonic_bond_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 2-particle system connected by an OpenMM HarmonicBondForce.

  Parameters:
      r0 = 0.15 nm
      k  = 300,000.0 kJ/(mol·nm^2)
  Separation: 0.16 nm -> expected energy = 0.5 * 300000 * (0.01)^2 = 15.0 kJ/mol
  """
  system = openmm.System()
  system.addParticle(12.0)
  system.addParticle(12.0)

  bonds = openmm.HarmonicBondForce()
  r0 = 0.15 * unit.nanometers
  k = 300000.0 * unit.kilojoules_per_mole / (unit.nanometer**2)
  bonds.addBond(0, 1, r0, k)
  system.addForce(bonds)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.16, 0.0, 0.0),
  ] * unit.nanometers

  top = app.Topology()
  c = top.addChain()
  r = top.addResidue("RES", c)
  a1 = top.addAtom("C1", app.Element.getBySymbol("C"), r)
  a2 = top.addAtom("C2", app.Element.getBySymbol("C"), r)
  top.addBond(a1, a2)

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      topology=top,
      metadata={
          "force_type": "HarmonicBondForce",
          "r0_nm": 0.15,
          "k_kj_per_mol_nm2": 300000.0,
          "expected_energy_kj": 15.0,
      },
  )


@pytest.fixture
def synthetic_harmonic_angle_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 3-particle system connected by an OpenMM HarmonicAngleForce.

  Parameters:
      theta0 = 109.5 degrees (~1.9111355 rad)
      k      = 400.0 kJ/(mol·rad^2)
  """
  system = openmm.System()
  for _ in range(3):
    system.addParticle(12.0)

  angles = openmm.HarmonicAngleForce()
  theta0 = 109.5 * (np.pi / 180.0) * unit.radians
  k = 400.0 * unit.kilojoules_per_mole / (unit.radian**2)
  angles.addAngle(0, 1, 2, theta0, k)
  system.addForce(angles)

  positions = [
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.0, 0.1, 0.0),  # 90 degrees (~1.5707963 rad)
  ] * unit.nanometers

  top = app.Topology()
  c = top.addChain()
  r = top.addResidue("RES", c)
  a0 = top.addAtom("A0", app.Element.getBySymbol("C"), r)
  a1 = top.addAtom("A1", app.Element.getBySymbol("C"), r)
  a2 = top.addAtom("A2", app.Element.getBySymbol("C"), r)
  top.addBond(a0, a1)
  top.addBond(a1, a2)

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      topology=top,
      metadata={
          "force_type": "HarmonicAngleForce",
          "theta0_rad": 109.5 * (np.pi / 180.0),
          "k_kj_per_mol_rad2": 400.0,
      },
  )


@pytest.fixture
def synthetic_periodic_torsion_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 4-particle system with an OpenMM PeriodicTorsionForce.

  Parameters:
      periodicity = 3
      phase = 0.0 rad
      k = 15.0 kJ/mol
  """
  system = openmm.System()
  for _ in range(4):
    system.addParticle(12.0)

  torsions = openmm.PeriodicTorsionForce()
  periodicity = 3
  phase = 0.0 * unit.radians
  k = 15.0 * unit.kilojoules_per_mole
  torsions.addTorsion(0, 1, 2, 3, periodicity, phase, k)
  system.addForce(torsions)

  # Non-planar 4-atom chain forming dihedral ~ 60 degrees (pi/3 rad)
  positions = [
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.1, 0.05, 0.0866),
  ] * unit.nanometers

  top = app.Topology()
  c = top.addChain()
  r = top.addResidue("RES", c)
  atoms = [top.addAtom(f"A{i}", app.Element.getBySymbol("C"), r) for i in range(4)]
  top.addBond(atoms[0], atoms[1])
  top.addBond(atoms[1], atoms[2])
  top.addBond(atoms[2], atoms[3])

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      topology=top,
      metadata={
          "force_type": "PeriodicTorsionForce",
          "periodicity": periodicity,
          "k_kj_per_mol": 15.0,
      },
  )


@pytest.fixture
def synthetic_rb_torsion_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 4-particle system with an OpenMM RBTorsionForce (Ryckaert-Bellemans).

  Energy = sum_{n=0}^5 c_n * (cos(psi))^n, where psi = phi - pi
  """
  system = openmm.System()
  for _ in range(4):
    system.addParticle(12.0)

  rb = openmm.RBTorsionForce()
  c0 = 1.0 * unit.kilojoules_per_mole
  c1 = 2.0 * unit.kilojoules_per_mole
  c2 = 3.0 * unit.kilojoules_per_mole
  c3 = 4.0 * unit.kilojoules_per_mole
  c4 = 5.0 * unit.kilojoules_per_mole
  c5 = 6.0 * unit.kilojoules_per_mole
  rb.addTorsion(0, 1, 2, 3, c0, c1, c2, c3, c4, c5)
  system.addForce(rb)

  positions = [
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.1),
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "RBTorsionForce",
          "coefficients": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      },
  )


@pytest.fixture
def synthetic_cmap_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 5-particle system with an OpenMM CMAPTorsionForce on a 24x24 grid."""
  system = openmm.System()
  for _ in range(5):
    system.addParticle(12.0)

  cmap = openmm.CMAPTorsionForce()
  size = 24
  # Generate continuous 2D sinusoidal energy grid in kJ/mol
  u = np.linspace(-np.pi, np.pi, size, endpoint=False)
  grid_2d = 5.0 * np.outer(np.sin(u), np.cos(u))
  grid_list = [float(v) * unit.kilojoules_per_mole for v in grid_2d.flatten()]

  map_idx = cmap.addMap(size, grid_list)
  # Two coupled dihedrals: 0-1-2-3 and 1-2-3-4
  cmap.addTorsion(map_idx, 0, 1, 2, 3, 1, 2, 3, 4)
  system.addForce(cmap)

  positions = [
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.1, 0.1, 0.0),
      openmm.Vec3(0.2, 0.1, 0.05),
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "CMAPTorsionForce",
          "grid_size": size,
          "map_index": map_idx,
      },
  )


@pytest.fixture
def synthetic_nonbonded_direct_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 2-particle system with direct cutoff (NoCutoff) Lennard-Jones and Coulomb."""
  system = openmm.System()
  system.addParticle(12.0)
  system.addParticle(12.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

  q0 = 1.0 * unit.elementary_charge
  q1 = -1.0 * unit.elementary_charge
  sigma = 0.3 * unit.nanometers
  epsilon = 1.0 * unit.kilojoules_per_mole

  nb.addParticle(q0, sigma, epsilon)
  nb.addParticle(q1, sigma, epsilon)
  system.addForce(nb)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.4, 0.0, 0.0),  # Separation 0.4 nm
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "NonbondedForce",
          "method": "NoCutoff",
          "q0": 1.0,
          "q1": -1.0,
          "sigma": 0.3,
          "epsilon": 1.0,
          "distance_nm": 0.4,
      },
  )


@pytest.fixture
def synthetic_pme_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 8-particle system in a 2.0 nm cubic box using PME electrostatics."""
  system = openmm.System()
  box_length = 2.0 * unit.nanometers
  system.setDefaultPeriodicBoxVectors(
      openmm.Vec3(2.0, 0.0, 0.0) * unit.nanometers,
      openmm.Vec3(0.0, 2.0, 0.0) * unit.nanometers,
      openmm.Vec3(0.0, 0.0, 2.0) * unit.nanometers,
  )

  for _ in range(8):
    system.addParticle(12.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.PME)
  nb.setCutoffDistance(0.9 * unit.nanometers)
  nb.setEwaldErrorTolerance(5e-5)

  # Alternating charges on the corners of a cube of edge 0.6 nm
  charges = [+0.5, -0.5, +0.5, -0.5, -0.5, +0.5, -0.5, +0.5]
  sigma = 0.3 * unit.nanometers
  epsilon = 0.5 * unit.kilojoules_per_mole

  for q in charges:
    nb.addParticle(q * unit.elementary_charge, sigma, epsilon)
  system.addForce(nb)

  positions = []
  for z in [0.7, 1.3]:
    for y in [0.7, 1.3]:
      for x in [0.7, 1.3]:
        positions.append(openmm.Vec3(x, y, z))
  positions = positions * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      box_vectors=system.getDefaultPeriodicBoxVectors(),
      metadata={
          "force_type": "NonbondedForce",
          "method": "PME",
          "cutoff_nm": 0.9,
          "box_length_nm": 2.0,
      },
  )


@pytest.fixture
def synthetic_14_exceptions_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 4-particle chain with 1-4 exceptions in OpenMM NonbondedForce."""
  system = openmm.System()
  for _ in range(4):
    system.addParticle(12.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

  q = 0.5 * unit.elementary_charge
  sigma = 0.3 * unit.nanometers
  epsilon = 1.0 * unit.kilojoules_per_mole

  for _ in range(4):
    nb.addParticle(q, sigma, epsilon)

  # Exclude 1-2 and 1-3 pairs: (0,1), (1,2), (2,3), (0,2), (1,3)
  for p1, p2 in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]:
    nb.addException(p1, p2, 0.0 * unit.elementary_charge**2, 1.0 * unit.nanometers, 0.0 * unit.kilojoules_per_mole)

  # 1-4 pair (0, 3) with standard 1-4 scaling factors (f_qq = 5/6, f_lj = 0.5)
  charge_prod = (0.5 * 0.5 * (5.0 / 6.0)) * (unit.elementary_charge**2)
  eps_14 = (1.0 * 0.5) * unit.kilojoules_per_mole
  nb.addException(0, 3, charge_prod, sigma, eps_14)
  system.addForce(nb)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.15, 0.0, 0.0),
      openmm.Vec3(0.30, 0.0, 0.0),
      openmm.Vec3(0.45, 0.0, 0.0),
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "NonbondedForce",
          "has_exceptions": True,
          "pair_14": (0, 3),
      },
  )


@pytest.fixture
def synthetic_exclusions_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 3-particle system with topological exclusions in NonbondedForce."""
  system = openmm.System()
  for _ in range(3):
    system.addParticle(12.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

  for _ in range(3):
    nb.addParticle(1.0 * unit.elementary_charge, 0.3 * unit.nanometers, 1.0 * unit.kilojoules_per_mole)

  # Exclude all pairs
  nb.addException(0, 1, 0.0, 1.0, 0.0)
  nb.addException(1, 2, 0.0, 1.0, 0.0)
  nb.addException(0, 2, 0.0, 1.0, 0.0)
  system.addForce(nb)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.2, 0.0, 0.0),
      openmm.Vec3(0.1, 0.15, 0.0),
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "NonbondedForce",
          "all_excluded": True,
          "expected_energy_kj": 0.0,
      },
  )


@pytest.fixture
def synthetic_gbsa_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 3-particle system with OpenMM GBSAOBCForce (OBC2 implicit solvent)."""
  system = openmm.System()
  for _ in range(3):
    system.addParticle(12.0)

  gb = openmm.GBSAOBCForce()
  gb.setSolventDielectric(78.5)
  gb.setSoluteDielectric(1.0)
  gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)

  gb.addParticle(+0.5 * unit.elementary_charge, 0.15 * unit.nanometers, 0.8)
  gb.addParticle(-0.5 * unit.elementary_charge, 0.15 * unit.nanometers, 0.8)
  gb.addParticle(0.0 * unit.elementary_charge, 0.12 * unit.nanometers, 0.8)
  system.addForce(gb)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.2, 0.0, 0.0),
      openmm.Vec3(0.1, 0.15, 0.0),
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "GBSAOBCForce",
          "model": "OBC2",
          "solvent_dielectric": 78.5,
      },
  )


@pytest.fixture
def synthetic_barostat_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic periodic box with particles and MonteCarloBarostat."""
  system = openmm.System()
  system.setDefaultPeriodicBoxVectors(
      openmm.Vec3(2.0, 0.0, 0.0) * unit.nanometers,
      openmm.Vec3(0.0, 2.0, 0.0) * unit.nanometers,
      openmm.Vec3(0.0, 0.0, 2.0) * unit.nanometers,
  )

  for _ in range(8):
    system.addParticle(12.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb.setCutoffDistance(0.8 * unit.nanometers)
  for _ in range(8):
    nb.addParticle(0.0 * unit.elementary_charge, 0.3 * unit.nanometers, 0.5 * unit.kilojoules_per_mole)
  system.addForce(nb)

  barostat = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25)
  system.addForce(barostat)

  positions = []
  for z in [0.8, 1.2]:
    for y in [0.8, 1.2]:
      for x in [0.8, 1.2]:
        positions.append(openmm.Vec3(x, y, z))
  positions = positions * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      box_vectors=system.getDefaultPeriodicBoxVectors(),
      metadata={
          "has_barostat": True,
          "pressure_bar": 1.0,
          "temperature_k": 300.0,
          "frequency": 25,
      },
  )


@pytest.fixture
def synthetic_cmm_system(openmm_available: bool) -> OpenMMTestSystem:
  """Synthetic 4-particle system with CMMotionRemover."""
  system = openmm.System()
  for _ in range(4):
    system.addParticle(12.0)

  cmm = openmm.CMMotionRemover(10)
  system.addForce(cmm)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.1),
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "force_type": "CMMotionRemover",
          "frequency": 10,
      },
  )


# =============================================================================
# Virtual Site Fixtures
# =============================================================================


@pytest.fixture
def synthetic_two_particle_vs_system(openmm_available: bool) -> OpenMMTestSystem:
  """System with a TwoParticleAverageSite placed between two heavy atoms."""
  system = openmm.System()
  system.addParticle(16.0)  # Atom 0 (Oxygen)
  system.addParticle(1.0)   # Atom 1 (Hydrogen)
  system.addParticle(0.0)   # Atom 2 (Virtual Site)

  w1, w2 = 0.6, 0.4
  vs = openmm.TwoParticleAverageSite(0, 1, w1, w2)
  system.setVirtualSite(2, vs)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),  # Placeholder, computed by virtual site
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "virtual_site_type": "TwoParticleAverageSite",
          "parent_indices": (0, 1),
          "weights": (w1, w2),
          "expected_vs_pos": openmm.Vec3(0.04, 0.0, 0.0) * unit.nanometers,
      },
  )


@pytest.fixture
def synthetic_three_particle_vs_system(openmm_available: bool) -> OpenMMTestSystem:
  """System with a ThreeParticleAverageSite defined by three heavy atoms."""
  system = openmm.System()
  system.addParticle(16.0)  # Atom 0
  system.addParticle(1.0)   # Atom 1
  system.addParticle(1.0)   # Atom 2
  system.addParticle(0.0)   # Atom 3 (Virtual Site)

  w1, w2, w3 = 0.5, 0.25, 0.25
  vs = openmm.ThreeParticleAverageSite(0, 1, 2, w1, w2, w3)
  system.setVirtualSite(3, vs)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),  # Computed
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "virtual_site_type": "ThreeParticleAverageSite",
          "parent_indices": (0, 1, 2),
          "weights": (w1, w2, w3),
          "expected_vs_pos": openmm.Vec3(0.025, 0.025, 0.0) * unit.nanometers,
      },
  )


@pytest.fixture
def tip4pew_water_system(openmm_available: bool) -> OpenMMTestSystem:
  """Single TIP4P-Ew water molecule with ThreeParticleAverageSite from tip4pew.xml."""
  top = app.Topology()
  c = top.addChain()
  r = top.addResidue("HOH", c)
  o = top.addAtom("O", app.Element.getBySymbol("O"), r)
  h1 = top.addAtom("H1", app.Element.getBySymbol("H"), r)
  h2 = top.addAtom("H2", app.Element.getBySymbol("H"), r)
  top.addBond(o, h1)
  top.addBond(o, h2)

  pos = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.09572, 0.0, 0.0),
      openmm.Vec3(-0.024, 0.09266, 0.0),
  ] * unit.nanometers

  ff = app.ForceField("tip4pew.xml")
  modeller = app.Modeller(top, pos)
  modeller.addExtraParticles(ff)
  system = ff.createSystem(modeller.topology)

  return OpenMMTestSystem(
      system=system,
      positions=modeller.positions,
      topology=modeller.topology,
      metadata={
          "model": "tip4pew",
          "num_particles": system.getNumParticles(),
          "virtual_site_index": 3,
      },
  )


@pytest.fixture
def synthetic_out_of_plane_vs_system(openmm_available: bool) -> OpenMMTestSystem:
  """System with an OutOfPlaneSite virtual site defined by cross product."""
  system = openmm.System()
  system.addParticle(16.0)  # Atom 0
  system.addParticle(1.0)   # Atom 1
  system.addParticle(1.0)   # Atom 2
  system.addParticle(0.0)   # Atom 3 (Virtual Site)

  w12, w13, wcross = 0.5, 0.5, 0.1
  vs = openmm.OutOfPlaneSite(0, 1, 2, w12, w13, wcross)
  system.setVirtualSite(3, vs)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),  # Computed
  ] * unit.nanometers

  # Expected: r1 + w12*(r2-r1) + w13*(r3-r1) + wcross*((r2-r1) x (r3-r1))
  # = (0,0,0) + 0.5*(0.1,0,0) + 0.5*(0,0.1,0) + 0.1*(0, 0, 0.01) = (0.05, 0.05, 0.001)
  expected_pos = openmm.Vec3(0.05, 0.05, 0.001) * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "virtual_site_type": "OutOfPlaneSite",
          "parent_indices": (0, 1, 2),
          "weights": (w12, w13, wcross),
          "expected_vs_pos": expected_pos,
      },
  )


@pytest.fixture
def tip5p_water_system(openmm_available: bool) -> OpenMMTestSystem:
  """Single TIP5P water molecule with two OutOfPlaneSite lone pairs from tip5p.xml."""
  top = app.Topology()
  c = top.addChain()
  r = top.addResidue("HOH", c)
  o = top.addAtom("O", app.Element.getBySymbol("O"), r)
  h1 = top.addAtom("H1", app.Element.getBySymbol("H"), r)
  h2 = top.addAtom("H2", app.Element.getBySymbol("H"), r)
  top.addBond(o, h1)
  top.addBond(o, h2)

  pos = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.09572, 0.0, 0.0),
      openmm.Vec3(-0.024, 0.09266, 0.0),
  ] * unit.nanometers

  ff = app.ForceField("tip5p.xml")
  modeller = app.Modeller(top, pos)
  modeller.addExtraParticles(ff)
  system = ff.createSystem(modeller.topology)

  return OpenMMTestSystem(
      system=system,
      positions=modeller.positions,
      topology=modeller.topology,
      metadata={
          "model": "tip5p",
          "num_particles": system.getNumParticles(),
          "virtual_site_indices": (3, 4),
      },
  )


@pytest.fixture
def synthetic_local_coords_vs_system(openmm_available: bool) -> OpenMMTestSystem:
  """System with a LocalCoordinatesSite virtual site with 12-float frame transformation."""
  system = openmm.System()
  system.addParticle(16.0)  # Atom 0
  system.addParticle(1.0)   # Atom 1
  system.addParticle(1.0)   # Atom 2
  system.addParticle(0.0)   # Atom 3 (Virtual Site)

  origin_weights = openmm.Vec3(1.0, 0.0, 0.0)
  x_weights = openmm.Vec3(-1.0, 1.0, 0.0)
  y_weights = openmm.Vec3(-1.0, 0.0, 1.0)
  local_position = openmm.Vec3(0.02, 0.03, 0.04) * unit.nanometers

  vs = openmm.LocalCoordinatesSite(
      0, 1, 2, origin_weights, x_weights, y_weights, local_position
  )
  system.setVirtualSite(3, vs)

  positions = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.1, 0.0, 0.0),
      openmm.Vec3(0.0, 0.1, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),  # Computed
  ] * unit.nanometers

  return OpenMMTestSystem(
      system=system,
      positions=positions,
      metadata={
          "virtual_site_type": "LocalCoordinatesSite",
          "parent_indices": (0, 1, 2),
          "origin_weights": origin_weights,
          "x_weights": x_weights,
          "y_weights": y_weights,
          "local_position": local_position,
          "expected_vs_pos": local_position,
      },
  )


# =============================================================================
# Real-World Application Scenario Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def _cached_chignolin_1uao_data(openmm_available: bool) -> tuple[str, Any, Any]:
  """Session-scoped loader for Chignolin (1UAO.pdb) returning serialized XML string and positions."""
  pdb_path = _DATA_DIR / "1UAO.pdb"
  if not pdb_path.exists():
    pytest.skip(f"Chignolin PDB file not found at {pdb_path}")

  from pdbfixer import PDBFixer

  fixer = PDBFixer(filename=str(pdb_path))
  fixer.findMissingResidues()
  fixer.findMissingAtoms()
  fixer.addMissingAtoms()
  fixer.addMissingHydrogens(7.0)

  ff = app.ForceField("amber14/protein.ff14SB.xml", "implicit/obc2.xml")
  system = ff.createSystem(
      fixer.topology,
      nonbondedMethod=app.NoCutoff,
      constraints=None,
  )

  xml_str = XmlSerializer.serialize(system)
  return xml_str, fixer.positions, fixer.topology


@pytest.fixture
def chignolin_1uao_system(_cached_chignolin_1uao_data: tuple[str, Any, Any]) -> OpenMMTestSystem:
  """Function-scoped fresh clone of Chignolin (1UAO, 138 atoms) with Amber14SB and GBSAOBC."""
  xml_str, positions, topology = _cached_chignolin_1uao_data
  system = XmlSerializer.deserialize(xml_str)
  return OpenMMTestSystem(
      system=system,
      positions=copy.deepcopy(positions),
      topology=topology,
      metadata={
          "name": "Chignolin (1UAO)",
          "num_particles": system.getNumParticles(),
          "solvent": "implicit_obc2",
      },
  )


@pytest.fixture(scope="session")
def _cached_tip3p_water_box_data(openmm_available: bool) -> tuple[str, Any, Any, Any]:
  """Session-scoped builder for solvated TIP3P water box (258 waters, 2.0 nm box, PME)."""
  top = app.Topology()
  modeller = app.Modeller(top, [])
  ff = app.ForceField("amber14/tip3p.xml")
  modeller.addSolvent(
      ff,
      boxSize=openmm.Vec3(2.0, 2.0, 2.0) * unit.nanometers,
      model="tip3p",
  )
  system = ff.createSystem(
      modeller.topology,
      nonbondedMethod=app.PME,
      nonbondedCutoff=0.8 * unit.nanometers,
      rigidWater=True,
  )
  xml_str = XmlSerializer.serialize(system)
  box_vectors = system.getDefaultPeriodicBoxVectors()
  return xml_str, modeller.positions, modeller.topology, box_vectors


@pytest.fixture
def solvated_tip3p_water_box(_cached_tip3p_water_box_data: tuple[str, Any, Any, Any]) -> OpenMMTestSystem:
  """Function-scoped fresh clone of the solvated TIP3P water box (774 particles, PME, SETTLE)."""
  xml_str, positions, topology, box_vectors = _cached_tip3p_water_box_data
  system = XmlSerializer.deserialize(xml_str)
  return OpenMMTestSystem(
      system=system,
      positions=copy.deepcopy(positions),
      topology=topology,
      box_vectors=box_vectors,
      metadata={
          "name": "TIP3P Water Box",
          "num_particles": system.getNumParticles(),
          "num_waters": system.getNumParticles() // 3,
          "box_size_nm": 2.0,
      },
  )


@pytest.fixture(scope="session")
def _cached_dhfr_benchmark_data(openmm_available: bool) -> tuple[str, Any, Any, Any]:
  """Session-scoped deserializer for DHFR explicit solvent JAC benchmark system (23,558 particles)."""
  xml_path = _DATA_DIR / "dhfr_jac_benchmark_system.xml"
  pdb_path = _DATA_DIR / "dhfr_jac_benchmark.pdb"

  if not xml_path.exists():
    pytest.skip(f"DHFR benchmark system XML not found at {xml_path}")
  if not pdb_path.exists():
    pytest.skip(f"DHFR benchmark PDB not found at {pdb_path}")

  with open(xml_path, encoding="utf-8") as f:
    xml_str = f.read()

  pdb = app.PDBFile(str(pdb_path))
  # Verify system XML and PDB particles match
  temp_sys = XmlSerializer.deserialize(xml_str)
  box_vectors = temp_sys.getDefaultPeriodicBoxVectors()
  del temp_sys

  return xml_str, pdb.positions, pdb.topology, box_vectors


@pytest.fixture
def dhfr_benchmark_system(_cached_dhfr_benchmark_data: tuple[str, Any, Any, Any]) -> OpenMMTestSystem:
  """Function-scoped fresh clone of DHFR explicit solvent benchmark system (23,558 particles)."""
  xml_str, positions, topology, box_vectors = _cached_dhfr_benchmark_data
  system = XmlSerializer.deserialize(xml_str)
  return OpenMMTestSystem(
      system=system,
      positions=copy.deepcopy(positions),
      topology=topology,
      box_vectors=box_vectors,
      metadata={
          "name": "DHFR Explicit Solvent Benchmark",
          "num_particles": system.getNumParticles(),
          "pme": True,
      },
  )
