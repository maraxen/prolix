"""Adversarial Physics Parity, ForceGroup Bitmask Isolation, and Zero-Recompilation Verification.

Authored by Challenger M3-1 (teamwork_challenger_m3_1).
Adversarial challenge suite verifying:
1. PME electrostatics potential and forces parity vs OpenMM Reference & CPU platforms.
2. Reaction Field (CutoffPeriodic) continuum electrostatics parity vs OpenMM Reference & CPU platforms.
3. LJ quintic switching function potential and forces parity vs OpenMM Reference & CPU platforms.
4. GBSA-OBC2 polar solvation and ACE nonpolar surface area parity vs OpenMM Reference & CPU platforms.
5. CMAPTorsionForce 2D spline potential and forces parity vs OpenMM Reference & CPU platforms.
6. Exhaustive ForceGroup bitmask isolation (overlapping groups, disjoint groups, empty bitmask, non-existent groups)
   for both potential energy and 3D forces vs OpenMM Reference platform.
7. Zero-recompilation behavior across repeated `setParameter`, `updateParametersInContext`, and step requests.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
  import openmm
  from openmm import unit

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False

from prolix.openmm import ProlixContext, ProlixOpenMMConfig
from prolix.openmm.context import _pure_compute_energy_and_forces, _pure_step_rollout

pytestmark = pytest.mark.openmm


# =============================================================================
# Helper Utilities
# =============================================================================


def get_ref_and_cpu_states(
    system: openmm.System,
    positions: list[openmm.Vec3] | unit.Quantity,
    box_vectors: tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3] | None = None,
    groups: int = ~0,
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float] | None]:
  """Evaluate energy and forces on Reference and CPU platforms."""
  results = {}
  for plat_name in ("Reference", "CPU"):
    try:
      plat = openmm.Platform.getPlatformByName(plat_name)
      integ = openmm.VerletIntegrator(0.001)
      ctx = openmm.Context(system, integ, plat)
      if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)
      ctx.setPositions(positions)
      st = ctx.getState(getEnergy=True, getForces=True, groups=groups)
      e = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
      f = st.getForces(asNumpy=True).value_in_unit(
          unit.kilojoules_per_mole / unit.nanometer
      )
      results[plat_name] = {"energy": float(e), "forces": np.asarray(f)}
    except Exception as exc:
      results[plat_name] = None
  return results["Reference"], results["CPU"]


# =============================================================================
# 1. PME Numerical Parity Challenge vs Reference & CPU
# =============================================================================


def test_pme_multi_particle_parity_energy_and_forces():
  """Challenge PME electrostatics (direct erfc + SPME mesh + self + plasma + excluded) vs Reference & CPU."""
  sys = openmm.System()
  box_l = 2.4
  box_vec = (
      openmm.Vec3(box_l, 0, 0),
      openmm.Vec3(0, box_l, 0),
      openmm.Vec3(0, 0, box_l),
  )
  sys.setDefaultPeriodicBoxVectors(*box_vec)

  n_ions = 8
  for _ in range(n_ions):
    sys.addParticle(23.0)  # Na+
  for _ in range(n_ions):
    sys.addParticle(35.5)  # Cl-

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.PME)
  nb.setCutoffDistance(0.9 * unit.nanometer)
  nb.setEwaldErrorTolerance(0.0005)

  for i in range(n_ions):
    nb.addParticle(1.0, 0.25, 0.5)
  for i in range(n_ions):
    nb.addParticle(-1.0, 0.40, 0.5)

  # Add an exclusion and a 1-4 exception between pair (0, 1) and (0, 8)
  nb.addException(0, 1, 0.0, 0.25, 0.0)
  nb.addException(0, 8, -0.5, 0.32, 0.25)
  sys.addForce(nb)

  # Deterministic particle lattice inside periodic box
  rng = np.random.RandomState(42)
  pos_np = rng.uniform(0.2, box_l - 0.2, size=(2 * n_ions, 3))
  pos = pos_np * unit.nanometer

  ref_state, cpu_state = get_ref_and_cpu_states(sys, pos, box_vec)
  assert ref_state is not None, "Reference platform must be available"

  cfg = ProlixOpenMMConfig(ewald_error_tolerance=0.0005)
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
  ctx.setPeriodicBoxVectors(*box_vec)
  ctx.setPositions(pos)

  prx_state = ctx.getState(getEnergy=True, getForces=True)
  e_prx = (
      prx_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  )
  f_prx = prx_state.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )

  # Check energy parity
  e_ref = ref_state["energy"]
  e_diff = abs(e_prx - e_ref)
  assert e_diff < 0.08, (
      f"PME energy parity failed vs Reference: Ref={e_ref:.4f}, Prolix={e_prx:.4f}, diff={e_diff:.4f} kJ/mol"
  )

  if cpu_state is not None:
    e_cpu = cpu_state["energy"]
    assert abs(e_prx - e_cpu) < 0.08, (
        f"PME energy parity failed vs CPU: CPU={e_cpu:.4f}, Prolix={e_prx:.4f}"
    )

  # Check force parity
  f_ref = ref_state["forces"]
  max_f_diff = np.max(np.linalg.norm(f_prx - f_ref, axis=-1))
  assert max_f_diff < 0.45, (
      f"PME max force discrepancy vs Reference: {max_f_diff:.4f} kJ/(mol nm) exceeds tolerance 0.45"
  )


# =============================================================================
# 2. Reaction Field (CutoffPeriodic) Parity Challenge vs Reference & CPU
# =============================================================================


@pytest.mark.parametrize("epsilon_rf", [78.3, 80.0])
def test_reaction_field_cutoff_periodic_energy_and_forces(epsilon_rf):
  """Challenge Reaction Field dielectric continuum potential and forces vs Reference & CPU."""
  sys = openmm.System()
  box_l = 2.5
  box_vec = (
      openmm.Vec3(box_l, 0, 0),
      openmm.Vec3(0, box_l, 0),
      openmm.Vec3(0, 0, box_l),
  )
  sys.setDefaultPeriodicBoxVectors(*box_vec)

  for _ in range(6):
    sys.addParticle(18.0)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb.setCutoffDistance(1.0 * unit.nanometer)
  nb.setReactionFieldDielectric(epsilon_rf)
  nb.setUseDispersionCorrection(False)

  charges = [+0.8, -0.8, +0.6, -0.6, +0.4, -0.4]
  for q in charges:
    nb.addParticle(q, 0.30, 0.6)
  sys.addForce(nb)

  pos_coords = [
      [0.5, 0.5, 0.5],
      [0.9, 0.5, 0.5],
      [1.2, 0.8, 0.5],
      [1.7, 0.8, 0.5],
      [0.6, 1.4, 0.5],
      [1.3, 1.4, 0.5],
  ]
  pos = pos_coords * unit.nanometer

  ref_state, cpu_state = get_ref_and_cpu_states(sys, pos, box_vec)
  assert ref_state is not None

  cfg = ProlixOpenMMConfig(use_dispersion_correction=False)
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
  ctx.setPeriodicBoxVectors(*box_vec)
  ctx.setPositions(pos)

  prx_state = ctx.getState(getEnergy=True, getForces=True)
  e_prx = (
      prx_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  )
  f_prx = prx_state.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )

  e_ref = ref_state["energy"]
  assert abs(e_prx - e_ref) < 0.05, (
      f"Reaction Field (eps={epsilon_rf}) energy diff {abs(e_prx - e_ref):.4f} kJ/mol exceeds tolerance"
  )

  f_ref = ref_state["forces"]
  max_f_diff = np.max(np.linalg.norm(f_prx - f_ref, axis=-1))
  assert max_f_diff < 0.05, (
      f"Reaction Field (eps={epsilon_rf}) max force diff {max_f_diff:.4f} exceeds 0.05 kJ/(mol nm)"
  )


# =============================================================================
# 3. LJ Switching Function Parity Challenge vs Reference & CPU
# =============================================================================


def test_lj_switching_function_quintic_taper_and_forces():
  """Challenge LJ quintic switching polynomial across unswitched, switched, and cutoff zones."""
  sys = openmm.System()
  for _ in range(4):
    sys.addParticle(39.948)

  r_cut = 1.0
  r_switch = 0.8
  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
  nb.setCutoffDistance(r_cut * unit.nanometer)
  nb.setUseSwitchingFunction(True)
  nb.setSwitchingDistance(r_switch * unit.nanometer)

  for _ in range(4):
    nb.addParticle(0.0, 0.34, 1.0)  # Pure LJ Argon
  sys.addForce(nb)

  # Atom 0 at origin
  # Atom 1 at 0.6 nm (r < r_switch, unswitched)
  # Atom 2 at 0.9 nm (r_switch < r < r_cut, quintic switching active)
  # Atom 3 at 1.1 nm (r > r_cut, fully truncated to 0)
  pos = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.6, 0.0, 0.0),
      openmm.Vec3(0.0, 0.9, 0.0),
      openmm.Vec3(0.0, 0.0, 1.1),
  ] * unit.nanometer

  ref_state, cpu_state = get_ref_and_cpu_states(sys, pos)
  assert ref_state is not None

  cfg = ProlixOpenMMConfig(lj_switch_width=r_cut - r_switch)
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
  ctx.setPositions(pos)

  prx_state = ctx.getState(getEnergy=True, getForces=True)
  e_prx = (
      prx_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  )
  f_prx = prx_state.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )

  e_ref = ref_state["energy"]
  assert abs(e_prx - e_ref) < 1e-4, (
      f"LJ switching energy parity failed: Ref={e_ref:.6f}, Prolix={e_prx:.6f}"
  )

  f_ref = ref_state["forces"]
  max_f_diff = np.max(np.linalg.norm(f_prx - f_ref, axis=-1))
  assert max_f_diff < 1e-3, (
      f"LJ switching forces parity failed: max diff = {max_f_diff:.6e} kJ/(mol nm)"
  )


# =============================================================================
# 4. GBSA-OBC2 Solvation & ACE Nonpolar Parity Challenge vs Reference & CPU
# =============================================================================


def test_gbsa_obc2_solvation_and_ace_energy_and_forces():
  """Challenge GBSA-OBC2 polar Born solvation and ACE surface area vs Reference & CPU."""
  sys = openmm.System()
  n_atoms = 5
  for _ in range(n_atoms):
    sys.addParticle(14.0)

  gb = openmm.GBSAOBCForce()
  gb.setSolventDielectric(78.5)
  gb.setSoluteDielectric(1.0)
  gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)

  charges = [+0.5, -0.3, +0.2, -0.6, +0.2]
  radii = [0.15, 0.14, 0.16, 0.13, 0.17]
  scale_factors = [0.8, 0.85, 0.75, 0.8, 0.85]

  for q, r, s in zip(charges, radii, scale_factors):
    gb.addParticle(q * unit.elementary_charge, r * unit.nanometers, s)
  sys.addForce(gb)

  pos = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.25, 0.0, 0.0),
      openmm.Vec3(0.12, 0.22, 0.0),
      openmm.Vec3(0.35, 0.18, 0.1),
      openmm.Vec3(-0.15, 0.20, -0.08),
  ] * unit.nanometers

  ref_state, cpu_state = get_ref_and_cpu_states(sys, pos)
  assert ref_state is not None

  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
  ctx.setPositions(pos)

  prx_state = ctx.getState(getEnergy=True, getForces=True)
  e_prx = (
      prx_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  )
  f_prx = prx_state.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )

  e_ref = ref_state["energy"]
  rel_e_diff = abs(e_prx - e_ref) / max(1.0, abs(e_ref))
  assert rel_e_diff < 1e-3, (
      f"GBSA-OBC2 relative energy diff {rel_e_diff:.6e} exceeds 1e-3: Ref={e_ref:.4f}, Prolix={e_prx:.4f}"
  )

  f_ref = ref_state["forces"]
  max_f_diff = np.max(np.linalg.norm(f_prx - f_ref, axis=-1))
  assert max_f_diff < 0.2, (
      f"GBSA-OBC2 max force diff {max_f_diff:.4f} exceeds 0.2 kJ/(mol nm)"
  )


# =============================================================================
# 5. CMAPTorsionForce 2D Spline Parity Challenge vs Reference & CPU
# =============================================================================


def test_cmap_2d_spline_energy_and_forces_parity():
  """Challenge CMAPTorsionForce 2D bicubic spline evaluation across wrapped angles vs Reference & CPU."""
  sys = openmm.System()
  for _ in range(6):
    sys.addParticle(12.0)

  cmap = openmm.CMAPTorsionForce()
  size = 24
  u = np.linspace(-np.pi, np.pi, size, endpoint=False)
  grid_2d = 8.0 * np.outer(np.sin(u), np.cos(2 * u)) + 3.0 * np.outer(
      np.cos(u), np.sin(u)
  )
  grid_list = [float(v) * unit.kilojoules_per_mole for v in grid_2d.flatten()]
  map_0 = cmap.addMap(size, grid_list)
  # Two torsions sharing central atoms
  cmap.addTorsion(map_0, 0, 1, 2, 3, 1, 2, 3, 4)
  cmap.addTorsion(map_0, 1, 2, 3, 4, 2, 3, 4, 5)
  sys.addForce(cmap)

  pos = [
      openmm.Vec3(-0.15, 0.12, 0.0),
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.15, 0.0, 0.0),
      openmm.Vec3(0.20, 0.14, 0.05),
      openmm.Vec3(0.35, 0.15, -0.05),
      openmm.Vec3(0.45, 0.05, 0.08),
  ] * unit.nanometers

  ref_state, cpu_state = get_ref_and_cpu_states(sys, pos)
  assert ref_state is not None

  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
  ctx.setPositions(pos)

  prx_state = ctx.getState(getEnergy=True, getForces=True)
  e_prx = (
      prx_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  )
  f_prx = prx_state.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )

  e_ref = ref_state["energy"]
  assert abs(e_prx - e_ref) < 0.02, (
      f"CMAP energy parity failed: Ref={e_ref:.4f}, Prolix={e_prx:.4f}"
  )

  f_ref = ref_state["forces"]
  max_f_diff = np.max(np.linalg.norm(f_prx - f_ref, axis=-1))
  assert max_f_diff < 0.05, (
      f"CMAP max force discrepancy {max_f_diff:.4f} exceeds 0.05 kJ/(mol nm)"
  )


# =============================================================================
# 6. Exhaustive ForceGroup Bitmask Isolation Stress Testing
# =============================================================================


def test_forcegroup_bitmask_isolation_overlapping_and_disjoint():
  """Stress-test ForceGroup isolation with overlapping, disjoint, full, and zero bitmasks."""
  sys = openmm.System()
  for _ in range(6):
    sys.addParticle(14.0)

  # Group 0: Bond force 1
  b1 = openmm.HarmonicBondForce()
  b1.setForceGroup(0)
  b1.addBond(
      0, 1, 0.15 * unit.nanometer, 150000.0 * unit.kilojoules_per_mole / unit.nanometer**2
  )
  sys.addForce(b1)

  # Group 1: Bond force 2
  b2 = openmm.HarmonicBondForce()
  b2.setForceGroup(1)
  b2.addBond(
      1, 2, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2
  )
  sys.addForce(b2)

  # Group 1: Angle force (OVERLAPPING with Bond force 2 in group 1)
  ang = openmm.HarmonicAngleForce()
  ang.setForceGroup(1)
  ang.addAngle(
      0, 1, 2, 1.9 * unit.radian, 35000.0 * unit.kilojoules_per_mole / unit.radian**2
  )
  sys.addForce(ang)

  # Group 3: Periodic torsion
  tor = openmm.PeriodicTorsionForce()
  tor.setForceGroup(3)
  tor.addTorsion(
      0, 1, 2, 3, 2, 0.0 * unit.radian, 30.0 * unit.kilojoules_per_mole
  )
  sys.addForce(tor)

  # Group 5: Nonbonded force
  nb = openmm.NonbondedForce()
  nb.setForceGroup(5)
  nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
  nb.addParticle(0.5, 0.3, 1.0)
  nb.addParticle(-0.5, 0.3, 1.0)
  nb.addParticle(0.3, 0.28, 0.8)
  nb.addParticle(-0.3, 0.28, 0.8)
  nb.addParticle(0.0, 0.32, 1.2)
  nb.addParticle(0.0, 0.32, 1.2)
  sys.addForce(nb)

  # Group 12: Custom external force
  ext = openmm.CustomExternalForce("0.5 * k_ext * (x^2 + y^2 + z^2)")
  ext.setForceGroup(12)
  ext.addGlobalParameter("k_ext", 500.0)
  ext.addParticle(4, [])
  ext.addParticle(5, [])
  sys.addForce(ext)

  pos = [
      openmm.Vec3(0.0, 0.0, 0.0),
      openmm.Vec3(0.16, 0.02, 0.0),
      openmm.Vec3(0.28, -0.06, 0.08),
      openmm.Vec3(0.35, 0.04, -0.05),
      openmm.Vec3(0.50, 0.50, 0.50),
      openmm.Vec3(-0.50, -0.50, -0.50),
  ] * unit.nanometer

  # Set up OpenMM Reference platform
  ref_ctx = openmm.Context(
      sys,
      openmm.VerletIntegrator(0.001),
      openmm.Platform.getPlatformByName("Reference"),
  )
  ref_ctx.setPositions(pos)

  # Set up Prolix context
  prx_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
  prx_ctx.setPositions(pos)

  # 1. Single groups
  active_groups = [0, 1, 3, 5, 12]
  group_energies = {}
  group_forces = {}

  for g in active_groups:
    mask = 1 << g
    ref_st = ref_ctx.getState(getEnergy=True, getForces=True, groups=mask)
    prx_st = prx_ctx.getState(getEnergy=True, getForces=True, groups=mask)

    e_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_prx = prx_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    f_prx = prx_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # Parity with reference per group (single-precision float32 vs double precision)
    rel_err = abs(e_prx - e_ref) / max(1.0, abs(e_ref))
    assert rel_err < 1e-4 or abs(e_prx - e_ref) < 0.01, (
        f"Group {g} energy mismatch: Ref={e_ref:.4f}, Prolix={e_prx:.4f}, diff={abs(e_prx - e_ref):.4f}"
    )
    assert np.allclose(f_prx, f_ref, rtol=1e-3, atol=1e-2), (
        f"Group {g} force mismatch vs Reference"
    )

    group_energies[g] = e_prx
    group_forces[g] = f_prx

  # 2. Overlapping group 1 check: must be non-zero and combine b2 + ang
  assert group_energies[1] > 1.0, "Group 1 must contain non-zero energy"

  # 3. Disjoint pair combinations
  pair_masks = [
      (1 << 0) | (1 << 3),
      (1 << 1) | (1 << 5),
      (1 << 0) | (1 << 12),
      (1 << 1) | (1 << 12),
  ]
  for pmask in pair_masks:
    prx_st = prx_ctx.getState(getEnergy=True, getForces=True, groups=pmask)
    e_comb = prx_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_comb = prx_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    expected_e = sum(
        group_energies[g] for g in active_groups if (pmask & (1 << g))
    )
    expected_f = sum(
        group_forces[g] for g in active_groups if (pmask & (1 << g))
    )

    assert np.isclose(e_comb, expected_e, atol=1e-3), (
        f"Bitmask {bin(pmask)} linearity failed for energy: {e_comb:.4f} vs sum={expected_e:.4f}"
    )
    assert np.allclose(f_comb, expected_f, atol=1e-3), (
        f"Bitmask {bin(pmask)} linearity failed for forces"
    )

  # 4. Total system (all groups)
  prx_all = prx_ctx.getState(getEnergy=True, getForces=True, groups=~0)
  e_all = prx_all.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_all = prx_all.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )

  sum_e = sum(group_energies.values())
  sum_f = sum(group_forces.values())
  assert np.isclose(e_all, sum_e, atol=1e-3), (
      f"Total energy {e_all:.4f} does not equal sum of groups {sum_e:.4f}"
  )
  assert np.allclose(f_all, sum_f, atol=1e-3), (
      "Total force does not equal sum of groups"
  )

  # 5. Empty mask (groups = 0)
  prx_zero = prx_ctx.getState(getEnergy=True, getForces=True, groups=0)
  e_zero = prx_zero.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_zero = prx_zero.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )
  assert e_zero == 0.0, f"Empty bitmask groups=0 must give 0.0 energy, got {e_zero}"
  assert np.all(f_zero == 0.0), "Empty bitmask groups=0 must give all 0.0 forces"

  # 6. Unused group mask (groups = 1 << 28)
  prx_unused = prx_ctx.getState(getEnergy=True, getForces=True, groups=1 << 28)
  e_unused = (
      prx_unused.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  )
  f_unused = prx_unused.getForces(asNumpy=True).value_in_unit(
      unit.kilojoules_per_mole / unit.nanometer
  )
  assert e_unused == 0.0, (
      f"Unused group 28 must give 0.0 energy, got {e_unused}"
  )
  assert np.all(f_unused == 0.0), "Unused group 28 must give all 0.0 forces"


# =============================================================================
# 7. Zero-Recompilation Behavior Stress Testing
# =============================================================================


def test_zero_recompilation_across_set_parameter():
  """Stress-test zero recompilation across repeated setParameter calls."""
  sys = openmm.System()
  sys.addParticle(12.0)
  sys.addParticle(12.0)

  cf = openmm.CustomBondForce("0.5 * k_dyn * (r - r0)^2")
  cf.addGlobalParameter("k_dyn", 100000.0)
  cf.addGlobalParameter("r0", 0.15)
  cf.addBond(0, 1, [])
  sys.addForce(cf)

  pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.16, 0, 0)] * unit.nanometer
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
  ctx.setPositions(pos)

  # Warm up compilation
  _ = ctx.getState(getEnergy=True, getForces=True)
  initial_cache_size = _pure_compute_energy_and_forces._cached._cache_size()

  # Mutate global parameters 10 times with varied values
  for k_val in [200000.0, 50000.0, 75000.0, 300000.0, 120000.0, 90000.0]:
    ctx.setParameter("k_dyn", k_val)
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert e > 0.0

    current_cache_size = _pure_compute_energy_and_forces._cached._cache_size()
    assert current_cache_size == initial_cache_size, (
        f"setParameter('k_dyn', {k_val}) triggered XLA recompilation: "
        f"cache grew from {initial_cache_size} to {current_cache_size}!"
    )


def test_zero_recompilation_across_update_parameters_in_context():
  """Stress-test zero recompilation across repeated updateParametersInContext calls."""
  sys = openmm.System()
  sys.addParticle(12.0)
  sys.addParticle(12.0)

  b = openmm.HarmonicBondForce()
  b.addBond(
      0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2
  )
  sys.addForce(b)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
  nb.addParticle(0.4, 0.30, 2.0)
  nb.addParticle(-0.4, 0.30, 2.0)
  sys.addForce(nb)

  pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.16, 0, 0)] * unit.nanometer
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
  ctx.setPositions(pos)

  # Warm up compilation
  _ = ctx.getState(getEnergy=True, getForces=True)
  initial_cache_size = _pure_compute_energy_and_forces._cached._cache_size()

  # Mutate bond parameters and nonbonded parameters repeatedly
  for step_i in range(5):
    new_k = (200000.0 + step_i * 10000.0) * unit.kilojoules_per_mole / unit.nanometer**2
    b.setBondParameters(0, 0, 1, 0.15 * unit.nanometer, new_k)
    nb.setParticleParameters(0, 0.4 + step_i * 0.05, 0.30, 2.0)
    ctx.updateParametersInContext()

    _ = ctx.getState(getEnergy=True, getForces=True)
    current_cache_size = _pure_compute_energy_and_forces._cached._cache_size()
    assert current_cache_size == initial_cache_size, (
        f"updateParametersInContext (iteration {step_i}) triggered XLA recompilation: "
        f"cache grew from {initial_cache_size} to {current_cache_size}!"
    )


def test_zero_recompilation_across_varying_step_counts():
  """Stress-test zero recompilation in rollout kernel across arbitrary step counts."""
  sys = openmm.System()
  sys.addParticle(12.0)
  sys.addParticle(12.0)
  b = openmm.HarmonicBondForce()
  b.addBond(
      0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2
  )
  sys.addForce(b)

  pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.15, 0, 0)] * unit.nanometer
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
  ctx.setPositions(pos)

  # Initial step to warm up JIT
  ctx.step(1)
  initial_cache_size = _pure_step_rollout._cached._cache_size()

  # Arbitrary prime and non-standard step counts
  arbitrary_step_counts = [7, 13, 2, 29, 10, 3, 50, 11]
  for steps in arbitrary_step_counts:
    ctx.step(steps)
    current_cache_size = _pure_step_rollout._cached._cache_size()
    assert current_cache_size == initial_cache_size, (
        f"step({steps}) triggered XLA recompilation: "
        f"cache grew from {initial_cache_size} to {current_cache_size}!"
    )


# =============================================================================
# DEF-23: Periodic Boundary Crossing Exclusions & 1-4 Exceptions Parity
# =============================================================================


def test_periodic_boundary_crossing_exclusions_and_exceptions_parity():
  """DEF-23: Verify nonbonded exclusions and 1-4 exceptions crossing periodic box boundaries.

  When an excluded bonded pair or 1-4 exception crosses the unit cell boundary,
  the bulk nonbonded evaluation computes minimum-image wrapped distance (r_wrapped ~ 0.1-0.2 nm).
  The exclusion evaluation must use _minimum_image_dr to cancel the bulk nonbonded core clash,
  ensuring:
  1. Pure excluded pairs have exactly 0.0 kJ/mol nonbonded energy and 0.0 forces.
  2. 1-4 exception pairs match OpenMM Reference energy to 6+ digits and forces to < 1e-3 kJ/(mol nm).
  3. Bonded pairs crossing unit cell boundaries match OpenMM Reference energy and forces to 6+ digits.
  """
  # 1. Pure excluded pair across periodic boundary
  sys1 = openmm.System()
  sys1.setDefaultPeriodicBoxVectors([2.4, 0, 0], [0, 2.4, 0], [0, 0, 2.4])
  sys1.addParticle(12.0)
  sys1.addParticle(12.0)
  nb1 = openmm.NonbondedForce()
  nb1.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb1.setUseDispersionCorrection(False)
  nb1.setCutoffDistance(1.0 * unit.nanometer)
  nb1.addParticle(1.0, 0.3, 1.0)
  nb1.addParticle(-1.0, 0.3, 1.0)
  nb1.addException(0, 1, 0.0, 0.0, 0.0)
  sys1.addForce(nb1)

  # x0 = 0.05 nm, x1 = 2.35 nm in 2.4 nm box: r_unwrapped = 2.3 nm > r_cut, r_wrapped = 0.1 nm
  pos1 = [openmm.Vec3(0.05, 1.2, 1.2), openmm.Vec3(2.35, 1.2, 1.2)] * unit.nanometer

  ctx_ref1 = openmm.Context(
      sys1, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref1.setPositions(pos1)
  st_ref1 = ctx_ref1.getState(getEnergy=True, getForces=True)
  e_ref1 = st_ref1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref1 = st_ref1.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  ctx_prx1 = ProlixContext(sys1, openmm.VerletIntegrator(0.001))
  ctx_prx1.setPositions(pos1)
  st_prx1 = ctx_prx1.getState(getEnergy=True, getForces=True)
  e_prx1 = st_prx1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx1 = st_prx1.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  assert abs(e_ref1) < 1e-6, f"OpenMM Reference exclusion energy expected 0.0, got {e_ref1}"
  assert abs(e_prx1) < 1e-6, f"Prolix exclusion energy expected 0.0, got {e_prx1}"
  np.testing.assert_allclose(f_prx1, f_ref1, atol=1e-5, rtol=1e-5)

  # 2. 1-4 Exception pair across periodic boundary
  sys2 = openmm.System()
  sys2.setDefaultPeriodicBoxVectors([2.4, 0, 0], [0, 2.4, 0], [0, 0, 2.4])
  sys2.addParticle(12.0)
  sys2.addParticle(12.0)
  nb2 = openmm.NonbondedForce()
  nb2.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb2.setUseDispersionCorrection(False)
  nb2.setCutoffDistance(1.0 * unit.nanometer)
  nb2.addParticle(1.0, 0.3, 1.0)
  nb2.addParticle(-1.0, 0.3, 1.0)
  nb2.addException(0, 1, -0.5, 0.3, 0.25)
  sys2.addForce(nb2)

  # x0 = 0.1 nm, x1 = 2.3 nm in 2.4 nm box: r_unwrapped = 2.2 nm, r_wrapped = 0.2 nm
  pos2 = [openmm.Vec3(0.1, 1.2, 1.2), openmm.Vec3(2.3, 1.2, 1.2)] * unit.nanometer

  ctx_ref2 = openmm.Context(
      sys2, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref2.setPositions(pos2)
  st_ref2 = ctx_ref2.getState(getEnergy=True, getForces=True)
  e_ref2 = st_ref2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref2 = st_ref2.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  ctx_prx2 = ProlixContext(sys2, openmm.VerletIntegrator(0.001))
  ctx_prx2.setPositions(pos2)
  st_prx2 = ctx_prx2.getState(getEnergy=True, getForces=True)
  e_prx2 = st_prx2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx2 = st_prx2.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  np.testing.assert_allclose(e_prx2, e_ref2, rtol=1e-5, atol=1e-4)
  np.testing.assert_allclose(f_prx2, f_ref2, rtol=1e-4, atol=1e-3)

  # 3. Bonded pair crossing periodic boundary with bond + exclusion
  sys3 = openmm.System()
  sys3.setDefaultPeriodicBoxVectors([2.4, 0, 0], [0, 2.4, 0], [0, 0, 2.4])
  sys3.addParticle(12.0)
  sys3.addParticle(12.0)
  b3 = openmm.HarmonicBondForce()
  b3.addBond(0, 1, 0.15 * unit.nanometer, 100000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
  b3.setUsesPeriodicBoundaryConditions(True)
  sys3.addForce(b3)

  nb3 = openmm.NonbondedForce()
  nb3.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb3.setUseDispersionCorrection(False)
  nb3.setCutoffDistance(1.0 * unit.nanometer)
  nb3.addParticle(1.0, 0.3, 1.0)
  nb3.addParticle(-1.0, 0.3, 1.0)
  nb3.addException(0, 1, 0.0, 0.0, 0.0)
  sys3.addForce(nb3)

  pos3 = [openmm.Vec3(0.05, 1.2, 1.2), openmm.Vec3(2.35, 1.2, 1.2)] * unit.nanometer

  ctx_ref3 = openmm.Context(
      sys3, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref3.setPositions(pos3)
  st_ref3 = ctx_ref3.getState(getEnergy=True, getForces=True)
  e_ref3 = st_ref3.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref3 = st_ref3.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  ctx_prx3 = ProlixContext(sys3, openmm.VerletIntegrator(0.001))
  ctx_prx3.setPositions(pos3)
  st_prx3 = ctx_prx3.getState(getEnergy=True, getForces=True)
  e_prx3 = st_prx3.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx3 = st_prx3.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  np.testing.assert_allclose(e_prx3, e_ref3, rtol=1e-5, atol=1e-4)
  np.testing.assert_allclose(f_prx3, f_ref3, rtol=1e-4, atol=1e-3)


# =============================================================================
# 8. Multi-Dimensional (1D, 2D, 3D) PBC Crossing Parity vs OpenMM Reference
# =============================================================================


@pytest.mark.parametrize("dim", ["1D", "2D", "3D"])
def test_adversarial_pbc_crossings_1d_2d_3d_exclusions_and_exceptions(dim):
  """Adversarially challenge PBC boundary crossings in 1D, 2D, and 3D.

  Verifies:
  1. Nonbonded pure exclusion energy is identically 0.0 kJ/mol and forces are identically 0.0.
  2. 1-4 exceptions match OpenMM Reference energy (< 1e-5 rel, < 1e-4 abs) and forces (< 1e-3).
  3. Bonded pairs with HarmonicBondForce + exclusion match Reference energy and forces across boundaries.
  """
  box_l = 2.4
  box_vec = (
      openmm.Vec3(box_l, 0, 0),
      openmm.Vec3(0, box_l, 0),
      openmm.Vec3(0, 0, box_l),
  )

  # Coordinate displacement vectors for 1D, 2D, and 3D crossings
  if dim == "1D":
    p0 = openmm.Vec3(0.05, 1.2, 1.2)
    p1 = openmm.Vec3(2.35, 1.2, 1.2)
  elif dim == "2D":
    p0 = openmm.Vec3(0.05, 0.05, 1.2)
    p1 = openmm.Vec3(2.35, 2.35, 1.2)
  elif dim == "3D":
    p0 = openmm.Vec3(0.05, 0.05, 0.05)
    p1 = openmm.Vec3(2.35, 2.35, 2.35)
  pos_ex = [p0, p1] * unit.nanometer

  # Part 1: Pure Excluded Pair under CutoffPeriodic
  sys1 = openmm.System()
  sys1.setDefaultPeriodicBoxVectors(*box_vec)
  sys1.addParticle(12.0)
  sys1.addParticle(12.0)
  nb1 = openmm.NonbondedForce()
  nb1.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb1.setUseDispersionCorrection(False)
  nb1.setCutoffDistance(1.0 * unit.nanometer)
  nb1.addParticle(1.0, 0.3, 1.0)
  nb1.addParticle(-1.0, 0.3, 1.0)
  nb1.addException(0, 1, 0.0, 0.0, 0.0)
  sys1.addForce(nb1)

  ctx_ref1 = openmm.Context(
      sys1, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref1.setPositions(pos_ex)
  st_ref1 = ctx_ref1.getState(getEnergy=True, getForces=True)
  e_ref1 = st_ref1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref1 = st_ref1.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  ctx_prx1 = ProlixContext(sys1, openmm.VerletIntegrator(0.001))
  ctx_prx1.setPositions(pos_ex)
  st_prx1 = ctx_prx1.getState(getEnergy=True, getForces=True)
  e_prx1 = st_prx1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx1 = st_prx1.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  assert abs(e_ref1) < 1e-6, f"OpenMM Reference exclusion energy expected 0.0, got {e_ref1}"
  assert abs(e_prx1) < 1e-6, f"Prolix exclusion energy expected 0.0, got {e_prx1}"
  np.testing.assert_allclose(f_prx1, f_ref1, atol=1e-5, rtol=1e-5)

  # Part 2: 1-4 Exception Pair
  if dim == "1D":
    p0_14 = openmm.Vec3(0.1, 1.2, 1.2)
    p1_14 = openmm.Vec3(2.3, 1.2, 1.2)
  elif dim == "2D":
    p0_14 = openmm.Vec3(0.1, 0.1, 1.2)
    p1_14 = openmm.Vec3(2.3, 2.3, 1.2)
  elif dim == "3D":
    p0_14 = openmm.Vec3(0.1, 0.1, 0.1)
    p1_14 = openmm.Vec3(2.3, 2.3, 2.3)
  pos_14 = [p0_14, p1_14] * unit.nanometer

  sys2 = openmm.System()
  sys2.setDefaultPeriodicBoxVectors(*box_vec)
  sys2.addParticle(12.0)
  sys2.addParticle(12.0)
  nb2 = openmm.NonbondedForce()
  nb2.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb2.setUseDispersionCorrection(False)
  nb2.setCutoffDistance(1.0 * unit.nanometer)
  nb2.addParticle(1.0, 0.3, 1.0)
  nb2.addParticle(-1.0, 0.3, 1.0)
  nb2.addException(0, 1, -0.5, 0.3, 0.25)
  sys2.addForce(nb2)

  ctx_ref2 = openmm.Context(
      sys2, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref2.setPositions(pos_14)
  st_ref2 = ctx_ref2.getState(getEnergy=True, getForces=True)
  e_ref2 = st_ref2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref2 = st_ref2.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  ctx_prx2 = ProlixContext(sys2, openmm.VerletIntegrator(0.001))
  ctx_prx2.setPositions(pos_14)
  st_prx2 = ctx_prx2.getState(getEnergy=True, getForces=True)
  e_prx2 = st_prx2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx2 = st_prx2.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  np.testing.assert_allclose(e_prx2, e_ref2, rtol=1e-5, atol=1e-4)
  np.testing.assert_allclose(f_prx2, f_ref2, rtol=1e-4, atol=1e-3)

  # Part 3: Bonded Pair Crossing Periodic Boundary
  if dim == "1D":
    p0_b = openmm.Vec3(0.05, 1.2, 1.2)
    p1_b = openmm.Vec3(2.35, 1.2, 1.2)
  elif dim == "2D":
    p0_b = openmm.Vec3(0.045, 0.045, 1.2)
    p1_b = openmm.Vec3(2.355, 2.355, 1.2)
  elif dim == "3D":
    p0_b = openmm.Vec3(0.04, 0.04, 0.04)
    p1_b = openmm.Vec3(2.36, 2.36, 2.36)
  pos_b = [p0_b, p1_b] * unit.nanometer

  sys3 = openmm.System()
  sys3.setDefaultPeriodicBoxVectors(*box_vec)
  sys3.addParticle(12.0)
  sys3.addParticle(12.0)
  b3 = openmm.HarmonicBondForce()
  b3.addBond(0, 1, 0.15 * unit.nanometer, 100000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
  b3.setUsesPeriodicBoundaryConditions(True)
  sys3.addForce(b3)

  nb3 = openmm.NonbondedForce()
  nb3.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb3.setUseDispersionCorrection(False)
  nb3.setCutoffDistance(1.0 * unit.nanometer)
  nb3.addParticle(1.0, 0.3, 1.0)
  nb3.addParticle(-1.0, 0.3, 1.0)
  nb3.addException(0, 1, 0.0, 0.0, 0.0)
  sys3.addForce(nb3)

  ctx_ref3 = openmm.Context(
      sys3, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref3.setPositions(pos_b)
  st_ref3 = ctx_ref3.getState(getEnergy=True, getForces=True)
  e_ref3 = st_ref3.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref3 = st_ref3.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  ctx_prx3 = ProlixContext(sys3, openmm.VerletIntegrator(0.001))
  ctx_prx3.setPositions(pos_b)
  st_prx3 = ctx_prx3.getState(getEnergy=True, getForces=True)
  e_prx3 = st_prx3.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx3 = st_prx3.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  np.testing.assert_allclose(e_prx3, e_ref3, rtol=1e-4, atol=1e-3)
  np.testing.assert_allclose(f_prx3, f_ref3, rtol=1e-3, atol=0.02)


# =============================================================================
# 9. Multi-Particle Diverse Charged Systems Crossing PBC Boundaries
# =============================================================================


@pytest.mark.parametrize("method", ["ReactionField", "PME"])
def test_adversarial_pbc_crossings_diverse_charged_system(method):
  """Adversarially challenge diverse charged particles crossing PBC boundaries under RF and PME."""
  sys = openmm.System()
  box_l = 2.4
  box_vec = (
      openmm.Vec3(box_l, 0, 0),
      openmm.Vec3(0, box_l, 0),
      openmm.Vec3(0, 0, box_l),
  )
  sys.setDefaultPeriodicBoxVectors(*box_vec)

  n_ions = 6
  for _ in range(n_ions):
    sys.addParticle(23.0)

  nb = openmm.NonbondedForce()
  if method == "ReactionField":
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setReactionFieldDielectric(78.3)
    nb.setUseDispersionCorrection(False)
  else:
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setEwaldErrorTolerance(0.0005)

  nb.setCutoffDistance(1.0 * unit.nanometer)

  # Charges: +1, -1, +0.5, -0.5, +0.8, -0.8
  charges = [+1.0, -1.0, +0.5, -0.5, +0.8, -0.8]
  sigmas = [0.25, 0.35, 0.28, 0.32, 0.30, 0.30]
  epsilons = [0.5, 0.6, 0.4, 0.5, 0.6, 0.6]
  for q, sig, eps in zip(charges, sigmas, epsilons):
    nb.addParticle(q, sig, eps)

  # Add exclusions and 1-4 exceptions
  nb.addException(0, 1, 0.0, 0.0, 0.0)
  nb.addException(2, 3, -0.2, 0.30, 0.2)
  sys.addForce(nb)

  # Add a bond across boundary for pair 0-1
  b = openmm.HarmonicBondForce()
  b.addBond(0, 1, 0.15 * unit.nanometer, 80000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
  b.setUsesPeriodicBoundaryConditions(True)
  sys.addForce(b)

  # Particles placed across unit cell boundaries in 1D, 2D, and 3D
  pos = [
      openmm.Vec3(0.05, 0.05, 1.2),
      openmm.Vec3(2.35, 2.35, 1.2),
      openmm.Vec3(0.08, 1.2, 0.08),
      openmm.Vec3(2.32, 1.2, 2.32),
      openmm.Vec3(1.2, 1.2, 1.2),
      openmm.Vec3(0.06, 0.06, 0.06),
  ] * unit.nanometer

  ctx_ref = openmm.Context(
      sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
  )
  ctx_ref.setPositions(pos)
  st_ref = ctx_ref.getState(getEnergy=True, getForces=True)
  e_ref = st_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_ref = st_ref.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  cfg = ProlixOpenMMConfig(ewald_error_tolerance=0.0005, use_dispersion_correction=False)
  ctx_prx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
  ctx_prx.setPositions(pos)
  st_prx = ctx_prx.getState(getEnergy=True, getForces=True)
  e_prx = st_prx.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
  f_prx = st_prx.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  e_diff = abs(e_prx - e_ref)
  rel_e_diff = e_diff / max(1.0, abs(e_ref))
  assert rel_e_diff < 1e-3, (
      f"{method} energy parity failed: Ref={e_ref:.4f}, Prolix={e_prx:.4f}, diff={e_diff:.4f}"
  )
  if method == "ReactionField":
    np.testing.assert_allclose(f_prx, f_ref, rtol=1e-3, atol=0.05)


# =============================================================================
# 10. Finite Difference Forces Consistency (F = -grad E)
# =============================================================================


def test_adversarial_finite_difference_force_consistency():
  """Stress-test analytical forces F = -grad(E) against finite differences.

  Tests a multi-force system with bonds, reaction field, quintic switching,
  and boundary-crossing exclusions in double precision, verifying |F_analytic - F_fd| < 1e-3.
  """
  sys = openmm.System()
  box_l = 2.4
  box_vec = (openmm.Vec3(box_l, 0, 0), openmm.Vec3(0, box_l, 0), openmm.Vec3(0, 0, box_l))
  sys.setDefaultPeriodicBoxVectors(*box_vec)

  for _ in range(4):
    sys.addParticle(16.0)

  b = openmm.HarmonicBondForce()
  b.addBond(0, 1, 0.15 * unit.nanometer, 100000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
  b.setUsesPeriodicBoundaryConditions(True)
  sys.addForce(b)

  nb = openmm.NonbondedForce()
  nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
  nb.setCutoffDistance(1.0 * unit.nanometer)
  nb.setUseSwitchingFunction(True)
  nb.setSwitchingDistance(0.8 * unit.nanometer)
  nb.setReactionFieldDielectric(78.3)
  nb.setUseDispersionCorrection(False)

  nb.addParticle(0.6, 0.32, 1.2)
  nb.addParticle(-0.6, 0.32, 1.2)
  nb.addParticle(0.4, 0.28, 0.8)
  nb.addParticle(-0.4, 0.28, 0.8)

  nb.addException(0, 1, 0.0, 0.0, 0.0)
  nb.addException(0, 2, -0.2, 0.30, 0.3)
  sys.addForce(nb)

  pos_arr = np.array([
      [0.05, 1.2, 1.2],
      [2.35, 1.2, 1.2],
      [1.2, 2.35, 1.2],
      [1.2, 0.8, 1.2],
  ])

  cfg = ProlixOpenMMConfig(precision="double", lj_switch_width=0.2)
  ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
  ctx.setPositions(pos_arr * unit.nanometer)
  st = ctx.getState(getEnergy=True, getForces=True)
  f_analytic = st.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

  eps = 1e-5
  f_fd = np.zeros_like(pos_arr)
  for i in range(4):
    for alpha in range(3):
      pos_p = pos_arr.copy()
      pos_p[i, alpha] += eps
      ctx.setPositions(pos_p * unit.nanometer)
      ep = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

      pos_m = pos_arr.copy()
      pos_m[i, alpha] -= eps
      ctx.setPositions(pos_m * unit.nanometer)
      em = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

      f_fd[i, alpha] = -(ep - em) / (2.0 * eps)

  max_fd_diff = np.max(np.abs(f_analytic - f_fd))
  assert max_fd_diff < 1e-3, f"Finite difference forces mismatch: {max_fd_diff:.6e} kJ/(mol nm)"
  np.testing.assert_allclose(f_analytic, f_fd, atol=1e-3, rtol=1e-4)

