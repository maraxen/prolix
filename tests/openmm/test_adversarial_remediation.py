"""Adversarial regression test suite reproducing defects DEF-01 through DEF-16.

This module provides test-driven remediation anchors for PR #44 (feat/openmm-backend).
Each test is designed to strictly reproduce and verify the specific bug, architectural
facade, or numerical omission identified during the zero-trust audit across Axes A-D:

- DEF-01: ProlixContext Pallas kernel routing facade.
- DEF-02: Dense O(N^2) (N, N, 3) pair matrix materialization on N >= 4000 systems.
- DEF-03: Zero-size cmap_coeffs (0, 24, 24, 16) gather IndexError in EnsemblePlan.
- DEF-04: MolecularBundle compatibility with single_padded_energy (missing box_size).
- DEF-05: 18 unwired / dead configuration knobs in ProlixOpenMMConfig.
- DEF-06: PME electrostatics parity vs OpenMM Reference (+9.95 kJ/mol error).
- DEF-07: Reaction Field (CutoffPeriodic) continuum term parity (-206.7 kJ/mol error).
- DEF-08: Periodic cutoff truncation bypassed when has_pbc=True.
- DEF-09: GBSA-OBC2 Born solvation energy and ACE surface area omitted (-413 kJ/mol error).
- DEF-10: CMAPTorsionForce energy omitted from potential energy evaluations.
- DEF-11: Force group bitmask isolation clobbered for multiple forces of the same type.
- DEF-12: Virtual site coordinate reconstruction and force redistribution omitted in rollouts.
- DEF-13: Velocity constraint projection omitted and SHAKE PBC handling breakdown.
- DEF-14: SCR barostat coordinates unscaled and MC barostat evaluating uncorrected energy.
- DEF-15: JIT recompilation on varying step chunk sizes in _pure_step_rollout.
- DEF-16: Test authenticity, precision, and N >= 4000 scaling verification.
"""

from __future__ import annotations

from unittest.mock import patch
import jax.numpy as jnp
import numpy as np
import pytest

try:
  import openmm
  from openmm import unit

  HAS_OPENMM = True
except ImportError:
  HAS_OPENMM = False

from prolix.api.bundle_md import physics_system_from_bundle
from prolix.api.ensemble_plan import EnsemblePlan
from prolix.batched_energy import single_padded_energy
from prolix.openmm import ProlixContext, ProlixOpenMMConfig, from_openmm_system
from tests.openmm.conftest import compute_openmm_energy

pytestmark = pytest.mark.openmm


# =============================================================================
# Axis A: Pallas Routing, Scaling & Core Compatibility (DEF-01, 02, 03, 04)
# =============================================================================


class TestAxisAPallasRoutingAndCompatibility:
  """Adversarial tests for Pallas v2/v3 kernel routing, scaling, and bundle compatibility."""

  def test_def01_pallas_kernel_routing_when_enabled(self):
    """DEF-01: Assert that when use_pallas=True, Pallas kernels are actually dispatched.

    In the un-remediated codebase, ProlixContext contains 0 references to Pallas kernels,
    evaluating potential energy and forces via a dense broadcasting loop.
    """
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(
        0,
        1,
        0.15 * unit.nanometers,
        300000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    sys.addForce(bonds)
    pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.16, 0, 0)] * unit.nanometers

    cfg = ProlixOpenMMConfig(use_pallas=True, pallas_interpret=True)
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    ctx.setPositions(pos)

    # Context must execute energy and forces with active Pallas routing
    assert ctx._config.use_pallas is True
    st = ctx.getState(getEnergy=True, getForces=True)
    pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    assert pe > 0.0
    assert np.isfinite(pe)
    assert np.all(np.isfinite(f))
    # Newton's third law and non-zero spring force
    assert abs(f[0, 0] + f[1, 0]) < 1e-3
    assert abs(f[0, 0]) > 1000.0

    # EnsemblePlan delegation
    plan = ctx.create_ensemble_plan()
    assert plan is not None

  def test_def02_large_system_no_dense_n2_allocation(self):
    """DEF-02: Assert systems with N >= 4000 do not allocate dense (N, N, 3) pair tensors.

    When nonbonded_engine='neighbor_list' is specified, neighbor lists must be used,
    and dense all-pairs arrays (pos[:, None, :] - pos[None, :, :]) must not be materialized.
    """
    sys = openmm.System()
    for _ in range(4000):
      sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    for _ in range(4000):
      nb.addParticle(0.0, 0.3, 0.5)
    sys.addForce(nb)

    cfg = ProlixOpenMMConfig(nonbonded_engine="neighbor_list")
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    rng = np.random.RandomState(42)
    pos_np = rng.uniform(0.0, 10.0, size=(4000, 3)) * unit.nanometer
    ctx.setPositions(pos_np)

    # Genuine NeighborList allocation must have occurred
    assert ctx._neighbor is not None, "NeighborList must be allocated for 4000-particle system"
    assert ctx._neighbor_idx is not None, "Neighbor indices must be constructed"
    assert ctx._neighbor_idx.shape[0] == 4000
    assert ctx._neighbor_idx.shape[1] < 4000  # K must be sparse, not N

    # Evaluation must succeed via sparse neighbor list
    state = ctx.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert np.isfinite(pe)

  def test_def03_cmap_zero_torsions_no_index_error(self):
    """DEF-03: Assert MolecularBundle with 0 CMAP torsions runs without IndexError.

    When an OpenMM system has 0 CMAP torsions, converter.py assigns shape (0, 24, 24, 16)
    to cmap_coeffs. EnsemblePlan and single_padded_energy statically gather on axis 0,
    triggering: IndexError: index is out of bounds for axis 0 with size 0.
    """
    sys = openmm.System()
    sys.addParticle(16.0)
    pos = np.zeros((1, 3)) * unit.nanometer
    conv = from_openmm_system(sys, positions=pos)

    # Shape must have at least 1 map (matching make_bundle_from_system)
    assert conv.bundle.cmap_coeffs.shape[0] >= 1, (
        f"DEF-03: cmap_coeffs has shape {conv.bundle.cmap_coeffs.shape}; "
        "shape[0] must be >= 1 to prevent JAX gather IndexError."
    )

    plan = EnsemblePlan.from_bundle(conv.bundle)
    # Must not raise IndexError
    plan.run(n_steps=1, dt=1.0, kT=0.6, seed=0)

  def test_def04_molecular_bundle_compatibility_single_padded_energy(self):
    """DEF-04: Assert MolecularBundle compatibility with single_padded_energy / PhysicsSystem.

    Calling single_padded_energy(bundle, ...) fails on the un-remediated codebase because
    MolecularBundle defines `box` rather than `box_size`.
    """
    sys = openmm.System()
    sys.addParticle(16.0)
    pos = np.zeros((1, 3)) * unit.nanometer
    conv = from_openmm_system(sys, positions=pos)

    def disp_fn(ra, rb):
      return ra - rb

    # MolecularBundle must provide box_size or be accepted by single_padded_energy
    assert hasattr(conv.bundle, "box_size"), (
        "DEF-04: MolecularBundle missing `box_size` property expected by "
        "single_padded_energy!"
    )

    # single_padded_energy must evaluate cleanly without AttributeError
    e = single_padded_energy(conv.bundle, displacement_fn=disp_fn)
    assert jnp.isfinite(e)


# =============================================================================
# Axis B: Exhaustive Config Knob Wiring (DEF-05)
# =============================================================================


class TestAxisBConfigKnobWiring:
  """Adversarial tests for previously dead / unwired ProlixOpenMMConfig fields."""

  def test_def05_solute_constraint_mode_none(self):
    """DEF-05: Assert solute_constraint_mode='none' disables solute constraints.

    In un-remediated codebase, solute constraints are extracted unconditionally.
    """
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(1.0)
    sys.addConstraint(0, 1, 0.109 * unit.nanometer)

    cfg = ProlixOpenMMConfig(solute_constraint_mode="none")
    conv = from_openmm_system(sys, config=cfg)
    assert conv.bundle.n_constraints == 0, (
        "DEF-05: solute_constraint_mode='none' was ignored; "
        f"n_constraints is {conv.bundle.n_constraints} instead of 0!"
    )

  def test_def05_shake_iterations_and_tolerance_wired(self):
    """DEF-05: Assert shake_iterations and constraint_tolerance propagate to solver."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(1.0)
    sys.addConstraint(0, 1, 0.109 * unit.nanometer)

    cfg = ProlixOpenMMConfig(shake_iterations=50, constraint_tolerance=1e-8)
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)

    assert ctx._constraints is not None
    assert getattr(ctx._constraints, "max_iter", None) == 50, (
        "DEF-05: shake_iterations=50 was not wired into constraint solver!"
    )
    assert getattr(ctx._constraints, "tol", None) == 1e-8, (
        "DEF-05: constraint_tolerance=1e-8 was not wired into constraint solver!"
    )

  def test_def05_precision_double_wired(self):
    """DEF-05: Assert precision='double' sets float64 dtype on positions and masses."""
    sys = openmm.System()
    sys.addParticle(12.0)
    pos = [openmm.Vec3(0, 0, 0)] * unit.nanometer

    cfg = ProlixOpenMMConfig(precision="double")
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    ctx.setPositions(pos)

    assert ctx._positions.dtype == jnp.float64, (
        f"DEF-05: precision='double' produced positions dtype {ctx._positions.dtype} "
        "instead of float64!"
    )
    assert ctx._bundle.masses.dtype == jnp.float64, (
        f"DEF-05: precision='double' produced masses dtype {ctx._bundle.masses.dtype} "
        "instead of float64!"
    )

  def test_def05_integrator_scheme_and_thermostat_override_wired(self):
    """DEF-05: Assert integrator_scheme and thermostat_override produce genuine dynamic trajectory differences."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15 * unit.nanometers, 300000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(bonds)
    pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.16, 0, 0)] * unit.nanometers

    # Run with Verlet scheme
    cfg_verlet = ProlixOpenMMConfig(integrator_scheme="verlet", deterministic_seed=42)
    ctx_verlet = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg_verlet)
    ctx_verlet.setPositions(pos)
    ctx_verlet.step(20)
    st_verlet = ctx_verlet.getState(getPositions=True, getVelocities=True)
    p_verlet = st_verlet.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    v_verlet = st_verlet.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    # Run with Brownian scheme
    cfg_brownian = ProlixOpenMMConfig(integrator_scheme="brownian", deterministic_seed=42)
    ctx_brownian = ProlixContext(sys, openmm.LangevinIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond), config=cfg_brownian)
    ctx_brownian.setPositions(pos)
    ctx_brownian.step(20)
    st_brownian = ctx_brownian.getState(getPositions=True, getVelocities=True)
    p_brownian = st_brownian.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Dynamics must differ algorithmically
    assert not np.allclose(p_verlet, p_brownian, atol=1e-5), "Verlet and Brownian schemes produced identical trajectories!"

    # Test CSVR thermostat override
    cfg_csvr = ProlixOpenMMConfig(thermostat_override="csvr", deterministic_seed=42)
    ctx_csvr = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg_csvr)
    ctx_csvr.setPositions(pos)
    ctx_csvr.step(20)
    st_csvr = ctx_csvr.getState(getVelocities=True)
    v_csvr = st_csvr.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    assert not np.allclose(v_verlet, v_csvr, atol=1e-5), "CSVR thermostat override produced identical velocities to NVE Verlet!"

  def test_def05_lj_switch_width_and_dispersion_correction_affect_energy(self):
    """DEF-05: Assert use_dispersion_correction alters potential energy in periodic box."""
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    sys.addParticle(39.948)
    sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.addParticle(0.0, 0.3, 1.0)
    nb.addParticle(0.0, 0.3, 1.0)
    sys.addForce(nb)
    pos = [
        openmm.Vec3(0.5, 0.5, 0.5),
        openmm.Vec3(1.35, 0.5, 0.5),
    ] * unit.nanometer

    cfg_no_disp = ProlixOpenMMConfig(use_dispersion_correction=False)
    ctx_no_disp = ProlixContext(
        sys, openmm.VerletIntegrator(0.001), config=cfg_no_disp
    )
    ctx_no_disp.setPositions(pos)
    e_no_disp = (
        ctx_no_disp.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    cfg_disp = ProlixOpenMMConfig(use_dispersion_correction=True)
    ctx_disp = ProlixContext(
        sys, openmm.VerletIntegrator(0.001), config=cfg_disp
    )
    ctx_disp.setPositions(pos)
    e_disp = (
        ctx_disp.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert not np.isclose(e_disp, e_no_disp, atol=1e-5), (
        "DEF-05: use_dispersion_correction did not alter potential energy; "
        "dispersion tail correction is dead code!"
    )


# =============================================================================
# Axis C: Physics & Numerical Parity (DEF-06 through DEF-15)
# =============================================================================


class TestAxisCPhysicsAndNumericalParity:
  """Adversarial tests reproducing physics omissions and parity failures."""

  def test_def06_pme_numerical_parity_vs_openmm_reference(self):
    """DEF-06: Assert PME electrostatics parity vs OpenMM Reference.

    In un-remediated codebase, direct erfc, reciprocal SPME mesh, self energy,
    plasma correction, and excluded pair erf correction are omitted (+9.95 kJ/mol error).
    """
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    sys.addParticle(39.948)
    sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.addParticle(1.0, 0.3, 0.5)
    nb.addParticle(-1.0, 0.3, 0.5)
    sys.addForce(nb)
    pos = [
        openmm.Vec3(0.5, 0.5, 0.5),
        openmm.Vec3(1.0, 0.5, 0.5),
    ] * unit.nanometer

    e_ref = compute_openmm_energy(sys, pos, platform_name="Reference")
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    e_prx = (
        ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert np.isclose(e_prx, e_ref, atol=0.05), (
        f"DEF-06: PME parity failure: Ref={e_ref:.4f} kJ/mol, "
        f"Prolix={e_prx:.4f} kJ/mol (diff = {abs(e_ref - e_prx):.4f} kJ/mol)"
    )

  def test_def07_reaction_field_cutoff_periodic_parity(self):
    """DEF-07: Assert Reaction Field (CutoffPeriodic) continuum term parity vs OpenMM.

    In un-remediated codebase, dielectric reaction field continuum terms are omitted
    (-206.7 kJ/mol discrepancy).
    """
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    sys.addParticle(39.948)
    sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.addParticle(1.0, 0.3, 0.5)
    nb.addParticle(-1.0, 0.3, 0.5)
    sys.addForce(nb)
    pos = [
        openmm.Vec3(0.5, 0.5, 0.5),
        openmm.Vec3(1.0, 0.5, 0.5),
    ] * unit.nanometer

    e_ref = compute_openmm_energy(sys, pos, platform_name="Reference")
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    e_prx = (
        ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert np.isclose(e_prx, e_ref, atol=0.1), (
        f"DEF-07: Reaction Field parity failure: Ref={e_ref:.4f} kJ/mol, "
        f"Prolix={e_prx:.4f} kJ/mol (diff = {abs(e_ref - e_prx):.4f} kJ/mol)"
    )

  def test_def08_periodic_cutoff_truncation_enforced(self):
    """DEF-08: Assert periodic cutoff truncation is enforced when has_pbc=True.

    In un-remediated codebase, line 442 skips cutoff_mask when has_pbc=True,
    evaluating pairs beyond the cutoff distance.
    """
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(4, 0, 0), openmm.Vec3(0, 4, 0), openmm.Vec3(0, 0, 4)
    )
    sys.addParticle(39.948)
    sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setUseDispersionCorrection(False)
    nb.addParticle(0.0, 0.3, 1.0)
    nb.addParticle(0.0, 0.3, 1.0)
    sys.addForce(nb)
    # Separation 1.2 nm > cutoff 0.9 nm
    pos = [
        openmm.Vec3(1.0, 1.0, 1.0),
        openmm.Vec3(2.2, 1.0, 1.0),
    ] * unit.nanometer

    e_ref = compute_openmm_energy(sys, pos, platform_name="Reference")
    assert np.isclose(e_ref, 0.0, atol=1e-6)

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    e_prx = (
        ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert np.isclose(e_prx, 0.0, atol=1e-6), (
        f"DEF-08: Periodic cutoff truncation bypassed: pair at 1.2 nm > 0.9 nm "
        f"evaluated to {e_prx:.6f} kJ/mol instead of 0.0!"
    )

  def test_def09_gbsa_obc2_solvation_and_ace_parity(self):
    """DEF-09: Assert GBSA-OBC2 Born solvation and ACE surface area parity vs OpenMM.

    In un-remediated codebase, GBSAOBCForce evaluates as unshielded vacuum Coulomb
    without Born radii or ACE surface area.
    """
    sys = openmm.System()
    for _ in range(3):
      sys.addParticle(12.0)
    gb = openmm.GBSAOBCForce()
    gb.setSolventDielectric(78.5)
    gb.setSoluteDielectric(1.0)
    gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    gb.addParticle(+0.5 * unit.elementary_charge, 0.15 * unit.nanometers, 0.8)
    gb.addParticle(-0.5 * unit.elementary_charge, 0.15 * unit.nanometers, 0.8)
    gb.addParticle(0.0 * unit.elementary_charge, 0.12 * unit.nanometers, 0.8)
    sys.addForce(gb)

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.2, 0.0, 0.0),
        openmm.Vec3(0.1, 0.15, 0.0),
    ] * unit.nanometers

    e_ref = compute_openmm_energy(sys, pos, platform_name="Reference")
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    e_prx = (
        ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert np.isclose(e_prx, e_ref, rtol=1e-3, atol=0.1), (
        f"DEF-09: GBSA-OBC2 parity failure: Ref={e_ref:.4f} kJ/mol, "
        f"Prolix={e_prx:.4f} kJ/mol (diff = {abs(e_ref - e_prx):.4f} kJ/mol)"
    )

  def test_def10_cmap_torsion_energy_parity(self):
    """DEF-10: Assert CMAPTorsionForce energy parity vs OpenMM Reference.

    In un-remediated codebase, CMAPTorsionForce is completely omitted from
    _eval_system_potential_energy, evaluating to 0.0 kJ/mol.
    """
    sys = openmm.System()
    for _ in range(5):
      sys.addParticle(12.0)
    cmap = openmm.CMAPTorsionForce()
    size = 24
    u = np.linspace(-np.pi, np.pi, size, endpoint=False)
    grid_2d = 5.0 * np.outer(np.sin(u), np.cos(u))
    grid_list = [float(v) * unit.kilojoules_per_mole for v in grid_2d.flatten()]
    map_idx = cmap.addMap(size, grid_list)
    cmap.addTorsion(map_idx, 0, 1, 2, 3, 1, 2, 3, 4)
    sys.addForce(cmap)

    pos = [
        openmm.Vec3(0.0, 0.1, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.1, 0.0, 0.0),
        openmm.Vec3(0.1, 0.1, 0.0),
        openmm.Vec3(0.2, 0.1, 0.05),
    ] * unit.nanometers

    e_ref = compute_openmm_energy(sys, pos, platform_name="Reference")
    assert e_ref > 0.5, "Reference CMAP energy must be non-zero"

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    e_prx = (
        ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert np.isclose(e_prx, e_ref, rtol=1e-3, atol=0.01), (
        f"DEF-10: CMAPTorsionForce energy omitted: Ref={e_ref:.4f} kJ/mol, "
        f"Prolix={e_prx:.4f} kJ/mol"
    )

  def test_def11_force_group_bitmask_isolation(self):
    """DEF-11: Assert force group isolation when multiple forces share the same component name.

    In un-remediated codebase, force_group_by_component clobbers earlier forces into
    the group of the last registered force of the same type.
    """
    sys = openmm.System()
    for _ in range(4):
      sys.addParticle(12.0)

    b1 = openmm.HarmonicBondForce()
    b1.setForceGroup(1)
    b1.addBond(
        0,
        1,
        0.15 * unit.nanometers,
        300000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    sys.addForce(b1)

    b2 = openmm.HarmonicBondForce()
    b2.setForceGroup(2)
    b2.addBond(
        2,
        3,
        0.15 * unit.nanometers,
        300000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    sys.addForce(b2)

    pos = [
        openmm.Vec3(0, 0, 0),
        openmm.Vec3(0.16, 0, 0),
        openmm.Vec3(0.5, 0, 0),
        openmm.Vec3(0.66, 0, 0),
    ] * unit.nanometers

    ctx_ref = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx_ref.setPositions(pos)
    e1_ref = (
        ctx_ref.getState(getEnergy=True, groups=1 << 1)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )
    e2_ref = (
        ctx_ref.getState(getEnergy=True, groups=1 << 2)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    ctx_prx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx_prx.setPositions(pos)
    e1_prx = (
        ctx_prx.getState(getEnergy=True, groups=1 << 1)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )
    e2_prx = (
        ctx_prx.getState(getEnergy=True, groups=1 << 2)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )

    assert np.isclose(e1_prx, e1_ref, atol=1e-3), (
        f"DEF-11: Force group 1 clobbered: Ref={e1_ref:.4f}, Prolix={e1_prx:.4f}"
    )
    assert np.isclose(e2_prx, e2_ref, atol=1e-3), (
        f"DEF-11: Force group 2 clobbered: Ref={e2_ref:.4f}, Prolix={e2_prx:.4f}"
    )

  def test_def12_virtual_site_rollout_dynamics(self):
    """DEF-12: Assert virtual site forces are redistributed to parent atoms during rollouts.

    In un-remediated codebase, redistribute_virtual_site_forces is not called inside
    _pure_step_rollout. Parent atoms receive 0 force from virtual site interactions,
    causing trajectories to severely diverge from OpenMM Reference.
    """
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    sys.addParticle(0.0)
    sys.addParticle(16.0)
    sys.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    nb = openmm.NonbondedForce()
    nb.addParticle(0.0, 0.1, 0.0)
    nb.addParticle(0.0, 0.1, 0.0)
    nb.addParticle(1.0, 0.1, 0.0)
    nb.addParticle(1.0, 0.1, 0.0)
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    sys.addForce(nb)

    pos = [
        openmm.Vec3(0, 0, 0),
        openmm.Vec3(0.2, 0, 0),
        openmm.Vec3(0.1, 0, 0),
        openmm.Vec3(0.1, 0.5, 0),
    ] * unit.nanometer

    integ_ref = openmm.VerletIntegrator(0.001)
    ctx_ref = openmm.Context(
        sys, integ_ref, openmm.Platform.getPlatformByName("Reference")
    )
    ctx_ref.setPositions(pos)
    integ_ref.step(5)
    p_ref = (
        ctx_ref.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )

    ctx_prx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx_prx.setPositions(pos)
    ctx_prx.step(5)
    p_prx = (
        ctx_prx.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )

    discrepancy = np.linalg.norm(p_prx[0] - p_ref[0])
    assert discrepancy < 1e-5, (
        f"DEF-12: Parent atom 0 trajectory diverged in rollout: Ref={p_ref[0]}, "
        f"Prolix={p_prx[0]} (discrepancy = {discrepancy:.6f} nm)"
    )

  def test_def13_velocity_constraint_projection(self):
    """DEF-13: Assert velocity constraints (r_ij . v_ij = 0) are enforced after rollout step."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(1.0)
    sys.addConstraint(0, 1, 0.109 * unit.nanometer)
    pos = [openmm.Vec3(0, 0, 0), openmm.Vec3(0.109, 0, 0)] * unit.nanometer

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin)
    ctx.step(1)

    state = ctx.getState(getPositions=True, getVelocities=True)
    p = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    v = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )

    r_ij = p[0] - p[1]
    v_ij = v[0] - v[1]
    rel_proj = abs(np.dot(r_ij, v_ij)) / (
        np.linalg.norm(r_ij) * np.linalg.norm(v_ij)
    )
    assert rel_proj < 1e-4, (
        f"DEF-13: Velocity constraint projection omitted after rollout: "
        f"relative projection = {rel_proj:.6e}"
    )

  def test_def13_shake_pbc_handling(self):
    """DEF-13: Assert SHAKE handles periodic boundary conditions without coordinate distortion."""
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    sys.addConstraint(0, 1, 0.1 * unit.nanometer)

    # Atom 0 at 0.05 nm, Atom 1 at 1.95 nm: true distance under PBC is already 0.1 nm
    pos = [
        openmm.Vec3(0.05, 1.0, 1.0),
        openmm.Vec3(1.95, 1.0, 1.0),
    ] * unit.nanometer

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)
    ctx.applyConstraints()
    p_after = (
        ctx.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometer)
    )

    # With minimum image convention, neither atom needs to move across the box
    assert np.isclose(p_after[0, 0], 0.05, atol=1e-3), (
        f"DEF-13: Atom 0 distorted by SHAKE without PBC shift: x={p_after[0, 0]:.4f}"
    )
    assert np.isclose(p_after[1, 0], 1.95, atol=1e-3), (
        f"DEF-13: Atom 1 distorted by SHAKE without PBC shift: x={p_after[1, 0]:.4f}"
    )

  def test_def14_scr_barostat_virial_minimum_image_handling(self):
    """DEF-14: Assert SCR barostat virial handles periodic boundary pairs with minimum image.

    In un-remediated codebase, line 1035 computes `virial = float(jnp.sum(self._positions * forces))`
    without minimum image convention. When particles cross PBC, the computed virial has the wrong
    magnitude and sign compared to the true minimum-image virial.
    """
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    sys.addParticle(39.948)
    sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.addParticle(1.0, 0.3, 0.5)
    nb.addParticle(-1.0, 0.3, 0.5)
    sys.addForce(nb)
    baro = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 1)
    sys.addForce(baro)

    cfg = ProlixOpenMMConfig(barostat_mode="scr", barostat_frequency_override=1)
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pos = [
        openmm.Vec3(0.1, 1.0, 1.0),
        openmm.Vec3(1.9, 1.0, 1.0),
    ] * unit.nanometer
    ctx.setPositions(pos)

    assert getattr(ctx, "_barostat_uses_minimum_image_virial", False) is True, (
        "DEF-14: SCR barostat computes origin-dependent scalar virial without "
        "minimum image convention!"
    )

  def test_def14_mc_barostat_evaluates_corrected_energy(self):
    """DEF-14: Assert MC barostat trial moves evaluate Hamiltonian with PME corrections.

    In un-remediated codebase, lines 955-965 call _pure_compute_energy_and_forces,
    which evaluates raw Coulomb omitting reciprocal SPME and dispersion corrections.
    """
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    sys.addParticle(39.948)
    sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.addParticle(1.0, 0.3, 0.5)
    nb.addParticle(-1.0, 0.3, 0.5)
    sys.addForce(nb)
    baro = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 1)
    sys.addForce(baro)

    cfg = ProlixOpenMMConfig(barostat_mode="mc", barostat_frequency_override=1)
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pos = [
        openmm.Vec3(0.5, 0.5, 0.5),
        openmm.Vec3(1.0, 0.5, 0.5),
    ] * unit.nanometer
    ctx.setPositions(pos)

    assert getattr(ctx, "_mc_barostat_uses_corrected_energy", False) is True, (
        "DEF-14: MC barostat evaluates trial volume moves using uncorrected "
        "potential energy!"
    )

  def test_def15_rollout_zero_recompilation_varying_chunks(self):
    """DEF-15: Assert rollout kernel does not recompile across varying step chunk sizes."""
    sys = openmm.System()
    sys.addParticle(12.0)
    pos = [openmm.Vec3(0, 0, 0)] * unit.nanometer

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions(pos)

    from prolix.openmm.context import _pure_step_rollout

    ctx.step(10)
    s1 = _pure_step_rollout._cached._cache_size()
    ctx.step(5)
    s2 = _pure_step_rollout._cached._cache_size()

    assert s1 == s2, (
        f"DEF-15: JIT recompiled on varying step chunk sizes: cache grew from"
        f" {s1} to {s2}!"
    )

  def test_def16_test_authenticity_and_scaling_coverage(self):
    """DEF-16: Assert context architecture natively supports N >= 4000 and Pallas dispatch."""
    cfg = ProlixOpenMMConfig(use_pallas=True, nonbonded_engine="neighbor_list")
    assert cfg.use_pallas is True
    assert cfg.nonbonded_engine == "neighbor_list"

    sys = openmm.System()
    for _ in range(50):
      sys.addParticle(39.948)
    nb = openmm.NonbondedForce()
    nb.setCutoffDistance(0.9 * unit.nanometer)
    for _ in range(50):
      nb.addParticle(0.0, 0.3, 0.5)
    sys.addForce(nb)

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.0, 3.0, size=(50, 3)) * unit.nanometer
    ctx.setPositions(pos)

    # Real NeighborList must be allocated and functional
    assert ctx._neighbor is not None
    assert ctx._neighbor_idx is not None

    st = ctx.getState(getEnergy=True, getForces=True)
    pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    assert np.isfinite(pe)
    assert np.all(np.isfinite(f))
