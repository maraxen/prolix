"""Comprehensive unit tests for ProlixContext and ProlixState."""

from __future__ import annotations

import numpy as np
import openmm
from openmm import unit
import pytest

from prolix.openmm.config import ProlixOpenMMConfig, ProlixPlatform
from prolix.openmm.context import ProlixContext, ProlixState


# =============================================================================
# 1. ProlixState Observable Contracts & XML Serialization Tests
# =============================================================================


def test_prolix_state_observables_and_units():
    """Verify ProlixState returns exact OpenMM Quantity units and numpy formats."""
    pos_nm = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float64)
    vel_nm_ps = np.array([[0.1, -0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=np.float64)
    f_kj_nm = np.array([[-500.0, 0.0, 0.0], [500.0, 0.0, 0.0]], dtype=np.float64)
    box_nm = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    params = {"lambda_test": 0.75}
    derivs = {"lambda_test": 12.5}

    state = ProlixState(
        positions_nm=pos_nm,
        velocities_omm=vel_nm_ps,
        forces_omm=f_kj_nm,
        box_vectors_nm=box_nm,
        time_ps=1.25,
        step_count=500,
        pe_kj=-120.5,
        ke_kj=45.2,
        parameters=params,
        param_derivatives=derivs,
        data_types=0xFF,
    )

    # Time and Step Count
    assert state.getTime().value_in_unit(unit.picosecond) == pytest.approx(1.25)
    assert state.getStepCount() == 500

    # Box Vectors and Volume
    bv = state.getPeriodicBoxVectors()
    assert len(bv) == 3
    assert bv[0].x == pytest.approx(2.0)
    bv_np = state.getPeriodicBoxVectors(asNumpy=True)
    np.testing.assert_allclose(bv_np.value_in_unit(unit.nanometer), box_nm)
    assert state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) == pytest.approx(8.0)

    # Positions
    pos = state.getPositions()
    assert len(pos) == 2
    assert pos[1].x == pytest.approx(0.15)
    pos_arr = state.getPositions(asNumpy=True)
    np.testing.assert_allclose(pos_arr.value_in_unit(unit.nanometer), pos_nm)

    # Velocities
    vel = state.getVelocities()
    assert len(vel) == 2
    assert vel[0].x == pytest.approx(0.1)
    vel_arr = state.getVelocities(asNumpy=True)
    np.testing.assert_allclose(vel_arr.value_in_unit(unit.nanometer / unit.picosecond), vel_nm_ps)

    # Forces
    forces = state.getForces()
    assert len(forces) == 2
    assert forces[0].x == pytest.approx(-500.0)
    forces_arr = state.getForces(asNumpy=True)
    np.testing.assert_allclose(forces_arr.value_in_unit(unit.kilojoules_per_mole / unit.nanometer), f_kj_nm)

    # Energies
    assert state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(-120.5)
    assert state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(45.2)

    # Parameters & Derivatives
    assert state.getParameters() == params
    assert state.getEnergyParameterDerivatives() == derivs
    assert state.getDataTypes() == 0xFF


def test_prolix_state_unrequested_fields_raise():
    """Verify unrequested fields in ProlixState raise exact OpenMMException."""
    empty_state = ProlixState(time_ps=0.0, step_count=0)

    with pytest.raises(openmm.OpenMMException, match="Invoked getPositions"):
        empty_state.getPositions()

    with pytest.raises(openmm.OpenMMException, match="Invoked getVelocities"):
        empty_state.getVelocities()

    with pytest.raises(openmm.OpenMMException, match="Invoked getForces"):
        empty_state.getForces()

    with pytest.raises(openmm.OpenMMException, match="Invoked getPotentialEnergy"):
        empty_state.getPotentialEnergy()

    with pytest.raises(openmm.OpenMMException, match="Invoked getKineticEnergy"):
        empty_state.getKineticEnergy()

    with pytest.raises(openmm.OpenMMException, match="Invoked getParameters"):
        empty_state.getParameters()

    with pytest.raises(openmm.OpenMMException, match="Invoked getEnergyParameterDerivatives"):
        empty_state.getEnergyParameterDerivatives()


def test_prolix_state_xml_serialization_roundtrip():
    """Verify XmlSerializer.serialize(prolix_state) round-trips natively without errors."""
    pos_nm = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    vel_nm_ps = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=np.float64)
    box_nm = np.eye(3, dtype=np.float64) * 3.0
    state = ProlixState(
        positions_nm=pos_nm,
        velocities_omm=vel_nm_ps,
        box_vectors_nm=box_nm,
        time_ps=2.5,
        step_count=1000,
        pe_kj=-250.0,
        ke_kj=80.0,
        parameters={"alchemical": 0.5},
    )

    xml_str = openmm.XmlSerializer.serialize(state)
    assert "<State" in xml_str
    assert 'time="2.5"' in xml_str
    assert 'stepCount="1000"' in xml_str

    deserialized = openmm.XmlSerializer.deserialize(xml_str)
    assert isinstance(deserialized, openmm.State)
    assert deserialized.getTime().value_in_unit(unit.picosecond) == pytest.approx(2.5)
    assert deserialized.getStepCount() == 1000
    assert deserialized.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(-250.0)
    assert deserialized.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(80.0)


# =============================================================================
# 2. Context Initialization & State Management Tests
# =============================================================================


def test_context_getters_and_setters():
    """Verify ProlixContext drop-in state management methods."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(1.0)
    sys.addParticle(1.0)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    config = ProlixOpenMMConfig(deterministic_seed=123)
    ctx = ProlixContext(sys, integ, config=config)

    assert ctx.getSystem() is sys
    assert ctx.getIntegrator() is integ
    assert isinstance(ctx.getPlatform(), ProlixPlatform)

    # Time and Step Count
    ctx.setTime(0.5 * unit.picosecond)
    assert ctx.getTime().value_in_unit(unit.picosecond) == pytest.approx(0.5)
    ctx.setStepCount(250)
    assert ctx.getStepCount() == 250

    # Positions and Velocities
    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.1, 0.0, 0.0),
        openmm.Vec3(0.0, 0.1, 0.0),
    ] * unit.nanometer
    ctx.setPositions(pos)

    state = ctx.getState(getPositions=True)
    pos_ret = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(pos_ret, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])

    vel = [
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ] * (unit.nanometer / unit.picosecond)
    ctx.setVelocities(vel)

    state_vel = ctx.getState(getVelocities=True)
    vel_ret = state_vel.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    np.testing.assert_allclose(vel_ret, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_context_periodic_box_vectors_and_reduction():
    """Verify setPeriodicBoxVectors converts to lower-triangular lattice form."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2.5, 0.0, 0.0),
        openmm.Vec3(0.5, 2.5, 0.0),
        openmm.Vec3(0.2, 0.3, 2.5),
    )
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    a = openmm.Vec3(3.0, 0.0, 0.0) * unit.nanometer
    b = openmm.Vec3(0.2, 3.0, 0.0) * unit.nanometer
    c = openmm.Vec3(0.1, 0.2, 3.0) * unit.nanometer
    ctx.setPeriodicBoxVectors(a, b, c)

    state = ctx.getState()
    bv = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    assert bv[0, 1] == pytest.approx(0.0, abs=1e-7)
    assert bv[0, 2] == pytest.approx(0.0, abs=1e-7)
    assert bv[1, 2] == pytest.approx(0.0, abs=1e-7)


def test_context_set_velocities_to_temperature():
    """Verify setVelocitiesToTemperature samples Gaussian thermal velocities and zeroes COM."""
    sys = openmm.System()
    for _ in range(50):
        sys.addParticle(16.0)

    integ = openmm.VerletIntegrator(0.001)
    config = ProlixOpenMMConfig(remove_com_motion=True, deterministic_seed=42)
    ctx = ProlixContext(sys, integ, config=config)
    ctx.setPositions([[i * 0.1, 0.0, 0.0] for i in range(50)] * unit.nanometer)

    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=100)
    state = ctx.getState(getVelocities=True, getEnergy=True)
    vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    # COM momentum should be zero
    p_com = np.sum(vel * 16.0, axis=0)
    np.testing.assert_allclose(p_com, [0.0, 0.0, 0.0], atol=1e-4)

    # Kinetic energy should be positive and order of 0.5 * DOF * kT
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert ke > 0.0


# =============================================================================
# 3. Virtual Sites and Constraints Tests
# =============================================================================


def test_context_virtual_sites_reconstruction():
    """Verify virtual site positions are computed upon setPositions and computeVirtualSites."""
    sys = openmm.System()
    sys.addParticle(16.0)  # Atom 0
    sys.addParticle(16.0)  # Atom 1
    sys.addParticle(0.0)   # VS 2 (midpoint)
    sys.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # Provide initial positions with virtual site at origin
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ] * unit.nanometer)

    state = ctx.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    # Virtual site should be at midpoint (1.0, 0.0, 0.0)
    np.testing.assert_allclose(pos[2], [1.0, 0.0, 0.0], atol=1e-5)


def test_context_molecules_connected_components():
    """Verify getMolecules returns correct bonded components."""
    sys = openmm.System()
    for _ in range(6):
        sys.addParticle(12.0)

    # Molecule 1: 0-1, 1-2
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    bonds.addBond(1, 2, 0.15, 1000.0)
    # Molecule 2: 3-4, 4-5
    bonds.addBond(3, 4, 0.15, 1000.0)
    bonds.addBond(4, 5, 0.15, 1000.0)
    sys.addForce(bonds)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    mols = ctx.getMolecules()

    assert len(mols) == 2
    assert mols[0] == (0, 1, 2)
    assert mols[1] == (3, 4, 5)


# =============================================================================
# 4. Force Groups & Parameter Derivatives Parity
# =============================================================================


def test_context_force_groups_decomposition():
    """Verify getState(groups=...) correctly isolates force groups with linear superposition."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    # Force Group 1: Bond between 0 and 1
    b1 = openmm.HarmonicBondForce()
    b1.addBond(0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b1.setForceGroup(1)
    sys.addForce(b1)

    # Force Group 2: Custom Bond between 1 and 2
    b2 = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    b2.addPerBondParameter("r0")
    b2.addPerBondParameter("k")
    b2.addBond(1, 2, [0.15, 300000.0])
    b2.setForceGroup(2)
    sys.addForce(b2)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.16, 0.0, 0.0),
        openmm.Vec3(0.33, 0.0, 0.0),
    ] * unit.nanometer)

    # Query Group 1 only
    s1 = ctx.getState(getEnergy=True, getForces=True, groups=1 << 1)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f1 = s1.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

    # Query Group 2 only
    s2 = ctx.getState(getEnergy=True, getForces=True, groups=1 << 2)
    e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f2 = s2.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

    # Query Both Groups (-1)
    s_all = ctx.getState(getEnergy=True, getForces=True, groups=-1)
    e_all = s_all.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_all = s_all.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

    # Verify linear superposition
    assert e_all == pytest.approx(e1 + e2, rel=1e-5)
    np.testing.assert_allclose(f_all, f1 + f2, rtol=1e-5, atol=1e-4)
    # Particle 2 should have zero force under group 1
    np.testing.assert_allclose(f1[2], [0.0, 0.0, 0.0], atol=1e-4)
    # Particle 0 should have zero force under group 2
    np.testing.assert_allclose(f2[0], [0.0, 0.0, 0.0], atol=1e-4)


def test_context_parameter_derivatives_parity():
    """Verify analytical parameter derivatives match OpenMM Reference Platform."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    cf = openmm.CustomBondForce("lambda_weight * k * (r - r0)^2")
    cf.addGlobalParameter("lambda_weight", 0.6)
    cf.addPerBondParameter("k")
    cf.addPerBondParameter("r0")
    cf.addEnergyParameterDerivative("lambda_weight")
    cf.addBond(0, 1, [150.0, 0.15])
    sys.addForce(cf)

    pos = [openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(0.17, 0.0, 0.0)] * unit.nanometer
    integ = openmm.VerletIntegrator(0.001)

    # OpenMM Reference
    omm_ctx = openmm.Context(sys, integ, openmm.Platform.getPlatformByName("Reference"))
    omm_ctx.setPositions(pos)
    omm_state = omm_ctx.getState(getEnergy=True, getParameterDerivatives=True)
    omm_pe = omm_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    omm_deriv = omm_state.getEnergyParameterDerivatives()["lambda_weight"]

    # ProlixContext
    p_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    p_ctx.setPositions(pos)
    p_state = p_ctx.getState(getEnergy=True, getParameterDerivatives=True)
    p_pe = p_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    p_deriv = p_state.getEnergyParameterDerivatives()["lambda_weight"]

    assert p_pe == pytest.approx(omm_pe, rel=1e-4)
    assert p_deriv == pytest.approx(omm_deriv, rel=1e-4)


# =============================================================================
# 5. Checkpoints & Reinitialize Tests
# =============================================================================


def test_context_checkpoint_roundtrip():
    """Verify binary createCheckpoint / loadCheckpoint round-trips state exactly."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    sys.addForce(bonds)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    pos_orig = [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]
    ctx.setPositions(pos_orig * unit.nanometer)
    ctx.setVelocities([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]] * (unit.nanometer / unit.picosecond))
    ctx.setTime(3.5 * unit.picosecond)
    ctx.setStepCount(700)

    chk = ctx.createCheckpoint()
    assert isinstance(chk, bytes)

    # Mutate context state
    ctx.setPositions([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]] * unit.nanometer)
    ctx.setTime(0.0 * unit.picosecond)
    ctx.setStepCount(0)

    # Restore checkpoint
    ctx.loadCheckpoint(chk)

    assert ctx.getTime().value_in_unit(unit.picosecond) == pytest.approx(3.5)
    assert ctx.getStepCount() == 700
    state = ctx.getState(getPositions=True, getVelocities=True)
    np.testing.assert_allclose(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        pos_orig,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        atol=1e-5,
    )


# =============================================================================
# 6. Milestone 4 Iteration 2 Remediation Tests (Defects 1, 2, 3, 4, 5, 7)
# =============================================================================


def test_context_enforce_periodic_box_triclinic_parity():
    """Verify enforcePeriodicBox=True matches OpenMM Reference down to machine precision on triclinic boxes."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    sys.addConstraint(0, 1, 0.1 * unit.nanometer)

    a = openmm.Vec3(2.0, 0.0, 0.0) * unit.nanometer
    b = openmm.Vec3(0.5, 2.0, 0.0) * unit.nanometer
    c = openmm.Vec3(0.3, 0.4, 2.0) * unit.nanometer
    sys.setDefaultPeriodicBoxVectors(a, b, c)

    pos = [
        openmm.Vec3(2.5, 3.1, 4.1),
        openmm.Vec3(2.6, 3.1, 4.1),
    ] * unit.nanometer

    ctx_ref = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx_ref.setPositions(pos)
    s_ref = ctx_ref.getState(getPositions=True, enforcePeriodicBox=True)
    pos_ref = s_ref.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    ctx_pro = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx_pro.setPositions(pos)
    s_pro = ctx_pro.getState(getPositions=True, enforcePeriodicBox=True)
    pos_pro = s_pro.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    max_diff = np.max(np.abs(pos_ref - pos_pro))
    assert max_diff < 1e-6, f"Triclinic wrapping mismatch: max_diff={max_diff} nm"


def test_context_non_pbc_default_box_vectors_and_volume():
    """Verify non-PBC systems retain system.getDefaultPeriodicBoxVectors() and report correct box volume."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(4.0, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 4.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, 4.0) * unit.nanometer,
    )
    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)

    state = ctx.getState(getPositions=True)
    vol = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    assert vol == pytest.approx(64.0, rel=1e-5), f"Expected volume 64.0 nm^3, got {vol}"


def test_context_periodic_torsion_nonzero_phase_parity():
    """Verify PeriodicTorsionForce parity with phase != 0 matches OpenMM Reference."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)
    tor = openmm.PeriodicTorsionForce()
    tor.addTorsion(0, 1, 2, 3, 1, 0.5 * unit.radian, 20.0 * unit.kilojoules_per_mole)
    sys.addForce(tor)

    pos = [
        openmm.Vec3(1.0, 1.0, 0.0),
        openmm.Vec3(0.0, 1.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 1.0),
    ] * unit.nanometer

    ctx_ref = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx_ref.setPositions(pos)
    e_ref = ctx_ref.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    ctx_pro = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx_pro.setPositions(pos)
    e_pro = ctx_pro.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    assert abs(e_ref - e_pro) / e_ref < 1e-5, f"Energy mismatch: e_ref={e_ref}, e_pro={e_pro}"


def test_context_apply_constraints_padding_slice():
    """Verify applyConstraints() on single water does not clobber oxygen atom 0 due to padding."""
    sys = openmm.System()
    for m in [15.999, 1.008, 1.008]:
        sys.addParticle(m)
    sys.addConstraint(0, 1, 0.09572 * unit.nanometer)
    sys.addConstraint(0, 2, 0.09572 * unit.nanometer)
    sys.addConstraint(1, 2, 0.15139 * unit.nanometer)

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001 * unit.picosecond), config=ProlixOpenMMConfig())
    ctx.setPositions([
        openmm.Vec3(0, 0, 0),
        openmm.Vec3(0.1, 0, 0),
        openmm.Vec3(-0.03, 0.09, 0),
    ] * unit.nanometer)
    ctx.applyConstraints()
    p = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    d01 = np.linalg.norm(p[0] - p[1])
    d02 = np.linalg.norm(p[0] - p[2])
    d12 = np.linalg.norm(p[1] - p[2])
    assert abs(d01 - 0.09572) < 1e-5, f"d01={d01}"
    assert abs(d02 - 0.09572) < 1e-5, f"d02={d02}"
    assert abs(d12 - 0.15139) < 1e-5, f"d12={d12}"


def test_context_constrained_rollout_water_bond_stability():
    """Verify constrained rollout maintains water bond geometry within 1e-4 nm."""
    sys = openmm.System()
    for m in [15.999, 1.008, 1.008]:
        sys.addParticle(m)
    sys.addConstraint(0, 1, 0.09572 * unit.nanometer)
    sys.addConstraint(0, 2, 0.09572 * unit.nanometer)
    sys.addConstraint(1, 2, 0.15139 * unit.nanometer)

    integ = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond
    )
    ctx = ProlixContext(sys, integ, config=ProlixOpenMMConfig())
    ctx.setPositions([
        openmm.Vec3(0, 0, 0),
        openmm.Vec3(0.09572, 0, 0),
        openmm.Vec3(-0.024, 0.09266, 0),
    ] * unit.nanometer)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin)
    ctx.step(100)

    p = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    d01 = np.linalg.norm(p[0] - p[1])
    assert abs(d01 - 0.09572) < 1e-4, f"Water bond dilated: d01={d01}"
