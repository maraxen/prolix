"""Adversarial stress test suite for ProlixContext and ProlixState.

Challenger 1 - Milestone 4: State Management & Reporter Interoperability Adversarial Stress.

Areas stressed:
1. StateDataReporter and DCDReporter interoperability, asserting frame data and formatting match OpenMM Reference.
2. openmm.XmlSerializer.serialize(state) and deserialize round-trip preservation of all attributes and unrequested states.
3. Unrequested field exception contracts matching OpenMM Reference.
4. Input validation and checkpoint corruption resistance (NaN, Inf, wrong dimensions, corrupted checkpoints, nested virtual sites).
"""

from __future__ import annotations

import io
import pickle
import tempfile
from typing import Any
import mdtraj as md
import numpy as np
import openmm
from openmm import app, unit
import pytest

from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation, ProlixState


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================


def _build_test_simulation(
    num_particles: int = 2,
    periodic: bool = True,
    box_size_nm: float = 2.0,
    is_prolix: bool = True,
) -> app.Simulation | ProlixSimulation:
    """Build a calibrated test simulation on either Prolix or OpenMM Reference platform."""
    sys = openmm.System()
    for _ in range(num_particles):
        sys.addParticle(12.0)

    box_vec = openmm.Vec3(box_size_nm, 0.0, 0.0)
    box_vec_y = openmm.Vec3(0.0, box_size_nm, 0.0)
    box_vec_z = openmm.Vec3(0.0, 0.0, box_size_nm)
    sys.setDefaultPeriodicBoxVectors(box_vec, box_vec_y, box_vec_z)

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("RES", chain)
    for i in range(num_particles):
        top.addAtom(f"A{i}", app.Element.getBySymbol("C"), res)

    if periodic:
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(0.8 * unit.nanometer)
        for _ in range(num_particles):
            nb.addParticle(0.0, 0.3 * unit.nanometer, 1.0 * unit.kilojoules_per_mole)
        sys.addForce(nb)

    if num_particles >= 2:
        b = openmm.HarmonicBondForce()
        b.addBond(
            0,
            1,
            0.15 * unit.nanometer,
            100000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
        )
        sys.addForce(b)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picosecond,
    )

    positions = [
        openmm.Vec3(float(i) * 0.15, 0.0, 0.0) for i in range(num_particles)
    ] * unit.nanometer

    if is_prolix:
        cfg = ProlixOpenMMConfig(deterministic_seed=42)
        sim = ProlixSimulation(top, sys, integ, config=cfg)
        sim.context.setPositions(positions)
        sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)
        return sim

    ref_platform = openmm.Platform.getPlatformByName("Reference")
    sim = app.Simulation(top, sys, integ, ref_platform)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
    return sim


# =============================================================================
# 1. StateDataReporter and DCDReporter Interoperability Stress
# =============================================================================


def test_statedatareporter_full_headers_formatting_parity():
    """Stress StateDataReporter requesting all headers and compare against OpenMM Reference."""
    sim_ref = _build_test_simulation(num_particles=2, periodic=True, is_prolix=False)
    sim_pro = _build_test_simulation(num_particles=2, periodic=True, is_prolix=True)

    out_ref = io.StringIO()
    rep_ref = app.StateDataReporter(
        out_ref,
        reportInterval=5,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
    )
    sim_ref.reporters.append(rep_ref)
    sim_ref.step(10)

    out_pro = io.StringIO()
    rep_pro = app.StateDataReporter(
        out_pro,
        reportInterval=5,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
    )
    sim_pro.reporters.append(rep_pro)
    sim_pro.step(10)

    lines_ref = [line.strip() for line in out_ref.getvalue().strip().split("\n") if line.strip()]
    lines_pro = [line.strip() for line in out_pro.getvalue().strip().split("\n") if line.strip()]

    # Assert headers match OpenMM Reference exactly
    assert lines_ref[0] == lines_pro[0], f"Header mismatch: {lines_pro[0]} vs {lines_ref[0]}"
    assert len(lines_pro) == len(lines_ref) == 3  # Header + 2 intervals

    # Parse and assert volume and density match Reference
    headers = [h.strip('"# ') for h in lines_pro[0].split(",")]
    vol_idx = headers.index("Box Volume (nm^3)")
    density_idx = headers.index("Density (g/mL)")

    row_ref = [float(v) for v in lines_ref[1].split(",")]
    row_pro = [float(v) for v in lines_pro[1].split(",")]

    assert row_pro[vol_idx] == pytest.approx(row_ref[vol_idx], rel=1e-5), (
        f"Box Volume mismatch in StateDataReporter: Prolix={row_pro[vol_idx]} vs Ref={row_ref[vol_idx]}"
    )
    assert row_pro[density_idx] == pytest.approx(row_ref[density_idx], rel=1e-5), (
        f"Density mismatch in StateDataReporter: Prolix={row_pro[density_idx]} vs Ref={row_ref[density_idx]}"
    )


def test_statedatareporter_box_volume_and_density_on_non_pbc_system():
    """Adversarial stress: StateDataReporter on system with default box vectors but no PBC forces.

    OpenMM Reference retains system.getDefaultPeriodicBoxVectors() on Context, reporting volume and density.
    ProlixContext must not drop box vectors to None / 1.0 nm^3.
    """
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(4.0, 0.0, 0.0),
        openmm.Vec3(0.0, 4.0, 0.0),
        openmm.Vec3(0.0, 0.0, 4.0),
    )
    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 10000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    top = app.Topology()
    c = top.addChain()
    r = top.addResidue("RES", c)
    top.addAtom("A1", app.Element.getBySymbol("C"), r)
    top.addAtom("A2", app.Element.getBySymbol("C"), r)

    # Reference
    integ_ref = openmm.VerletIntegrator(0.001 * unit.picosecond)
    sim_ref = app.Simulation(top, sys, integ_ref, openmm.Platform.getPlatformByName("Reference"))
    sim_ref.context.setPositions([[0, 0, 0], [0.15, 0, 0]] * unit.nanometer)
    out_ref = io.StringIO()
    rep_ref = app.StateDataReporter(out_ref, 1, step=True, volume=True, density=True)
    sim_ref.reporters.append(rep_ref)
    sim_ref.step(1)

    # Prolix
    integ_pro = openmm.VerletIntegrator(0.001 * unit.picosecond)
    sim_pro = ProlixSimulation(top, sys, integ_pro)
    sim_pro.context.setPositions([[0, 0, 0], [0.15, 0, 0]] * unit.nanometer)
    out_pro = io.StringIO()
    rep_pro = app.StateDataReporter(out_pro, 1, step=True, volume=True, density=True)
    sim_pro.reporters.append(rep_pro)
    sim_pro.step(1)

    lines_ref = out_ref.getvalue().strip().split("\n")
    lines_pro = out_pro.getvalue().strip().split("\n")

    vol_ref = float(lines_ref[1].split(",")[1])
    vol_pro = float(lines_pro[1].split(",")[1])

    # Expected 4^3 = 64.0 nm^3
    assert vol_ref == pytest.approx(64.0, rel=1e-5)
    assert vol_pro == pytest.approx(vol_ref, rel=1e-5), (
        f"Non-PBC system Box Volume dropped: Prolix={vol_pro} nm^3 vs Reference={vol_ref} nm^3 (expected 64.0)"
    )


def test_dcdreporter_non_pbc_frames_and_coordinates():
    """Verify DCDReporter outputs exact frames and coordinates on non-periodic simulations."""
    sim = _build_test_simulation(num_particles=2, periodic=False, is_prolix=True)
    sim.context.setPositions([[0.0, 0.0, 0.0], [0.18, 0.0, 0.0]] * unit.nanometer)
    sim.context.setVelocities([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]] * (unit.nanometer / unit.picosecond))

    with tempfile.NamedTemporaryFile(suffix=".dcd") as f_dcd:
        reporter = app.DCDReporter(f_dcd.name, reportInterval=2)
        sim.reporters.append(reporter)
        sim.step(10)
        del reporter

        md_top = md.Topology.from_openmm(sim.topology)
        traj = md.load_dcd(f_dcd.name, top=md_top)

        assert len(traj) == 5
        assert traj.n_atoms == 2
        assert not np.isnan(traj.xyz).any()
        assert not np.isinf(traj.xyz).any()


def test_dcdreporter_frame_count_coordinates_and_box_vectors_mdtraj():
    """Verify DCDReporter outputs exact frames, coordinates, and periodic box vectors read via mdtraj."""
    sim = _build_test_simulation(num_particles=2, periodic=True, box_size_nm=3.0, is_prolix=True)
    sim.context.setPositions([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]] * unit.nanometer)
    sim.context.setVelocities([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]] * (unit.nanometer / unit.picosecond))

    with tempfile.NamedTemporaryFile(suffix=".dcd") as f_dcd:
        reporter = app.DCDReporter(f_dcd.name, reportInterval=2)
        sim.reporters.append(reporter)
        sim.step(10)
        del reporter  # Ensure file handles are closed

        md_top = md.Topology.from_openmm(sim.topology)
        traj = md.load_dcd(f_dcd.name, top=md_top)

        # 10 steps at report interval 2 => exactly 5 frames
        assert len(traj) == 5
        assert traj.n_atoms == 2
        # Verify box lengths in DCD header match 3.0 nm (30.0 Å in mdtraj/dcd format)
        np.testing.assert_allclose(traj.unitcell_lengths[0], [3.0, 3.0, 3.0], rtol=1e-3)
        # Verify coordinates evolve continuously without NaNs
        assert not np.isnan(traj.xyz).any()
        assert not np.isinf(traj.xyz).any()
        # Verify frame 0 is non-zero
        assert np.linalg.norm(traj.xyz[0, 1] - traj.xyz[0, 0]) > 0.1


# =============================================================================
# 2. openmm.XmlSerializer Round-Trip Preservation Stress
# =============================================================================


def test_xmlserializer_full_state_roundtrip_bit_exact():
    """Stress XmlSerializer round-trip on complex ProlixState preserving all attributes."""
    pos = np.array([[0.123456789, -0.987654321, 0.555555555], [1.111111111, 2.222222222, -3.333333333]], dtype=np.float64)
    vel = np.array([[10.1234, -20.5678, 30.9012], [-40.3456, 50.7890, -60.1234]], dtype=np.float64)
    forces = np.array([[-123.456, 789.012, -345.678], [901.234, -567.890, 123.456]], dtype=np.float64)
    box = np.array([[3.5, 0.0, 0.0], [0.2, 3.5, 0.0], [0.1, 0.3, 3.5]], dtype=np.float64)
    params = {"charge_scale": 0.85, "restraint_k": 500.0}

    state = ProlixState(
        positions_nm=pos,
        velocities_omm=vel,
        forces_omm=forces,
        box_vectors_nm=box,
        time_ps=12.34567,
        step_count=98765,
        pe_kj=-9876.5432,
        ke_kj=1234.5678,
        parameters=params,
        data_types=0xFFFFFFFF,
    )

    xml_str = openmm.XmlSerializer.serialize(state)
    assert "<State" in xml_str
    assert 'time="12.34567"' in xml_str
    assert 'stepCount="98765"' in xml_str

    deserialized = openmm.XmlSerializer.deserialize(xml_str)
    assert isinstance(deserialized, openmm.State)

    # Time and StepCount
    assert deserialized.getTime().value_in_unit(unit.picosecond) == pytest.approx(12.34567, rel=1e-9)
    assert deserialized.getStepCount() == 98765

    # Energies
    assert deserialized.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(-9876.5432, rel=1e-9)
    assert deserialized.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) == pytest.approx(1234.5678, rel=1e-9)

    # Box Vectors and Volume
    bv_rt = deserialized.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(bv_rt, box, rtol=1e-9)

    # Positions, Velocities, Forces
    pos_rt = deserialized.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(pos_rt, pos, rtol=1e-9)

    vel_rt = deserialized.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    np.testing.assert_allclose(vel_rt, vel, rtol=1e-9)

    f_rt = deserialized.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    np.testing.assert_allclose(f_rt, forces, rtol=1e-9)

    # Parameters
    p_rt = dict(deserialized.getParameters())
    assert set(p_rt.keys()) == set(params.keys())
    for k in params:
        assert p_rt[k] == pytest.approx(params[k], rel=1e-9)


def test_xmlserializer_unrequested_parameters_contract_preservation():
    """Adversarial stress: Serializing and deserializing a state with NO parameters requested.

    OpenMM Reference: When parameters are unrequested (parameters=False), NO <Parameters> tag is written.
    Upon deserialization, deserialized.getParameters() must raise OpenMMException.
    ProlixState must not emit a spurious <Parameters/> tag that silences this exception!
    """
    state_no_params = ProlixState(time_ps=1.0, step_count=100)

    # ProlixState raises OpenMMException before serialization
    with pytest.raises(openmm.OpenMMException, match="Invoked getParameters"):
        state_no_params.getParameters()

    xml_str = openmm.XmlSerializer.serialize(state_no_params)

    # Assert XML does not contain <Parameters/> when parameters were never requested
    assert "<Parameters" not in xml_str, (
        "XmlSerializer emitted <Parameters/> for unrequested parameters, which corrupts deserialized state contract!"
    )

    deserialized = openmm.XmlSerializer.deserialize(xml_str)
    # After round-trip, unrequested parameters contract MUST be preserved
    with pytest.raises(openmm.OpenMMException, match="Invoked getParameters"):
        deserialized.getParameters()


def test_xmlserializer_unrequested_positions_and_velocities_preservation():
    """Verify deserializing state with unrequested positions/velocities raises OpenMMException."""
    state_energy_only = ProlixState(pe_kj=-100.0, ke_kj=50.0)
    xml_str = openmm.XmlSerializer.serialize(state_energy_only)
    deserialized = openmm.XmlSerializer.deserialize(xml_str)

    with pytest.raises(openmm.OpenMMException, match="Invoked getPositions"):
        deserialized.getPositions()

    with pytest.raises(openmm.OpenMMException, match="Invoked getVelocities"):
        deserialized.getVelocities()

    with pytest.raises(openmm.OpenMMException, match="Invoked getForces"):
        deserialized.getForces()


# =============================================================================
# 3. Unrequested Field Exception Contracts Stress
# =============================================================================


def test_unrequested_field_exceptions_parity_with_reference():
    """Assert all unrequested fields on ProlixState raise exact openmm.OpenMMException with reference messages."""
    empty_state = ProlixState()

    contracts = [
        ("getPositions", "Invoked getPositions() on a State which does not contain positions."),
        ("getVelocities", "Invoked getVelocities() on a State which does not contain velocities."),
        ("getForces", "Invoked getForces() on a State which does not contain forces."),
        ("getPotentialEnergy", "Invoked getPotentialEnergy() on a State which does not contain energies."),
        ("getKineticEnergy", "Invoked getKineticEnergy() on a State which does not contain energies."),
        ("getParameters", "Invoked getParameters() on a State which does not contain parameters."),
        (
            "getEnergyParameterDerivatives",
            "Invoked getEnergyParameterDerivatives() on a State which does not contain parameter derivatives.",
        ),
    ]

    for method_name, expected_msg in contracts:
        with pytest.raises(openmm.OpenMMException) as exc_info:
            getattr(empty_state, method_name)()
        assert expected_msg in str(exc_info.value), f"{method_name} raised '{exc_info.value}' instead of '{expected_msg}'"

    # Also test asNumpy=True on unrequested array fields
    for array_method in ["getPositions", "getVelocities", "getForces"]:
        with pytest.raises(openmm.OpenMMException) as exc_info:
            getattr(empty_state, array_method)(asNumpy=True)
        assert "Invoked" in str(exc_info.value)


# =============================================================================
# 4. Input Validation & Checkpoint Robustness Stress
# =============================================================================


def test_set_positions_and_velocities_wrong_particle_count_raises():
    """Adversarial stress: Passing wrong particle counts to setPositions/setVelocities must raise OpenMMException."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # 1 particle instead of 2
    with pytest.raises((openmm.OpenMMException, ValueError)):
        ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)

    # 3 particles instead of 2
    with pytest.raises((openmm.OpenMMException, ValueError)):
        ctx.setPositions([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]] * unit.nanometer)

    # Wrong particle count in setVelocities
    with pytest.raises((openmm.OpenMMException, ValueError)):
        ctx.setVelocities([[0.0, 0.0, 0.0]] * (unit.nanometer / unit.picosecond))

    with pytest.raises((openmm.OpenMMException, ValueError)):
        ctx.setVelocities(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]] * (unit.nanometer / unit.picosecond)
        )


def test_set_positions_and_velocities_wrong_dimensionality_raises():
    """Adversarial stress: Passing non-3D arrays (e.g. shape (N, 2)) must raise ValueError/OpenMMException."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # 2D coordinates instead of 3D
    with pytest.raises((openmm.OpenMMException, ValueError)):
        ctx.setPositions(np.zeros((2, 2)))

    with pytest.raises((openmm.OpenMMException, ValueError)):
        ctx.setVelocities(np.zeros((2, 2)))


def test_checkpoint_corruption_resistance():
    """Adversarial stress: Corrupted, truncated, or malformed checkpoints must raise OpenMMException."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # OpenMM Reference raises OpenMMException("loadCheckpoint: Checkpoint header was not correct")
    corrupted_payloads = [
        b"",
        b"garbage_not_a_checkpoint",
        b"\x00" * 32,
        pickle.dumps({"positions": np.zeros((2, 3))}),  # Missing velocities/time
        pickle.dumps({"positions": np.zeros((5, 3)), "velocities": np.zeros((5, 3)), "time": 0.0, "step_count": 0}),  # Wrong particle count
    ]

    for bad_chk in corrupted_payloads:
        with pytest.raises(openmm.OpenMMException):
            ctx.loadCheckpoint(bad_chk)


def test_nested_virtual_site_allocation_and_velocity_zeroing():
    """Adversarial stress: Nested virtual sites must not cause particle count double-counting or step crashes.

    System:
    Atom 0: real atom (mass 16.0)
    Atom 1: real atom (mass 16.0)
    VS 2: TwoParticleAverageSite(0, 1, 0.5, 0.5) (mass 0.0)
    VS 3: TwoParticleAverageSite(0, 2, 0.5, 0.5) (mass 0.0) - nested dependency on VS 2!
    """
    sys = openmm.System()
    sys.addParticle(16.0)  # 0
    sys.addParticle(16.0)  # 1
    sys.addParticle(0.0)   # 2
    sys.addParticle(0.0)   # 3

    sys.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    sys.setVirtualSite(3, openmm.TwoParticleAverageSite(0, 2, 0.5, 0.5))

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # Particle count MUST be exactly 4 (not 4 + 2 = 6!)
    assert ctx._n_particles == 4, (
        f"Particle count double-counts virtual sites: ctx._n_particles={ctx._n_particles} vs expected 4"
    )

    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ] * unit.nanometer)

    # Assert nested virtual site reconstruction
    st = ctx.getState(getPositions=True, getVelocities=True)
    pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(pos[0], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(pos[1], [2.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(pos[2], [1.0, 0.0, 0.0], atol=1e-5)   # Midpoint 0 and 1
    np.testing.assert_allclose(pos[3], [0.5, 0.0, 0.0], atol=1e-5)   # Midpoint 0 and 2

    # Assert velocities for all virtual sites are zeroed even after setVelocities
    ctx.setVelocities([[1.0, 1.0, 1.0]] * 4 * (unit.nanometer / unit.picosecond))
    st_vel = ctx.getState(getVelocities=True)
    vel = st_vel.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    np.testing.assert_allclose(vel[2], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vel[3], [0.0, 0.0, 0.0], atol=1e-6)

    # Assert integration step advances without shape broadcasting crash
    ctx.step(5)
    assert ctx.getStepCount() == 5
