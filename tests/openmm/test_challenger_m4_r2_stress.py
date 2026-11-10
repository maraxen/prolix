"""Empirical Adversarial Stress Test Suite for Milestone 4 Iteration 2.

Challenger M4-R2-1: Drop-In Context and Simulation Adversarial Stress.

Areas stressed:
1. Input validation on setPositions and setVelocities (wrong shape (N, 2), wrong count N != numParticles, NaNs, 1D/3D).
2. Checkpoint corruption resistance (corrupted headers, truncated bytes, missing dictionary keys, shape mismatches).
3. StateDataReporter box volume and density reporting on non-PBC systems with default box vectors (varying aspect ratios).
4. enforcePeriodicBox=True on canonical lower-triangular triclinic boxes comparing against OpenMM Reference Platform (< 10^-6 nm).
5. openmm.XmlSerializer.deserialize() asserting state.getParameters() raises openmm.OpenMMException when unrequested.
"""

from __future__ import annotations

import io
import pickle
from typing import Any
import numpy as np
import openmm
from openmm import app, unit
import pytest

from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation, ProlixState
from prolix.physics.pbc import reduce_triclinic_box_vectors
import jax.numpy as jnp


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================


def _build_non_pbc_test_system(
    box_vectors: tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3],
    masses: list[float] | None = None,
) -> tuple[openmm.System, app.Topology]:
    """Build a non-PBC system with explicit default periodic box vectors."""
    if masses is None:
        masses = [16.0, 12.0, 1.0]

    sys = openmm.System()
    for m in masses:
        sys.addParticle(m)
    sys.setDefaultPeriodicBoxVectors(*box_vectors)

    # Non-periodic forces (HarmonicBondForce only, no PBC nonbonded)
    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 50000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b.addBond(1, 2, 0.11 * unit.nanometer, 40000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("RES", chain)
    top.addAtom("O", app.Element.getBySymbol("O"), res)
    top.addAtom("C", app.Element.getBySymbol("C"), res)
    top.addAtom("H", app.Element.getBySymbol("H"), res)
    atoms = list(top.atoms())
    top.addBond(atoms[0], atoms[1])
    top.addBond(atoms[1], atoms[2])

    return sys, top


def _build_triclinic_test_system(
    a: openmm.Vec3,
    b: openmm.Vec3,
    c: openmm.Vec3,
) -> tuple[openmm.System, app.Topology]:
    """Build a periodic system with canonical lower-triangular triclinic box vectors."""
    sys = openmm.System()
    # Molecule 1: Dimer (0, 1)
    sys.addParticle(16.0)  # 0
    sys.addParticle(16.0)  # 1
    # Molecule 2: Trimer (2, 3, 4)
    sys.addParticle(12.0)  # 2
    sys.addParticle(1.0)   # 3
    sys.addParticle(1.0)   # 4
    # Molecule 3: Monomer (5)
    sys.addParticle(40.0)  # 5

    sys.setDefaultPeriodicBoxVectors(a, b, c)

    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.12 * unit.nanometer, 100000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(2, 3, 0.11 * unit.nanometer, 80000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(2, 4, 0.11 * unit.nanometer, 80000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(bonds)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    for _ in range(6):
        nb.addParticle(0.0, 0.3 * unit.nanometer, 0.5 * unit.kilojoules_per_mole)
    sys.addForce(nb)

    top = app.Topology()
    chain = top.addChain()
    r1 = top.addResidue("M1", chain)
    top.addAtom("O1", app.Element.getBySymbol("O"), r1)
    top.addAtom("O2", app.Element.getBySymbol("O"), r1)
    r2 = top.addResidue("M2", chain)
    top.addAtom("C1", app.Element.getBySymbol("C"), r2)
    top.addAtom("H1", app.Element.getBySymbol("H"), r2)
    top.addAtom("H2", app.Element.getBySymbol("H"), r2)
    r3 = top.addResidue("M3", chain)
    top.addAtom("Ar", app.Element.getBySymbol("Ar"), r3)

    atoms = list(top.atoms())
    top.addBond(atoms[0], atoms[1])
    top.addBond(atoms[2], atoms[3])
    top.addBond(atoms[2], atoms[4])

    return sys, top


# =============================================================================
# Task 2: Stress Test Input Validation on setPositions and setVelocities
# =============================================================================


def test_set_positions_and_velocities_wrong_shapes():
    """Stress test setPositions and setVelocities with 1D, 2D (N, 2), (N, 4), and 3D arrays."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    invalid_shapes = [
        np.zeros((3, 2)),          # (N, 2)
        np.zeros((3, 4)),          # (N, 4)
        np.zeros(9),               # 1D flattened
        np.zeros((3, 3, 1)),       # 3D
        np.zeros((1, 3, 3)),       # 3D
    ]

    for arr in invalid_shapes:
        with pytest.raises(ValueError, match="must have shape"):
            ctx.setPositions(arr)
        with pytest.raises(ValueError, match="must have shape"):
            ctx.setVelocities(arr)


def test_set_positions_and_velocities_wrong_particle_counts():
    """Stress test setPositions and setVelocities with particle counts N != numParticles."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # Context has 4 particles
    bad_counts = [0, 1, 2, 3, 5, 10, 100]

    for count in bad_counts:
        bad_pos = np.zeros((count, 3)) * unit.nanometer
        bad_vel = np.zeros((count, 3)) * (unit.nanometer / unit.picosecond)

        with pytest.raises(openmm.OpenMMException, match="wrong number of positions"):
            ctx.setPositions(bad_pos)

        with pytest.raises(openmm.OpenMMException, match="wrong number of velocities"):
            ctx.setVelocities(bad_vel)


def test_set_positions_and_velocities_nan_and_inf_inputs():
    """Stress test setPositions and setVelocities when inputs contain NaN or Inf values."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(12.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    nan_pos = np.array([[np.nan, 0.0, 1.0], [0.5, np.nan, -0.2]]) * unit.nanometer
    ctx.setPositions(nan_pos)
    st = ctx.getState(getPositions=True)
    pos_out = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.isnan(pos_out[0, 0])
    assert np.isnan(pos_out[1, 1])
    assert not np.isnan(pos_out[0, 2])

    nan_vel = np.array([[10.0, np.nan, 0.0], [np.nan, -5.0, 2.0]]) * (unit.nanometer / unit.picosecond)
    ctx.setVelocities(nan_vel)
    st_v = ctx.getState(getVelocities=True)
    vel_out = st_v.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    assert np.isnan(vel_out[0, 1])
    assert np.isnan(vel_out[1, 0])
    assert not np.isnan(vel_out[0, 0])

    # Inf inputs
    inf_pos = np.array([[np.inf, 0.0, 1.0], [-np.inf, 2.0, 3.0]]) * unit.nanometer
    ctx.setPositions(inf_pos)
    st_inf = ctx.getState(getPositions=True)
    pos_inf = st_inf.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    assert np.isinf(pos_inf[0, 0])
    assert np.isinf(pos_inf[1, 0])


def test_set_positions_with_virtual_sites_and_wrong_counts():
    """Verify setPositions validates against real particles count before virtual site construction."""
    sys = openmm.System()
    sys.addParticle(16.0)  # 0: O
    sys.addParticle(1.0)   # 1: H1
    sys.addParticle(1.0)   # 2: H2
    sys.addParticle(0.0)   # 3: M-site (virtual site)
    sys.setVirtualSite(3, openmm.ThreeParticleAverageSite(0, 1, 2, 0.8, 0.1, 0.1))

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    # Passing 3 positions (real atoms only) when system particle count is 4 must raise
    with pytest.raises(openmm.OpenMMException, match="wrong number of positions"):
        ctx.setPositions(np.zeros((3, 3)) * unit.nanometer)

    # Passing 5 positions when system particle count is 4 must raise
    with pytest.raises(openmm.OpenMMException, match="wrong number of positions"):
        ctx.setPositions(np.zeros((5, 3)) * unit.nanometer)

    # Passing exact 4 positions must succeed and reconstruct virtual site
    valid_pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.1, 0.0, 0.0),
        openmm.Vec3(0.0, 0.1, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),  # dummy for VS
    ] * unit.nanometer
    ctx.setPositions(valid_pos)
    st = ctx.getState(getPositions=True)
    p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    # VS at 0.8*p0 + 0.1*p1 + 0.1*p2 = (0.01, 0.01, 0.0)
    np.testing.assert_allclose(p[3], [0.01, 0.01, 0.0], atol=1e-5)


# =============================================================================
# Task 3: Stress Test Checkpoint Corruption Resistance
# =============================================================================


def test_checkpoint_corruption_header_and_truncated_bytes():
    """Stress test loadCheckpoint with corrupted headers, truncated bytes, and non-pickle data."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]] * unit.nanometer)

    valid_chk = ctx.createCheckpoint()
    assert isinstance(valid_chk, bytes)
    assert len(valid_chk) > 0

    bad_payloads = [
        b"",                                      # Empty bytes
        b"\x00" * 8,                              # Null bytes
        b"BAD_MAGIC_HEADER_12345678",             # Random header
        valid_chk[:10],                           # Truncated header
        valid_chk[: len(valid_chk) // 2],         # Half-truncated
        valid_chk[: len(valid_chk) - 1],          # Truncated final byte
        b"random arbitrary binary string \xff\xfe",
    ]

    for bad in bad_payloads:
        with pytest.raises(openmm.OpenMMException, match="Checkpoint header was not correct"):
            ctx.loadCheckpoint(bad)


def test_checkpoint_missing_required_dictionary_keys():
    """Stress test loadCheckpoint when essential dictionary keys are missing."""
    sys = openmm.System()
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)

    base_dict = {
        "positions": np.zeros((1, 3)),
        "velocities": np.zeros((1, 3)),
        "time": 5.0,
        "step_count": 50,
    }

    # Test removing each required key individually
    for required_key in ["positions", "velocities", "time", "step_count"]:
        corrupted = dict(base_dict)
        del corrupted[required_key]

        # Test both pickled bytes and raw dict
        with pytest.raises(openmm.OpenMMException, match=f"missing key '{required_key}'"):
            ctx.loadCheckpoint(pickle.dumps(corrupted))

        with pytest.raises(openmm.OpenMMException, match=f"missing key '{required_key}'"):
            ctx.loadCheckpoint(corrupted)


def test_checkpoint_shape_mismatches_raises():
    """Stress test loadCheckpoint when positions or velocities have mismatched particle counts or dimensions."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]] * unit.nanometer)

    # N = 3 particles expected
    mismatched_positions = [
        np.zeros((2, 3)),    # N = 2
        np.zeros((4, 3)),    # N = 4
        np.zeros((3, 2)),    # 2D
        np.zeros((3, 4)),    # 4D
        np.zeros(9),         # 1D
    ]

    for bad_pos in mismatched_positions:
        bad_state = {
            "positions": bad_pos,
            "velocities": np.zeros((3, 3)),
            "time": 0.0,
            "step_count": 0,
        }
        with pytest.raises(openmm.OpenMMException, match="invalid positions shape"):
            ctx.loadCheckpoint(bad_state)

    mismatched_velocities = [
        np.zeros((2, 3)),
        np.zeros((4, 3)),
        np.zeros((3, 2)),
        np.zeros(9),
    ]

    for bad_vel in mismatched_velocities:
        bad_state = {
            "positions": np.zeros((3, 3)),
            "velocities": bad_vel,
            "time": 0.0,
            "step_count": 0,
        }
        with pytest.raises(openmm.OpenMMException, match="invalid velocities shape"):
            ctx.loadCheckpoint(bad_state)


def test_checkpoint_roundtrip_state_fidelity():
    """Verify that a valid checkpoint restores positions, velocities, time, and step count bit-for-bit."""
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(12.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)

    init_pos = np.array([[1.234, -0.567, 3.891], [-2.345, 6.789, 0.123]]) * unit.nanometer
    init_vel = np.array([[-0.12, 0.34, -0.56], [0.78, -0.90, 0.11]]) * (unit.nanometer / unit.picosecond)
    ctx.setPositions(init_pos)
    ctx.setVelocities(init_vel)
    ctx.setTime(4.567 * unit.picosecond)
    ctx.setStepCount(1234)

    chk_bytes = ctx.createCheckpoint()

    # Mutate context state
    ctx.setPositions(np.zeros((2, 3)) * unit.nanometer)
    ctx.setVelocities(np.zeros((2, 3)) * (unit.nanometer / unit.picosecond))
    ctx.setTime(0.0 * unit.picosecond)
    ctx.setStepCount(0)

    # Restore from checkpoint
    ctx.loadCheckpoint(chk_bytes)

    st = ctx.getState(getPositions=True, getVelocities=True)
    np.testing.assert_allclose(
        st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        init_pos.value_in_unit(unit.nanometer),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
        init_vel.value_in_unit(unit.nanometer / unit.picosecond),
        rtol=1e-6,
    )
    assert ctx.getTime().value_in_unit(unit.picosecond) == pytest.approx(4.567, rel=1e-6)
    assert ctx.getStepCount() == 1234


# =============================================================================
# Task 4: Stress Test StateDataReporter on Non-PBC Systems
# =============================================================================


@pytest.mark.parametrize(
    ("box_lx", "box_ly", "box_lz"),
    [
        (2.5, 3.5, 4.5),  # 39.375 nm^3
        (5.0, 2.0, 8.0),  # 80.0 nm^3
        (3.0, 3.0, 3.0),  # 27.0 nm^3
        (1.5, 6.0, 4.0),  # 36.0 nm^3
    ],
)
def test_statedatareporter_box_volume_and_density_non_pbc_aspect_ratios(
    box_lx: float, box_ly: float, box_lz: float
):
    """Stress test StateDataReporter on non-PBC systems across varied box dimensions.

    Must compute Volume = Lx * Ly * Lz and exact density M / V, matching OpenMM Reference Platform.
    Must NOT drop to 1.0 nm^3!
    """
    box_vectors = (
        openmm.Vec3(box_lx, 0.0, 0.0),
        openmm.Vec3(0.0, box_ly, 0.0),
        openmm.Vec3(0.0, 0.0, box_lz),
    )
    expected_vol = box_lx * box_ly * box_lz

    # OpenMM Reference
    sys_ref, top_ref = _build_non_pbc_test_system(box_vectors)
    integ_ref = openmm.VerletIntegrator(0.001 * unit.picosecond)
    sim_ref = app.Simulation(top_ref, sys_ref, integ_ref, openmm.Platform.getPlatformByName("Reference"))
    sim_ref.context.setPositions([[0, 0, 0], [0.15, 0, 0], [0.15, 0.11, 0]] * unit.nanometer)

    out_ref = io.StringIO()
    rep_ref = app.StateDataReporter(out_ref, 1, step=True, volume=True, density=True)
    sim_ref.reporters.append(rep_ref)
    sim_ref.step(1)

    # Prolix
    sys_pro, top_pro = _build_non_pbc_test_system(box_vectors)
    integ_pro = openmm.VerletIntegrator(0.001 * unit.picosecond)
    sim_pro = ProlixSimulation(top_pro, sys_pro, integ_pro)
    sim_pro.context.setPositions([[0, 0, 0], [0.15, 0, 0], [0.15, 0.11, 0]] * unit.nanometer)

    out_pro = io.StringIO()
    rep_pro = app.StateDataReporter(out_pro, 1, step=True, volume=True, density=True)
    sim_pro.reporters.append(rep_pro)
    sim_pro.step(1)

    lines_ref = out_ref.getvalue().strip().split("\n")
    lines_pro = out_pro.getvalue().strip().split("\n")

    vol_ref = float(lines_ref[1].split(",")[1])
    density_ref = float(lines_ref[1].split(",")[2])

    vol_pro = float(lines_pro[1].split(",")[1])
    density_pro = float(lines_pro[1].split(",")[2])

    # Assert volume matches expected product of dimensions
    assert vol_ref == pytest.approx(expected_vol, rel=1e-5)
    assert vol_pro == pytest.approx(expected_vol, rel=1e-5), (
        f"Prolix non-PBC volume {vol_pro} nm^3 != expected {expected_vol} nm^3"
    )

    # Assert volume and density match OpenMM Reference
    assert vol_pro == pytest.approx(vol_ref, rel=1e-5)
    assert density_pro == pytest.approx(density_ref, rel=1e-5), (
        f"Prolix non-PBC density {density_pro} != Reference {density_ref}"
    )


# =============================================================================
# Task 5: Stress Test enforcePeriodicBox=True on Triclinic Boxes
# =============================================================================


def test_triclinic_molecular_wrapping_parity_with_openmm_reference():
    """Adversarial stress: enforcePeriodicBox=True on canonical lower-triangular triclinic box.

    Positions span positive and negative cells across multiple unit vectors.
    Must match OpenMM Reference Platform to < 10^-6 nm across all coordinates.
    Molecules must remain unfragmented (bond lengths preserved).
    """
    # Canonical lower-triangular triclinic box vectors
    # a = (3.0, 0, 0)
    # b = (1.0, 3.0, 0)
    # c = (0.5, 1.0, 3.0)
    a = openmm.Vec3(3.0, 0.0, 0.0)
    b = openmm.Vec3(1.0, 3.0, 0.0)
    c = openmm.Vec3(0.5, 1.0, 3.0)

    sys_ref, top_ref = _build_triclinic_test_system(a, b, c)
    sys_pro, top_pro = _build_triclinic_test_system(a, b, c)

    # Construct coordinates where:
    # - Molecule 1 (0, 1): COM is far in positive octant (e.g. +2*a + 3*b + 1*c)
    # - Molecule 2 (2, 3, 4): COM is far in negative octant (e.g. -1*a - 2*b - 3*c)
    # - Molecule 3 (5): Isolated atom exactly across boundary
    r_mol1 = np.array([
        [7.5, 9.95, 3.0],
        [7.5, 10.07, 3.0],
    ])
    r_mol2 = np.array([
        [-4.5, -6.0, -9.0],
        [-4.5, -5.89, -9.0],
        [-4.5, -6.0, -8.89],
    ])
    r_mol3 = np.array([
        [-2.0, -1.0, -3.5],
    ])
    all_pos = np.vstack([r_mol1, r_mol2, r_mol3]) * unit.nanometer

    # OpenMM Reference
    integ_ref = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx_ref = openmm.Context(sys_ref, integ_ref, openmm.Platform.getPlatformByName("Reference"))
    ctx_ref.setPositions(all_pos)
    st_ref = ctx_ref.getState(getPositions=True, enforcePeriodicBox=True)
    pos_ref = st_ref.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Prolix
    integ_pro = openmm.VerletIntegrator(0.001 * unit.picosecond)
    ctx_pro = ProlixContext(sys_pro, integ_pro)
    ctx_pro.setPositions(all_pos)
    st_pro = ctx_pro.getState(getPositions=True, enforcePeriodicBox=True)
    pos_pro = st_pro.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Max coordinate difference must be < 10^-6 nm
    diff = np.abs(pos_ref - pos_pro)
    max_err = float(np.max(diff))
    assert max_err < 1e-6, f"Triclinic wrapping error {max_err} exceeds 1e-6 nm vs OpenMM Reference"

    # Verify molecular integrity (bond lengths intact after wrapping)
    # Dimer bond 0-1
    d_ref_01 = np.linalg.norm(pos_ref[1] - pos_ref[0])
    d_pro_01 = np.linalg.norm(pos_pro[1] - pos_pro[0])
    assert d_pro_01 == pytest.approx(d_ref_01, abs=1e-6)
    assert d_pro_01 == pytest.approx(0.12, abs=1e-4)

    # Trimer bonds 2-3 and 2-4
    d_pro_23 = np.linalg.norm(pos_pro[3] - pos_pro[2])
    d_pro_24 = np.linalg.norm(pos_pro[4] - pos_pro[2])
    assert d_pro_23 == pytest.approx(0.11, abs=1e-4)
    assert d_pro_24 == pytest.approx(0.11, abs=1e-4)


def test_triclinic_reduced_octahedron_wrapping_parity():
    """Stress test triclinic wrapping on truncated octahedron cell reduced to lower-triangular form."""
    # Truncated octahedron box vectors in reduced lower-triangular form
    l_box = 3.5
    a = openmm.Vec3(l_box, 0.0, 0.0)
    b = openmm.Vec3(l_box / 3.0, l_box * np.sqrt(8.0) / 3.0, 0.0)
    c = openmm.Vec3(-l_box / 3.0, l_box * np.sqrt(2.0) / 3.0, l_box * np.sqrt(6.0) / 3.0)

    # Canonicalize via Prolix reduce_triclinic_box_vectors
    raw_box = np.array([
        [l_box, 0.0, 0.0],
        [l_box / 3.0, l_box * np.sqrt(8.0) / 3.0, 0.0],
        [-l_box / 3.0, l_box * np.sqrt(2.0) / 3.0, l_box * np.sqrt(6.0) / 3.0],
    ])
    canon_box = np.asarray(reduce_triclinic_box_vectors(jnp.asarray(raw_box, dtype=jnp.float32)))

    vec_a = openmm.Vec3(float(canon_box[0, 0]), float(canon_box[0, 1]), float(canon_box[0, 2]))
    vec_b = openmm.Vec3(float(canon_box[1, 0]), float(canon_box[1, 1]), float(canon_box[1, 2]))
    vec_c = openmm.Vec3(float(canon_box[2, 0]), float(canon_box[2, 1]), float(canon_box[2, 2]))

    sys_ref, _ = _build_triclinic_test_system(vec_a, vec_b, vec_c)
    sys_pro, _ = _build_triclinic_test_system(vec_a, vec_b, vec_c)

    # Coordinates spanning positive and negative cells across ~2 unit cells
    pos = np.array([
        [5.0, 7.0, 6.0],
        [5.1, 7.0, 6.0],
        [-4.0, -5.0, -6.0],
        [-4.1, -5.0, -6.0],
        [-4.0, -5.1, -6.0],
        [7.0, -3.0, 5.0],
    ]) * unit.nanometer

    ctx_ref = openmm.Context(sys_ref, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx_ref.setPositions(pos)
    pos_ref = ctx_ref.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    ctx_pro = ProlixContext(sys_pro, openmm.VerletIntegrator(0.001))
    ctx_pro.setPositions(pos)
    pos_pro = ctx_pro.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    diff = np.abs(pos_ref - pos_pro)
    assert np.max(diff) < 1e-6, f"Reduced octahedron wrapping difference {np.max(diff)} exceeds 1e-6 nm"


# =============================================================================
# Task 6: Stress Test XmlSerializer Unrequested Parameter Exception Contract
# =============================================================================


def test_xmlserializer_deserialization_unrequested_parameters_raises_openmm_exception():
    """Adversarial stress: Serializing a state with getParameters=False and deserializing via XmlSerializer.

    OpenMM Reference contract:
    - Deserialized State must raise openmm.OpenMMException when getParameters() is invoked.
    - ProlixState must NOT write <Parameters/> or <EnergyParameterDerivatives/> when unrequested.
    """
    sys = openmm.System()
    sys.addParticle(16.0)
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]] * unit.nanometer)

    # Query state with getParameters=False (default) and getParameterDerivatives=False (default)
    state = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)

    # Prior to serialization, ProlixState raises OpenMMException on unrequested parameters
    with pytest.raises(openmm.OpenMMException, match="Invoked getParameters"):
        state.getParameters()
    with pytest.raises(openmm.OpenMMException, match="Invoked getEnergyParameterDerivatives"):
        state.getEnergyParameterDerivatives()

    # Serialize to XML
    xml_str = openmm.XmlSerializer.serialize(state)

    # Verify XML does not contain unrequested tags
    assert "<Parameters" not in xml_str, "XmlSerializer leaked <Parameters> tag for unrequested parameters!"
    assert "<EnergyParameterDerivatives" not in xml_str, "XmlSerializer leaked <EnergyParameterDerivatives> tag!"

    # Deserialize back via OpenMM C++ XmlSerializer
    deserialized = openmm.XmlSerializer.deserialize(xml_str)
    assert isinstance(deserialized, openmm.State)

    # Contract verification: Deserialized state MUST raise OpenMMException
    with pytest.raises(openmm.OpenMMException, match="Invoked getParameters"):
        deserialized.getParameters()

    with pytest.raises(openmm.OpenMMException, match="Invoked getEnergyParameterDerivatives"):
        deserialized.getEnergyParameterDerivatives()


def test_xmlserializer_requested_parameters_roundtrip_contract():
    """Verify that when parameters ARE requested, XmlSerializer preserves them accurately."""
    sys = openmm.System()
    sys.addParticle(16.0)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
    ctx.setParameter("test_lambda", 0.75)

    state = ctx.getState(getParameters=True)
    xml_str = openmm.XmlSerializer.serialize(state)

    assert "<Parameters" in xml_str
    assert "test_lambda=" in xml_str

    deserialized = openmm.XmlSerializer.deserialize(xml_str)
    params = deserialized.getParameters()
    assert "test_lambda" in params
    assert params["test_lambda"] == pytest.approx(0.75, rel=1e-6)
