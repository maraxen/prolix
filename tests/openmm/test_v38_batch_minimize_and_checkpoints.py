"""Track 3A (v3.8) tests: vectorized ProlixBatchSimulation.minimizeEnergy, batch serialization, and LJPME/restraints parity."""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import openmm
import openmm.unit as _omm_unit
import pytest
from openmm import app

from prolix.openmm import (
    ProlixBatchSimulation,
    ProlixOpenMMConfig,
    ProlixSimulation,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

unit = cast("Any", _omm_unit)


@pytest.fixture(autouse=True)
def _restore_jax_x64_flag() -> Generator[None]:
    """Restore global jax_enable_x64 after double-precision tests."""
    prev = bool(jax.config.read("jax_enable_x64"))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _build_water_and_solute_system(
    *,
    n_waters: int = 2,
    include_solute_dimer: bool = True,
    method: int = openmm.NonbondedForce.PME,
    restraint_target_nm: tuple[float, float, float] | None = None,
    restraint_k_omm: float = 2500.0,
    with_alchemical_offset: bool = False,
    with_virtual_site: bool = False,
    with_custom_quartic: bool = False,
    box_nm: float = 2.4,
) -> tuple[app.Topology, openmm.System, np.ndarray]:
    """Construct a compact periodic test system with optional rigid waters, SHAKE solute, restraints, LJPME, VS, or Lepton."""
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )

    top = app.Topology()
    top.setUnitCellDimensions(openmm.Vec3(box_nm, box_nm, box_nm) * unit.nanometer)
    chain = top.addChain()
    elem_o = app.Element.getBySymbol("O")
    elem_h = app.Element.getBySymbol("H")
    elem_c = app.Element.getBySymbol("C")

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(method)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setEwaldErrorTolerance(5e-4)
    if method in (openmm.NonbondedForce.PME, openmm.NonbondedForce.LJPME):
        nb.setPMEParameters(3.2, 40, 40, 40)
        nb.setLJPMEParameters(3.0, 20, 20, 20)
    if method == openmm.NonbondedForce.LJPME:
        nb.setUseSwitchingFunction(True)
        nb.setSwitchingDistance(0.75 * unit.nanometer)
        nb.setUseDispersionCorrection(True)
    else:
        nb.setUseDispersionCorrection(False)

    if with_alchemical_offset:
        nb.addGlobalParameter("lambda_vdw", 0.5)

    bonds = openmm.HarmonicBondForce()
    coords: list[np.ndarray] = []

    theta = math.radians(104.52)
    r_oh = 0.09572
    r_hh = 0.15139
    for w in range(n_waters):
        res = top.addResidue("HOH", chain)
        ao = top.addAtom("O", elem_o, res)
        ah1 = top.addAtom("H1", elem_h, res)
        ah2 = top.addAtom("H2", elem_h, res)
        top.addBond(ao, ah1)
        top.addBond(ao, ah2)

        o_pos = np.array([0.45 + 0.65 * w, 0.55, 0.60], dtype=np.float64)
        h1_pos = o_pos + np.array([r_oh, 0.0, 0.0], dtype=np.float64)
        h2_pos = o_pos + np.array(
            [r_oh * math.cos(theta), r_oh * math.sin(theta), 0.0], dtype=np.float64
        )
        coords.extend([o_pos, h1_pos, h2_pos])

        base = 3 * w
        sys.addParticle(15.9994)
        sys.addParticle(1.00794)
        sys.addParticle(1.00794)
        sys.addConstraint(base, base + 1, r_oh * unit.nanometer)
        sys.addConstraint(base, base + 2, r_oh * unit.nanometer)
        sys.addConstraint(base + 1, base + 2, r_hh * unit.nanometer)

        nb.addParticle(-0.834, 0.315075, 0.635968)
        nb.addParticle(0.417, 0.04, 0.0)
        nb.addParticle(0.417, 0.04, 0.0)
        nb.addException(base, base + 1, 0.0, 0.1, 0.0)
        nb.addException(base, base + 2, 0.0, 0.1, 0.0)
        nb.addException(base + 1, base + 2, 0.0, 0.1, 0.0)

    if include_solute_dimer:
        res_s = top.addResidue("SOL", chain)
        c0 = top.addAtom("C1", elem_c, res_s)
        c1 = top.addAtom("C2", elem_c, res_s)
        h_s = top.addAtom("H1", elem_h, res_s)
        top.addBond(c0, c1)
        top.addBond(c1, h_s)

        idx_c0 = len(coords)
        idx_c1 = idx_c0 + 1
        idx_hs = idx_c0 + 2
        sys.addParticle(12.011)
        sys.addParticle(12.011)
        sys.addParticle(1.008)

        coords.append(np.array([1.35, 1.30, 1.25], dtype=np.float64))
        coords.append(np.array([1.50, 1.30, 1.25], dtype=np.float64))
        coords.append(np.array([1.609, 1.30, 1.25], dtype=np.float64))

        bonds.addBond(idx_c0, idx_c1, 0.152 * unit.nanometer, 220000.0)
        # Constrained C-H bond for SHAKE verification
        sys.addConstraint(idx_c1, idx_hs, 0.109 * unit.nanometer)

        nb.addParticle(-0.15, 0.34, 0.45)
        nb.addParticle(0.10, 0.34, 0.45)
        nb.addParticle(0.05, 0.25, 0.06)
        nb.addException(idx_c0, idx_c1, 0.0, 0.34, 0.0)
        nb.addException(idx_c1, idx_hs, 0.0, 0.29, 0.0)
        nb.addException(idx_c0, idx_hs, -0.004, 0.30, 0.12)

        if with_alchemical_offset:
            nb.addParticleParameterOffset("lambda_vdw", idx_c0, 0.0, 0.02, 0.30)

        if with_virtual_site:
            top.addAtom("M", None, res_s)
            idx_vs = len(coords)
            sys.addParticle(0.0)
            sys.setVirtualSite(
                idx_vs, openmm.TwoParticleAverageSite(idx_c0, idx_c1, 0.5, 0.5)
            )
            coords.append(np.array([1.425, 1.30, 1.25], dtype=np.float64))
            nb.addParticle(-0.10, 0.10, 0.0)
            nb.addException(idx_c0, idx_vs, 0.0, 0.1, 0.0)
            nb.addException(idx_c1, idx_vs, 0.0, 0.1, 0.0)
            nb.addException(idx_hs, idx_vs, 0.0, 0.1, 0.0)

        if with_custom_quartic:
            cbf = openmm.CustomBondForce("k_q * (r - r0)^4")
            cbf.addPerBondParameter("k_q")
            cbf.addPerBondParameter("r0")
            cbf.addBond(idx_c0, idx_c1, [80000.0, 0.152])
            sys.addForce(cbf)

    if bonds.getNumBonds() > 0:
        sys.addForce(bonds)
    sys.addForce(nb)

    if restraint_target_nm is not None:
        cef = openmm.CustomExternalForce(
            "0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
        )
        cef.addPerParticleParameter("k")
        cef.addPerParticleParameter("x0")
        cef.addPerParticleParameter("y0")
        cef.addPerParticleParameter("z0")
        tx, ty, tz = restraint_target_nm
        target_idx = 3 * n_waters if include_solute_dimer else 0
        cef.addParticle(target_idx, [restraint_k_omm, tx, ty, tz])
        sys.addForce(cef)

    return top, sys, np.asarray(coords, dtype=np.float64)


# =============================================================================
# 1. Vectorized ProlixBatchSimulation.minimizeEnergy (fire, lbfgs, openmm_cpu & fallbacks)
# =============================================================================


@pytest.mark.parametrize("backend", ["fire", "lbfgs", "openmm_cpu"])
def test_batch_minimize_energy_vmapped_and_cpu_projects_constraints_and_zeros_vel(
    backend: str,
) -> None:
    """Batched minimizeEnergy lowers energy across all B replicas, projects SETTLE/SHAKE, and zeros velocities."""
    _, sys, base_pos = _build_water_and_solute_system(
        n_waters=2, include_solute_dimer=True, method=openmm.NonbondedForce.PME
    )
    rng = np.random.default_rng(19)
    pos_list = []
    for b in range(3):
        p = base_pos.copy()
        # Stretch C1-C2 harmonic bond (indices 6, 7) and perturb constrained H atoms
        p[7, 0] += 0.045 + 0.01 * b
        p[8, 0] += 0.065 + 0.01 * b
        p[1, 1] += 0.018
        p += rng.normal(0.0, 0.004, size=p.shape)
        pos_list.append(p)

    bsim = ProlixBatchSimulation(
        [sys, sys, sys],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=pos_list,
        config=ProlixOpenMMConfig(precision="single", pallas_interpret=True),
    )
    assert bsim.plan is not None
    assert bsim._contexts is None
    assert bsim._can_vmap_batch_observables(-1) is True

    bsim.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)
    pre_states = bsim.getStates(getEnergy=True, getVelocities=True)
    pre_pe = [
        st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        for st in pre_states
    ]
    assert all(
        st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) > 0.0
        for st in pre_states
    )

    bsim.minimizeEnergy(
        tolerance=2.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=250,
        backend=backend,
    )
    # Vectorized GPU & openmm_cpu paths must NOT allocate fallback ProlixContexts
    assert bsim._contexts is None

    post_states = bsim.getStates(
        getPositions=True, getVelocities=True, getEnergy=True, getForces=True
    )
    for b, st in enumerate(post_states):
        pe_after = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke_after = st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        vel_after = st.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        pos_after = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        assert pe_after < pre_pe[b] - 20.0
        assert ke_after == pytest.approx(0.0, abs=1e-12)
        assert np.max(np.abs(vel_after)) == 0.0

        # Verify SETTLE water geometry (O=0, H1=1, H2=2) and SHAKE C-H bond (7, 8)
        r_oh1 = float(np.linalg.norm(pos_after[1] - pos_after[0]))
        r_oh2 = float(np.linalg.norm(pos_after[2] - pos_after[0]))
        r_hh = float(np.linalg.norm(pos_after[2] - pos_after[1]))
        r_ch = float(np.linalg.norm(pos_after[8] - pos_after[7]))
        assert r_oh1 == pytest.approx(0.09572, abs=2e-4)
        assert r_oh2 == pytest.approx(0.09572, abs=2e-4)
        assert r_hh == pytest.approx(0.15139, abs=2e-4)
        assert r_ch == pytest.approx(0.10900, abs=2e-4)


def test_batch_minimize_energy_fallback_with_virtual_sites_and_custom_forces() -> None:
    """Fallback minimizeEnergy reconstructs virtual sites, handles Lepton forces, and syncs _contexts."""
    _, sys_vs, base_pos = _build_water_and_solute_system(
        n_waters=1,
        include_solute_dimer=True,
        method=openmm.NonbondedForce.CutoffPeriodic,
        with_virtual_site=True,
        with_custom_quartic=True,
    )
    p0 = base_pos.copy()
    p1 = base_pos.copy()
    p0[4, 0] += 0.05  # Stretch C1-C2 (indices 3, 4)
    p1[4, 0] -= 0.04
    # Deliberately corrupt virtual site coordinate (index 6) before minimizeEnergy
    p0[6] += np.array([0.2, -0.15, 0.1])
    p1[6] -= np.array([0.1, 0.25, -0.1])

    bsim = ProlixBatchSimulation(
        [sys_vs, sys_vs],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=[p0, p1],
        config=ProlixOpenMMConfig(precision="double", custom_force_mode="lepton_only"),
    )
    assert bsim._contexts is not None
    bsim.setVelocitiesToTemperature(250.0, randomSeed=11)

    e_before = [
        st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        for st in bsim.getStates(getEnergy=True)
    ]
    bsim.minimizeEnergy(tolerance=1.0, maxIterations=200, backend="lbfgs")

    states = bsim.getStates(getPositions=True, getVelocities=True, getEnergy=True)
    for i, st in enumerate(states):
        pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel_nm = st.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        e_after = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert e_after < e_before[i]
        assert np.max(np.abs(vel_nm)) == 0.0
        # Virtual site 6 must sit at exact midpoint of C1 (3) and C2 (4)
        np.testing.assert_allclose(pos_nm[6], 0.5 * (pos_nm[3] + pos_nm[4]), atol=1e-6)
        # Context positions & zeroed velocities must be synced
        ctx_pos = (
            bsim._contexts[i]
            .getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )
        np.testing.assert_allclose(ctx_pos, pos_nm, atol=1e-7)

    with pytest.raises(ValueError, match="Unknown minimizer backend"):
        bsim.minimizeEnergy(backend="cg_unknown")


# =============================================================================
# 2. Batch Serialization: saveStates / loadStates & Checkpoints + Validation
# =============================================================================


def test_batch_save_load_states_and_checkpoints_roundtrip_and_validation(
    tmp_path: Path,
) -> None:
    """Verify saveStates/loadStates, checkpoint round-trips, deterministic continuation, and OpenMMException checks."""
    _, sys_a, pos_a = _build_water_and_solute_system(
        n_waters=2,
        include_solute_dimer=True,
        method=openmm.NonbondedForce.PME,
        with_alchemical_offset=True,
    )
    bsim = ProlixBatchSimulation(
        [sys_a, sys_a, sys_a],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=[pos_a, pos_a + 0.005, pos_a - 0.005],
        config=ProlixOpenMMConfig(
            precision="mixed", pallas_interpret=True, deterministic_seed=123
        ),
    )
    bsim.setParameter("lambda_vdw", [0.1, 0.5, 0.9])
    bsim.setVelocitiesToTemperature(300.0, randomSeed=77)
    bsim.step(2)

    ref_states = bsim.getStates(
        getPositions=True, getVelocities=True, getEnergy=True, getParameters=True
    )

    # 1. Per-replica XML saveStates / loadStates round-trip
    xml_paths = [tmp_path / f"state_rep_{i}.xml" for i in range(3)]
    bsim.saveStates(xml_paths)

    bsim_loaded_xml = ProlixBatchSimulation(
        [sys_a, sys_a, sys_a],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True),
    )
    bsim_loaded_xml.loadStates(xml_paths)
    assert bsim_loaded_xml.currentStep == 2
    got_states_xml = bsim_loaded_xml.getStates(
        getPositions=True, getVelocities=True, getEnergy=True, getParameters=True
    )
    for ref_s, got_s in zip(ref_states, got_states_xml, strict=True):
        np.testing.assert_allclose(
            got_s.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            ref_s.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            got_s.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            ref_s.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            atol=1e-6,
        )
        assert got_s.getParameters()["lambda_vdw"] == pytest.approx(
            ref_s.getParameters()["lambda_vdw"]
        )

    # 2. Combined single-file / stream saveStates & loadStates round-trip
    buf_states = io.StringIO()
    bsim.saveStates(buf_states)
    buf_states.seek(0)
    bsim_loaded_json = ProlixBatchSimulation(
        [sys_a, sys_a, sys_a],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True),
    )
    bsim_loaded_json.loadStates(buf_states)
    assert [p["lambda_vdw"] for p in bsim_loaded_json.getParameters()] == pytest.approx(
        [0.1, 0.5, 0.9]
    )

    # 3. createCheckpoint / saveCheckpoint / loadCheckpoint exact continuation
    chk_bytes = bsim.createCheckpoint()
    chk_path = tmp_path / "batch.chk"
    bsim.saveCheckpoint(chk_path)

    # Advance original simulation by 2 more steps
    bsim.step(2)
    cont_orig = bsim.getStates(getPositions=True, getVelocities=True)

    # Restore into fresh batch simulation from file and advance 2 steps
    bsim_restored = ProlixBatchSimulation(
        [sys_a, sys_a, sys_a],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True),
    )
    bsim_restored.loadCheckpoint(chk_path)
    assert bsim_restored.currentStep == 2
    assert [p["lambda_vdw"] for p in bsim_restored.getParameters()] == pytest.approx(
        [0.1, 0.5, 0.9]
    )
    bsim_restored.step(2)
    cont_restored = bsim_restored.getStates(getPositions=True, getVelocities=True)

    for s_orig, s_rest in zip(cont_orig, cont_restored, strict=True):
        np.testing.assert_allclose(
            s_rest.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            s_orig.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            s_rest.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            s_orig.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            atol=1e-7,
        )

    # 4. Validation errors must raise openmm.OpenMMException on batch_size & atom count mismatch
    bsim_b2 = ProlixBatchSimulation(
        [sys_a, sys_a],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True),
    )
    with pytest.raises(openmm.OpenMMException, match="batch_size"):
        bsim_b2.loadCheckpoint(chk_bytes)
    with pytest.raises(openmm.OpenMMException, match="batch_size"):
        bsim_b2.loadStates(xml_paths)

    _, sys_small, _ = _build_water_and_solute_system(
        n_waters=1, include_solute_dimer=False, method=openmm.NonbondedForce.PME
    )
    bsim_wrong_atoms = ProlixBatchSimulation(
        [sys_a, sys_small, sys_a],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True),
    )
    with pytest.raises(openmm.OpenMMException, match="atom count"):
        bsim_wrong_atoms.loadCheckpoint(chk_bytes)
    with pytest.raises(openmm.OpenMMException, match="atom count"):
        bsim_wrong_atoms.loadStates(xml_paths)
    with pytest.raises(openmm.OpenMMException):
        bsim.loadCheckpoint(b"corrupt-header")


# =============================================================================
# 3. Complete LJPME & PositionRestraints Parity (ProlixSimulation & ProlixBatchSimulation)
# =============================================================================


@pytest.mark.parametrize("use_config_upgrade", [False, True])
@pytest.mark.parametrize("min_backend", ["lbfgs", "fire"])
def test_prolix_simulation_ljpme_minimize_and_staged_reporter_parity(
    use_config_upgrade: bool,
    min_backend: str,
) -> None:
    """ProlixSimulation._minimize_{lbfgs,fire} & _capture_device_report_snapshot honor effective LJPME."""
    omm_method = (
        openmm.NonbondedForce.PME
        if use_config_upgrade
        else openmm.NonbondedForce.LJPME
    )
    top, sys, pos_nm = _build_water_and_solute_system(
        n_waters=1,
        include_solute_dimer=True,
        method=omm_method,
        restraint_target_nm=(1.32, 1.28, 1.24),
    )
    _, sys_ref_ljpme, _ = _build_water_and_solute_system(
        n_waters=1,
        include_solute_dimer=True,
        method=openmm.NonbondedForce.LJPME,
        restraint_target_nm=(1.32, 1.28, 1.24),
    )

    perturbed = pos_nm.copy()
    perturbed[4, 0] += 0.035  # Stretch C1-C2 bond

    cfg = ProlixOpenMMConfig(
        precision="double",
        use_lj_pme=use_config_upgrade,
        minimizer_backend=cast("Any", min_backend),
    )
    sim = ProlixSimulation(
        top,
        sys,
        openmm.VerletIntegrator(0.0005),
        config=cfg,
    )
    sim.context.setPositions(perturbed * unit.nanometer)
    sim.minimizeEnergy(
        tolerance=0.5 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=300,
    )
    min_state = sim.context.getState(getPositions=True, getEnergy=True)
    min_pos = min_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    e_prolix_min = min_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    # Evaluate minimized coordinates in OpenMM Reference LJPME to verify exact energy surface
    ref_ctx = openmm.Context(
        sys_ref_ljpme,
        openmm.VerletIntegrator(0.0005),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ref_ctx.setPositions(min_pos * unit.nanometer)
    e_ref_at_min = (
        ref_ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoules_per_mole)
    )
    np.testing.assert_allclose(e_prolix_min, e_ref_at_min, rtol=5e-4, atol=5e-3)

    # Verify staged device report snapshot (_capture_device_report_snapshot) matches OpenMM Reference LJPME
    captured: list[tuple[int, float, np.ndarray, np.ndarray]] = []

    class _CaptureReporter:
        def describeNextReport(self, simulation: Any) -> tuple[int, bool, bool, bool, bool]:  # noqa: N802
            rem = 2 - (simulation.currentStep % 2)
            return (rem, True, False, True, True)

        def report(self, simulation: Any, state: Any) -> None:
            captured.append(
                (
                    int(simulation.currentStep),
                    float(
                        state.getPotentialEnergy().value_in_unit(
                            unit.kilojoules_per_mole
                        )
                    ),
                    state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoules_per_mole / unit.nanometer
                    ),
                )
            )

    sim.reporters.append(_CaptureReporter())
    assert sim._plan_staged_report_batch(0, 4) is not None
    sim.step(4)
    assert [c[0] for c in captured] == [2, 4]

    for _step_k, pe_rep, pos_rep, frc_rep in captured:
        ref_ctx.setPositions(pos_rep * unit.nanometer)
        ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
        np.testing.assert_allclose(
            pe_rep,
            ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            rtol=1e-3,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            frc_rep,
            ref_st.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            ),
            rtol=5e-3,
            atol=8e-2,
        )


def test_batch_vmapped_ljpme_and_position_restraints_parity_vs_openmm_reference() -> None:
    """ProlixBatchSimulation vmaps uniform-shape PositionRestraints + LJPME + dU/dλ accurately."""
    targets = [
        (-0.15, 1.25, 1.20),  # Includes negative x0 outside [0, L)
        (1.30, 1.35, 1.25),
        (1.42, 1.22, 1.32),
    ]
    k_vals = [1800.0, 2600.0, 3400.0]
    lam_vals = [0.15, 0.50, 0.85]

    systems = []
    pos_list = []
    rng = np.random.default_rng(101)
    for b in range(3):
        _, sys_b, base_p = _build_water_and_solute_system(
            n_waters=1,
            include_solute_dimer=True,
            method=openmm.NonbondedForce.LJPME,
            restraint_target_nm=targets[b],
            restraint_k_omm=k_vals[b],
            with_alchemical_offset=True,
        )
        systems.append(sys_b)
        p_b = base_p.copy() + rng.normal(0.0, 0.015, size=base_p.shape)
        if b == 0:
            # Place solute near negative x0 target for replica 0
            p_b[3:, 0] -= 1.45
        pos_list.append(p_b)

    bsim = ProlixBatchSimulation(
        systems,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=pos_list,
        config=ProlixOpenMMConfig(precision="double"),
    )
    bsim.setParameter("lambda_vdw", lam_vals)
    assert bsim._can_vmap_batch_observables(-1) is True

    states = bsim.getStates(
        getPositions=True,
        getForces=True,
        getEnergy=True,
        getParameters=True,
        getParameterDerivatives=True,
    )
    vg_e, vg_grad = bsim.value_and_grad_positions()
    vg_grad_arr = np.asarray(vg_grad, dtype=np.float64)
    param_derivs = bsim.compute_parameter_derivatives()

    for b in range(3):
        ref_ctx = openmm.Context(
            systems[b],
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        ref_ctx.setPositions(pos_list[b] * unit.nanometer)
        ref_ctx.setParameter("lambda_vdw", lam_vals[b])
        ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
        e_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )

        e_got = states[b].getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        f_got = states[b].getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        np.testing.assert_allclose(e_got, e_ref, rtol=1e-3, atol=2e-2)
        np.testing.assert_allclose(float(vg_e[b]), e_ref, rtol=1e-3, atol=2e-2)
        np.testing.assert_allclose(f_got, f_ref, rtol=5e-3, atol=8e-2)
        np.testing.assert_allclose(-vg_grad_arr[b], f_ref, rtol=5e-3, atol=8e-2)

        # Compare dU/d(lambda_vdw) against OpenMM Reference central finite difference
        dl = 1e-4
        ref_ctx.setParameter("lambda_vdw", lam_vals[b] + dl)
        ep = (
            ref_ctx.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        ref_ctx.setParameter("lambda_vdw", lam_vals[b] - dl)
        em = (
            ref_ctx.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        fd_dlam = (ep - em) / (2.0 * dl)
        assert states[b].getEnergyParameterDerivatives()["lambda_vdw"] == pytest.approx(
            fd_dlam, rel=2e-3, abs=5e-3
        )
        assert param_derivs[b]["lambda_vdw"] == pytest.approx(
            fd_dlam, rel=2e-3, abs=5e-3
        )

    # Also verify vmapped minimizeEnergy works on restrained LJPME batch
    bsim.minimizeEnergy(
        tolerance=2.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=200,
        backend="lbfgs",
    )
    min_states = bsim.getStates(getPositions=True, getEnergy=True)
    for b in range(3):
        assert (
            min_states[b].getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            < float(vg_e[b])
        )
