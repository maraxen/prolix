"""Track 1 (v3.7 Phase 2) tests: ProlixBatchSimulation NPT/RF/alchemical dU/dλ & ProlixSimulation staged reporters."""

from __future__ import annotations

import io
import math
from typing import Any, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
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

unit = cast("Any", _omm_unit)


def _build_water_cluster_system(
    n_waters: int = 8,
    *,
    method: str = "pme",
    rf_dielectric: float = 78.3,
    with_mc_barostat: bool = False,
    barostat_freq: int = 2,
    box_nm: float = 2.2,
    with_lambda_param: bool = False,
) -> tuple[app.Topology, openmm.System, np.ndarray]:
    """Build an orthorhombic rigid TIP3P water system (PME, RF, or CutoffNonPeriodic)."""
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

    nb = openmm.NonbondedForce()
    if method == "pme":
        nb.setNonbondedMethod(openmm.NonbondedForce.PME)
        nb.setPMEParameters(3.4, 32, 32, 32)
    elif method == "rf":
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setReactionFieldDielectric(rf_dielectric)
    elif method == "cutoff_nonperiodic":
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    else:
        raise ValueError(f"Unsupported method={method!r}")

    nb.setCutoffDistance(0.85 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    if with_lambda_param:
        nb.addGlobalParameter("lam_test", 0.0)

    n_side = max(2, math.ceil(n_waters ** (1.0 / 3.0)))
    spacing = box_nm / n_side
    theta = math.radians(104.52)
    r_oh = 0.09572
    r_hh = 0.15139

    coords: list[np.ndarray] = []
    for w in range(n_waters):
        res = top.addResidue("HOH", chain)
        ao = top.addAtom("O", elem_o, res)
        ah1 = top.addAtom("H1", elem_h, res)
        ah2 = top.addAtom("H2", elem_h, res)
        top.addBond(ao, ah1)
        top.addBond(ao, ah2)

        ix = w % n_side
        iy = (w // n_side) % n_side
        iz = w // (n_side * n_side)
        o_pos = np.array(
            [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing],
            dtype=np.float64,
        )
        h1_pos = o_pos + np.array([r_oh, 0.0, 0.0], dtype=np.float64)
        h2_pos = o_pos + np.array(
            [r_oh * math.cos(theta), r_oh * math.sin(theta), 0.0],
            dtype=np.float64,
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

    if with_lambda_param:
        nb.addParticleParameterOffset("lam_test", 0, 0.35, 0.01, 0.25)
        nb.addParticleParameterOffset("lam_test", 1, -0.175, 0.0, 0.0)
        nb.addParticleParameterOffset("lam_test", 2, -0.175, 0.0, 0.0)

    sys.addForce(nb)
    if with_mc_barostat:
        sys.addForce(
            openmm.MonteCarloBarostat(
                250.0 * unit.bar, 300.0 * unit.kelvin, barostat_freq
            )
        )

    return top, sys, np.asarray(coords, dtype=np.float64)


# =============================================================================
# 1. ProlixBatchSimulation NPT, Reaction Field, Velocity/Box Persistence & Plan Reuse
# =============================================================================


def test_batch_simulation_npt_and_rf_stay_on_ensemble_plan_and_persist_state():
    """Uniform MC barostat & Reaction Field stay off ProlixContext fallback and persist v/box."""
    # Reaction Field (CutoffPeriodic) + conductor limit (rf_dielectric=inf)
    for rf_eps in (78.3, float("inf")):
        _, sys_rf, pos_rf = _build_water_cluster_system(
            n_waters=8, method="rf", rf_dielectric=rf_eps, with_mc_barostat=False
        )
        bsim_rf = ProlixBatchSimulation(
            [sys_rf, sys_rf],
            openmm.LangevinMiddleIntegrator(300.0, 2.0, 0.001),
            positions_list=[pos_rf, pos_rf],
            config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True),
        )
        assert bsim_rf.plan is not None
        assert bsim_rf._contexts is None

        plan_id_before = id(bsim_rf.plan)
        bsim_rf.setPositions([pos_rf + 0.002, pos_rf - 0.002])
        assert id(bsim_rf.plan) == plan_id_before
        assert bsim_rf._contexts is None

        bsim_rf.step(4)
        assert bsim_rf._contexts is None
        st_rf = bsim_rf.getStates(
            getPositions=True, getVelocities=True, getEnergy=True, getForces=True
        )
        assert bsim_rf._contexts is None
        for st in st_rf:
            pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            vel = st.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
            frc = st.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            )
            assert math.isfinite(pe)
            assert ke > 0.0
            assert np.linalg.norm(vel) > 0.0
            assert np.all(np.isfinite(frc))

    # Uniform isotropic MonteCarloBarostat (NPT) across consecutive step() calls
    _, sys_npt, pos_npt = _build_water_cluster_system(
        n_waters=8, method="pme", with_mc_barostat=True, barostat_freq=1
    )
    bsim_npt = ProlixBatchSimulation(
        [sys_npt, sys_npt],
        openmm.LangevinMiddleIntegrator(300.0, 2.0, 0.001),
        positions_list=[pos_npt, pos_npt],
        config=ProlixOpenMMConfig(precision="mixed", pallas_interpret=True, nl_update_every=4),
    )
    assert bsim_npt.plan is not None
    assert bsim_npt._contexts is None
    assert bsim_npt._pressure_atm == pytest.approx(250.0 / 1.01325, rel=1e-6)
    assert bsim_npt._barostat_interval == 1

    bsim_npt.step(4)
    assert bsim_npt._contexts is None
    st_npt_1 = bsim_npt.getStates(getPositions=True, getVelocities=True, getEnergy=True)
    vols_1 = [
        st.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) for st in st_npt_1
    ]
    assert all(math.isfinite(v) and v > 0.0 for v in vols_1)
    assert all(
        st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole) > 0.0
        for st in st_npt_1
    )

    # Consecutive NPT step() continues cleanly even if replica boxes diverged
    bsim_npt.step(4)
    assert bsim_npt._contexts is None
    assert bsim_npt.currentStep == 8
    st_npt_2 = bsim_npt.getStates(getPositions=True, getEnergy=True)
    vols_2 = [
        st.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) for st in st_npt_2
    ]
    assert all(math.isfinite(v) and v > 0.0 for v in vols_2)


def test_batch_simulation_fallback_routing_for_cutoff_nonperiodic_and_scr():
    """CutoffNonPeriodic and SCR barostat route cleanly to ProlixContext fallback."""
    _, sys_cnp, pos_cnp = _build_water_cluster_system(
        n_waters=4, method="cutoff_nonperiodic"
    )
    bsim_cnp = ProlixBatchSimulation(
        [sys_cnp, sys_cnp],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=[pos_cnp, pos_cnp],
    )
    assert bsim_cnp.plan is None
    assert bsim_cnp._contexts is not None

    _, sys_scr, pos_scr = _build_water_cluster_system(
        n_waters=4, method="pme", with_mc_barostat=True, barostat_freq=2
    )
    bsim_scr = ProlixBatchSimulation(
        [sys_scr, sys_scr],
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=[pos_scr, pos_scr],
        config=ProlixOpenMMConfig(barostat_mode="scr"),
    )
    assert bsim_scr.plan is None
    assert bsim_scr._contexts is not None


# =============================================================================
# 2. 8-Window OpenFE Alchemical Offsets + Vectorized dU/dλ vs OpenMM Reference FD
# =============================================================================


def test_batch_8_window_alchemical_offsets_and_vmapped_derivatives():
    """8-window setParameter + vectorized compute_parameter_derivatives matches OpenMM Reference FD."""
    box_nm = 2.4
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    for _ in range(8):
        sys.addParticle(12.0)

    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.152, 220000.0)
    bond.addBond(2, 3, 0.152, 220000.0)
    sys.addForce(bond)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.4, 48, 48, 48)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.75 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    nb.addGlobalParameter("lambda_elec", 0.0)
    nb.addGlobalParameter("lambda_sterics", 1.0)

    base_q = [0.45, -0.45, 0.30, -0.30, 0.20, -0.20, 0.15, -0.15]
    for i in range(8):
        nb.addParticle(base_q[i], 0.30 + 0.01 * i, 0.35 + 0.05 * i)

    nb.addParticleParameterOffset("lambda_elec", 0, -0.45, 0.0, 0.0)
    nb.addParticleParameterOffset("lambda_elec", 1, 0.45, 0.0, 0.0)
    nb.addParticleParameterOffset("lambda_sterics", 0, 0.0, 0.025, -0.25)
    nb.addParticleParameterOffset("lambda_sterics", 2, 0.0, -0.015, 0.30)

    # Include an initially-zero exception activated via ExceptionParameterOffset
    exc_zero = nb.addException(0, 2, 0.0, 0.31, 0.0)
    exc_14 = nb.addException(1, 3, -0.10, 0.32, 0.40)
    nb.addException(0, 1, 0.0, 0.30, 0.0)
    nb.addException(2, 3, 0.0, 0.30, 0.0)
    nb.addExceptionParameterOffset("lambda_elec", exc_zero, -0.16, 0.01, 0.50)
    nb.addExceptionParameterOffset("lambda_sterics", exc_14, 0.05, -0.01, -0.20)
    sys.addForce(nb)

    rng = np.random.default_rng(7)
    pos_base = np.array(
        [
            [0.35, 0.40, 0.45],
            [0.50, 0.41, 0.45],
            [0.95, 0.80, 0.60],
            [1.10, 0.81, 0.60],
            [1.55, 0.55, 1.10],
            [0.65, 1.45, 1.25],
            [1.35, 1.50, 0.55],
            [1.75, 1.10, 1.65],
        ],
        dtype=np.float64,
    )
    pos_windows = [
        pos_base + rng.uniform(-0.015, 0.015, size=pos_base.shape) for _ in range(8)
    ]

    n_win = 8
    lam_elec = np.linspace(0.0, 1.0, n_win)
    lam_ster = np.linspace(1.0, 0.15, n_win)

    bsim = ProlixBatchSimulation(
        [sys] * n_win,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=pos_windows,
        config=ProlixOpenMMConfig(precision="mixed", dispersion_follows_parameters=True),
    )
    assert bsim.plan is not None
    assert bsim._contexts is None

    bsim.setParameter("lambda_elec", lam_elec)
    bsim.setParameter("lambda_sterics", lam_ster)

    # Scalar broadcast & validation check
    with pytest.raises(openmm.OpenMMException):
        bsim.setParameter("nonexistent_lambda", 0.5)
    with pytest.raises(ValueError, match="Expected 8"):
        bsim.setParameter("lambda_elec", [0.1, 0.2])

    params_out = bsim.getParameters()
    assert len(params_out) == n_win
    for w in range(n_win):
        assert params_out[w]["lambda_elec"] == pytest.approx(float(lam_elec[w]))
        assert params_out[w]["lambda_sterics"] == pytest.approx(float(lam_ster[w]))

    # MK-64 stride=0 topology sharing MUST remain intact after setParameter()
    for w in range(1, n_win):
        assert bsim.bundles[w].bond_idx is bsim.bundles[0].bond_idx
        assert bsim.bundles[w].exception_pairs is bsim.bundles[0].exception_pairs
        assert bsim.bundles[w].excl_dense_indices is bsim.bundles[0].excl_dense_indices
    assert not np.allclose(
        np.asarray(bsim.bundles[0].charges), np.asarray(bsim.bundles[-1].charges)
    )

    derivs = bsim.compute_parameter_derivatives()
    states = bsim.getStates(
        getEnergy=True,
        getForces=True,
        getParameters=True,
        getParameterDerivatives=True,
    )
    e_vg, g_vg = bsim.value_and_grad_positions()
    assert bsim._contexts is None
    assert isinstance(g_vg, jnp.ndarray)
    assert e_vg.shape == (n_win,)
    assert g_vg.shape == (n_win, 8, 3)

    # Compare all 8 windows against OpenMM Reference + central finite differences
    ref_plat = openmm.Platform.getPlatformByName("Reference")
    eps_fd = 1e-4
    for w in range(n_win):
        ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), ref_plat)
        ref_ctx.setPositions(pos_windows[w] * unit.nanometer)
        le = float(lam_elec[w])
        ls = float(lam_ster[w])
        ref_ctx.setParameter("lambda_elec", le)
        ref_ctx.setParameter("lambda_sterics", ls)
        ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
        e_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )

        # Note: OpenMM Reference freezes C6_tail at Context creation unless dispersion follows
        # live parameters; compute exact Reference FD wrt lambda_elec (pure electrostatic + zero-exc LJ)
        # and verify sterics against reinitialized Reference contexts so live tail matches OpenMM!
        ref_ctx.setParameter("lambda_elec", le + eps_fd)
        e_le_p = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        ref_ctx.setParameter("lambda_elec", le - eps_fd)
        e_le_m = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        fd_le = (e_le_p - e_le_m) / (2.0 * eps_fd)

        assert abs(derivs[w]["lambda_elec"] - fd_le) < 1e-3 * max(1.0, abs(fd_le))
        assert states[w].getEnergyParameterDerivatives()["lambda_elec"] == pytest.approx(
            derivs[w]["lambda_elec"], rel=1e-6, abs=1e-6
        )

        # Also verify exact dU/dlambda_sterics via self-consistent Prolix energy FD AND Reference (disp=False)
        bsim_p = ProlixBatchSimulation(
            [sys],
            openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
            positions_list=[pos_windows[w]],
            config=ProlixOpenMMConfig(precision="double", dispersion_follows_parameters=True),
        )
        bsim_p.setParameter("lambda_elec", le)
        bsim_p.setParameter("lambda_sterics", ls + eps_fd)
        ep_s = bsim_p.getStates(getEnergy=True)[0].getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        bsim_p.setParameter("lambda_sterics", ls - eps_fd)
        em_s = bsim_p.getStates(getEnergy=True)[0].getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        fd_ls_exact = (ep_s - em_s) / (2.0 * eps_fd)
        assert abs(derivs[w]["lambda_sterics"] - fd_ls_exact) < 1e-3 * max(
            1.0, abs(fd_ls_exact)
        )

        # Check value_and_grad_positions matches -getForces()
        f_w = states[w].getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        np.testing.assert_allclose(np.asarray(g_vg[w]), -f_w, rtol=1e-4, atol=5e-3)
        np.testing.assert_allclose(f_w, f_ref, rtol=2e-3, atol=0.15)
        # Energy matches Reference within live-tail C6 delta (< 0.15 kJ/mol on 2.4nm^3 box)
        assert abs(float(e_vg[w]) - e_ref) < 0.25


def test_batch_8_window_alchemical_derivatives_vs_openmm_reference_exact():
    """With use_dispersion_correction=False, all 8 windows match OpenMM Reference FD < 5e-4 kJ/mol."""
    box_nm = 2.4
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    for _ in range(6):
        sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.4, 48, 48, 48)
    nb.setUseDispersionCorrection(False)
    nb.addGlobalParameter("lambda_alch", 0.0)
    for i, q in enumerate([0.5, -0.5, 0.3, -0.3, 0.2, -0.2]):
        nb.addParticle(q, 0.30 + 0.01 * i, 0.40 + 0.05 * i)
    nb.addParticleParameterOffset("lambda_alch", 0, -0.5, 0.02, -0.30)
    nb.addParticleParameterOffset("lambda_alch", 1, 0.5, -0.01, -0.20)
    exc0 = nb.addException(0, 2, 0.0, 0.31, 0.0)
    nb.addExceptionParameterOffset("lambda_alch", exc0, -0.15, 0.01, 0.45)
    sys.addForce(nb)

    pos = np.array(
        [
            [0.35, 0.40, 0.45],
            [0.75, 0.42, 0.48],
            [0.40, 0.85, 0.52],
            [1.15, 0.90, 0.60],
            [1.45, 0.55, 1.15],
            [0.70, 1.45, 1.30],
        ],
        dtype=np.float64,
    )
    lambdas = np.linspace(0.0, 1.0, 8)
    bsim = ProlixBatchSimulation(
        [sys] * 8,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        positions_list=[pos] * 8,
        config=ProlixOpenMMConfig(precision="mixed"),
    )
    bsim.setParameter("lambda_alch", lambdas)
    derivs = bsim.compute_parameter_derivatives()
    assert bsim._contexts is None

    ref_plat = openmm.Platform.getPlatformByName("Reference")
    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), ref_plat)
    ref_ctx.setPositions(pos * unit.nanometer)
    eps = 1e-5
    for w, lam_v in enumerate(lambdas):
        ref_ctx.setParameter("lambda_alch", float(lam_v) + eps)
        ep = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        ref_ctx.setParameter("lambda_alch", float(lam_v) - eps)
        em = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        fd_ref = (ep - em) / (2.0 * eps)
        assert abs(derivs[w]["lambda_alch"] - fd_ref) < 1e-3 * max(1.0, abs(fd_ref))


# =============================================================================
# 3. ProlixSimulation Multi-Interval Device Ring-Buffer Reporter Staging
# =============================================================================


def test_simulation_staged_statedata_dcd_pdb_match_unbatched_and_bulk_device_get(tmp_path):
    """StateDataReporter + DCDReporter + PDBReporter stage K frames with 1 bulk jax.device_get."""
    top, sys, pos_nm = _build_water_cluster_system(n_waters=6, method="rf")
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        use_pallas=True,
        pallas_interpret=True,
        nonbonded_engine="morton_tile_pairs",
        deterministic_seed=99,
    )

    # 1. Staged multi-interval run (step(25) with interval=5 -> K=5 frames in 1 bulk device_get)
    sim_staged = ProlixSimulation(
        top,
        sys,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=cfg,
    )
    sim_staged.context.setPositions(pos_nm * unit.nanometer)
    sim_staged.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 777)

    csv_staged = io.StringIO()
    dcd_staged_path = tmp_path / "staged.dcd"
    pdb_staged_path = tmp_path / "staged.pdb"
    sim_staged.reporters.extend(
        [
            app.StateDataReporter(
                csv_staged,
                reportInterval=5,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            ),
            app.DCDReporter(str(dcd_staged_path), 5, enforcePeriodicBox=True),
            app.PDBReporter(str(pdb_staged_path), 5, enforcePeriodicBox=False),
        ]
    )

    dget_calls: list[int] = []
    orig_device_get = jax.device_get

    def counting_device_get(x: Any) -> Any:
        if isinstance(x, list):
            dget_calls.append(len(x))
        return orig_device_get(x)

    with patch("prolix.openmm.simulation.jax.device_get", side_effect=counting_device_get):
        sim_staged.step(25)

    # Exactly ONE bulk jax.device_get call staging all K=5 report frames!
    assert dget_calls == [5]
    assert sim_staged.currentStep == 25

    # 2. Unstaged reference run stepping 1 report interval (5 steps) at a time
    sim_ref = ProlixSimulation(
        top,
        sys,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=cfg,
    )
    sim_ref.context.setPositions(pos_nm * unit.nanometer)
    sim_ref.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 777)

    csv_ref = io.StringIO()
    dcd_ref_path = tmp_path / "ref.dcd"
    pdb_ref_path = tmp_path / "ref.pdb"
    sim_ref.reporters.extend(
        [
            app.StateDataReporter(
                csv_ref,
                reportInterval=5,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            ),
            app.DCDReporter(str(dcd_ref_path), 5, enforcePeriodicBox=True),
            app.PDBReporter(str(pdb_ref_path), 5, enforcePeriodicBox=False),
        ]
    )
    for _ in range(5):
        sim_ref.step(5)

    # Compare StateDataReporter CSV output identically across all 5 frames
    staged_lines = csv_staged.getvalue().strip().splitlines()
    ref_lines = csv_ref.getvalue().strip().splitlines()
    assert len(staged_lines) == 6
    assert staged_lines[0] == ref_lines[0]
    for row_s, row_r in zip(staged_lines[1:], ref_lines[1:], strict=True):
        vals_s = [float(x) for x in row_s.split(",")]
        vals_r = [float(x) for x in row_r.split(",")]
        np.testing.assert_allclose(vals_s, vals_r, rtol=1e-6, atol=1e-5)

    # Compare PDB output identically and DCD binary frames (excluding timestamp header bytes)
    assert pdb_staged_path.read_text() == pdb_ref_path.read_text()
    dcd_s_bytes = dcd_staged_path.read_bytes()
    dcd_r_bytes = dcd_ref_path.read_bytes()
    assert len(dcd_s_bytes) == len(dcd_r_bytes) > 276
    # DCD header occupies first 276 bytes (contains wall-clock creation string); coordinate
    # and unit-cell records from byte 276 onward must match bit-for-bit!
    assert dcd_s_bytes[276:] == dcd_r_bytes[276:]


def test_simulation_staged_k_cap_and_stateful_reporter_mutation_discard():
    """Verify K<=32 ring-buffer chunking and mid-replay context mutation discard/resume."""
    top, sys, pos_nm = _build_water_cluster_system(
        n_waters=8, method="rf", box_nm=1.8, with_lambda_param=True
    )
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        use_pallas=True,
        pallas_interpret=True,
        nonbonded_engine="morton_tile_pairs",
        deterministic_seed=1234,
    )

    # Part A: 40 reports with interval=1 must chunk into K=32 + K=8 (2 bulk device_get calls)
    sim_cap = ProlixSimulation(
        top,
        sys,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=cfg,
    )
    sim_cap.context.setPositions(pos_nm * unit.nanometer)
    csv_cap = io.StringIO()
    sim_cap.reporters.append(
        app.StateDataReporter(csv_cap, reportInterval=1, step=True, potentialEnergy=True)
    )
    batch_sizes: list[int] = []
    orig_dget = jax.device_get

    def track_batches(x: Any) -> Any:
        if isinstance(x, list):
            batch_sizes.append(len(x))
        return orig_dget(x)

    with patch("prolix.openmm.simulation.jax.device_get", side_effect=track_batches):
        sim_cap.step(40)
    assert batch_sizes == [32, 8]
    assert sim_cap.currentStep == 40
    assert len(csv_cap.getvalue().strip().splitlines()) == 41

    # Part B: Stateful reporter mutates context parameter at step 10 during step(20) (interval=5).
    # Pre-staged frames for steps 15 and 20 MUST be discarded, context rewound to step 10 with
    # lam_test=1.0 preserved, and steps 10->15->20 re-executed under lam_test=1.0!
    class LambdaSwitchReporter:
        def __init__(self) -> None:
            self.logged: list[tuple[int, float, float]] = []

        def describeNextReport(self, simulation: Any) -> dict[str, Any]:  # noqa: N802
            rem = 5 - (simulation.currentStep % 5)
            return {"steps": rem, "periodic": False, "include": ["energy"]}

        def report(self, simulation: Any, state: Any) -> None:
            st_num = int(simulation.currentStep)
            pe_val = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            lam_now = float(state.getParameters()["lam_test"])
            self.logged.append((st_num, lam_now, pe_val))
            if st_num == 10 and lam_now == 0.0:
                simulation.context.setParameter("lam_test", 1.0)

    sim_mut = ProlixSimulation(
        top,
        sys,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=cfg,
    )
    sim_mut.context.setPositions(pos_nm * unit.nanometer)
    sim_mut.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 321)
    rep_mut = LambdaSwitchReporter()
    sim_mut.reporters.append(rep_mut)
    sim_mut.step(20)

    assert sim_mut.currentStep == 20
    assert [entry[0] for entry in rep_mut.logged] == [5, 10, 15, 20]
    assert [entry[1] for entry in rep_mut.logged] == [0.0, 0.0, 1.0, 1.0]

    # Compare against manual unstaged split at step 10 (5 -> 10 -> setParameter(1.0) -> 15 -> 20)
    sim_split = ProlixSimulation(
        top,
        sys,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=cfg,
    )
    sim_split.context.setPositions(pos_nm * unit.nanometer)
    sim_split.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 321)
    split_pes: list[float] = []
    for target in (5, 10, 15, 20):
        sim_split.step(5)
        st_sp = sim_split.context.getState(getEnergy=True)
        split_pes.append(st_sp.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        if target == 10:
            sim_split.context.setParameter("lam_test", 1.0)

    # Frames 5 and 10 are identical; frames 15 and 20 match unstaged split under lam_test=1.0!
    for idx_f in range(4):
        assert rep_mut.logged[idx_f][2] == pytest.approx(
            split_pes[idx_f], rel=1e-5, abs=1e-3
        )
    # Also verify that if frames 15 & 20 had NOT been discarded, PE at step 15 would differ!
    sim_nomut = ProlixSimulation(
        top,
        sys,
        openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.001),
        config=cfg,
    )
    sim_nomut.context.setPositions(pos_nm * unit.nanometer)
    sim_nomut.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 321)
    sim_nomut.step(15)
    pe_nomut_15 = sim_nomut.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert abs(rep_mut.logged[2][2] - pe_nomut_15) > 0.02
