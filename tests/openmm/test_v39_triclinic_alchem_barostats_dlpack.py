"""Pallas v3.9 Track 4 tests: triclinic fast-path, fast CustomForces, GPU barostats, HMR, and JAX/DLPack APIs."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import openmm as mm
import pytest
from openmm import app
from openmm import unit as _omm_unit

from prolix.governance.pending_knobs import DOCUMENTED_INERT_KNOBS, PENDING_EPIC_KNOBS
from prolix.governance.redsox_adapter import (
    SUPERSET_ALIAS_MAP,
    check_prolix_reachability,
    check_superset,
    get_reference_config_schemas,
)
from prolix.knob_surface import KnobSurface
from prolix.openmm import (
    ProlixBatchSimulation,
    ProlixContext,
    ProlixOpenMMConfig,
    ProlixSimulation,
)
from prolix.openmm.converter import from_openmm_system
from prolix.openmm.exceptions import (
    UNSUPPORTED_OPENMM_FORCES,
    UnsupportedOpenMMForceError,
)

unit = cast("Any", _omm_unit)


class _StrictDLPackView:
    """Zero-copy DLPack wrapper that forbids host NumPy materialization."""

    def __init__(self, arr: jax.Array) -> None:
        self._arr = arr

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return self._arr.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        return cast("tuple[int, int]", cast("Any", self._arr).__dlpack_device__())

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("DLPack tensor was unexpectedly copied to host NumPy!")


def _triclinic_box_nm(shape: str, edge_nm: float) -> np.ndarray:
    """Return canonical lower-triangular box vectors (nm) matching OpenMM reduced form."""
    d = float(edge_nm)
    if shape == "rhombic_dodecahedron":
        return np.array(
            [
                [d, 0.0, 0.0],
                [0.0, d, 0.0],
                [0.5 * d, 0.5 * d, 0.5 * np.sqrt(2.0) * d],
            ],
            dtype=np.float64,
        )
    if shape == "truncated_octahedron":
        return np.array(
            [
                [d, 0.0, 0.0],
                [d / 3.0, (2.0 * np.sqrt(2.0) / 3.0) * d, 0.0],
                [-d / 3.0, (np.sqrt(2.0) / 3.0) * d, (np.sqrt(6.0) / 3.0) * d],
            ],
            dtype=np.float64,
        )
    return np.diag([d, d, d]).astype(np.float64)


def _build_water_system(
    n_waters: int,
    *,
    shape: str = "cubic",
    edge_nm: float = 2.65,
    cutoff_nm: float = 0.80,
    pme_alpha_nm_inv: float = 3.5,
    pme_grid: int = 24,
    hmr_h_mass_amu: float | None = None,
    seed: int = 101,
) -> tuple[mm.System, app.Topology, np.ndarray, np.ndarray]:
    """Build a non-overlapping rigid-TIP3P water box in cubic, dodecahedron, or octahedron cell."""
    rng = np.random.default_rng(seed)
    box_nm = _triclinic_box_nm(shape, edge_nm)
    sys = mm.System()
    sys.setDefaultPeriodicBoxVectors(
        mm.Vec3(*box_nm[0]) * unit.nanometer,
        mm.Vec3(*box_nm[1]) * unit.nanometer,
        mm.Vec3(*box_nm[2]) * unit.nanometer,
    )
    top = app.Topology()
    chain = top.addChain()
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(cutoff_nm * unit.nanometer)
    nb.setPMEParameters(pme_alpha_nm_inv / unit.nanometer, pme_grid, pme_grid, pme_grid)
    nb.setUseDispersionCorrection(True)

    r_oh = 0.09572
    r_hh = 0.15139
    theta = 104.52 * np.pi / 180.0
    h1_loc = np.array([r_oh, 0.0, 0.0])
    h2_loc = np.array([r_oh * np.cos(theta), r_oh * np.sin(theta), 0.0])

    if hmr_h_mass_amu is not None:
        m_h = float(hmr_h_mass_amu)
        m_o = 18.015 - 2.0 * m_h
    else:
        m_o = 15.999
        m_h = 1.008

    n_side = max(2, int(np.ceil(n_waters ** (1.0 / 3.0))))
    frac_1d = (np.arange(n_side, dtype=np.float64) + 0.5) / float(n_side)
    grid_f = np.stack(np.meshgrid(frac_1d, frac_1d, frac_1d, indexing="ij"), axis=-1).reshape(-1, 3)

    el_o = app.Element.getBySymbol("O")
    el_h = app.Element.getBySymbol("H")
    coords: list[np.ndarray] = []
    for w in range(n_waters):
        res = top.addResidue("HOH", chain)
        ao = top.addAtom("O", el_o, res)
        ah1 = top.addAtom("H1", el_h, res)
        ah2 = top.addAtom("H2", el_h, res)
        top.addBond(ao, ah1)
        top.addBond(ao, ah2)

        base_o = grid_f[w] @ box_nm + rng.uniform(-0.01, 0.01, size=(3,))
        # Perturbed SO(3) orientation via QR avoiding pathological total-energy cancellation
        q_rot, _ = np.linalg.qr(np.eye(3) + rng.normal(scale=0.35, size=(3, 3)))
        if np.linalg.det(q_rot) < 0:
            q_rot[:, 0] *= -1.0
        p_o = base_o
        p_h1 = base_o + h1_loc @ q_rot.T
        p_h2 = base_o + h2_loc @ q_rot.T
        coords.extend([p_o, p_h1, p_h2])

        io = sys.addParticle(m_o * unit.amu)
        ih1 = sys.addParticle(m_h * unit.amu)
        ih2 = sys.addParticle(m_h * unit.amu)
        sys.addConstraint(io, ih1, r_oh * unit.nanometer)
        sys.addConstraint(io, ih2, r_oh * unit.nanometer)
        sys.addConstraint(ih1, ih2, r_hh * unit.nanometer)

        nb.addParticle(-0.834 * unit.elementary_charge, 0.31507524 * unit.nanometer, 0.635968 * unit.kilojoules_per_mole)
        nb.addParticle(0.417 * unit.elementary_charge, 0.04 * unit.nanometer, 0.0 * unit.kilojoules_per_mole)
        nb.addParticle(0.417 * unit.elementary_charge, 0.04 * unit.nanometer, 0.0 * unit.kilojoules_per_mole)
        nb.addException(io, ih1, 0.0, 0.1, 0.0)
        nb.addException(io, ih2, 0.0, 0.1, 0.0)
        nb.addException(ih1, ih2, 0.0, 0.1, 0.0)

    sys.addForce(nb)
    sys.addForce(mm.CMMotionRemover(1))
    return sys, top, np.asarray(coords, dtype=np.float64), box_nm


def _max_water_bond_err_angstrom(pos_nm: np.ndarray, n_waters: int) -> float:
    pos_a = np.asarray(pos_nm, dtype=np.float64) * 10.0
    o = pos_a[0 : 3 * n_waters : 3]
    h1 = pos_a[1 : 3 * n_waters : 3]
    h2 = pos_a[2 : 3 * n_waters : 3]
    d_oh1 = np.max(np.abs(np.linalg.norm(h1 - o, axis=-1) - 0.9572))
    d_oh2 = np.max(np.abs(np.linalg.norm(h2 - o, axis=-1) - 0.9572))
    d_hh = np.max(np.abs(np.linalg.norm(h2 - h1, axis=-1) - 1.5139))
    return float(max(d_oh1, d_oh2, d_hh))


@pytest.mark.parametrize("shape", ["rhombic_dodecahedron", "truncated_octahedron"])
def test_triclinic_context_and_simulation_fastpath_parity_and_wrapping(shape: str) -> None:
    """Reduced-form triclinic boxes run on `_morton_runner` with OpenMM Reference parity & whole-molecule wrapping."""
    n_waters = 48
    sys_mm, top, pos_nm, box_nm = _build_water_system(
        n_waters, shape=shape, edge_nm=2.55, cutoff_nm=0.80, pme_grid=48, seed=301
    )
    cfg = ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs")
    integ_p = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = ProlixSimulation(top, sys_mm, integ_p, config=cfg)
    ctx = sim.context
    assert ctx._is_triclinic is True
    assert ctx._is_reduced_triclinic_supported() is True
    assert ctx._can_use_morton_obs() is True

    # Shift several waters across periodic faces to verify pre-step & post-step whole-molecule wrapping
    pos_shifted = pos_nm.copy()
    pos_shifted[0:3] += 2.0 * box_nm[2] - 1.0 * box_nm[0]
    ctx.setPositions(pos_shifted * unit.nanometer)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)

    # Pre-step getState & getForcesJax must lazily activate `_morton_runner`
    st0 = ctx.getState(getEnergy=True, getForces=True, getPositions=True, enforcePeriodicBox=True)
    assert ctx._morton_runner is not None
    runner_ref = ctx._morton_runner

    e_p = st0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_p = np.asarray(st0.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer))
    f_jax = np.asarray(sim.getForcesJax(in_kj_per_mol_nm=True))
    np.testing.assert_allclose(f_jax, f_p, rtol=1e-5, atol=1e-4)

    ref_plat = mm.Platform.getPlatformByName("Reference")
    ctx_ref = mm.Context(
        sys_mm,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
        ref_plat,
    )
    ctx_ref.setPositions(pos_shifted * unit.nanometer)
    st_ref = ctx_ref.getState(getEnergy=True, getForces=True, getPositions=True, enforcePeriodicBox=True)
    e_ref = st_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = np.asarray(st_ref.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer))

    rel_e = abs(e_p - e_ref) / max(abs(e_ref), 1.0)
    max_f_rel = float(
        np.max(np.linalg.norm(f_p - f_ref, axis=-1))
        / max(float(np.max(np.linalg.norm(f_ref, axis=-1))), 1.0)
    )
    assert rel_e < 3e-4
    assert max_f_rel < 3e-3

    # Verify whole-molecule triclinic wrapping matches OpenMM Reference
    pos_w_state = np.asarray(st0.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    pos_w_jax = np.asarray(sim.getPositionsJax(in_nm=True, enforcePeriodicBox=True))
    pos_w_ref = np.asarray(st_ref.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    np.testing.assert_allclose(pos_w_jax, pos_w_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pos_w_state, pos_w_ref, rtol=1e-5, atol=1e-5)

    # Step on persistent runner without rebuilding
    sim.step(6)
    assert ctx._morton_runner is runner_ref
    pos_after = np.asarray(sim.getPositionsJax(in_nm=True))
    assert _max_water_bond_err_angstrom(pos_after, n_waters) < 1e-5


def test_triclinic_homogeneous_and_cross_shape_ragged_batch_simulation() -> None:
    """Homogeneous and cross-shape ragged triclinic `ProlixBatchSimulation` execute on `EnsemblePlan`."""
    sys_dodec, top_dodec, pos_dodec, _ = _build_water_system(
        20, shape="rhombic_dodecahedron", edge_nm=2.45, seed=401
    )
    cfg = ProlixOpenMMConfig(
        precision="mixed", nonbonded_engine="morton_tile_pairs", deterministic_forces=True
    )
    integ = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)

    # 1. Homogeneous triclinic batch
    batch_h = ProlixBatchSimulation(
        systems=[sys_dodec, sys_dodec],
        integrators=integ,
        topologies=top_dodec,
        positions_list=[pos_dodec * unit.nanometer, pos_dodec * unit.nanometer],
        config=cfg,
    )
    assert batch_h._contexts is None
    assert batch_h.plan is not None
    batch_h.setVelocitiesToTemperature(300.0 * unit.kelvin, 501)
    f_h = batch_h.getForcesJax(in_kj_per_mol_nm=True)
    assert f_h.shape == (2, 60, 3)
    assert batch_h._contexts is None
    batch_h.step(4)
    states_h = batch_h.getStates(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
    assert len(states_h) == 2
    for st in states_h:
        assert np.isfinite(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        assert _max_water_bond_err_angstrom(
            np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)), 20
        ) < 1e-4

    # 2. Cross-shape ragged batch: cubic (16w=48) + dodecahedron (20w=60) + octahedron (24w=72)
    sys_cub, top_cub, pos_cub, _ = _build_water_system(16, shape="cubic", edge_nm=2.25, seed=402)
    sys_oct, top_oct, pos_oct, _ = _build_water_system(24, shape="truncated_octahedron", edge_nm=2.60, seed=403)
    batch_r = ProlixBatchSimulation(
        systems=[sys_cub, sys_dodec, sys_oct],
        integrators=integ,
        topologies=[top_cub, top_dodec, top_oct],
        positions_list=[
            pos_cub * unit.nanometer,
            pos_dodec * unit.nanometer,
            pos_oct * unit.nanometer,
        ],
        config=cfg,
    )
    assert batch_r._contexts is None
    batch_r.setVelocitiesToTemperature(300.0 * unit.kelvin, 502)
    fa = np.asarray(batch_r.getForcesJax(in_kj_per_mol_nm=True))
    fb = np.asarray(batch_r.getForcesJax(in_kj_per_mol_nm=True))
    assert fa.shape == (3, 72, 3)
    np.testing.assert_allclose(fa[0, 48:], 0.0, atol=0.0)
    np.testing.assert_allclose(fa[1, 60:], 0.0, atol=0.0)
    assert float(np.max(np.abs(fa - fb))) == 0.0
    assert batch_r._contexts is None

    batch_r.step(4)
    pos_r = np.asarray(batch_r.getPositionsJax(in_nm=True, enforcePeriodicBox=True))
    assert pos_r.shape == (3, 72, 3)
    assert np.all(np.isfinite(pos_r))


def test_fast_custom_forces_on_morton_gbsa_and_batch_zero_recompile() -> None:
    """Fast `CompiledLeptonForce`s stay on `_morton_runner` & `_gbsa_runner` with 0 recompilations on `setParameter`."""
    n_waters = 27
    sys_alch, top_alch, pos_w_nm, _ = _build_water_system(
        n_waters, shape="cubic", edge_nm=2.35, pme_grid=60, seed=601
    )

    # Append a 2-atom alchemical diatomic ligand molecule so rigid waters stay 3-atom SETTLE waters
    lig_a = sys_alch.addParticle(12.0 * unit.amu)
    lig_b = sys_alch.addParticle(12.0 * unit.amu)
    nb_f = next(
        sys_alch.getForce(i)
        for i in range(sys_alch.getNumForces())
        if isinstance(sys_alch.getForce(i), mm.NonbondedForce)
    )
    nb_f.addParticle(0.0 * unit.elementary_charge, 0.30 * unit.nanometer, 0.0 * unit.kilojoules_per_mole)
    nb_f.addParticle(0.0 * unit.elementary_charge, 0.30 * unit.nanometer, 0.0 * unit.kilojoules_per_mole)
    nb_f.addException(lig_a, lig_b, 0.0, 0.1, 0.0)
    lig_chain = top_alch.addChain()
    lig_res = top_alch.addResidue("LIG", lig_chain)
    at_a = top_alch.addAtom("C1", app.Element.getBySymbol("C"), lig_res)
    at_b = top_alch.addAtom("C2", app.Element.getBySymbol("C"), lig_res)
    top_alch.addBond(at_a, at_b)
    pos_nm = np.vstack(
        [
            pos_w_nm,
            np.array([[0.15, 0.15, 0.15], [0.30, 0.15, 0.15]], dtype=np.float64),
        ]
    )
    n_tot = int(pos_nm.shape[0])

    # Add CustomBondForce, CustomExternalForce, and sparse alchemical CustomNonbondedForce
    cb = mm.CustomBondForce("0.5 * lambda_bond * kb * (r - r0)^2")
    cb.addGlobalParameter("lambda_bond", 1.0)
    cb.addEnergyParameterDerivative("lambda_bond")
    cb.addPerBondParameter("kb")
    cb.addPerBondParameter("r0")
    cb.addBond(lig_a, lig_b, [800.0, 0.15])
    sys_alch.addForce(cb)

    ce = mm.CustomExternalForce("0.5 * k_ext * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    ce.addGlobalParameter("k_ext", 10.0)
    ce.addPerParticleParameter("x0")
    ce.addPerParticleParameter("y0")
    ce.addPerParticleParameter("z0")
    ce.addParticle(lig_a, list(pos_nm[lig_a]))
    sys_alch.addForce(ce)

    cnb = mm.CustomNonbondedForce(
        "lambda_sterics * 4 * eps * ((sig / r_eff)^12 - (sig / r_eff)^6);"
        "r_eff = (0.5 * sig^6 * (1 - lambda_sterics) + r^6)^(1/6);"
        "sig = 0.5 * (sig1 + sig2); eps = sqrt(eps1 * eps2)"
    )
    cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(0.80 * unit.nanometer)
    cnb.setUseLongRangeCorrection(False)
    cnb.addGlobalParameter("lambda_sterics", 1.0)
    cnb.addEnergyParameterDerivative("lambda_sterics")
    cnb.addPerParticleParameter("sig")
    cnb.addPerParticleParameter("eps")
    for idx in range(n_tot):
        is_heavy = idx >= 3 * n_waters or (idx % 3) == 0
        cnb.addParticle([0.315 if is_heavy else 0.10, 0.50 if is_heavy else 0.0])
    lig_grp = {lig_a, lig_b}
    env_grp = set(range(3 * n_waters))
    cnb.addInteractionGroup(lig_grp, env_grp)
    cnb.addExclusion(lig_a, lig_b)
    for w in range(n_waters):
        b = 3 * w
        cnb.addExclusion(b, b + 1)
        cnb.addExclusion(b, b + 2)
        cnb.addExclusion(b + 1, b + 2)
    sys_alch.addForce(cnb)

    cfg = ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs")
    ctx = ProlixContext(
        sys_alch,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
        config=cfg,
    )
    assert len(ctx._custom_forces) == 3
    assert all(cf.is_gpu_runner_fast for cf in ctx._custom_forces)

    ctx.setPositions(pos_nm * unit.nanometer)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 602)
    ctx.step(2)
    runner_initial = ctx._morton_runner
    assert runner_initial is not None

    ref_plat = mm.Platform.getPlatformByName("Reference")
    ctx_ref = mm.Context(
        sys_alch,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
        ref_plat,
    )
    st_cur = ctx.getState(getPositions=True)
    ctx_ref.setPositions(st_cur.getPositions(asNumpy=True))

    for lam in (0.0, 0.5, 1.0):
        ctx.setParameter("lambda_sterics", lam)
        ctx_ref.setParameter("lambda_sterics", lam)
        st_p = ctx.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
        st_r = ctx_ref.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
        assert ctx._morton_runner is runner_initial
        ep = st_p.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        er = st_r.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        np.testing.assert_allclose(ep, er, rtol=5e-4, atol=2e-2)
        dp = st_p.getEnergyParameterDerivatives()["lambda_sterics"]
        dr = st_r.getEnergyParameterDerivatives()["lambda_sterics"]
        np.testing.assert_allclose(dp, dr, rtol=2e-3, atol=5e-2)
        ctx.step(2)
        assert ctx._morton_runner is runner_initial
        ctx_ref.setPositions(ctx.getState(getPositions=True).getPositions(asNumpy=True))

    # Multi-replica batch sharing 1 compiled runner + both `derivs[i][k]` and `derivs[k]` indexing
    lambdas = [0.2, 0.5, 0.8]
    batch = ProlixBatchSimulation(
        systems=[sys_alch] * 3,
        integrators=mm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        ),
        topologies=top_alch,
        positions_list=[pos_nm * unit.nanometer] * 3,
        config=cfg,
    )
    batch.setParameter("lambda_sterics", lambdas)
    batch.setVelocitiesToTemperature(300.0 * unit.kelvin, 603)
    batch.step(2)
    assert batch._contexts is not None
    assert batch._contexts[0]._morton_runner is not None
    assert batch._contexts[0]._morton_runner is batch._contexts[1]._morton_runner
    assert batch._contexts[1]._morton_runner is batch._contexts[2]._morton_runner

    derivs = batch.compute_parameter_derivatives()
    arr_d = derivs["lambda_sterics"]
    assert isinstance(arr_d, np.ndarray)
    assert arr_d.shape == (3,)
    for i in range(3):
        assert np.isclose(arr_d[i], derivs[i]["lambda_sterics"])

    # Implicit-solvent GBSA runner with fast CustomBondForce
    sys_gb = mm.System()
    gb = mm.GBSAOBCForce()
    nb_gb = mm.NonbondedForce()
    nb_gb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    rng = np.random.default_rng(604)
    pos_gb = np.zeros((16, 3), dtype=np.float64)
    for i in range(16):
        pos_gb[i] = np.array([0.38 * i, 0.05 * np.sin(i), 0.05 * np.cos(i)]) + rng.uniform(-0.01, 0.01, 3)
        sys_gb.addParticle(12.0 * unit.amu)
        q_i = 0.25 if (i % 2 == 0) else -0.25
        nb_gb.addParticle(q_i * unit.elementary_charge, 0.32 * unit.nanometer, 0.4 * unit.kilojoules_per_mole)
        gb.addParticle(q_i * unit.elementary_charge, 0.17 * unit.nanometer, 0.8)
    cb_gb = mm.CustomBondForce("0.5 * lambda_gb * k * (r - r0)^2")
    cb_gb.addGlobalParameter("lambda_gb", 1.0)
    cb_gb.addPerBondParameter("k")
    cb_gb.addPerBondParameter("r0")
    for i in range(15):
        cb_gb.addBond(i, i + 1, [300.0, 0.38])
        nb_gb.addException(i, i + 1, 0.0, 0.1, 0.0)
    sys_gb.addForce(nb_gb)
    sys_gb.addForce(gb)
    sys_gb.addForce(cb_gb)

    ctx_gb = ProlixContext(
        sys_gb,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds),
        config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="gbsa_tile"),
    )
    ctx_gb.setPositions(pos_gb * unit.nanometer)
    ctx_gb.setVelocitiesToTemperature(300.0 * unit.kelvin, 605)
    ctx_gb.step(3)
    gb_runner_0 = ctx_gb._gbsa_runner
    assert gb_runner_0 is not None
    ctx_gb.setParameter("lambda_gb", 0.4)
    ctx_gb.step(3)
    assert ctx_gb._gbsa_runner is gb_runner_0


def test_in_loop_anisotropic_and_membrane_barostats() -> None:
    """In-loop GPU `MonteCarloAnisotropicBarostat` & `MonteCarloMembraneBarostat` honor axis & Z-mode invariants."""
    assert "MonteCarloAnisotropicBarostat" in UNSUPPORTED_OPENMM_FORCES
    assert "MonteCarloMembraneBarostat" in UNSUPPORTED_OPENMM_FORCES

    n_waters = 20
    cfg = ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs")

    # Raw converter still enforces explicit `config.anisotropic_barostat=True`, while ProlixContext auto-enables it
    sys_aniso, _, pos_nm, box0_nm = _build_water_system(n_waters, shape="cubic", edge_nm=2.45, seed=701)
    sys_aniso.addForce(
        mm.MonteCarloAnisotropicBarostat(
            mm.Vec3(1.0, 1.0, 8000.0) * unit.bar,
            300.0 * unit.kelvin,
            False,
            False,
            True,
            1,
        )
    )
    with pytest.raises(UnsupportedOpenMMForceError):
        from_openmm_system(sys_aniso)

    ctx_aniso = ProlixContext(
        sys_aniso,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
        config=cfg,
    )
    ctx_aniso.setPositions(pos_nm * unit.nanometer)
    ctx_aniso.setVelocitiesToTemperature(300.0 * unit.kelvin, 702)
    ctx_aniso.step(8)
    assert ctx_aniso._morton_runner is not None
    bv_aniso = np.asarray(ctx_aniso.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
    np.testing.assert_allclose(bv_aniso[0, 0], box0_nm[0, 0], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(bv_aniso[1, 1], box0_nm[1, 1], rtol=0.0, atol=1e-7)
    assert abs(bv_aniso[2, 2] - box0_nm[2, 2]) > 1e-6
    assert _max_water_bond_err_angstrom(np.asarray(ctx_aniso.getPositionsJax(in_nm=True)), n_waters) < 1e-5

    # Membrane barostat: ZFixed preserves Lz while XYIsotropic keeps Lx == Ly
    sys_zf, _, pos_zf, box_zf = _build_water_system(n_waters, shape="cubic", edge_nm=2.45, seed=703)
    sys_zf.addForce(
        mm.MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            500.0 * unit.bar * unit.nanometer,
            300.0 * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ZFixed,
            1,
        )
    )
    ctx_zf = ProlixContext(
        sys_zf,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
        config=cfg,
    )
    ctx_zf.setPositions(pos_zf * unit.nanometer)
    ctx_zf.setVelocitiesToTemperature(300.0 * unit.kelvin, 704)
    ctx_zf.step(8)
    assert ctx_zf._morton_runner is not None
    bv_zf = np.asarray(ctx_zf.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
    np.testing.assert_allclose(bv_zf[2, 2], box_zf[2, 2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(bv_zf[0, 0], bv_zf[1, 1], rtol=1e-6, atol=1e-6)

    # Membrane barostat: ConstantVolume conserves box volume Lx * Ly * Lz
    sys_cv, _, pos_cv, box_cv = _build_water_system(n_waters, shape="cubic", edge_nm=2.45, seed=705)
    sys_cv.addForce(
        mm.MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            500.0 * unit.bar * unit.nanometer,
            300.0 * unit.kelvin,
            mm.MonteCarloMembraneBarostat.XYIsotropic,
            mm.MonteCarloMembraneBarostat.ConstantVolume,
            1,
        )
    )
    ctx_cv = ProlixContext(
        sys_cv,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds),
        config=cfg,
    )
    ctx_cv.setPositions(pos_cv * unit.nanometer)
    ctx_cv.setVelocitiesToTemperature(300.0 * unit.kelvin, 706)
    v0 = float(np.linalg.det(box_cv))
    ctx_cv.step(8)
    assert ctx_cv._morton_runner is not None
    v1 = float(ctx_cv.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3))
    np.testing.assert_allclose(v1, v0, rtol=1e-5, atol=1e-5)


def test_dlpack_zero_copy_roundtrips_virtual_sites_and_hmr_4fs() -> None:
    """Zero-copy JAX/DLPack APIs avoid host NumPy copies, reconstruct virtual sites, and run 4.0 fs HMR."""
    n_waters = 20
    sys_hmr, top_hmr, pos_nm, _ = _build_water_system(
        n_waters, shape="cubic", edge_nm=2.40, hmr_h_mass_amu=3.0, seed=801
    )
    sim = ProlixSimulation(
        top_hmr,
        sys_hmr,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds),
        config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs"),
    )

    pos_j = jnp.asarray(pos_nm, dtype=jnp.float64)
    vel_j = jnp.full_like(pos_j, 0.05)
    sim.setPositionsJax(_StrictDLPackView(pos_j), in_nm=True)
    sim.setVelocitiesJax(_StrictDLPackView(vel_j), in_nm_per_ps=True)

    np.testing.assert_allclose(np.asarray(sim.getPositionsJax(in_nm=True)), pos_nm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sim.getPositionsJax(in_nm=False)), pos_nm * 10.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sim.getVelocitiesJax(in_nm_per_ps=True)), 0.05, rtol=1e-5, atol=1e-5)

    with pytest.raises(mm.OpenMMException):
        sim.setPositionsJax(jnp.zeros((5, 3)))
    with pytest.raises(mm.OpenMMException):
        sim.setVelocitiesJax(jnp.zeros((5, 3)))

    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 802)
    sim.step(10)
    assert sim.context._morton_runner is not None
    p_hmr = np.asarray(sim.getPositionsJax(in_nm=True))
    f_hmr_kj = np.asarray(sim.getForcesJax(in_kj_per_mol_nm=True))
    f_hmr_akma = np.asarray(sim.getForcesJax(in_kj_per_mol_nm=False))
    assert np.all(np.isfinite(p_hmr))
    assert np.all(np.isfinite(f_hmr_kj))
    np.testing.assert_allclose(f_hmr_kj, f_hmr_akma * 41.84, rtol=1e-5, atol=1e-4)
    assert _max_water_bond_err_angstrom(p_hmr, n_waters) < 1e-5

    # Virtual site reconstruction & zeroed VS velocity on `setPositionsJax` / `setVelocitiesJax`
    sys_vs = mm.System()
    p0 = sys_vs.addParticle(16.0 * unit.amu)
    p1 = sys_vs.addParticle(1.0 * unit.amu)
    p2 = sys_vs.addParticle(1.0 * unit.amu)
    pv = sys_vs.addParticle(0.0 * unit.amu)
    sys_vs.setVirtualSite(pv, mm.ThreeParticleAverageSite(p0, p1, p2, 0.5, 0.25, 0.25))
    nb_vs = mm.NonbondedForce()
    nb_vs.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for q_v in (0.0, 0.4, 0.4, -0.8):
        nb_vs.addParticle(q_v * unit.elementary_charge, 0.3 * unit.nanometer, 0.0 * unit.kilojoules_per_mole)
    for a_i, b_i in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        nb_vs.addException(a_i, b_i, 0.0, 0.1, 0.0)
    sys_vs.addForce(nb_vs)
    ctx_vs = ProlixContext(
        sys_vs,
        mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds),
    )
    raw_vs_pos = jnp.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [9.9, 9.9, 9.9]],
        dtype=jnp.float32,
    )
    ctx_vs.setPositionsJax(_StrictDLPackView(raw_vs_pos), in_nm=True)
    ctx_vs.setVelocitiesJax(_StrictDLPackView(jnp.ones((4, 3), dtype=jnp.float32)), in_nm_per_ps=True)
    out_vs_pos = np.asarray(ctx_vs.getPositionsJax(in_nm=True))
    out_vs_vel = np.asarray(ctx_vs.getVelocitiesJax(in_nm_per_ps=True))
    np.testing.assert_allclose(out_vs_pos[3], [0.025, 0.025, 0.0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_vs_vel[3], [0.0, 0.0, 0.0], atol=0.0)

    # Batch DLPack 3D & sequence round-trips
    batch = ProlixBatchSimulation(
        systems=[sys_hmr, sys_hmr],
        integrators=mm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds
        ),
        topologies=top_hmr,
        config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="morton_tile_pairs"),
    )
    pos_b = jnp.stack([pos_j, pos_j + 0.01], axis=0)
    vel_b = jnp.full_like(pos_b, -0.03)
    batch.setPositionsJax(_StrictDLPackView(pos_b), in_nm=True)
    batch.setVelocitiesJax(_StrictDLPackView(vel_b), in_nm_per_ps=True)
    np.testing.assert_allclose(np.asarray(batch.getPositionsJax(in_nm=True)), np.asarray(pos_b), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(batch.getVelocitiesJax(in_nm_per_ps=True)), -0.03, rtol=1e-5, atol=1e-5)
    batch.step(4)
    assert np.all(np.isfinite(np.asarray(batch.getPositionsJax(in_nm=True))))


def test_v39_redsox_u1_u5_knob_governance_invariants() -> None:
    """Verify U5 superset (0 missing knobs) and U1 reachability (100% consumer coverage, 0 pending)."""
    for name, schema in get_reference_config_schemas().items():
        superset = check_superset(
            KnobSurface,
            schema,
            alias_map=SUPERSET_ALIAS_MAP,
            inert_knobs=DOCUMENTED_INERT_KNOBS,
        )
        assert superset.passed, f"U5 failed for {name}: {superset.missing_fields}"
        assert superset.missing_fields == frozenset()

    reach = check_prolix_reachability(KnobSurface)
    assert reach.passed, f"U1 failed: {reach.unread_fields}"
    assert reach.unread_fields == frozenset()
    assert PENDING_EPIC_KNOBS == frozenset()
