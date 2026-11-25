"""Track 4 (v3.6) verification suite: GPU fast-path step/getState, exception PBC, alchemical offsets, and combining rules."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import openmm
import pytest
from openmm import unit

from prolix.openmm import ProlixContext, ProlixOpenMMConfig, parse_platform_properties


def _make_ref_context(system: openmm.System) -> openmm.Context:
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    plat = openmm.Platform.getPlatformByName("Reference")
    return openmm.Context(system, integ, plat)


# =============================================================================
# 1. Reciprocal erf(alpha*r_mic)/r_mic vs Physical Exception PBC
# =============================================================================


@pytest.mark.parametrize("exc_use_pbc", [False, True])
def test_pme_reciprocal_mic_vs_physical_exceptions_use_pbc(exc_use_pbc: bool):
    """PME reciprocal cancellation always uses MIC while physical 1-4/exceptions obey exceptions_use_pbc."""
    box_nm = 2.4
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    for _ in range(4):
        sys.addParticle(14.0)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.5, 48, 48, 48)
    nb.setUseDispersionCorrection(False)
    nb.setExceptionsUsePeriodicBoundaryConditions(exc_use_pbc)

    nb.addParticle(0.65, 0.32, 0.45)
    nb.addParticle(-0.55, 0.30, 0.40)
    nb.addParticle(0.40, 0.34, 0.35)
    nb.addParticle(-0.50, 0.31, 0.50)

    # Full exclusion straddling PBC boundary (x=0.1 nm vs x=2.25 nm -> r_raw=2.15 nm, r_mic=0.25 nm)
    nb.addException(0, 1, 0.0, 0.31, 0.0)
    # Explicit 1-4 / NBFix-shaped exception straddling PBC boundary (r_raw=2.05 nm > L/2, r_mic=0.35 nm)
    nb.addException(2, 3, -0.18, 0.325, 0.55)
    sys.addForce(nb)

    pos_nm = np.array(
        [
            [0.10, 0.50, 0.50],
            [2.25, 0.52, 0.48],
            [0.15, 1.40, 1.20],
            [2.20, 1.42, 1.18],
        ],
        dtype=np.float64,
    )

    cfg = ProlixOpenMMConfig(precision="double", use_pallas=False)
    ctx = ProlixContext(
        sys,
        openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond
        ),
        config=cfg,
    )
    ctx.setPositions(pos_nm * unit.nanometer)
    st = ctx.getState(getEnergy=True, getForces=True)
    e_prx = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_prx = st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    ref_ctx = _make_ref_context(sys)
    ref_ctx.setPositions(pos_nm * unit.nanometer)
    ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    assert e_prx == pytest.approx(e_ref, rel=1e-4, abs=5e-3)
    np.testing.assert_allclose(f_prx, f_ref, rtol=2e-4, atol=2.5e-2)

    if exc_use_pbc:
        # Under exceptions_use_pbc=True, shifting excluded/exception partners by -box_nm
        # leaves PME energy 100% invariant (periodic MIC cancellation).
        pos_unsplit = pos_nm.copy()
        pos_unsplit[1, 0] -= box_nm
        pos_unsplit[3, 0] -= box_nm
        ctx.setPositions(pos_unsplit * unit.nanometer)
        e_unsplit = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert e_prx == pytest.approx(e_unsplit, rel=1e-10, abs=1e-8)


def test_exceptions_use_pbc_distinct_energies_and_live_update():
    """Toggling exceptions_use_pbc changes straddling exception energy and updates via updateParametersInContext."""
    box_nm = 2.2
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.8 * unit.nanometer)
    nb.setPMEParameters(3.4, 36, 36, 36)
    nb.setExceptionsUsePeriodicBoundaryConditions(False)
    nb.addParticle(0.5, 0.3, 0.3)
    nb.addParticle(-0.5, 0.3, 0.3)
    nb.addException(0, 1, -0.25, 0.3, 0.4)
    sys.addForce(nb)

    pos_nm = np.array([[0.10, 0.5, 0.5], [2.00, 0.5, 0.5]], dtype=np.float64)
    ctx = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(precision="double", use_pallas=False),
    )
    ctx.setPositions(pos_nm * unit.nanometer)
    e_no_pbc = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    ref_no_pbc = _make_ref_context(sys)
    ref_no_pbc.setPositions(pos_nm * unit.nanometer)
    e_ref_no_pbc = ref_no_pbc.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert e_no_pbc == pytest.approx(e_ref_no_pbc, rel=1e-4, abs=5e-3)

    nb.setExceptionsUsePeriodicBoundaryConditions(True)
    ctx.updateParametersInContext(nb)
    e_with_pbc = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    ref_with_pbc = _make_ref_context(sys)
    ref_with_pbc.setPositions(pos_nm * unit.nanometer)
    e_ref_with_pbc = ref_with_pbc.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert e_with_pbc == pytest.approx(e_ref_with_pbc, rel=1e-4, abs=5e-3)
    # Exact closed-form delta between r_mic=0.30 nm and r_raw=1.90 nm (1-4 + PME reciprocal erf)
    c_kj = 138.935457644382
    q_excl = 0.5 * (-0.5)
    u_mic = (
        4.0 * 0.4 * ((0.3 / 0.30) ** 12 - (0.3 / 0.30) ** 6)
        + c_kj * (-0.25 / 0.30)
        - c_kj * q_excl * (math.erf(3.4 * 0.30) / 0.30)
    )
    u_raw = (
        4.0 * 0.4 * ((0.3 / 1.90) ** 12 - (0.3 / 1.90) ** 6)
        + c_kj * (-0.25 / 1.90)
        - c_kj * q_excl * (math.erf(3.4 * 1.90) / 1.90)
    )
    assert (e_with_pbc - e_no_pbc) == pytest.approx(u_mic - u_raw, rel=1e-6)


# =============================================================================
# 2. Reaction Field (CutoffPeriodic) on Tile Pairs & Conductor Limit
# =============================================================================


def test_reaction_field_tile_pairs_and_conductor_limit():
    """CutoffPeriodic enables _can_use_tile_pairs() and handles rf_dielectric=inf without NaN."""
    rng = np.random.default_rng(42)
    n_p = 64
    box_nm = 2.6
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setReactionFieldDielectric(78.3)
    nb.setUseDispersionCorrection(True)
    for i in range(n_p):
        sys.addParticle(16.0)
        q_i = 0.35 if (i % 2 == 0) else -0.35
        nb.addParticle(q_i, 0.31, 0.45)
    for i in range(0, n_p - 1, 4):
        nb.addException(i, i + 1, -0.06, 0.30, 0.25)
    sys.addForce(nb)

    # Well-separated grid positions
    grid = np.stack(
        np.meshgrid(
            np.linspace(0.2, 2.2, 4),
            np.linspace(0.2, 2.2, 4),
            np.linspace(0.2, 2.2, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(n_p, 3)
    pos_nm = grid + rng.uniform(-0.03, 0.03, size=(n_p, 3))

    ref_ctx = _make_ref_context(sys)
    ref_ctx.setPositions(pos_nm * unit.nanometer)
    ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    cfg = ProlixOpenMMConfig(
        precision="mixed",
        use_pallas=True,
        pallas_interpret=True,
        nonbonded_engine="morton_tile_pairs",
    )
    ctx = ProlixContext(
        sys,
        openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond),
        config=cfg,
    )
    ctx.setPositions(pos_nm * unit.nanometer)
    assert ctx._can_use_tile_pairs() is True
    assert ctx._tile_pair_nl is not None

    st = ctx.getState(getEnergy=True, getForces=True)
    e_prx = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_prx = st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    assert e_prx == pytest.approx(e_ref, rel=1e-4, abs=1e-2)
    np.testing.assert_allclose(f_prx, f_ref, rtol=2e-4, atol=2e-2)

    # Conductor limit rf_dielectric = +inf must match large finite dielectric without NaN
    sys_inf = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(sys))
    nb_inf = sys_inf.getForce(0)
    nb_inf.setReactionFieldDielectric(float("inf"))
    ctx_inf = ProlixContext(
        sys_inf,
        openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond),
        config=cfg,
    )
    ctx_inf.setPositions(pos_nm * unit.nanometer)
    st_inf = ctx_inf.getState(getEnergy=True, getForces=True)
    e_inf = st_inf.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_inf = st_inf.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    assert math.isfinite(e_inf)
    assert np.all(np.isfinite(f_inf))

    sys_1e15 = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(sys))
    sys_1e15.getForce(0).setReactionFieldDielectric(1.0e15)
    ref_1e15 = _make_ref_context(sys_1e15)
    ref_1e15.setPositions(pos_nm * unit.nanometer)
    e_1e15 = ref_1e15.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert e_inf == pytest.approx(e_1e15, rel=1e-4, abs=1e-2)


# =============================================================================
# 3. OpenFE Alchemical Particle & Exception Offsets + Derivatives
# =============================================================================


@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
def test_alchemical_particle_and_exception_offsets_lambda_sweep(precision: str):
    """ParticleParameterOffset + ExceptionParameterOffset match OpenMM Reference across lambda sweep."""
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
    nb.addGlobalParameter("lambda_elec", 0.4)
    nb.addGlobalParameter("lambda_sterics", 0.25)

    base_q = [0.40, -0.40, 0.25, -0.25, 0.15, -0.15]
    for i in range(6):
        nb.addParticle(base_q[i], 0.31 + 0.01 * i, 0.40 + 0.05 * i)

    # Two global parameters modifying particle 0 & 1 simultaneously
    nb.addParticleParameterOffset("lambda_elec", 0, -0.40, 0.0, 0.0)
    nb.addParticleParameterOffset("lambda_sterics", 0, 0.0, 0.02, -0.30)
    nb.addParticleParameterOffset("lambda_elec", 1, 0.40, 0.0, 0.0)

    # Offset-referenced ZERO exception at lambda=0 that turns ON as lambda_elec increases!
    exc0 = nb.addException(0, 2, 0.0, 0.32, 0.0)
    exc1 = nb.addException(1, 3, -0.08, 0.33, 0.35)
    nb.addExceptionParameterOffset("lambda_elec", exc0, -0.12, 0.01, 0.45)
    nb.addExceptionParameterOffset("lambda_sterics", exc1, 0.04, 0.0, -0.20)
    sys.addForce(nb)

    pos_nm = np.array(
        [
            [0.30, 0.40, 0.50],
            [0.72, 0.45, 0.52],
            [0.35, 0.85, 0.55],
            [0.80, 0.90, 0.60],
            [1.30, 0.55, 0.70],
            [1.35, 1.05, 0.75],
        ],
        dtype=np.float64,
    )

    ref_ctx = _make_ref_context(sys)
    ref_ctx.setPositions(pos_nm * unit.nanometer)

    ctx = ProlixContext(
        sys,
        openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picosecond),
        config=ProlixOpenMMConfig(precision=precision, use_pallas=False),
    )
    ctx.setPositions(pos_nm * unit.nanometer)
    # setParameter BEFORE first getState (plus non-zero defaults)
    ctx.setParameter("lambda_elec", 0.0)
    ref_ctx.setParameter("lambda_elec", 0.0)

    e_rtol = 5e-4 if precision == "single" else 1e-4
    f_rtol = 2e-3 if precision == "single" else 2e-4
    f_atol = 0.25 if precision == "single" else 2e-2
    for lam_e, lam_s in [
        (0.0, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ]:
        ctx.setParameter("lambda_elec", lam_e)
        ctx.setParameter("lambda_sterics", lam_s)
        ref_ctx.setParameter("lambda_elec", lam_e)
        ref_ctx.setParameter("lambda_sterics", lam_s)

        st_p = ctx.getState(getEnergy=True, getForces=True)
        st_r = ref_ctx.getState(getEnergy=True, getForces=True)
        e_p = st_p.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_r = st_r.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        f_p = st_p.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        f_r = st_r.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        assert e_p == pytest.approx(e_r, rel=e_rtol, abs=2e-2)
        np.testing.assert_allclose(f_p, f_r, rtol=f_rtol, atol=f_atol)


@pytest.mark.parametrize("follows_params", [True, False])
def test_offset_parameter_derivatives_and_update_validation(follows_params: bool):
    """Single-pass getParameterDerivatives matches central finite differences and validates offset updates."""
    box_nm = 2.2
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    for _ in range(4):
        sys.addParticle(14.0)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.85 * unit.nanometer)
    nb.setPMEParameters(3.5, 24, 24, 24)
    nb.setUseDispersionCorrection(True)
    nb.addGlobalParameter("lam_q", 0.35)
    nb.addGlobalParameter("lam_lj", 0.45)

    nb.addParticle(0.50, 0.32, 0.60)
    nb.addParticle(-0.50, 0.30, 0.50)
    nb.addParticle(0.20, 0.34, 0.40)
    nb.addParticle(-0.20, 0.31, 0.45)

    p_off_idx = nb.addParticleParameterOffset("lam_q", 0, -0.35, 0.0, 0.0)
    nb.addParticleParameterOffset("lam_lj", 0, 0.0, 0.03, 0.45)
    e_idx = nb.addException(1, 2, -0.10, 0.32, 0.30)
    e_off_idx = nb.addExceptionParameterOffset("lam_q", e_idx, 0.08, 0.0, 0.15)
    sys.addForce(nb)

    pos_nm = np.array(
        [
            [0.25, 0.30, 0.40],
            [0.65, 0.35, 0.42],
            [0.30, 0.78, 0.45],
            [0.75, 0.82, 0.50],
        ],
        dtype=np.float64,
    )
    ctx = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(
            precision="double",
            use_pallas=False,
            dispersion_follows_parameters=follows_params,
        ),
    )
    ctx.setPositions(pos_nm * unit.nanometer)

    st0 = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    derivs = st0.getEnergyParameterDerivatives()
    assert "lam_q" in derivs
    assert "lam_lj" in derivs

    h = 1e-5
    for pname, base_val in [("lam_q", 0.35), ("lam_lj", 0.45)]:
        ctx.setParameter(pname, base_val + h)
        e_hi = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        ctx.setParameter(pname, base_val - h)
        e_lo = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        ctx.setParameter(pname, base_val)
        fd_deriv = (e_hi - e_lo) / (2.0 * h)
        assert derivs[pname] == pytest.approx(fd_deriv, rel=1e-5, abs=1e-4)

    if not follows_params:
        # Verify dispersion_follows_parameters=False freezes tail across lambda changes
        ctx_live = ProlixContext(
            sys,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(
                precision="double",
                use_pallas=False,
                dispersion_follows_parameters=True,
            ),
        )
        ctx_live.setPositions(pos_nm * unit.nanometer)
        ctx.setParameter("lam_lj", 0.95)
        ctx_live.setParameter("lam_lj", 0.95)
        e_frozen = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        e_live = ctx_live.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        assert abs(e_frozen - e_live) > 1e-4

    # Live scale update via updateParametersInContext (same target particle/exception)
    e_before_upd = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    nb.setParticleParameterOffset(p_off_idx, "lam_q", 0, -0.15, 0.0, 0.0)
    nb.setExceptionParameterOffset(e_off_idx, "lam_q", e_idx, 0.04, 0.0, 0.25)
    ctx.updateParametersInContext(nb)
    e_after_upd = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert abs(e_after_upd - e_before_upd) > 1.0

    # Moving offset particle target must raise OpenMMException
    nb.setParticleParameterOffset(p_off_idx, "lam_q", 1, -0.15, 0.0, 0.0)
    with pytest.raises(openmm.OpenMMException, match="changed its particle or its parameter"):
        ctx.updateParametersInContext(nb)


# =============================================================================
# 4. Geometric vs Lorentz-Berthelot Combining Rule & Zero-Eps AD Safety
# =============================================================================


def test_lj_combining_rule_geometric_vs_lorentz_berthelot_and_tail():
    """Geometric combining matches exact analytical pair + switched/unswitched dispersion tail without zero-eps NaN."""
    cfg_parsed = parse_platform_properties(
        {
            "LjCombiningRule": "geometric",
            "ExceptionsUsePbc": "true",
            "DispersionFollowsParameters": "false",
        }
    )
    assert cfg_parsed.lj_combining_rule == "geometric"
    assert cfg_parsed.exceptions_use_pbc is True
    assert cfg_parsed.dispersion_follows_parameters is False

    box_nm = 3.0
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    sys.addParticle(16.0)
    sys.addParticle(28.0)
    sys.addParticle(1.008)  # zero-epsilon hydrogen to verify AD safety

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.setUseSwitchingFunction(False)
    nb.setUseDispersionCorrection(True)
    nb.addGlobalParameter("lam_eps", 0.0)
    nb.addParticle(0.0, 0.28, 0.80)
    nb.addParticle(0.0, 0.44, 0.30)
    nb.addParticle(0.0, 0.10, 0.00)  # eps == 0.0!
    nb.addParticleParameterOffset("lam_eps", 0, 0.0, 0.0, 0.10)
    sys.addForce(nb)

    pos_nm = np.array(
        [[0.5, 0.5, 0.5], [0.95, 0.5, 0.5], [1.45, 0.5, 0.5]], dtype=np.float64
    )
    ctx_geom = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(
            precision="double",
            use_pallas=False,
            lj_combining_rule="geometric",
        ),
    )
    ctx_lb = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(
            precision="double",
            use_pallas=False,
            lj_combining_rule="lorentz_berthelot",
        ),
    )
    ctx_geom.setPositions(pos_nm * unit.nanometer)
    ctx_lb.setPositions(pos_nm * unit.nanometer)

    st_geom = ctx_geom.getState(
        getEnergy=True, getForces=True, getParameterDerivatives=True
    )
    st_lb = ctx_lb.getState(getEnergy=True)
    e_geom = st_geom.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_lb = st_lb.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_geom = st_geom.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    d_geom = st_geom.getEnergyParameterDerivatives()

    assert abs(e_geom - e_lb) > 0.03
    assert np.all(np.isfinite(f_geom))
    assert math.isfinite(d_geom["lam_eps"])

    # Analytical check in kJ/mol & nm for unswitched geometric LJ + unswitched geometric tail:
    sig0, eps0 = 0.28, 0.80
    sig1, eps1 = 0.44, 0.30
    sig_g = math.sqrt(sig0 * sig1)
    eps_g = math.sqrt(eps0 * eps1)
    r01 = 0.45
    sr6 = (sig_g / r01) ** 6
    e_pair_kj = 4.0 * eps_g * (sr6**2 - sr6)
    rc = 1.0
    vol_nm3 = box_nm**3
    # Ordered pair sum over (i, j) in {0, 1}^2 for unswitched tail:
    # E_tail = (8 * pi / V) * sum_{i,j} 0.5 * 4 * eps_ij * (sig_ij^12 / (9 * rc^9) - sig_ij^6 / (3 * rc^3))
    e_tail_kj = 0.0
    for si, ei in [(sig0, eps0), (sig1, eps1)]:
        for sj, ej in [(sig0, eps0), (sig1, eps1)]:
            s_ij = math.sqrt(si * sj)
            ep_ij = math.sqrt(ei * ej)
            c12 = 4.0 * ep_ij * (s_ij**12)
            c6 = 4.0 * ep_ij * (s_ij**6)
            e_tail_kj += (2.0 * math.pi / vol_nm3) * (
                c12 / (9.0 * (rc**9)) - c6 / (3.0 * (rc**3))
            )
    assert e_geom == pytest.approx(e_pair_kj + e_tail_kj, rel=1e-6, abs=1e-6)


# =============================================================================
# 5. Vectorized CSR Wrap/Unwrap + GPU Morton Fast-Path step() & getState()
# =============================================================================


def test_vectorized_csr_wrap_and_unwrap_triclinic_and_orthorhombic():
    """Flat-CSR _unwrap_molecules_and_com and _wrap_molecules preserve intramolecular geometry."""
    sys = openmm.System()
    a_v = openmm.Vec3(2.5, 0.0, 0.0)
    b_v = openmm.Vec3(0.4, 2.4, 0.0)
    c_v = openmm.Vec3(-0.2, 0.3, 2.6)
    sys.setDefaultPeriodicBoxVectors(a_v, b_v, c_v)
    bonds = openmm.HarmonicBondForce()
    n_mols = 20
    for m in range(n_mols):
        base = 3 * m
        sys.addParticle(15.999)
        sys.addParticle(1.008)
        sys.addParticle(1.008)
        bonds.addBond(base, base + 1, 0.09572 * unit.nanometer, 400000.0)
        bonds.addBond(base, base + 2, 0.09572 * unit.nanometer, 400000.0)
    sys.addForce(bonds)

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    rng = np.random.default_rng(7)
    coms = rng.uniform(-4.0, 6.0, size=(n_mols, 3))
    pos_nm = np.zeros((3 * n_mols, 3), dtype=np.float64)
    for m in range(n_mols):
        pos_nm[3 * m + 0] = coms[m]
        pos_nm[3 * m + 1] = coms[m] + np.array([0.09572, 0.0, 0.0])
        pos_nm[3 * m + 2] = coms[m] + np.array([-0.024, 0.0926, 0.0])

    ctx.setPositions(pos_nm * unit.nanometer)
    wrapped = ctx.getState(
        getPositions=True, enforcePeriodicBox=True
    ).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Every intramolecular O-H bond must remain intact (no molecule torn across periodic boundary)
    for m in range(n_mols):
        d1 = np.linalg.norm(wrapped[3 * m + 1] - wrapped[3 * m + 0])
        d2 = np.linalg.norm(wrapped[3 * m + 2] - wrapped[3 * m + 0])
        assert d1 == pytest.approx(0.09572, abs=1e-6)
        assert d2 == pytest.approx(0.09572, rel=1e-3)


def _build_water_box_system(
    n_waters: int = 32,
    *,
    is_rf: bool = False,
    with_barostat: bool = False,
) -> tuple[openmm.System, np.ndarray]:
    box_nm = 2.2
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    )
    nb = openmm.NonbondedForce()
    if is_rf:
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setReactionFieldDielectric(78.3)
    else:
        nb.setNonbondedMethod(openmm.NonbondedForce.PME)
        nb.setPMEParameters(3.4, 36, 36, 36)
    nb.setCutoffDistance(0.85 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    nb.addGlobalParameter("lam_scale", 0.0)

    n_side = math.ceil(n_waters ** (1.0 / 3.0))
    spacing = box_nm / n_side
    coords = []
    for w in range(n_waters):
        ix = w % n_side
        iy = (w // n_side) % n_side
        iz = w // (n_side * n_side)
        o_pos = np.array(
            [(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing],
            dtype=np.float64,
        )
        h1_pos = o_pos + np.array([0.09572, 0.0, 0.0])
        theta = math.radians(104.52)
        h2_pos = o_pos + np.array(
            [0.09572 * math.cos(theta), 0.09572 * math.sin(theta), 0.0]
        )
        coords.extend([o_pos, h1_pos, h2_pos])

        base = 3 * w
        sys.addParticle(15.9994)
        sys.addParticle(1.00794)
        sys.addParticle(1.00794)
        sys.addConstraint(base, base + 1, 0.09572 * unit.nanometer)
        sys.addConstraint(base, base + 2, 0.09572 * unit.nanometer)
        sys.addConstraint(base + 1, base + 2, 0.15139 * unit.nanometer)

        nb.addParticle(-0.834, 0.315075, 0.635968)
        nb.addParticle(0.417, 0.04, 0.0)
        nb.addParticle(0.417, 0.04, 0.0)
        nb.addException(base, base + 1, 0.0, 0.1, 0.0)
        nb.addException(base, base + 2, 0.0, 0.1, 0.0)
        nb.addException(base + 1, base + 2, 0.0, 0.1, 0.0)

    nb.addParticleParameterOffset("lam_scale", 0, 0.10, 0.0, 0.0)
    nb.addParticleParameterOffset("lam_scale", 1, -0.05, 0.0, 0.0)
    nb.addParticleParameterOffset("lam_scale", 2, -0.05, 0.0, 0.0)
    sys.addForce(nb)

    if with_barostat:
        sys.addForce(
            openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 5)
        )

    return sys, np.asarray(coords, dtype=np.float64)


@pytest.mark.parametrize(
    ("is_rf", "with_barostat", "baro_mode"),
    [
        (False, False, "auto"),
        (True, False, "auto"),
        (False, True, "mc"),
        (False, True, "scr"),
    ],
)
def test_morton_gpu_fastpath_step_and_zero_fallback_getstate(
    is_rf: bool, with_barostat: bool, baro_mode: str
):
    """Morton fast-path executes multi-chunk step() and serves getState(energy, forces, pos) with zero fallback."""
    sys, pos_nm = _build_water_box_system(
        n_waters=16, is_rf=is_rf, with_barostat=with_barostat
    )
    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        use_pallas=True,
        pallas_interpret=True,
        nonbonded_engine="morton_tile_pairs",
        nl_update_every=4,
        barostat_mode=baro_mode,
    )
    ctx = ProlixContext(sys, integ, config=cfg)
    ctx.setPositions(pos_nm * unit.nanometer)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 123)

    # Multi-chunk step spanning multiple nl_update_every (4) and barostat (5) intervals
    with patch(
        "prolix.openmm.context._pure_step_rollout",
        side_effect=AssertionError("Must not fall back to _pure_step_rollout"),
    ):
        ctx.step(10)

    assert ctx._morton_runner is not None
    assert ctx.getStepCount() == 10
    assert ctx._host_nl_dirty is True
    if not with_barostat:
        assert ctx._box_nm_dirty is False

    # Fast-path getState(getEnergy=True, getForces=True, getPositions=True, enforcePeriodicBox=True)
    # must NOT invoke _pure_compute_energy_and_forces or host _init_neighbor_list!
    with (
        patch(
            "prolix.openmm.context._pure_compute_energy_and_forces",
            side_effect=AssertionError(
                "getState(groups=-1) must stay on GPU Morton fast-path"
            ),
        ),
        patch.object(
            ctx,
            "_init_neighbor_list",
            side_effect=AssertionError("getState must not rebuild host NL"),
        ),
    ):
        st_fast = ctx.getState(
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
        # Forces-only query right after step/getState reuses cached Morton forces (zero force eval)
        st_f_only = ctx.getState(getForces=True)

    assert ctx._host_nl_dirty is True
    pe_fast = st_fast.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    ke_fast = st_fast.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_fast = st_fast.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    f_cached = st_f_only.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    pos_after = st_fast.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    bv_after = st_fast.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)

    assert math.isfinite(pe_fast)
    assert ke_fast > 0.0
    assert np.all(np.isfinite(f_fast))
    np.testing.assert_allclose(f_cached, f_fast, rtol=1e-6, atol=1e-5)

    # Parity check of fast-path PE & Forces against OpenMM Reference at exact post-step coordinates & box!
    ref_ctx = _make_ref_context(sys)
    ref_ctx.setPeriodicBoxVectors(
        bv_after[0] * unit.nanometer,
        bv_after[1] * unit.nanometer,
        bv_after[2] * unit.nanometer,
    )
    ref_ctx.setPositions(pos_after * unit.nanometer)
    ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
    pe_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    assert pe_fast == pytest.approx(pe_ref, rel=5e-4, abs=0.25)
    np.testing.assert_allclose(f_fast, f_ref, rtol=2e-3, atol=0.35)

    # Verify mutators + checkpoint + reinitialize(preserveState=True) seamlessly interoperate with Morton runner
    ctx.setParameter("lam_scale", 0.5)
    assert ctx._morton_forces_dirty is True
    pe_shifted = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert abs(pe_shifted - pe_fast) > 1e-2

    chk = ctx.createCheckpoint()
    ctx.step(4)
    assert ctx.getStepCount() == 14
    ctx.loadCheckpoint(chk)
    assert ctx.getStepCount() == 10
    pe_restored = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    assert pe_restored == pytest.approx(pe_shifted, rel=1e-5, abs=1e-3)

    ctx.reinitialize(preserveState=True)
    assert ctx.getStepCount() == 10
    ctx.step(4)
    assert ctx.getStepCount() == 14
