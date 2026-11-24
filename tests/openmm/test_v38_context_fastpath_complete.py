"""v3.8 Track 1 tests: ProlixContext GPU fast-path completion.

Verifies persistent GPU Morton BAOAB (`_morton_runner`) and implicit-solvent
OBC2 GBSA BAOAB (`_gbsa_runner`) fast-paths against OpenMM `Reference`:
1. `PositionRestraints` with negative / outside-box `x0` + per-substep unwrapped
   PBC tracking + zero-recompilation `updateParametersInContext`.
2. `VirtualSites` (`TIP4P-Ew` `ThreeParticleAverageSite` & `LocalCoordinatesSite`)
   across NVT, PBC-boundary-straddling molecules, `getForces`, and NPT `MonteCarloBarostat`.
3. OPLS `"geometric"` LJ combining rule on Morton fast-path.
4. `NonbondedForce.LJPME` (`"lj_pme"`) direct + reciprocal + exclusion + NPT on Morton fast-path.
5. Persistent GPU `OBC2 GBSA` BAOAB step runner (`_gbsa_runner`) with bonds, angles,
   torsions, 1-4 exceptions, H-bond constraints, and `PositionRestraints`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import openmm
import openmm.unit as _omm_unit

from prolix.openmm import ProlixContext, ProlixOpenMMConfig

unit = cast("Any", _omm_unit)


def _make_reference_context(
    system: openmm.System, step_ps: float = 0.001
) -> openmm.Context:
    return openmm.Context(
        system,
        openmm.VerletIntegrator(step_ps),
        openmm.Platform.getPlatformByName("Reference"),
    )


def _assert_reference_energy_and_forces(
    system: openmm.System,
    ctx: ProlixContext,
    *,
    e_atol_kj: float = 0.08,
    f_atol_kj_nm: float = 0.2,
) -> None:
    st = ctx.getState(getPositions=True, getEnergy=True, getForces=True)
    pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    bv_nm = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    pe_prolix = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_prolix = st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    ref_ctx = _make_reference_context(system)
    if system.usesPeriodicBoundaryConditions():
        ref_ctx.setPeriodicBoxVectors(bv_nm[0], bv_nm[1], bv_nm[2])
    ref_ctx.setPositions(pos_nm)
    if any(system.isVirtualSite(i) for i in range(system.getNumParticles())):
        ref_ctx.computeVirtualSites()
    for k, v in ctx.getParameters().items():
        ref_ctx.setParameter(k, v)

    ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
    pe_ref = ref_st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = ref_st.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    assert abs(pe_prolix - pe_ref) < e_atol_kj, (
        f"PE mismatch: Prolix={pe_prolix:.6f} vs Reference={pe_ref:.6f} kJ/mol"
    )
    np.testing.assert_allclose(f_prolix, f_ref, atol=f_atol_kj_nm, rtol=2e-3)


def test_morton_fastpath_position_restraints_negative_x0_and_zero_recompile() -> None:
    """PositionRestraints (`x0 < 0` & `y0 > L`) stay on `_morton_runner` with exact unwrapped PBC."""
    n_atoms = 16
    box_nm = 2.6
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        [box_nm, 0.0, 0.0], [0.0, box_nm, 0.0], [0.0, 0.0, box_nm]
    )

    bonds = openmm.HarmonicBondForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.2, 48, 48, 48)
    nb.setUseDispersionCorrection(True)

    cef = openmm.CustomExternalForce(
        "0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    cef.addPerParticleParameter("x0")
    cef.addPerParticleParameter("y0")
    cef.addPerParticleParameter("z0")
    cef.addPerParticleParameter("k")

    pos0 = np.zeros((n_atoms, 3), dtype=np.float64)
    bond_pairs: list[tuple[int, int]] = []
    for i in range(n_atoms):
        sys.addParticle(12.0)
        q = 0.25 if (i % 2 == 0) else -0.25
        nb.addParticle(q, 0.33, 0.45)
        if i % 2 == 1:
            bonds.addBond(i - 1, i, 0.15, 250000.0)
            bond_pairs.append((i - 1, i))

    # Dimer 0 placed at negative x and above-box y, restrained to (-0.35, 3.15, 1.30) nm
    pos0[0] = [-0.33, 3.13, 1.30]
    pos0[1] = [-0.18, 3.14, 1.30]
    cef.addParticle(0, [-0.35, 3.15, 1.30, 1200.0])
    cef.addParticle(1, [-0.20, 3.15, 1.30, 800.0])

    for i in range(2, n_atoms):
        mol_k = i // 2
        offset = 0.55 * (mol_k % 3) + 0.35
        pos0[i] = [
            offset + (0.15 if i % 2 == 1 else 0.0),
            0.60 + 0.55 * (mol_k // 3),
            1.20,
        ]

    nb.createExceptionsFromBonds(bond_pairs, 0.5, 0.5)
    sys.addForce(bonds)
    sys.addForce(nb)
    sys.addForce(cef)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond
    )
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        nonbonded_engine="morton_tile_pairs",
        use_pallas=True,
        pallas_interpret=True,
        nl_update_every=3,
    )
    ctx = ProlixContext(sys, integ, config=cfg)
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=42)

    assert len(ctx._position_restraints) == 1
    assert ctx._can_use_morton_step_runner(False, None, None, "baoab")

    ctx.step(6)
    assert ctx._morton_runner is not None
    assert ctx._can_use_morton_obs()

    pos_after = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    # Restrained dimer 0 must remain near unwrapped (-0.35, 3.15, 1.30) nm
    assert pos_after[0, 0] < 0.0
    assert pos_after[0, 1] > box_nm
    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.05, f_atol_kj_nm=0.15)

    # Dynamic restraint update must NOT recompile / rebuild _morton_runner
    runner_before = ctx._morton_runner
    cef.setParticleParameters(0, 0, [-0.40, 3.20, 1.25, 1800.0])
    ctx.updateParametersInContext(cef)
    assert ctx._morton_runner is runner_before
    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.05, f_atol_kj_nm=0.15)


def _build_tip4pew_box(
    n_waters: int = 8,
    box_nm: float = 2.4,
    with_barostat: bool = False,
) -> tuple[openmm.System, np.ndarray]:
    """Construct an authentic periodic TIP4P-Ew rigid water box (O, H1, H2, M)."""
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        [box_nm, 0.0, 0.0], [0.0, box_nm, 0.0], [0.0, 0.0, box_nm]
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.2, 48, 48, 48)
    nb.setUseDispersionCorrection(True)

    r_oh = 0.09572
    r_hh = 0.15139
    half_hh = 0.5 * r_hh
    z_h = np.sqrt(r_oh**2 - half_hh**2)

    pos = np.zeros((4 * n_waters, 3), dtype=np.float64)
    excl_bonds: list[tuple[int, int]] = []

    for w in range(n_waters):
        base = 4 * w
        o_idx = sys.addParticle(15.9994)
        h1_idx = sys.addParticle(1.007947)
        h2_idx = sys.addParticle(1.007947)
        m_idx = sys.addParticle(0.0)
        assert (o_idx, h1_idx, h2_idx, m_idx) == (base, base + 1, base + 2, base + 3)

        sys.addConstraint(o_idx, h1_idx, r_oh)
        sys.addConstraint(o_idx, h2_idx, r_oh)
        sys.addConstraint(h1_idx, h2_idx, r_hh)
        sys.setVirtualSite(
            m_idx,
            openmm.ThreeParticleAverageSite(
                o_idx, h1_idx, h2_idx, 0.786646558, 0.106676721, 0.106676721
            ),
        )

        nb.addParticle(0.0, 0.316435, 0.680946)
        nb.addParticle(0.52422, 1.0, 0.0)
        nb.addParticle(0.52422, 1.0, 0.0)
        nb.addParticle(-1.04844, 1.0, 0.0)

        for a in range(4):
            for b in range(a + 1, 4):
                excl_bonds.append((base + a, base + b))

        if w == 0:
            # Place water 0 straddling x = 0 PBC plane (O at +0.02 nm, H1 at -0.056 nm)
            ox, oy, oz = 0.02, 0.55, 0.55
            pos[o_idx] = [ox, oy, oz]
            pos[h1_idx] = [ox - half_hh, oy + z_h, oz]
            pos[h2_idx] = [ox + half_hh, oy + z_h, oz]
        else:
            gx = w % 2
            gy = (w // 2) % 2
            gz = w // 4
            ox = 0.55 + 0.85 * gx
            oy = 0.55 + 0.85 * gy
            oz = 0.55 + 0.85 * gz
            pos[o_idx] = [ox, oy, oz]
            pos[h1_idx] = [ox + half_hh, oy + z_h, oz]
            pos[h2_idx] = [ox - half_hh, oy + z_h, oz]

    nb.createExceptionsFromBonds(excl_bonds, 1.0, 1.0)
    sys.addForce(nb)
    if with_barostat:
        sys.addForce(
            openmm.MonteCarloBarostat(
                1.0 * unit.bar, 300.0 * unit.kelvin, 2
            )
        )
    return sys, pos


def test_morton_fastpath_tip4pew_virtual_sites_nvt_and_getforces() -> None:
    """TIP4P-Ew virtual sites run on `_morton_runner` in NVT with exact PBC straddle & `getForces`."""
    sys, pos0 = _build_tip4pew_box(n_waters=8, with_barostat=False)
    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        nonbonded_engine="morton_tile_pairs",
        use_pallas=True,
        pallas_interpret=True,
        nl_update_every=2,
    )
    ctx = ProlixContext(sys, integ, config=cfg)
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=101)

    assert not ctx._has_frozen_atoms
    ref_ctx = _make_reference_context(sys)
    assert ctx.getMolecules() == ref_ctx.getMolecules()
    assert len(ctx.getMolecules()) == 8
    assert ctx._can_use_morton_step_runner(False, None, None, "baoab")

    ctx.step(5)
    assert ctx._morton_runner is not None
    assert ctx._can_use_morton_obs()

    # Check cached force unpermutation path (`getForces=True, getEnergy=False`) BEFORE recompute
    f_cached = (
        ctx.getState(getForces=True)
        .getForces(asNumpy=True)
        .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    )
    st_full = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
    f_full = st_full.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    np.testing.assert_allclose(f_cached, f_full, atol=3e-3, rtol=1e-3)

    vel = st_full.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    vs_slots = np.arange(3, 32, 4)
    np.testing.assert_allclose(vel[vs_slots], 0.0, atol=1e-12)

    pos_after = st_full.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    for w in range(8):
        b = 4 * w
        d_oh1 = np.linalg.norm(pos_after[b + 1] - pos_after[b])
        d_oh2 = np.linalg.norm(pos_after[b + 2] - pos_after[b])
        d_hh = np.linalg.norm(pos_after[b + 2] - pos_after[b + 1])
        m_expected = (
            0.786646558 * pos_after[b]
            + 0.106676721 * pos_after[b + 1]
            + 0.106676721 * pos_after[b + 2]
        )
        assert abs(d_oh1 - 0.09572) < 1e-5
        assert abs(d_oh2 - 0.09572) < 1e-5
        assert abs(d_hh - 0.15139) < 1e-5
        np.testing.assert_allclose(pos_after[b + 3], m_expected, atol=1e-5)

    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.06, f_atol_kj_nm=0.18)


def test_morton_fastpath_tip4pew_virtual_sites_npt_barostat() -> None:
    """TIP4P-Ew virtual sites + MonteCarloBarostat stay on `_morton_runner` and match Reference."""
    sys, pos0 = _build_tip4pew_box(n_waters=8, with_barostat=True)
    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        nonbonded_engine="morton_tile_pairs",
        use_pallas=True,
        pallas_interpret=True,
        nl_update_every=2,
    )
    ctx = ProlixContext(sys, integ, config=cfg)
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=202)

    baro_info = ctx._converted_system.barostat
    assert ctx._can_use_morton_step_runner(True, "mc", baro_info, "baoab")

    ctx.step(6)
    assert ctx._morton_runner is not None
    assert ctx._can_use_morton_obs()

    st_wr = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos_wr = st_wr.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel = st_wr.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    np.testing.assert_allclose(vel[3::4], 0.0, atol=1e-12)
    for w in range(8):
        b = 4 * w
        # Whole-molecule wrap keeps M right next to O (< 0.02 nm)
        assert np.linalg.norm(pos_wr[b + 3] - pos_wr[b]) < 0.02

    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.06, f_atol_kj_nm=0.18)


def test_morton_fastpath_local_coordinates_site_parity() -> None:
    """`LocalCoordinatesSite` virtual sites execute on `_morton_runner` with exact VJP forces."""
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors([2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 2.5])
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.2, 48, 48, 48)

    pos0 = np.zeros((8, 3), dtype=np.float64)
    excl_bonds: list[tuple[int, int]] = []
    for m in range(2):
        b = 4 * m
        p0 = sys.addParticle(12.0)
        p1 = sys.addParticle(14.0)
        p2 = sys.addParticle(12.0)
        vs = sys.addParticle(0.0)
        assert (p0, p1, p2, vs) == (b, b + 1, b + 2, b + 3)

        bonds.addBond(p0, p1, 0.14, 220000.0)
        bonds.addBond(p1, p2, 0.14, 220000.0)
        angles.addAngle(p0, p1, p2, 2.0, 450.0)
        sys.setVirtualSite(
            vs,
            openmm.LocalCoordinatesSite(
                p0,
                p1,
                p2,
                [0.2, 0.6, 0.2],
                [-0.5, 0.0, 0.5],
                [-0.5, 1.0, -0.5],
                [0.02, 0.03, 0.015],
            ),
        )
        nb.addParticle(0.15, 0.33, 0.40)
        nb.addParticle(-0.10, 0.32, 0.45)
        nb.addParticle(0.15, 0.33, 0.40)
        nb.addParticle(-0.20, 0.28, 0.20)
        for i in range(4):
            for j in range(i + 1, 4):
                excl_bonds.append((b + i, b + j))

        ox = 0.55 + 0.85 * m
        pos0[p0] = [ox - 0.12, 0.80, 0.90]
        pos0[p1] = [ox, 0.88, 0.92]
        pos0[p2] = [ox + 0.12, 0.80, 0.91]

    nb.createExceptionsFromBonds(excl_bonds, 0.5, 0.5)
    sys.addForce(bonds)
    sys.addForce(angles)
    sys.addForce(nb)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond
    )
    ctx = ProlixContext(
        sys,
        integ,
        config=ProlixOpenMMConfig(
            precision="mixed",
            nonbonded_engine="morton_tile_pairs",
            use_pallas=True,
            pallas_interpret=True,
        ),
    )
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=77)
    ctx.step(4)
    assert ctx._morton_runner is not None
    assert ctx._can_use_morton_obs()
    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.05, f_atol_kj_nm=0.15)


def test_morton_fastpath_opls_geometric_combining_rule_parity() -> None:
    """OPLS `lj_combining_rule='geometric'` activates `_morton_runner` and matches OpenMM Reference."""
    n_atoms = 16
    box_nm = 2.5
    # System for Prolix (NonbondedForce PME + config lj_combining_rule="geometric")
    sys_prolix = openmm.System()
    sys_prolix.setDefaultPeriodicBoxVectors(
        [box_nm, 0.0, 0.0], [0.0, box_nm, 0.0], [0.0, 0.0, box_nm]
    )
    # Matching OpenMM Reference system: NonbondedForce PME (charges only) + CustomNonbondedForce (geometric LJ)
    sys_ref = openmm.System()
    sys_ref.setDefaultPeriodicBoxVectors(
        [box_nm, 0.0, 0.0], [0.0, box_nm, 0.0], [0.0, 0.0, box_nm]
    )

    nb_p = openmm.NonbondedForce()
    nb_p.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb_p.setCutoffDistance(0.9 * unit.nanometer)
    nb_p.setPMEParameters(3.2, 48, 48, 48)
    nb_p.setUseDispersionCorrection(False)

    nb_r = openmm.NonbondedForce()
    nb_r.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb_r.setCutoffDistance(0.9 * unit.nanometer)
    nb_r.setPMEParameters(3.2, 48, 48, 48)
    nb_r.setUseDispersionCorrection(False)

    cnb_r = openmm.CustomNonbondedForce(
        "4*eps*((sig/r)^12 - (sig/r)^6); sig=sqrt(sig1*sig2); eps=sqrt(eps1*eps2)"
    )
    cnb_r.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cnb_r.setCutoffDistance(0.9 * unit.nanometer)
    cnb_r.setUseLongRangeCorrection(False)
    cnb_r.addPerParticleParameter("sig")
    cnb_r.addPerParticleParameter("eps")

    pos0 = np.zeros((n_atoms, 3), dtype=np.float64)
    # Deliberately widely contrasting sigmas (0.22 nm vs 0.44 nm) so LB (0.33) != geometric (0.3111)
    for i in range(n_atoms):
        sys_prolix.addParticle(14.0)
        sys_ref.addParticle(14.0)
        q = 0.30 if (i % 2 == 0) else -0.30
        sig = 0.22 if (i % 2 == 0) else 0.44
        eps = 0.35 if (i % 3 == 0) else 0.85
        nb_p.addParticle(q, sig, eps)
        nb_r.addParticle(q, 0.30, 0.0)
        cnb_r.addParticle([sig, eps])
        pos0[i] = [
            0.40 + 0.48 * (i % 4),
            0.40 + 0.48 * ((i // 4) % 2),
            0.40 + 0.48 * (i // 8),
        ]

    sys_prolix.addForce(nb_p)
    sys_ref.addForce(nb_r)
    sys_ref.addForce(cnb_r)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond
    )
    ctx = ProlixContext(
        sys_prolix,
        integ,
        config=ProlixOpenMMConfig(
            precision="mixed",
            lj_combining_rule="geometric",
            nonbonded_engine="morton_tile_pairs",
            use_pallas=True,
            pallas_interpret=True,
            use_dispersion_correction=False,
        ),
    )
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=55)

    assert ctx._lj_combining_rule == "geometric"
    assert ctx._can_use_morton_step_runner(False, None, None, "baoab")

    ctx.step(5)
    assert ctx._morton_runner is not None
    assert ctx._can_use_morton_obs()
    _assert_reference_energy_and_forces(sys_ref, ctx, e_atol_kj=0.06, f_atol_kj_nm=0.15)


def test_morton_fastpath_ljpme_rollout_and_npt_parity() -> None:
    """`NonbondedForce.LJPME` + exclusions + 1-4 + NPT executes on `_morton_runner` matching Reference."""
    n_chains = 4  # 4 * 4 = 16 atoms (with 1-2, 1-3, and 1-4 pairs)
    box_nm = 2.5
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        [box_nm, 0.0, 0.0], [0.0, box_nm, 0.0], [0.0, 0.0, box_nm]
    )
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    torsions = openmm.PeriodicTorsionForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    nb.setPMEParameters(3.2, 24, 24, 24)
    nb.setLJPMEParameters(3.0, 20, 20, 20)

    pos0 = np.zeros((4 * n_chains, 3), dtype=np.float64)
    bond_pairs: list[tuple[int, int]] = []
    for c in range(n_chains):
        b = 4 * c
        for k in range(4):
            sys.addParticle(12.0)
            q = (0.22 if k % 2 == 0 else -0.22)
            sig = 0.28 + 0.04 * k
            eps = 0.30 + 0.15 * k
            nb.addParticle(q, sig, eps)
            pos0[b + k] = [
                0.45 + 0.15 * k,
                0.45 + 0.50 * (c % 2),
                0.45 + 0.50 * (c // 2) + 0.03 * (k % 2),
            ]
        bonds.addBond(b, b + 1, 0.152, 220000.0)
        bonds.addBond(b + 1, b + 2, 0.152, 220000.0)
        bonds.addBond(b + 2, b + 3, 0.152, 220000.0)
        bond_pairs.extend([(b, b + 1), (b + 1, b + 2), (b + 2, b + 3)])
        angles.addAngle(b, b + 1, b + 2, 1.95, 420.0)
        angles.addAngle(b + 1, b + 2, b + 3, 1.95, 420.0)
        torsions.addTorsion(b, b + 1, b + 2, b + 3, 3, 0.0, 6.5)

    nb.createExceptionsFromBonds(bond_pairs, 0.83333333, 0.5)
    sys.addForce(bonds)
    sys.addForce(angles)
    sys.addForce(torsions)
    sys.addForce(nb)
    sys.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 3))

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond
    )
    ctx = ProlixContext(
        sys,
        integ,
        config=ProlixOpenMMConfig(
            precision="mixed",
            nonbonded_engine="morton_tile_pairs",
            use_pallas=True,
            pallas_interpret=True,
            nl_update_every=2,
        ),
    )
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=303)

    baro_info = ctx._converted_system.barostat
    assert ctx._effective_nonbonded_method() == "lj_pme"
    assert ctx._can_use_morton_step_runner(True, "mc", baro_info, "baoab")

    ctx.step(6)
    assert ctx._morton_runner is not None
    assert ctx._can_use_morton_obs()
    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.06, f_atol_kj_nm=0.18)


def test_gbsa_obc2_step_runner_restrained_and_constrained_rollout_parity() -> None:
    """Implicit-solvent `OBC2 GBSA` + constraints + `PositionRestraints` runs on `_gbsa_runner`."""
    n_atoms = 12
    sys = openmm.System()
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    torsions = openmm.PeriodicTorsionForce()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    gb = openmm.GBSAOBCForce()
    gb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    gb.setSoluteDielectric(1.0)
    gb.setSolventDielectric(78.5)
    gb.setSurfaceAreaEnergy(2.25936)

    cef = openmm.CustomExternalForce(
        "0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    cef.addPerParticleParameter("x0")
    cef.addPerParticleParameter("y0")
    cef.addPerParticleParameter("z0")
    cef.addPerParticleParameter("k")

    pos0 = np.zeros((n_atoms, 3), dtype=np.float64)
    bond_pairs: list[tuple[int, int]] = []
    # Build 4 heavy-atom backbone chain (0-1-2-3) each with 2 constrained H atoms (star clusters)
    # Atoms: [C0, H0a, H0b, C1, H1a, H1b, C2, H2a, H2b, C3, H3a, H3b]
    for blk in range(4):
        c_idx = sys.addParticle(12.011)
        h1_idx = sys.addParticle(1.008)
        h2_idx = sys.addParticle(1.008)
        assert (c_idx, h1_idx, h2_idx) == (3 * blk, 3 * blk + 1, 3 * blk + 2)

        q_c = -0.24 if blk % 2 == 0 else 0.16
        q_h = 0.12 if blk % 2 == 0 else -0.08
        nb.addParticle(q_c, 0.34, 0.45)
        nb.addParticle(q_h, 0.25, 0.065)
        nb.addParticle(q_h, 0.25, 0.065)
        gb.addParticle(q_c, 0.17, 0.72)
        gb.addParticle(q_h, 0.12, 0.85)
        gb.addParticle(q_h, 0.12, 0.85)

        sys.addConstraint(c_idx, h1_idx, 0.109)
        sys.addConstraint(c_idx, h2_idx, 0.109)
        bond_pairs.extend([(c_idx, h1_idx), (c_idx, h2_idx)])
        angles.addAngle(h1_idx, c_idx, h2_idx, 1.91, 300.0)

        cx = -0.25 + 0.152 * blk
        cy = 0.04 * (blk % 2)
        pos0[c_idx] = [cx, cy, 0.0]
        pos0[h1_idx] = [cx, cy + 0.089, 0.063]
        pos0[h2_idx] = [cx, cy + 0.089, -0.063]

        if blk > 0:
            prev_c = 3 * (blk - 1)
            bonds.addBond(prev_c, c_idx, 0.152, 240000.0)
            bond_pairs.append((prev_c, c_idx))
            angles.addAngle(prev_c, c_idx, h1_idx, 1.91, 350.0)
        if blk > 1:
            angles.addAngle(3 * (blk - 2), 3 * (blk - 1), c_idx, 1.95, 450.0)
        if blk > 2:
            torsions.addTorsion(0, 3, 6, 9, 3, 0.0, 8.0)

    # Restrain C0 at negative x0 = -0.25 nm
    cef.addParticle(0, [-0.25, 0.0, 0.0, 1500.0])

    nb.createExceptionsFromBonds(bond_pairs, 0.83333333, 0.5)
    sys.addForce(bonds)
    sys.addForce(angles)
    sys.addForce(torsions)
    sys.addForce(nb)
    sys.addForce(gb)
    sys.addForce(cef)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 2.0 / unit.picosecond, 0.001 * unit.picosecond
    )
    ctx = ProlixContext(
        sys,
        integ,
        config=ProlixOpenMMConfig(
            precision="mixed",
            nonbonded_engine="gbsa_tile",
            use_pallas=True,
            pallas_interpret=True,
            cmm_frequency=4,
        ),
    )
    ctx.setPositions(pos0)
    ctx.setVelocitiesToTemperature(300.0, randomSeed=404)

    assert ctx._can_use_gbsa_step_runner(False, "baoab")

    ctx.step(8)
    assert ctx._gbsa_runner is not None

    pos_after = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    assert pos_after[0, 0] < -0.15
    for blk in range(4):
        c_i = 3 * blk
        assert abs(np.linalg.norm(pos_after[c_i + 1] - pos_after[c_i]) - 0.109) < 1e-5
        assert abs(np.linalg.norm(pos_after[c_i + 2] - pos_after[c_i]) - 0.109) < 1e-5

    # Verify cached _gbsa_fx/fy/fz forces from _gbsa_runner match OpenMM Reference directly
    f_runner_omm = (
        np.stack(
            [
                np.asarray(ctx._gbsa_fx)[:n_atoms],
                np.asarray(ctx._gbsa_fy)[:n_atoms],
                np.asarray(ctx._gbsa_fz)[:n_atoms],
            ],
            axis=-1,
        )
        * 41.84
    )
    ref_ctx = _make_reference_context(sys)
    ref_ctx.setPositions(pos_after)
    f_ref_omm = (
        ref_ctx.getState(getForces=True)
        .getForces(asNumpy=True)
        .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    )
    np.testing.assert_allclose(f_runner_omm, f_ref_omm, atol=0.35, rtol=3e-3)
    _assert_reference_energy_and_forces(sys, ctx, e_atol_kj=0.25, f_atol_kj_nm=0.35)
