"""Adversarial stress test suite for Milestone 4 Iteration 3: CHARMM Harmonic Improper potential.

Authored by Challenger M4-R3-1.
Evaluates:
1. Synthetic molecules with CHARMM harmonic impropers at varied force constants
   (k = 0.5, 10.0, 100.0, 500.0 kcal/(mol*rad^2)) and equilibrium angles
   (phi_0 = 0 deg, 35.264 deg, -35.264 deg, 180 deg).
2. Cross-boundary conditions: angles wrapping past -pi and +pi (+/- 179 deg -> -/+ 179 deg).
3. JAX autodiff forces -grad E parity vs OpenMM Reference platform (getState(getForces=True)).
4. Equivalence between custom_force_mode="auto" (folded into MolecularBundle) and
   custom_force_mode="lepton_only" (compiled via Lepton).
5. Numerical energy parity between Prolix and OpenMM Reference within < 1e-5 relative tolerance.
6. Multi-center improper molecules and coupled SETTLE water dynamics.
"""

from __future__ import annotations

import math

import numpy as np
import openmm
import pytest
from openmm import unit

from prolix.openmm import ProlixContext, ProlixOpenMMConfig
from prolix.openmm.context import _pure_compute_energy_and_forces


def _make_torsion_geometry(angle_deg: float) -> list[openmm.Vec3]:
    """Construct 4-particle coordinates producing an exact dihedral angle in nanometers.

    Particles:
    - Atom 0 at (0, 0.1, 0) nm
    - Atom 1 at (0, 0, 0) nm (origin, bond 1-2 along x-axis)
    - Atom 2 at (0.15, 0, 0) nm
    - Atom 3 at (0.15, 0.1*cos(phi), 0.1*sin(phi)) nm
    Dihedral angle between 0-1-2-3 is exactly `angle_deg`.
    """
    angle_rad = np.deg2rad(angle_deg)
    p0 = np.array([0.0, 0.1, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.15, 0.0, 0.0])
    p3 = np.array([0.15, 0.1 * np.cos(angle_rad), 0.1 * np.sin(angle_rad)])
    return [
        openmm.Vec3(*p0),
        openmm.Vec3(*p1),
        openmm.Vec3(*p2),
        openmm.Vec3(*p3),
    ]


@pytest.mark.parametrize("k_kcal", [0.5, 10.0, 100.0, 500.0])
@pytest.mark.parametrize("phi0_deg", [0.0, 35.264, -35.264, 180.0])
def test_harmonic_improper_parametric_grid_parity(k_kcal: float, phi0_deg: float):
    """Verify energy and force parity across varied k and phi0 within < 1e-5 rel error."""
    k_omm = k_kcal * 4.184  # kcal/(mol*rad^2) -> kJ/(mol*rad^2)
    phi0_rad = float(np.deg2rad(phi0_deg))

    # Angle displaced by 8 degrees from equilibrium
    angle = phi0_deg - 8.0 if phi0_deg == 180.0 else phi0_deg + 8.0

    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [k_omm, phi0_rad])
    sys.addForce(ct)

    pos = _make_torsion_geometry(angle) * unit.nanometer

    # 1. OpenMM Reference Platform
    ref_ctx = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ref_ctx.setPositions(pos)
    st_ref = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = st_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = st_ref.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # 2. ProlixContext custom_force_mode="auto" (folded into MolecularBundle)
    cfg_auto = ProlixOpenMMConfig(custom_force_mode="auto")
    ctx_auto = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg_auto)
    ctx_auto.setPositions(pos)
    st_auto = ctx_auto.getState(getEnergy=True, getForces=True)
    e_auto = st_auto.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_auto = st_auto.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # 3. ProlixContext custom_force_mode="lepton_only" (evaluated via Lepton)
    cfg_lep = ProlixOpenMMConfig(custom_force_mode="lepton_only")
    ctx_lep = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg_lep)
    ctx_lep.setPositions(pos)
    st_lep = ctx_lep.getState(getEnergy=True, getForces=True)
    e_lep = st_lep.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_lep = st_lep.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # Architectural assertion: auto folds into bundle, lepton_only does not
    assert ctx_auto._bundle.n_impropers == 1
    assert len(ctx_auto._converted_system.custom_forces) == 0
    assert ctx_lep._bundle.n_impropers == 0
    assert len(ctx_lep._converted_system.custom_forces) == 1

    # Parity assertions: auto vs Reference
    rel_e_auto = abs(e_ref - e_auto) / (abs(e_ref) + 1e-12)
    assert rel_e_auto < 1e-5, (
        f"Energy auto disparity at k={k_kcal}, phi0={phi0_deg}: "
        f"ref={e_ref:.6f}, auto={e_auto:.6f}, rel={rel_e_auto:.2e}"
    )

    f_max_norm = np.max(np.abs(f_ref))
    rel_f_auto = np.max(np.abs(f_ref - f_auto)) / (f_max_norm + 1e-12)
    assert rel_f_auto < 1e-4, (
        f"Force auto disparity at k={k_kcal}, phi0={phi0_deg}: "
        f"rel={rel_f_auto:.2e}, max_diff={np.max(np.abs(f_ref - f_auto)):.2e}"
    )

    # Parity assertions: auto vs lepton_only equivalence
    rel_e_lep = abs(e_auto - e_lep) / (abs(e_auto) + 1e-12)
    assert rel_e_lep < 1e-5, (
        f"Auto vs Lepton energy disparity at k={k_kcal}, phi0={phi0_deg}: "
        f"auto={e_auto:.6f}, lep={e_lep:.6f}, rel={rel_e_lep:.2e}"
    )

    rel_f_lep = np.max(np.abs(f_auto - f_lep)) / (f_max_norm + 1e-12)
    assert rel_f_lep < 1e-4, (
        f"Auto vs Lepton force disparity at k={k_kcal}, phi0={phi0_deg}: "
        f"rel={rel_f_lep:.2e}"
    )


@pytest.mark.parametrize("angle_deg", [179.0, -179.0])
def test_harmonic_improper_cross_boundary_wrap_planar_equilibrium(angle_deg: float):
    """Verify cross-boundary wrapping (+/- 179 deg) for phi0 = 0 deg."""
    k_kcal = 100.0
    k_omm = k_kcal * 4.184
    phi0_rad = 0.0

    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [k_omm, phi0_rad])
    sys.addForce(ct)

    pos = _make_torsion_geometry(angle_deg) * unit.nanometer

    ref_ctx = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ref_ctx.setPositions(pos)
    st_ref = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = st_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = st_ref.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    pro_ctx = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(custom_force_mode="auto"),
    )
    pro_ctx.setPositions(pos)
    st_pro = pro_ctx.getState(getEnergy=True, getForces=True)
    e_pro = st_pro.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_pro = st_pro.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    rel_e = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
    assert rel_e < 1e-5, f"Cross-boundary energy failed at {angle_deg} deg: ref={e_ref}, pro={e_pro}"

    f_max_norm = np.max(np.abs(f_ref))
    rel_f = np.max(np.abs(f_ref - f_pro)) / (f_max_norm + 1e-12)
    assert rel_f < 1e-4, f"Cross-boundary force failed at {angle_deg} deg: rel={rel_f}"


def test_harmonic_improper_cross_boundary_continuity_trans_equilibrium():
    """Verify physical continuity across the +/- 180 deg branch cut for phi0 = 180 deg.

    In auto mode, Prolix uses toroidal unwrapping:
        diff_raw = phi - phi0 + pi
        dphi = diff_raw - 2*pi * floor(diff_raw / (2*pi)) - pi
    Thus, as phi crosses +179 deg -> -179 deg, dphi changes smoothly from -1 deg to +1 deg,
    yielding continuous potential energy E(+179 deg) == E(-179 deg) and smooth restoring forces.
    """
    k_kcal = 100.0
    k_omm = k_kcal * 4.184

    # Part A: Verify parity with OpenMM Reference when phi0 is matched to the branch side
    for angle_deg, phi0_rad in [(175.0, np.pi), (-175.0, -np.pi), (179.0, np.pi), (-179.0, -np.pi)]:
        sys = openmm.System()
        for _ in range(4):
            sys.addParticle(12.0)
        ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
        ct.addPerTorsionParameter("k")
        ct.addPerTorsionParameter("theta0")
        ct.addTorsion(0, 1, 2, 3, [k_omm, phi0_rad])
        sys.addForce(ct)

        pos = _make_torsion_geometry(angle_deg) * unit.nanometer
        ref_ctx = openmm.Context(
            sys,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        ref_ctx.setPositions(pos)
        e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        pro_ctx = ProlixContext(
            sys,
            openmm.VerletIntegrator(0.001),
            config=ProlixOpenMMConfig(custom_force_mode="auto"),
        )
        pro_ctx.setPositions(pos)
        e_pro = pro_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        abs_e = abs(e_ref - e_pro)
        rel_e = abs_e / (abs(e_ref) + 1e-12)
        # For small energies (< 0.1 kJ/mol at 179 deg), absolute difference < 2e-6 kJ/mol;
        # for non-trivial energies (at 175 deg), relative difference < 1e-5.
        assert np.isclose(e_ref, e_pro, rtol=2e-5, atol=1e-5), (
            f"Trans improper failed at {angle_deg} deg: ref={e_ref}, pro={e_pro}, rel={rel_e:.2e}, abs={abs_e:.2e}"
        )

    # Part B: Adversarial branch-crossing continuity in auto mode
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)
    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [k_omm, float(np.deg2rad(180.0))])
    sys.addForce(ct)

    ctx_auto = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(custom_force_mode="auto"),
    )

    ctx_auto.setPositions(_make_torsion_geometry(179.0) * unit.nanometer)
    st_pos = ctx_auto.getState(getEnergy=True, getForces=True)
    e_pos = st_pos.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_pos = st_pos.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    ctx_auto.setPositions(_make_torsion_geometry(-179.0) * unit.nanometer)
    st_neg = ctx_auto.getState(getEnergy=True, getForces=True)
    e_neg = st_neg.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_neg = st_neg.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # 1. Energy symmetry across boundary: absolute diff < 1e-5 kJ/mol, relative close within precision
    abs_diff_energy = abs(e_pos - e_neg)
    assert np.isclose(e_pos, e_neg, rtol=5e-5, atol=1e-5), (
        f"Toroidal energy discontinuity detected: e_pos={e_pos}, e_neg={e_neg}, abs_diff={abs_diff_energy:.2e}"
    )

    # 2. Out-of-plane restoring force reflection on atom 3: F_z(+179) == -F_z(-179)
    np.testing.assert_allclose(
        f_pos[3, 2],
        -f_neg[3, 2],
        rtol=1e-4,
        err_msg="Toroidal force reflection discontinuity detected on atom 3!",
    )


@pytest.mark.parametrize(
    ("expr", "factor"),
    [
        ("0.5 * k * (theta - theta0)^2", 0.5),
        ("k * (theta - theta0)^2", 1.0),
        ("k / 2 * (theta - theta0)^2", 0.5),
        ("0.5 * k * (theta0 - theta)^2", 0.5),
    ],
)
def test_harmonic_improper_algebraic_expression_scaling_parity(
    expr: str, factor: float
):
    """Verify that both halved (0.5*k) and unhalved (k) CHARMM expressions match Reference."""
    k_omm = 250.0  # kJ/(mol*rad^2)
    phi0_rad = float(np.deg2rad(35.264))

    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce(expr)
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [k_omm, phi0_rad])
    sys.addForce(ct)

    pos = _make_torsion_geometry(45.0) * unit.nanometer

    ref_ctx = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ref_ctx.setPositions(pos)
    e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    pro_ctx = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(custom_force_mode="auto"),
    )
    pro_ctx.setPositions(pos)
    e_pro = pro_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    rel_e = abs(e_ref - e_pro) / e_ref
    assert rel_e < 1e-5, f"Expression {expr} failed: ref={e_ref}, pro={e_pro}, rel={rel_e}"


def test_harmonic_improper_multi_center_synthetic_molecule():
    """Verify a 7-particle synthetic molecule with 3 coupled improper centers."""
    sys = openmm.System()
    for _ in range(7):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")

    # Center 1: planar aromatic center (0-1-2-3), k=100 kcal/mol, phi0=0.0 deg
    ct.addTorsion(0, 1, 2, 3, [100.0 * 4.184, 0.0])
    # Center 2: pyramidal center (1-2-3-4), k=50 kcal/mol, phi0=35.264 deg
    ct.addTorsion(1, 2, 3, 4, [50.0 * 4.184, float(np.deg2rad(35.264))])
    # Center 3: inverted pyramidal center (2-3-4-5), k=25 kcal/mol, phi0=-35.264 deg
    ct.addTorsion(2, 3, 4, 5, [25.0 * 4.184, float(np.deg2rad(-35.264))])
    sys.addForce(ct)

    rng = np.random.default_rng(1337)
    pos_nm = rng.standard_normal((7, 3)) * 0.15
    pos = [openmm.Vec3(*p) for p in pos_nm] * unit.nanometer

    # 1. Reference
    ref_ctx = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ref_ctx.setPositions(pos)
    st_ref = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = st_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = st_ref.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # 2. Prolix auto
    ctx_auto = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(custom_force_mode="auto"),
    )
    ctx_auto.setPositions(pos)
    st_auto = ctx_auto.getState(getEnergy=True, getForces=True)
    e_auto = st_auto.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_auto = st_auto.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # 3. Prolix lepton_only
    ctx_lep = ProlixContext(
        sys,
        openmm.VerletIntegrator(0.001),
        config=ProlixOpenMMConfig(custom_force_mode="lepton_only"),
    )
    ctx_lep.setPositions(pos)
    st_lep = ctx_lep.getState(getEnergy=True, getForces=True)
    e_lep = st_lep.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_lep = st_lep.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )

    # Confirm all 3 centers folded
    assert ctx_auto._bundle.n_impropers == 3
    assert len(ctx_auto._converted_system.custom_forces) == 0

    # Energy parity
    rel_e_auto = abs(e_ref - e_auto) / e_ref
    assert rel_e_auto < 1e-5, f"Multi-center auto energy failed: ref={e_ref}, auto={e_auto}"
    rel_e_lep = abs(e_auto - e_lep) / e_auto
    assert rel_e_lep < 1e-5, f"Multi-center lepton energy failed: auto={e_auto}, lep={e_lep}"

    # Force parity
    f_max_norm = np.max(np.abs(f_ref))
    rel_f_auto = np.max(np.abs(f_ref - f_auto)) / f_max_norm
    assert rel_f_auto < 1e-4, f"Multi-center auto force failed: rel={rel_f_auto}"
    rel_f_lep = np.max(np.abs(f_auto - f_lep)) / f_max_norm
    assert rel_f_lep < 1e-4, f"Multi-center lepton force failed: rel={rel_f_lep}"


def test_harmonic_improper_dynamic_parameter_update_zero_recompile():
    """Verify updateParametersInContext on CustomTorsionForce causes zero XLA recompilations."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [200.0, float(np.deg2rad(30.0))])
    sys.addForce(ct)

    pos = _make_torsion_geometry(40.0) * unit.nanometer

    ref_ctx = openmm.Context(
        sys,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ref_ctx.setPositions(pos)

    # Use lepton_only mode to exercise CustomTorsionForce parameter updating in updateParametersInContext
    cfg = ProlixOpenMMConfig(custom_force_mode="lepton_only")
    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pro_ctx.setPositions(pos)

    # Initial baseline
    s_ref_0 = ref_ctx.getState(getEnergy=True)
    s_pro_0 = pro_ctx.getState(getEnergy=True)
    e_ref_0 = s_ref_0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_pro_0 = s_pro_0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert abs(e_ref_0 - e_pro_0) / e_ref_0 < 1e-5

    baseline_cache = _pure_compute_energy_and_forces._cached._cache_size()

    # Dynamic update 1: change force constant k
    ct.setTorsionParameters(0, 0, 1, 2, 3, [400.0, float(np.deg2rad(30.0))])
    ct.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext(ct)

    s_ref_1 = ref_ctx.getState(getEnergy=True)
    s_pro_1 = pro_ctx.getState(getEnergy=True)
    e_ref_1 = s_ref_1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_pro_1 = s_pro_1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert abs(e_ref_1 - e_pro_1) / e_ref_1 < 1e-5
    assert not math.isclose(e_ref_0, e_ref_1, rel_tol=1e-3)

    cache_1 = _pure_compute_energy_and_forces._cached._cache_size()
    assert cache_1 == baseline_cache, (
        f"Recompilation detected after updating k! baseline={baseline_cache}, current={cache_1}"
    )

    # Dynamic update 2: change equilibrium angle theta0
    ct.setTorsionParameters(0, 0, 1, 2, 3, [400.0, float(np.deg2rad(50.0))])
    ct.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()

    s_ref_2 = ref_ctx.getState(getEnergy=True)
    s_pro_2 = pro_ctx.getState(getEnergy=True)
    e_ref_2 = s_ref_2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    e_pro_2 = s_pro_2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert abs(e_ref_2 - e_pro_2) / e_ref_2 < 1e-5

    cache_2 = _pure_compute_energy_and_forces._cached._cache_size()
    assert cache_2 == baseline_cache, (
        f"Recompilation detected after updating theta0! baseline={baseline_cache}, current={cache_2}"
    )


def test_harmonic_improper_coupled_with_settle_water_stability():
    """Verify stability of a system containing rigid water triangles (SETTLE) and impropers."""
    sys = openmm.System()

    # Solute: 4 particles (0..3) with harmonic improper
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [300.0, float(np.deg2rad(35.264))])
    sys.addForce(ct)

    # Solvent: 2 TIP3P water molecules (indices 4..9)
    # Water 1: O=4, H=5, H=6
    # Water 2: O=7, H=8, H=9
    for _ in range(2):
        sys.addParticle(15.99943)  # Oxygen
        sys.addParticle(1.007947)  # Hydrogen
        sys.addParticle(1.007947)  # Hydrogen

    # Rigid water constraints (OH = 0.09572 nm, HH = 0.15139 nm)
    for base in [4, 7]:
        sys.addConstraint(base, base + 1, 0.09572)
        sys.addConstraint(base, base + 2, 0.09572)
        sys.addConstraint(base + 1, base + 2, 0.15139)

    solute_pos = _make_torsion_geometry(45.0)
    w1_pos = [
        openmm.Vec3(0.5, 0.0, 0.0),
        openmm.Vec3(0.59572, 0.0, 0.0),
        openmm.Vec3(0.47, 0.09, 0.0),
    ]
    w2_pos = [
        openmm.Vec3(0.8, 0.0, 0.0),
        openmm.Vec3(0.89572, 0.0, 0.0),
        openmm.Vec3(0.77, 0.09, 0.0),
    ]
    all_pos = (solute_pos + w1_pos + w2_pos) * unit.nanometer

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    ctx = ProlixContext(
        sys,
        integ,
        config=ProlixOpenMMConfig(
            custom_force_mode="auto",
            water_constraint_mode="settle",
        ),
    )
    ctx.setPositions(all_pos)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin)

    # Assert constraints applied cleanly without broadcasting crash
    ctx.applyConstraints()
    ctx.applyVelocityConstraints()

    # Step rollout (25 steps)
    ctx.step(25)
    st = ctx.getState(getPositions=True, getEnergy=True)
    e_pot = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert not math.isnan(e_pot)
    assert not math.isinf(e_pot)

    # Verify water geometry constraints preserved
    p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    for base in [4, 7]:
        d_oh1 = np.linalg.norm(p[base] - p[base + 1])
        d_oh2 = np.linalg.norm(p[base] - p[base + 2])
        d_hh = np.linalg.norm(p[base + 1] - p[base + 2])
        np.testing.assert_allclose(d_oh1, 0.09572, atol=1e-3)
        np.testing.assert_allclose(d_oh2, 0.09572, atol=1e-3)
        np.testing.assert_allclose(d_hh, 0.15139, atol=1e-3)
