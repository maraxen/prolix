"""Adversarial stress test suite for Milestone 4 Iteration 2 (Challenger M4-R2-2).

Empirical and adversarial verification of:
1. Prior challenger stress suite execution.
2. Periodic bonded minimum-image displacement across bonds, angles, torsions, and nonbonded pairs.
3. Periodic torsions and harmonic impropers across non-zero phases (30°, 45°, 90°, 120°).
4. Water bond rigidity with SETTLE over 100-step (200 fs) Langevin rollouts and velocity constraint broadcasting.
5. Dynamic volume fluctuations and stability in NPT ("mc" and "scr" modes).
6. Multi-instance custom forces dynamic parameter updates with zero XLA recompilations.
"""

from __future__ import annotations

import numpy as np
import openmm
from openmm import app, unit
import pytest

from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation
from prolix.openmm.context import _pure_compute_energy_and_forces, _pure_step_rollout


# =============================================================================
# Task 2: Periodic Bonded Simulations without TypeError & Numerical Parity
# =============================================================================


def test_periodic_bonded_terms_minimum_image_dr_and_parity():
    """Verify _minimum_image_dr runs without TypeError across bonds, angles, torsions, and nonbonded pairs.

    Also tests numerical parity against OpenMM Reference Platform with periodic boundary conditions.
    """
    sys = openmm.System()
    box_l = 2.0
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_l, 0, 0),
        openmm.Vec3(0, box_l, 0),
        openmm.Vec3(0, 0, box_l),
    )
    for _ in range(4):
        sys.addParticle(12.0)

    # Bond spanning periodic boundary: atom 0 at x=0.05, atom 1 at x=1.95 (effective dx = -0.1 nm via PBC)
    b = openmm.HarmonicBondForce()
    b.setUsesPeriodicBoundaryConditions(True)
    b.addBond(0, 1, 0.10 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    # Angle spanning periodic boundary: atom 2 at x=1.85 (angle 0-1-2)
    ang = openmm.HarmonicAngleForce()
    ang.setUsesPeriodicBoundaryConditions(True)
    ang.addAngle(0, 1, 2, 2.0 * unit.radian, 50000.0 * unit.kilojoules_per_mole / unit.radian**2)
    sys.addForce(ang)

    # Torsion spanning periodic boundary: atom 3 at x=1.75
    tor = openmm.PeriodicTorsionForce()
    tor.setUsesPeriodicBoundaryConditions(True)
    tor.addTorsion(0, 1, 2, 3, 2, 0.5 * unit.radian, 25.0 * unit.kilojoules_per_mole)
    sys.addForce(tor)

    pos = [
        openmm.Vec3(0.05, 1.0, 1.0),
        openmm.Vec3(1.95, 1.0, 1.0),
        openmm.Vec3(1.85, 1.05, 1.0),
        openmm.Vec3(1.75, 1.08, 1.05),
    ] * unit.nanometer

    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref_ctx.setPositions(pos)
    s_ref = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref = s_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_ref = s_ref.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    pro_ctx.setPositions(pos)
    s_pro = pro_ctx.getState(getEnergy=True, getForces=True)
    e_pro = s_pro.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f_pro = s_pro.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

    rel_e = abs(e_ref - e_pro) / abs(e_ref)
    denom_f = float(np.max(np.abs(f_ref)) + 1e-12)
    rel_f = float(np.max(np.abs(f_ref - f_pro))) / denom_f

    # Assert zero TypeError and exact parity (< 1e-4 relative) across periodic bonded terms
    assert rel_e < 1e-4, f"Periodic bonded energy parity failed: ref={e_ref}, pro={e_pro}, rel_e={rel_e}"
    assert rel_f < 1e-4, f"Periodic bonded force parity failed: rel_f={rel_f}"


# =============================================================================
# Task 3: Periodic Torsions & Harmonic Impropers with Non-Zero Phase Values
# =============================================================================


@pytest.mark.parametrize("phase_deg", [30.0, 45.0, 90.0, 120.0])
def test_periodic_torsion_varying_phases_parity(phase_deg: float):
    """Verify PeriodicTorsionForce parity vs OpenMM Reference across non-zero phases."""
    phase_rad = float(np.deg2rad(phase_deg))
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    tor = openmm.PeriodicTorsionForce()
    tor.addTorsion(0, 1, 2, 3, 2, phase_rad * unit.radian, 25.0 * unit.kilojoules_per_mole)
    sys.addForce(tor)

    pos = [
        openmm.Vec3(0.0, 0.1, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.15, 0.0, 0.0),
        openmm.Vec3(0.15, 0.1, 0.05),
    ] * unit.nanometer

    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref_ctx.setPositions(pos)
    e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    pro_ctx.setPositions(pos)
    e_pro = pro_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    rel_diff = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
    assert rel_diff < 1e-5, f"Periodic torsion parity failed at {phase_deg}°: ref={e_ref}, pro={e_pro}, rel={rel_diff}"


@pytest.mark.parametrize("phase_deg", [30.0, 45.0, 90.0, 120.0])
def test_custom_periodic_torsion_auto_mode_parity(phase_deg: float):
    """Verify CustomTorsionForce with periodic expression in custom_force_mode='auto' matches Reference."""
    phase_rad = float(np.deg2rad(phase_deg))
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("k * (1 + cos(2 * theta - phase))")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("phase")
    ct.addTorsion(0, 1, 2, 3, [25.0, phase_rad])
    sys.addForce(ct)

    pos = [
        openmm.Vec3(-0.1, 0.12, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.12, 0.0, 0.0),
        openmm.Vec3(0.18, 0.1, 0.08),
    ] * unit.nanometer

    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref_ctx.setPositions(pos)
    e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    cfg = ProlixOpenMMConfig(custom_force_mode="auto")
    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pro_ctx.setPositions(pos)
    e_pro = pro_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    rel_diff = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
    assert rel_diff < 1e-5, f"Custom periodic torsion failed at {phase_deg}°: ref={e_ref}, pro={e_pro}, rel={rel_diff}"


@pytest.mark.parametrize("theta0_deg", [30.0, 45.0, 90.0, 120.0])
def test_harmonic_improper_lepton_only_parity(theta0_deg: float):
    """Verify CustomTorsionForce with harmonic improper in custom_force_mode='lepton_only' matches Reference."""
    theta0_rad = float(np.deg2rad(theta0_deg))
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [300.0, theta0_rad])
    sys.addForce(ct)

    pos = [
        openmm.Vec3(-0.1, 0.12, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.12, 0.0, 0.0),
        openmm.Vec3(0.18, 0.1, 0.08),
    ] * unit.nanometer

    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref_ctx.setPositions(pos)
    e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    cfg = ProlixOpenMMConfig(custom_force_mode="lepton_only")
    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pro_ctx.setPositions(pos)
    e_pro = pro_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    rel_diff = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
    assert rel_diff < 1e-5, f"Harmonic improper [lepton_only] failed at {theta0_deg}°: ref={e_ref}, pro={e_pro}, rel={rel_diff}"


def test_harmonic_improper_auto_mode_zero_energy_bug():
    """Adversarial challenge: Expose the harmonic improper zero-energy bug in custom_force_mode='auto'.

    In `pattern_matcher.py`, harmonic impropers are matched and folded into `bundle.improper_params`
    with `improper_is_periodic=False` and parameters `[k_akma, theta0, 0.0]`.
    However, `ProlixContext._eval_bonded_terms` unconditionally treats `bundle.improper_params` as
    periodic dihedrals `per = i_params[:, 0]`, `phase = i_params[:, 1]`, `k_i = i_params[:, 2]`,
    reading `k_i = 0.0` and yielding `e_pro = 0.0 kJ/mol` (100% relative error).
    """
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    ct = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    ct.addPerTorsionParameter("k")
    ct.addPerTorsionParameter("theta0")
    ct.addTorsion(0, 1, 2, 3, [300.0, float(np.deg2rad(30.0))])
    sys.addForce(ct)

    pos = [
        openmm.Vec3(-0.1, 0.12, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.12, 0.0, 0.0),
        openmm.Vec3(0.18, 0.1, 0.08),
    ] * unit.nanometer

    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref_ctx.setPositions(pos)
    e_ref = ref_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert e_ref > 1.0, f"Expected non-trivial Reference energy, got {e_ref}"

    cfg = ProlixOpenMMConfig(custom_force_mode="auto")
    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pro_ctx.setPositions(pos)
    e_pro = pro_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    rel_diff = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
    # This assertion mandates the < 1e-5 parity requirement from the dispatch prompt.
    # It demonstrates the bug because e_pro is currently 0.0!
    assert rel_diff < 1e-5, (
        f"Harmonic improper in auto mode failed: ref={e_ref:.5f}, pro={e_pro:.5f}, rel_diff={rel_diff:.4f}. "
        f"Bug in context.py: _eval_bonded_terms ignores bundle.improper_is_periodic and reads k_i=0.0!"
    )


# =============================================================================
# Task 4: Water Bond Rigidity with SETTLE & Velocity Constraints Broadcasting Bug
# =============================================================================


def test_water_bond_rigidity_settle_rollout_100_steps():
    """Verify 100-step (200 fs) Langevin rollout on water with SETTLE keeps r_OH and r_HH rigid (< 1e-4 nm)."""
    modeller = app.Modeller(app.Topology(), [])
    ff = app.ForceField("amber14/tip3p.xml")
    modeller.addSolvent(ff, boxSize=openmm.Vec3(1.5, 1.5, 1.5) * unit.nanometers, model="tip3p")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=0.7 * unit.nanometers,
        rigidWater=True,
    )
    integ = openmm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    sim = ProlixSimulation(modeller.topology, system, integ)
    sim.context.setPositions(modeller.positions)

    # Initial velocities set manually to avoid the setVelocitiesToTemperature shape bug
    vel = np.random.randn(*np.array(modeller.positions.value_in_unit(unit.nanometer)).shape) * 0.1
    sim.context.setVelocities(vel * unit.nanometer / unit.picosecond)

    sim.step(100)

    pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    r_oh_target = 0.09572
    r_hh_target = 0.15139

    n_waters = len(pos) // 3
    max_oh_dev = 0.0
    max_hh_dev = 0.0

    for w in range(n_waters):
        o = pos[3 * w]
        h1 = pos[3 * w + 1]
        h2 = pos[3 * w + 2]
        d_oh1 = np.linalg.norm(h1 - o)
        d_oh2 = np.linalg.norm(h2 - o)
        d_hh = np.linalg.norm(h2 - h1)
        max_oh_dev = max(max_oh_dev, abs(d_oh1 - r_oh_target), abs(d_oh2 - r_oh_target))
        max_hh_dev = max(max_hh_dev, abs(d_hh - r_hh_target))

    assert max_oh_dev < 1e-4, f"r_OH drift too large: {max_oh_dev:.2e} nm (expected < 1e-4 nm)"
    assert max_hh_dev < 1e-4, f"r_HH drift too large: {max_hh_dev:.2e} nm (expected < 1e-4 nm)"


def test_water_multi_molecule_set_velocities_broadcasting_bug():
    """Adversarial challenge: Expose the broadcasting crash in applyVelocityConstraints for multi-water systems.

    In `ProlixContext.applyVelocityConstraints`, `masses` is passed with 1D shape (N,).
    Inside `SETTLEConstraint.apply_velocities`, `velocities = momenta / mass` expects `mass`
    to have shape (N, 1).
    When N > 3, JAX raises ValueError: Incompatible shapes for broadcasting: shapes=[(N, 3), (N,)].
    """
    modeller = app.Modeller(app.Topology(), [])
    ff = app.ForceField("amber14/tip3p.xml")
    modeller.addSolvent(ff, boxSize=openmm.Vec3(1.2, 1.2, 1.2) * unit.nanometers, model="tip3p")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=0.5 * unit.nanometers,
        rigidWater=True,
    )
    ctx = ProlixContext(system, openmm.VerletIntegrator(0.001))
    ctx.setPositions(modeller.positions)

    # Calling setVelocitiesToTemperature triggers applyVelocityConstraints.
    # Must succeed without raising ValueError: Incompatible shapes for broadcasting.
    try:
        ctx.setVelocitiesToTemperature(300.0 * unit.kelvin)
    except ValueError as e:
        pytest.fail(
            f"setVelocitiesToTemperature crashed with broadcasting error on multi-water system: {e}. "
            f"Bug in context.py: applyVelocityConstraints passes 1D masses instead of masses[:, None]!"
        )


# =============================================================================
# Task 5: NPT Simulations with MonteCarloBarostat in "mc" and "scr" Modes
# =============================================================================


@pytest.mark.parametrize("baro_mode", ["mc", "scr"])
def test_npt_simulation_dynamic_volume_fluctuations(baro_mode: str):
    """Verify NPT simulations with MonteCarloBarostat in 'mc' and 'scr' modes fluctuate dynamically."""
    sys = openmm.System()
    box_size = 2.5
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_size, 0, 0),
        openmm.Vec3(0, box_size, 0),
        openmm.Vec3(0, 0, box_size),
    )
    for _ in range(8):
        sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.8 * unit.nanometer)
    for _ in range(8):
        nb.addParticle(0.0, 0.3, 0.5)
    sys.addForce(nb)

    # Barostat move attempted every 5 steps
    baro = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 5)
    sys.addForce(baro)

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("RES", ch)
    for i in range(8):
        top.addAtom(f"A{i}", app.Element.getBySymbol("C"), res)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond
    )

    cfg = ProlixOpenMMConfig(barostat_mode=baro_mode)
    sim = ProlixSimulation(top, sys, integ, config=cfg)
    sim.context.setPositions([
        openmm.Vec3(0.5, 0.5, 0.5), openmm.Vec3(1.5, 0.5, 0.5),
        openmm.Vec3(0.5, 1.5, 0.5), openmm.Vec3(1.5, 1.5, 0.5),
        openmm.Vec3(0.5, 0.5, 1.5), openmm.Vec3(1.5, 0.5, 1.5),
        openmm.Vec3(0.5, 1.5, 1.5), openmm.Vec3(1.5, 1.5, 1.5),
    ] * unit.nanometer)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    volumes: list[float] = []
    energies: list[float] = []

    for _ in range(10):
        sim.step(10)
        s = sim.context.getState(getEnergy=True)
        bv = sim.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        vol = float(np.linalg.det(bv))
        pe = s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        volumes.append(vol)
        energies.append(pe)

    # Assert volume fluctuates dynamically
    assert min(volumes) != max(volumes), f"[{baro_mode}] Volume remained completely static across 100 steps!"
    # Assert box stays finite and strictly positive
    assert all(v > 0 and not np.isnan(v) and not np.isinf(v) for v in volumes), f"[{baro_mode}] Invalid box volumes: {volumes}"
    # Assert energy remains finite and bounded
    assert all(not np.isnan(e) and not np.isinf(e) for e in energies), f"[{baro_mode}] Energy exploded: {energies}"


# =============================================================================
# Task 6: Multi-Instance Custom Forces Dynamic Updates & Zero Recompilation
# =============================================================================


def test_update_parameters_multi_instance_custom_forces_zero_recompile():
    """Verify updateParametersInContext with multi-instance custom forces of same type triggers 0 recompilations."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    # Two separate CustomBondForce instances with distinct functional forms and parameters
    cf1 = openmm.CustomBondForce("k1 * (r - r1)^2")
    cf1.addPerBondParameter("k1")
    cf1.addPerBondParameter("r1")
    cf1.addBond(0, 1, [100.0, 0.15])
    sys.addForce(cf1)

    cf2 = openmm.CustomBondForce("k2 * (r - r2)^4")
    cf2.addPerBondParameter("k2")
    cf2.addPerBondParameter("r2")
    cf2.addBond(2, 3, [200.0, 0.20])
    sys.addForce(cf2)

    pos = [
        openmm.Vec3(0, 0, 0),
        openmm.Vec3(0.16, 0, 0),
        openmm.Vec3(0.5, 0, 0),
        openmm.Vec3(0.72, 0, 0),
    ] * unit.nanometer

    cfg = ProlixOpenMMConfig(custom_force_mode="lepton_only")
    pro_ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001), config=cfg)
    pro_ctx.setPositions(pos)

    ref_ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ref_ctx.setPositions(pos)

    def assert_parity_and_cache(expected_cache: int, label: str):
        s_ref = ref_ctx.getState(getEnergy=True)
        s_pro = pro_ctx.getState(getEnergy=True)
        e_ref = s_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_pro = s_pro.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        rel = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
        assert rel < 1e-5, f"[{label}] Energy parity failed: ref={e_ref}, pro={e_pro}, rel={rel}"
        cur_cache = _pure_compute_energy_and_forces._cached._cache_size()
        assert cur_cache == expected_cache, f"[{label}] Recompilation detected! expected={expected_cache}, got={cur_cache}"

    # Prime compilation and record initial baseline
    s_ref_init = ref_ctx.getState(getEnergy=True)
    s_pro_init = pro_ctx.getState(getEnergy=True)
    baseline_cache = _pure_compute_energy_and_forces._cached._cache_size()
    assert_parity_and_cache(baseline_cache, "initial")

    # Update CF1 only via force-targeted update
    cf1.setBondParameters(0, 0, 1, [300.0, 0.17])
    cf1.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext(cf1)
    assert_parity_and_cache(baseline_cache, "after_cf1_update")

    # Update CF2 only via force-targeted update
    cf2.setBondParameters(0, 2, 3, [500.0, 0.23])
    cf2.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext(cf2)
    assert_parity_and_cache(baseline_cache, "after_cf2_update")

    # Global update without argument
    cf1.setBondParameters(0, 0, 1, [450.0, 0.18])
    cf2.setBondParameters(0, 2, 3, [700.0, 0.25])
    cf1.updateParametersInContext(ref_ctx)
    cf2.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()
    assert_parity_and_cache(baseline_cache, "after_global_update")

    # Verify rollout cache persists across parameter updates
    pro_ctx.step(5)
    rollout_cache = _pure_step_rollout._cached._cache_size()
    cf1.setBondParameters(0, 0, 1, [500.0, 0.19])
    pro_ctx.updateParametersInContext(cf1)
    pro_ctx.step(5)
    rollout_cache_after = _pure_step_rollout._cached._cache_size()
    assert rollout_cache == rollout_cache_after, (
        f"Rollout recompiled on custom force parameter update! before={rollout_cache}, after={rollout_cache_after}"
    )
