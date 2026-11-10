"""Adversarial stress test suite for Milestone 4 (Rollouts, Dynamic Updates, Minimization).

Author: Challenger 2
Target Modules:
- src/prolix/openmm/context.py (ProlixContext, updateParametersInContext, rollout kernels)
- src/prolix/openmm/simulation.py (ProlixSimulation, fused rollout bypass, minimizeEnergy)
"""

from __future__ import annotations

from typing import Any
import numpy as np
import openmm
from openmm import app, unit
import pytest

from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation
from prolix.openmm.context import _pure_compute_energy_and_forces, _pure_step_rollout


# =============================================================================
# 1. Dynamic Updates & Zero Recompilation with Numerical Parity
# =============================================================================


def test_dynamic_parameter_updates_zero_recompile_and_parity():
    """Verify sequential parameter mutations maintain _cache_size() == 1 and numerical parity (< 1e-5).

    Covers:
    - Particle charges, sigmas, epsilons
    - 1-4 exception charges, sigmas, epsilons
    - Harmonic bond length and force constant
    - Harmonic angle equilibrium angle and force constant
    - Periodic torsion force constant (at phase=0 baseline)
    - Custom force per-bond parameters
    - Step rollout kernel compilation cache size across parameter updates
    """
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    # 1. Harmonic Bonds
    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b.addBond(1, 2, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b.addBond(2, 3, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    # 2. Harmonic Angles
    ang = openmm.HarmonicAngleForce()
    ang.addAngle(0, 1, 2, 1.8 * unit.radian, 40000.0 * unit.kilojoules_per_mole / unit.radian**2)
    ang.addAngle(1, 2, 3, 1.8 * unit.radian, 40000.0 * unit.kilojoules_per_mole / unit.radian**2)
    sys.addForce(ang)

    # 3. Periodic Torsions (phase=0 where baseline parity holds)
    tor = openmm.PeriodicTorsionForce()
    tor.addTorsion(0, 1, 2, 3, 2, 0.0 * unit.radian, 25.0 * unit.kilojoules_per_mole)
    sys.addForce(tor)

    # 4. Nonbonded Force (particles and 1-4 exceptions)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb.addParticle(0.4, 0.30, 2.0)
    nb.addParticle(-0.2, 0.28, 1.5)
    nb.addParticle(-0.2, 0.28, 1.5)
    nb.addParticle(0.4, 0.30, 2.0)
    nb.addException(0, 1, 0.0, 0.1, 0.0)
    nb.addException(1, 2, 0.0, 0.1, 0.0)
    nb.addException(2, 3, 0.0, 0.1, 0.0)
    nb.addException(0, 2, 0.0, 0.1, 0.0)
    nb.addException(1, 3, 0.0, 0.1, 0.0)
    nb.addException(0, 3, 0.16, 0.32, 1.0)
    sys.addForce(nb)

    # 5. Custom Force (quartic bond)
    cf = openmm.CustomBondForce("k_q * (r - r0_q)^4")
    cf.addPerBondParameter("k_q")
    cf.addPerBondParameter("r0_q")
    cf.addBond(0, 3, [50000.0, 0.35])
    sys.addForce(cf)

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.14, 0.05, 0.0),
        openmm.Vec3(0.25, -0.05, 0.08),
        openmm.Vec3(0.38, 0.02, -0.02),
    ] * unit.nanometer

    # OpenMM Reference Platform Context
    ref_integ = openmm.VerletIntegrator(0.001)
    ref_platform = openmm.Platform.getPlatformByName("Reference")
    ref_ctx = openmm.Context(sys, ref_integ, ref_platform)
    ref_ctx.setPositions(pos)

    # Prolix Context
    pro_integ = openmm.VerletIntegrator(0.001)
    pro_ctx = ProlixContext(sys, pro_integ)
    pro_ctx.setPositions(pos)

    def verify_state_parity(step_label: str) -> None:
        s_ref = ref_ctx.getState(getEnergy=True, getForces=True)
        s_pro = pro_ctx.getState(getEnergy=True, getForces=True)

        e_ref = s_ref.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_pro = s_pro.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        rel_e = abs(e_ref - e_pro) / (abs(e_ref) + 1e-12)
        assert rel_e < 1e-5, f"[{step_label}] Energy parity failed: ref={e_ref}, pro={e_pro}, rel_diff={rel_e}"

        f_ref = s_ref.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        f_pro = s_pro.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        max_f_denom = np.max(np.abs(f_ref)) + 1e-12
        rel_f = np.max(np.abs(f_ref - f_pro)) / max_f_denom
        assert rel_f < 1e-4, f"[{step_label}] Force parity failed: max_diff={np.max(np.abs(f_ref - f_pro))}, rel_f={rel_f}"

    # Initial check & compile priming
    verify_state_parity("initial")
    cache_start = _pure_compute_energy_and_forces._cached._cache_size()

    # Mutation 1: Nonbonded particles
    nb.setParticleParameters(0, 0.8, 0.35, 3.0)
    nb.setParticleParameters(3, 0.8, 0.35, 3.0)
    nb.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()
    assert _pure_compute_energy_and_forces._cached._cache_size() == cache_start, "Recompile on particle update!"
    verify_state_parity("after_particle_update")

    # Mutation 2: 1-4 Exceptions
    nb.setExceptionParameters(5, 0, 3, 0.32, 0.38, 2.5)
    nb.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()
    assert _pure_compute_energy_and_forces._cached._cache_size() == cache_start, "Recompile on exception update!"
    verify_state_parity("after_exception_update")

    # Mutation 3: Harmonic Bonds
    b.setBondParameters(0, 0, 1, 0.16 * unit.nanometer, 250000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()
    assert _pure_compute_energy_and_forces._cached._cache_size() == cache_start, "Recompile on bond update!"
    verify_state_parity("after_bond_update")

    # Mutation 4: Harmonic Angles
    ang.setAngleParameters(0, 0, 1, 2, 2.0 * unit.radian, 60000.0 * unit.kilojoules_per_mole / unit.radian**2)
    ang.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()
    assert _pure_compute_energy_and_forces._cached._cache_size() == cache_start, "Recompile on angle update!"
    verify_state_parity("after_angle_update")

    # Mutation 5: Custom Force (quartic bond)
    cf.setBondParameters(0, 0, 3, [80000.0, 0.38])
    cf.updateParametersInContext(ref_ctx)
    pro_ctx.updateParametersInContext()
    assert _pure_compute_energy_and_forces._cached._cache_size() == cache_start, "Recompile on custom force update!"
    verify_state_parity("after_custom_force_update")

    # Rollout Cache Stress
    pro_ctx.step(5)
    rollout_cache_1 = _pure_step_rollout._cached._cache_size()

    # Dynamic mutation during rollout
    b.setBondParameters(1, 1, 2, 0.17 * unit.nanometer, 180000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    pro_ctx.updateParametersInContext()
    pro_ctx.step(5)
    rollout_cache_2 = _pure_step_rollout._cached._cache_size()

    assert rollout_cache_1 == rollout_cache_2, (
        f"Rollout recompiled on parameter update! before={rollout_cache_1}, after={rollout_cache_2}"
    )


# =============================================================================
# 2. Adversarial Parity Challenge: Periodic Torsion Nonzero Phase Sign
# =============================================================================


def test_periodic_torsion_nonzero_phase_parity_adversarial():
    """Adversarial stress: Verify numerical parity of PeriodicTorsionForce when phase != 0.

    Exposes the dihedral sign inversion bug in `src/prolix/openmm/context.py` (_eval_bonded_terms).
    When phase is 0.0, cos(per * phi) is symmetric to sign inversion, masking the bug.
    When phase is non-zero, cos(per * (-phi) - phase) != cos(per * phi - phase).
    """
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    # Periodic torsion with phase = 0.5 radian
    tor = openmm.PeriodicTorsionForce()
    tor.addTorsion(0, 1, 2, 3, 2, 0.5 * unit.radian, 25.0 * unit.kilojoules_per_mole)
    sys.addForce(tor)

    pos = [
        openmm.Vec3(0.0, 0.1, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.15, 0.0, 0.0),
        openmm.Vec3(0.15, 0.1, 0.05),
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
    max_f_diff = np.max(np.abs(f_ref - f_pro))

    # This assertion mandates the < 1e-5 relative parity requirement from dispatch
    assert rel_e < 1e-5, (
        f"Dihedral parity failure: e_ref={e_ref:.4f}, e_pro={e_pro:.4f}, "
        f"rel_diff={rel_e:.4f} (expected < 1e-5). "
        f"Bug in context.py: m = cross(c1, b2_u) inverts phi sign!"
    )
    assert max_f_diff < 1e-2, f"Dihedral force parity failure: max_f_diff={max_f_diff}"


# =============================================================================
# 3. Fused Rollout Bypass Verification
# =============================================================================


def test_fused_rollout_bypass_prime_intervals_no_slicing():
    """Verify _simulate dispatches exact step chunks directly without 10-step slicing across prime intervals."""
    sys = openmm.System()
    sys.addParticle(12.0)
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("RES", chain)
    top.addAtom("A", app.Element.getBySymbol("C"), res)

    integ = openmm.VerletIntegrator(0.001)
    sim = ProlixSimulation(top, sys, integ)
    sim.context.setPositions([openmm.Vec3(0, 0, 0)] * unit.nanometer)

    step_calls: list[int] = []
    original_step = sim.context.step

    def recording_step(n: int) -> None:
        step_calls.append(n)
        original_step(n)

    sim.context.step = recording_step

    class PrimeReporter:
        def __init__(self, interval: int) -> None:
            self.interval = interval
            self.report_steps: list[int] = []

        def describeNextReport(self, sim_obj: Any) -> tuple[int, bool, bool, bool, bool, bool]:
            steps = self.interval - (sim_obj.currentStep % self.interval)
            return (steps, False, False, False, False, False)

        def report(self, sim_obj: Any, state: Any) -> None:  # noqa: ARG002
            self.report_steps.append(sim_obj.currentStep)

    r17 = PrimeReporter(17)
    r31 = PrimeReporter(31)
    r50 = PrimeReporter(50)
    sim.reporters.extend([r17, r31, r50])

    # Run for 150 steps
    sim.step(150)

    # In OpenMM reference Simulation._simulate, 150 steps with 10-step slicing produces dozens of step(10) calls.
    # In Prolix Fused Rollout Bypass, chunks are exact intervals between required reports.
    expected_reports_17 = [17, 34, 51, 68, 85, 102, 119, 136]
    expected_reports_31 = [31, 62, 93, 124]
    expected_reports_50 = [50, 100, 150]

    assert r17.report_steps == expected_reports_17
    assert r31.report_steps == expected_reports_31
    assert r50.report_steps == expected_reports_50

    # Ensure total steps executed matches exactly 150
    assert sum(step_calls) == 150
    assert sim.currentStep == 150

    # Check that chunks were not sliced into arbitrary 10-step pieces:
    # Any step call equals the gap between distinct report event points
    all_events = sorted(set(expected_reports_17 + expected_reports_31 + expected_reports_50 + [150]))
    expected_chunks = [all_events[0]] + [all_events[i] - all_events[i - 1] for i in range(1, len(all_events))]
    assert step_calls == expected_chunks, f"Step chunks {step_calls} != expected {expected_chunks}"


# =============================================================================
# 4. NVE Conservation, NVT Equipartition, and NPT Stability
# =============================================================================


def test_nve_energy_conservation_1000_steps():
    """Verify NVE simulation with VerletIntegrator conserves total energy (Delta E / E < 1e-4 over 1000 steps)."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 20000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("RES", ch)
    top.addAtom("A1", app.Element.getBySymbol("C"), res)
    top.addAtom("A2", app.Element.getBySymbol("C"), res)

    integ = openmm.VerletIntegrator(0.0005 * unit.picosecond)  # 0.5 fs timestep
    sim = ProlixSimulation(top, sys, integ)
    sim.context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(0.155, 0, 0)] * unit.nanometer)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)

    s0 = sim.context.getState(getEnergy=True)
    e0 = s0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) + s0.getKineticEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    energies: list[float] = []
    for _ in range(10):
        sim.step(100)
        s = sim.context.getState(getEnergy=True)
        e = s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) + s.getKineticEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        energies.append(e)

    max_drift = max(abs(e - e0) for e in energies)
    rel_drift = max_drift / abs(e0)
    assert rel_drift < 1e-4, f"NVE energy drift too large: rel_drift={rel_drift:.2e} (expected < 1e-4)"


def test_nvt_temperature_equipartition():
    """Verify NVT Langevin thermostat reaches target temperature equipartition."""
    n = 20
    sys = openmm.System()
    for _ in range(n):
        sys.addParticle(12.0)

    b = openmm.HarmonicBondForce()
    for i in range(n - 1):
        b.addBond(i, i + 1, 0.15 * unit.nanometer, 20000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("RES", ch)
    for i in range(n):
        top.addAtom(f"A{i}", app.Element.getBySymbol("C"), res)

    target_t = 300.0
    integ = openmm.LangevinMiddleIntegrator(
        target_t * unit.kelvin,
        10.0 / unit.picosecond,
        0.001 * unit.picosecond,
    )
    sim = ProlixSimulation(top, sys, integ)
    positions = [openmm.Vec3(float(i) * 0.15, 0.0, 0.0) for i in range(n)] * unit.nanometer
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(target_t * unit.kelvin, randomSeed=101)

    # Equilibrate 500 steps
    sim.step(500)

    # Sample instantaneous temperatures
    k_b_kj = 0.008314462618  # kJ/(mol*K)
    ndof = 3 * n - 3  # COM motion removed
    temps: list[float] = []

    for _ in range(25):
        sim.step(40)
        s = sim.context.getState(getEnergy=True)
        ke = s.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        t_inst = (2.0 * ke) / (ndof * k_b_kj)
        temps.append(t_inst)

    mean_t = float(np.mean(temps))
    sem_t = float(np.std(temps) / np.sqrt(len(temps)))
    # Equipartition check: mean temperature within 3 standard errors of target
    assert abs(mean_t - target_t) < 3.0 * sem_t + 15.0, (
        f"NVT temperature equipartition failed: target={target_t}K, mean={mean_t:.1f}K, sem={sem_t:.1f}K"
    )


@pytest.mark.parametrize("baro_mode", ["mc", "scr"])
def test_npt_simulation_stability_and_pbc_bug_adversarial(baro_mode: str):
    """Adversarial stress: Verify NPT simulation with MonteCarloBarostat in 'mc' and 'scr' modes completes stably.

    Exposes the critical PBC bug in `src/prolix/openmm/context.py` line 245:
    `_minimum_image_dr` calls `minimum_image_displacement(dr, box)` (2 args)
    when `minimum_image_displacement(r1, r2, box)` requires 3 arguments.
    """
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2.5, 0, 0),
        openmm.Vec3(0, 2.5, 0),
        openmm.Vec3(0, 0, 2.5),
    )
    for _ in range(8):
        sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.8 * unit.nanometer)
    for _ in range(8):
        nb.addParticle(0.0, 0.3, 0.5)
    sys.addForce(nb)

    baro = openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25)
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

    # Step simulation: Must complete stably without crashing
    sim.step(50)

    state = sim.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert not np.isnan(pe) and not np.isinf(pe), f"NPT [{baro_mode}] produced invalid energy: {pe}"


# =============================================================================
# 5. Minimizer Convergence from Clashed Initial Coordinates
# =============================================================================


@pytest.mark.parametrize("backend", ["lbfgs", "fire", "openmm_cpu"])
def test_minimizer_convergence_from_clashed_coordinates(backend: str):
    """Stress test minimizer convergence from severely clashed coordinates (0.5 Angstrom distance)."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b.addBond(1, 2, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    b.addBond(2, 3, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("RES", ch)
    for i in range(4):
        top.addAtom(f"A{i}", app.Element.getBySymbol("C"), res)

    integ = openmm.VerletIntegrator(0.001)
    cfg = ProlixOpenMMConfig(minimizer_backend=backend)
    sim = ProlixSimulation(top, sys, integ, config=cfg)

    # Severely clashed initial coordinates: atoms 0 and 3 placed 0.05 nm (0.5 Å) apart
    sim.context.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.15, 0.0, 0.0),
        openmm.Vec3(0.15, 0.05, 0.0),
        openmm.Vec3(0.05, 0.0, 0.0),
    ] * unit.nanometer)

    s0 = sim.context.getState(getEnergy=True)
    e0 = s0.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert e0 > 1000.0, f"Expected high initial clash energy, got {e0}"

    tol = 10.0 * unit.kilojoules_per_mole / unit.nanometer
    sim.minimizeEnergy(tolerance=tol, maxIterations=500)

    s1 = sim.context.getState(getEnergy=True, getForces=True)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f1 = s1.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

    # Verify dramatic energy reduction and convergence
    assert e1 < 1.0, f"[{backend}] Minimization failed to reduce energy: e0={e0}, e1={e1}"
    max_f = np.max(np.abs(f1))
    assert max_f < 20.0, f"[{backend}] Force tolerance not satisfied: max_f={max_f}"
