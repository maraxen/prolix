"""Adversarial stress test suite for Milestone 4 Iteration 3 (Challenger M4-R3-2).

Empirical and adversarial verification of:
1. Multi-water systems of sizes N_waters in [1, 2, 5, 20, 100] (N=3, 6, 15, 60, 300 particles).
2. setVelocitiesToTemperature at 100 K, 300 K, 500 K across all system sizes never raises broadcasting or shape errors.
3. Constraint distance errors after applyConstraints() are < 10^-5 nm.
4. Constraint velocity errors after applyVelocityConstraints() are < 10^-6 nm/ps (internal water O-H and H-H
   relative velocities orthogonal to bond vectors).
5. NVT simulation stability with SETTLE over 50 steps on a 20-water box.
6. Deterministic seeding and constraint idempotency edge cases.
"""

from __future__ import annotations

import numpy as np
import openmm
from openmm import app, unit
import pytest

from prolix.openmm import ProlixContext, ProlixSimulation


def _build_water_system(n_waters: int) -> tuple[openmm.System, openmm.app.Topology, list[openmm.Vec3]]:
    """Helper to create an OpenMM System and Topology containing exactly `n_waters` TIP3P waters."""
    modeller = app.Modeller(app.Topology(), [])
    ff = app.ForceField("amber14/tip3p.xml")
    modeller.addSolvent(ff, model="tip3p", numAdded=n_waters)

    box_size = max(2.5, (n_waters / 33.0) ** (1.0 / 3.0) * 1.5 + 1.0)
    box_vec = openmm.Vec3(box_size, 0.0, 0.0) * unit.nanometers
    modeller.topology.setPeriodicBoxVectors((
        box_vec,
        openmm.Vec3(0.0, box_size, 0.0) * unit.nanometers,
        openmm.Vec3(0.0, 0.0, box_size) * unit.nanometers,
    ))

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=0.9 * unit.nanometers,
        rigidWater=True,
    )
    return system, modeller.topology, modeller.positions


# =============================================================================
# Task 1 & 2: Multi-Water Temperature Setting Across Sizes and Temperatures
# =============================================================================


@pytest.mark.parametrize("n_waters", [1, 2, 5, 20, 100])
@pytest.mark.parametrize("temperature_k", [100.0, 300.0, 500.0])
def test_multi_water_set_velocities_to_temperature_broadcasting(n_waters: int, temperature_k: float):
    """Verify setVelocitiesToTemperature across sizes N_waters in [1, 2, 5, 20, 100] and temperatures.

    Ensures zero broadcasting or shape mismatch errors and validates that COM motion is removed.
    """
    system, _, positions = _build_water_system(n_waters)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(system, integ)
    ctx.setPositions(positions)

    # Calling setVelocitiesToTemperature must never raise ValueError or shape error
    ctx.setVelocitiesToTemperature(temperature_k * unit.kelvin, randomSeed=1000 + n_waters)

    state = ctx.getState(getVelocities=True)
    vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    expected_particles = 3 * n_waters
    assert vel.shape == (expected_particles, 3), (
        f"Velocity shape mismatch: expected ({expected_particles}, 3), got {vel.shape}"
    )
    assert not np.isnan(vel).any(), "Velocities contain NaN values!"
    assert np.any(vel != 0.0), "Velocities are all zero!"

    # Center-of-mass velocity check: COM motion removal (< 5e-5 nm/ps)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(expected_particles)])
    v_com = np.sum(vel * masses[:, None], axis=0) / np.sum(masses)
    assert np.linalg.norm(v_com) < 5e-5, f"COM velocity not removed: ||v_com|| = {np.linalg.norm(v_com):.2e} nm/ps"


# =============================================================================
# Task 3: Constraint Distance Errors After applyConstraints() < 10^-5 nm
# =============================================================================


@pytest.mark.parametrize("n_waters", [1, 2, 5, 20, 100])
def test_multi_water_apply_constraints_distance_error(n_waters: int):
    """Verify constraint distance errors after applyConstraints() are < 10^-5 nm across all water molecules."""
    system, _, positions = _build_water_system(n_waters)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(system, integ)

    # Adversarial test: perturb initial positions with random noise (+- 0.005 nm = +- 0.05 A)
    rng = np.random.default_rng(2026 + n_waters)
    pos_array = np.array(positions.value_in_unit(unit.nanometer))
    pos_perturbed = pos_array + rng.normal(scale=0.005, size=pos_array.shape)
    ctx.setPositions(pos_perturbed * unit.nanometer)

    # Apply position constraints (SETTLE)
    ctx.applyConstraints(tolerance=1e-5)

    pos_out = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    # Standard TIP3P target distances
    r_oh_target = 0.09572
    r_hh_target = 0.15139

    max_oh_err = 0.0
    max_hh_err = 0.0

    for w in range(n_waters):
        o = pos_out[3 * w]
        h1 = pos_out[3 * w + 1]
        h2 = pos_out[3 * w + 2]

        d_oh1 = float(np.linalg.norm(h1 - o))
        d_oh2 = float(np.linalg.norm(h2 - o))
        d_hh = float(np.linalg.norm(h2 - h1))

        max_oh_err = max(max_oh_err, abs(d_oh1 - r_oh_target), abs(d_oh2 - r_oh_target))
        max_hh_err = max(max_hh_err, abs(d_hh - r_hh_target))

    assert max_oh_err < 1e-5, f"r_OH constraint distance error too large: {max_oh_err:.2e} nm (expected < 1e-5 nm)"
    assert max_hh_err < 1e-5, f"r_HH constraint distance error too large: {max_hh_err:.2e} nm (expected < 1e-5 nm)"


# =============================================================================
# Task 4: Constraint Velocity Errors After applyVelocityConstraints() < 10^-6 nm/ps
# =============================================================================


@pytest.mark.parametrize("n_waters", [1, 2, 5, 20, 100])
def test_multi_water_apply_velocity_constraints_orthogonality(n_waters: int):
    """Verify constraint velocity errors after applyVelocityConstraints() are < 10^-6 nm/ps.

    Relative velocities of water O-H and H-H must be orthogonal to their respective bond vectors:
    |(v_i - v_j) . (r_i - r_j) / ||r_i - r_j||| < 10^-6 nm/ps.
    """
    system, _, positions = _build_water_system(n_waters)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(system, integ)
    ctx.setPositions(positions)
    ctx.applyConstraints()

    # Assign velocities at 300 K
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=54321 + n_waters)

    # Explicitly call applyVelocityConstraints to enforce projection
    ctx.applyVelocityConstraints(tolerance=1e-6)

    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel = ctx.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    max_err_oh = 0.0
    max_err_hh = 0.0

    for w in range(n_waters):
        o = pos[3 * w]
        h1 = pos[3 * w + 1]
        h2 = pos[3 * w + 2]
        vo = vel[3 * w]
        vh1 = vel[3 * w + 1]
        vh2 = vel[3 * w + 2]

        u_oh1 = (h1 - o) / np.linalg.norm(h1 - o)
        u_oh2 = (h2 - o) / np.linalg.norm(h2 - o)
        u_hh = (h2 - h1) / np.linalg.norm(h2 - h1)

        err_oh1 = abs(float(np.dot(vh1 - vo, u_oh1)))
        err_oh2 = abs(float(np.dot(vh2 - vo, u_oh2)))
        err_hh = abs(float(np.dot(vh2 - vh1, u_hh)))

        max_err_oh = max(max_err_oh, err_oh1, err_oh2)
        max_err_hh = max(max_err_hh, err_hh)

    assert max_err_oh < 1e-6, (
        f"N={n_waters}: O-H constraint velocity error exceeds 1e-6 nm/ps: {max_err_oh:.2e} nm/ps"
    )
    assert max_err_hh < 1e-6, (
        f"N={n_waters}: H-H constraint velocity error exceeds 1e-6 nm/ps: {max_err_hh:.2e} nm/ps"
    )


# =============================================================================
# Task 5: NVT Simulation Stability with SETTLE Over 50 Steps on 20-Water Box
# =============================================================================


def test_nvt_simulation_stability_settle_50_steps_20_waters():
    """Verify NVT simulation stability with SETTLE over 50 steps (100 fs) on a 20-water box."""
    system, topology, positions = _build_water_system(20)
    dt_ps = 0.002
    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picosecond
    )
    sim = ProlixSimulation(topology, system, integ)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=777)

    # Rollout 50 steps
    sim.step(50)

    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    ke = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)

    assert not np.isnan(pos).any(), "Positions contain NaN after 50-step NVT rollout!"
    assert not np.isnan(vel).any(), "Velocities contain NaN after 50-step NVT rollout!"
    assert not np.isnan(pe), "Potential energy is NaN after 50-step NVT rollout!"
    assert not np.isnan(ke), "Kinetic energy is NaN after 50-step NVT rollout!"
    assert ke > 0.0, "Kinetic energy must be positive!"

    # Verify water geometry rigidity (< 1e-4 nm drift)
    r_oh_target = 0.09572
    r_hh_target = 0.15139

    max_oh_dev = 0.0
    max_hh_dev = 0.0
    for w in range(20):
        o = pos[3 * w]
        h1 = pos[3 * w + 1]
        h2 = pos[3 * w + 2]
        d_oh1 = float(np.linalg.norm(h1 - o))
        d_oh2 = float(np.linalg.norm(h2 - o))
        d_hh = float(np.linalg.norm(h2 - h1))
        max_oh_dev = max(max_oh_dev, abs(d_oh1 - r_oh_target), abs(d_oh2 - r_oh_target))
        max_hh_dev = max(max_hh_dev, abs(d_hh - r_hh_target))

    assert max_oh_dev < 1e-4, f"r_OH drift too large: {max_oh_dev:.2e} nm (expected < 1e-4 nm)"
    assert max_hh_dev < 1e-4, f"r_HH drift too large: {max_hh_dev:.2e} nm (expected < 1e-4 nm)"


# =============================================================================
# Task 6: Deterministic Seeding and Constraint Idempotency Edge Cases
# =============================================================================


def test_temperature_setting_reproducible_seeding():
    """Verify setVelocitiesToTemperature produces identical velocities with identical seeds and distinct with different seeds."""
    system, _, positions = _build_water_system(5)
    integ = openmm.VerletIntegrator(0.001)
    ctx1 = ProlixContext(system, integ)
    ctx1.setPositions(positions)
    ctx1.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=9999)
    v1 = ctx1.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    ctx2 = ProlixContext(system, integ)
    ctx2.setPositions(positions)
    ctx2.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=9999)
    v2 = ctx2.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    ctx3 = ProlixContext(system, integ)
    ctx3.setPositions(positions)
    ctx3.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=1111)
    v3 = ctx3.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)

    np.testing.assert_allclose(v1, v2, atol=1e-6, err_msg="Identical randomSeed did not produce identical velocities!")
    assert not np.allclose(v1, v3, atol=1e-3), "Different randomSeed produced identical velocities!"


def test_apply_constraints_idempotency():
    """Verify that calling applyConstraints() repeatedly on already constrained positions is idempotent (< 1e-7 nm change)."""
    system, _, positions = _build_water_system(5)
    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(system, integ)
    ctx.setPositions(positions)
    ctx.applyConstraints()

    pos1 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ctx.applyConstraints()
    pos2 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    max_delta = float(np.max(np.abs(pos2 - pos1)))
    assert max_delta < 1e-7, f"applyConstraints is not idempotent: max displacement = {max_delta:.2e} nm"
