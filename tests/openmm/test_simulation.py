"""Unit tests for ProlixSimulation drop-in replacement and fused rollout bypass."""

from __future__ import annotations

import io
from typing import Any
import numpy as np
import openmm
from openmm import app, unit
import pytest

from prolix.openmm import ProlixOpenMMConfig, ProlixSimulation


def _build_test_simulation(
    num_particles: int = 2,
    minimizer_backend: str = "lbfgs",
) -> ProlixSimulation:
    """Construct a minimal valid simulation with harmonic bond force."""
    sys = openmm.System()
    for _ in range(num_particles):
        sys.addParticle(12.0)

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("RES", chain)
    for i in range(num_particles):
        top.addAtom(f"A{i}", app.Element.getBySymbol("C"), res)

    if num_particles >= 2:
        b = openmm.HarmonicBondForce()
        b.addBond(0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
        sys.addForce(b)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picosecond,
    )
    cfg = ProlixOpenMMConfig(minimizer_backend=minimizer_backend)
    sim = ProlixSimulation(top, sys, integ, config=cfg)

    positions = [
        openmm.Vec3(float(i) * 0.15, 0.0, 0.0) for i in range(num_particles)
    ] * unit.nanometer
    sim.context.setPositions(positions)
    return sim


def test_simulation_initialization_and_properties():
    """Verify ProlixSimulation initialization, references, and currentStep property."""
    sim = _build_test_simulation()

    assert isinstance(sim, app.Simulation)
    assert sim.topology is not None
    assert sim.system is not None
    assert sim.integrator is not None
    assert sim.context is not None

    # Step count getter & setter
    assert sim.currentStep == 0
    sim.currentStep = 250
    assert sim.currentStep == 250
    assert sim.context.getStepCount() == 250


def test_simulation_fused_rollout_bypass_single_dispatch():
    """Verify _simulate dispatches full reporter intervals directly without 10-step slicing."""
    sim = _build_test_simulation()

    step_calls: list[int] = []
    original_step = sim.context.step

    def recording_step(steps: int) -> None:
        step_calls.append(steps)
        original_step(steps)

    sim.context.step = recording_step

    class IntervalReporter:
        def __init__(self, interval: int) -> None:
            self.interval = interval
            self.reports = 0

        def describeNextReport(self, simulation: Any) -> tuple[int, bool, bool, bool, bool, bool]:
            # Returns: (steps, getPositions, getVelocities, getForces, getEnergy, enforcePeriodic)
            steps = self.interval - (simulation.currentStep % self.interval)
            return (steps, True, False, False, False, False)

        def report(self, simulation: Any, state: Any) -> None:
            self.reports += 1

    rep = IntervalReporter(50)
    sim.reporters.append(rep)

    # Step 150 steps with report interval 50
    sim.step(150)

    # In standard OpenMM, 150 steps would be sliced into 10-step pieces: [10, 10, 10, ...] (15 calls)
    # Prolix Fused Rollout must dispatch full 50-step intervals: [50, 50, 50]
    assert step_calls == [50, 50, 50]
    assert rep.reports == 3
    assert sim.currentStep == 150


def test_simulation_step_advances_trajectory():
    """Verify sim.step() advances simulation time and updates coordinates."""
    sim = _build_test_simulation()
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)

    initial_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    sim.step(20)

    final_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    t = sim.context.getTime().value_in_unit(unit.picosecond)

    assert sim.currentStep == 20
    assert t == pytest.approx(0.02, rel=1e-4)
    # Positions should have evolved under Langevin dynamics
    assert not np.allclose(initial_pos, final_pos)


def test_simulation_statedata_reporter_integration():
    """Verify standard StateDataReporter produces correct headers and periodic records."""
    sim = _build_test_simulation()
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=123)

    out = io.StringIO()
    reporter = app.StateDataReporter(
        out,
        reportInterval=10,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
    )
    sim.reporters.append(reporter)

    sim.step(30)

    lines = [line.strip() for line in out.getvalue().strip().split("\n") if line.strip()]
    assert len(lines) == 4  # Header + 3 reports (steps 10, 20, 30)

    headers = lines[0].split(",")
    assert '#"Step"' in headers
    assert '"Time (ps)"' in headers
    assert '"Potential Energy (kJ/mole)"' in headers
    assert '"Temperature (K)"' in headers

    row30 = [float(val) for val in lines[3].split(",")]
    assert row30[0] == 30  # Step 30
    assert row30[1] == pytest.approx(0.03, rel=1e-3)  # Time 0.03 ps
    assert row30[5] > 0.0  # Temperature > 0 K


@pytest.mark.parametrize("backend", ["lbfgs", "fire", "openmm_cpu"])
def test_simulation_minimize_energy_backends(backend: str):
    """Verify energy minimization backends reduce potential energy to equilibrium."""
    sim = _build_test_simulation(minimizer_backend=backend)

    # Displace bond from 0.15 nm to 0.25 nm (high potential energy)
    sim.context.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.25, 0.0, 0.0),
    ] * unit.nanometer)

    e_init = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert e_init > 500.0

    sim.minimizeEnergy(maxIterations=100)

    e_min = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    # Energy should drop close to 0 kJ/mol (at equilibrium r0 = 0.15 nm)
    assert e_min < 1.0

    pos_min = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    dist = np.linalg.norm(pos_min[1] - pos_min[0])
    assert dist == pytest.approx(0.15, abs=1e-3)


class MockMinimizationReporter(openmm.MinimizationReporter):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def report(self, iteration, x, grad, args):
        self.call_count += 1
        return False


def test_minimizer_openmm_cpu_forwards_reporter():
    """Verify openmm_cpu minimization forwards user-provided MinimizationReporter."""
    sys = openmm.System()
    sys.addParticle(1.0)
    sys.addParticle(1.0)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 0.15, 1000.0)
    sys.addForce(force)

    config = ProlixOpenMMConfig(minimizer_backend="openmm_cpu")
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("RES", chain)
    top.addAtom("A0", app.Element.getBySymbol("H"), res)
    top.addAtom("A1", app.Element.getBySymbol("H"), res)

    sim = ProlixSimulation(
        top, sys, openmm.VerletIntegrator(0.001), config=config
    )
    sim.context.setPositions(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]] * unit.nanometer
    )

    reporter = MockMinimizationReporter()
    sim.minimizeEnergy(
        tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=10,
        reporter=reporter,
    )
    assert reporter.call_count > 0, (
        "MinimizationReporter was not invoked during openmm_cpu minimization!"
    )
