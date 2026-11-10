"""Unit tests for dynamic zero-recompile parameter updates and alchemical derivatives in ProlixContext."""

from __future__ import annotations

import numpy as np
import openmm
from openmm import unit
import pytest

from prolix.openmm import ProlixContext
from prolix.openmm.context import _pure_compute_energy_and_forces, _pure_step_rollout


def test_update_nonbonded_particles_zero_recompile():
    """Verify particle charge, sigma, and epsilon updates update energy without JIT recompilation."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.addParticle(0.5, 0.3, 1.0)
    nb.addParticle(-0.5, 0.3, 1.0)
    sys.addForce(nb)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.5, 0.0, 0.0),
    ] * unit.nanometer)

    # Prime compilation
    s1 = ctx.getState(getEnergy=True, getForces=True)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_before = _pure_compute_energy_and_forces._cached._cache_size()

    # Dynamic update of particle parameters in OpenMM system
    nb.setParticleParameters(0, 1.0, 0.35, 2.0)
    ctx.updateParametersInContext()

    s2 = ctx.getState(getEnergy=True, getForces=True)
    e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_after = _pure_compute_energy_and_forces._cached._cache_size()

    assert e1 != e2
    assert cache_after == cache_before, (
        f"Recompilation detected! cache_before={cache_before}, cache_after={cache_after}"
    )


def test_update_nonbonded_exceptions_zero_recompile():
    """Verify 1-4 exception parameter updates modify energy with zero recompilations."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.addParticle(0.0, 0.3, 0.0)
    nb.addParticle(0.0, 0.3, 0.0)
    nb.addException(0, 1, 0.2, 0.25, 1.5)
    sys.addForce(nb)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.3, 0.0, 0.0),
    ] * unit.nanometer)

    # Prime compilation
    s1 = ctx.getState(getEnergy=True)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_before = _pure_compute_energy_and_forces._cached._cache_size()

    # Modify 1-4 exception parameters
    nb.setExceptionParameters(0, 0, 1, 0.5, 0.35, 3.0)
    ctx.updateParametersInContext()

    s2 = ctx.getState(getEnergy=True)
    e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_after = _pure_compute_energy_and_forces._cached._cache_size()

    assert e1 != e2
    assert cache_after == cache_before, (
        f"Recompilation detected on exception update! cache_before={cache_before}, cache_after={cache_after}"
    )


def test_update_harmonic_bonds_and_angles_zero_recompile():
    """Verify HarmonicBondForce and HarmonicAngleForce parameter updates with zero recompilations."""
    sys = openmm.System()
    for _ in range(3):
        sys.addParticle(12.0)

    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    ang = openmm.HarmonicAngleForce()
    ang.addAngle(0, 1, 2, 1.57 * unit.radian, 50000.0 * unit.kilojoules_per_mole / unit.radian**2)
    sys.addForce(ang)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.18, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.18, 0.0),
    ] * unit.nanometer)

    s1 = ctx.getState(getEnergy=True)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_before = _pure_compute_energy_and_forces._cached._cache_size()

    # Double force constants
    b.setBondParameters(0, 0, 1, 0.15 * unit.nanometer, 400000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    ang.setAngleParameters(0, 0, 1, 2, 1.57 * unit.radian, 100000.0 * unit.kilojoules_per_mole / unit.radian**2)
    ctx.updateParametersInContext()

    s2 = ctx.getState(getEnergy=True)
    e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_after = _pure_compute_energy_and_forces._cached._cache_size()

    assert e2 == pytest.approx(2.0 * e1, rel=1e-4)
    assert cache_after == cache_before


def test_update_periodic_torsions_zero_recompile():
    """Verify PeriodicTorsionForce dynamic update with zero recompilations."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    tor = openmm.PeriodicTorsionForce()
    tor.addTorsion(0, 1, 2, 3, 2, 0.0 * unit.radian, 10.0 * unit.kilojoules_per_mole)
    sys.addForce(tor)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.0, 0.1, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.15, 0.0, 0.0),
        openmm.Vec3(0.15, 0.1, 0.0),
    ] * unit.nanometer)

    s1 = ctx.getState(getEnergy=True)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_before = _pure_compute_energy_and_forces._cached._cache_size()

    # Triple torsion force constant
    tor.setTorsionParameters(0, 0, 1, 2, 3, 2, 0.0 * unit.radian, 30.0 * unit.kilojoules_per_mole)
    ctx.updateParametersInContext()

    s2 = ctx.getState(getEnergy=True)
    e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    cache_after = _pure_compute_energy_and_forces._cached._cache_size()

    assert e2 == pytest.approx(3.0 * e1, rel=1e-4)
    assert cache_after == cache_before


def test_update_custom_forces_and_derivatives_zero_recompile():
    """Verify dynamic custom force updates and analytical parameter derivatives without recompilation."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    cf = openmm.CustomBondForce("lambda_param * k * (r - r0)^2")
    cf.addGlobalParameter("lambda_param", 1.0)
    cf.addPerBondParameter("k")
    cf.addPerBondParameter("r0")
    cf.addEnergyParameterDerivative("lambda_param")
    cf.addBond(0, 1, [100.0, 0.15])
    sys.addForce(cf)

    integ = openmm.VerletIntegrator(0.001)
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.20, 0.0, 0.0),
    ] * unit.nanometer)

    s1 = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    e1 = s1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    d1 = s1.getEnergyParameterDerivatives()["lambda_param"]
    cache_before = _pure_compute_energy_and_forces._cached._cache_size()

    # Double per-bond force constant
    cf.setBondParameters(0, 0, 1, [200.0, 0.15])
    ctx.updateParametersInContext()

    s2 = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    e2 = s2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    d2 = s2.getEnergyParameterDerivatives()["lambda_param"]
    cache_after = _pure_compute_energy_and_forces._cached._cache_size()

    assert e2 == pytest.approx(2.0 * e1, rel=1e-4)
    assert d2 == pytest.approx(2.0 * d1, rel=1e-4)
    assert cache_after == cache_before

    # Modify global parameter
    ctx.setParameter("lambda_param", 0.5)
    s3 = ctx.getState(getEnergy=True, getParameterDerivatives=True)
    e3 = s3.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    d3 = s3.getEnergyParameterDerivatives()["lambda_param"]
    cache_third = _pure_compute_energy_and_forces._cached._cache_size()

    assert e3 == pytest.approx(e1, rel=1e-4)
    assert d3 == pytest.approx(2.0 * d1, rel=1e-4)  # dE/dlambda = k*(r-r0)^2 (remains 200*(0.05)^2)
    assert cache_third == cache_after


def test_step_rollout_zero_recompile_on_parameter_update():
    """Verify GPU rollout kernel cache does not grow across dynamic parameter updates."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15 * unit.nanometer, 200000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(b)

    integ = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picosecond,
    )
    ctx = ProlixContext(sys, integ)
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.16, 0.0, 0.0),
    ] * unit.nanometer)

    ctx.step(5)
    cache_before = _pure_step_rollout._cached._cache_size()

    # Dynamic update of bond parameter
    b.setBondParameters(0, 0, 1, 0.18 * unit.nanometer, 250000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    ctx.updateParametersInContext()

    ctx.step(5)
    cache_after = _pure_step_rollout._cached._cache_size()

    assert cache_before == cache_after, (
        f"Rollout kernel recompiled! cache_before={cache_before}, cache_after={cache_after}"
    )


def test_update_multiple_custom_forces_exact_index_mapping():
    """Verify multiple custom forces of the same type update parameters without key collision or recompile."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    cf1 = openmm.CustomBondForce("k1 * (r - r0)^3")
    cf1.addPerBondParameter("k1")
    cf1.addPerBondParameter("r0")
    cf1.addBond(0, 1, [100.0, 0.15])
    sys.addForce(cf1)

    cf2 = openmm.CustomBondForce("k2 * (r - r0)^4")
    cf2.addPerBondParameter("k2")
    cf2.addPerBondParameter("r0")
    cf2.addBond(2, 3, [200.0, 0.20])
    sys.addForce(cf2)

    ctx = ProlixContext(sys, openmm.VerletIntegrator(0.001))
    ctx.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.2, 0.0, 0.0),
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(1.3, 0.0, 0.0),
    ] * unit.nanometer)

    # Prime compilation
    s1 = ctx.getState(getEnergy=True)
    cache_before = _pure_compute_energy_and_forces._cached._cache_size()

    # Update ONLY cf2 with force parameter
    cf2.setBondParameters(0, 2, 3, [400.0, 0.20])
    ctx.updateParametersInContext(force=cf2)

    s2 = ctx.getState(getEnergy=True)
    cache_after = _pure_compute_energy_and_forces._cached._cache_size()
    assert s2.getPotentialEnergy() != s1.getPotentialEnergy()
    assert cache_after == cache_before

    # Update cf1 as well and verify updateParametersInContext() without args updates both
    cf1.setBondParameters(0, 0, 1, [300.0, 0.15])
    ctx.updateParametersInContext()
    s3 = ctx.getState(getEnergy=True)
    cache_third = _pure_compute_energy_and_forces._cached._cache_size()
    assert s3.getPotentialEnergy() != s2.getPotentialEnergy()
    assert cache_third == cache_before
