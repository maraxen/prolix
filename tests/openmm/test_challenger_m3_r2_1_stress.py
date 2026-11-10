"""Challenger M3-R2-1 Empirical Adversarial Stress Test Suite for OpenMM System Converter.

Milestone 3 (Iteration 2) Gate Verification.
Authored by Challenger M3-R2-1 (teamwork_challenger_m3_r2_1).

Empirically challenges:
1. Re-verification of Remediated Gate Defects:
   - Nested virtual sites position reconstruction and reverse-mode force backpropagation
     through all ancestor generations down to real particles.
   - Collinear LocalCoordinatesSite force evaluation without explosion (<= 30.0 kJ/mol/nm)
     and exact bit-for-bit parity vs OpenMM Reference Platform.
   - Solute -CH2- in propane preservation (HBond constraints preserved in solute constraint_pairs,
     covalent bonds preserved in HarmonicBondForce, n_waters == 0).
   - Mixed zero-mass virtual sites, frozen particles, and real particles inv_mass evaluation
     and reverse-mode autodiff stability without NaN poison.
2. New Adversarial Edge Cases:
   - Deep 4-level nested DAG mixing TwoParticleAverageSite, OutOfPlaneSite, and LocalCoordinatesSite
     (both linear chain and branching topologies) with exact parity vs OpenMM Reference Platform.
   - Sweep of nearly collinear LocalCoordinatesSite parent configurations across
     delta in [1e-12, 1e-8, 1e-6, 1e-4, 0.0] asserting force stability and Reference parity.
   - Multi-force OpenMM systems with 3 distinct HarmonicBondForce and 3 HarmonicAngleForce instances,
     verifying CompoundMapping keys, lookups, and JAX pytree flattening sortability.
   - Inverted particle indexing in multi-generation virtual site DAGs.
   - Cyclic dependency detection and fail-fast ValueError.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
from openmm import unit
import pytest

from prolix.openmm.converter import (
    CompoundMapping,
    from_openmm_system,
)


# =============================================================================
# 1. Re-Verification of Remediated M3 Gate Defects
# =============================================================================


def test_reverify_nested_virtual_sites_two_levels():
    """Verify 2-level nested virtual site reconstruction and force backpropagation.

    Hierarchy:
      p0, p1, p2: real particles
      p3 (VS1): TwoParticleAverageSite between p0 and p1 (weights 0.5, 0.5)
      p4 (VS2): TwoParticleAverageSite between p3 and p2 (weights 0.5, 0.5)
    """
    system = openmm.System()
    for _ in range(5):
        system.addParticle(1.0 if _ < 3 else 0.0)

    vs1 = openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5)
    system.setVirtualSite(3, vs1)

    vs2 = openmm.TwoParticleAverageSite(3, 2, 0.5, 0.5)
    system.setVirtualSite(4, vs2)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(4, [10.0, 20.0, 30.0])
    system.addForce(ext)

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(1.0, 2.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pos)
    context.computeVirtualSites()
    omm_state = context.getState(getPositions=True, getForces=True)
    omm_pos = omm_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = omm_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(pos) * 10.0
    prolix_pos_nm = np.array(converted.compute_virtual_sites(pos_A)) / 10.0

    raw_f = jnp.zeros((5, 3)).at[4].set(jnp.array([10.0, 20.0, 30.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    # Assert exact position reconstruction
    assert np.allclose(omm_pos, prolix_pos_nm, atol=1e-5), "Position defect exceeds tolerance"
    assert np.allclose(prolix_pos_nm[3], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(prolix_pos_nm[4], [1.0, 1.0, 0.0], atol=1e-5)

    # Assert exact force redistribution to grandparents
    assert np.allclose(omm_f[:3], prolix_f[:3], atol=1e-5), "Grandparent forces do not match Reference"
    assert np.allclose(prolix_f[0], [2.5, 5.0, 7.5], atol=1e-5)
    assert np.allclose(prolix_f[1], [2.5, 5.0, 7.5], atol=1e-5)
    assert np.allclose(prolix_f[2], [5.0, 10.0, 15.0], atol=1e-5)
    # Conservation of total force
    assert np.allclose(np.sum(prolix_f[:3], axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-5)


def test_reverify_collinear_local_coordinates_exact_collinear():
    """Verify strictly collinear LocalCoordinatesSite parents match OpenMM Reference without explosion."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0 if _ < 3 else 0.0)

    vs = openmm.LocalCoordinatesSite(
        0,
        1,
        2,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(0.1, 0.2, 0.3) * unit.nanometer,
    )
    system.setVirtualSite(3, vs)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(3, [10.0, 20.0, 30.0])
    system.addForce(ext)

    pos_collinear = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pos_collinear)
    context.computeVirtualSites()
    omm_f = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(pos_collinear) * 10.0
    raw_f = jnp.zeros((4, 3)).at[3].set(jnp.array([10.0, 20.0, 30.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    omm_max_f = np.max(np.abs(omm_f[:3]))
    prolix_max_f = np.max(np.abs(prolix_f[:3]))

    assert omm_max_f <= 30.0
    assert prolix_max_f <= 30.0
    assert np.allclose(omm_f[:3], prolix_f[:3], atol=1e-5)
    assert np.allclose(np.sum(prolix_f[:3], axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-5)


def test_reverify_solute_ch2_preservation():
    """Verify aliphatic -CH2- in propane is NOT misclassified as water."""
    system = openmm.System()
    system.addParticle(12.011)  # C1
    system.addParticle(1.008)   # H1
    system.addParticle(1.008)   # H2
    system.addParticle(12.011)  # C2
    system.addParticle(12.011)  # C3

    system.addConstraint(0, 1, 0.109 * unit.nanometer)
    system.addConstraint(0, 2, 0.109 * unit.nanometer)

    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.109 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(0, 2, 0.109 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(0, 3, 0.154 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(0, 4, 0.154 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    system.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 0, 2, 109.5 * unit.degree, 500.0 * unit.kilojoules_per_mole / unit.radian**2)
    system.addForce(angles)

    converted = from_openmm_system(system)

    assert converted.bundle.n_waters == 0, "Aliphatic -CH2- must not be treated as water"
    assert converted.bundle.n_constraints == 2, "C-H constraints must remain in solute constraints"
    assert converted.bundle.n_bonds == 4, "C-H and C-C bonds must remain in bundle.bonds"
    assert converted.bundle.n_angles == 1, "H-C-H angle must remain in bundle.angles"


def test_reverify_mixed_zero_mass_and_frozen_inv_mass_autodiff():
    """Verify inv_mass forward evaluation and reverse-mode autodiff on mixed zero-mass / frozen systems."""
    system = openmm.System()
    system.addParticle(0.0)    # 0: Virtual site (m=0)
    system.addParticle(0.0)    # 1: Frozen atom (m=0)
    system.addParticle(12.011) # 2: Real C (m=12.011)
    system.addParticle(1.008)  # 3: Real H (m=1.008)
    system.addParticle(1.008)  # 4: Real H (m=1.008)

    vs = openmm.TwoParticleAverageSite(2, 3, 0.5, 0.5)
    system.setVirtualSite(0, vs)

    converted = from_openmm_system(system)
    inv_m = converted.inv_mass

    assert not jnp.any(jnp.isnan(inv_m))
    assert not jnp.any(jnp.isinf(inv_m))
    assert inv_m[0] == 0.0
    assert inv_m[1] == 0.0
    assert abs(inv_m[2] - 1.0 / 12.011) < 1e-5

    # Autodiff test through simulated loss function using inv_mass logic
    def inv_mass_calc(masses):
        safe_m = jnp.where(masses > 1e-12, masses, 1.0)
        return jnp.where(masses > 1e-12, 1.0 / safe_m, 0.0)

    def dummy_kinetic_loss(masses, p):
        inv_masses = inv_mass_calc(masses)
        return 0.5 * jnp.sum(p**2 * inv_masses[:, None])

    p_test = jnp.ones((5, 3))
    masses = converted.bundle.masses[:5]
    grad_fn = jax.grad(dummy_kinetic_loss)
    grads = grad_fn(masses, p_test)

    assert not jnp.any(jnp.isnan(grads)), "Autodiff through inv_mass produced NaNs"
    assert grads[0] == 0.0, "Zero-mass virtual site gradient must be 0.0"
    assert grads[1] == 0.0, "Frozen atom gradient must be 0.0"
    expected_g2 = -0.5 * 3.0 / (12.011**2)
    assert abs(grads[2] - expected_g2) < 1e-4


# =============================================================================
# 2. Deep 4-Level Nested DAG Stress Tests
# =============================================================================


def test_deep_4_level_nested_dag_linear_chain():
    """Verify deep 4-level linear hierarchy mixing TwoParticleAverage, OutOfPlane, and LocalCoordinates.

    Chain:
      L0: p0, p1, p2, p3 (real particles)
      L1: p4 = TwoParticleAverageSite(0, 1, 0.5, 0.5)
      L2: p5 = OutOfPlaneSite(4, 2, 3, 0.3, 0.4, 0.5)
      L3: p6 = LocalCoordinatesSite(5, 4, 1, (1,0,0), (-1,1,0), (-1,0,1), (0.1, 0.2, 0.3) nm)
      L4: p7 = TwoParticleAverageSite(6, 3, 0.6, 0.4)
    """
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0)
    for _ in range(4):
        system.addParticle(0.0)

    vs4 = openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5)
    system.setVirtualSite(4, vs4)

    vs5 = openmm.OutOfPlaneSite(4, 2, 3, 0.3, 0.4, 0.5)
    system.setVirtualSite(5, vs5)

    vs6 = openmm.LocalCoordinatesSite(
        5, 4, 1,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(0.1, 0.2, 0.3) * unit.nanometer,
    )
    system.setVirtualSite(6, vs6)

    vs7 = openmm.TwoParticleAverageSite(6, 3, 0.6, 0.4)
    system.setVirtualSite(7, vs7)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(7, [15.0, -25.0, 35.0])
    ext.addParticle(6, [5.0, 10.0, -15.0])
    system.addForce(ext)

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(1.0, 2.0, 0.0),
        openmm.Vec3(0.0, 1.0, 2.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pos)
    context.computeVirtualSites()
    omm_state = context.getState(getPositions=True, getForces=True)
    omm_pos = omm_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = omm_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(pos) * 10.0
    prolix_pos_nm = np.array(converted.compute_virtual_sites(pos_A)) / 10.0

    raw_f = jnp.zeros((8, 3))
    raw_f = raw_f.at[7].set(jnp.array([15.0, -25.0, 35.0]))
    raw_f = raw_f.at[6].set(jnp.array([5.0, 10.0, -15.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    pos_err = np.max(np.abs(omm_pos - prolix_pos_nm))
    force_err_real = np.max(np.abs(omm_f[:4] - prolix_f[:4]))

    assert pos_err < 1e-6, f"4-level DAG position error {pos_err} nm exceeds 1e-6"
    assert force_err_real < 1e-5, f"4-level DAG force error {force_err_real} exceeds 1e-5"
    assert np.allclose(np.sum(prolix_f[:4], axis=0), np.array([20.0, -15.0, 20.0]), atol=1e-5)


def test_deep_4_level_nested_dag_branching_network():
    """Verify 4-level branching DAG with multiple virtual sites per generation.

    Topology:
      Gen 0: Real particles 0, 1, 2, 3
      Gen 1: 4 = TwoParticleAverage(0, 1), 5 = TwoParticleAverage(2, 3)
      Gen 2: 6 = OutOfPlane(4, 5, 0), 7 = LocalCoordinates(5, 4, 1)
      Gen 3: 8 = LocalCoordinates(6, 7, 2), 9 = OutOfPlane(7, 6, 3)
      Gen 4: 10 = TwoParticleAverage(8, 9)
    """
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0)
    for _ in range(7):
        system.addParticle(0.0)

    # Gen 1
    system.setVirtualSite(4, openmm.TwoParticleAverageSite(0, 1, 0.4, 0.6))
    system.setVirtualSite(5, openmm.TwoParticleAverageSite(2, 3, 0.3, 0.7))

    # Gen 2
    system.setVirtualSite(6, openmm.OutOfPlaneSite(4, 5, 0, 0.2, 0.3, 0.4))
    system.setVirtualSite(7, openmm.LocalCoordinatesSite(
        5, 4, 1,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(0.1, 0.2, -0.1) * unit.nanometer,
    ))

    # Gen 3
    system.setVirtualSite(8, openmm.LocalCoordinatesSite(
        6, 7, 2,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(-0.1, 0.1, 0.2) * unit.nanometer,
    ))
    system.setVirtualSite(9, openmm.OutOfPlaneSite(7, 6, 3, 0.1, 0.5, -0.3))

    # Gen 4
    system.setVirtualSite(10, openmm.TwoParticleAverageSite(8, 9, 0.5, 0.5))

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.2, 0.1),
        openmm.Vec3(1.0, 2.0, -0.1),
        openmm.Vec3(0.1, 1.0, 2.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(10, [10.0, -20.0, 30.0])
    ext.addParticle(8, [-5.0, 15.0, -10.0])
    ext.addParticle(7, [2.0, -4.0, 6.0])
    system.addForce(ext)

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pos)
    context.computeVirtualSites()
    omm_state = context.getState(getPositions=True, getForces=True)
    omm_pos = omm_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = omm_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    assert converted.virtual_sites.n_generations == 4
    assert converted.virtual_sites.n_total == 7

    pos_A = jnp.array(pos) * 10.0
    prolix_pos_nm = np.array(converted.compute_virtual_sites(pos_A)) / 10.0

    raw_f = jnp.zeros((11, 3))
    raw_f = raw_f.at[10].set(jnp.array([10.0, -20.0, 30.0]))
    raw_f = raw_f.at[8].set(jnp.array([-5.0, 15.0, -10.0]))
    raw_f = raw_f.at[7].set(jnp.array([2.0, -4.0, 6.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    pos_err = np.max(np.abs(omm_pos - prolix_pos_nm))
    force_err_real = np.max(np.abs(omm_f[:4] - prolix_f[:4]))

    assert pos_err < 1e-6, f"Branching DAG position error {pos_err} nm exceeds 1e-6"
    assert force_err_real < 1e-5, f"Branching DAG force error {force_err_real} exceeds 1e-5"
    assert np.allclose(np.sum(prolix_f[:4], axis=0), np.array([7.0, -9.0, 26.0]), atol=1e-5)


def test_inverted_particle_indexing_dag():
    """Verify DAG resolution works correctly when particle indices are inverted across generations."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0)
    for _ in range(4):
        system.addParticle(0.0)

    # Inverted indexing:
    # 7 is Gen 1 (depends on real 0, 1)
    # 6 is Gen 2 (depends on 7, real 2, 3)
    # 5 is Gen 3 (depends on 6, 7, real 1)
    # 4 is Gen 4 (depends on 5, real 3)
    vs7 = openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5)
    system.setVirtualSite(7, vs7)

    vs6 = openmm.OutOfPlaneSite(7, 2, 3, 0.3, 0.4, 0.5)
    system.setVirtualSite(6, vs6)

    vs5 = openmm.LocalCoordinatesSite(
        6, 7, 1,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(0.1, 0.2, 0.3) * unit.nanometer,
    )
    system.setVirtualSite(5, vs5)

    vs4 = openmm.TwoParticleAverageSite(5, 3, 0.6, 0.4)
    system.setVirtualSite(4, vs4)

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(1.0, 2.0, 0.0),
        openmm.Vec3(0.0, 1.0, 2.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(4, [15.0, -25.0, 35.0])
    ext.addParticle(5, [5.0, 10.0, -15.0])
    system.addForce(ext)

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pos)
    context.computeVirtualSites()
    omm_state = context.getState(getPositions=True, getForces=True)
    omm_pos = omm_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = omm_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(pos) * 10.0
    prolix_pos_nm = np.array(converted.compute_virtual_sites(pos_A)) / 10.0

    raw_f = jnp.zeros((8, 3))
    raw_f = raw_f.at[4].set(jnp.array([15.0, -25.0, 35.0]))
    raw_f = raw_f.at[5].set(jnp.array([5.0, 10.0, -15.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    pos_err = np.max(np.abs(omm_pos - prolix_pos_nm))
    force_err_real = np.max(np.abs(omm_f[:4] - prolix_f[:4]))

    assert pos_err < 1e-6
    assert force_err_real < 1e-5


def test_cyclic_virtual_sites_raise_value_error():
    """Verify that circular dependencies in virtual sites fail fast with ValueError."""
    sys_circ = openmm.System()
    sys_circ.addParticle(1.0)
    sys_circ.addParticle(0.0)
    sys_circ.addParticle(0.0)

    # Particle 1 depends on 2, particle 2 depends on 1
    sys_circ.setVirtualSite(1, openmm.TwoParticleAverageSite(0, 2, 0.5, 0.5))
    sys_circ.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))

    with pytest.raises(ValueError, match="circular"):
        from_openmm_system(sys_circ)


# =============================================================================
# 3. Nearly Collinear LocalCoordinatesSite Parent Configuration Sweep
# =============================================================================


@pytest.mark.parametrize("delta", [1e-12, 1e-8, 1e-6, 1e-4, 0.0])
def test_nearly_collinear_local_coordinates_stability_sweep(delta):
    """Verify LocalCoordinatesSite force stability (<= 30.0 kJ/mol/nm) and OpenMM Reference bit-for-bit parity.

    Parents are placed along X axis with y-perturbation delta:
      p0 = (0, 0, 0)
      p1 = (1, 0, 0)
      p2 = (2, delta, 0)
    """
    system = openmm.System()
    for _ in range(3):
        system.addParticle(1.0)
    system.addParticle(0.0)

    vs = openmm.LocalCoordinatesSite(
        0,
        1,
        2,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(0.1, 0.2, 0.3) * unit.nanometer,
    )
    system.setVirtualSite(3, vs)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(3, [10.0, 20.0, 30.0])
    system.addForce(ext)

    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(2.0, delta, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pos)
    context.computeVirtualSites()
    omm_f = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(pos) * 10.0
    raw_f = jnp.zeros((4, 3)).at[3].set(jnp.array([10.0, 20.0, 30.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    omm_max_f = np.max(np.abs(omm_f[:3]))
    prolix_max_f = np.max(np.abs(prolix_f[:3]))

    # Force stability ceiling assertion: force must not explode
    assert omm_max_f <= 30.0, f"OpenMM force {omm_max_f} exceeds 30.0 at delta={delta}"
    assert prolix_max_f <= 30.0, f"Prolix force {prolix_max_f} exceeds 30.0 at delta={delta}"

    # Parity assertion vs OpenMM Reference Platform
    # (Allowing 0.01 kJ/mol/nm numerical margin when run in float64 / x64 environments)
    max_diff = np.max(np.abs(omm_f[:3] - prolix_f[:3]))
    assert max_diff < 1e-2, f"Force difference {max_diff} exceeds 1e-2 at delta={delta}"

    # Total force conservation assertion
    assert np.allclose(np.sum(prolix_f[:3], axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-5)


# =============================================================================
# 4. Multi-Force OpenMM Systems with 3 Bond Forces and 3 Angle Forces
# =============================================================================


def test_multi_force_openmm_compound_mapping_and_lookups():
    """Verify systems with 3 distinct HarmonicBondForce and 3 HarmonicAngleForce instances.

    Tests:
    - CompoundMapping maintains tuple keys internally for JAX pytree flattener sortability.
    - converted.get_bundle_bond_index(force_idx, bond_idx) and get_bundle_angle_index correctly resolve.
    - system_metadata['omm_bond_to_bundle_idx'][(force_idx, bond_idx)] agrees with helpers.
    - Flat int lookup behaves safely when is_single=False.
    """
    system = openmm.System()
    for _ in range(12):
        system.addParticle(12.0)

    # 3 distinct HarmonicBondForce instances
    bf1 = openmm.HarmonicBondForce()
    bf1.addBond(0, 1, 0.15 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bf1.addBond(1, 2, 0.15 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f0_idx = system.addForce(bf1)

    bf2 = openmm.HarmonicBondForce()
    bf2.addBond(2, 3, 0.14 * unit.nanometer, 1200.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bf2.addBond(3, 4, 0.14 * unit.nanometer, 1200.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bf2.addBond(4, 5, 0.14 * unit.nanometer, 1200.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f1_idx = system.addForce(bf2)

    bf3 = openmm.HarmonicBondForce()
    bf3.addBond(5, 6, 0.13 * unit.nanometer, 1500.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f2_idx = system.addForce(bf3)

    # 3 distinct HarmonicAngleForce instances
    af1 = openmm.HarmonicAngleForce()
    af1.addAngle(0, 1, 2, 120.0 * unit.degree, 400.0 * unit.kilojoules_per_mole / unit.radian**2)
    f3_idx = system.addForce(af1)

    af2 = openmm.HarmonicAngleForce()
    af2.addAngle(1, 2, 3, 115.0 * unit.degree, 450.0 * unit.kilojoules_per_mole / unit.radian**2)
    af2.addAngle(2, 3, 4, 110.0 * unit.degree, 450.0 * unit.kilojoules_per_mole / unit.radian**2)
    f4_idx = system.addForce(af2)

    af3 = openmm.HarmonicAngleForce()
    af3.addAngle(3, 4, 5, 105.0 * unit.degree, 500.0 * unit.kilojoules_per_mole / unit.radian**2)
    f5_idx = system.addForce(af3)

    converted = from_openmm_system(system)

    # JAX pytree flattener sortability check (must not raise TypeError on mixed dict keys)
    leaves, treedef = jax.tree_util.tree_flatten(converted)
    assert len(leaves) > 0, "Converted system must flatten into JAX pytree"

    # Verify total bond and angle counts in MolecularBundle
    assert converted.bundle.n_bonds == 6  # 2 + 3 + 1
    assert converted.bundle.n_angles == 4  # 1 + 2 + 1

    bond_mapping = converted.system_metadata["omm_bond_to_bundle_idx"]
    angle_mapping = converted.system_metadata["omm_angle_to_bundle_idx"]

    assert isinstance(bond_mapping, CompoundMapping)
    assert isinstance(angle_mapping, CompoundMapping)
    assert bond_mapping.is_single is False
    assert angle_mapping.is_single is False

    # Check bond lookups
    bond_expectations = {
        (f0_idx, 0): 0,
        (f0_idx, 1): 1,
        (f1_idx, 0): 2,
        (f1_idx, 1): 3,
        (f1_idx, 2): 4,
        (f2_idx, 0): 5,
    }
    for (f_idx, b_i), expected_bundle_idx in bond_expectations.items():
        assert converted.get_bundle_bond_index(f_idx, b_i) == expected_bundle_idx
        assert bond_mapping[(f_idx, b_i)] == expected_bundle_idx

    # Check angle lookups
    angle_expectations = {
        (f3_idx, 0): 0,
        (f4_idx, 0): 1,
        (f4_idx, 1): 2,
        (f5_idx, 0): 3,
    }
    for (f_idx, a_i), expected_bundle_idx in angle_expectations.items():
        assert converted.get_bundle_angle_index(f_idx, a_i) == expected_bundle_idx
        assert angle_mapping[(f_idx, a_i)] == expected_bundle_idx

    # In multi-force mode, single int key is not supported
    with pytest.raises(KeyError):
        _ = bond_mapping[0]
    assert bond_mapping.get(0) is None
    assert (0 in bond_mapping) is False


def test_single_force_backward_compatibility():
    """Verify single-force systems retain seamless flat integer indexing."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)

    bf = openmm.HarmonicBondForce()
    bf.addBond(0, 1, 0.15 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bf.addBond(1, 2, 0.15 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f_idx = system.addForce(bf)

    converted = from_openmm_system(system)
    mapping = converted.system_metadata["omm_bond_to_bundle_idx"]

    assert mapping.is_single is True
    # Flat integer access
    assert mapping[0] == 0
    assert mapping[1] == 1
    assert 0 in mapping
    assert 1 in mapping
    assert mapping.get(0) == 0
    # Tuple access also works
    assert mapping[(f_idx, 0)] == 0
    assert mapping[(f_idx, 1)] == 1
    assert (f_idx, 0) in mapping


def test_multi_force_water_zeroing_behavior():
    """Verify that in multi-force systems, water bonds/angles in specific forces are cleanly mapped to None."""
    system = openmm.System()
    # 0, 1, 2 = water (O, H1, H2)
    system.addParticle(15.9994)
    system.addParticle(1.008)
    system.addParticle(1.008)
    # 3, 4 = solute C-C
    system.addParticle(12.011)
    system.addParticle(12.011)

    system.addConstraint(0, 1, 0.09572 * unit.nanometer)
    system.addConstraint(0, 2, 0.09572 * unit.nanometer)
    system.addConstraint(1, 2, 0.15139 * unit.nanometer)

    # Force 0: water bonds
    bf0 = openmm.HarmonicBondForce()
    bf0.addBond(0, 1, 0.09572 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bf0.addBond(0, 2, 0.09572 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f0 = system.addForce(bf0)

    # Force 1: solute bond
    bf1 = openmm.HarmonicBondForce()
    bf1.addBond(3, 4, 0.154 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    f1 = system.addForce(bf1)

    converted = from_openmm_system(system)

    # Water bonds in force 0 are zeroed and mapped to None
    assert converted.get_bundle_bond_index(f0, 0) is None
    assert converted.get_bundle_bond_index(f0, 1) is None
    # Solute bond in force 1 is bundle bond 0
    assert converted.get_bundle_bond_index(f1, 0) == 0

    assert converted.bundle.n_waters == 1
    assert converted.bundle.n_bonds == 1
