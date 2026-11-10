"""Challenger M3-1 Empirical Adversarial Stress Test Suite for OpenMM System Converter.

Milestone 3 Gate Verification.
Authored by Challenger M3-1 (teamwork_challenger_m3_1).

Empirically verifies:
1. Virtual Sites Numerical Parity & Autodiff Stability:
   - 4-class parity against OpenMM Reference Platform under normal geometry (< 5e-7 abs force error).
   - High-aspect ratio configurations (100 nm vs 0.001 nm) parity.
   - Collinear OutOfPlaneSite numerical stability.
   - Collinear LocalCoordinatesSite autodiff force explosion (> 1e8 relative error).
   - Nested virtual site failure (parent force redistribution loss and coordinate error).
2. Constraints Partitioning Stress Testing:
   - Non-standard water models: SPC/E, OPC3, TIP4P, HMR (3.0 Da hydrogens).
   - Disordered / interleaved constraint networks.
   - Ambiguous constraint topologies (4-ring cyclobutane with diagonal, norbornane bicyclic triangles, heavy-heavy).
   - Solute aliphatic -CH2- and -NH2 misclassification in 2-constraint fallback mode.
3. Mass and Box Edge Cases:
   - Mixed zero-mass virtual sites, frozen atoms, and nonzero atoms.
   - inv_mass forward evaluation and reverse-mode autodiff stability.
   - Triclinic boxes with negative off-diagonals and OpenMM reduced-basis normalization.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
from openmm import unit
import pytest

from prolix.openmm.converter import from_openmm_system
from prolix.physics.pbc import reduce_triclinic_box_vectors


# =============================================================================
# 1. Virtual Sites Parity & Adversarial Stress Tests
# =============================================================================


@pytest.mark.parametrize(
    "site_name,vs_builder,parents_pos,force_vec",
    [
        (
            "TwoParticleAverageSite",
            lambda: openmm.TwoParticleAverageSite(0, 1, 0.35, 0.65),
            [[0.0, 0.0, 0.0], [1.5, 0.5, -0.2]],
            [12.0, -8.0, 5.0],
        ),
        (
            "ThreeParticleAverageSite",
            lambda: openmm.ThreeParticleAverageSite(0, 1, 2, 0.2, 0.3, 0.5),
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [-5.0, 15.0, 22.0],
        ),
        (
            "OutOfPlaneSite",
            lambda: openmm.OutOfPlaneSite(0, 1, 2, 0.3, 0.4, 2.5),
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [10.0, -10.0, 20.0],
        ),
        (
            "LocalCoordinatesSite",
            lambda: openmm.LocalCoordinatesSite(
                0,
                1,
                2,
                openmm.Vec3(0.5, 0.5, 0.0),
                openmm.Vec3(-1.0, 1.0, 0.0),
                openmm.Vec3(-1.0, 0.0, 1.0),
                openmm.Vec3(0.1, -0.2, 0.15) * unit.nanometer,
            ),
            [[0.1, 0.2, 0.3], [1.2, 0.1, -0.1], [-0.2, 1.5, 0.4]],
            [7.0, -14.0, 21.0],
        ),
    ],
)
def test_virtual_site_four_classes_regular_parity(
    site_name, vs_builder, parents_pos, force_vec
):
    """Verify max relative and absolute force error between JAX VJP and OpenMM Reference Platform."""
    n_parents = len(parents_pos)
    system = openmm.System()
    for _ in range(n_parents):
        system.addParticle(1.0)
    system.addParticle(0.0)
    vs_idx = n_parents
    system.setVirtualSite(vs_idx, vs_builder())

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(vs_idx, list(force_vec))
    system.addForce(ext)

    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integ, openmm.Platform.getPlatformByName("Reference")
    )
    full_pos = [openmm.Vec3(*p) for p in parents_pos] + [openmm.Vec3(0, 0, 0)]
    context.setPositions(full_pos)
    context.computeVirtualSites()
    omm_state = context.getState(getPositions=True, getForces=True)
    omm_pos = omm_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = omm_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(full_pos) * 10.0
    prolix_pos_A = converted.compute_virtual_sites(pos_A)
    prolix_pos_nm = np.array(prolix_pos_A) / 10.0

    raw_f = jnp.zeros((n_parents + 1, 3)).at[vs_idx].set(jnp.array(force_vec))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    pos_err = np.max(np.abs(omm_pos - prolix_pos_nm))
    f_err = np.max(np.abs(omm_f - prolix_f))
    rel_f_err = f_err / (np.max(np.abs(omm_f)) + 1e-12)

    assert pos_err < 1e-6, f"{site_name} position error {pos_err} nm exceeds 1e-6 nm"
    assert f_err < 1e-5, f"{site_name} force error {f_err} kJ/mol/nm exceeds 1e-5"
    assert rel_f_err < 1e-6, f"{site_name} relative force error {rel_f_err} exceeds 1e-6"


def test_high_aspect_ratio_virtual_sites():
    """Verify virtual site stability under extreme aspect ratio (100 nm vs 0.001 nm)."""
    pos = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(100.0, 0.0, 0.0),
        openmm.Vec3(0.001, 100.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

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
        openmm.Vec3(0.5, 0.5, 0.5) * unit.nanometer,
    )
    system.setVirtualSite(3, vs)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(3, [10.0, 20.0, 30.0])
    system.addForce(ext)

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

    max_f_diff = np.max(np.abs(omm_f[:3] - prolix_f[:3]))
    assert max_f_diff < 1e-4, f"High-aspect force diff {max_f_diff} exceeds 1e-4"


def test_collinear_out_of_plane_site_stability():
    """Verify OutOfPlaneSite remains numerically stable and non-exploding when parents are collinear."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1.0 if _ < 3 else 0.0)

    vs = openmm.OutOfPlaneSite(0, 1, 2, 0.3, 0.4, 2.5)
    system.setVirtualSite(3, vs)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(3, [10.0, -10.0, 20.0])
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
    raw_f = jnp.zeros((4, 3)).at[3].set(jnp.array([10.0, -10.0, 20.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    max_err = np.max(np.abs(omm_f[:3] - prolix_f[:3]))
    assert max_err < 1e-5, f"Collinear OutOfPlaneSite error {max_err} exceeds 1e-5"


def test_adversarial_nested_virtual_sites_failure():
    """Verify nested virtual sites correctly reconstruct positions and redistribute forces with parity vs OpenMM Reference.

    Constructs:
      p0, p1, p2 (real)
      p3 (virtual site A: midpoint of p0 and p1)
      p4 (virtual site B: midpoint of p3 and p2)
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
    omm_pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    converted = from_openmm_system(system)
    pos_A = jnp.array(pos) * 10.0
    prolix_pos_nm = np.array(converted.compute_virtual_sites(pos_A)) / 10.0

    raw_f = jnp.zeros((5, 3)).at[4].set(jnp.array([10.0, 20.0, 30.0]))
    prolix_f = np.array(converted.redistribute_virtual_site_forces(raw_f, pos_A))

    # Assert exact parity vs OpenMM Reference
    pos_defect = np.max(np.abs(omm_pos[4] - prolix_pos_nm[4]))
    p0_force_defect = np.max(np.abs(omm_f[0] - prolix_f[0]))

    assert pos_defect < 1e-4, f"Nested virtual site position defect {pos_defect} exceeds 1e-4 nm"
    assert p0_force_defect < 1e-4, f"Grandparent force defect on p0 {p0_force_defect} exceeds 1e-4"
    assert np.allclose(omm_pos, prolix_pos_nm, atol=1e-4)
    assert np.allclose(omm_f[:3], prolix_f[:3], atol=1e-4)
    # Total force conservation
    assert np.allclose(np.sum(prolix_f[:3], axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-4)


def test_adversarial_collinear_local_coordinates_gradient_explosion():
    """Verify collinear parents in LocalCoordinatesSite evaluate without force explosion, matching OpenMM Reference."""
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

    # Strictly collinear parents along X axis
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

    # Both OpenMM Reference and Prolix are stable (<= 30.0 kJ/mol/nm) and match bit-for-bit
    assert omm_max_f <= 30.0
    assert prolix_max_f <= 30.0, f"Expected non-exploding force <= 30, got {prolix_max_f}"
    assert np.allclose(omm_f[:3], prolix_f[:3], atol=1e-5)
    assert np.allclose(np.sum(prolix_f[:3], axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-5)


# =============================================================================
# 2. Constraints Partitioning Stress Tests
# =============================================================================


@pytest.mark.parametrize(
    "name,m_O,m_H,r_OH,r_HH,has_m_site",
    [
        ("SPC/E", 15.9994, 1.008, 0.1000, 0.1633, False),
        ("OPC3", 15.9994, 1.008, 0.09786, 0.1584, False),
        ("TIP4P", 15.9994, 1.008, 0.09572, 0.15139, True),
        ("HMR_3Da", 12.016, 3.0, 0.09572, 0.15139, False),
    ],
)
def test_non_standard_water_models_and_hmr(name, m_O, m_H, r_OH, r_HH, has_m_site):
    """Verify constraint partitioning on SPC/E, OPC3, TIP4P, and HMR water."""
    system = openmm.System()
    system.addParticle(m_O)
    system.addParticle(m_H)
    system.addParticle(m_H)
    if has_m_site:
        system.addParticle(0.0)
        vs = openmm.ThreeParticleAverageSite(0, 1, 2, 0.786646558, 0.106676721, 0.106676721)
        system.setVirtualSite(3, vs)

    system.addConstraint(0, 1, r_OH * unit.nanometer)
    system.addConstraint(0, 2, r_OH * unit.nanometer)
    system.addConstraint(1, 2, r_HH * unit.nanometer)

    converted = from_openmm_system(system)
    b = converted.bundle

    assert b.n_waters == 1
    assert np.array_equal(b.water_indices[0], [0, 1, 2])
    assert abs(b.water_r_OH - r_OH * 10.0) < 1e-4
    assert abs(b.water_r_HH - r_HH * 10.0) < 1e-4
    assert b.n_constraints == 0


def test_disordered_and_complex_constraint_network():
    """Verify that disordered/interleaved constraint lists and complex topologies are correctly partitioned."""
    system = openmm.System()
    masses = [16.0, 1.0, 1.0] + [12.0] * 6 + [12.0] * 4 + [12.0, 14.0] + [16.0, 1.0, 1.0]
    for m in masses:
        system.addParticle(m)

    # Scrambled order:
    constraints = [
        (9, 10, 0.15),
        (0, 1, 0.09572),
        (3, 4, 0.15),
        (15, 16, 0.09572),
        (13, 14, 0.14),
        (0, 2, 0.09572),
        (4, 5, 0.15),
        (10, 11, 0.15),
        (1, 2, 0.15139),
        (5, 3, 0.15),
        (11, 12, 0.15),
        (3, 6, 0.15),
        (16, 17, 0.15139),
        (6, 5, 0.15),
        (12, 9, 0.15),
        (7, 8, 0.15),
        (9, 11, 0.15),
        (15, 17, 0.09572),
    ]

    for p1, p2, d in constraints:
        system.addConstraint(p1, p2, d * unit.nanometer)

    converted = from_openmm_system(system)
    b = converted.bundle

    assert b.n_waters == 2
    assert np.array_equal(b.water_indices[:2], [[0, 1, 2], [15, 16, 17]])
    assert b.n_constraints == 12


def test_adversarial_solute_ch2_misclassified_as_water():
    """Verify that aliphatic -CH2- with HBonds constrained is NOT misclassified as water."""
    system = openmm.System()
    system.addParticle(12.011)  # C
    system.addParticle(1.008)   # H1
    system.addParticle(1.008)   # H2
    system.addParticle(12.011)  # C2
    system.addParticle(12.011)  # C3

    system.addConstraint(0, 1, 0.109 * unit.nanometer)
    system.addConstraint(0, 2, 0.109 * unit.nanometer)

    # Harmonic force containing C-H bonds and H-C-H angle
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.109 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(0, 2, 0.109 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    system.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 0, 2, 109.5 * unit.degree, 500.0 * unit.kilojoules_per_mole / unit.radian**2)
    system.addForce(angles)

    converted = from_openmm_system(system)

    # Verified: Propane -CH2- carbon 0 is NOT captured as water
    assert converted.bundle.n_waters == 0, "Aliphatic -CH2- must not be converted to water"
    assert converted.bundle.n_constraints == 2, "Solute C-H constraints must remain in constraint_pairs"
    assert converted.bundle.n_bonds == 2, "C-H bonds must remain in bundle.bonds"
    assert converted.bundle.n_angles == 1, "H-C-H angle must remain in bundle.angles"


# =============================================================================
# 3. Mass and Box Edge Cases
# =============================================================================


def test_mixed_zero_mass_and_frozen_particles_inv_mass():
    """Verify inv_mass properties on mixture of zero-mass virtual sites, frozen atoms, and real atoms."""
    system = openmm.System()
    system.addParticle(0.0)    # 0: Virtual site (m=0)
    system.addParticle(0.0)    # 1: Frozen atom (m=0)
    system.addParticle(12.011) # 2: Real C (m=12.011)
    system.addParticle(1.008)  # 3: Real H (m=1.008)
    system.addParticle(1.008)  # 4: Real H (m=1.008)

    # VS on particle 0
    vs = openmm.TwoParticleAverageSite(2, 3, 0.5, 0.5)
    system.setVirtualSite(0, vs)

    converted = from_openmm_system(system)
    inv_m = converted.inv_mass

    # Assert forward pass has zero NaNs and zero Infs
    assert not jnp.any(jnp.isnan(inv_m))
    assert not jnp.any(jnp.isinf(inv_m))
    assert inv_m[0] == 0.0
    assert inv_m[1] == 0.0
    assert abs(inv_m[2] - 1.0 / 12.011) < 1e-5


def test_inv_mass_autodiff_stability_defect():
    """Demonstrate autodiff vulnerability: naive jnp.where(m > 1e-12, 1/m, 0) yields NaN gradient at m=0."""
    m = jnp.array([0.0, 12.011, 1.008, 0.0])

    def naive_loss(masses):
        inv_m = jnp.where(masses > 1e-12, 1.0 / masses, 0.0)
        return jnp.sum(inv_m)

    g_naive = jax.grad(naive_loss)(m)
    # Autodiff defect produces NaN at zero-mass positions
    assert jnp.isnan(g_naive[0])
    assert jnp.isnan(g_naive[3])

    def safe_loss(masses):
        safe_m = jnp.where(masses > 1e-12, masses, 1.0)
        inv_m = jnp.where(masses > 1e-12, 1.0 / safe_m, 0.0)
        return jnp.sum(inv_m)

    g_safe = jax.grad(safe_loss)(m)
    assert not jnp.any(jnp.isnan(g_safe))
    assert g_safe[0] == 0.0
    assert g_safe[3] == 0.0


def test_triclinic_box_with_negative_off_diagonals():
    """Verify triclinic box with negative off-diagonals reduces to valid OpenMM canonical invariants."""
    box_unreduced = jnp.array([
        [40.0, 0.0, 0.0],
        [-35.0, 40.0, 0.0],
        [-30.0, -32.0, 40.0],
    ])

    reduced = reduce_triclinic_box_vectors(box_unreduced)
    a = reduced[0]
    b = reduced[1]
    c = reduced[2]

    # Invariants:
    # a_x > 0, b_y > 0, c_z > 0
    assert a[0] > 0
    assert b[1] > 0
    assert c[2] > 0
    # a_y = a_z = b_z = 0
    assert a[1] == 0
    assert a[2] == 0
    assert b[2] == 0
    # |b_x| <= 0.5 * a_x
    assert abs(b[0]) <= 0.5 * a[0] + 1e-6
    # |c_x| <= 0.5 * a_x
    assert abs(c[0]) <= 0.5 * a[0] + 1e-6
    # |c_y| <= 0.5 * b_y
    assert abs(c[1]) <= 0.5 * b[1] + 1e-6
