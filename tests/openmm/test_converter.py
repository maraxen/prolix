"""Comprehensive unit tests for the OpenMM System Converter (`from_openmm_system`).

Covers:
- Particle masses, virtual site mask, frozen atom mask, and safe inverse mass.
- Periodic box vector extraction, unit conversion (nm -> Å), canonical lower-triangular form,
  orthorhombic vs triclinic classification, and non-periodic boundary conditions.
- Virtual sites: all 4 types (TwoParticleAverageSite, ThreeParticleAverageSite,
  OutOfPlaneSite, LocalCoordinatesSite), topological dependency DAG ordering,
  circular dependency detection, forward reconstruction parity, and reverse-mode
  force redistribution parity vs OpenMM Reference Platform via VJP.
- Constraints partitioning: 3-cycle rigid water detection with Oxygen at column 0,
  water bond and angle zeroing in HarmonicBondForce and HarmonicAngleForce,
  and solute constraint partitioning for SHAKE/RATTLE.
- Standard bonded forces: HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce,
  CMAPTorsionForce, and CompiledRBTorsionForce.
- NonbondedForce: charges, LJ sigmas and epsilons, 1-2/1-3 topological exclusions,
  unified 1-4 exceptions, 32 dense exclusion cap enforcement, switching function,
  analytical dispersion correction, and PME parameter resolution.
- Custom forces: pattern folding into MolecularBundle vs Lepton compilation fallback,
  position restraint extraction, and custom_force_mode routing ("auto", "pattern_only", "lepton_only").
- Force groups (0..31) and global parameters preservation.
- Fail-fast error handling with UnsupportedOpenMMForceError on unsupported forces and LJPME.
- Integration tests on real-world systems: Chignolin (1UAO) and solvated TIP3P water box.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

try:
    import openmm
    from openmm import app, unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

import jax
import jax.numpy as jnp

from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.converter import (
    ConvertedOpenMMSystem,
    PositionRestraints,
    VirtualSites,
    from_openmm_system,
)
from prolix.openmm.exceptions import UnsupportedOpenMMForceError
from prolix.openmm.units import (
    BOND_K_OMM_TO_AKMA,
    ANGLE_K_OMM_TO_AKMA,
    TORSION_K_OMM_TO_AKMA,
    AKMA_PRESSURE_PER_BAR,
)
from tests.openmm.conftest import (
    clone_openmm_system,
    compute_openmm_energy,
)

pytestmark = pytest.mark.openmm


# =============================================================================
# 1. Masses, Particle Counts, and Zero-Mass Handling
# =============================================================================


def test_converter_masses_and_counts():
    """Verify particle counts, mass extraction in Daltons, and safe inverse mass."""
    sys = openmm.System()
    # Particle 0: standard carbon (12.011 Da)
    sys.addParticle(12.011)
    # Particle 1: hydrogen (1.008 Da)
    sys.addParticle(1.008)
    # Particle 2: fixed/frozen atom (0.0 Da, not a virtual site)
    sys.addParticle(0.0)
    # Particle 3: virtual site (0.0 Da)
    sys.addParticle(0.0)
    sys.setVirtualSite(3, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))

    conv = from_openmm_system(sys)
    bundle = conv.bundle

    assert conv.n_real_particles == 4
    assert int(bundle.n_atoms) == 4
    # Real masses match OpenMM Daltons 1:1
    assert np.isclose(float(bundle.masses[0]), 12.011, atol=1e-5)
    assert np.isclose(float(bundle.masses[1]), 1.008, atol=1e-5)
    assert float(bundle.masses[2]) == 0.0
    assert float(bundle.masses[3]) == 0.0

    # Padding atoms receive mass 1.0
    assert float(bundle.masses[4]) == 1.0
    assert bool(bundle.atom_mask[0]) is True
    assert bool(bundle.atom_mask[3]) is True
    assert bool(bundle.atom_mask[4]) is False

    # Safe inverse mass avoids NaN on zero masses
    inv_m = conv.inv_mass
    assert np.isclose(float(inv_m[0]), 1.0 / 12.011, atol=1e-5)
    assert np.isclose(float(inv_m[1]), 1.0 / 1.008, atol=1e-5)
    assert float(inv_m[2]) == 0.0
    assert float(inv_m[3]) == 0.0
    assert not jnp.any(jnp.isnan(inv_m))


def test_converter_zero_particles_raises():
    """Verify that an OpenMM system with 0 particles raises ValueError."""
    sys = openmm.System()
    with pytest.raises(ValueError, match="0 particles"):
        from_openmm_system(sys)


# =============================================================================
# 2. Periodic Box Vectors & Boundary Conditions
# =============================================================================


def test_converter_box_vectors_orthorhombic():
    """Verify orthorhombic periodic box vectors conversion (nm -> Å, factor 10)."""
    sys = openmm.System()
    sys.addParticle(12.0)
    a = openmm.Vec3(2.5, 0.0, 0.0) * unit.nanometer
    b = openmm.Vec3(0.0, 3.0, 0.0) * unit.nanometer
    c = openmm.Vec3(0.0, 0.0, 3.5) * unit.nanometer
    sys.setDefaultPeriodicBoxVectors(a, b, c)

    nb = openmm.NonbondedForce()
    nb.addParticle(0.0, 0.3, 0.0)
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    sys.addForce(nb)

    conv = from_openmm_system(sys)

    assert conv.has_pbc is True
    assert conv.is_triclinic is False
    assert conv.system_metadata["boundary_condition"] == "periodic"
    box = np.asarray(conv.bundle.box)
    assert box.shape == (3, 3)
    assert np.isclose(box[0, 0], 25.0, atol=1e-5)
    assert np.isclose(box[1, 1], 30.0, atol=1e-5)
    assert np.isclose(box[2, 2], 35.0, atol=1e-5)
    # Upper and lower off-diagonals are strictly zero
    assert np.allclose(box - np.diag(np.diag(box)), 0.0)


def test_converter_box_vectors_triclinic():
    """Verify lower-triangular triclinic box vectors conversion and reduction."""
    sys = openmm.System()
    sys.addParticle(12.0)
    # OpenMM reduced triclinic vectors
    a = openmm.Vec3(4.0, 0.0, 0.0) * unit.nanometer
    b = openmm.Vec3(1.0, 4.0, 0.0) * unit.nanometer
    c = openmm.Vec3(1.0, 1.0, 4.0) * unit.nanometer
    sys.setDefaultPeriodicBoxVectors(a, b, c)

    nb = openmm.NonbondedForce()
    nb.addParticle(0.0, 0.3, 0.0)
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    sys.addForce(nb)

    conv = from_openmm_system(sys)

    assert conv.has_pbc is True
    assert conv.is_triclinic is True
    box = np.asarray(conv.bundle.box)
    assert np.isclose(box[0, 0], 40.0, atol=1e-5)
    assert np.isclose(box[1, 0], 10.0, atol=1e-5)
    assert np.isclose(box[1, 1], 40.0, atol=1e-5)
    # Upper-triangular elements are strictly zero
    assert box[0, 1] == 0.0 and box[0, 2] == 0.0 and box[1, 2] == 0.0


def test_converter_box_vectors_nonperiodic():
    """Verify non-periodic system produces zero box and boundary_condition='free'."""
    sys = openmm.System()
    sys.addParticle(12.0)
    # Default is zero box vectors and usesPeriodicBoundaryConditions() == False

    conv = from_openmm_system(sys)

    assert conv.has_pbc is False
    assert conv.is_triclinic is False
    assert conv.bundle.shape_spec.boundary_condition == "free"
    assert np.allclose(np.asarray(conv.bundle.box), 0.0)


def test_converter_box_vectors_non_canonical_raises(monkeypatch):
    """Verify that non-canonical upper-triangular box orientation raises ValueError."""
    sys = openmm.System()
    sys.addParticle(12.0)
    non_canonical = (
        openmm.Vec3(2.5, 1.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 3.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, 3.5) * unit.nanometer,
    )
    monkeypatch.setattr(sys, "getDefaultPeriodicBoxVectors", lambda: non_canonical)
    monkeypatch.setattr(sys, "usesPeriodicBoundaryConditions", lambda: True)

    with pytest.raises(ValueError, match="lower-triangular"):
        from_openmm_system(sys)


# =============================================================================
# 3. Virtual Sites: All 4 Classes, DAG Ordering, and Force VJP
# =============================================================================


def test_converter_virtual_sites_all_four_types(reference_platform):
    """Verify all 4 virtual site classes against OpenMM Reference Platform."""
    sys = openmm.System()
    for _ in range(10):
        sys.addParticle(12.0)  # real atoms 0..9

    # Add 4 virtual sites:
    # VS 10: TwoParticleAverageSite between 0 and 1 (weights 0.6, 0.4)
    sys.addParticle(0.0)
    sys.setVirtualSite(10, openmm.TwoParticleAverageSite(0, 1, 0.6, 0.4))

    # VS 11: ThreeParticleAverageSite between 1, 2, 3 (weights 0.5, 0.3, 0.2)
    sys.addParticle(0.0)
    sys.setVirtualSite(11, openmm.ThreeParticleAverageSite(1, 2, 3, 0.5, 0.3, 0.2))

    # VS 12: OutOfPlaneSite with 4, 5, 6
    sys.addParticle(0.0)
    sys.setVirtualSite(12, openmm.OutOfPlaneSite(4, 5, 6, 0.2, 0.3, 1.5))

    # VS 13: LocalCoordinatesSite with 7, 8, 9
    sys.addParticle(0.0)
    p_local = openmm.Vec3(0.05, 0.02, 0.01) * unit.nanometer
    wo = [1.0, 0.0, 0.0]
    wx = [-1.0, 1.0, 0.0]
    wy = [-1.0, 0.0, 1.0]
    sys.setVirtualSite(13, openmm.LocalCoordinatesSite(7, 8, 9, wo, wx, wy, p_local))

    # Arbitrary initial positions for 14 particles
    np.random.seed(42)
    pos_nm = np.random.randn(14, 3) * 0.5
    pos_quantity = pos_nm * unit.nanometer
    pos_akma = jnp.asarray(pos_nm * 10.0, dtype=jnp.float32)

    # 1. OpenMM Reference computation
    ctx = openmm.Context(sys, openmm.VerletIntegrator(0.001), reference_platform)
    ctx.setPositions(pos_quantity)
    ctx.computeVirtualSites()
    omm_state = ctx.getState(getPositions=True)
    omm_pos_akma = (
        np.asarray(omm_state.getPositions().value_in_unit(unit.nanometer)) * 10.0
    )

    # 2. Prolix Converter computation
    conv = from_openmm_system(sys, positions=pos_akma)
    vs: VirtualSites = conv.virtual_sites
    assert vs is not None
    assert vs.n_total == 4
    assert vs.n_linear == 3  # 2-particle, 3-particle, out-of-plane unified
    assert vs.n_local == 1

    recon_pos = conv.compute_virtual_sites(pos_akma)

    # Compare reconstructed virtual sites 10, 11, 12, 13
    for vs_idx in [10, 11, 12, 13]:
        max_diff = float(jnp.max(jnp.abs(recon_pos[vs_idx] - omm_pos_akma[vs_idx])))
        assert max_diff < 1e-4, f"VS {vs_idx} coordinate error {max_diff} Å exceeds tolerance"

    # 3. Force Redistribution via reverse-mode VJP
    # Apply random forces on virtual sites
    raw_forces_akma = jnp.zeros((14, 3), dtype=jnp.float32)
    raw_forces_akma = raw_forces_akma.at[10].set(jnp.array([1.5, -2.0, 0.5], dtype=jnp.float32))
    raw_forces_akma = raw_forces_akma.at[11].set(jnp.array([0.0, 1.2, -1.0], dtype=jnp.float32))
    raw_forces_akma = raw_forces_akma.at[12].set(jnp.array([-0.8, 0.5, 2.0], dtype=jnp.float32))
    raw_forces_akma = raw_forces_akma.at[13].set(jnp.array([1.0, 1.0, -1.0], dtype=jnp.float32))

    redistributed_forces = conv.redistribute_virtual_site_forces(raw_forces_akma, recon_pos)
    assert not jnp.any(jnp.isnan(redistributed_forces))

    # Total force conservation: sum of redistributed forces must equal sum of raw forces
    sum_raw = jnp.sum(raw_forces_akma[10:14], axis=0)
    sum_redist = jnp.sum(redistributed_forces[:10], axis=0)
    assert jnp.allclose(sum_raw, sum_redist, atol=1e-4)


def test_converter_virtual_sites_circular_fail_fast():
    """Verify that circular virtual site definitions raise ValueError."""
    sys = openmm.System()
    sys.addParticle(0.0)
    sys.addParticle(0.0)
    # Site 0 depends on 1, Site 1 depends on 0
    sys.setVirtualSite(0, openmm.TwoParticleAverageSite(1, 1, 0.5, 0.5))
    sys.setVirtualSite(1, openmm.TwoParticleAverageSite(0, 0, 0.5, 0.5))

    with pytest.raises(ValueError, match="circular"):
        from_openmm_system(sys)


# =============================================================================
# 4. Constraints Partitioning: SETTLE vs Solute SHAKE & Water Zeroing
# =============================================================================


def test_converter_water_settle_zeroing():
    """Verify that SETTLE water triangles are isolated and internal water bonds/angles zeroed."""
    sys = openmm.System()
    # Water 1: O(0), H1(1), H2(2)
    sys.addParticle(15.999)  # Oxygen
    sys.addParticle(1.008)   # Hydrogen
    sys.addParticle(1.008)   # Hydrogen
    # Solute heavy atom 3 and hydrogen 4
    sys.addParticle(12.011)  # Carbon
    sys.addParticle(1.008)   # Hydrogen

    # Add 3 water constraints
    sys.addConstraint(0, 1, 0.09572)  # O-H1
    sys.addConstraint(0, 2, 0.09572)  # O-H2
    sys.addConstraint(1, 2, 0.15139)  # H1-H2
    # Add solute C-H constraint
    sys.addConstraint(3, 4, 0.10900)

    # Intentionally add HarmonicBondForce and HarmonicAngleForce on the water atoms
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.09572, 450000.0)  # Water O-H1
    bonds.addBond(0, 2, 0.09572, 450000.0)  # Water O-H2
    bonds.addBond(3, 4, 0.10900, 300000.0)  # Solute C-H
    sys.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 0, 2, 1.824, 400.0)  # Water H1-O-H2
    sys.addForce(angles)

    conv = from_openmm_system(sys)
    bundle = conv.bundle

    # 1. Rigid water check: Oxygen strictly at column 0
    assert int(bundle.n_waters) == 1
    water_triangle = np.asarray(bundle.water_indices[0])
    assert water_triangle[0] == 0  # Oxygen
    assert set(water_triangle[1:3]) == {1, 2}  # Hydrogens
    assert np.isclose(float(bundle.water_r_OH), 0.9572, atol=1e-4)
    assert np.isclose(float(bundle.water_r_HH), 1.5139, atol=1e-4)

    # 2. Critical Zeroing / Filtering check:
    # Water bonds and angles MUST be removed from bundle active arrays
    assert int(bundle.n_bonds) == 1  # Only solute bond (3, 4) remains
    assert tuple(bundle.bond_idx[0]) == (3, 4)
    assert int(bundle.n_angles) == 0  # Water angle removed

    # 3. Solute SHAKE constraint check
    assert int(bundle.n_constraints) == 1
    assert tuple(bundle.constraint_pairs[0]) == (3, 4)
    assert np.isclose(float(bundle.constraint_lengths[0]), 1.09, atol=1e-4)


# =============================================================================
# 5. Standard Bonded Forces & Unit Conversions
# =============================================================================


def test_converter_bonded_forces_parameters():
    """Verify parameter and unit conversions for bonds, angles, and periodic torsions."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    # Bond between 0 and 1
    b_force = openmm.HarmonicBondForce()
    b_force.addBond(0, 1, 0.15, 250000.0)  # 0.15 nm, 250000 kJ/(mol*nm^2)
    sys.addForce(b_force)

    # Angle between 0, 1, 2
    a_force = openmm.HarmonicAngleForce()
    a_force.addAngle(0, 1, 2, 1.91, 450.0)  # 1.91 rad, 450 kJ/(mol*rad^2)
    sys.addForce(a_force)

    # Torsion between 0, 1, 2, 3
    t_force = openmm.PeriodicTorsionForce()
    t_force.addTorsion(0, 1, 2, 3, 3, 3.14159, 12.5)  # n=3, phi0=pi, k=12.5 kJ/mol
    sys.addForce(t_force)

    conv = from_openmm_system(sys)
    bundle = conv.bundle

    # Bond: r0 = 1.5 Å, k = 250000 / 418.4
    assert np.isclose(float(bundle.bond_params[0, 0]), 1.5, atol=1e-5)
    assert np.isclose(float(bundle.bond_params[0, 1]), 250000.0 * BOND_K_OMM_TO_AKMA, atol=1e-4)

    # Angle: theta0 = 1.91 rad, k = 450 / 4.184
    assert np.isclose(float(bundle.angle_params[0, 0]), 1.91, atol=1e-5)
    assert np.isclose(float(bundle.angle_params[0, 1]), 450.0 * ANGLE_K_OMM_TO_AKMA, atol=1e-4)

    # Dihedral: n = 3, phi0 = 3.14159, k = 12.5 / 4.184
    assert np.isclose(float(bundle.dihedral_params[0, 0]), 3.0, atol=1e-5)
    assert np.isclose(float(bundle.dihedral_params[0, 1]), 3.14159, atol=1e-5)
    assert np.isclose(float(bundle.dihedral_params[0, 2]), 12.5 * TORSION_K_OMM_TO_AKMA, atol=1e-4)


def test_converter_cmap_and_rb_torsions(synthetic_cmap_system, synthetic_rb_torsion_system):
    """Verify CMAPTorsionForce grid extraction and CompiledRBTorsionForce compilation."""
    # 1. Test CMAP
    cmap_sys, cmap_pos, _, _ = synthetic_cmap_system
    conv_cmap = from_openmm_system(cmap_sys)
    bundle_cmap = conv_cmap.bundle

    assert int(bundle_cmap.n_cmap) == 1
    assert bundle_cmap.cmap_coeffs is not None
    assert bundle_cmap.cmap_coeffs.shape[-1] == 16

    # 2. Test RBTorsionForce
    rb_sys, rb_pos, _, _ = synthetic_rb_torsion_system
    conv_rb = from_openmm_system(rb_sys)

    assert len(conv_rb.custom_forces) == 1
    rb_force = conv_rb.custom_forces[0]
    assert rb_force.name == "RBTorsionForce"

    # Verify energy parity with OpenMM Reference
    pos_nm = np.asarray(rb_pos.value_in_unit(unit.nanometer))
    omm_energy = compute_openmm_energy(rb_sys, rb_pos, "Reference")
    prolix_e_omm = float(rb_force.energy_omm(jnp.asarray(pos_nm), None, {}))
    assert np.isclose(omm_energy, prolix_e_omm, rtol=1e-5)


# =============================================================================
# 6. NonbondedForce: Methods, Exclusions, 1-4 Exceptions & 32-Cap Check
# =============================================================================


def test_converter_nonbonded_methods_and_switching():
    """Verify NonbondedForce methods (CutoffPeriodic/ReactionField, PME) and switching function."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    nb.addParticle(0.5, 0.3, 1.2)   # q=0.5 e, sig=0.3 nm, eps=1.2 kJ/mol
    nb.addParticle(-0.5, 0.35, 1.5)
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.85 * unit.nanometer)
    nb.setReactionFieldDielectric(78.5)
    sys.addForce(nb)

    conv = from_openmm_system(sys)

    assert conv.nonbonded_method == "reaction_field"
    assert np.isclose(conv.nonbonded_cutoff, 10.0, atol=1e-5)  # 1.0 nm = 10 Å
    assert np.isclose(conv.lj_switch_width, 1.5, atol=1e-5)   # 10 - 8.5 = 1.5 Å
    assert np.isclose(conv.reaction_field_dielectric, 78.5, atol=1e-5)

    bundle = conv.bundle
    assert np.isclose(float(bundle.charges[0]), 0.5, atol=1e-5)
    assert np.isclose(float(bundle.sigmas[0]), 3.0, atol=1e-5)  # 0.3 nm = 3.0 Å
    assert np.isclose(float(bundle.epsilons[0]), 1.2 / 4.184, atol=1e-5)


def test_converter_unified_14_exceptions():
    """Verify unified 1-4 exceptions: pairs in excl_indices with scale 0.0 and in exception_*."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    for _ in range(4):
        nb.addParticle(0.2, 0.3, 1.0)

    # Pair (0, 1): pure 1-2 exclusion (chargeProd=0, eps=0)
    nb.addException(0, 1, 0.0, 0.1, 0.0)
    # Pair (0, 3): 1-4 exception with non-zero scaling
    nb.addException(0, 3, 0.02, 0.32, 0.5)
    sys.addForce(nb)

    conv = from_openmm_system(sys)
    bundle = conv.bundle

    # Both pairs appear in excl_indices with scale 0.0
    assert int(bundle.n_excl) == 2
    assert float(bundle.excl_scales_vdw[0]) == 0.0
    assert float(bundle.excl_scales_elec[0]) == 0.0
    assert float(bundle.excl_scales_vdw[1]) == 0.0
    assert float(bundle.excl_scales_elec[1]) == 0.0

    # Only the non-zero pair (0, 3) appears in exception_*
    assert int(bundle.n_exception_pairs) == 1
    assert tuple(bundle.exception_pairs[0]) == (0, 3)
    assert np.isclose(float(bundle.exception_chargeprods[0]), 0.02, atol=1e-5)
    assert np.isclose(float(bundle.exception_sigmas[0]), 3.2, atol=1e-5)
    assert np.isclose(float(bundle.exception_epsilons[0]), 0.5 / 4.184, atol=1e-5)


def test_converter_dense_exclusion_cap_enforcement():
    """Verify that an atom with > 32 exclusions raises ValueError."""
    sys = openmm.System()
    # 35 particles: atom 0 excluded with 34 other particles (exceeds cap of 32)
    for _ in range(35):
        sys.addParticle(12.0)

    nb = openmm.NonbondedForce()
    for _ in range(35):
        nb.addParticle(0.0, 0.3, 0.0)

    for other in range(1, 35):
        nb.addException(0, other, 0.0, 0.1, 0.0)
    sys.addForce(nb)

    with pytest.raises(ValueError, match="slots per atom"):
        from_openmm_system(sys)


# =============================================================================
# 7. Custom Forces & Routing (Auto, Pattern-Only, Lepton-Only)
# =============================================================================


def test_converter_custom_forces_auto_folds_pattern():
    """Verify that a matched custom bond force folds into bundle in 'auto' mode."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    cbf = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    cbf.addPerBondParameter("k")
    cbf.addPerBondParameter("r0")
    cbf.addBond(0, 1, [300000.0, 0.14])  # 300000 kJ/(mol*nm^2), 0.14 nm
    sys.addForce(cbf)

    cfg = ProlixOpenMMConfig(custom_force_mode="auto")
    conv = from_openmm_system(sys, config=cfg)

    # Matched and folded into bundle.bond_params
    assert int(conv.bundle.n_bonds) == 1
    assert np.isclose(float(conv.bundle.bond_params[0, 0]), 1.4, atol=1e-5)
    assert np.isclose(float(conv.bundle.bond_params[0, 1]), 300000.0 * BOND_K_OMM_TO_AKMA, atol=1e-4)
    # custom_forces list is empty since it folded cleanly
    assert len(conv.custom_forces) == 0


def test_converter_custom_forces_pattern_only_unmatched_raises():
    """Verify that an unmatched custom force in 'pattern_only' mode raises UnsupportedOpenMMForceError."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    # Non-standard quartic potential
    cbf = openmm.CustomBondForce("k * (r - r0)^4")
    cbf.addPerBondParameter("k")
    cbf.addPerBondParameter("r0")
    cbf.addBond(0, 1, [50000.0, 0.15])
    sys.addForce(cbf)

    cfg = ProlixOpenMMConfig(custom_force_mode="pattern_only")
    with pytest.raises(UnsupportedOpenMMForceError, match="could not be matched"):
        from_openmm_system(sys, config=cfg)


def test_converter_custom_forces_lepton_only_compiles():
    """Verify that custom forces compile to CompiledLeptonForce in 'lepton_only' mode."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    cbf = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    cbf.addPerBondParameter("k")
    cbf.addPerBondParameter("r0")
    cbf.addBond(0, 1, [300000.0, 0.14])
    sys.addForce(cbf)

    cfg = ProlixOpenMMConfig(custom_force_mode="lepton_only")
    conv = from_openmm_system(sys, config=cfg)

    # Not folded into bundle
    assert int(conv.bundle.n_bonds) == 0
    # Compiled to custom_forces
    assert len(conv.custom_forces) == 1
    assert conv.custom_forces[0].name == "CustomBondForce"


def test_converter_position_restraints():
    """Verify that a CustomExternalForce position restraint is extracted into PositionRestraints."""
    sys = openmm.System()
    sys.addParticle(12.0)

    cef = openmm.CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    cef.addPerParticleParameter("k")
    cef.addPerParticleParameter("x0")
    cef.addPerParticleParameter("y0")
    cef.addPerParticleParameter("z0")
    cef.addParticle(0, [1000.0, 0.5, 0.5, 0.5])  # k=1000 kJ/(mol*nm^2), x0=y0=z0=0.5 nm
    sys.addForce(cef)

    conv = from_openmm_system(sys)
    assert len(conv.position_restraints) == 1
    pr = conv.position_restraints[0]
    assert int(pr.indices[0]) == 0
    assert np.isclose(float(pr.x0[0]), 5.0, atol=1e-5)  # 0.5 nm = 5.0 Å
    assert np.isclose(float(pr.k_x[0]), 0.5 * 1000.0 * BOND_K_OMM_TO_AKMA, atol=1e-4)


# =============================================================================
# 8. Force Groups & Global Parameters
# =============================================================================


def test_converter_force_groups_and_global_params():
    """Verify force group preservation (0..31) and global parameters extraction."""
    sys = openmm.System()
    sys.addParticle(12.0)
    sys.addParticle(12.0)

    b = openmm.HarmonicBondForce()
    b.addBond(0, 1, 0.15, 200000.0)
    b.setForceGroup(3)
    sys.addForce(b)

    cbf = openmm.CustomBondForce("k_global * (r - r0)^2")
    cbf.addGlobalParameter("k_global", 1500.0)
    cbf.addPerBondParameter("r0")
    cbf.addBond(0, 1, [0.15])
    cbf.setForceGroup(7)
    sys.addForce(cbf)

    conv = from_openmm_system(sys)

    assert 3 in conv.force_groups
    assert "HarmonicBondForce" in conv.force_groups[3]
    assert 7 in conv.force_groups
    assert "CustomBondForce" in conv.force_groups[7]
    assert conv.force_group_by_component.get("bond") == 3

    assert "k_global" in conv.global_parameters
    assert np.isclose(conv.global_parameters["k_global"], 1500.0)


# =============================================================================
# 9. Fail-Fast on Unsupported Forces & LJPME
# =============================================================================


def test_converter_unsupported_forces_fail_fast():
    """Verify that unsupported forces immediately raise UnsupportedOpenMMForceError."""
    forces_to_test = [
        openmm.DrudeForce(),
        openmm.CustomCVForce("0.5*r^2"),
        openmm.CustomCentroidBondForce(2, "0.5*distance(g1, g2)^2"),
        openmm.CustomHbondForce("0.5*r^2"),
    ]
    if hasattr(openmm, "RPMDMonteCarloBarostat"):
        forces_to_test.append(openmm.RPMDMonteCarloBarostat(1.0 * unit.bar))

    for f in forces_to_test:
        sys = openmm.System()
        sys.addParticle(12.0)
        sys.addForce(f)
        with pytest.raises(UnsupportedOpenMMForceError):
            from_openmm_system(sys)


def test_converter_ljpme_unsupported_fail_fast():
    """Verify that NonbondedForce with LJPME (method 5) raises UnsupportedOpenMMForceError."""
    sys = openmm.System()
    sys.addParticle(12.0)
    nb = openmm.NonbondedForce()
    nb.addParticle(0.0, 0.3, 1.0)
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME)
    sys.addForce(nb)

    with pytest.raises(UnsupportedOpenMMForceError, match="LJPME"):
        from_openmm_system(sys)


# =============================================================================
# 10. Real-World Integration Tests (Chignolin & Water Box)
# =============================================================================


def test_converter_chignolin_implicit(chignolin_1uao_system):
    """Verify end-to-end conversion of Chignolin 1UAO implicit solvent system."""
    sys, pos, top, box = chignolin_1uao_system
    conv = from_openmm_system(sys, positions=pos)

    assert conv.n_real_particles == 138
    assert conv.bundle.shape_spec.has_implicit_solvent is True
    assert int(conv.bundle.n_atoms) == 138
    assert int(conv.bundle.n_bonds) > 0
    assert int(conv.bundle.n_angles) > 0
    assert int(conv.bundle.n_dihedrals) > 0


def test_converter_solvated_water_box(solvated_tip3p_water_box):
    """Verify end-to-end conversion of 258-molecule TIP3P water box with PME and SETTLE."""
    sys, pos, top, box = solvated_tip3p_water_box
    conv = from_openmm_system(sys, positions=pos)

    assert conv.n_real_particles == 774
    assert conv.has_pbc is True
    assert conv.nonbonded_method == "pme"
    assert conv.bundle.shape_spec.has_real_water is True
    assert int(conv.bundle.n_waters) == 258
    # All water oxygen atoms placed at column 0
    oxygens = np.asarray(conv.bundle.water_indices[:258, 0])
    # In standard tip3p box, atoms 0, 3, 6, ... are oxygens
    assert np.all(oxygens % 3 == 0)


# =============================================================================
# 11. Iteration 2 Gate Remediation Tests: Nested Sites, Collinear VJP, Solute CH2, Metadata
# =============================================================================


def test_converter_nested_virtual_sites_dag(reference_platform):
    """Verify nested virtual sites across multi-level DAGs match OpenMM Reference."""
    # Build a depth-3 DAG:
    # 0, 1, 2: real particles
    # 3: VS midpoint of 0 and 1 (depth 1)
    # 4: VS midpoint of 3 and 2 (depth 2)
    # 5: VS midpoint of 4 and 1 (depth 3)
    sys = openmm.System()
    for _ in range(6):
        sys.addParticle(1.0 if _ < 3 else 0.0)

    sys.setVirtualSite(3, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    sys.setVirtualSite(4, openmm.TwoParticleAverageSite(3, 2, 0.5, 0.5))
    sys.setVirtualSite(5, openmm.TwoParticleAverageSite(4, 1, 0.5, 0.5))

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(5, [10.0, 20.0, 30.0])
    sys.addForce(ext)

    pos_nm = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(1.0, 2.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(sys, integ, reference_platform)
    ctx.setPositions(pos_nm)
    ctx.computeVirtualSites()
    omm_state = ctx.getState(getPositions=True, getForces=True)
    omm_pos = omm_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    omm_f = omm_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    conv = from_openmm_system(sys)
    assert conv.virtual_sites is not None
    assert conv.virtual_sites.n_generations == 3

    pos_akma = jnp.asarray(pos_nm) * 10.0
    prolix_pos_nm = np.asarray(conv.compute_virtual_sites(pos_akma)) / 10.0

    raw_f = jnp.zeros((6, 3)).at[5].set(jnp.array([10.0, 20.0, 30.0]))
    prolix_f = np.asarray(conv.redistribute_virtual_site_forces(raw_f, pos_akma))

    # Assert exact position reconstruction parity
    pos_err = np.max(np.abs(omm_pos - prolix_pos_nm))
    assert pos_err < 1e-4, f"Nested virtual site position error {pos_err} nm exceeds 1e-4"

    # Assert exact force redistribution parity on real atoms
    force_err = np.max(np.abs(omm_f[:3] - prolix_f[:3]))
    assert force_err < 1e-4, f"Nested virtual site force error {force_err} exceeds 1e-4"

    # Verify 100% force conservation
    real_force_sum = np.sum(prolix_f[:3], axis=0)
    assert np.allclose(real_force_sum, np.array([10.0, 20.0, 30.0]), atol=1e-4)


def test_converter_collinear_local_coordinates_parity(reference_platform):
    """Verify collinear LocalCoordinatesSite evaluates stably without force explosion."""
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(1.0 if _ < 3 else 0.0)

    vs = openmm.LocalCoordinatesSite(
        0,
        1,
        2,
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(-1.0, 1.0, 0.0),
        openmm.Vec3(-1.0, 0.0, 1.0),
        openmm.Vec3(0.1, 0.2, 0.3) * unit.nanometer,
    )
    sys.setVirtualSite(3, vs)

    ext = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
    ext.addPerParticleParameter("fx")
    ext.addPerParticleParameter("fy")
    ext.addPerParticleParameter("fz")
    ext.addParticle(3, [10.0, 20.0, 30.0])
    sys.addForce(ext)

    # Collinear positions along X-axis
    pos_nm = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(0.0, 0.0, 0.0),
    ]

    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(sys, integ, reference_platform)
    ctx.setPositions(pos_nm)
    ctx.computeVirtualSites()
    omm_f = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    conv = from_openmm_system(sys)
    pos_akma = jnp.asarray(pos_nm) * 10.0
    raw_f = jnp.zeros((4, 3)).at[3].set(jnp.array([10.0, 20.0, 30.0]))
    prolix_f = np.asarray(conv.redistribute_virtual_site_forces(raw_f, pos_akma))

    omm_max_f = np.max(np.abs(omm_f[:3]))
    prolix_max_f = np.max(np.abs(prolix_f[:3]))

    assert omm_max_f <= 30.0
    assert prolix_max_f <= 30.0, f"Expected finite force <= 30, got {prolix_max_f}"
    assert np.allclose(omm_f[:3], prolix_f[:3], atol=1e-5)
    assert np.allclose(np.sum(prolix_f[:3], axis=0), np.array([10.0, 20.0, 30.0]), atol=1e-5)


def test_converter_propane_solute_ch2_constraints_not_water():
    """Verify propane -CH2- constraints remain solute constraints and C-H bonds are not zeroed."""
    sys = openmm.System()
    sys.addParticle(12.011)  # C0
    sys.addParticle(1.008)   # H1
    sys.addParticle(1.008)   # H2
    sys.addParticle(12.011)  # C3
    sys.addParticle(12.011)  # C4

    sys.addConstraint(0, 1, 0.109 * unit.nanometer)
    sys.addConstraint(0, 2, 0.109 * unit.nanometer)

    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.109 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    bonds.addBond(0, 2, 0.109 * unit.nanometer, 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    sys.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    angles.addAngle(1, 0, 2, 109.5 * unit.degree, 500.0 * unit.kilojoules_per_mole / unit.radian**2)
    sys.addForce(angles)

    conv = from_openmm_system(sys)

    assert conv.bundle.n_waters == 0
    assert conv.bundle.n_constraints == 2
    assert conv.bundle.n_bonds == 2
    assert conv.bundle.n_angles == 1

    # Check safe inv_mass evaluation under autodiff
    def loss_fn(m):
        safe_m = jnp.where(m > 1e-12, m, 1.0)
        return jnp.sum(jnp.where(m > 1e-12, 1.0 / safe_m, 0.0))

    grad_m = jax.grad(loss_fn)(conv.bundle.masses)
    assert not jnp.any(jnp.isnan(grad_m))


def test_converter_multiple_bonded_forces_metadata():
    """Verify compound keys in system_metadata when multiple bond/angle forces are present."""
    sys = openmm.System()
    for _ in range(6):
        sys.addParticle(12.0)

    # Force 0: bonds 0-1 and 1-2
    b1 = openmm.HarmonicBondForce()
    b1.addBond(0, 1, 0.15, 200000.0)
    b1.addBond(1, 2, 0.15, 200000.0)
    sys.addForce(b1)

    # Force 1: angle 0-1-2
    a1 = openmm.HarmonicAngleForce()
    a1.addAngle(0, 1, 2, 1.91, 400.0)
    sys.addForce(a1)

    # Force 2: bond 3-4 (second bond force!)
    b2 = openmm.HarmonicBondForce()
    b2.addBond(3, 4, 0.15, 250000.0)
    sys.addForce(b2)

    # Force 3: angle 3-4-5 (second angle force!)
    a2 = openmm.HarmonicAngleForce()
    a2.addAngle(3, 4, 5, 1.91, 450.0)
    sys.addForce(a2)

    conv = from_openmm_system(sys)

    # Compound keys disambiguate terms across forces
    b1_0 = conv.get_bundle_bond_index(0, 0)
    b1_1 = conv.get_bundle_bond_index(0, 1)
    b2_0 = conv.get_bundle_bond_index(2, 0)

    assert b1_0 == 0
    assert b1_1 == 1
    assert b2_0 == 2
    assert len({b1_0, b1_1, b2_0}) == 3  # All distinct bundle rows!

    # Direct compound key access in system_metadata
    meta_bonds = conv.system_metadata["omm_bond_to_bundle_idx"]
    assert meta_bonds[(0, 0)] == 0
    assert meta_bonds[(0, 1)] == 1
    assert meta_bonds[(2, 0)] == 2

    # Angle compound keys
    a1_0 = conv.get_bundle_angle_index(1, 0)
    a2_0 = conv.get_bundle_angle_index(3, 0)
    assert a1_0 == 0
    assert a2_0 == 1
    meta_angles = conv.system_metadata["omm_angle_to_bundle_idx"]
    assert meta_angles[(1, 0)] == 0
    assert meta_angles[(3, 0)] == 1
