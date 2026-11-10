"""Challenger 2 Empirical Stress Test Suite for Pattern Matcher & Folding Engine.

Tests mandated by Challenger 2 Mandate:
1. Full adversarial suite verification (run separately via pytest).
2. Stress test _match_position_restraint:
   - Non-matching expressions: CustomExternalForce('a * x^4 + exp(y)') must return None.
   - 1D/2D restraints: k * (z - z0)^2 and k * ((x - x0)^2 + (y - y0)^2) must return None.
   - Factored vs distributed forms with custom parameter names:
     verify target coordinates and k are extracted exactly, and
     k_prolix = mult * (k_omm / 418.4) evaluates exact energy against OpenMM Reference.
3. Stress test _match_periodic_torsion:
   - Verify k * (1 - cos(theta)) returns None (rejection of inverted sign).
   - Verify cos(theta) + exp(theta) returns None (rejection of extra terms).
   - Verify 2.5 * k * (1 + cos(3 * theta - 0.5)) matches with exact 2.5 * k energy scaling.
4. Stress test _match_compound_bond_3p:
   - Verify that adding trailing terms like + distance(p1, p2)^4 returns None.
5. Stress test companion array padding in fold_matched_pattern_into_bundle:
   - Fold 50 Urey-Bradley terms into a bundle with 10 angles, expanding angle_bucket_idx from 0 (32) to 1 (128).
     Assert that BOTH angle_idx.shape[0] == 128 and urey_bradley_idx.shape[0] == 128.
   - Fold 50 improper terms into a bundle with 10 dihedrals, expanding dihedral_bucket_idx from 0 (32) to 1 (128).
     Assert that BOTH dihedral_idx.shape[0] == 128 and improper_idx.shape[0] == 128.
"""

from __future__ import annotations

import dataclasses
import math
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as unit
import pytest

from tests.openmm.conftest import compute_openmm_energy
from tests.openmm.test_adversarial_pattern_matcher import _create_flexible_test_bundle
from prolix.openmm.pattern_matcher import (
    MatchedPattern,
    fold_matched_pattern_into_bundle,
    match_custom_force_pattern,
)
from prolix.openmm.units import (
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
)
from prolix.types.bundles import (
    ANGLE_BUCKETS,
    DIHEDRAL_BUCKETS,
    MolecularBundle,
)


# =============================================================================
# 1. Stress Test: _match_position_restraint
# =============================================================================


def test_position_restraint_rejection_non_matching():
    """Verify non-matching expressions return None."""
    force = openmm.CustomExternalForce("a * x^4 + exp(y)")
    force.addPerParticleParameter("a")
    force.addParticle(0, [2.5])
    matched = match_custom_force_pattern(force)
    assert matched is None, "CustomExternalForce('a * x^4 + exp(y)') should return None"


def test_position_restraint_rejection_1d_and_2d():
    """Verify 1D and 2D harmonic restraints return None."""
    # 1D restraint on z
    f_1d = openmm.CustomExternalForce("k * (z - z0)^2")
    f_1d.addPerParticleParameter("k")
    f_1d.addPerParticleParameter("z0")
    f_1d.addParticle(0, [100.0, 1.5])
    assert match_custom_force_pattern(f_1d) is None, "1D restraint k*(z-z0)^2 must return None"

    # 1D restraint on x
    f_1dx = openmm.CustomExternalForce("0.5 * k * (x - x0)^2")
    f_1dx.addPerParticleParameter("k")
    f_1dx.addPerParticleParameter("x0")
    f_1dx.addParticle(0, [100.0, 1.5])
    assert match_custom_force_pattern(f_1dx) is None, "1D restraint 0.5*k*(x-x0)^2 must return None"

    # 2D restraint on x, y
    f_2d = openmm.CustomExternalForce("k * ((x - x0)^2 + (y - y0)^2)")
    f_2d.addPerParticleParameter("k")
    f_2d.addPerParticleParameter("x0")
    f_2d.addPerParticleParameter("y0")
    f_2d.addParticle(0, [100.0, 1.0, 2.0])
    assert match_custom_force_pattern(f_2d) is None, "2D restraint k*((x-x0)^2+(y-y0)^2) must return None"

    # 2D distributed restraint on x, y
    f_2d_dist = openmm.CustomExternalForce("0.5 * k * (x - x0)^2 + 0.5 * k * (y - y0)^2")
    f_2d_dist.addPerParticleParameter("k")
    f_2d_dist.addPerParticleParameter("x0")
    f_2d_dist.addPerParticleParameter("y0")
    f_2d_dist.addParticle(0, [100.0, 1.0, 2.0])
    assert match_custom_force_pattern(f_2d_dist) is None, "2D distributed restraint must return None"


@pytest.mark.parametrize(
    "expr,mult",
    [
        ("0.5 * k_fc * ((x - tx)^2 + (y - ty)^2 + (z - tz)^2)", 0.5),
        ("k_fc * ((x - tx)^2 + (y - ty)^2 + (z - tz)^2)", 1.0),
        ("2.0 * k_fc * ((x - tx)^2 + (y - ty)^2 + (z - tz)^2)", 2.0),
        ("0.5 * k_fc * (x - tx)^2 + 0.5 * k_fc * (y - ty)^2 + 0.5 * k_fc * (z - tz)^2", 0.5),
        ("k_fc * (x - tx)^2 + k_fc * (y - ty)^2 + k_fc * (z - tz)^2", 1.0),
        ("0.5 * k_fc * periodicdistance(x, y, z, tx, ty, tz)^2", 0.5),
        ("k_fc * periodicdistance(x, y, z, tx, ty, tz)^2", 1.0),
    ],
)
def test_position_restraint_factored_and_distributed_parity(expr: str, mult: float):
    """Verify target coordinates, k extraction, and exact energy parity vs OpenMM Reference."""
    force = openmm.CustomExternalForce(expr)
    force.addPerParticleParameter("k_fc")
    force.addPerParticleParameter("tx")
    force.addPerParticleParameter("ty")
    force.addPerParticleParameter("tz")

    k_omm = 500.0  # kJ/(mol * nm^2)
    tx, ty, tz = 1.2, 2.3, 3.4  # nm

    force.addParticle(0, [k_omm, tx, ty, tz])

    matched = match_custom_force_pattern(force)
    assert matched is not None, f"Failed to match position restraint expr: {expr}"
    assert matched.pattern_name == "harmonic_position_restraint"
    assert matched.target_term == "position_restraint"
    assert matched.indices.shape == (1, 1)
    assert matched.indices[0, 0] == 0

    # Verify extracted params: [x0_akma, y0_akma, z0_akma, k_akma]
    p = matched.params[0]
    expected_x0 = tx * NM_TO_ANGSTROM  # 12.0 A
    expected_y0 = ty * NM_TO_ANGSTROM  # 23.0 A
    expected_z0 = tz * NM_TO_ANGSTROM  # 34.0 A
    expected_k = mult * (k_omm / 418.4)  # kcal/(mol * A^2)

    np.testing.assert_allclose(p[0], expected_x0, rtol=1e-5)
    np.testing.assert_allclose(p[1], expected_y0, rtol=1e-5)
    np.testing.assert_allclose(p[2], expected_z0, rtol=1e-5)
    np.testing.assert_allclose(p[3], expected_k, rtol=1e-5)

    # Evaluate exact energy vs OpenMM Reference at test positions
    sys = openmm.System()
    sys.addParticle(1.0)
    sys.addForce(force)

    pos_omm = np.array([[1.5, 2.0, 3.0]], dtype=np.float64)  # nm
    pos_openmm_quantity = pos_omm * unit.nanometer

    e_omm_kj = compute_openmm_energy(sys, pos_openmm_quantity, platform_name="Reference")
    e_omm_kcal = e_omm_kj * KJ_PER_MOL_TO_KCAL_PER_MOL

    # Prolix evaluated energy in AKMA (kcal/mol):
    # E_prolix = k_prolix * ||r - r0||^2
    pos_akma = pos_omm[0] * NM_TO_ANGSTROM
    r0_akma = np.array([expected_x0, expected_y0, expected_z0])
    dr_akma = pos_akma - r0_akma
    dist_sq_akma = np.sum(dr_akma ** 2)
    e_prolix = expected_k * dist_sq_akma

    np.testing.assert_allclose(
        e_prolix,
        e_omm_kcal,
        rtol=1e-5,
        err_msg=f"Energy mismatch for {expr}: prolix={e_prolix}, omm={e_omm_kcal}",
    )


# =============================================================================
# 2. Stress Test: _match_periodic_torsion
# =============================================================================


def test_periodic_torsion_rejection_inverted_sign():
    """Verify k * (1 - cos(theta)) returns None."""
    force = openmm.CustomTorsionForce("k * (1 - cos(theta))")
    force.addPerTorsionParameter("k")
    force.addTorsion(0, 1, 2, 3, [50.0])
    matched = match_custom_force_pattern(force)
    assert matched is None, "k * (1 - cos(theta)) must return None (rejected inverted sign)"


def test_periodic_torsion_rejection_extra_terms():
    """Verify cos(theta) + exp(theta) returns None."""
    force = openmm.CustomTorsionForce("cos(theta) + exp(theta)")
    force.addTorsion(0, 1, 2, 3, [])
    matched = match_custom_force_pattern(force)
    assert matched is None, "cos(theta) + exp(theta) must return None (rejected extra terms)"


@pytest.mark.parametrize(
    "bad_expr",
    [
        "k * (1 + sin(theta))",               # sin instead of cos
        "k * (1 + cos(theta)^2)",             # cos squared
        "k * (1 + cos(theta)) + 5.0",         # extra additive constant
        "k * (2 + cos(theta))",               # non-unity baseline offset
        "k * cos(theta)",                     # bare cosine without (1 + cos) offset
    ],
)
def test_periodic_torsion_adversarial_rejections(bad_expr: str):
    """Verify non-standard periodic torsion variants are rejected."""
    force = openmm.CustomTorsionForce(bad_expr)
    if "k" in bad_expr:
        force.addPerTorsionParameter("k")
        force.addTorsion(0, 1, 2, 3, [10.0])
    else:
        force.addTorsion(0, 1, 2, 3, [])
    matched = match_custom_force_pattern(force)
    assert matched is None, f"Expression '{bad_expr}' should return None"


def test_periodic_torsion_scaled_matching_and_energy():
    """Verify 2.5 * k * (1 + cos(3 * theta - 0.5)) matches with exact 2.5 * k scaling."""
    expr = "2.5 * k * (1 + cos(3 * theta - 0.5))"
    force = openmm.CustomTorsionForce(expr)
    force.addPerTorsionParameter("k")
    k_omm = 12.0  # kJ/mol
    force.addTorsion(0, 1, 2, 3, [k_omm])

    matched = match_custom_force_pattern(force)
    assert matched is not None, f"Failed to match {expr}"
    assert matched.pattern_name == "periodic_torsion"
    assert matched.target_term == "dihedral"
    assert matched.indices.shape == (1, 4)

    # Param format: [periodicity, phase, k_akma, 0.0]
    p = matched.params[0]
    assert math.isclose(p[0], 3.0), f"Expected periodicity 3.0, got {p[0]}"
    assert math.isclose(p[1], 0.5), f"Expected phase 0.5, got {p[1]}"

    # Expected k in AKMA: 2.5 * (k_omm / 4.184)
    expected_k_akma = 2.5 * (k_omm / 4.184)
    np.testing.assert_allclose(p[2], expected_k_akma, rtol=1e-5)

    # Parity check against OpenMM Reference
    sys = openmm.System()
    for _ in range(4):
        sys.addParticle(1.0)
    sys.addForce(force)

    # Choose non-planar geometry
    pos_omm = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    pos_quantity = pos_omm * unit.nanometer

    e_omm_kj = compute_openmm_energy(sys, pos_quantity, platform_name="Reference")
    e_omm_kcal = e_omm_kj * KJ_PER_MOL_TO_KCAL_PER_MOL

    # Dihedral angle calculation per OpenMM convention
    b1 = pos_omm[1] - pos_omm[0]
    b2 = pos_omm[2] - pos_omm[1]
    b3 = pos_omm[3] - pos_omm[2]
    c1 = np.cross(b1, b2)
    c2 = np.cross(b2, b3)
    cos_th = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
    sin_th = np.dot(np.cross(c1, c2), b2 / np.linalg.norm(b2)) / (np.linalg.norm(c1) * np.linalg.norm(c2))
    theta = math.atan2(sin_th, cos_th)

    e_prolix = expected_k_akma * (1.0 + math.cos(3.0 * theta - 0.5))
    np.testing.assert_allclose(
        e_prolix,
        e_omm_kcal,
        rtol=1e-5,
        err_msg=f"Periodic torsion energy mismatch: prolix={e_prolix}, omm={e_omm_kcal}",
    )


# =============================================================================
# 3. Stress Test: _match_compound_bond_3p
# =============================================================================


def test_compound_bond_3p_rejection_trailing_terms():
    """Verify that adding trailing terms like + distance(p1, p2)^4 returns None."""
    # Composite Angle + Urey-Bradley with trailing quartic term
    expr = (
        "0.5 * k_a * (angle(p1, p2, p3) - theta0)^2 + "
        "0.5 * k_ub * (distance(p1, p3) - r_ub0)^2 + "
        "distance(p1, p2)^4"
    )
    force = openmm.CustomCompoundBondForce(3, expr)
    force.addPerBondParameter("k_a")
    force.addPerBondParameter("theta0")
    force.addPerBondParameter("k_ub")
    force.addPerBondParameter("r_ub0")
    force.addBond([0, 1, 2], [100.0, 1.8, 200.0, 0.25])

    matched = match_custom_force_pattern(force)
    assert matched is None, "CustomCompoundBondForce with trailing quartic term must return None"

    # Pure Urey-Bradley with trailing linear term
    expr_ub_bad = "0.5 * k_ub * (distance(p1, p3) - r_ub0)^2 + 0.1 * distance(p1, p2)"
    force_ub_bad = openmm.CustomCompoundBondForce(3, expr_ub_bad)
    force_ub_bad.addPerBondParameter("k_ub")
    force_ub_bad.addPerBondParameter("r_ub0")
    force_ub_bad.addBond([0, 1, 2], [200.0, 0.25])

    matched_ub = match_custom_force_pattern(force_ub_bad)
    assert matched_ub is None, "CustomCompoundBondForce with trailing linear term must return None"

    # 3 harmonic terms (angle + ub + extra 1-2 bond) -> rejected for 3p compound bond
    expr_3terms = (
        "0.5 * k_a * (angle(p1, p2, p3) - theta0)^2 + "
        "0.5 * k_ub * (distance(p1, p3) - r_ub0)^2 + "
        "0.5 * k_b * (distance(p1, p2) - r_b0)^2"
    )
    force_3terms = openmm.CustomCompoundBondForce(3, expr_3terms)
    force_3terms.addPerBondParameter("k_a")
    force_3terms.addPerBondParameter("theta0")
    force_3terms.addPerBondParameter("k_ub")
    force_3terms.addPerBondParameter("r_ub0")
    force_3terms.addPerBondParameter("k_b")
    force_3terms.addPerBondParameter("r_b0")
    force_3terms.addBond([0, 1, 2], [100.0, 1.8, 200.0, 0.25, 300.0, 0.15])
    assert match_custom_force_pattern(force_3terms) is None, "3-term compound bond must return None"


# =============================================================================
# 4. Stress Test: Companion Array Padding in fold_matched_pattern_into_bundle
# =============================================================================


def test_companion_array_padding_angle_and_urey_bradley():
    """Fold 50 Urey-Bradley terms into a bundle with 10 angles, expanding angle_bucket_idx from 0 (32) to 1 (128).

    Assert that BOTH angle_idx.shape[0] == 128 and urey_bradley_idx.shape[0] == 128.
    """
    # Create bundle with 10 angles in 32-capacity bucket
    bundle = _create_flexible_test_bundle(n_atoms=64, atom_cap=64, angle_cap=32, dihedral_cap=32)
    # Set 10 active angles
    bundle = dataclasses.replace(
        bundle,
        n_angles=jnp.array(10, dtype=jnp.int32),
        angle_mask=jnp.arange(32) < 10,
    )
    assert bundle.angle_idx.shape[0] == 32
    assert bundle.urey_bradley_idx.shape[0] == 32
    assert bundle.shape_spec.angle_bucket_idx == 0

    # Construct MatchedPattern with 50 Urey-Bradley terms
    ub_indices = np.array([[i, i + 2, 0] for i in range(50)], dtype=np.int32)
    ub_params = np.array([[2.5, 50.0] for _ in range(50)], dtype=np.float32)
    matched_ub = MatchedPattern(
        pattern_name="urey_bradley",
        target_term="urey_bradley",
        indices=ub_indices,
        params=ub_params,
    )

    new_bundle = fold_matched_pattern_into_bundle(bundle, matched_ub)

    # Ladder check: 50 UB terms exceeds 32 capacity -> bucket 1 (128)
    expected_cap = ANGLE_BUCKETS[1]  # 128
    assert expected_cap == 128

    assert new_bundle.shape_spec.angle_bucket_idx == 1, (
        f"Expected angle_bucket_idx == 1, got {new_bundle.shape_spec.angle_bucket_idx}"
    )

    # CRITICAL: Both angle_idx and urey_bradley_idx MUST be synchronized to 128!
    assert new_bundle.angle_idx.shape[0] == 128, (
        f"angle_idx capacity desynchronized! Expected 128, got {new_bundle.angle_idx.shape[0]}"
    )
    assert new_bundle.urey_bradley_idx.shape[0] == 128, (
        f"urey_bradley_idx capacity desynchronized! Expected 128, got {new_bundle.urey_bradley_idx.shape[0]}"
    )

    # Also check angle_params, angle_mask, urey_bradley_params, urey_bradley_mask
    assert new_bundle.angle_params.shape[0] == 128
    assert new_bundle.angle_mask.shape[0] == 128
    assert new_bundle.urey_bradley_params.shape[0] == 128
    assert new_bundle.urey_bradley_mask.shape[0] == 128

    # Check active counts
    assert int(new_bundle.n_angles) == 10
    assert int(new_bundle.n_urey_bradley) == 50
    assert int(jnp.sum(new_bundle.angle_mask)) == 10
    assert int(jnp.sum(new_bundle.urey_bradley_mask)) == 50


def test_companion_array_padding_dihedral_and_improper():
    """Fold 50 improper terms into a bundle with 10 dihedrals, expanding dihedral_bucket_idx from 0 (32) to 1 (128).

    Assert that BOTH dihedral_idx.shape[0] == 128 and improper_idx.shape[0] == 128.
    """
    # Create bundle with 10 dihedrals in 32-capacity bucket
    bundle = _create_flexible_test_bundle(n_atoms=64, atom_cap=64, angle_cap=32, dihedral_cap=32)
    bundle = dataclasses.replace(
        bundle,
        n_dihedrals=jnp.array(10, dtype=jnp.int32),
        dihedral_mask=jnp.arange(32) < 10,
    )
    assert bundle.dihedral_idx.shape[0] == 32
    assert bundle.improper_idx.shape[0] == 32
    assert bundle.shape_spec.dihedral_bucket_idx == 0

    # Construct MatchedPattern with 50 improper terms
    imp_indices = np.array([[i, i + 1, i + 2, i + 3] for i in range(50)], dtype=np.int32)
    imp_params = np.array([[10.0, 0.0, 0.0] for _ in range(50)], dtype=np.float32)
    matched_imp = MatchedPattern(
        pattern_name="harmonic_improper",
        target_term="improper",
        indices=imp_indices,
        params=imp_params,
    )

    new_bundle = fold_matched_pattern_into_bundle(bundle, matched_imp)

    # Ladder check: 50 improper terms exceeds 32 capacity -> bucket 1 (128)
    expected_cap = DIHEDRAL_BUCKETS[1]  # 128
    assert expected_cap == 128

    assert new_bundle.shape_spec.dihedral_bucket_idx == 1, (
        f"Expected dihedral_bucket_idx == 1, got {new_bundle.shape_spec.dihedral_bucket_idx}"
    )

    # CRITICAL: Both dihedral_idx and improper_idx MUST be synchronized to 128!
    assert new_bundle.dihedral_idx.shape[0] == 128, (
        f"dihedral_idx capacity desynchronized! Expected 128, got {new_bundle.dihedral_idx.shape[0]}"
    )
    assert new_bundle.improper_idx.shape[0] == 128, (
        f"improper_idx capacity desynchronized! Expected 128, got {new_bundle.improper_idx.shape[0]}"
    )

    # Also check params and masks
    assert new_bundle.dihedral_params.shape[0] == 128
    assert new_bundle.dihedral_mask.shape[0] == 128
    assert new_bundle.improper_params.shape[0] == 128
    assert new_bundle.improper_mask.shape[0] == 128

    # Check active counts
    assert int(new_bundle.n_dihedrals) == 10
    assert int(new_bundle.n_impropers) == 50
    assert int(jnp.sum(new_bundle.dihedral_mask)) == 10
    assert int(jnp.sum(new_bundle.improper_mask)) == 50
