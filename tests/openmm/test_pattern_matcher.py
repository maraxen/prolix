"""Comprehensive test suite for OpenMM custom force pattern matcher and folding engine.

Verifies:
1. Canonical AST normalization:
   - Commutative addition and multiplication ordering
   - Constant folding and fraction simplification
   - Symmetrization of coordinate differences (e.g. (r0 - r)^2 -> (r - r0)^2)
   - Inlining of definitions and subexpressions
2. Algebraic matching for all 6 native potentials:
   - Harmonic bonds (CustomBondForce, CustomCompoundBondForce 2-particle)
   - Harmonic angles (CustomAngleForce)
   - Periodic torsions (CustomTorsionForce)
   - CHARMM harmonic impropers (CustomTorsionForce) with unhalved formula scaling
   - Urey-Bradley 1-3 interactions (CustomCompoundBondForce 3-particle)
   - Combined Angle + Urey-Bradley composite interactions
   - Harmonic position restraints (CustomExternalForce)
3. Dynamic parameter fallback rule:
   - Energy parameter derivatives (addEnergyParameterDerivative) decline pattern matching
   - Global parameters (addGlobalParameter) decline pattern matching
   - Tabulated functions decline pattern matching
   - Arbitrary/unmatched non-linear expressions decline pattern matching
4. Parameter folding engine into MolecularBundle:
   - Active count increments (n_bonds, n_angles, etc.)
   - Mask updates and index/param array padding
   - Bucket ladder expansion when capacity is exceeded
   - Composite pattern multi-term folding
5. Numerical parity:
   - Exact energy agreement between OpenMM Context and folded Prolix calculations
   - Parity of improper scaling (k_omm / 8.368) vs OpenMM reference
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as unit
import pytest
from jax_md import space

from tests.openmm.conftest import compute_openmm_energy
from prolix.openmm.lepton_compiler import (
    BinaryOpNode,
    ConstNode,
    VarNode,
    parse_lepton_expression,
)
from prolix.openmm.pattern_matcher import (
    MatchedPattern,
    canonicalize_expression,
    fold_matched_pattern_into_bundle,
    match_custom_force_pattern,
    rewrite_ast,
)
from prolix.openmm.units import (
    ANGLE_K_OMM_TO_AKMA,
    BOND_K_OMM_TO_AKMA,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
    TORSION_K_OMM_TO_AKMA,
)
from prolix.physics.bonded import (
    make_angle_energy_fn,
    make_bond_energy_fn,
    make_dihedral_energy_fn,
    make_harmonic_improper_energy_fn,
)
from prolix.types.bundles import (
    ANGLE_BUCKETS,
    BOND_BUCKETS,
    DIHEDRAL_BUCKETS,
    MolecularBundle,
    MolecularShapeSpec,
)


# =============================================================================
# Helper Fixtures & Utilities
# =============================================================================


def _create_minimal_test_bundle(
    n_atoms: int = 16,
    n_bonds: int = 0,
    n_angles: int = 0,
    n_dihedrals: int = 0,
    n_impropers: int = 0,
    n_urey_bradley: int = 0,
    bond_cap: int = 16,
    angle_cap: int = 32,
    dihedral_cap: int = 32,
) -> MolecularBundle:
    """Create a minimal well-formed MolecularBundle for testing folding operations."""
    a = 64
    b = bond_cap
    ang = angle_cap
    dih = dihedral_cap

    shape_spec = MolecularShapeSpec(
        atom_bucket_idx=0,
        bond_bucket_idx=0,
        angle_bucket_idx=0,
        dihedral_bucket_idx=0,
        water_bucket_idx=0,
        excl_bucket_idx=0,
        cmap_bucket_idx=0,
        exception_bucket_idx=0,
        has_pbc=False,
        has_implicit_solvent=False,
        boundary_condition="free",
    )

    return MolecularBundle(
        positions=jnp.zeros((a, 3)),
        masses=jnp.ones(a),
        charges=jnp.zeros(a),
        sigmas=jnp.ones(a),
        epsilons=jnp.ones(a),
        radii=jnp.ones(a),
        scaled_radii=jnp.ones(a),
        atom_mask=jnp.concatenate(
            [jnp.ones(n_atoms, dtype=bool), jnp.zeros(a - n_atoms, dtype=bool)]
        ),
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=jnp.zeros((3, 3)),
        # Bonds
        bond_idx=jnp.zeros((b, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((b, 2)),
        bond_mask=jnp.concatenate(
            [jnp.ones(n_bonds, dtype=bool), jnp.zeros(b - n_bonds, dtype=bool)]
        ),
        n_bonds=jnp.array(n_bonds, dtype=jnp.int32),
        # Angles
        angle_idx=jnp.zeros((ang, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((ang, 2)),
        angle_mask=jnp.concatenate(
            [jnp.ones(n_angles, dtype=bool), jnp.zeros(ang - n_angles, dtype=bool)]
        ),
        n_angles=jnp.array(n_angles, dtype=jnp.int32),
        # Dihedrals
        dihedral_idx=jnp.zeros((dih, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((dih, 4)),
        dihedral_mask=jnp.concatenate(
            [jnp.ones(n_dihedrals, dtype=bool), jnp.zeros(dih - n_dihedrals, dtype=bool)]
        ),
        n_dihedrals=jnp.array(n_dihedrals, dtype=jnp.int32),
        # Impropers
        improper_idx=jnp.zeros((dih, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((dih, 3)),
        improper_mask=jnp.concatenate(
            [jnp.ones(n_impropers, dtype=bool), jnp.zeros(dih - n_impropers, dtype=bool)]
        ),
        improper_is_periodic=jnp.array(False, dtype=bool),
        n_impropers=jnp.array(n_impropers, dtype=jnp.int32),
        # Urey-Bradley
        urey_bradley_idx=jnp.zeros((ang, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((ang, 2)),
        urey_bradley_mask=jnp.concatenate(
            [jnp.ones(n_urey_bradley, dtype=bool), jnp.zeros(ang - n_urey_bradley, dtype=bool)]
        ),
        n_urey_bradley=jnp.array(n_urey_bradley, dtype=jnp.int32),
        # Padding for non-bonded and other fields
        cmap_torsion_idx=jnp.zeros((8, 8), dtype=jnp.int32),
        cmap_energy_grids=jnp.zeros((8, 24, 24)),
        cmap_mask=jnp.zeros(8, dtype=bool),
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=jnp.zeros((8, 3), dtype=jnp.int32),
        water_mask=jnp.zeros(8, dtype=bool),
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=jnp.zeros((32, 2), dtype=jnp.int32),
        excl_scales_vdw=jnp.zeros(32),
        excl_scales_elec=jnp.zeros(32),
        excl_mask=jnp.zeros(32, dtype=bool),
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=jnp.zeros((32, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros(32),
        exception_epsilons=jnp.zeros(32),
        exception_chargeprods=jnp.zeros(32),
        exception_mask=jnp.zeros(32, dtype=bool),
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=jnp.array(0.0),
        cutoff_distance=jnp.array(10.0),
        shape_spec=shape_spec,
    )


# =============================================================================
# 1. AST Canonical Normalization Tests
# =============================================================================


def test_ast_canonicalize_commutative_add_and_mul():
    """Verify that commutative permutations of addition and multiplication normalize identically."""
    expr1 = "0.5 * k * (r - r0)^2"
    expr2 = "(r - r0)^2 * k * 0.5"
    expr3 = "k * 0.5 * (r - r0)^2"

    ast1, defs1 = parse_lepton_expression(expr1)
    ast2, defs2 = parse_lepton_expression(expr2)
    ast3, defs3 = parse_lepton_expression(expr3)

    norm1 = canonicalize_expression(ast1, defs1)
    norm2 = canonicalize_expression(ast2, defs2)
    norm3 = canonicalize_expression(ast3, defs3)

    assert repr(norm1) == repr(norm2)
    assert repr(norm1) == repr(norm3)


def test_ast_canonicalize_division_to_multiplication():
    """Verify that division by constants is folded to multiplication by reciprocal."""
    expr1 = "k * (r - r0)^2 / 2"
    expr2 = "0.5 * k * (r - r0)^2"

    ast1, defs1 = parse_lepton_expression(expr1)
    ast2, defs2 = parse_lepton_expression(expr2)

    norm1 = canonicalize_expression(ast1, defs1)
    norm2 = canonicalize_expression(ast2, defs2)

    assert repr(norm1) == repr(norm2)


def test_ast_canonicalize_sq_function():
    """Verify that sq(x) is rewritten to x^2 and normalized."""
    expr1 = "0.5 * k * sq(r - r0)"
    expr2 = "0.5 * k * (r - r0)^2"

    ast1, defs1 = parse_lepton_expression(expr1)
    ast2, defs2 = parse_lepton_expression(expr2)

    norm1 = canonicalize_expression(ast1, defs1)
    norm2 = canonicalize_expression(ast2, defs2)

    assert repr(norm1) == repr(norm2)


def test_ast_canonicalize_difference_symmetrization():
    """Verify that (r0 - r)^2 is rewritten to (r - r0)^2 for coordinate variables."""
    expr1 = "0.5 * k * (r0 - r)^2"
    expr2 = "0.5 * k * (r - r0)^2"

    ast1, defs1 = parse_lepton_expression(expr1)
    ast2, defs2 = parse_lepton_expression(expr2)

    norm1 = canonicalize_expression(ast1, defs1)
    norm2 = canonicalize_expression(ast2, defs2)

    assert repr(norm1) == repr(norm2)


def test_ast_canonicalize_definitions_inlining():
    """Verify that semicolon-delimited definitions are fully inlined and folded."""
    expr = "energy; energy = k * diff^2 / 2; diff = r - r0"
    primary_ast, defs = parse_lepton_expression(expr)
    norm = canonicalize_expression(primary_ast, defs)

    ref_ast, ref_defs = parse_lepton_expression("0.5 * k * (r - r0)^2")
    norm_ref = canonicalize_expression(ref_ast, ref_defs)

    assert repr(norm) == repr(norm_ref)


# =============================================================================
# 2. Algebraic Pattern Matching for Native Potentials
# =============================================================================


def test_match_harmonic_bond_force():
    """Verify pattern matching and parameter conversion for CustomBondForce."""
    force = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    force.addBond(0, 1, [300000.0, 0.15])  # k in kJ/(mol*nm^2), r0 in nm
    force.addBond(2, 3, [250000.0, 0.12])

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "harmonic_bond"
    assert matched.target_term == "bond"
    assert matched.indices.shape == (2, 2)
    assert matched.params.shape == (2, 2)
    np.testing.assert_array_equal(matched.indices, [[0, 1], [2, 3]])

    # Verify AKMA conversion:
    # r0 in Angstrom = nm * 10
    # k in kcal/(mol*A^2) = k_omm / 418.4
    expected_r0_0 = 0.15 * NM_TO_ANGSTROM
    expected_k_0 = 300000.0 * BOND_K_OMM_TO_AKMA
    np.testing.assert_allclose(matched.params[0, 0], expected_r0_0, rtol=1e-5)
    np.testing.assert_allclose(matched.params[0, 1], expected_k_0, rtol=1e-5)


def test_match_harmonic_angle_force():
    """Verify pattern matching and parameter conversion for CustomAngleForce."""
    force = openmm.CustomAngleForce("0.5 * k * (theta - theta0)^2")
    force.addPerAngleParameter("k")
    force.addPerAngleParameter("theta0")
    force.addAngle(0, 1, 2, [400.0, 1.91])  # k in kJ/(mol*rad^2), theta0 in rad

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "harmonic_angle"
    assert matched.target_term == "angle"
    assert matched.indices.shape == (1, 3)
    assert matched.params.shape == (1, 2)

    # theta0 in rad, k in kcal/(mol*rad^2) = k_omm / 4.184
    np.testing.assert_allclose(matched.params[0, 0], 1.91, rtol=1e-5)
    np.testing.assert_allclose(matched.params[0, 1], 400.0 * ANGLE_K_OMM_TO_AKMA, rtol=1e-5)


def test_match_charmm_harmonic_improper_unhalved_convention():
    """Verify that CHARMM harmonic impropers correctly scale for Prolix's unhalved formula."""
    # OpenMM standard formula with 0.5: E = 0.5 * k * (theta - theta0)^2
    force = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    force.addPerTorsionParameter("k")
    force.addPerTorsionParameter("theta0")
    k_omm = 209.2  # kJ/(mol*rad^2)
    theta0_val = 0.0
    force.addTorsion(0, 1, 2, 3, [k_omm, theta0_val])

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "harmonic_improper"
    assert matched.target_term == "improper"

    # CRITICAL DISCREPANCY:
    # Prolix formula: E = sum k_prolix * (theta - theta0)^2
    # OpenMM formula: E = 0.5 * k_omm * (theta - theta0)^2
    # So k_prolix = 0.5 * k_omm / 4.184 = k_omm / 8.368
    expected_k_prolix = 0.5 * (k_omm / 4.184)
    assert math.isclose(expected_k_prolix, k_omm / 8.368, rel_tol=1e-5)
    np.testing.assert_allclose(matched.params[0, 0], expected_k_prolix, rtol=1e-5)
    np.testing.assert_allclose(matched.params[0, 1], theta0_val, rtol=1e-5)


def test_match_periodic_torsion_force():
    """Verify pattern matching for periodic dihedrals in CustomTorsionForce."""
    force = openmm.CustomTorsionForce("k * (1 + cos(n * theta - phase))")
    force.addPerTorsionParameter("k")
    force.addPerTorsionParameter("n")
    force.addPerTorsionParameter("phase")
    force.addTorsion(0, 1, 2, 3, [15.0, 3.0, 3.14159265])

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "periodic_torsion"
    assert matched.target_term == "dihedral"

    # params: [periodicity, phase, k_akma, 0.0]
    expected_k_akma = 15.0 * TORSION_K_OMM_TO_AKMA
    np.testing.assert_allclose(matched.params[0, 0], 3.0, rtol=1e-5)
    np.testing.assert_allclose(matched.params[0, 1], 3.14159265, rtol=1e-5)
    np.testing.assert_allclose(matched.params[0, 2], expected_k_akma, rtol=1e-5)


def test_match_urey_bradley_compound_bond_force():
    """Verify 3-particle Urey-Bradley interaction in CustomCompoundBondForce."""
    force = openmm.CustomCompoundBondForce(3, "0.5 * k * (distance(p1, p3) - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    force.addBond([1, 2, 5], [100000.0, 0.25])

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "urey_bradley"
    assert matched.target_term == "urey_bradley"
    np.testing.assert_array_equal(matched.indices, [[1, 5, 0]])
    np.testing.assert_allclose(matched.params[0, 0], 0.25 * NM_TO_ANGSTROM, rtol=1e-5)
    np.testing.assert_allclose(matched.params[0, 1], 100000.0 * BOND_K_OMM_TO_AKMA, rtol=1e-5)


def test_match_combined_angle_urey_bradley_force():
    """Verify composite pattern matching for simultaneous Angle + Urey-Bradley."""
    expr = (
        "0.5 * k_ang * (angle(p1, p2, p3) - theta0)^2 + "
        "0.5 * k_ub * (distance(p1, p3) - r0)^2"
    )
    force = openmm.CustomCompoundBondForce(3, expr)
    force.addPerBondParameter("k_ang")
    force.addPerBondParameter("theta0")
    force.addPerBondParameter("k_ub")
    force.addPerBondParameter("r0")
    force.addBond([0, 1, 2], [500.0, 1.91, 120000.0, 0.24])

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "combined_angle_urey_bradley"
    assert matched.target_term == "composite"
    assert len(matched.sub_patterns) == 2

    ang_sub = next(s for s in matched.sub_patterns if s.target_term == "angle")
    ub_sub = next(s for s in matched.sub_patterns if s.target_term == "urey_bradley")

    np.testing.assert_array_equal(ang_sub.indices, [[0, 1, 2]])
    np.testing.assert_array_equal(ub_sub.indices, [[0, 2, 0]])


def test_match_position_restraint_force():
    """Verify position restraint matching in CustomExternalForce."""
    force = openmm.CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addParticle(0, [1000.0, 1.0, 2.0, 3.0])

    matched = match_custom_force_pattern(force)
    assert matched is not None
    assert matched.pattern_name == "harmonic_position_restraint"
    assert matched.target_term == "position_restraint"
    assert matched.indices[0, 0] == 0


# =============================================================================
# 3. Dynamic Parameter Fallback Rule Tests
# =============================================================================


def test_fallback_when_energy_parameter_derivatives_present():
    """Verify that match_custom_force_pattern returns None if energy derivatives exist."""
    force = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    force.addGlobalParameter("k", 300000.0)
    force.addPerBondParameter("r0")
    force.addBond(0, 1, [0.15])
    force.addEnergyParameterDerivative("k")  # Autodiff requested!

    matched = match_custom_force_pattern(force)
    assert matched is None, "Should fallback to Tier B Lepton compiler when derivatives are requested"


def test_fallback_when_global_parameters_present():
    """Verify that match_custom_force_pattern returns None if global parameters exist."""
    force = openmm.CustomBondForce("0.5 * k * (r - r0)^2 * lambda_scale")
    force.addGlobalParameter("lambda_scale", 1.0)
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    force.addBond(0, 1, [300000.0, 0.15])

    matched = match_custom_force_pattern(force)
    assert matched is None, "Should fallback to Lepton compiler when global parameters exist"


def test_fallback_when_tabulated_functions_present():
    """Verify that match_custom_force_pattern returns None if tabulated functions exist."""
    force = openmm.CustomCompoundBondForce(2, "table(distance(p1, p2))")
    tab = openmm.Continuous1DFunction([0.0, 1.0, 4.0], 0.0, 2.0)
    force.addTabulatedFunction("table", tab)
    force.addBond([0, 1], [])

    matched = match_custom_force_pattern(force)
    assert matched is None, "Should fallback to Lepton compiler when tabulated functions exist"


def test_fallback_when_unmatched_expression():
    """Verify that non-linear or arbitrary expressions decline pattern matching."""
    force = openmm.CustomBondForce("a * r^4 - b / r^2")
    force.addPerBondParameter("a")
    force.addPerBondParameter("b")
    force.addBond(0, 1, [1.0, 2.0])

    matched = match_custom_force_pattern(force)
    assert matched is None, "Should decline matching for arbitrary non-harmonic potential"


# =============================================================================
# 4. Parameter Folding Engine into MolecularBundle Tests
# =============================================================================


def test_fold_bonds_into_bundle_within_bucket():
    """Verify folding new bonds into an existing bundle without exceeding bucket capacity."""
    bundle = _create_minimal_test_bundle(n_atoms=16, n_bonds=2, bond_cap=16)

    # Matched pattern with 3 new bonds
    new_indices = np.array([[2, 3], [4, 5], [6, 7]], dtype=np.int32)
    new_params = np.array([[1.5, 300.0], [1.2, 250.0], [1.4, 280.0]], dtype=np.float32)
    matched = MatchedPattern(
        pattern_name="harmonic_bond",
        target_term="bond",
        indices=new_indices,
        params=new_params,
    )

    folded = fold_matched_pattern_into_bundle(bundle, matched)

    assert int(folded.n_bonds) == 5
    assert folded.bond_idx.shape[0] == 16
    assert folded.shape_spec.bond_bucket_idx == 0
    # First 5 are active
    np.testing.assert_array_equal(folded.bond_mask[:5], [True] * 5)
    np.testing.assert_array_equal(folded.bond_mask[5:], [False] * 11)
    np.testing.assert_array_equal(folded.bond_idx[2:5], new_indices)
    np.testing.assert_allclose(folded.bond_params[2:5], new_params)


def test_fold_bonds_into_bundle_with_bucket_expansion():
    """Verify that when total bonds exceed current bucket, bucket expands to next ladder entry."""
    bundle = _create_minimal_test_bundle(n_atoms=32, n_bonds=10, bond_cap=16)

    # 10 existing + 15 new = 25 total bonds -> exceeds 16 (BOND_BUCKETS[0]), should expand to 64 (BOND_BUCKETS[1])
    new_indices = np.array([[i, i + 1] for i in range(15)], dtype=np.int32)
    new_params = np.ones((15, 2), dtype=np.float32)
    matched = MatchedPattern(
        pattern_name="harmonic_bond",
        target_term="bond",
        indices=new_indices,
        params=new_params,
    )

    folded = fold_matched_pattern_into_bundle(bundle, matched)

    assert int(folded.n_bonds) == 25
    assert folded.shape_spec.bond_bucket_idx == 1  # 64
    assert folded.bond_idx.shape[0] == BOND_BUCKETS[1]  # 64
    assert jnp.sum(folded.bond_mask) == 25


def test_fold_composite_angle_urey_bradley_into_bundle():
    """Verify composite pattern folds both angle and Urey-Bradley terms properly."""
    bundle = _create_minimal_test_bundle(n_atoms=16, n_angles=1, n_urey_bradley=0)

    sub1 = MatchedPattern(
        pattern_name="harmonic_angle",
        target_term="angle",
        indices=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        params=np.array([[1.91, 50.0], [2.00, 60.0]], dtype=np.float32),
    )
    sub2 = MatchedPattern(
        pattern_name="urey_bradley",
        target_term="urey_bradley",
        indices=np.array([[0, 2, 0], [1, 3, 0]], dtype=np.int32),
        params=np.array([[2.5, 30.0], [2.6, 35.0]], dtype=np.float32),
    )
    composite = MatchedPattern(
        pattern_name="combined_angle_urey_bradley",
        target_term="composite",
        indices=np.empty((0, 3), dtype=np.int32),
        params=np.empty((0, 2), dtype=np.float32),
        sub_patterns=[sub1, sub2],
    )

    folded = fold_matched_pattern_into_bundle(bundle, composite)

    assert int(folded.n_angles) == 3  # 1 old + 2 new
    assert int(folded.n_urey_bradley) == 2  # 0 old + 2 new
    np.testing.assert_array_equal(folded.angle_mask[:3], [True] * 3)
    np.testing.assert_array_equal(folded.urey_bradley_mask[:2], [True] * 2)


# =============================================================================
# 5. Numerical Parity Verification Tests vs OpenMM Reference
# =============================================================================


def test_harmonic_bond_numerical_parity_vs_openmm():
    """Verify exact energy parity between OpenMM CustomBondForce and folded Prolix bond energy."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)

    # OpenMM CustomBondForce
    force = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    force.addBond(0, 1, [250000.0, 0.15])  # k in kJ/(mol*nm^2), r0 in nm
    force.addBond(2, 3, [320000.0, 0.12])
    system.addForce(force)

    # Test positions (in nm)
    positions_nm = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.16, 0.00, 0.00],  # r = 0.16 nm, r0 = 0.15 nm
            [0.50, 0.50, 0.50],
            [0.61, 0.50, 0.50],  # r = 0.11 nm, r0 = 0.12 nm
        ],
        dtype=np.float64,
    )
    pos_quantity = positions_nm * unit.nanometers
    omm_energy_kj = compute_openmm_energy(system, pos_quantity, platform_name="Reference")
    omm_energy_akma = omm_energy_kj * KJ_PER_MOL_TO_KCAL_PER_MOL

    # Match pattern and fold into bundle
    matched = match_custom_force_pattern(force)
    assert matched is not None

    bundle = _create_minimal_test_bundle(n_atoms=4, n_bonds=0, bond_cap=16)
    folded = fold_matched_pattern_into_bundle(bundle, matched)

    # Evaluate using Prolix bond energy function
    displacement_fn, _ = space.free()
    bond_energy_fn = make_bond_energy_fn(displacement_fn, folded.bond_idx[: int(folded.n_bonds)])
    pos_angstrom = jnp.asarray(positions_nm * NM_TO_ANGSTROM, dtype=jnp.float32)
    prolix_energy_akma = float(
        bond_energy_fn(pos_angstrom, folded.bond_params[: int(folded.n_bonds)])
    )

    np.testing.assert_allclose(prolix_energy_akma, omm_energy_akma, rtol=1e-5)


def test_harmonic_angle_numerical_parity_vs_openmm():
    """Verify exact energy parity between OpenMM CustomAngleForce and folded Prolix angle energy."""
    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.0)

    force = openmm.CustomAngleForce("0.5 * k * (theta - theta0)^2")
    force.addPerAngleParameter("k")
    force.addPerAngleParameter("theta0")
    # k in kJ/(mol*rad^2), theta0 in rad
    force.addAngle(0, 1, 2, [500.0, 1.80])
    system.addForce(force)

    # 3 atoms forming angle ~ 1.57 rad (90 deg)
    positions_nm = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    pos_quantity = positions_nm * unit.nanometers
    omm_energy_kj = compute_openmm_energy(system, pos_quantity, platform_name="Reference")
    omm_energy_akma = omm_energy_kj * KJ_PER_MOL_TO_KCAL_PER_MOL

    matched = match_custom_force_pattern(force)
    assert matched is not None

    bundle = _create_minimal_test_bundle(n_atoms=3, n_angles=0, angle_cap=32)
    folded = fold_matched_pattern_into_bundle(bundle, matched)

    displacement_fn, _ = space.free()
    angle_energy_fn = make_angle_energy_fn(displacement_fn, folded.angle_idx[: int(folded.n_angles)])
    pos_angstrom = jnp.asarray(positions_nm * NM_TO_ANGSTROM, dtype=jnp.float32)
    prolix_energy_akma = float(
        angle_energy_fn(pos_angstrom, folded.angle_params[: int(folded.n_angles)])
    )

    np.testing.assert_allclose(prolix_energy_akma, omm_energy_akma, rtol=1e-5)


def test_charmm_harmonic_improper_unhalved_numerical_parity_vs_openmm():
    """Verify that unhalved improper scaling (k_omm / 8.368) yields exact numerical parity with OpenMM."""
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)

    # Standard OpenMM CustomTorsionForce with 0.5: E = 0.5 * k * (theta - theta0)^2
    force = openmm.CustomTorsionForce("0.5 * k * (theta - theta0)^2")
    force.addPerTorsionParameter("k")
    force.addPerTorsionParameter("theta0")
    force.addTorsion(0, 1, 2, 3, [300.0, 0.2])  # k in kJ/(mol*rad^2)
    system.addForce(force)

    positions_nm = np.array(
        [
            [-0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.1, 0.05],
        ],
        dtype=np.float64,
    )
    pos_quantity = positions_nm * unit.nanometers
    omm_energy_kj = compute_openmm_energy(system, pos_quantity, platform_name="Reference")
    omm_energy_akma = omm_energy_kj * KJ_PER_MOL_TO_KCAL_PER_MOL

    matched = match_custom_force_pattern(force)
    assert matched is not None

    bundle = _create_minimal_test_bundle(n_atoms=4, n_impropers=0, dihedral_cap=32)
    folded = fold_matched_pattern_into_bundle(bundle, matched)

    displacement_fn, _ = space.free()
    improper_energy_fn = make_harmonic_improper_energy_fn(
        displacement_fn, folded.improper_idx[: int(folded.n_impropers)]
    )
    pos_angstrom = jnp.asarray(positions_nm * NM_TO_ANGSTROM, dtype=jnp.float32)

    # Note: make_harmonic_improper_energy_fn expects improper_params with shape (N, terms, 2) or (N, 2)
    # when stored as [k_prolix, phi0]
    p_imp = folded.improper_params[: int(folded.n_impropers), :2]
    # Broadcast terms dimension if needed
    p_imp_3d = p_imp[:, None, :]
    prolix_energy_akma = float(improper_energy_fn(pos_angstrom, p_imp_3d))

    np.testing.assert_allclose(prolix_energy_akma, omm_energy_akma, rtol=1e-5)
