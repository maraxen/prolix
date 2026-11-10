"""Adversarial stress-test suite for Milestone 2 Lepton compiler and custom force engine.

Authored by Challenger 1 (teamwork_challenger_m2_1) on branch feat/openmm-backend.
Empirically stress-tests:
1. Complex Lepton syntax: nested sub-expressions, topological DAG resolution,
   right-associative power (2^3^2 == 512, not 64), unary minus precedence (-2^2 == -4).
2. Negative base integer powers under jax.grad: absence of NaNs at cos(theta) <= 0.
3. Tabulated function stress tests: Continuous1D/2D/3D functions evaluated near boundary
   and domain limits against OpenMM Reference calculations.
4. CustomNonbondedForce edge cases: exclusions, interaction groups, NoCutoff distance,
   switching function at exact boundaries, and zero distance avoidance.
5. Analytical parameter derivatives dE/dlambda via jax.grad vs finite differences.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import pytest

from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.lepton_compiler import (
    BinaryOpNode,
    ConstNode,
    UnaryOpNode,
    VarNode,
    compile_continuous_1d_function,
    compile_continuous_2d_function,
    compile_continuous_3d_function,
    compile_custom_force,
    compile_tabulated_function,
    evaluate_custom_force,
    evaluate_integer_power,
    evaluate_lepton_program,
    lower_ast_to_jax,
    parse_lepton_expression,
    parse_lepton_single_expression,
    resolve_lepton_program,
)


# =============================================================================
# 1. Complex Lepton Syntax & Operator Precedence
# =============================================================================


class TestComplexLeptonSyntaxAdversarial:
    """Stress-test recursive-descent parser, operator precedence, and DAG resolution."""

    def test_right_associativity_power_deeply_nested(self):
        """Verify right-associativity: 2^3^2 == 512 (2^(3^2)), NOT 64 ((2^3)^2)."""
        # 1. Single expression AST structure
        ast = parse_lepton_single_expression("2^3^2")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "^"
        assert isinstance(ast.left, ConstNode) and ast.left.value == 2.0
        assert isinstance(ast.right, BinaryOpNode) and ast.right.op == "^"
        assert isinstance(ast.right.left, ConstNode) and ast.right.left.value == 3.0
        assert isinstance(ast.right.right, ConstNode) and ast.right.right.value == 2.0

        val = float(lower_ast_to_jax(ast, {}))
        assert val == 512.0, f"Expected 512.0 for 2^3^2, got {val}"

        # 2. Deep nesting: 2^2^2^2 == 2^(2^(2^2)) == 2^(2^4) == 2^16 == 65536
        ast_deep = parse_lepton_single_expression("2^2^2^2")
        val_deep = float(lower_ast_to_jax(ast_deep, {}))
        assert val_deep == 65536.0, f"Expected 65536.0 for 2^2^2^2, got {val_deep}"

        # 3. Negative base with right-associative power: (-2)^3^2 == (-2)^9 == -512
        ast_neg = parse_lepton_single_expression("(-2)^3^2")
        val_neg = float(lower_ast_to_jax(ast_neg, {}))
        assert val_neg == -512.0, f"Expected -512.0 for (-2)^3^2, got {val_neg}"

    def test_unary_minus_operator_precedence(self):
        """Verify unary minus precedence: -2^2 == -4 (-(2^2)), NOT 4 ((-2)^2)."""
        ast = parse_lepton_single_expression("-2^2")
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "-"
        assert isinstance(ast.operand, BinaryOpNode)
        assert ast.operand.op == "^"

        val = float(lower_ast_to_jax(ast, {}))
        assert val == -4.0, f"Expected -4.0 for -2^2, got {val}"

        # Multi-factor expressions with unary minus and power
        # -2 * 3^2 == -2 * 9 == -18
        val_mul = float(lower_ast_to_jax(parse_lepton_single_expression("-2 * 3^2"), {}))
        assert val_mul == -18.0, f"Expected -18.0 for -2 * 3^2, got {val_mul}"

        # 2 * -3^2 == 2 * -(3^2) == 2 * -9 == -18
        val_mul2 = float(lower_ast_to_jax(parse_lepton_single_expression("2 * -3^2"), {}))
        assert val_mul2 == -18.0, f"Expected -18.0 for 2 * -3^2, got {val_mul2}"

        # -(-2)^2 == -(4) == -4
        val_paren = float(lower_ast_to_jax(parse_lepton_single_expression("-(-2)^2"), {}))
        assert val_paren == -4.0, f"Expected -4.0 for -(-2)^2, got {val_paren}"

    def test_arbitrary_topological_definitions_deep_dag(self):
        """Verify arbitrary ordering resolution of a deep 8-stage dependency DAG."""
        # Program defined in reverse dependency order
        prog = (
            "result; "
            "result = step7 + 1.0; "
            "step7 = step6 * 2.0; "
            "step6 = step5 - 3.0; "
            "step5 = step4 + step3; "
            "step4 = step2 * 3.0; "
            "step3 = step1 + 4.0; "
            "step2 = base_var * 2.0; "
            "step1 = base_var + 10.0; "
            "base_var = 5.0"
        )
        primary, sorted_defs = resolve_lepton_program(prog)

        # base_var = 5
        # step1 = 15, step2 = 10
        # step3 = 19, step4 = 30
        # step5 = 49
        # step6 = 46
        # step7 = 92
        # result = 93
        val_eval = float(evaluate_lepton_program(primary, sorted_defs, {}))
        assert val_eval == 93.0, f"Expected 93.0 for deep DAG, got {val_eval}"

    def test_cycle_detection_indirect(self):
        """Verify cycle detector spots indirect cycles: a -> b -> c -> d -> a."""
        prog = "out; out = a + 1; a = b * 2; b = c - 3; c = d + 4; d = a / 2"
        with pytest.raises(ValueError, match="Cyclic dependency detected"):
            resolve_lepton_program(prog)


# =============================================================================
# 2. Negative Base Integer Powers Under jax.grad
# =============================================================================


class TestNegativeBaseIntegerPowersAdversarial:
    """Stress-test negative base integer powers and absence of NaNs under jax.grad."""

    def test_custom_angle_cos_cube_at_orthogonality(self):
        """Verify d/dtheta (cos(theta)^3) at theta = pi/2 (where cos = 0)."""
        omm_force = openmm.CustomAngleForce("cos(theta)^3")
        omm_force.addAngle(0, 1, 2, [])
        compiled = compile_custom_force(omm_force)

        # 90 degrees in Angstroms: atom 0 on +x, atom 1 at origin, atom 2 on +y
        pos_akma = jnp.array(
            [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            dtype=jnp.float32,
        )

        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        assert not jnp.isnan(e_akma), "Energy at theta=pi/2 is NaN"
        assert not jnp.any(jnp.isnan(forces_akma)), "Forces at theta=pi/2 contain NaN"
        assert np.isclose(float(e_akma), 0.0, atol=1e-12)
        assert np.allclose(forces_akma, 0.0, atol=1e-5)

    def test_custom_angle_cos_cube_at_obtuse_negative_cosine(self):
        """Verify d/dtheta (cos(theta)^3) at theta = 120 deg (where cos = -0.5 < 0) matches OpenMM."""
        omm_force = openmm.CustomAngleForce("cos(theta)^3")
        omm_force.addAngle(0, 1, 2, [])

        system = openmm.System()
        for _ in range(3):
            system.addParticle(1.0)
        system.addForce(omm_force)

        # 120 degrees in nm
        pos_omm = [
            openmm.Vec3(1.0, 0.0, 0.0),
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(-0.5, math.sqrt(3) / 2.0, 0.0),
        ]
        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(pos_omm)
        st = context.getState(getEnergy=True, getForces=True)
        omm_e = st.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
        omm_f = st.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoules_per_mole / openmm.unit.nanometer
        )

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array(
            [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 5.0 * math.sqrt(3), 0.0]],
            dtype=jnp.float32,
        )
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})

        assert not jnp.isnan(e_akma)
        assert not jnp.any(jnp.isnan(forces_akma))

        # Check energy parity
        prolix_e = float(e_akma) * 4.184
        assert np.isclose(prolix_e, omm_e, rtol=1e-5)

        # Check force parity: convert Prolix kcal/(mol*A) to kJ/(mol*nm) (* 41.84)
        prolix_f = np.array(forces_akma) * 41.84
        assert np.allclose(prolix_f, omm_f, atol=1e-4)

    @pytest.mark.parametrize("exp", [1, 2, 3, 4, 5, 6, 7])
    def test_integer_power_unrolled_gradients_negative_base(self, exp: int):
        """Verify d/dx (x^exp) produces finite, correct gradients for negative base x = -1.5."""
        x = jnp.array(-1.5, dtype=jnp.float32)

        def _fn(val):
            return evaluate_integer_power(val, exp)

        res, grad = jax.value_and_grad(_fn)(x)
        assert not jnp.isnan(res)
        assert not jnp.isnan(grad)

        expected_res = (-1.5) ** exp
        expected_grad = exp * ((-1.5) ** (exp - 1))
        assert np.isclose(float(res), expected_res, rtol=1e-4)
        assert np.isclose(float(grad), expected_grad, rtol=1e-4)


# =============================================================================
# 3. Tabulated Function Stress Tests
# =============================================================================


class TestTabulatedFunctionsAdversarial:
    """Stress-test TabulatedFunction classes at boundary and domain limits against OpenMM Reference."""

    def test_continuous_1d_boundary_and_extrapolation_parity(self):
        """Verify Continuous1DFunction behavior at exact boundaries and outside [xmin, xmax]."""
        vals1d = [1.0, 4.0, 9.0, 16.0]
        xmin, xmax = 1.0, 4.0
        omm_1d = openmm.Continuous1DFunction(vals1d, xmin, xmax)
        jax_1d = compile_tabulated_function(omm_1d)

        # Reference setup
        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        force = openmm.CustomCompoundBondForce(2, "f(distance(p1, p2))")
        force.addTabulatedFunction("f", omm_1d)
        force.addBond([0, 1], [])
        system.addForce(force)

        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )

        test_points = [
            0.5,      # Far outside left -> 0.0
            0.9999,   # Just outside left -> 0.0
            1.0,      # Exact left boundary -> 1.0
            1.0001,   # Just inside left
            2.5,      # Interior midpoint
            3.9999,   # Just inside right
            4.0,      # Exact right boundary -> 16.0
            4.0001,   # Just outside right -> 0.0
            5.0,      # Far outside right -> 0.0
        ]
        for r in test_points:
            context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(r, 0, 0)])
            omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            jax_e = float(jax_1d(jnp.array(r, dtype=jnp.float32)))
            assert np.isclose(jax_e, omm_e, atol=1e-5), (
                f"1D spline mismatch at r={r}: omm={omm_e}, jax={jax_e}"
            )

    def test_continuous_2d_boundary_and_extrapolation_parity(self):
        """Verify Continuous2DFunction behavior at grid boundaries and corners vs OpenMM Reference."""
        xsize, ysize = 3, 3
        vals_2d = [0.0, 1.0, 4.0, 1.0, 2.0, 5.0, 4.0, 5.0, 8.0]
        xmin, xmax = 0.0, 2.0
        ymin, ymax = 0.0, 2.0
        omm_2d = openmm.Continuous2DFunction(xsize, ysize, vals_2d, xmin, xmax, ymin, ymax)
        jax_2d = compile_tabulated_function(omm_2d)

        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        force = openmm.CustomCompoundBondForce(2, "f(x1, y2)")
        force.addTabulatedFunction("f", omm_2d)
        force.addBond([0, 1], [])
        system.addForce(force)

        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )

        test_points = [
            (0.0, 0.0),          # Origin corner
            (2.0, 2.0),          # Far corner
            (0.0001, 0.0001),    # Near corner
            (1.9999, 1.9999),    # Near far corner
            (0.5, 1.2),          # Interior point
            (-0.1, 1.0),         # Outside left -> 0.0
            (2.1, 1.0),          # Outside right -> 0.0
            (1.0, -0.1),         # Outside bottom -> 0.0
            (1.0, 2.1),          # Outside top -> 0.0
        ]
        for x, y in test_points:
            context.setPositions([openmm.Vec3(x, 0, 0), openmm.Vec3(0, y, 0)])
            omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            jax_e = float(jax_2d(jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)))
            assert np.isclose(jax_e, omm_e, atol=1e-5), (
                f"2D spline mismatch at ({x}, {y}): omm={omm_e}, jax={jax_e}"
            )

    def test_continuous_3d_discrepancy_on_non_trivial_grid(self):
        """EMPERICAL FINDING: Continuous3DFunction on 3x3x3 grid exhibits 35% error.

        In lepton_compiler.py line 794, Continuous3DFunction is implemented via
        TRILINEAR interpolation rather than the claimed OpenMM 64-parameter TRICUBIC spline.
        This test documents the empirical divergence against OpenMM Reference.
        """
        xsize, ysize, zsize = 3, 3, 3
        vals_3d = [float(i**2) for i in range(27)]
        xmin, xmax = 0.0, 2.0
        ymin, ymax = 0.0, 2.0
        zmin, zmax = 0.0, 2.0
        omm_3d = openmm.Continuous3DFunction(
            xsize, ysize, zsize, vals_3d, xmin, xmax, ymin, ymax, zmin, zmax
        )
        jax_3d = compile_tabulated_function(omm_3d)

        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        force = openmm.CustomCompoundBondForce(2, "f(x1, y1, z1)")
        force.addTabulatedFunction("f", omm_3d)
        force.addBond([0, 1], [])
        system.addForce(force)

        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions([openmm.Vec3(0.5, 0.5, 0.5), openmm.Vec3(0, 0, 0)])
        omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoules_per_mole
        )
        jax_e = float(jax_3d(jnp.array(0.5), jnp.array(0.5), jnp.array(0.5)))

        rel_error = abs(omm_e - jax_e) / abs(omm_e)
        # Expected OpenMM tricubic: 47.9375 kJ/mol; Prolix trilinear: 65.0 kJ/mol (35.6% error)
        # We document this discrepancy: it will fail an exact parity check < 1e-4!
        if rel_error > 0.01:
            pytest.fail(
                f"Continuous3DFunction tricubic vs trilinear discrepancy: "
                f"OpenMM Reference={omm_e:.4f}, Prolix={jax_e:.4f}, rel_error={rel_error:.2%}"
            )


# =============================================================================
# 4. CustomNonbondedForce Edge Cases
# =============================================================================


class TestCustomNonbondedForceAdversarial:
    """Stress-test exclusions, interaction groups, cutoff modes, and zero distances."""

    def test_custom_nonbonded_exclusions_correctness(self):
        """Verify 1-2 and 1-3 exclusions are masked out correctly."""
        omm_force = openmm.CustomNonbondedForce("q1*q2/r")
        omm_force.addPerParticleParameter("q")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(2.0)

        for _ in range(3):
            omm_force.addParticle([1.0])

        # Exclude pairs 0-1 and 1-2, leaving ONLY pair 0-2
        omm_force.addExclusion(0, 1)
        omm_force.addExclusion(1, 2)

        system = openmm.System()
        for _ in range(3):
            system.addParticle(1.0)
        system.addForce(omm_force)

        # Positions: atom 0 at 0.0 nm, atom 1 at 0.2 nm, atom 2 at 0.5 nm
        # Distance (0, 2) = 0.5 nm -> Energy = 1.0 * 1.0 / 0.5 = 2.0 kJ/mol
        positions = [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.2, 0.0, 0.0),
            openmm.Vec3(0.5, 0.0, 0.0),
        ]
        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(positions)
        omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoules_per_mole
        )

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})

        assert not jnp.isnan(e_akma)
        assert not jnp.any(jnp.isnan(forces_akma))
        prolix_e = float(e_akma) * 4.184
        assert np.isclose(prolix_e, omm_e, rtol=1e-5)

    def test_custom_nonbonded_interaction_groups_ignored_defect(self):
        """EMPERICAL FINDING: CustomNonbondedForce interaction_groups is ignored in energy_omm.

        When addInteractionGroup([0, 1], [2, 3]) is set, OpenMM restricts evaluation
        to pairs spanning group1 and group2. Prolix computes all pairs, yielding wrong energy.
        """
        omm_force = openmm.CustomNonbondedForce("q1*q2/r")
        omm_force.addPerParticleParameter("q")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(2.0)

        for _ in range(4):
            omm_force.addParticle([1.0])

        # Interaction group between {0, 1} and {2, 3}
        omm_force.addInteractionGroup([0, 1], [2, 3])

        system = openmm.System()
        for _ in range(4):
            system.addParticle(1.0)
        system.addForce(omm_force)

        # Positions: 0.0, 0.1, 0.3, 0.4 nm
        positions = [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.1, 0.0, 0.0),
            openmm.Vec3(0.3, 0.0, 0.0),
            openmm.Vec3(0.4, 0.0, 0.0),
        ]
        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(positions)
        omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoules_per_mole
        )

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        e_akma, _, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        prolix_e = float(e_akma) * 4.184

        diff = abs(omm_e - prolix_e)
        # OpenMM Reference: 14.1667 kJ/mol; Prolix: 34.1667 kJ/mol (diff = 20.0 kJ/mol)
        if diff > 1e-3:
            pytest.fail(
                f"CustomNonbondedForce interaction_groups bug: "
                f"OpenMM Reference={omm_e:.4f} kJ/mol, Prolix={prolix_e:.4f} kJ/mol, diff={diff:.4f}"
            )

    def test_custom_nonbonded_no_cutoff_inappropriate_truncation_defect(self):
        """EMPERICAL FINDING: NoCutoff mode is incorrectly truncated at default cutoff_distance.

        In lepton_compiler.py lines 1431-1432, `is_valid` checks `if self.cutoff_distance > 0:`.
        Because cutoff_distance defaults to 1.0 in OpenMM, Prolix truncates NoCutoff forces at 1.0 nm!
        """
        omm_force = openmm.CustomNonbondedForce("q1*q2/r")
        omm_force.addPerParticleParameter("q")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

        omm_force.addParticle([1.0])
        omm_force.addParticle([1.0])

        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addForce(omm_force)

        # Distance = 1.5 nm (> 1.0 nm default cutoff)
        positions = [openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(1.5, 0.0, 0.0)]
        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(positions)
        omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoules_per_mole
        )

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]], dtype=jnp.float32)
        e_akma, _, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        prolix_e = float(e_akma) * 4.184

        diff = abs(omm_e - prolix_e)
        # OpenMM Reference: 1.0 / 1.5 = 0.6667 kJ/mol; Prolix: 0.0 kJ/mol
        if diff > 1e-3:
            pytest.fail(
                f"CustomNonbondedForce NoCutoff truncation bug: "
                f"OpenMM Reference={omm_e:.4f} kJ/mol, Prolix={prolix_e:.4f} kJ/mol, diff={diff:.4f}"
            )

    def test_custom_nonbonded_switching_exact_boundaries(self):
        """Verify switching function continuity at r_switch and r_cutoff."""
        omm_force = openmm.CustomNonbondedForce("1.0/r")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(1.0)
        omm_force.setUseSwitchingFunction(True)
        omm_force.setSwitchingDistance(0.8)

        omm_force.addParticle([])
        omm_force.addParticle([])

        compiled = compile_custom_force(omm_force)

        # At r = 0.8 nm: s = 1.0, E = 1/0.8 = 1.25 kJ/mol
        pos_switch = jnp.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=jnp.float32)
        e_switch, _, _ = evaluate_custom_force(compiled, pos_switch, None, {})
        assert np.isclose(float(e_switch) * 4.184, 1.25, rtol=1e-4)

        # At r >= 1.0 nm: s = 0.0, E = 0.0
        pos_cutoff = jnp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=jnp.float32)
        e_cutoff, f_cutoff, _ = evaluate_custom_force(compiled, pos_cutoff, None, {})
        assert float(e_cutoff) == 0.0
        assert np.allclose(f_cutoff, 0.0)

    def test_custom_nonbonded_zero_distance_avoidance(self):
        """Verify that excluded particles at exactly identical coordinates (r=0) do NOT cause NaNs."""
        omm_force = openmm.CustomNonbondedForce("4 * ((0.3/r)^12 - (0.3/r)^6)")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(1.0)

        omm_force.addParticle([])
        omm_force.addParticle([])
        omm_force.addExclusion(0, 1)

        compiled = compile_custom_force(omm_force)
        # Identical coordinates: r = 0.0
        pos_zero = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
        e, f, _ = evaluate_custom_force(compiled, pos_zero, None, {})

        assert not jnp.isnan(e)
        assert not jnp.any(jnp.isnan(f))
        assert float(e) == 0.0
        assert np.allclose(f, 0.0)


# =============================================================================
# 5. Analytical Parameter Derivatives via jax.grad vs Finite Differences
# =============================================================================


class TestParameterDerivativesAdversarial:
    """Stress-test analytical parameter derivatives dE/dlambda against finite differences."""

    def test_custom_bond_nonlinear_lambda_derivative(self):
        """Verify dE/dlambda for nonlinear alchemical coupling: E = (1-lambda)^4 * E_A + lambda^4 * E_B."""
        expr = "(1.0 - lambda)^4 * 0.5 * k * (r - r0)^2 + lambda^4 * 0.5 * k * (r - r1)^2"
        omm_force = openmm.CustomBondForce(expr)
        omm_force.addPerBondParameter("r0")
        omm_force.addPerBondParameter("r1")
        omm_force.addPerBondParameter("k")
        omm_force.addGlobalParameter("lambda", 0.3)
        omm_force.addEnergyParameterDerivative("lambda")
        omm_force.addBond(0, 1, [0.15, 0.25, 1000.0])

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)  # r = 0.2 nm

        lam_val = 0.3
        gp = {"lambda": jnp.array(lam_val, dtype=jnp.float32)}
        _, _, pderivs = evaluate_custom_force(compiled, pos_akma, None, gp, compute_derivatives=True)
        dE_dlam_jax = float(pderivs["lambda"])

        # Central finite difference
        h = 1e-4
        gp_plus = {"lambda": jnp.array(lam_val + h, dtype=jnp.float32)}
        gp_minus = {"lambda": jnp.array(lam_val - h, dtype=jnp.float32)}
        pos_nm = pos_akma * 0.1
        e_plus = float(compiled.energy_omm(pos_nm, None, gp_plus))
        e_minus = float(compiled.energy_omm(pos_nm, None, gp_minus))
        dE_dlam_fd = (e_plus - e_minus) / (2.0 * h)

        rel_diff = abs(dE_dlam_jax - dE_dlam_fd) / abs(dE_dlam_fd)
        assert rel_diff < 1e-3, (
            f"Parameter derivative mismatch: analytical={dE_dlam_jax}, finite_diff={dE_dlam_fd}"
        )

    def test_custom_nonbonded_computed_value_lambda_derivative(self):
        """Verify dE/dlambda propagating through ComputedValues in CustomNonbondedForce."""
        expr = "4 * eps * ((sig/r)^12 - (sig/r)^6); sig = sig1 + sig2; eps = eps1 * eps2"
        omm_force = openmm.CustomNonbondedForce(expr)
        omm_force.addPerParticleParameter("sigma_base")
        omm_force.addPerParticleParameter("epsilon_base")
        omm_force.addGlobalParameter("lambda", 0.5)
        omm_force.addEnergyParameterDerivative("lambda")
        omm_force.addComputedValue("sig", "sigma_base * lambda")
        omm_force.addComputedValue("eps", "epsilon_base * (1.0 - lambda)")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(2.0)

        omm_force.addParticle([0.15, 1.0])
        omm_force.addParticle([0.15, 1.0])

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float32)  # r = 0.4 nm

        lam_val = 0.5
        gp = {"lambda": jnp.array(lam_val, dtype=jnp.float32)}
        _, _, pderivs = evaluate_custom_force(compiled, pos_akma, None, gp, compute_derivatives=True)
        dE_dlam_jax = float(pderivs["lambda"])

        # Central finite difference
        h = 1e-4
        gp_plus = {"lambda": jnp.array(lam_val + h, dtype=jnp.float32)}
        gp_minus = {"lambda": jnp.array(lam_val - h, dtype=jnp.float32)}
        pos_nm = pos_akma * 0.1
        e_plus = float(compiled.energy_omm(pos_nm, None, gp_plus))
        e_minus = float(compiled.energy_omm(pos_nm, None, gp_minus))
        dE_dlam_fd = (e_plus - e_minus) / (2.0 * h)

        rel_diff = abs(dE_dlam_jax - dE_dlam_fd) / abs(dE_dlam_fd)
        assert rel_diff < 1e-3, (
            f"CustomNonbondedForce dE/dlambda mismatch: analytical={dE_dlam_jax}, finite_diff={dE_dlam_fd}"
        )
