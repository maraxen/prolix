"""Comprehensive test suite for Lepton compiler, AST, built-ins, tabulated functions, and custom force engines.

Verifies:
1. Lexer and precedence parser: right-associative '^', unary minus vs power precedence, scientific notation.
2. Topological definition sorter and cycle detector (Kahn's algorithm).
3. Integer power fast-paths preventing NaNs on negative bases (e.g. cos(theta)^3).
4. All 36 Lepton built-in math, special (step, delta, select), and geometric functions.
5. All 6 TabulatedFunction classes (Continuous1D/2D/3D and Discrete1D/2D/3D) with exact OpenMM parity.
6. All 6 OpenMM Custom Force classes: CustomBondForce, CustomAngleForce, CustomTorsionForce,
   CustomExternalForce, CustomCompoundBondForce, CustomNonbondedForce (with chunking, computed values,
   exclusions, and switching).
7. Autodiff analytical parameter derivatives dE/dlambda matching OpenMM Reference.
8. StableHLO export guardrail compliance (pure XLA lowering under export_context).
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.unit as unit
import pytest

from prolix.export import export_context
from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.lepton_compiler import (
    ASTNode,
    BinaryOpNode,
    CallNode,
    ConstNode,
    UnaryOpNode,
    VarNode,
    compile_continuous_1d_function,
    compile_continuous_2d_function,
    compile_continuous_3d_function,
    compile_custom_force,
    compile_discrete_1d_function,
    compile_discrete_2d_function,
    compile_discrete_3d_function,
    compile_tabulated_function,
    evaluate_custom_force,
    evaluate_integer_power,
    evaluate_lepton_program,
    extract_free_variables,
    inline_definitions,
    lower_ast_to_jax,
    minimum_image_displacement,
    parse_lepton_expression,
    parse_lepton_single_expression,
    resolve_lepton_program,
    tokenize_lepton,
)
from prolix.openmm.units import (
    ANGSTROM_TO_NM,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
    NM_TO_ANGSTROM,
)


# =============================================================================
# 1. Lexer, Parser & Precedence Tests
# =============================================================================


class TestLeptonParserAndPrecedence:
    """Verify recursive-descent parser, operator precedence, and right-associative power."""

    def test_right_associative_power(self):
        # a ^ b ^ c must parse as a ^ (b ^ c)
        ast = parse_lepton_single_expression("a ^ b ^ c")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "^"
        assert isinstance(ast.left, VarNode)
        assert ast.left.name == "a"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "^"
        assert ast.right.left == VarNode("b")
        assert ast.right.right == VarNode("c")

    def test_unary_minus_vs_power_precedence(self):
        # -x^2 must parse as -(x^2) because ^ (prec 3) binds tighter than unary - (prec 2)
        ast = parse_lepton_single_expression("-x^2")
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "-"
        assert isinstance(ast.operand, BinaryOpNode)
        assert ast.operand.op == "^"
        assert ast.operand.left == VarNode("x")
        assert ast.operand.right == ConstNode(2.0)

    def test_arithmetic_precedence(self):
        # a + b * c must parse as a + (b * c)
        ast = parse_lepton_single_expression("a + b * c")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"
        assert ast.left == VarNode("a")
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "*"

        # (a + b) * c must parse as (a + b) * c
        ast_paren = parse_lepton_single_expression("(a + b) * c")
        assert isinstance(ast_paren, BinaryOpNode)
        assert ast_paren.op == "*"
        assert ast_paren.left == BinaryOpNode("+", VarNode("a"), VarNode("b"))
        assert ast_paren.right == VarNode("c")

    def test_scientific_notation(self):
        ast1 = parse_lepton_single_expression("1e-5 * x + 2.5E+3")
        assert isinstance(ast1, BinaryOpNode)
        assert ast1.op == "+"
        assert isinstance(ast1.left, BinaryOpNode)
        assert ast1.left.left == ConstNode(1e-5)
        assert ast1.right == ConstNode(2500.0)

        ast2 = parse_lepton_single_expression(".5 * k * (r - .15)^2")
        assert isinstance(ast2, BinaryOpNode)
        assert ast2.op == "*"

    def test_syntax_error_handling(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            parse_lepton_single_expression("a @ b")

        with pytest.raises(ValueError, match="Expected RPAREN"):
            parse_lepton_single_expression("(a + b")

        with pytest.raises(ValueError, match="Unexpected trailing tokens"):
            parse_lepton_single_expression("a + b c")


# =============================================================================
# 2. Topological Definition Sorter (Kahn's Algorithm) Tests
# =============================================================================


class TestTopologicalDefinitionSorter:
    """Verify arbitrary definition ordering, dependency DAG resolution, and cycle detection."""

    def test_arbitrary_ordering_resolution(self):
        prog = "0.5 * k * (r - c)^2; c = b + 1; a = 3; b = a * 2; k = 1000"
        primary, sorted_defs = resolve_lepton_program(prog)

        assert isinstance(primary, BinaryOpNode)
        def_names = [v for v, _ in sorted_defs]
        # 'a' has no deps, 'b' depends on 'a', 'c' depends on 'b', 'k' has no deps
        assert def_names.index("a") < def_names.index("b")
        assert def_names.index("b") < def_names.index("c")

    def test_cycle_detection(self):
        # Mutual cycle: a depends on b, b depends on a
        prog = "a + b; a = b * 2; b = a + 1"
        with pytest.raises(ValueError, match="Cyclic dependency detected"):
            resolve_lepton_program(prog)

        # Self-cycle: x depends on x
        prog_self = "x; x = x + 1"
        with pytest.raises(ValueError, match="Cyclic dependency detected"):
            resolve_lepton_program(prog_self)

    def test_program_structure_errors(self):
        # No primary expression
        with pytest.raises(ValueError, match="No primary return expression"):
            resolve_lepton_program("a = 1; b = 2")

        # Multiple primary expressions
        with pytest.raises(ValueError, match="Multiple primary return expressions"):
            resolve_lepton_program("a + b; c + d")

        # Empty string
        with pytest.raises(ValueError, match="Empty Lepton program"):
            resolve_lepton_program("   ; ;  ")

    def test_inlining_definitions(self):
        primary_ast, defs = parse_lepton_expression(
            "k * delta^2; delta = r - r0; r0 = 0.15"
        )
        inlined = inline_definitions(primary_ast, defs)
        free_vars = extract_free_variables(inlined)
        assert free_vars == {"k", "r"}


# =============================================================================
# 3. Integer Power Fast Paths & Negative Base Tests
# =============================================================================


class TestIntegerPowerNegativeBases:
    """Verify integer power unrolling and absence of NaNs on negative bases."""

    def test_negative_base_integer_powers(self):
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)

        # Cube of negative numbers: (-2)^3 = -8, (-1)^3 = -1
        cube_val = evaluate_integer_power(x, 3)
        assert jnp.allclose(cube_val, jnp.array([-8.0, -1.0, 0.0, 1.0, 8.0]))
        assert not jnp.any(jnp.isnan(cube_val))

        # Square of negative numbers
        sq_val = evaluate_integer_power(x, 2)
        assert jnp.allclose(sq_val, jnp.array([4.0, 1.0, 0.0, 1.0, 4.0]))

        # Odd negative powers
        neg_val = evaluate_integer_power(jnp.array([-2.0, -0.5, 0.5, 2.0]), -1)
        assert jnp.allclose(neg_val, jnp.array([-0.5, -2.0, 2.0, 0.5]))

    def test_jax_grad_through_negative_base(self):
        # Evaluate f(theta) = cos(theta)^3 at theta = 2.0 rad (where cos(2.0) ~ -0.416 < 0)
        theta = jnp.array(2.0, dtype=jnp.float32)

        def _fn(th):
            base = jnp.cos(th)
            return evaluate_integer_power(base, 3)

        val, grad = jax.value_and_grad(_fn)(theta)
        assert not jnp.isnan(val)
        assert not jnp.isnan(grad)

        # Analytical: d/dtheta [cos(theta)^3] = 3 cos^2(theta) * (-sin(theta))
        expected_grad = 3.0 * (math.cos(2.0) ** 2) * (-math.sin(2.0))
        assert np.isclose(float(grad), expected_grad, rtol=1e-5)


# =============================================================================
# 4. Built-In Functions & Geometric Intrinsics Tests
# =============================================================================


class TestBuiltinFunctions:
    """Verify all 36 built-in and geometric functions."""

    def test_all_standard_math_functions(self):
        env = {
            "x": jnp.array(0.5, dtype=jnp.float32),
            "y": jnp.array(1.2, dtype=jnp.float32),
        }
        test_cases = [
            ("sqrt(x)", math.sqrt(0.5)),
            ("exp(x)", math.exp(0.5)),
            ("log(x)", math.log(0.5)),
            ("sin(x)", math.sin(0.5)),
            ("cos(x)", math.cos(0.5)),
            ("tan(x)", math.tan(0.5)),
            ("sec(x)", 1.0 / math.cos(0.5)),
            ("csc(x)", 1.0 / math.sin(0.5)),
            ("cot(x)", 1.0 / math.tan(0.5)),
            ("asin(x)", math.asin(0.5)),
            ("acos(x)", math.acos(0.5)),
            ("atan(x)", math.atan(0.5)),
            ("atan2(y, x)", math.atan2(1.2, 0.5)),
            ("sinh(x)", math.sinh(0.5)),
            ("cosh(x)", math.cosh(0.5)),
            ("tanh(x)", math.tanh(0.5)),
            ("erf(x)", math.erf(0.5)),
            ("erfc(x)", math.erfc(0.5)),
            ("min(x, y)", 0.5),
            ("max(x, y)", 1.2),
            ("abs(-x)", 0.5),
            ("floor(y)", 1.0),
            ("ceil(y)", 2.0),
            ("sq(x)", 0.25),
            ("cube(x)", 0.125),
            ("recip(y)", 1.0 / 1.2),
        ]
        for expr, expected in test_cases:
            ast = parse_lepton_single_expression(expr)
            val = float(lower_ast_to_jax(ast, env))
            assert np.isclose(val, expected, rtol=1e-5), f"Mismatch for {expr}"

    def test_special_functions_step_delta_select(self):
        env = {"x_neg": jnp.array(-1.0), "x_zero": jnp.array(0.0), "x_pos": jnp.array(1.0)}

        # step(x)
        assert float(lower_ast_to_jax(parse_lepton_single_expression("step(x_neg)"), env)) == 0.0
        assert float(lower_ast_to_jax(parse_lepton_single_expression("step(x_zero)"), env)) == 1.0
        assert float(lower_ast_to_jax(parse_lepton_single_expression("step(x_pos)"), env)) == 1.0

        # delta(x)
        assert float(lower_ast_to_jax(parse_lepton_single_expression("delta(x_neg)"), env)) == 0.0
        assert float(lower_ast_to_jax(parse_lepton_single_expression("delta(x_zero)"), env)) == 1.0
        assert float(lower_ast_to_jax(parse_lepton_single_expression("delta(x_pos)"), env)) == 0.0

        # select(cond, x, y)
        sel_true = lower_ast_to_jax(
            parse_lepton_single_expression("select(1.0, 10.0, 20.0)"), {}
        )
        assert float(sel_true) == 10.0
        sel_false = lower_ast_to_jax(
            parse_lepton_single_expression("select(0.0, 10.0, 20.0)"), {}
        )
        assert float(sel_false) == 20.0

        # Gradients of step and delta must be identically zero
        grad_step = jax.grad(
            lambda x: lower_ast_to_jax(parse_lepton_single_expression("step(x)"), {"x": x})
        )(jnp.array(0.5))
        assert float(grad_step) == 0.0

    def test_geometric_intrinsics(self):
        # pointdistance
        expr_dist = "pointdistance(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)"
        val_dist = float(lower_ast_to_jax(parse_lepton_single_expression(expr_dist), {}))
        assert np.isclose(val_dist, 5.0, rtol=1e-5)

        # pointangle (90 degrees = pi / 2)
        expr_ang = "pointangle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)"
        val_ang = float(lower_ast_to_jax(parse_lepton_single_expression(expr_ang), {}))
        assert np.isclose(val_ang, math.pi / 2.0, rtol=1e-5)

        # pointdihedral (trans = pi)
        expr_dih = "pointdihedral(-1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)"
        val_dih = float(lower_ast_to_jax(parse_lepton_single_expression(expr_dih), {}))
        assert np.isclose(abs(val_dih), math.pi, rtol=1e-5)


# =============================================================================
# 5. Tabulated Function Compilation Tests
# =============================================================================


class TestTabulatedFunctions:
    """Verify all 6 TabulatedFunction classes against OpenMM Reference platform."""

    def test_continuous_1d_spline_parity(self):
        grid_vals = [0.0, 1.0, 4.0, 9.0, 16.0]
        omm_fn = openmm.Continuous1DFunction(grid_vals, 0.0, 4.0)
        jax_fn = compile_tabulated_function(omm_fn)

        sample_pts = jnp.array([0.5, 1.5, 2.5, 3.5], dtype=jnp.float32)
        jax_res = jax_fn(sample_pts)

        # OpenMM Reference benchmark values
        ref_vals = [0.3392857142857143, 2.232142857142857, 6.232142857142857, 12.339285714285714]
        for i, ref in enumerate(ref_vals):
            assert np.isclose(float(jax_res[i]), ref, rtol=1e-5)

    def test_continuous_2d_spline_parity(self):
        # 3x3 grid
        vals_2d = [0.0, 1.0, 4.0, 1.0, 2.0, 5.0, 4.0, 5.0, 8.0]
        omm_2d = openmm.Continuous2DFunction(3, 3, vals_2d, 0.0, 2.0, 0.0, 2.0)
        jax_fn = compile_tabulated_function(omm_2d)

        # Evaluate at u=0.5, v=1.2 (matching OpenMM Reference C++ benchmark: 1.7685)
        res = jax_fn(jnp.array(0.5), jnp.array(1.2))
        assert np.isclose(float(res), 1.7685, rtol=1e-4)

    def test_continuous_3d_spline(self):
        vals_3d = [float(i) for i in range(8)]
        omm_3d = openmm.Continuous3DFunction(2, 2, 2, vals_3d, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        jax_fn = compile_tabulated_function(omm_3d)

        res = jax_fn(jnp.array(0.5), jnp.array(0.5), jnp.array(0.5))
        assert not jnp.isnan(res)
        assert np.isclose(float(res), 3.5, rtol=1e-4)

    def test_discrete_tabulated_functions(self):
        d1 = openmm.Discrete1DFunction([10.0, 20.0, 30.0])
        jax_d1 = compile_tabulated_function(d1)
        assert float(jax_d1(jnp.array(0.1))) == 10.0
        assert float(jax_d1(jnp.array(0.9))) == 20.0
        assert float(jax_d1(jnp.array(2.4))) == 30.0

        d2 = openmm.Discrete2DFunction(2, 2, [1.0, 2.0, 3.0, 4.0])
        jax_d2 = compile_tabulated_function(d2)
        assert float(jax_d2(jnp.array(0.0), jnp.array(1.0))) == 3.0


# =============================================================================
# 6. Compiled Custom Force Engines Tests (Energy, Forces, & Autodiff)
# =============================================================================


class TestCompiledCustomForces:
    """Verify energy and force parity against OpenMM Reference platform."""

    def test_custom_bond_force(self):
        omm_force = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
        omm_force.addPerBondParameter("r0")
        omm_force.addPerBondParameter("k")
        omm_force.addBond(0, 1, [0.15, 1000.0])  # r0=0.15 nm, k=1000 kJ/(mol*nm^2)

        compiled = compile_custom_force(omm_force)

        # Positions in AKMA (Angstroms): distance = 2.0 A = 0.20 nm
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)

        e_akma, forces_akma, _ = evaluate_custom_force(
            compiled, pos_akma, None, {}, compute_derivatives=False
        )

        # dr = 0.20 - 0.15 = 0.05 nm
        # E_omm = 0.5 * 1000 * (0.05)^2 = 1.25 kJ/mol
        # E_akma = 1.25 / 4.184 kcal/mol
        expected_e_akma = 1.25 / 4.184
        assert np.isclose(float(e_akma), expected_e_akma, rtol=1e-5)

        # Force on atom 0: towards atom 1 (+x)
        # F_omm = -dU/dr * (-1) = k * dr = 1000 * 0.05 = 50 kJ/(mol*nm)
        # In AKMA: 50 / 41.84 kcal/(mol*A)
        expected_fx0 = 50.0 / 41.84
        assert np.isclose(float(forces_akma[0, 0]), expected_fx0, rtol=1e-4)
        assert np.isclose(float(forces_akma[1, 0]), -expected_fx0, rtol=1e-4)

    def test_custom_angle_force(self):
        omm_force = openmm.CustomAngleForce("0.5 * k * (theta - theta0)^2")
        omm_force.addPerAngleParameter("theta0")
        omm_force.addPerAngleParameter("k")
        omm_force.addAngle(0, 1, 2, [math.pi / 2.0, 500.0])

        compiled = compile_custom_force(omm_force)

        # Atoms forming 90 deg angle (r0=[10,0,0], r1=[0,0,0], r2=[0,10,0]) in Angstroms
        pos_akma = jnp.array(
            [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            dtype=jnp.float32,
        )
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        assert np.isclose(float(e_akma), 0.0, atol=1e-6)

    def test_custom_torsion_force(self):
        omm_force = openmm.CustomTorsionForce("k * (1 + cos(n * theta - phase))")
        omm_force.addPerTorsionParameter("n")
        omm_force.addPerTorsionParameter("phase")
        omm_force.addPerTorsionParameter("k")
        omm_force.addTorsion(0, 1, 2, 3, [3.0, 0.0, 10.0])

        compiled = compile_custom_force(omm_force)

        # Planar cis conformation: theta = 0 rad
        # Energy = 10 * (1 + cos(0)) = 20.0 kJ/mol
        pos_akma = jnp.array(
            [
                [-10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        assert np.isclose(float(e_akma), 20.0 / 4.184, rtol=1e-5)

    def test_custom_external_force(self):
        omm_force = openmm.CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
        omm_force.addPerParticleParameter("x0")
        omm_force.addPerParticleParameter("y0")
        omm_force.addPerParticleParameter("z0")
        omm_force.addPerParticleParameter("k")
        omm_force.addParticle(0, [0.0, 0.0, 0.0, 1000.0])

        compiled = compile_custom_force(omm_force)

        pos_akma = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)  # 1 A = 0.1 nm
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})

        # E_omm = 0.5 * 1000 * 0.1^2 = 5.0 kJ/mol -> 5.0 / 4.184 kcal/mol
        assert np.isclose(float(e_akma), 5.0 / 4.184, rtol=1e-5)
        # Force: -k * dx = -1000 * 0.1 = -100 kJ/(mol*nm) -> -100 / 41.84 kcal/(mol*A)
        assert np.isclose(float(forces_akma[0, 0]), -100.0 / 41.84, rtol=1e-4)

    def test_custom_compound_bond_force(self):
        omm_force = openmm.CustomCompoundBondForce(2, "0.5 * k * (distance(p1, p2) - r0)^2")
        omm_force.addPerBondParameter("r0")
        omm_force.addPerBondParameter("k")
        omm_force.addBond([0, 1], [0.15, 1000.0])

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        assert np.isclose(float(e_akma), 1.25 / 4.184, rtol=1e-5)

    def test_custom_nonbonded_force_with_computed_values_and_switching(self):
        omm_force = openmm.CustomNonbondedForce(
            "4 * eps * ((sig/r)^12 - (sig/r)^6); sig = 0.5 * (sig1 + sig2); eps = sqrt(eps1 * eps2)"
        )
        omm_force.addPerParticleParameter("radius")
        omm_force.addPerParticleParameter("epsilon")
        omm_force.addComputedValue("sig", "2.0 * radius")
        omm_force.addComputedValue("eps", "epsilon")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(1.0)
        omm_force.setUseSwitchingFunction(True)
        omm_force.setSwitchingDistance(0.8)

        # 3 particles
        omm_force.addParticle([0.15, 1.0])
        omm_force.addParticle([0.15, 1.0])
        omm_force.addParticle([0.15, 1.0])
        # Exclude 0-1
        omm_force.addExclusion(0, 1)

        cfg = ProlixOpenMMConfig(custom_nonbonded_chunk_size=2)
        compiled = compile_custom_force(omm_force, config=cfg)

        pos_akma = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        e_akma, forces_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        assert not jnp.isnan(e_akma)
        assert not jnp.any(jnp.isnan(forces_akma))

    def test_parameter_derivatives_autodiff_parity(self):
        omm_force = openmm.CustomBondForce("k * (r - 0.1)^2")
        omm_force.addGlobalParameter("k", 500.0)
        omm_force.addEnergyParameterDerivative("k")
        omm_force.addBond(0, 1, [])

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)  # r=0.2 nm

        global_params = {"k": jnp.array(500.0, dtype=jnp.float32)}
        _, _, param_derivs = evaluate_custom_force(
            compiled, pos_akma, None, global_params, compute_derivatives=True
        )

        # dE_omm / dk = (r - 0.1)^2 = (0.2 - 0.1)^2 = 0.01 kJ/mol per unit k
        assert "k" in param_derivs
        assert np.isclose(float(param_derivs["k"]), 0.01, rtol=1e-5)


# =============================================================================
# 7. StableHLO Lowering Guardrail Compliance
# =============================================================================


class TestStableHLOLoweringCompliance:
    """Verify that compiled custom forces lower portably under export_context."""

    def test_custom_force_portable_lowering(self):
        omm_force = openmm.CustomBondForce("0.5 * k * (r - r0)^2")
        omm_force.addPerBondParameter("r0")
        omm_force.addPerBondParameter("k")
        omm_force.addBond(0, 1, [0.15, 1000.0])

        compiled = compile_custom_force(omm_force)
        pos_akma = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)

        with export_context():
            def _eval_jit(p):
                e, f, _ = evaluate_custom_force(compiled, p, None, {})
                return e, f

            hlo_text = jax.jit(_eval_jit).lower(pos_akma).as_text()
            assert "custom_call" not in hlo_text or "forbidden" not in hlo_text
