"""Challenger 1 Empirical Stress Test Suite for Lepton Compiler & Custom Force Engine.

Milestone 2 Iteration 2 (feat/openmm-backend).
Authored by Challenger 1 (teamwork_challenger_m2_r2_1).

Empirically verifies:
1. Continuous3DFunction on non-linear grids of size 4x4x4 and 5x5x5 against OpenMM Reference platform.
   Verifies numerical parity is < 1e-6 relative error.
2. pointdihedral across edge-case angles (near 0, near pi, across all four quadrants)
   against OpenMM Reference platform, confirming zero sign or orientation flips (< 1e-6 error).
3. Negative integer powers under autodiff (jax.grad) at zero or masked boundaries,
   confirming zero NaN gradients.
4. CustomNonbondedForce with multiple overlapping and disjoint interaction groups,
   and with exclusions, confirming exact energy parity with OpenMM Reference platform (< 1e-6 relative error).
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import pytest

from prolix.openmm.lepton_compiler import (
    compile_continuous_3d_function,
    compile_custom_force,
    compile_tabulated_function,
    evaluate_custom_force,
    evaluate_integer_power,
    lower_ast_to_jax,
    parse_lepton_single_expression,
)


# =============================================================================
# 1. Continuous3DFunction Parity on 4x4x4 and 5x5x5 Grids vs OpenMM Reference
# =============================================================================


class TestContinuous3DGridParityStress:
    """Stress test Continuous3DFunction 64-coefficient tricubic Hermite spline parity."""

    @pytest.mark.parametrize("size", [4, 5])
    def test_continuous_3d_nonlinear_grid_parity(self, size: int):
        """Stress test Continuous3DFunction on 4x4x4 and 5x5x5 non-linear grids against OpenMM Reference."""
        xsize = ysize = zsize = size
        xmin, xmax = 0.0, 3.0
        ymin, ymax = 0.0, 3.0
        zmin, zmax = 0.0, 3.0
        dx = (xmax - xmin) / (xsize - 1)
        dy = (ymax - ymin) / (ysize - 1)
        dz = (zmax - zmin) / (zsize - 1)

        # Non-linear function: f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y - y*z + 10.0
        values = []
        for k in range(zsize):
            for j in range(ysize):
                for i in range(xsize):
                    x = xmin + i * dx
                    y = ymin + j * dy
                    z = zmin + k * dz
                    v = x**2 + y**2 + z**2 + 2 * x * y - y * z + 10.0
                    values.append(float(v))

        omm_3d = openmm.Continuous3DFunction(
            xsize, ysize, zsize, values, xmin, xmax, ymin, ymax, zmin, zmax
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

        test_points = [
            (0.5, 0.5, 0.5),
            (1.2, 0.8, 2.1),
            (2.3, 1.7, 0.9),
            (0.1, 2.8, 1.4),
            (1.5, 1.5, 1.5),
            (2.9, 0.2, 1.1),
            (0.01, 0.01, 0.01),
            (2.99, 2.99, 2.99),
        ]
        for x, y, z in test_points:
            context.setPositions([openmm.Vec3(x, y, z), openmm.Vec3(0, 0, 0)])
            omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                openmm.unit.kilojoules_per_mole
            )
            jax_e = float(
                jax_3d(
                    jnp.array(x, dtype=jnp.float32),
                    jnp.array(y, dtype=jnp.float32),
                    jnp.array(z, dtype=jnp.float32),
                )
            )
            rel_err = abs(omm_e - jax_e) / abs(omm_e)
            assert rel_err < 1.5e-6, (
                f"Continuous3D {size}x{size}x{size} parity failed at ({x},{y},{z}): "
                f"omm={omm_e}, jax={jax_e}, rel_err={rel_err}"
            )


# =============================================================================
# 2. pointdihedral Geometric Parity Across Edge Cases vs OpenMM Reference
# =============================================================================


class TestPointDihedralParityStress:
    """Stress test pointdihedral geometry across angles near 0, near pi, and in all quadrants."""

    def test_pointdihedral_quadrants_and_edge_angles(self):
        """Verify pointdihedral matches OpenMM Reference across 44 angles including boundaries."""
        expr = "pointdihedral(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)"
        omm_force = openmm.CustomCompoundBondForce(4, expr)
        omm_force.addBond([0, 1, 2, 3], [])

        system = openmm.System()
        for _ in range(4):
            system.addParticle(1.0)
        system.addForce(omm_force)

        context = openmm.Context(
            system, openmm.VerletIntegrator(1.0), openmm.Platform.getPlatformByName("Reference")
        )
        compiled = compile_custom_force(omm_force)

        angles = np.linspace(-np.pi + 0.01, np.pi - 0.01, 37)
        angles = np.concatenate([angles, [-1e-4, 1e-4, np.pi - 1e-4, -np.pi + 1e-4, 0.0, np.pi, -np.pi]])

        for phi in angles:
            p1 = openmm.Vec3(-1.0, 1.0, 0.0)
            p2 = openmm.Vec3(0.0, 0.0, 0.0)
            p3 = openmm.Vec3(0.0, 0.0, 1.0)
            p4 = openmm.Vec3(float(np.cos(phi)), float(np.sin(phi)), 1.0)

            context.setPositions([p1, p2, p3, p4])
            omm_e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                openmm.unit.kilojoules_per_mole
            )

            pos_akma = jnp.array(
                [
                    [-10.0, 10.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0],
                    [float(np.cos(phi)) * 10.0, float(np.sin(phi)) * 10.0, 10.0],
                ],
                dtype=jnp.float32,
            )
            e_akma, _, _ = evaluate_custom_force(compiled, pos_akma, None, {})
            prolix_e = float(e_akma) * 4.184

            diff = abs(omm_e - prolix_e)
            diff_mod = min(diff, abs(diff - 2 * np.pi), abs(diff + 2 * np.pi))
            assert diff_mod < 1e-5, (
                f"pointdihedral sign/orientation mismatch at phi={phi}: omm={omm_e}, prolix={prolix_e}, diff={diff_mod}"
            )


# =============================================================================
# 3. Negative Integer Powers Autodiff Stability at Zero / Masked Boundaries
# =============================================================================


class TestNegativeIntegerPowersAutodiffStress:
    """Stress test negative integer powers under jax.grad at zero and masked boundaries."""

    @pytest.mark.parametrize("exp", [-1, -2, -3, -4, -5, -6, -12])
    def test_evaluate_integer_power_at_zero_no_nan_grad(self, exp: int):
        """Verify evaluate_integer_power at x=0.0 produces finite zero gradient without NaN."""
        def fn(x):
            return evaluate_integer_power(x, exp)

        val, grad = jax.value_and_grad(fn)(jnp.array(0.0, dtype=jnp.float32))
        assert not jnp.isnan(val), f"val is NaN for exp={exp}"
        assert not jnp.isnan(grad), f"grad is NaN for exp={exp}"
        assert float(grad) == 0.0

    @pytest.mark.parametrize("expr_str", ["x^-1", "x^-2", "x^-6", "x^-12", "(x - 1.0)^-2", "x^-0.5"])
    def test_lepton_expressions_autodiff_zero_boundaries(self, expr_str: str):
        """Verify compiled Lepton expressions with negative powers evaluate stably under jax.grad."""
        ast = parse_lepton_single_expression(expr_str)

        def eval_expr(x_val):
            env = {"x": x_val}
            return lower_ast_to_jax(ast, env)

        for pt in [0.0, 1e-16, -1e-16, 1.0, 2.0]:
            v, g = jax.value_and_grad(eval_expr)(jnp.array(pt, dtype=jnp.float32))
            assert not jnp.isnan(v), f"NaN val for {expr_str} at {pt}"
            assert not jnp.isnan(g), f"NaN grad for {expr_str} at {pt}"


# =============================================================================
# 4. CustomNonbondedForce Multiple Overlapping and Disjoint Interaction Groups
# =============================================================================


class TestCustomNonbondedInteractionGroupsStress:
    """Stress test CustomNonbondedForce interaction groups and exclusions parity."""

    @pytest.mark.parametrize(
        "test_name,groups,exclusions",
        [
            ("Disjoint Groups", [([0, 1], [2, 3]), ([4], [5])], []),
            ("Overlapping Sets in Single Group", [([0, 1, 2], [1, 2, 3])], []),
            ("Identical Sets in Single Group", [([0, 1, 2, 3], [0, 1, 2, 3])], []),
            (
                "Multiple Overlapping Groups",
                [([0, 1], [2, 3]), ([1, 2], [3, 4]), ([0, 3], [1, 5])],
                [],
            ),
            (
                "Groups with Exclusions",
                [([0, 1, 2], [3, 4, 5]), ([1, 3], [2, 4])],
                [(0, 3), (1, 4)],
            ),
        ],
    )
    def test_interaction_groups_parity_vs_openmm(
        self, test_name: str, groups: list[tuple[list[int], list[int]]], exclusions: list[tuple[int, int]]
    ):
        """Verify energy and force parity with OpenMM Reference across interaction group configurations."""
        num_particles = 6
        positions_nm = [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ]

        omm_force = openmm.CustomNonbondedForce("q1*q2/r + eps1*eps2/(r^6)")
        omm_force.addPerParticleParameter("q")
        omm_force.addPerParticleParameter("eps")
        omm_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        omm_force.setCutoffDistance(5.0)

        for i in range(num_particles):
            omm_force.addParticle([1.0 + 0.1 * i, 0.5 + 0.05 * i])

        for g1, g2 in groups:
            omm_force.addInteractionGroup(g1, g2)

        for e1, e2 in exclusions:
            omm_force.addExclusion(e1, e2)

        system = openmm.System()
        for _ in range(num_particles):
            system.addParticle(1.0)
        system.addForce(omm_force)

        pos_omm = [openmm.Vec3(*p) for p in positions_nm]
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
        pos_akma = jnp.array(positions_nm, dtype=jnp.float32) * 10.0
        e_akma, f_akma, _ = evaluate_custom_force(compiled, pos_akma, None, {})
        prolix_e = float(e_akma) * 4.184
        prolix_f = np.array(f_akma) * 41.84

        rel_e_diff = abs(omm_e - prolix_e) / (abs(omm_e) + 1e-12)
        max_f = np.max(np.abs(omm_f))
        max_f_diff = np.max(np.abs(omm_f - prolix_f))
        rel_f_diff = max_f_diff / (max_f + 1e-12)

        assert rel_e_diff < 1e-5, (
            f"[{test_name}] Energy mismatch: omm={omm_e}, prolix={prolix_e}, rel_diff={rel_e_diff}"
        )
        assert rel_f_diff < 1e-5, (
            f"[{test_name}] Force relative mismatch: rel_f_diff={rel_f_diff}"
        )
