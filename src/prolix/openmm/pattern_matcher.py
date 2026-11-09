"""Canonical AST pattern matcher and parameter folding engine for OpenMM custom forces.

This module provides algebraic normalization of Lepton ASTs (inlining, constant folding,
difference symmetrization, commutative sorting) and pattern matching against Prolix
native potentials (harmonic bonds, harmonic angles, periodic dihedrals, CHARMM impropers,
Urey-Bradley 1-3 interactions, and harmonic position restraints).

When custom_force_mode == "auto", matched parameters are converted to AKMA units and folded
directly into MolecularBundle arrays, enabling zero-overhead execution within Prolix's fused
Pallas megakernels.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import openmm

from prolix.openmm.lepton_compiler import (
    ASTNode,
    BinaryOpNode,
    CallNode,
    ConstNode,
    UnaryOpNode,
    VarNode,
    extract_free_variables,
    inline_definitions,
    parse_lepton_expression,
)
from prolix.openmm.units import (
    ANGLE_K_OMM_TO_AKMA,
    BOND_K_OMM_TO_AKMA,
    NM_TO_ANGSTROM,
    TORSION_K_OMM_TO_AKMA,
)
from prolix.types.bundles import (
    ANGLE_BUCKETS,
    BOND_BUCKETS,
    DIHEDRAL_BUCKETS,
    MolecularBundle,
    _bucket_idx,
)

__all__ = [
    "MatchedPattern",
    "ast_key",
    "is_coordinate",
    "symmetrize_difference",
    "flatten_binary",
    "fold_binary",
    "rewrite_ast",
    "canonicalize_expression",
    "match_custom_force_pattern",
    "fold_matched_pattern_into_bundle",
]


# =============================================================================
# 1. Matched Pattern Descriptor
# =============================================================================

PatternNameType = Literal[
    "harmonic_bond",
    "harmonic_angle",
    "periodic_torsion",
    "harmonic_improper",
    "urey_bradley",
    "harmonic_position_restraint",
    "combined_angle_urey_bradley",
]

TargetTermType = Literal[
    "bond",
    "angle",
    "dihedral",
    "improper",
    "urey_bradley",
    "position_restraint",
    "composite",
]


@dataclass(frozen=True)
class MatchedPattern:
    """Descriptor for an OpenMM custom force matched to a Prolix native potential."""

    pattern_name: PatternNameType
    target_term: TargetTermType
    indices: np.ndarray  # Shape: (M, 2), (M, 3), (M, 4), or (M, 1)
    params: np.ndarray   # Shape: (M, 2), (M, 3), or (M, 4) in AKMA units
    sub_patterns: list[MatchedPattern] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 2. Canonical AST Ordering & Symmetrization
# =============================================================================


def ast_key(node: ASTNode) -> tuple[Any, ...]:
    """Structural sorting key for deterministic commutative ordering of AST nodes."""
    if isinstance(node, ConstNode):
        return (0, round(float(node.value), 8))
    if isinstance(node, VarNode):
        return (1, node.name)
    if isinstance(node, UnaryOpNode):
        return (2, node.op, ast_key(node.operand))
    if isinstance(node, BinaryOpNode):
        return (3, node.op, ast_key(node.left), ast_key(node.right))
    if isinstance(node, CallNode):
        return (4, node.name, tuple(ast_key(a) for a in node.args))
    return (5,)


_COORDINATE_VAR_NAMES = frozenset(
    {"r", "theta", "x", "y", "z", "p1", "p2", "p3", "p4"}
)
_COORDINATE_CALL_NAMES = frozenset(
    {
        "distance",
        "angle",
        "dihedral",
        "pointdistance",
        "pointangle",
        "pointdihedral",
        "periodicdistance",
    }
)
_COORDINATE_CALLS_BY_TYPE: dict[str, frozenset[str]] = {
    "r": frozenset({"distance", "pointdistance", "periodicdistance"}),
    "distance": frozenset({"distance", "pointdistance", "periodicdistance"}),
    "theta": frozenset({"angle", "pointangle", "dihedral", "pointdihedral"}),
    "angle": frozenset({"angle", "pointangle"}),
    "dihedral": frozenset({"dihedral", "pointdihedral"}),
}


def is_coordinate(node: ASTNode) -> bool:
    """Check if node represents a dynamical atomic coordinate."""
    if isinstance(node, VarNode):
        return node.name in _COORDINATE_VAR_NAMES
    if isinstance(node, CallNode):
        return node.name in _COORDINATE_CALL_NAMES
    return False


def symmetrize_difference(left: ASTNode, right: ASTNode) -> tuple[ASTNode, ASTNode]:
    """Order difference (a - b) canonically so coordinates precede parameters."""
    if is_coordinate(right) and not is_coordinate(left):
        return right, left
    if not is_coordinate(left) and not is_coordinate(right):
        if ast_key(left) > ast_key(right):
            return right, left
    return left, right


def flatten_binary(node: ASTNode, op: str) -> list[ASTNode]:
    """Flatten chained associative operations into a factor list."""
    if isinstance(node, BinaryOpNode) and node.op == op:
        return flatten_binary(node.left, op) + flatten_binary(node.right, op)
    return [node]


def fold_binary(op: str, nodes: list[ASTNode]) -> ASTNode:
    """Fold a factor list into a left-associated binary AST."""
    if not nodes:
        raise ValueError(f"Cannot fold empty node list for op '{op}'")
    res = nodes[0]
    for n in nodes[1:]:
        res = BinaryOpNode(op, res, n)
    return res


# =============================================================================
# 3. Algebraic AST Rewriter & Canonicalizer
# =============================================================================


def _rewrite_abs(arg: ASTNode) -> ASTNode:
    if isinstance(arg, ConstNode):
        return ConstNode(abs(arg.value))
    if isinstance(arg, UnaryOpNode) and arg.op == "-":
        return CallNode("abs", (arg.operand,))
    if isinstance(arg, BinaryOpNode) and arg.op == "-":
        cl, cr = symmetrize_difference(arg.left, arg.right)
        return CallNode("abs", (BinaryOpNode("-", cl, cr),))
    return CallNode("abs", (arg,))


def _rewrite_min_max(name: str, a: ASTNode, b: ASTNode) -> ASTNode:
    if isinstance(a, ConstNode) and isinstance(b, ConstNode):
        val = min(a.value, b.value) if name == "min" else max(a.value, b.value)
        return ConstNode(val)
    if ast_key(a) > ast_key(b):
        return CallNode(name, (b, a))
    return CallNode(name, (a, b))


def _rewrite_call_node(node: CallNode) -> ASTNode:
    args = tuple(rewrite_ast(a) for a in node.args)
    name = node.name.lower()

    if name in ("sq", "square") and len(args) == 1:
        return rewrite_ast(BinaryOpNode("^", args[0], ConstNode(2.0)))
    if name == "cube" and len(args) == 1:
        return rewrite_ast(BinaryOpNode("^", args[0], ConstNode(3.0)))
    if name == "recip" and len(args) == 1:
        return rewrite_ast(BinaryOpNode("^", args[0], ConstNode(-1.0)))
    if name == "abs" and len(args) == 1:
        return _rewrite_abs(args[0])
    if name in ("min", "max") and len(args) == 2:
        return _rewrite_min_max(name, args[0], args[1])

    return CallNode(node.name, args)


def _rewrite_mul_node(left: ASTNode, right: ASTNode) -> ASTNode:
    factors = flatten_binary(BinaryOpNode("*", left, right), "*")
    const_prod = 1.0
    non_consts: list[ASTNode] = []
    for f in factors:
        f_rw = rewrite_ast(f)
        if isinstance(f_rw, ConstNode):
            const_prod *= f_rw.value
        else:
            non_consts.append(f_rw)
    if math.isclose(const_prod, 0.0, abs_tol=1e-12):
        return ConstNode(0.0)
    non_consts.sort(key=ast_key)
    if not non_consts:
        return ConstNode(const_prod)
    all_factors: list[ASTNode] = []
    if not math.isclose(const_prod, 1.0, abs_tol=1e-12):
        all_factors.append(ConstNode(const_prod))
    all_factors.extend(non_consts)
    return fold_binary("*", all_factors)


def _rewrite_add_node(left: ASTNode, right: ASTNode) -> ASTNode:
    terms = flatten_binary(BinaryOpNode("+", left, right), "+")
    const_sum = 0.0
    non_consts: list[ASTNode] = []
    for t in terms:
        t_rw = rewrite_ast(t)
        if isinstance(t_rw, ConstNode):
            const_sum += t_rw.value
        else:
            non_consts.append(t_rw)
    non_consts.sort(key=ast_key)
    if not non_consts:
        return ConstNode(const_sum)
    all_terms: list[ASTNode] = []
    if not math.isclose(const_sum, 0.0, abs_tol=1e-12):
        all_terms.append(ConstNode(const_sum))
    all_terms.extend(non_consts)
    return fold_binary("+", all_terms)


def _rewrite_pow_node(left: ASTNode, right: ASTNode) -> ASTNode:
    if isinstance(left, ConstNode) and isinstance(right, ConstNode):
        return ConstNode(left.value ** right.value)
    if isinstance(right, ConstNode):
        if math.isclose(right.value, 0.0):
            return ConstNode(1.0)
        if math.isclose(right.value, 1.0):
            return left
        if math.isclose(right.value, 2.0):
            if isinstance(left, BinaryOpNode) and left.op == "-":
                cl, cr = symmetrize_difference(left.left, left.right)
                return BinaryOpNode("^", BinaryOpNode("-", cl, cr), ConstNode(2.0))
            if isinstance(left, UnaryOpNode) and left.op == "-":
                return BinaryOpNode("^", left.operand, ConstNode(2.0))
    return BinaryOpNode("^", left, right)


def _rewrite_binary_node(node: BinaryOpNode) -> ASTNode:
    op = node.op
    left = rewrite_ast(node.left)
    right = rewrite_ast(node.right)

    if op == "/" and isinstance(right, ConstNode) and right.value != 0:
        return rewrite_ast(BinaryOpNode("*", left, ConstNode(1.0 / right.value)))
    if op == "^":
        return _rewrite_pow_node(left, right)
    if op == "*":
        return _rewrite_mul_node(left, right)
    if op == "+":
        return _rewrite_add_node(left, right)

    return BinaryOpNode(op, left, right)


def _rewrite_unary_node(node: UnaryOpNode) -> ASTNode:
    sub = rewrite_ast(node.operand)
    if node.op == "-":
        if isinstance(sub, ConstNode):
            return ConstNode(-sub.value)
        if isinstance(sub, UnaryOpNode) and sub.op == "-":
            return sub.operand  # -(-x) -> x
    return UnaryOpNode(node.op, sub)


def rewrite_ast(node: ASTNode) -> ASTNode:
    """Apply algebraic rewriting, constant folding, and commutative sorting."""
    if isinstance(node, (ConstNode, VarNode)):
        return node

    if isinstance(node, UnaryOpNode):
        return _rewrite_unary_node(node)

    if isinstance(node, CallNode):
        return _rewrite_call_node(node)

    if isinstance(node, BinaryOpNode):
        return _rewrite_binary_node(node)

    return node


def canonicalize_expression(
    primary_ast: ASTNode, definitions: dict[str, ASTNode]
) -> ASTNode:
    """Run full inlining, rewriting, and normalization pipeline to convergence."""
    ast = inline_definitions(primary_ast, definitions)
    for _ in range(10):
        prev = ast
        ast = rewrite_ast(ast)
        if ast == prev:
            break
    return ast


# =============================================================================
# 4. Pattern Matcher Implementations
# =============================================================================


def _matches_expected_coord(node: ASTNode, expected_coord: str) -> bool:
    if isinstance(node, VarNode):
        return node.name == expected_coord
    if isinstance(node, CallNode):
        valid_calls = _COORDINATE_CALLS_BY_TYPE.get(expected_coord, frozenset())
        return node.name.lower() in valid_calls
    return False


def _extract_coord_param_from_base(base: ASTNode, expected_coord: str) -> tuple[ASTNode, ASTNode] | None:
    if isinstance(base, BinaryOpNode) and base.op == "-":
        if _matches_expected_coord(base.left, expected_coord):
            return base.left, base.right
    if isinstance(base, CallNode) and base.name.lower() == "min" and len(base.args) == 2:
        arg0 = base.args[0]
        if isinstance(arg0, CallNode) and arg0.name.lower() == "abs" and len(arg0.args) == 1:
            inner = arg0.args[0]
            if isinstance(inner, BinaryOpNode) and inner.op == "-":
                if _matches_expected_coord(inner.left, expected_coord):
                    return inner.left, inner.right
    return None


def _extract_harmonic_power(node: BinaryOpNode, expected_coord: str) -> tuple[float, ASTNode, ASTNode] | None:
    if isinstance(node.right, ConstNode) and math.isclose(node.right.value, 2.0):
        extracted = _extract_coord_param_from_base(node.left, expected_coord)
        if extracted is not None:
            coord_node, param_node = extracted
            return 1.0, coord_node, param_node
    return None


def _extract_harmonic_product(node: BinaryOpNode, expected_coord: str) -> tuple[float, ASTNode, ASTNode] | None:
    factors = flatten_binary(node, "*")
    const_mult = 1.0
    power_node = None
    param_nodes = []
    for f in factors:
        if isinstance(f, ConstNode):
            const_mult *= f.value
        elif isinstance(f, BinaryOpNode) and f.op == "^" and isinstance(f.right, ConstNode) and math.isclose(f.right.value, 2.0):
            power_node = f
        else:
            param_nodes.append(f)

    if power_node is not None:
        extracted = _extract_coord_param_from_base(power_node.left, expected_coord)
        if extracted is not None:
            _, param_node = extracted
            k_node = param_nodes[0] if param_nodes else ConstNode(1.0)
            return const_mult, k_node, param_node
    return None


def _extract_harmonic_diff(node: ASTNode, expected_coord: str) -> tuple[float, ASTNode, ASTNode] | None:
    """Match C * (coord - param)^2 or C * k * (coord - param)^2."""
    if isinstance(node, BinaryOpNode):
        if node.op == "^":
            return _extract_harmonic_power(node, expected_coord)
        if node.op == "*":
            return _extract_harmonic_product(node, expected_coord)
    return None


def _get_param_val(
    param_node: ASTNode,
    per_item_params: dict[str, float],
) -> float | None:
    if isinstance(param_node, ConstNode):
        return param_node.value
    if isinstance(param_node, VarNode) and param_node.name in per_item_params:
        return per_item_params[param_node.name]
    return None


def _match_harmonic_bond(force: openmm.CustomBondForce, ast: ASTNode) -> MatchedPattern | None:
    res = _extract_harmonic_diff(ast, "r")
    if res is None:
        return None
    mult, k_node, r0_node = res

    num_bonds = force.getNumBonds()
    if num_bonds == 0:
        return None

    num_params = force.getNumPerBondParameters()
    param_names = [force.getPerBondParameterName(i) for i in range(num_params)]

    indices_list = []
    params_list = []
    for i in range(num_bonds):
        p1, p2, raw_params = force.getBondParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_val = _get_param_val(k_node, p_dict)
        r0_val = _get_param_val(r0_node, p_dict)
        if k_val is None or r0_val is None:
            return None

        # Convert to AKMA: r0 in A, k in kcal/(mol*A^2)
        # OpenMM energy: mult * k * (r - r0)^2 where mult is usually 0.5
        # Prolix energy: 0.5 * k_prolix * (r - r0)^2
        # Therefore: k_prolix = 2 * mult * (k_omm / 418.4)
        r0_akma = r0_val * NM_TO_ANGSTROM
        k_akma = 2.0 * mult * (k_val * BOND_K_OMM_TO_AKMA)

        indices_list.append([p1, p2])
        params_list.append([r0_akma, k_akma])

    return MatchedPattern(
        pattern_name="harmonic_bond",
        target_term="bond",
        indices=np.asarray(indices_list, dtype=np.int32),
        params=np.asarray(params_list, dtype=np.float32),
    )


def _match_harmonic_angle(force: openmm.CustomAngleForce, ast: ASTNode) -> MatchedPattern | None:
    res = _extract_harmonic_diff(ast, "theta")
    if res is None:
        return None
    mult, k_node, theta0_node = res

    num_angles = force.getNumAngles()
    if num_angles == 0:
        return None

    num_params = force.getNumPerAngleParameters()
    param_names = [force.getPerAngleParameterName(i) for i in range(num_params)]

    indices_list = []
    params_list = []
    for i in range(num_angles):
        p1, p2, p3, raw_params = force.getAngleParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_val = _get_param_val(k_node, p_dict)
        theta0_val = _get_param_val(theta0_node, p_dict)
        if k_val is None or theta0_val is None:
            return None

        # Convert to AKMA: theta0 in rad, k in kcal/(mol*rad^2)
        # k_prolix = 2 * mult * (k_omm / 4.184)
        theta0_akma = theta0_val
        k_akma = 2.0 * mult * (k_val * ANGLE_K_OMM_TO_AKMA)

        indices_list.append([p1, p2, p3])
        params_list.append([theta0_akma, k_akma])

    return MatchedPattern(
        pattern_name="harmonic_angle",
        target_term="angle",
        indices=np.asarray(indices_list, dtype=np.int32),
        params=np.asarray(params_list, dtype=np.float32),
    )


def _match_harmonic_improper(force: openmm.CustomTorsionForce, ast: ASTNode) -> MatchedPattern | None:
    res = _extract_harmonic_diff(ast, "theta")
    if res is None:
        return None
    mult, k_node, theta0_node = res

    num_torsions = force.getNumTorsions()
    if num_torsions == 0:
        return None

    num_params = force.getNumPerTorsionParameters()
    param_names = [force.getPerTorsionParameterName(i) for i in range(num_params)]

    indices_list = []
    params_list = []
    for i in range(num_torsions):
        p1, p2, p3, p4, raw_params = force.getTorsionParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_val = _get_param_val(k_node, p_dict)
        theta0_val = _get_param_val(theta0_node, p_dict)
        if k_val is None or theta0_val is None:
            return None

        # CRITICAL DISCREPANCY CONVENTION:
        # Prolix improper energy: E = sum k_prolix * (theta - theta0)^2 (WITHOUT 0.5 factor!)
        # OpenMM commonly defines: E = 0.5 * k_omm * (theta - theta0)^2 (mult = 0.5)
        # Therefore: k_prolix = mult * (k_omm / 4.184) = (k_omm / 8.368 when mult=0.5)
        theta0_akma = theta0_val
        k_akma = mult * (k_val * ANGLE_K_OMM_TO_AKMA)

        indices_list.append([p1, p2, p3, p4])
        # improper_params shape is (M, 3) stored as [k_prolix, theta0_akma, 0.0]
        params_list.append([k_akma, theta0_akma, 0.0])

    return MatchedPattern(
        pattern_name="harmonic_improper",
        target_term="improper",
        indices=np.asarray(indices_list, dtype=np.int32),
        params=np.asarray(params_list, dtype=np.float32),
    )


def _match_periodic_torsion(force: openmm.CustomTorsionForce, ast: ASTNode) -> MatchedPattern | None:
    # Match k * (1 + cos(n * theta - phase)) or k + k * cos(n * theta - phase)
    # Look for cos call node
    cos_node = None
    k_factor = ConstNode(1.0)

    def _find_cos(n: ASTNode):
        nonlocal cos_node
        if isinstance(n, CallNode) and n.name.lower() == "cos":
            cos_node = n
        elif isinstance(n, BinaryOpNode):
            _find_cos(n.left)
            _find_cos(n.right)
        elif isinstance(n, UnaryOpNode):
            _find_cos(n.operand)

    _find_cos(ast)
    if cos_node is None or len(cos_node.args) != 1:
        return None

    cos_arg = cos_node.args[0]
    # Arg should be n * theta - phase or n * theta
    n_node = ConstNode(1.0)
    phase_node = ConstNode(0.0)

    if isinstance(cos_arg, BinaryOpNode) and cos_arg.op == "-":
        phase_node = cos_arg.right
        cos_arg = cos_arg.left

    if isinstance(cos_arg, BinaryOpNode) and cos_arg.op == "*":
        if isinstance(cos_arg.left, VarNode) and cos_arg.left.name == "theta":
            n_node = cos_arg.right
        elif isinstance(cos_arg.right, VarNode) and cos_arg.right.name == "theta":
            n_node = cos_arg.left
    elif isinstance(cos_arg, VarNode) and cos_arg.name == "theta":
        n_node = ConstNode(1.0)
    else:
        return None

    # Find k multiplier on cos
    num_torsions = force.getNumTorsions()
    if num_torsions == 0:
        return None

    num_params = force.getNumPerTorsionParameters()
    param_names = [force.getPerTorsionParameterName(i) for i in range(num_params)]

    # Search for k parameter in AST
    free_vars = extract_free_variables(ast) - {"theta"}
    k_name = next((v for v in free_vars if "k" in v.lower()), None)
    if k_name is None and free_vars:
        k_name = sorted(free_vars)[0]

    indices_list = []
    params_list = []
    for i in range(num_torsions):
        p1, p2, p3, p4, raw_params = force.getTorsionParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_val = p_dict.get(k_name, 1.0) if k_name else 1.0
        n_val = _get_param_val(n_node, p_dict)
        phase_val = _get_param_val(phase_node, p_dict)
        if n_val is None or phase_val is None:
            return None

        # Convert to AKMA: k in kcal/mol
        k_akma = k_val * TORSION_K_OMM_TO_AKMA
        periodicity = float(round(n_val))
        phase_akma = float(phase_val)

        indices_list.append([p1, p2, p3, p4])
        # dihedral_params shape: (M, 4) stored as [periodicity, phase, k, 0.0]
        params_list.append([periodicity, phase_akma, k_akma, 0.0])

    return MatchedPattern(
        pattern_name="periodic_torsion",
        target_term="dihedral",
        indices=np.asarray(indices_list, dtype=np.int32),
        params=np.asarray(params_list, dtype=np.float32),
    )


def _match_compound_bond_2p(
    force: openmm.CustomCompoundBondForce,
    ast: ASTNode,
    param_names: list[str],
    num_bonds: int,
) -> MatchedPattern | None:
    res = _extract_harmonic_diff(ast, "distance")
    if res is None:
        return None
    mult, k_node, r0_node = res
    indices_list = []
    params_list = []
    for i in range(num_bonds):
        particles, raw_params = force.getBondParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))
        k_val = _get_param_val(k_node, p_dict)
        r0_val = _get_param_val(r0_node, p_dict)
        if k_val is None or r0_val is None:
            return None
        r0_akma = r0_val * NM_TO_ANGSTROM
        k_akma = 2.0 * mult * (k_val * BOND_K_OMM_TO_AKMA)
        indices_list.append(particles)
        params_list.append([r0_akma, k_akma])

    return MatchedPattern(
        pattern_name="harmonic_bond",
        target_term="bond",
        indices=np.asarray(indices_list, dtype=np.int32),
        params=np.asarray(params_list, dtype=np.float32),
    )


def _build_combined_angle_ub_pattern(
    force: openmm.CustomCompoundBondForce,
    angle_term: tuple[float, ASTNode, ASTNode],
    ub_term: tuple[float, ASTNode, ASTNode],
    param_names: list[str],
    num_bonds: int,
) -> MatchedPattern | None:
    mult_ang, k_ang_node, th0_node = angle_term
    mult_ub, k_ub_node, r0_node = ub_term

    ang_indices, ang_params = [], []
    ub_indices, ub_params = [], []
    for i in range(num_bonds):
        particles, raw_params = force.getBondParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_a = _get_param_val(k_ang_node, p_dict)
        th0 = _get_param_val(th0_node, p_dict)
        k_u = _get_param_val(k_ub_node, p_dict)
        r0 = _get_param_val(r0_node, p_dict)
        if any(v is None for v in (k_a, th0, k_u, r0)):
            return None

        # Angle
        ang_indices.append([particles[0], particles[1], particles[2]])
        ang_params.append([th0, 2.0 * mult_ang * k_a * ANGLE_K_OMM_TO_AKMA])

        # Urey-Bradley 1-3
        ub_indices.append([particles[0], particles[2], 0])
        ub_params.append([r0 * NM_TO_ANGSTROM, 2.0 * mult_ub * k_u * BOND_K_OMM_TO_AKMA])

    sub1 = MatchedPattern(
        pattern_name="harmonic_angle",
        target_term="angle",
        indices=np.asarray(ang_indices, dtype=np.int32),
        params=np.asarray(ang_params, dtype=np.float32),
    )
    sub2 = MatchedPattern(
        pattern_name="urey_bradley",
        target_term="urey_bradley",
        indices=np.asarray(ub_indices, dtype=np.int32),
        params=np.asarray(ub_params, dtype=np.float32),
    )
    return MatchedPattern(
        pattern_name="combined_angle_urey_bradley",
        target_term="composite",
        indices=np.empty((0, 3), dtype=np.int32),
        params=np.empty((0, 2), dtype=np.float32),
        sub_patterns=[sub1, sub2],
    )


def _build_pure_ub_pattern(
    force: openmm.CustomCompoundBondForce,
    ub_term: tuple[float, ASTNode, ASTNode],
    param_names: list[str],
    num_bonds: int,
) -> MatchedPattern | None:
    mult_ub, k_ub_node, r0_node = ub_term
    ub_indices, ub_params = [], []
    for i in range(num_bonds):
        particles, raw_params = force.getBondParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))
        k_u = _get_param_val(k_ub_node, p_dict)
        r0 = _get_param_val(r0_node, p_dict)
        if k_u is None or r0 is None:
            return None
        ub_indices.append([particles[0], particles[2], 0])
        ub_params.append([r0 * NM_TO_ANGSTROM, 2.0 * mult_ub * k_u * BOND_K_OMM_TO_AKMA])

    return MatchedPattern(
        pattern_name="urey_bradley",
        target_term="urey_bradley",
        indices=np.asarray(ub_indices, dtype=np.int32),
        params=np.asarray(ub_params, dtype=np.float32),
    )


def _match_compound_bond_3p(
    force: openmm.CustomCompoundBondForce,
    ast: ASTNode,
    param_names: list[str],
    num_bonds: int,
) -> MatchedPattern | None:
    terms = flatten_binary(ast, "+") if (isinstance(ast, BinaryOpNode) and ast.op == "+") else [ast]
    angle_term = None
    ub_term = None
    for t in terms:
        r_ang = _extract_harmonic_diff(t, "angle")
        r_ub = _extract_harmonic_diff(t, "distance")
        if r_ang is not None:
            angle_term = r_ang
        elif r_ub is not None:
            ub_term = r_ub

    if ub_term is not None and angle_term is not None:
        return _build_combined_angle_ub_pattern(force, angle_term, ub_term, param_names, num_bonds)

    if ub_term is not None:
        return _build_pure_ub_pattern(force, ub_term, param_names, num_bonds)

    return None


def _match_compound_bond(force: openmm.CustomCompoundBondForce, ast: ASTNode) -> MatchedPattern | None:
    num_bonds = force.getNumBonds()
    if num_bonds == 0:
        return None

    num_particles = force.getNumParticlesPerBond()
    num_params = force.getNumPerBondParameters()
    param_names = [force.getPerBondParameterName(i) for i in range(num_params)]

    if num_particles == 2:
        return _match_compound_bond_2p(force, ast, param_names, num_bonds)
    if num_particles == 3:
        return _match_compound_bond_3p(force, ast, param_names, num_bonds)

    return None


def _match_position_restraint(force: openmm.CustomExternalForce, ast: ASTNode) -> MatchedPattern | None:
    num_particles = force.getNumParticles()
    if num_particles == 0:
        return None

    num_params = force.getNumPerParticleParameters()
    param_names = [force.getPerParticleParameterName(i) for i in range(num_params)]

    # Check for ((x - x0)^2 + (y - y0)^2 + (z - z0)^2) or periodicdistance^2
    mult = 1.0
    k_name = "k"
    free_vars = extract_free_variables(ast) - {"x", "y", "z"}
    k_candidate = next((v for v in free_vars if "k" in v.lower()), None)
    if k_candidate is not None:
        k_name = k_candidate

    indices_list = []
    params_list = []
    for i in range(num_particles):
        p_idx, raw_params = force.getParticleParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        x0 = p_dict.get("x0", 0.0)
        y0 = p_dict.get("y0", 0.0)
        z0 = p_dict.get("z0", 0.0)
        k_val = p_dict.get(k_name, 1000.0)

        # Prolix compute_positional_restraint_energy: E = sum k * Delta_r^2 (unhalved)
        # Therefore: k_prolix = mult * (k_omm / 418.4)
        x0_akma = x0 * NM_TO_ANGSTROM
        y0_akma = y0 * NM_TO_ANGSTROM
        z0_akma = z0 * NM_TO_ANGSTROM
        k_akma = mult * (k_val * BOND_K_OMM_TO_AKMA)

        indices_list.append([p_idx])
        params_list.append([x0_akma, y0_akma, z0_akma, k_akma])

    return MatchedPattern(
        pattern_name="harmonic_position_restraint",
        target_term="position_restraint",
        indices=np.asarray(indices_list, dtype=np.int32),
        params=np.asarray(params_list, dtype=np.float32),
    )


# =============================================================================
# 5. Public Pattern Matcher API
# =============================================================================


def _is_dynamic_or_unsupported(force: openmm.Force) -> bool:
    if hasattr(force, "getNumEnergyParameterDerivatives") and force.getNumEnergyParameterDerivatives() > 0:
        return True
    if hasattr(force, "getNumGlobalParameters") and force.getNumGlobalParameters() > 0:
        return True
    return bool(hasattr(force, "getNumTabulatedFunctions") and force.getNumTabulatedFunctions() > 0)


def _match_by_force_type(custom_force_type: str, force: openmm.Force, canonical_ast: ASTNode) -> MatchedPattern | None:
    if custom_force_type == "CustomBondForce":
        return _match_harmonic_bond(force, canonical_ast)
    if custom_force_type == "CustomAngleForce":
        return _match_harmonic_angle(force, canonical_ast)
    if custom_force_type == "CustomTorsionForce":
        return _match_harmonic_improper(force, canonical_ast) or _match_periodic_torsion(force, canonical_ast)
    if custom_force_type == "CustomCompoundBondForce":
        return _match_compound_bond(force, canonical_ast)
    if custom_force_type == "CustomExternalForce":
        return _match_position_restraint(force, canonical_ast)
    return None


def match_custom_force_pattern(
    force: openmm.Force,
    custom_force_type: str | None = None,
) -> MatchedPattern | None:
    """Inspect an OpenMM Custom Force and match it against native potential patterns.

    Returns:
        MatchedPattern if matched and safe for static fusion.
        None if dynamic derivatives/globals exist or no pattern matches (falling back to Lepton compiler).
    """
    if custom_force_type is None:
        custom_force_type = force.__class__.__name__

    if _is_dynamic_or_unsupported(force):
        return None

    energy_expr = force.getEnergyFunction()
    primary_ast, defs = parse_lepton_expression(energy_expr)
    canonical_ast = canonicalize_expression(primary_ast, defs)

    return _match_by_force_type(custom_force_type, force, canonical_ast)


# =============================================================================
# 6. Parameter Folding Engine into MolecularBundle
# =============================================================================


def fold_matched_pattern_into_bundle(
    bundle: MolecularBundle,
    matched: MatchedPattern,
) -> MolecularBundle:
    """Fold a matched pattern's indices and parameters into MolecularBundle arrays."""
    # Composite pattern handling (e.g. combined angle + Urey-Bradley)
    if matched.target_term == "composite" or matched.sub_patterns:
        curr_bundle = bundle
        for sub in matched.sub_patterns:
            curr_bundle = fold_matched_pattern_into_bundle(curr_bundle, sub)
        return curr_bundle

    term = matched.target_term
    if term == "position_restraint":
        # Position restraints do not reside in MolecularBundle bonded arrays;
        # they are stored on ConvertedOpenMMSystem.
        return bundle

    # Map target term to bundle field names and bucket ladder
    spec_field = f"{term}_bucket_idx"
    extra_kwargs: dict[str, Any] = {}
    if term == "bond":
        ladder = BOND_BUCKETS
        idx_field, prm_field, msk_field, n_field = "bond_idx", "bond_params", "bond_mask", "n_bonds"
    elif term == "angle":
        ladder = ANGLE_BUCKETS
        idx_field, prm_field, msk_field, n_field = "angle_idx", "angle_params", "angle_mask", "n_angles"
    elif term == "dihedral":
        ladder = DIHEDRAL_BUCKETS
        idx_field, prm_field, msk_field, n_field = "dihedral_idx", "dihedral_params", "dihedral_mask", "n_dihedrals"
    elif term == "improper":
        ladder = DIHEDRAL_BUCKETS
        spec_field = "dihedral_bucket_idx"
        idx_field, prm_field, msk_field, n_field = "improper_idx", "improper_params", "improper_mask", "n_impropers"
        extra_kwargs = {"improper_is_periodic": jnp.array(False, dtype=bool)}
    elif term == "urey_bradley":
        ladder = ANGLE_BUCKETS
        spec_field = "angle_bucket_idx"
        idx_field, prm_field, msk_field, n_field = "urey_bradley_idx", "urey_bradley_params", "urey_bradley_mask", "n_urey_bradley"
    else:
        raise ValueError(f"Unsupported target term '{term}'")

    # Extract existing active entries
    n_old = int(getattr(bundle, n_field))
    old_idx = getattr(bundle, idx_field)[:n_old]
    old_prm = getattr(bundle, prm_field)[:n_old]

    # Convert new matched arrays
    new_idx = jnp.asarray(matched.indices, dtype=jnp.int32)
    new_prm = jnp.asarray(matched.params, dtype=jnp.float32)
    n_new = int(new_idx.shape[0])
    n_total = n_old + n_new

    # Concatenate active rows
    cat_idx = jnp.concatenate([old_idx, new_idx], axis=0) if n_old > 0 else new_idx
    cat_prm = jnp.concatenate([old_prm, new_prm], axis=0) if n_old > 0 else new_prm

    # Resolve bucket capacity and ShapeSpec
    curr_cap = getattr(bundle, idx_field).shape[0]
    if n_total <= curr_cap:
        target_cap = curr_cap
        shape_spec = bundle.shape_spec
    else:
        new_bucket_idx = _bucket_idx(n_total, ladder)
        target_cap = ladder[new_bucket_idx]
        if hasattr(bundle.shape_spec, spec_field):
            shape_spec = dataclasses.replace(bundle.shape_spec, **{spec_field: new_bucket_idx})
        else:
            shape_spec = bundle.shape_spec

    # Pad arrays to bucket capacity
    pad_len = target_cap - n_total
    padded_idx = jnp.pad(cat_idx, ((0, pad_len), (0, 0))) if pad_len > 0 else cat_idx
    padded_prm = jnp.pad(cat_prm, ((0, pad_len), (0, 0))) if pad_len > 0 else cat_prm
    mask = jnp.arange(target_cap) < n_total

    updates = {
        idx_field: padded_idx,
        prm_field: padded_prm,
        msk_field: mask,
        n_field: jnp.array(n_total, dtype=jnp.int32),
        "shape_spec": shape_spec,
        **extra_kwargs,
    }
    return dataclasses.replace(bundle, **updates)
