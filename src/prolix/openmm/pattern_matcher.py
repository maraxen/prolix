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
            _, param_node = extracted
            # Return ConstNode(1.0) as k_node so unscaled differences (r - r0)^2 match with k = 1.0
            return 1.0, ConstNode(1.0), param_node
    return None


def _extract_harmonic_product(node: BinaryOpNode, expected_coord: str) -> tuple[float, ASTNode, ASTNode] | None:
    factors = flatten_binary(node, "*")
    const_mult = 1.0
    power_nodes: list[BinaryOpNode] = []
    param_nodes: list[ASTNode] = []
    for f in factors:
        if isinstance(f, ConstNode):
            const_mult *= f.value
        elif (
            isinstance(f, BinaryOpNode)
            and f.op == "^"
            and isinstance(f.right, ConstNode)
            and math.isclose(f.right.value, 2.0)
        ):
            power_nodes.append(f)
        else:
            param_nodes.append(f)

    # Strictly enforce: exactly ONE squared power term, and at most ONE variable multiplier
    if len(power_nodes) != 1 or len(param_nodes) > 1:
        return None

    power_node = power_nodes[0]
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


def _extract_periodic_cos_call(ast: ASTNode) -> tuple[CallNode, float, ASTNode] | None:
    """Extract (cos_call, const_mult, k_node) from factored or expanded periodic torsion AST."""
    # --- Pattern Form 1: Factored Product Form: C * k * (1 + cos(n * theta - phase)) ---
    factors = flatten_binary(ast, "*")
    plus_node: BinaryOpNode | None = None
    param_nodes: list[ASTNode] = []
    c_prod = 1.0

    for f in factors:
        if isinstance(f, ConstNode):
            c_prod *= f.value
        elif isinstance(f, BinaryOpNode) and f.op == "+":
            plus_node = f
        else:
            param_nodes.append(f)

    if plus_node is not None:
        terms = flatten_binary(plus_node, "+")
        cos_call = None
        if len(terms) == 2:
            t1, t2 = terms
            if (
                isinstance(t1, ConstNode)
                and math.isclose(t1.value, 1.0)
                and isinstance(t2, CallNode)
                and t2.name.lower() == "cos"
                and len(t2.args) == 1
            ):
                cos_call = t2
            elif (
                isinstance(t2, ConstNode)
                and math.isclose(t2.value, 1.0)
                and isinstance(t1, CallNode)
                and t1.name.lower() == "cos"
                and len(t1.args) == 1
            ):
                cos_call = t1

        if cos_call is not None and len(param_nodes) <= 1:
            k_node = param_nodes[0] if param_nodes else ConstNode(1.0)
            return cos_call, c_prod, k_node

    # --- Pattern Form 2: Expanded Sum Form: C * k + C * k * cos(n * theta - phase) ---
    if isinstance(ast, BinaryOpNode) and ast.op == "+":
        terms = flatten_binary(ast, "+")
        if len(terms) == 2:
            t1, t2 = terms

            def _split_cos_term(node: ASTNode) -> tuple[CallNode, float, ASTNode] | None:
                f_list = flatten_binary(node, "*")
                cos_list = [
                    f for f in f_list
                    if isinstance(f, CallNode) and f.name.lower() == "cos" and len(f.args) == 1
                ]
                if len(cos_list) != 1:
                    return None
                other_factors = [f for f in f_list if f is not cos_list[0]]
                cm = 1.0
                pn: list[ASTNode] = []
                for f in other_factors:
                    if isinstance(f, ConstNode):
                        cm *= f.value
                    else:
                        pn.append(f)
                if len(pn) > 1:
                    return None
                kn = pn[0] if pn else ConstNode(1.0)
                return cos_list[0], cm, kn

            def _split_base_term(node: ASTNode) -> tuple[float, ASTNode] | None:
                f_list = flatten_binary(node, "*")
                if any(isinstance(f, CallNode) and f.name.lower() == "cos" for f in f_list):
                    return None
                cm = 1.0
                pn: list[ASTNode] = []
                for f in f_list:
                    if isinstance(f, ConstNode):
                        cm *= f.value
                    else:
                        pn.append(f)
                if len(pn) > 1:
                    return None
                kn = pn[0] if pn else ConstNode(1.0)
                return cm, kn

            s1 = _split_cos_term(t1)
            s2 = _split_base_term(t2)
            if s1 is None or s2 is None:
                s1 = _split_cos_term(t2)
                s2 = _split_base_term(t1)

            if s1 is not None and s2 is not None:
                cos_cand, cm1, kn1 = s1
                cm2, kn2 = s2
                if math.isclose(cm1, cm2, rel_tol=1e-5) and kn1 == kn2:
                    return cos_cand, cm1, kn1

    return None


def _extract_cos_angle_params(cos_arg: ASTNode) -> tuple[ASTNode, ASTNode] | None:
    """Extract (n_node, phase_node) from cosine argument (e.g. n * theta - phase)."""
    n_node: ASTNode = ConstNode(1.0)
    phase_node: ASTNode = ConstNode(0.0)

    if isinstance(cos_arg, BinaryOpNode) and cos_arg.op == "-":
        phase_node = cos_arg.right
        n_part = cos_arg.left
    elif isinstance(cos_arg, BinaryOpNode) and cos_arg.op == "+":
        terms_arg = flatten_binary(cos_arg, "+")
        if len(terms_arg) == 2:
            if isinstance(terms_arg[0], ConstNode):
                phase_node = ConstNode(-terms_arg[0].value)
                n_part = terms_arg[1]
            elif isinstance(terms_arg[1], ConstNode):
                phase_node = ConstNode(-terms_arg[1].value)
                n_part = terms_arg[0]
            else:
                return None
        else:
            return None
    else:
        n_part = cos_arg

    if isinstance(n_part, VarNode) and n_part.name == "theta":
        n_node = ConstNode(1.0)
    elif isinstance(n_part, BinaryOpNode) and n_part.op == "*":
        factors_n = flatten_binary(n_part, "*")
        theta_factors = [f for f in factors_n if isinstance(f, VarNode) and f.name == "theta"]
        if len(theta_factors) != 1:
            return None
        other_n = [f for f in factors_n if f is not theta_factors[0]]
        if len(other_n) != 1:
            return None
        n_node = other_n[0]
    else:
        return None

    return n_node, phase_node


def _match_periodic_torsion(force: openmm.CustomTorsionForce, ast: ASTNode) -> MatchedPattern | None:
    num_torsions = force.getNumTorsions()
    if num_torsions == 0:
        return None

    cos_match = _extract_periodic_cos_call(ast)
    if cos_match is None:
        return None
    cos_call, const_mult, k_node = cos_match

    angle_params = _extract_cos_angle_params(cos_call.args[0])
    if angle_params is None:
        return None
    n_node, phase_node = angle_params

    # Per-torsion parameter resolution and conversion
    num_params = force.getNumPerTorsionParameters()
    param_names = [force.getPerTorsionParameterName(i) for i in range(num_params)]

    indices_list = []
    params_list = []
    for i in range(num_torsions):
        p1, p2, p3, p4, raw_params = force.getTorsionParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_val = _get_param_val(k_node, p_dict)
        n_val = _get_param_val(n_node, p_dict)
        phase_val = _get_param_val(phase_node, p_dict)
        if k_val is None or n_val is None or phase_val is None:
            return None

        # Convert to AKMA: k in kcal/mol
        k_akma = const_mult * (k_val * TORSION_K_OMM_TO_AKMA)
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
        if r_ang is not None and angle_term is None:
            angle_term = r_ang
        elif r_ub is not None and ub_term is None:
            ub_term = r_ub
        else:
            # Term did not match either, was duplicate, or is an unrecognized non-harmonic term
            return None

    # Strict term length assertions:
    # Combined Angle + Urey-Bradley requires exactly 2 terms
    if len(terms) == 2 and ub_term is not None and angle_term is not None:
        return _build_combined_angle_ub_pattern(force, angle_term, ub_term, param_names, num_bonds)

    # Pure Urey-Bradley requires exactly 1 term
    if len(terms) == 1 and ub_term is not None:
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

    # Step 1: Structural AST Matching
    # Target canonical forms:
    # Form 1: mult * k * periodicdistance(x, y, z, x0, y0, z0)^2
    # Form 2A (factored): mult * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)
    # Form 2B (distributed): mult * k * (x - x0)^2 + mult * k * (y - y0)^2 + mult * k * (z - z0)^2

    const_mult = 1.0
    k_node: ASTNode | None = None
    x0_node: ASTNode | None = None
    y0_node: ASTNode | None = None
    z0_node: ASTNode | None = None

    # --- Pattern Form 2B: Distributed sum of 3 harmonic coordinate differences ---
    if isinstance(ast, BinaryOpNode) and ast.op == "+":
        terms = flatten_binary(ast, "+")
        if len(terms) == 3:
            extracted: dict[str, tuple[float, ASTNode, ASTNode]] = {}
            for t in terms:
                for c in ("x", "y", "z"):
                    res = _extract_harmonic_diff(t, c)
                    if res is not None and c not in extracted:
                        extracted[c] = res
                        break
            if len(extracted) == 3:
                mult_x, k_x, x0_cand = extracted["x"]
                mult_y, k_y, y0_cand = extracted["y"]
                mult_z, k_z, z0_cand = extracted["z"]
                if (
                    math.isclose(mult_x, mult_y, rel_tol=1e-5)
                    and math.isclose(mult_x, mult_z, rel_tol=1e-5)
                    and k_x == k_y == k_z
                ):
                    const_mult = mult_x
                    k_node = k_x
                    x0_node, y0_node, z0_node = x0_cand, y0_cand, z0_cand

    # --- Pattern Form 1 and Form 2A: Factored products ---
    if k_node is None:
        factors = flatten_binary(ast, "*")
        c_prod = 1.0
        param_nodes: list[ASTNode] = []
        pdist_call: CallNode | None = None
        sum_of_squares_node: BinaryOpNode | None = None

        for f in factors:
            if isinstance(f, ConstNode):
                c_prod *= f.value
            elif (
                isinstance(f, BinaryOpNode)
                and f.op == "^"
                and isinstance(f.right, ConstNode)
                and math.isclose(f.right.value, 2.0)
            ):
                if isinstance(f.left, CallNode) and f.left.name.lower() == "periodicdistance":
                    pdist_call = f.left
                else:
                    param_nodes.append(f)
            elif isinstance(f, BinaryOpNode) and f.op == "+":
                sum_of_squares_node = f
            else:
                param_nodes.append(f)

        # Enforce multi-variable multiplier guard
        if len(param_nodes) > 1:
            return None

        # Try Form 1: periodicdistance^2
        if pdist_call is not None and len(pdist_call.args) == 6:
            args = pdist_call.args
            coord_names_first = [getattr(a, "name", None) for a in args[:3]]
            coord_names_last = [getattr(a, "name", None) for a in args[3:]]
            if coord_names_first == ["x", "y", "z"]:
                const_mult = c_prod
                k_node = param_nodes[0] if param_nodes else ConstNode(1.0)
                x0_node, y0_node, z0_node = args[3], args[4], args[5]
            elif coord_names_last == ["x", "y", "z"]:
                const_mult = c_prod
                k_node = param_nodes[0] if param_nodes else ConstNode(1.0)
                x0_node, y0_node, z0_node = args[0], args[1], args[2]

        # Try Form 2A: sum of squares ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)
        elif sum_of_squares_node is not None:
            terms = flatten_binary(sum_of_squares_node, "+")
            if len(terms) == 3:
                coords: dict[str, ASTNode] = {}
                for t in terms:
                    if (
                        isinstance(t, BinaryOpNode)
                        and t.op == "^"
                        and isinstance(t.right, ConstNode)
                        and math.isclose(t.right.value, 2.0)
                    ):
                        for c in ("x", "y", "z"):
                            extracted_base = _extract_coord_param_from_base(t.left, c)
                            if extracted_base is not None and c not in coords:
                                coords[c] = extracted_base[1]
                                break
                if len(coords) == 3:
                    const_mult = c_prod
                    k_node = param_nodes[0] if param_nodes else ConstNode(1.0)
                    x0_node, y0_node, z0_node = coords["x"], coords["y"], coords["z"]

    # If neither pattern matched, reject
    if k_node is None or x0_node is None or y0_node is None or z0_node is None:
        return None

    # Step 2: Per-particle parameter extraction and unit conversion
    num_params = force.getNumPerParticleParameters()
    param_names = [force.getPerParticleParameterName(i) for i in range(num_params)]

    indices_list = []
    params_list = []
    for i in range(num_particles):
        p_idx, raw_params = force.getParticleParameters(i)
        p_dict = dict(zip(param_names, raw_params, strict=True))

        k_val = _get_param_val(k_node, p_dict)
        x0_val = _get_param_val(x0_node, p_dict)
        y0_val = _get_param_val(y0_node, p_dict)
        z0_val = _get_param_val(z0_node, p_dict)

        if any(v is None for v in (k_val, x0_val, y0_val, z0_val)):
            return None

        # AKMA conversion:
        # Prolix compute_positional_restraint_energy evaluates E = sum k_prolix * Delta_r^2 (unhalved)
        # Therefore: k_prolix = mult * (k_omm / 418.4)
        x0_akma = x0_val * NM_TO_ANGSTROM
        y0_akma = y0_val * NM_TO_ANGSTROM
        z0_akma = z0_val * NM_TO_ANGSTROM
        k_akma = const_mult * (k_val * BOND_K_OMM_TO_AKMA)

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


# Shared bucket ladder families and their associated bundle fields
_BUCKET_FAMILIES: dict[str, tuple[tuple[int, ...], str, list[tuple[str, str, str]]]] = {
    "bond": (
        BOND_BUCKETS,
        "bond_bucket_idx",
        [("bond_idx", "bond_params", "bond_mask")],
    ),
    "angle": (
        ANGLE_BUCKETS,
        "angle_bucket_idx",
        [
            ("angle_idx", "angle_params", "angle_mask"),
            ("urey_bradley_idx", "urey_bradley_params", "urey_bradley_mask"),
        ],
    ),
    "urey_bradley": (
        ANGLE_BUCKETS,
        "angle_bucket_idx",
        [
            ("angle_idx", "angle_params", "angle_mask"),
            ("urey_bradley_idx", "urey_bradley_params", "urey_bradley_mask"),
        ],
    ),
    "dihedral": (
        DIHEDRAL_BUCKETS,
        "dihedral_bucket_idx",
        [
            ("dihedral_idx", "dihedral_params", "dihedral_mask"),
            ("improper_idx", "improper_params", "improper_mask"),
        ],
    ),
    "improper": (
        DIHEDRAL_BUCKETS,
        "dihedral_bucket_idx",
        [
            ("dihedral_idx", "dihedral_params", "dihedral_mask"),
            ("improper_idx", "improper_params", "improper_mask"),
        ],
    ),
}


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
    if term not in _BUCKET_FAMILIES:
        raise ValueError(f"Unsupported target term '{term}'")

    ladder, spec_field, family_fields = _BUCKET_FAMILIES[term]
    extra_kwargs: dict[str, Any] = {}
    if term == "bond":
        idx_field, prm_field, msk_field, n_field = "bond_idx", "bond_params", "bond_mask", "n_bonds"
    elif term == "angle":
        idx_field, prm_field, msk_field, n_field = "angle_idx", "angle_params", "angle_mask", "n_angles"
    elif term == "urey_bradley":
        idx_field, prm_field, msk_field, n_field = "urey_bradley_idx", "urey_bradley_params", "urey_bradley_mask", "n_urey_bradley"
    elif term == "dihedral":
        idx_field, prm_field, msk_field, n_field = "dihedral_idx", "dihedral_params", "dihedral_mask", "n_dihedrals"
    elif term == "improper":
        idx_field, prm_field, msk_field, n_field = "improper_idx", "improper_params", "improper_mask", "n_impropers"
        extra_kwargs = {"improper_is_periodic": jnp.array(False, dtype=bool)}

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

    updates: dict[str, Any] = {
        idx_field: padded_idx,
        prm_field: padded_prm,
        msk_field: mask,
        n_field: jnp.array(n_total, dtype=jnp.int32),
        "shape_spec": shape_spec,
        **extra_kwargs,
    }

    # CRITICAL: Synchronize all companion arrays in the same bucket family
    for c_idx, c_prm, c_msk in family_fields:
        if c_idx != idx_field:
            curr_comp_cap = getattr(bundle, c_idx).shape[0]
            if curr_comp_cap < target_cap:
                c_pad = target_cap - curr_comp_cap
                updates[c_idx] = jnp.pad(getattr(bundle, c_idx), ((0, c_pad), (0, 0)))
                updates[c_prm] = jnp.pad(getattr(bundle, c_prm), ((0, c_pad), (0, 0)))
                updates[c_msk] = jnp.pad(getattr(bundle, c_msk), ((0, c_pad),), constant_values=False)

    return dataclasses.replace(bundle, **updates)
