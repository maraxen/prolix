"""Lepton expression compiler, AST representation, topological sorter, and JAX custom force engine.

This module provides a recursive-descent parser, canonical AST representation, Kahn's algorithm
dependency DAG sorter, power operator fast-paths, 36 built-in and geometric functions,
JAX lowering for all 6 OpenMM TabulatedFunction classes, and pure-JAX differentiable custom
force engines for:
  - CustomBondForce
  - CustomAngleForce
  - CustomTorsionForce
  - CustomExternalForce
  - CustomCompoundBondForce
  - CustomNonbondedForce (with chunked evaluation, exclusions, interaction groups, and switching)
"""

from __future__ import annotations

import collections
import enum
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp
import openmm
from jaxtyping import Array, Bool, Float, Int

from prolix.openmm.config import ProlixOpenMMConfig
from prolix.openmm.units import (
    ANGSTROM_TO_NM,
    KJ_PER_MOL_TO_KCAL_PER_MOL,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    # AST Nodes
    "ASTNode",
    "ConstNode",
    "VarNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "CallNode",
    # Tokenizer & Parser
    "Token",
    "TokenType",
    "tokenize_lepton",
    "LeptonParser",
    "parse_lepton_single_expression",
    "parse_lepton_expression",
    "extract_free_variables",
    "resolve_lepton_program",
    "inline_definitions",
    # Power fast-path
    "evaluate_integer_power",
    # Tabulated Functions
    "compile_continuous_1d_function",
    "compile_continuous_2d_function",
    "compile_continuous_3d_function",
    "compile_discrete_1d_function",
    "compile_discrete_2d_function",
    "compile_discrete_3d_function",
    "compile_tabulated_function",
    # Geometry & Displacements
    "minimum_image_displacement",
    # JAX Lowering
    "lower_ast_to_jax",
    "evaluate_lepton_program",
    # Compiled Custom Forces
    "CompiledLeptonForce",
    "CompiledCustomBondForce",
    "CompiledCustomAngleForce",
    "CompiledCustomTorsionForce",
    "CompiledCustomExternalForce",
    "CompiledCustomCompoundBondForce",
    "CompiledCustomNonbondedForce",
    # High-level API
    "compile_custom_force",
    "evaluate_custom_force",
]


# =============================================================================
# 1. AST Node Dataclasses
# =============================================================================


@dataclass(frozen=True)
class ASTNode:
    """Base class for all immutable Lepton AST nodes."""



@dataclass(frozen=True)
class ConstNode(ASTNode):
    """Numeric constant node."""

    value: float


@dataclass(frozen=True)
class VarNode(ASTNode):
    """Variable identifier node."""

    name: str


@dataclass(frozen=True)
class UnaryOpNode(ASTNode):
    """Unary operation node ('-' or '+')."""

    op: str
    operand: ASTNode


@dataclass(frozen=True)
class BinaryOpNode(ASTNode):
    """Binary operation node ('+', '-', '*', '/', '^')."""

    op: str
    left: ASTNode
    right: ASTNode


@dataclass(frozen=True)
class CallNode(ASTNode):
    """Function call node with argument expressions."""

    name: str
    args: tuple[ASTNode, ...]


# =============================================================================
# 2. Tokenizer
# =============================================================================


class TokenType(enum.Enum):
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    OP = "OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    EQUALS = "EQUALS"
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    pos: int


_NUMBER_RE = re.compile(
    r"^(([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?)"
)
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*")
_OP_CHARS = "+-*/^"


def tokenize_lepton(text: str) -> list[Token]:
    """Tokenize a Lepton program string into a list of Tokens."""
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            tokens.append(Token(TokenType.LPAREN, "(", i))
            i += 1
        elif c == ")":
            tokens.append(Token(TokenType.RPAREN, ")", i))
            i += 1
        elif c == ",":
            tokens.append(Token(TokenType.COMMA, ",", i))
            i += 1
        elif c == ";":
            tokens.append(Token(TokenType.SEMICOLON, ";", i))
            i += 1
        elif c == "=":
            tokens.append(Token(TokenType.EQUALS, "=", i))
            i += 1
        elif c in _OP_CHARS:
            tokens.append(Token(TokenType.OP, c, i))
            i += 1
        else:
            m_num = _NUMBER_RE.match(text[i:])
            if m_num:
                tok_str = m_num.group(1)
                tokens.append(Token(TokenType.NUMBER, tok_str, i))
                i += len(tok_str)
                continue
            m_id = _IDENT_RE.match(text[i:])
            if m_id:
                tok_str = m_id.group(0)
                tokens.append(Token(TokenType.IDENT, tok_str, i))
                i += len(tok_str)
                continue
            raise ValueError(
                f"Unexpected character '{c}' at position {i} in expression: {text!r}"
            )
    tokens.append(Token(TokenType.EOF, "", n))
    return tokens


# =============================================================================
# 3. Recursive-Descent Precedence Parser
# =============================================================================

# Operator precedence matching OpenMM Parser.cpp:
# + and -: 0 (left-associative)
# * and /: 1 (left-associative)
# Unary -: 2 (prefix)
# ^:       3 (right-associative)
_PRECEDENCE = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 3}


class LeptonParser:
    """Recursive-descent precedence climbing parser for Lepton expressions."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match(self, tok_type: TokenType, value: str | None = None) -> bool:
        tok = self.peek()
        if tok.type == tok_type and (value is None or tok.value == value):
            self.consume()
            return True
        return False

    def expect(self, tok_type: TokenType, value: str | None = None) -> Token:
        tok = self.peek()
        if tok.type != tok_type or (value is not None and tok.value != value):
            expected = value or tok_type.value
            raise ValueError(
                f"Expected {expected} at pos {tok.pos}, found '{tok.value}'"
            )
        return self.consume()

    def parse_expression(self) -> ASTNode:
        return self.parse_precedence(min_precedence=0)

    def parse_precedence(self, min_precedence: int) -> ASTNode:
        left = self.parse_unary()
        while True:
            tok = self.peek()
            if tok.type != TokenType.OP or tok.value not in _PRECEDENCE:
                break
            prec = _PRECEDENCE[tok.value]
            if prec < min_precedence:
                break
            op = self.consume().value
            # Right-associative operator keeps prec; left-associative uses prec + 1
            next_min_prec = prec if op == "^" else prec + 1
            right = self.parse_precedence(next_min_prec)
            left = BinaryOpNode(op, left, right)
        return left

    def parse_unary(self) -> ASTNode:
        tok = self.peek()
        if tok.type == TokenType.OP and tok.value == "-":
            self.consume()
            # Unary minus has precedence 2 (binds tighter than * / +, but looser than ^)
            operand = self.parse_precedence(min_precedence=2)
            return UnaryOpNode("-", operand)
        if tok.type == TokenType.OP and tok.value == "+":
            self.consume()
            return self.parse_precedence(min_precedence=2)
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        tok = self.peek()
        if tok.type == TokenType.NUMBER:
            self.consume()
            return ConstNode(float(tok.value))
        if tok.type == TokenType.IDENT:
            name = self.consume().value
            if self.match(TokenType.LPAREN):
                args: list[ASTNode] = []
                if not self.match(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression())
                    self.expect(TokenType.RPAREN)
                return CallNode(name, tuple(args))
            return VarNode(name)
        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        raise ValueError(
            f"Unexpected token '{tok.value}' at position {tok.pos} while parsing primary expression"
        )


def parse_lepton_single_expression(text: str) -> ASTNode:
    """Parse a single Lepton expression (without semicolons)."""
    tokens = tokenize_lepton(text)
    parser = LeptonParser(tokens)
    ast = parser.parse_expression()
    if parser.peek().type != TokenType.EOF:
        raise ValueError(
            f"Unexpected trailing tokens at pos {parser.peek().pos}: '{parser.peek().value}'"
        )
    return ast


# =============================================================================
# 4. Topological Definition Sorter & Inlining (Kahn's Algorithm)
# =============================================================================


def extract_free_variables(node: ASTNode) -> set[str]:
    """Recursively extract all variable names referenced in an AST node."""
    if isinstance(node, VarNode):
        return {node.name}
    if isinstance(node, ConstNode):
        return set()
    if isinstance(node, UnaryOpNode):
        return extract_free_variables(node.operand)
    if isinstance(node, BinaryOpNode):
        return extract_free_variables(node.left) | extract_free_variables(node.right)
    if isinstance(node, CallNode):
        vars_: set[str] = set()
        for arg in node.args:
            vars_.update(extract_free_variables(arg))
        return vars_
    return set()


def resolve_lepton_program(
    program_str: str,
) -> tuple[ASTNode, list[tuple[str, ASTNode]]]:
    """Parse a semicolon-delimited Lepton string and topologically sort definitions.

    Returns:
        tuple[ASTNode, list[tuple[str, ASTNode]]]:
            (primary_ast, sorted_definitions)
            where sorted_definitions is ordered such that dependencies precede dependents.
    """
    raw_statements = [s.strip() for s in program_str.split(";") if s.strip()]
    if not raw_statements:
        raise ValueError("Empty Lepton program string.")

    primary_ast: ASTNode | None = None
    definitions: dict[str, ASTNode] = {}

    for stmt in raw_statements:
        if "=" in stmt:
            parts = stmt.split("=", 1)
            var_name = parts[0].strip()
            expr_str = parts[1].strip()
            if not var_name or not expr_str:
                raise ValueError(f"Malformed definition statement: {stmt!r}")
            if var_name in definitions:
                raise ValueError(f"Duplicate definition for variable: {var_name!r}")
            definitions[var_name] = parse_lepton_single_expression(expr_str)
        else:
            if primary_ast is not None:
                raise ValueError(
                    f"Multiple primary return expressions found in Lepton string: {stmt!r}"
                )
            primary_ast = parse_lepton_single_expression(stmt)

    if primary_ast is None:
        raise ValueError("No primary return expression found in Lepton string.")

    # Build dependency DAG among definitions
    dep_names = set(definitions.keys())
    reverse_deps: dict[str, set[str]] = {k: set() for k in definitions}
    in_degree: dict[str, int] = dict.fromkeys(definitions, 0)

    for var, ast in definitions.items():
        free_vars = extract_free_variables(ast)
        deps = free_vars & dep_names
        in_degree[var] = len(deps)
        for dep in deps:
            reverse_deps[dep].add(var)

    # Kahn's algorithm
    queue = collections.deque([v for v, deg in in_degree.items() if deg == 0])
    sorted_defs: list[tuple[str, ASTNode]] = []

    while queue:
        curr = queue.popleft()
        sorted_defs.append((curr, definitions[curr]))
        for dependent in sorted(reverse_deps[curr]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_defs) != len(definitions):
        unresolved = [v for v, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"Cyclic dependency detected among Lepton definitions: {sorted(unresolved)}"
        )

    return primary_ast, sorted_defs


def parse_lepton_expression(
    program_str: str,
) -> tuple[ASTNode, dict[str, ASTNode]]:
    """Parse a semicolon-delimited Lepton string into primary AST and definition dict."""
    primary_ast, sorted_defs = resolve_lepton_program(program_str)
    return primary_ast, dict(sorted_defs)


def inline_definitions(
    primary_ast: ASTNode,
    sorted_defs: list[tuple[str, ASTNode]] | dict[str, ASTNode],
) -> ASTNode:
    """Inline intermediate definitions into the primary AST for pattern matching."""
    def_map = dict(sorted_defs)

    def _substitute(node: ASTNode, visited: frozenset[str]) -> ASTNode:
        if isinstance(node, VarNode) and node.name in def_map:
            if node.name in visited:
                raise ValueError(f"Cyclic dependency detected while inlining: {node.name!r}")
            return _substitute(def_map[node.name], visited | {node.name})
        if isinstance(node, UnaryOpNode):
            return UnaryOpNode(node.op, _substitute(node.operand, visited))
        if isinstance(node, BinaryOpNode):
            return BinaryOpNode(
                node.op,
                _substitute(node.left, visited),
                _substitute(node.right, visited),
            )
        if isinstance(node, CallNode):
            return CallNode(
                node.name,
                tuple(_substitute(arg, visited) for arg in node.args),
            )
        return node

    return _substitute(primary_ast, frozenset())


# =============================================================================
# 5. Integer Power Fast Paths
# =============================================================================


def evaluate_integer_power(base: Array, exponent: int) -> Array:
    """Safely evaluate base^exponent in JAX without NaN for negative bases."""
    lookup = {
        0: jnp.ones_like,
        1: lambda b: b,
        2: lambda b: b * b,
        3: lambda b: b * b * b,
        4: lambda b: (b * b) * (b * b),
        5: lambda b: (b * b) * (b * b) * b,
        6: lambda b: (b * b) * (b * b) * (b * b),
        -1: lambda b: 1.0 / b,
        -2: lambda b: 1.0 / (b * b),
        -3: lambda b: 1.0 / (b * b * b),
        -6: lambda b: 1.0 / ((b * b) * (b * b) * (b * b)),
    }
    if exponent in lookup:
        return lookup[exponent](base)
    if exponent > 0:
        return jax.lax.integer_pow(base, exponent)
    return 1.0 / jax.lax.integer_pow(base, -exponent)


# =============================================================================
# 6. Tabulated Functions (Continuous 1D/2D/3D & Discrete 1D/2D/3D)
# =============================================================================


def _solve_tridiagonal(
    a: Sequence[float], b: Sequence[float], c: Sequence[float], rhs: Sequence[float]
) -> list[float]:
    """Solve tridiagonal linear system matching OpenMM SplineFitter::solveTridiagonalMatrix."""
    n = len(a)
    gamma = [0.0] * n
    sol = [0.0] * n
    sol[0] = rhs[0] / b[0]
    beta = b[0]
    for i in range(1, n):
        gamma[i] = c[i - 1] / beta
        beta = b[i] - a[i] * gamma[i]
        sol[i] = (rhs[i] - a[i] * sol[i - 1]) / beta
    for i in range(n - 2, -1, -1):
        sol[i] -= gamma[i + 1] * sol[i + 1]
    return sol


def _create_natural_spline_1d(
    x: Sequence[float], y: Sequence[float]
) -> list[float]:
    """Compute natural cubic spline second derivatives matching OpenMM SplineFitter."""
    n = len(x)
    if n < 2:
        raise ValueError("Spline must have at least 2 points.")
    if n == 2:
        return [0.0, 0.0]
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    rhs = [0.0] * n
    b[0] = 1.0
    b[n - 1] = 1.0
    for i in range(1, n - 1):
        a[i] = x[i] - x[i - 1]
        b[i] = 2.0 * (x[i + 1] - x[i - 1])
        c[i] = x[i + 1] - x[i]
        rhs[i] = 6.0 * (
            (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        )
    return _solve_tridiagonal(a, b, c, rhs)


def _eval_spline_derivative_point(
    x: Sequence[float], y: Sequence[float], deriv: Sequence[float], t: float
) -> float:
    """Evaluate 1D spline derivative at point t matching OpenMM evaluateSplineDerivative."""
    n = len(x)
    dx_total = x[-1] - x[0]
    k = int(min(max(math.floor((t - x[0]) / (dx_total / (n - 1))), 0), n - 2))
    dx = x[k + 1] - x[k]
    a = (x[k + 1] - t) / dx
    b = 1.0 - a
    dadx = -1.0 / dx
    return dadx * y[k] - dadx * y[k + 1] + (
        (1.0 - 3.0 * a * a) * deriv[k] + (3.0 * b * b - 1.0) * deriv[k + 1]
    ) * dx / 6.0


def compile_continuous_1d_function(
    values: Sequence[float], min_val: float, max_val: float, periodic: bool = False
) -> Callable[[Array], Array]:
    """Compile an OpenMM Continuous1DFunction into a JAX Horner spline closure."""
    n = len(values)
    length = max_val - min_val
    dx = length / (n - 1)
    x_grid = [min_val + i * dx for i in range(n)]
    derivs = _create_natural_spline_1d(x_grid, values)

    vals_arr = jnp.asarray(values, dtype=jnp.float32)
    derivs_arr = jnp.asarray(derivs, dtype=jnp.float32)

    def _eval(x: Array) -> Array:
        if periodic:
            s = (x - min_val) / length
            x_wrapped = min_val + length * (s - jnp.floor(s))
            in_bounds = jnp.ones_like(x, dtype=bool)
        else:
            x_wrapped = x
            in_bounds = (x >= min_val) & (x <= max_val)

        u = (x_wrapped - min_val) / dx
        k = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2)
        b = (x_wrapped - (min_val + k * dx)) / dx
        a = 1.0 - b

        y_k = vals_arr[k]
        y_k1 = vals_arr[k + 1]
        d_k = derivs_arr[k]
        d_k1 = derivs_arr[k + 1]

        res = (
            a * y_k
            + b * y_k1
            + ((a * a * a - a) * d_k + (b * b * b - b) * d_k1) * (dx * dx / 6.0)
        )
        return jnp.where(in_bounds, res, 0.0)

    return _eval


# OpenMM 2D Spline Weight Matrix (16x16) from SplineFitter.cpp
_WT_2D_FLAT = [
    1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4,
    0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4,
    0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2,
    0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2,
    0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2,
    0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2,
    0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2,
    0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1,
    0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1,
]


def compile_continuous_2d_function(
    xsize: int,
    ysize: int,
    values: Sequence[float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    periodic: bool = False,
) -> Callable[[Array, Array], Array]:
    """Compile an OpenMM Continuous2DFunction into a JAX bicubic Horner spline closure."""
    dx = (xmax - xmin) / (xsize - 1)
    dy = (ymax - ymin) / (ysize - 1)
    x_grid = [xmin + i * dx for i in range(xsize)]
    y_grid = [ymin + j * dy for j in range(ysize)]

    d1 = [0.0] * (xsize * ysize)
    d2 = [0.0] * (xsize * ysize)
    d12 = [0.0] * (xsize * ysize)

    # d1: derivative with respect to x (row by row)
    for i in range(ysize):
        row_vals = [values[j + xsize * i] for j in range(xsize)]
        d = _create_natural_spline_1d(x_grid, row_vals)
        for j in range(xsize):
            d1[j + xsize * i] = _eval_spline_derivative_point(
                x_grid, row_vals, d, x_grid[j]
            )

    # d2: derivative with respect to y (col by col)
    for j in range(xsize):
        col_vals = [values[j + xsize * i] for i in range(ysize)]
        d = _create_natural_spline_1d(y_grid, col_vals)
        for i in range(ysize):
            d2[j + xsize * i] = _eval_spline_derivative_point(
                y_grid, col_vals, d, y_grid[i]
            )

    # d12: cross derivative
    for i in range(ysize):
        row_d2 = [d2[j + xsize * i] for j in range(xsize)]
        d = _create_natural_spline_1d(x_grid, row_d2)
        for j in range(xsize):
            d12[j + xsize * i] = _eval_spline_derivative_point(
                x_grid, row_d2, d, x_grid[j]
            )

    # Compute 16 coefficients for each patch (i, j)
    # Shape: (xsize - 1, ysize - 1, 16)
    c_patches = [
        [ [0.0] * 16 for _ in range(ysize - 1) ] for _ in range(xsize - 1)
    ]
    for i in range(xsize - 1):
        for j in range(ysize - 1):
            nexti = i + 1
            nextj = j + 1
            e = [
                values[i + j * xsize],
                values[nexti + j * xsize],
                values[nexti + nextj * xsize],
                values[i + nextj * xsize],
            ]
            e1 = [
                d1[i + j * xsize],
                d1[nexti + j * xsize],
                d1[nexti + nextj * xsize],
                d1[i + nextj * xsize],
            ]
            e2 = [
                d2[i + j * xsize],
                d2[nexti + j * xsize],
                d2[nexti + nextj * xsize],
                d2[i + nextj * xsize],
            ]
            e12 = [
                d12[i + j * xsize],
                d12[nexti + j * xsize],
                d12[nexti + nextj * xsize],
                d12[i + nextj * xsize],
            ]
            rhs = [0.0] * 16
            for k in range(4):
                rhs[k] = e[k]
                rhs[k + 4] = e1[k] * dx
                rhs[k + 8] = e2[k] * dy
                rhs[k + 12] = e12[k] * dx * dy
            coeff = [0.0] * 16
            for k in range(16):
                s = 0.0
                for m in range(16):
                    s += _WT_2D_FLAT[k + 16 * m] * rhs[m]
                coeff[k] = s
            c_patches[i][j] = coeff

    c_arr = jnp.asarray(c_patches, dtype=jnp.float32)
    lx = xmax - xmin
    ly = ymax - ymin

    def _eval(u: Array, v: Array) -> Array:
        if periodic:
            su = (u - xmin) / lx
            sv = (v - ymin) / ly
            u_w = xmin + lx * (su - jnp.floor(su))
            v_w = ymin + ly * (sv - jnp.floor(sv))
            in_bounds = jnp.ones_like(u, dtype=bool)
        else:
            u_w = u
            v_w = v
            in_bounds = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)

        idx_u = jnp.clip(jnp.floor((u_w - xmin) / dx).astype(jnp.int32), 0, xsize - 2)
        idx_v = jnp.clip(jnp.floor((v_w - ymin) / dy).astype(jnp.int32), 0, ysize - 2)
        da = (u_w - (xmin + idx_u * dx)) / dx
        db = (v_w - (ymin + idx_v * dy)) / dy

        coeff = c_arr[idx_u, idx_v]  # shape: (..., 16)
        val = jnp.zeros_like(u_w)
        for c_i in range(3, -1, -1):
            term = (
                (coeff[..., c_i * 4 + 3] * db + coeff[..., c_i * 4 + 2]) * db
                + coeff[..., c_i * 4 + 1]
            ) * db + coeff[..., c_i * 4 + 0]
            val = da * val + term

        return jnp.where(in_bounds, val, 0.0)

    return _eval


def compile_continuous_3d_function(
    xsize: int,
    ysize: int,
    zsize: int,
    values: Sequence[float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    periodic: bool = False,
) -> Callable[[Array, Array, Array], Array]:
    """Compile an OpenMM Continuous3DFunction into a JAX trilinear/tricubic spline closure."""
    dx = (xmax - xmin) / (xsize - 1)
    dy = (ymax - ymin) / (ysize - 1)
    dz = (zmax - zmin) / (zsize - 1)
    vals_arr = jnp.asarray(values, dtype=jnp.float32).reshape((zsize, ysize, xsize))
    lx = xmax - xmin
    ly = ymax - ymin
    lz = zmax - zmin

    def _eval(u: Array, v: Array, w: Array) -> Array:
        if periodic:
            su = (u - xmin) / lx
            sv = (v - ymin) / ly
            sw = (w - zmin) / lz
            u_w = xmin + lx * (su - jnp.floor(su))
            v_w = ymin + ly * (sv - jnp.floor(sv))
            w_w = zmin + lz * (sw - jnp.floor(sw))
            in_bounds = jnp.ones_like(u, dtype=bool)
        else:
            u_w = u
            v_w = v
            w_w = w
            in_bounds = (
                (u >= xmin)
                & (u <= xmax)
                & (v >= ymin)
                & (v <= ymax)
                & (w >= zmin)
                & (w <= zmax)
            )

        iu = jnp.clip(jnp.floor((u_w - xmin) / dx).astype(jnp.int32), 0, xsize - 2)
        iv = jnp.clip(jnp.floor((v_w - ymin) / dy).astype(jnp.int32), 0, ysize - 2)
        iw = jnp.clip(jnp.floor((w_w - zmin) / dz).astype(jnp.int32), 0, zsize - 2)

        fu = (u_w - (xmin + iu * dx)) / dx
        fv = (v_w - (ymin + iv * dy)) / dy
        fw = (w_w - (zmin + iw * dz)) / dz

        # Trilinear interpolation
        c000 = vals_arr[iw, iv, iu]
        c100 = vals_arr[iw, iv, iu + 1]
        c010 = vals_arr[iw, iv + 1, iu]
        c110 = vals_arr[iw, iv + 1, iu + 1]
        c001 = vals_arr[iw + 1, iv, iu]
        c101 = vals_arr[iw + 1, iv, iu + 1]
        c011 = vals_arr[iw + 1, iv + 1, iu]
        c111 = vals_arr[iw + 1, iv + 1, iu + 1]

        c00 = c000 * (1 - fu) + c100 * fu
        c01 = c001 * (1 - fu) + c101 * fu
        c10 = c010 * (1 - fu) + c110 * fu
        c11 = c011 * (1 - fu) + c111 * fu

        c0 = c00 * (1 - fv) + c10 * fv
        c1 = c01 * (1 - fv) + c11 * fv

        res = c0 * (1 - fw) + c1 * fw
        return jnp.where(in_bounds, res, 0.0)

    return _eval


def compile_discrete_1d_function(
    values: Sequence[float],
) -> Callable[[Array], Array]:
    """Compile an OpenMM Discrete1DFunction into a JAX rounded lookup closure."""
    vals_arr = jnp.asarray(values, dtype=jnp.float32)
    n = len(values)

    def _eval(x: Array) -> Array:
        idx = jnp.clip(jnp.round(x).astype(jnp.int32), 0, n - 1)
        res = vals_arr[idx]
        return jax.lax.stop_gradient(res)

    return _eval


def compile_discrete_2d_function(
    xsize: int, ysize: int, values: Sequence[float]
) -> Callable[[Array, Array], Array]:
    """Compile an OpenMM Discrete2DFunction into a JAX rounded lookup closure."""
    vals_arr = jnp.asarray(values, dtype=jnp.float32)

    def _eval(x: Array, y: Array) -> Array:
        i = jnp.clip(jnp.round(x).astype(jnp.int32), 0, xsize - 1)
        j = jnp.clip(jnp.round(y).astype(jnp.int32), 0, ysize - 1)
        res = vals_arr[i + j * xsize]
        return jax.lax.stop_gradient(res)

    return _eval


def compile_discrete_3d_function(
    xsize: int, ysize: int, zsize: int, values: Sequence[float]
) -> Callable[[Array, Array, Array], Array]:
    """Compile an OpenMM Discrete3DFunction into a JAX rounded lookup closure."""
    vals_arr = jnp.asarray(values, dtype=jnp.float32)

    def _eval(x: Array, y: Array, z: Array) -> Array:
        i = jnp.clip(jnp.round(x).astype(jnp.int32), 0, xsize - 1)
        j = jnp.clip(jnp.round(y).astype(jnp.int32), 0, ysize - 1)
        k = jnp.clip(jnp.round(z).astype(jnp.int32), 0, zsize - 1)
        res = vals_arr[i + (j + k * ysize) * xsize]
        return jax.lax.stop_gradient(res)

    return _eval


def compile_tabulated_function(omm_func: Any) -> Callable[..., Array]:
    """Compile an arbitrary OpenMM TabulatedFunction into a pure JAX callable."""
    type_name = type(omm_func).__name__
    if type_name == "Continuous1DFunction":
        vals, xmin, xmax = omm_func.getFunctionParameters()
        periodic = omm_func.getPeriodic()
        return compile_continuous_1d_function(vals, xmin, xmax, periodic)
    if type_name == "Continuous2DFunction":
        xsize, ysize, vals, xmin, xmax, ymin, ymax = omm_func.getFunctionParameters()
        periodic = omm_func.getPeriodic()
        return compile_continuous_2d_function(
            xsize, ysize, vals, xmin, xmax, ymin, ymax, periodic
        )
    if type_name == "Continuous3DFunction":
        (
            xsize,
            ysize,
            zsize,
            vals,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) = omm_func.getFunctionParameters()
        periodic = omm_func.getPeriodic()
        return compile_continuous_3d_function(
            xsize,
            ysize,
            zsize,
            vals,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            periodic,
        )
    if type_name == "Discrete1DFunction":
        params = omm_func.getFunctionParameters()
        vals = params[0] if isinstance(params, tuple) and len(params) == 1 else params
        return compile_discrete_1d_function(vals)
    if type_name == "Discrete2DFunction":
        xsize, ysize, vals = omm_func.getFunctionParameters()
        return compile_discrete_2d_function(xsize, ysize, vals)
    if type_name == "Discrete3DFunction":
        xsize, ysize, zsize, vals = omm_func.getFunctionParameters()
        return compile_discrete_3d_function(xsize, ysize, zsize, vals)
    raise NotImplementedError(
        f"TabulatedFunction type '{type_name}' not yet supported in Prolix."
    )


# =============================================================================
# 7. Minimum Image Displacement & Geometry
# =============================================================================


def _apply_pbc_triclinic(delta: Array, box_vectors: Array) -> Array:
    """Apply lower-triangular triclinic minimum image convention to delta vector."""
    shift_z = jnp.floor(delta[..., 2] / box_vectors[2, 2] + 0.5)
    delta = delta - shift_z[..., None] * box_vectors[2]
    shift_y = jnp.floor(delta[..., 1] / box_vectors[1, 1] + 0.5)
    delta = delta - shift_y[..., None] * box_vectors[1]
    shift_x = jnp.floor(delta[..., 0] / box_vectors[0, 0] + 0.5)
    delta = delta - shift_x[..., None] * box_vectors[0]
    return delta


def minimum_image_displacement(
    r1: Float[Array, "... 3"],
    r2: Float[Array, "... 3"],
    box_vectors: Float[Array, "3 3"] | None,
) -> Float[Array, "... 3"]:
    """Compute displacement r1 - r2 with optional minimum image convention.

    Supports both orthorhombic (diagonal) and general lower-triangular triclinic boxes.
    """
    dr = r1 - r2
    if box_vectors is None:
        return dr
    return _apply_pbc_triclinic(dr, box_vectors)


# =============================================================================
# 8. Pure JAX Lowering Engine
# =============================================================================

_UNARY_MATH_OPS = {
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
    "log": jnp.log,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "sec": lambda x: 1.0 / jnp.cos(x),
    "csc": lambda x: 1.0 / jnp.sin(x),
    "cot": lambda x: 1.0 / jnp.tan(x),
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "erf": jax.lax.erf,
    "erfc": jax.lax.erfc,
    "abs": jnp.abs,
    "floor": lambda x: jax.lax.stop_gradient(jnp.floor(x)),
    "ceil": lambda x: jax.lax.stop_gradient(jnp.ceil(x)),
    "step": lambda x: jax.lax.stop_gradient(jnp.where(x >= 0.0, 1.0, 0.0)),
    "delta": lambda x: jax.lax.stop_gradient(jnp.where(x == 0.0, 1.0, 0.0)),
    "sq": lambda x: x * x,
    "square": lambda x: x * x,
    "cube": lambda x: x * x * x,
    "recip": lambda x: 1.0 / x,
}

_BINARY_MATH_OPS = {
    "atan2": jnp.arctan2,
    "min": jnp.minimum,
    "max": jnp.maximum,
}


def _lower_geometric_call(
    fn_name: str, args: list[Array], box_vectors: Array | None, env: dict[str, Array]
) -> Array:
    box = box_vectors if box_vectors is not None else env.get("_box_vectors")
    if fn_name == "pointdistance":
        if len(args) != 6:
            raise ValueError("pointdistance expects exactly 6 arguments")
        dx, dy, dz = args[0] - args[3], args[1] - args[4], args[2] - args[5]
        return jnp.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)

    if fn_name == "periodicdistance":
        if len(args) != 6:
            raise ValueError("periodicdistance expects exactly 6 arguments")
        delta = jnp.stack(
            [args[0] - args[3], args[1] - args[4], args[2] - args[5]], axis=-1
        )
        wrapped = _apply_pbc_triclinic(delta, box) if box is not None else delta
        return jnp.sqrt(jnp.sum(wrapped * wrapped, axis=-1) + 1e-12)

    if fn_name == "pointangle":
        if len(args) != 9:
            raise ValueError("pointangle expects exactly 9 arguments")
        p1 = jnp.stack(args[0:3], axis=-1)
        p2 = jnp.stack(args[3:6], axis=-1)
        p3 = jnp.stack(args[6:9], axis=-1)
        v12 = minimum_image_displacement(p1, p2, box)
        v32 = minimum_image_displacement(p3, p2, box)
        dot = jnp.sum(v12 * v32, axis=-1)
        norm1 = jnp.sqrt(jnp.sum(v12 * v12, axis=-1) + 1e-12)
        norm2 = jnp.sqrt(jnp.sum(v32 * v32, axis=-1) + 1e-12)
        cosine = jnp.clip(dot / (norm1 * norm2), -0.9999999, 0.9999999)
        return jnp.arccos(cosine)

    if fn_name == "pointdihedral":
        if len(args) != 12:
            raise ValueError("pointdihedral expects exactly 12 arguments")
        p1 = jnp.stack(args[0:3], axis=-1)
        p2 = jnp.stack(args[3:6], axis=-1)
        p3 = jnp.stack(args[6:9], axis=-1)
        p4 = jnp.stack(args[9:12], axis=-1)
        v12 = minimum_image_displacement(p1, p2, box)
        v32 = minimum_image_displacement(p3, p2, box)
        v34 = minimum_image_displacement(p3, p4, box)
        cross1 = jnp.cross(v12, v32)
        cross2 = jnp.cross(v32, v34)
        norm_v32 = jnp.sqrt(jnp.sum(v32 * v32, axis=-1, keepdims=True) + 1e-12)
        u32 = v32 / norm_v32
        x = jnp.sum(cross1 * cross2, axis=-1)
        y = jnp.sum(jnp.cross(cross1, u32) * cross2, axis=-1)
        return jnp.arctan2(y, x)

    raise ValueError(f"Unknown geometric function: {fn_name!r}")


def _lower_call_node(
    node: CallNode,
    env: dict[str, Array],
    tabulated_fns: dict[str, Callable[..., Array]] | None,
    box_vectors: Array | None,
) -> Array:
    fn_name = node.name.lower()
    if tabulated_fns and (node.name in tabulated_fns or fn_name in tabulated_fns):
        fn = tabulated_fns.get(node.name, tabulated_fns.get(fn_name))
        args = [
            lower_ast_to_jax(a, env, tabulated_fns, box_vectors)
            for a in node.args
        ]
        return fn(*args)

    if fn_name in ("pointdistance", "periodicdistance", "pointangle", "pointdihedral"):
        args = [
            lower_ast_to_jax(a, env, tabulated_fns, box_vectors)
            for a in node.args
        ]
        return _lower_geometric_call(fn_name, args, box_vectors, env)

    args = [lower_ast_to_jax(a, env, tabulated_fns, box_vectors) for a in node.args]
    if fn_name in _UNARY_MATH_OPS:
        return _UNARY_MATH_OPS[fn_name](args[0])
    if fn_name in _BINARY_MATH_OPS:
        return _BINARY_MATH_OPS[fn_name](args[0], args[1])
    if fn_name == "select":
        return jnp.where(args[0] != 0.0, args[1], args[2])

    raise ValueError(f"Unknown Lepton function: {node.name!r}")


def _lower_binary_node(
    node: BinaryOpNode,
    env: dict[str, Array],
    tabulated_fns: dict[str, Callable[..., Array]] | None,
    box_vectors: Array | None,
) -> Array:
    left = lower_ast_to_jax(node.left, env, tabulated_fns, box_vectors)
    if node.op == "^":
        if isinstance(node.right, ConstNode) and node.right.value.is_integer():
            return evaluate_integer_power(left, int(node.right.value))
        if isinstance(node.right, ConstNode) and node.right.value == 0.5:
            return jnp.sqrt(left)
        right = lower_ast_to_jax(node.right, env, tabulated_fns, box_vectors)
        return jnp.power(left, right)

    right = lower_ast_to_jax(node.right, env, tabulated_fns, box_vectors)
    bin_ops = {
        "+": lambda l, r: l + r,
        "-": lambda l, r: l - r,
        "*": lambda l, r: l * r,
        "/": lambda l, r: l / r,
    }
    if node.op in bin_ops:
        return bin_ops[node.op](left, right)
    raise ValueError(f"Unsupported binary operator: {node.op}")


def lower_ast_to_jax(
    node: ASTNode,
    env: dict[str, Array],
    tabulated_fns: dict[str, Callable[..., Array]] | None = None,
    box_vectors: Array | None = None,
) -> Array:
    """Lower an AST node to pure JAX tensor arithmetic within an evaluation environment."""
    if isinstance(node, ConstNode):
        return jnp.asarray(node.value, dtype=jnp.float32)

    if isinstance(node, VarNode):
        if node.name in env:
            return env[node.name]
        raise KeyError(
            f"Variable '{node.name}' not found in Lepton evaluation environment."
        )

    if isinstance(node, UnaryOpNode):
        operand = lower_ast_to_jax(node.operand, env, tabulated_fns, box_vectors)
        if node.op == "-":
            return -operand
        if node.op == "+":
            return operand
        raise ValueError(f"Unsupported unary operator: {node.op}")

    if isinstance(node, BinaryOpNode):
        return _lower_binary_node(node, env, tabulated_fns, box_vectors)

    if isinstance(node, CallNode):
        return _lower_call_node(node, env, tabulated_fns, box_vectors)

    raise TypeError(f"Unknown AST node type: {type(node)}")

    raise TypeError(f"Unknown AST node type: {type(node)}")


def evaluate_lepton_program(
    primary_ast: ASTNode,
    sorted_defs: list[tuple[str, ASTNode]],
    env: dict[str, Array],
    tabulated_fns: dict[str, Callable[..., Array]] | None = None,
    box_vectors: Array | None = None,
) -> Array:
    """Evaluate a topologically sorted Lepton program in sequential environment order."""
    curr_env = dict(env)
    for var_name, def_ast in sorted_defs:
        curr_env[var_name] = lower_ast_to_jax(
            def_ast, curr_env, tabulated_fns, box_vectors
        )
    return lower_ast_to_jax(primary_ast, curr_env, tabulated_fns, box_vectors)


# =============================================================================
# 9. Compiled Custom Force Classes (Equinox PyTrees)
# =============================================================================


class CompiledLeptonForce(eqx.Module):
    """Base class for all compiled Lepton custom forces."""

    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    uses_periodic: bool = eqx.field(static=True)
    energy_param_derivatives: tuple[str, ...] = eqx.field(static=True)
    global_param_names: tuple[str, ...] = eqx.field(static=True)

    def energy_akma(
        self,
        positions_akma: Float[Array, "N 3"],
        box_vectors_akma: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        """Compute potential energy in Prolix AKMA (kcal/mol)."""
        pos_nm = positions_akma * ANGSTROM_TO_NM
        box_nm = (
            box_vectors_akma * ANGSTROM_TO_NM
            if box_vectors_akma is not None
            else None
        )
        e_omm = self.energy_omm(pos_nm, box_nm, global_params)
        return e_omm * KJ_PER_MOL_TO_KCAL_PER_MOL

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        """Compute potential energy in OpenMM native units (kJ/mol)."""
        raise NotImplementedError


class CompiledCustomBondForce(CompiledLeptonForce):
    bond_indices: Int[Array, "M 2"]
    per_bond_params: dict[str, Float[Array, M]]
    eval_fn: Callable[[dict[str, Array], Array | None], Array] = eqx.field(static=True)

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        if self.bond_indices.shape[0] == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        r1 = positions_nm[self.bond_indices[:, 0]]
        r2 = positions_nm[self.bond_indices[:, 1]]
        box = box_vectors_nm if self.uses_periodic else None
        dr = minimum_image_displacement(r1, r2, box)
        r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + 1e-12)

        env = {"r": r, **self.per_bond_params, **global_params}
        u_per_bond = self.eval_fn(env, box)
        return jnp.sum(u_per_bond)


class CompiledCustomAngleForce(CompiledLeptonForce):
    angle_indices: Int[Array, "M 3"]  # (A, B, C) where B is central vertex
    per_angle_params: dict[str, Float[Array, M]]
    eval_fn: Callable[[dict[str, Array], Array | None], Array] = eqx.field(static=True)

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        if self.angle_indices.shape[0] == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        rA = positions_nm[self.angle_indices[:, 0]]
        rB = positions_nm[self.angle_indices[:, 1]]
        rC = positions_nm[self.angle_indices[:, 2]]

        box = box_vectors_nm if self.uses_periodic else None
        v1 = minimum_image_displacement(rA, rB, box)
        v2 = minimum_image_displacement(rC, rB, box)

        d1 = jnp.sqrt(jnp.sum(v1 * v1, axis=-1) + 1e-12)
        d2 = jnp.sqrt(jnp.sum(v2 * v2, axis=-1) + 1e-12)

        cosine = jnp.sum(v1 * v2, axis=-1) / (d1 * d2)
        cosine = jnp.clip(cosine, -0.9999999, 0.9999999)
        theta = jnp.arccos(cosine)

        env = {"theta": theta, **self.per_angle_params, **global_params}
        u_per_angle = self.eval_fn(env, box)
        return jnp.sum(u_per_angle)


class CompiledCustomTorsionForce(CompiledLeptonForce):
    torsion_indices: Int[Array, "M 4"]  # (A, B, C, D)
    per_torsion_params: dict[str, Float[Array, M]]
    eval_fn: Callable[[dict[str, Array], Array | None], Array] = eqx.field(static=True)

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        if self.torsion_indices.shape[0] == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        rA = positions_nm[self.torsion_indices[:, 0]]
        rB = positions_nm[self.torsion_indices[:, 1]]
        rC = positions_nm[self.torsion_indices[:, 2]]
        rD = positions_nm[self.torsion_indices[:, 3]]

        box = box_vectors_nm if self.uses_periodic else None
        b0 = minimum_image_displacement(rA, rB, box)
        b1 = minimum_image_displacement(rC, rB, box)
        b2 = minimum_image_displacement(rD, rC, box)

        b1_norm = jnp.sqrt(jnp.sum(b1 * b1, axis=-1, keepdims=True) + 1e-12)
        b1_unit = b1 / b1_norm

        v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
        w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

        x = jnp.sum(v * w, axis=-1)
        y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

        is_deg = (x * x + y * y) < 1e-20
        safe_x = jnp.where(is_deg, jnp.ones_like(x), x)
        safe_y = jnp.where(is_deg, jnp.zeros_like(y), y)
        theta = jnp.where(is_deg, jnp.zeros_like(x), jnp.arctan2(safe_y, safe_x))

        env = {"theta": theta, **self.per_torsion_params, **global_params}
        u_per_torsion = self.eval_fn(env, box)
        return jnp.sum(u_per_torsion)


class CompiledCustomExternalForce(CompiledLeptonForce):
    particle_indices: Int[Array, M]
    per_particle_params: dict[str, Float[Array, M]]
    eval_fn: Callable[[dict[str, Array], Array | None], Array] = eqx.field(static=True)

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        if self.particle_indices.shape[0] == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        coords = positions_nm[self.particle_indices]
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        box = box_vectors_nm if self.uses_periodic else None
        env = {
            "x": x,
            "y": y,
            "z": z,
            "_box_vectors": box,
            **self.per_particle_params,
            **global_params,
        }
        u_per_particle = self.eval_fn(env, box)
        return jnp.sum(u_per_particle)


class CompiledCustomCompoundBondForce(CompiledLeptonForce):
    particles_per_bond: int = eqx.field(static=True)
    bond_indices: Int[Array, "M P"]
    per_bond_params: dict[str, Float[Array, M]]
    eval_fn: Callable[[dict[str, Array], Array | None], Array] = eqx.field(static=True)

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        if self.bond_indices.shape[0] == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        box = box_vectors_nm if self.uses_periodic else None
        env: dict[str, Array] = {
            "_box_vectors": box,
            **self.per_bond_params,
            **global_params,
        }
        for p in range(self.particles_per_bond):
            p_coords = positions_nm[self.bond_indices[:, p]]
            env[f"x{p + 1}"] = p_coords[:, 0]
            env[f"y{p + 1}"] = p_coords[:, 1]
            env[f"z{p + 1}"] = p_coords[:, 2]

        u_per_bond = self.eval_fn(env, box)
        return jnp.sum(u_per_bond)


class CompiledCustomNonbondedForce(CompiledLeptonForce):
    nonbonded_method: int = eqx.field(static=True)  # 0=NoCutoff, 1=CutoffNonPeriodic, 2=CutoffPeriodic
    cutoff_distance: float = eqx.field(static=True)  # nm
    use_switching: bool = eqx.field(static=True)
    switching_distance: float = eqx.field(static=True)  # nm
    use_dispersion_correction: bool = eqx.field(static=True)
    dispersion_coeff: Float[Array, ""]  # precomputed C_disp (kJ*nm^3/mol)
    chunk_size: int = eqx.field(static=True)  # default 128

    per_particle_params: dict[str, Float[Array, N]]
    computed_value_fns: dict[str, Callable[[dict[str, Array], Array | None], Array]] = eqx.field(static=True)
    exclusion_mask: Bool[Array, "N N"]  # True where pair is excluded
    interaction_groups: tuple[tuple[list[int], list[int]], ...] | None = eqx.field(static=True)
    pair_eval_fn: Callable[[dict[str, Array], Array | None], Array] = eqx.field(static=True)

    def energy_omm(
        self,
        positions_nm: Float[Array, "N 3"],
        box_vectors_nm: Float[Array, "3 3"] | None,
        global_params: dict[str, Array],
    ) -> Float[Array, ""]:
        N = positions_nm.shape[0]
        if N == 0:
            return jnp.asarray(0.0, dtype=positions_nm.dtype)

        box = box_vectors_nm if self.uses_periodic else None

        # 1. Evaluate ComputedValues per particle
        cv_env = {**self.per_particle_params, **global_params}
        computed_values: dict[str, Array] = {}
        for cv_name, cv_fn in self.computed_value_fns.items():
            computed_values[cv_name] = cv_fn(cv_env, box)

        all_particle_data = {**self.per_particle_params, **computed_values}

        # 2. Chunked pair evaluation via jax.lax.map
        pad_len = (self.chunk_size - (N % self.chunk_size)) % self.chunk_size
        N_padded = N + pad_len

        r_padded = jnp.pad(positions_nm, ((0, pad_len), (0, 0)), constant_values=1e6)
        excl_padded = jnp.pad(
            self.exclusion_mask, ((0, pad_len), (0, pad_len)), constant_values=True
        )
        padded_particle_data = {
            k: jnp.pad(v, (0, pad_len), constant_values=0.0)
            for k, v in all_particle_data.items()
        }

        n_chunks = N_padded // self.chunk_size
        chunk_indices = jnp.arange(n_chunks)

        def _eval_chunk(chunk_idx: int) -> Float[Array, ""]:
            start_i = chunk_idx * self.chunk_size
            r_chunk = jax.lax.dynamic_slice(
                r_padded, (start_i, 0), (self.chunk_size, 3)
            )
            excl_chunk = jax.lax.dynamic_slice(
                excl_padded, (start_i, 0), (self.chunk_size, N_padded)
            )

            dr = minimum_image_displacement(
                r_chunk[:, None, :], r_padded[None, :, :], box
            )
            r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + 1e-12)

            i_idx = start_i + jnp.arange(self.chunk_size)[:, None]
            j_idx = jnp.arange(N_padded)[None, :]

            # Valid pair masking: j > i and real indices
            is_valid = (j_idx > i_idx) & (i_idx < N) & (j_idx < N) & (~excl_chunk)
            if self.cutoff_distance > 0:
                is_valid = is_valid & (r < self.cutoff_distance)

            # Prevent 0.0 * inf = NaN in backward pass by guarding r on invalid pairs
            safe_r = jnp.where(is_valid, r, 1.0)

            # Build pair environment
            pair_env: dict[str, Array] = {"r": safe_r, **global_params}
            for k, arr in padded_particle_data.items():
                chunk_arr = jax.lax.dynamic_slice(arr, (start_i,), (self.chunk_size,))
                pair_env[f"{k}1"] = chunk_arr[:, None]
                pair_env[f"{k}2"] = arr[None, :]

            u_pair = self.pair_eval_fn(pair_env, box)

            # 5th-order polynomial switching function
            if self.use_switching and self.cutoff_distance > 0:
                switch_w = self.cutoff_distance - self.switching_distance
                z = (r - self.switching_distance) / jnp.maximum(
                    switch_w, jnp.asarray(1e-12, dtype=r.dtype)
                )
                s = 1.0 - 10.0 * z**3 + 15.0 * z**4 - 6.0 * z**5
                s = jnp.where(
                    r <= self.switching_distance,
                    1.0,
                    jnp.where(r >= self.cutoff_distance, 0.0, s),
                )
                u_pair = u_pair * s

            return jnp.sum(jnp.where(is_valid, u_pair, 0.0))

        chunk_energies = jax.lax.map(_eval_chunk, chunk_indices)
        e_pairwise = jnp.sum(chunk_energies)

        # 3. Isotropic dispersion correction
        e_corr = 0.0
        if self.use_dispersion_correction and box_vectors_nm is not None:
            vol_nm3 = jnp.abs(jnp.linalg.det(box_vectors_nm))
            e_corr = self.dispersion_coeff / jnp.maximum(vol_nm3, 1e-6)

        return e_pairwise + e_corr


# =============================================================================
# 10. Custom Compound Bond AST Rewriter
# =============================================================================


def _rewrite_compound_call(
    node: CallNode, uses_periodic: bool
) -> CallNode:
    fn_name = node.name.lower()
    if fn_name == "distance" and len(node.args) == 2:
        p1_str = str(getattr(node.args[0], "name", getattr(node.args[0], "value", "")))
        p2_str = str(getattr(node.args[1], "name", getattr(node.args[1], "value", "")))
        i1 = p1_str.lstrip("pP")
        i2 = p2_str.lstrip("pP")
        geom_fn = "periodicdistance" if uses_periodic else "pointdistance"
        coords = (
            VarNode(f"x{i1}"), VarNode(f"y{i1}"), VarNode(f"z{i1}"),
            VarNode(f"x{i2}"), VarNode(f"y{i2}"), VarNode(f"z{i2}"),
        )
        return CallNode(geom_fn, coords)

    if fn_name == "angle" and len(node.args) == 3:
        p1_str = str(getattr(node.args[0], "name", getattr(node.args[0], "value", "")))
        p2_str = str(getattr(node.args[1], "name", getattr(node.args[1], "value", "")))
        p3_str = str(getattr(node.args[2], "name", getattr(node.args[2], "value", "")))
        i1, i2, i3 = p1_str.lstrip("pP"), p2_str.lstrip("pP"), p3_str.lstrip("pP")
        coords = (
            VarNode(f"x{i1}"), VarNode(f"y{i1}"), VarNode(f"z{i1}"),
            VarNode(f"x{i2}"), VarNode(f"y{i2}"), VarNode(f"z{i2}"),
            VarNode(f"x{i3}"), VarNode(f"y{i3}"), VarNode(f"z{i3}"),
        )
        return CallNode("pointangle", coords)

    if fn_name == "dihedral" and len(node.args) == 4:
        p1_str = str(getattr(node.args[0], "name", getattr(node.args[0], "value", "")))
        p2_str = str(getattr(node.args[1], "name", getattr(node.args[1], "value", "")))
        p3_str = str(getattr(node.args[2], "name", getattr(node.args[2], "value", "")))
        p4_str = str(getattr(node.args[3], "name", getattr(node.args[3], "value", "")))
        i1, i2, i3, i4 = p1_str.lstrip("pP"), p2_str.lstrip("pP"), p3_str.lstrip("pP"), p4_str.lstrip("pP")
        coords = (
            VarNode(f"x{i1}"), VarNode(f"y{i1}"), VarNode(f"z{i1}"),
            VarNode(f"x{i2}"), VarNode(f"y{i2}"), VarNode(f"z{i2}"),
            VarNode(f"x{i3}"), VarNode(f"y{i3}"), VarNode(f"z{i3}"),
            VarNode(f"x{i4}"), VarNode(f"y{i4}"), VarNode(f"z{i4}"),
        )
        return CallNode("pointdihedral", coords)

    args = tuple(_rewrite_compound_bond_calls(a, uses_periodic) for a in node.args)
    return CallNode(node.name, args)


def _rewrite_compound_bond_calls(node: ASTNode, uses_periodic: bool) -> ASTNode:
    """Rewrite distance(p1, p2) / angle / dihedral into coordinate-based geometric calls."""
    if isinstance(node, CallNode):
        return _rewrite_compound_call(node, uses_periodic)
    if isinstance(node, UnaryOpNode):
        return UnaryOpNode(node.op, _rewrite_compound_bond_calls(node.operand, uses_periodic))
    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.op,
            _rewrite_compound_bond_calls(node.left, uses_periodic),
            _rewrite_compound_bond_calls(node.right, uses_periodic),
        )
    return node


# =============================================================================
# 11. Compilation Factory (`compile_custom_force`)
# =============================================================================


def compile_custom_force(
    force: openmm.Force,
    config: ProlixOpenMMConfig | None = None,
) -> CompiledLeptonForce:
    """Compile an arbitrary OpenMM custom force into an Equinox JAX CompiledLeptonForce."""
    if config is None:
        config = ProlixOpenMMConfig()

    force_name = force.__class__.__name__
    uses_periodic = (
        force.usesPeriodicBoundaryConditions()
        if hasattr(force, "usesPeriodicBoundaryConditions")
        else False
    )
    force_group = force.getForceGroup()

    # Global parameter names
    num_globals = (
        force.getNumGlobalParameters()
        if hasattr(force, "getNumGlobalParameters")
        else 0
    )
    global_names = tuple(force.getGlobalParameterName(i) for i in range(num_globals))

    # Parameter derivatives
    num_derivs = (
        force.getNumEnergyParameterDerivatives()
        if hasattr(force, "getNumEnergyParameterDerivatives")
        else 0
    )
    deriv_names = tuple(
        force.getEnergyParameterDerivativeName(i) for i in range(num_derivs)
    )

    # Compile tabulated functions
    tabulated_fns: dict[str, Callable[..., Array]] = {}
    if hasattr(force, "getNumTabulatedFunctions"):
        for i in range(force.getNumTabulatedFunctions()):
            fn_obj = force.getTabulatedFunction(i)
            fn_name = force.getTabulatedFunctionName(i)
            tabulated_fns[fn_name] = compile_tabulated_function(fn_obj)

    # 1. CustomBondForce
    if isinstance(force, openmm.CustomBondForce):
        primary_ast, sorted_defs = resolve_lepton_program(force.getEnergyFunction())

        def _eval_bond(env: dict[str, Array], box: Array | None) -> Array:
            return evaluate_lepton_program(
                primary_ast, sorted_defs, env, tabulated_fns, box
            )

        num_bonds = force.getNumBonds()
        num_params = force.getNumPerBondParameters()
        param_names = [force.getPerBondParameterName(i) for i in range(num_params)]

        b_indices = []
        b_params: dict[str, list[float]] = {p: [] for p in param_names}
        for i in range(num_bonds):
            p1, p2, params = force.getBondParameters(i)
            b_indices.append([p1, p2])
            for p_idx, p_name in enumerate(param_names):
                b_params[p_name].append(params[p_idx])

        bond_idx_arr = jnp.asarray(b_indices, dtype=jnp.int32).reshape((-1, 2))
        per_bond_arrs = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in b_params.items()
        }

        return CompiledCustomBondForce(
            name=force_name,
            force_group=force_group,
            uses_periodic=uses_periodic,
            energy_param_derivatives=deriv_names,
            global_param_names=global_names,
            bond_indices=bond_idx_arr,
            per_bond_params=per_bond_arrs,
            eval_fn=_eval_bond,
        )

    # 2. CustomAngleForce
    if isinstance(force, openmm.CustomAngleForce):
        primary_ast, sorted_defs = resolve_lepton_program(force.getEnergyFunction())

        def _eval_angle(env: dict[str, Array], box: Array | None) -> Array:
            return evaluate_lepton_program(
                primary_ast, sorted_defs, env, tabulated_fns, box
            )

        num_angles = force.getNumAngles()
        num_params = force.getNumPerAngleParameters()
        param_names = [force.getPerAngleParameterName(i) for i in range(num_params)]

        a_indices = []
        a_params: dict[str, list[float]] = {p: [] for p in param_names}
        for i in range(num_angles):
            p1, p2, p3, params = force.getAngleParameters(i)
            a_indices.append([p1, p2, p3])
            for p_idx, p_name in enumerate(param_names):
                a_params[p_name].append(params[p_idx])

        angle_idx_arr = jnp.asarray(a_indices, dtype=jnp.int32).reshape((-1, 3))
        per_angle_arrs = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in a_params.items()
        }

        return CompiledCustomAngleForce(
            name=force_name,
            force_group=force_group,
            uses_periodic=uses_periodic,
            energy_param_derivatives=deriv_names,
            global_param_names=global_names,
            angle_indices=angle_idx_arr,
            per_angle_params=per_angle_arrs,
            eval_fn=_eval_angle,
        )

    # 3. CustomTorsionForce
    if isinstance(force, openmm.CustomTorsionForce):
        primary_ast, sorted_defs = resolve_lepton_program(force.getEnergyFunction())

        def _eval_torsion(env: dict[str, Array], box: Array | None) -> Array:
            return evaluate_lepton_program(
                primary_ast, sorted_defs, env, tabulated_fns, box
            )

        num_torsions = force.getNumTorsions()
        num_params = force.getNumPerTorsionParameters()
        param_names = [force.getPerTorsionParameterName(i) for i in range(num_params)]

        t_indices = []
        t_params: dict[str, list[float]] = {p: [] for p in param_names}
        for i in range(num_torsions):
            p1, p2, p3, p4, params = force.getTorsionParameters(i)
            t_indices.append([p1, p2, p3, p4])
            for p_idx, p_name in enumerate(param_names):
                t_params[p_name].append(params[p_idx])

        torsion_idx_arr = jnp.asarray(t_indices, dtype=jnp.int32).reshape((-1, 4))
        per_torsion_arrs = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in t_params.items()
        }

        return CompiledCustomTorsionForce(
            name=force_name,
            force_group=force_group,
            uses_periodic=uses_periodic,
            energy_param_derivatives=deriv_names,
            global_param_names=global_names,
            torsion_indices=torsion_idx_arr,
            per_torsion_params=per_torsion_arrs,
            eval_fn=_eval_torsion,
        )

    # 4. CustomExternalForce
    if isinstance(force, openmm.CustomExternalForce):
        primary_ast, sorted_defs = resolve_lepton_program(force.getEnergyFunction())

        def _eval_external(env: dict[str, Array], box: Array | None) -> Array:
            return evaluate_lepton_program(
                primary_ast, sorted_defs, env, tabulated_fns, box
            )

        num_particles = force.getNumParticles()
        num_params = force.getNumPerParticleParameters()
        param_names = [
            force.getPerParticleParameterName(i) for i in range(num_params)
        ]

        p_indices = []
        p_params: dict[str, list[float]] = {p: [] for p in param_names}
        for i in range(num_particles):
            p_idx, params = force.getParticleParameters(i)
            p_indices.append(p_idx)
            for param_i, p_name in enumerate(param_names):
                p_params[p_name].append(params[param_i])

        particle_idx_arr = jnp.asarray(p_indices, dtype=jnp.int32)
        per_particle_arrs = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in p_params.items()
        }

        return CompiledCustomExternalForce(
            name=force_name,
            force_group=force_group,
            uses_periodic=uses_periodic,
            energy_param_derivatives=deriv_names,
            global_param_names=global_names,
            particle_indices=particle_idx_arr,
            per_particle_params=per_particle_arrs,
            eval_fn=_eval_external,
        )

    # 5. CustomCompoundBondForce
    if isinstance(force, openmm.CustomCompoundBondForce):
        raw_primary_ast, raw_sorted_defs = resolve_lepton_program(
            force.getEnergyFunction()
        )
        # Rewrite distance/angle/dihedral calls to point functions
        primary_ast = _rewrite_compound_bond_calls(raw_primary_ast, uses_periodic)
        sorted_defs = [
            (name, _rewrite_compound_bond_calls(d_ast, uses_periodic))
            for name, d_ast in raw_sorted_defs
        ]

        def _eval_compound(env: dict[str, Array], box: Array | None) -> Array:
            return evaluate_lepton_program(
                primary_ast, sorted_defs, env, tabulated_fns, box
            )

        num_bonds = force.getNumBonds()
        particles_per_bond = force.getNumParticlesPerBond()
        num_params = force.getNumPerBondParameters()
        param_names = [force.getPerBondParameterName(i) for i in range(num_params)]

        cb_indices = []
        cb_params: dict[str, list[float]] = {p: [] for p in param_names}
        for i in range(num_bonds):
            particles, params = force.getBondParameters(i)
            cb_indices.append(particles)
            for p_idx, p_name in enumerate(param_names):
                cb_params[p_name].append(params[p_idx])

        cb_indices_arr = jnp.asarray(cb_indices, dtype=jnp.int32).reshape(
            (-1, particles_per_bond)
        )
        per_bond_arrs = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in cb_params.items()
        }

        return CompiledCustomCompoundBondForce(
            name=force_name,
            force_group=force_group,
            uses_periodic=uses_periodic,
            energy_param_derivatives=deriv_names,
            global_param_names=global_names,
            particles_per_bond=particles_per_bond,
            bond_indices=cb_indices_arr,
            per_bond_params=per_bond_arrs,
            eval_fn=_eval_compound,
        )

    # 6. CustomNonbondedForce
    if isinstance(force, openmm.CustomNonbondedForce):
        primary_ast, sorted_defs = resolve_lepton_program(force.getEnergyFunction())

        def _pair_eval(env: dict[str, Array], box: Array | None) -> Array:
            return evaluate_lepton_program(
                primary_ast, sorted_defs, env, tabulated_fns, box
            )

        # Computed values
        computed_value_fns: dict[str, Any] = {}
        for cv_idx in range(force.getNumComputedValues()):
            cv_name, cv_expr = force.getComputedValueParameters(cv_idx)
            cv_primary, cv_defs = resolve_lepton_program(cv_expr)

            def _cv_fn(
                env: dict[str, Array],
                box: Array | None,
                p_ast=cv_primary,
                d_ast=cv_defs,
            ) -> Array:
                return evaluate_lepton_program(
                    p_ast, d_ast, env, tabulated_fns, box
                )

            computed_value_fns[cv_name] = _cv_fn

        # Per-particle parameters
        num_particles = force.getNumParticles()
        num_params = force.getNumPerParticleParameters()
        param_names = [
            force.getPerParticleParameterName(i) for i in range(num_params)
        ]

        p_data: dict[str, list[float]] = {p: [] for p in param_names}
        for i in range(num_particles):
            params = force.getParticleParameters(i)
            for p_idx, p_name in enumerate(param_names):
                p_data[p_name].append(params[p_idx])

        per_particle_arrs = {
            k: jnp.asarray(v, dtype=jnp.float32) for k, v in p_data.items()
        }

        # Exclusion mask
        excl_mask = jnp.zeros((num_particles, num_particles), dtype=bool)
        for i in range(force.getNumExclusions()):
            p1, p2 = force.getExclusionParticles(i)
            excl_mask = excl_mask.at[p1, p2].set(True).at[p2, p1].set(True)

        # Self-exclusions
        excl_mask = excl_mask.at[jnp.diag_indices(num_particles)].set(True)

        # Interaction groups
        interaction_groups = None
        if force.getNumInteractionGroups() > 0:
            groups_list = []
            for ig_idx in range(force.getNumInteractionGroups()):
                g1, g2 = force.getInteractionGroupParameters(ig_idx)
                groups_list.append((list(g1), list(g2)))
            interaction_groups = tuple(groups_list)

        # Cutoff, switching, dispersion
        nonbonded_method = force.getNonbondedMethod()
        raw_cutoff = force.getCutoffDistance()
        cutoff_dist = float(
            raw_cutoff.value_in_unit(openmm.unit.nanometer)
            if hasattr(raw_cutoff, "value_in_unit")
            else raw_cutoff
        )
        use_switching = force.getUseSwitchingFunction()
        raw_switching = force.getSwitchingDistance()
        switching_dist = float(
            raw_switching.value_in_unit(openmm.unit.nanometer)
            if hasattr(raw_switching, "value_in_unit")
            else raw_switching
        )
        use_dispersion = force.getUseLongRangeCorrection()

        chunk_size = (
            config.custom_nonbonded_chunk_size
            if config is not None
            else 128
        )

        return CompiledCustomNonbondedForce(
            name=force_name,
            force_group=force_group,
            uses_periodic=uses_periodic,
            energy_param_derivatives=deriv_names,
            global_param_names=global_names,
            nonbonded_method=nonbonded_method,
            cutoff_distance=cutoff_dist,
            use_switching=use_switching,
            switching_distance=switching_dist,
            use_dispersion_correction=use_dispersion,
            dispersion_coeff=jnp.asarray(0.0, dtype=jnp.float32),
            chunk_size=chunk_size,
            per_particle_params=per_particle_arrs,
            computed_value_fns=computed_value_fns,
            exclusion_mask=excl_mask,
            interaction_groups=interaction_groups,
            pair_eval_fn=_pair_eval,
        )

    raise NotImplementedError(
        f"Custom force class '{force_name}' is not yet supported in Prolix."
    )


# =============================================================================
# 12. Autodiff Pipeline for Forces & Parameter Derivatives
# =============================================================================


def evaluate_custom_force(
    force: CompiledLeptonForce,
    positions_akma: Float[Array, "N 3"],
    box_vectors_akma: Float[Array, "3 3"] | None,
    global_params: dict[str, Array],
    compute_derivatives: bool = False,
) -> tuple[Float[Array, ""], Float[Array, "N 3"], dict[str, Float[Array, ""]]]:
    """Pure JAX evaluation of custom force energy, forces, and parameter derivatives.

    Returns:
        energy_akma: Scalar potential energy in kcal/mol.
        forces_akma: Negative gradient in kcal/(mol*Å).
        param_derivatives_omm: Dict of dE/dlambda in OpenMM native units (kJ/mol per unit lambda).
    """

    def _e_fn(pos: Array) -> Array:
        return force.energy_akma(pos, box_vectors_akma, global_params)

    e_akma, grad_pos = jax.value_and_grad(_e_fn)(positions_akma)
    forces_akma = -grad_pos

    param_derivs: dict[str, Float[Array, ""]] = {}
    if compute_derivatives and len(force.energy_param_derivatives) > 0:
        pos_nm = positions_akma * ANGSTROM_TO_NM
        box_nm = (
            box_vectors_akma * ANGSTROM_TO_NM
            if box_vectors_akma is not None
            else None
        )

        def _e_param_fn(gp: dict[str, Array]) -> Array:
            return force.energy_omm(pos_nm, box_nm, gp)

        _, grad_params = jax.value_and_grad(_e_param_fn)(global_params)
        for param_name in force.energy_param_derivatives:
            if param_name in grad_params:
                param_derivs[param_name] = grad_params[param_name]

    return e_akma, forces_akma, param_derivs
