# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted for JAX/Pallas Triton from NVIDIA MatX `matx::fltflt` (`fltflt.hpp`):
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the BSD 3-Clause License (see root NOTICE for full text).
#
# Algorithmic references:
# - D. E. Knuth, TAOCP Vol. 2 (TwoSum).
# - T. J. Dekker, Numerische Mathematik 18 (1971), 224-242 (QuickTwoSum, Veltkamp split).
# - A. Thall, "Extended-Precision Floating-Point Numbers for GPU Computation", SIGGRAPH 2006.
# - H. Zhang & A. Aiken, SC'25 (FPAN depth-10 branch-free float-float addition).

"""Pallas/Triton float-float ``(hi, lo)`` ~48-bit compensated arithmetic (v3.6 P1-A).

Provides IEEE-754 round-to-nearest-even (``rn.f32``, non-FTZ, non-contracted)
compensated ``(hi, lo)`` primitives on 32-bit FP32 CUDA pipes via
``jax.experimental.pallas.triton.elementwise_inline_asm`` inside GPU Pallas
kernels, with an exact contraction-proof pure-JAX fallback
(``jax.lax.optimization_barrier`` + 12-bit Veltkamp/Dekker split) for CPU,
``interpret=True``, XLA host/JIT calls, and ``prolix.export`` StableHLO lowering.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import triton as plt

from prolix.export import in_export_context

_IN_TRITON_ASM_CONTEXT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_IN_TRITON_ASM_CONTEXT", default=False
)


class FltFlt(NamedTuple):
    """Normalized non-overlapping float32 pair ``(hi, lo)`` with ``|lo| <= 0.5 * ulp(hi)``."""

    hi: jax.Array
    lo: jax.Array


FltFltLike = FltFlt | tuple[jax.Array, jax.Array]


@contextlib.contextmanager
def fltflt_triton_asm_context(enabled: bool = True):
    """Enable PTX ``rn.f32`` ``elementwise_inline_asm`` lowering inside a GPU Pallas kernel."""
    use_asm = bool(enabled) and (not in_export_context()) and (jax.default_backend() == "gpu")
    token = _IN_TRITON_ASM_CONTEXT.set(use_asm)
    try:
        yield
    finally:
        _IN_TRITON_ASM_CONTEXT.reset(token)


def should_use_triton_asm(
    use_triton_asm: bool | None = None,
    *,
    interpret: bool = False,
) -> bool:
    """Return True iff PTX ``elementwise_inline_asm`` should be emitted."""
    if interpret or in_export_context() or jax.default_backend() != "gpu":
        return False
    if use_triton_asm is not None:
        return bool(use_triton_asm)
    return _IN_TRITON_ASM_CONTEXT.get()


def _as_f32_same_shape(*xs: jax.Array | float) -> tuple[list[jax.Array], tuple[int, ...]]:
    arrs = [jnp.asarray(x, dtype=jnp.float32) for x in xs]
    bcast = list(jnp.broadcast_arrays(*arrs))
    orig_shape = bcast[0].shape
    if orig_shape == ():
        bcast = [a.reshape((1,)) for a in bcast]
    return bcast, orig_shape


def _restore_shape(x: jax.Array, orig_shape: tuple[int, ...]) -> jax.Array:
    if orig_shape == ():
        return x[0]
    return x


def _ptx_f32_op(asm: str, constraints: str, *xs: jax.Array | float) -> jax.Array:
    bcast, orig_shape = _as_f32_same_shape(*xs)
    [out] = plt.elementwise_inline_asm(
        asm,
        args=bcast,
        constraints=constraints,
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(bcast[0].shape, jnp.float32)],
    )
    return _restore_shape(out, orig_shape)


def fadd_rn(
    a: jax.Array | float,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> jax.Array:
    """IEEE round-to-nearest-even ``a + b`` in ``float32`` without contraction or FTZ."""
    if should_use_triton_asm(use_triton_asm, interpret=interpret):
        return _ptx_f32_op("add.rn.f32 $0, $1, $2;", "=f,f,f", a, b)
    return jnp.asarray(a, dtype=jnp.float32) + jnp.asarray(b, dtype=jnp.float32)


def fsub_rn(
    a: jax.Array | float,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> jax.Array:
    """IEEE round-to-nearest-even ``a - b`` in ``float32`` without contraction or FTZ."""
    if should_use_triton_asm(use_triton_asm, interpret=interpret):
        return _ptx_f32_op("sub.rn.f32 $0, $1, $2;", "=f,f,f", a, b)
    return jnp.asarray(a, dtype=jnp.float32) - jnp.asarray(b, dtype=jnp.float32)


def fmul_rn(
    a: jax.Array | float,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> jax.Array:
    """IEEE round-to-nearest-even ``a * b`` in ``float32`` without FMA fusion or FTZ."""
    if should_use_triton_asm(use_triton_asm, interpret=interpret):
        return _ptx_f32_op("mul.rn.f32 $0, $1, $2;", "=f,f,f", a, b)
    return jnp.asarray(a, dtype=jnp.float32) * jnp.asarray(b, dtype=jnp.float32)


def fneg_f32(
    a: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Negate a ``float32`` array without ``lax.neg_p`` on ``TypedNdArray`` constants."""
    af = jnp.asarray(a, dtype=jnp.float32)
    z = jnp.zeros_like(af)
    return fsub_rn(z, af, use_triton_asm=use_triton_asm, interpret=interpret)


def _veltkamp_split_f32(a: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Exact 12-bit + 12-bit Dekker/Veltkamp split of an IEEE binary32 mantissa."""
    split_c = jnp.asarray(4097.0, dtype=jnp.float32)
    c = split_c * a
    ahi = c - (c - a)
    alo = a - ahi
    return ahi, alo


def fmaf_rn(
    a: jax.Array | float,
    b: jax.Array | float,
    c: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> jax.Array:
    """IEEE round-to-nearest-even fused multiply-add ``a * b + c`` in ``float32``."""
    if should_use_triton_asm(use_triton_asm, interpret=interpret):
        return _ptx_f32_op("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", a, b, c)
    af = jnp.asarray(a, dtype=jnp.float32)
    bf = jnp.asarray(b, dtype=jnp.float32)
    cf = jnp.asarray(c, dtype=jnp.float32)
    p_hi = af * bf
    ah, al = _veltkamp_split_f32(af)
    bh, bl = _veltkamp_split_f32(bf)
    p_lo = ((ah * bh - p_hi) + ah * bl + al * bh) + al * bl
    s_hi = p_hi + cf
    v = s_hi - p_hi
    s_err = (p_hi - (s_hi - v)) + (cf - v)
    return s_hi + (p_lo + s_err)


def ff_from_f32(x: jax.Array | float) -> FltFlt:
    """Lift a ``float32`` array to ``FltFlt(x, 0.0f)``."""
    xf = jnp.asarray(x, dtype=jnp.float32)
    return FltFlt(xf, jnp.zeros_like(xf))


def ff_from_float64(x: jax.Array) -> FltFlt:
    """Split a ``float64`` (or ``float32``) array into a normalized ``FltFlt(hi, lo)`` pair."""
    xa = jnp.asarray(x)
    hi = xa.astype(jnp.float32)
    if xa.dtype == jnp.float64:
        lo = (xa - hi.astype(jnp.float64)).astype(jnp.float32)
        return FltFlt(hi, lo)
    return FltFlt(hi, jnp.zeros_like(hi))


def ff_to_float64(a: FltFltLike) -> jax.Array:
    """Recombine ``FltFlt(hi, lo)`` into ``float64`` (when ``jax_enable_x64`` is active)."""
    hi, lo = a[0], a[1]
    return hi.astype(jnp.float64) + lo.astype(jnp.float64)


def ff_to_f32(
    a: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Round ``FltFlt(hi, lo)`` to ``float32``."""
    return fadd_rn(a[0], a[1], use_triton_asm=use_triton_asm, interpret=interpret)


def ff_neg(
    a: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Exact negation ``(-a.hi, -a.lo)``."""
    return FltFlt(
        fneg_f32(a[0], use_triton_asm=use_triton_asm, interpret=interpret),
        fneg_f32(a[1], use_triton_asm=use_triton_asm, interpret=interpret),
    )


def ff_where(cond: jax.Array, a: FltFltLike, b: FltFltLike) -> FltFlt:
    """Elementwise ``jnp.where`` over ``FltFlt`` pairs."""
    return FltFlt(jnp.where(cond, a[0], b[0]), jnp.where(cond, a[1], b[1]))


def two_sum(
    a: jax.Array | float,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Knuth 6-op error-free ``a + b = s + e`` without magnitude ordering assumption."""
    s = fadd_rn(a, b, use_triton_asm=use_triton_asm, interpret=interpret)
    v = fsub_rn(s, a, use_triton_asm=use_triton_asm, interpret=interpret)
    sv = fsub_rn(s, v, use_triton_asm=use_triton_asm, interpret=interpret)
    da = fsub_rn(a, sv, use_triton_asm=use_triton_asm, interpret=interpret)
    db = fsub_rn(b, v, use_triton_asm=use_triton_asm, interpret=interpret)
    e = fadd_rn(da, db, use_triton_asm=use_triton_asm, interpret=interpret)
    return FltFlt(s, e)


def fast_two_sum(
    a: jax.Array | float,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Dekker 3-op QuickTwoSum ``a + b = s + e`` assuming ``|a| >= |b|`` (or ``a == 0``)."""
    s = fadd_rn(a, b, use_triton_asm=use_triton_asm, interpret=interpret)
    sa = fsub_rn(s, a, use_triton_asm=use_triton_asm, interpret=interpret)
    e = fsub_rn(b, sa, use_triton_asm=use_triton_asm, interpret=interpret)
    return FltFlt(s, e)


def two_prod_fma(
    a: jax.Array | float,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Hida/Dekker error-free ``a * b = x + y`` (2 PTX ops on Triton GPU, Veltkamp split on XLA)."""
    if should_use_triton_asm(use_triton_asm, interpret=interpret):
        x = _ptx_f32_op("mul.rn.f32 $0, $1, $2;", "=f,f,f", a, b)
        neg_x = fneg_f32(x, use_triton_asm=True, interpret=False)
        y = _ptx_f32_op("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", a, b, neg_x)
        return FltFlt(x, y)
    af = jnp.asarray(a, dtype=jnp.float32)
    bf = jnp.asarray(b, dtype=jnp.float32)
    x = jax.lax.optimization_barrier(af * bf)
    ah, al = _veltkamp_split_f32(af)
    bh, bl = _veltkamp_split_f32(bf)
    t1 = jax.lax.optimization_barrier(x - jax.lax.optimization_barrier(ah * bh))
    t2 = jax.lax.optimization_barrier(t1 - jax.lax.optimization_barrier(al * bh))
    t3 = jax.lax.optimization_barrier(t2 - jax.lax.optimization_barrier(ah * bl))
    y = jax.lax.optimization_barrier(jax.lax.optimization_barrier(al * bl) - t3)
    return FltFlt(x, y)


def ff_add_f32(
    a: FltFltLike,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Add ``FltFlt`` and ``float32`` (``fltflt.hpp:312-317``, 10 FP32 ops)."""
    s = two_sum(a[0], b, use_triton_asm=use_triton_asm, interpret=interpret)
    s_lo = fadd_rn(s.lo, a[1], use_triton_asm=use_triton_asm, interpret=interpret)
    return fast_two_sum(s.hi, s_lo, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_add(
    a: FltFltLike,
    b: FltFltLike | jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Zhang & Aiken (SC'25) FPAN branch-free ``fltflt`` addition (``fltflt.hpp:298-305``)."""
    if not isinstance(b, tuple):
        return ff_add_f32(a, b, use_triton_asm=use_triton_asm, interpret=interpret)
    s = two_sum(a[0], b[0], use_triton_asm=use_triton_asm, interpret=interpret)
    t = two_sum(a[1], b[1], use_triton_asm=use_triton_asm, interpret=interpret)
    q = fast_two_sum(s.hi, t.hi, use_triton_asm=use_triton_asm, interpret=interpret)
    st_lo = fadd_rn(s.lo, t.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    stq_lo = fadd_rn(st_lo, q.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    return fast_two_sum(q.hi, stq_lo, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_sub_f32(
    a: FltFltLike,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Subtract ``float32`` ``b`` from ``FltFlt`` ``a``."""
    nb = fneg_f32(b, use_triton_asm=use_triton_asm, interpret=interpret)
    return ff_add_f32(a, nb, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_sub(
    a: FltFltLike,
    b: FltFltLike | jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Subtract ``b`` from ``a`` (``fltflt.hpp:352-361``)."""
    if not isinstance(b, tuple):
        return ff_sub_f32(a, b, use_triton_asm=use_triton_asm, interpret=interpret)
    b_ff: tuple[jax.Array, jax.Array] = (b[0], b[1])
    return ff_add(
        a,
        ff_neg(b_ff, use_triton_asm=use_triton_asm, interpret=interpret),
        use_triton_asm=use_triton_asm,
        interpret=interpret,
    )


def ff_mul_f32(
    a: FltFltLike,
    b: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Multiply ``FltFlt`` ``a`` by ``float32`` ``b`` (``fltflt.hpp:382-387``, 6 FP32 ops)."""
    p = two_prod_fma(a[0], b, use_triton_asm=use_triton_asm, interpret=interpret)
    p_lo = fmaf_rn(a[1], b, p.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    return fast_two_sum(p.hi, p_lo, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_mul(
    a: FltFltLike,
    b: FltFltLike | jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Thall ``df64_mult`` ``fltflt`` multiplication (``fltflt.hpp:372-378``, 7 FP32 ops)."""
    if not isinstance(b, tuple):
        return ff_mul_f32(a, b, use_triton_asm=use_triton_asm, interpret=interpret)
    p = two_prod_fma(a[0], b[0], use_triton_asm=use_triton_asm, interpret=interpret)
    p_lo = fmaf_rn(a[0], b[1], p.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    p_lo = fmaf_rn(a[1], b[0], p_lo, use_triton_asm=use_triton_asm, interpret=interpret)
    return fast_two_sum(p.hi, p_lo, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_fma(
    a: FltFltLike,
    b: FltFltLike,
    c: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Two-normalization ``a * b + c`` including ``a.lo * b.lo`` (``fltflt.hpp:397-418``)."""
    p = two_prod_fma(a[0], b[0], use_triton_asm=use_triton_asm, interpret=interpret)
    p_lo = fmaf_rn(a[0], b[1], p.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    p_lo = fmaf_rn(a[1], b[0], p_lo, use_triton_asm=use_triton_asm, interpret=interpret)
    s = two_sum(p.hi, c[0], use_triton_asm=use_triton_asm, interpret=interpret)
    s_lo = fadd_rn(s.lo, p_lo, use_triton_asm=use_triton_asm, interpret=interpret)
    s = fast_two_sum(s.hi, s_lo, use_triton_asm=use_triton_asm, interpret=interpret)
    s_lo2 = fadd_rn(s.lo, c[1], use_triton_asm=use_triton_asm, interpret=interpret)
    s_lo2 = fmaf_rn(a[1], b[1], s_lo2, use_triton_asm=use_triton_asm, interpret=interpret)
    return fast_two_sum(s.hi, s_lo2, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_div(
    a: FltFltLike,
    b: FltFltLike | jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Karp ``fltflt_div`` ``a / b`` (``fltflt.hpp:563-579``), zero-safe when ``b.hi == 0``."""
    b_pair: tuple[jax.Array, jax.Array] = (b[0], b[1]) if isinstance(b, tuple) else ff_from_f32(b)
    b_hi = jnp.asarray(b_pair[0], dtype=jnp.float32)
    z = jnp.zeros_like(b_hi)
    one = jnp.ones_like(b_hi)
    nz = b_hi != z
    safe_b_hi = jnp.where(nz, b_hi, one)
    xn = jnp.where(nz, one / safe_b_hi, z)
    yn = fmul_rn(a[0], xn, use_triton_asm=use_triton_asm, interpret=interpret)
    neg_yn = fneg_f32(yn, use_triton_asm=use_triton_asm, interpret=interpret)
    diff = ff_fma(
        ff_from_f32(neg_yn),
        b_pair,
        a,
        use_triton_asm=use_triton_asm,
        interpret=interpret,
    )
    prod = two_prod_fma(xn, diff.hi, use_triton_asm=use_triton_asm, interpret=interpret)
    res = ff_add_f32(prod, yn, use_triton_asm=use_triton_asm, interpret=interpret)
    return ff_where(nz, res, FltFlt(z, z))


def ff_sqrt(
    a: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Karp full ~48-bit ``fltflt_sqrt`` (``fltflt.hpp:695-702``), safe at ``a.hi <= 0``."""
    a_hi = jnp.asarray(a[0], dtype=jnp.float32)
    z = jnp.zeros_like(a_hi)
    one = jnp.ones_like(a_hi)
    half = jnp.asarray(0.5, dtype=jnp.float32)
    pos = a_hi > z
    safe_hi = jnp.where(pos, a_hi, one)
    xn = jnp.where(pos, jax.lax.rsqrt(safe_hi), z)
    yn = fmul_rn(a_hi, xn, use_triton_asm=use_triton_asm, interpret=interpret)
    ynsqr = two_prod_fma(yn, yn, use_triton_asm=use_triton_asm, interpret=interpret)
    diff = ff_sub(a, ynsqr, use_triton_asm=use_triton_asm, interpret=interpret)
    half_diff = fmul_rn(half, diff.hi, use_triton_asm=use_triton_asm, interpret=interpret)
    prod = two_prod_fma(xn, half_diff, use_triton_asm=use_triton_asm, interpret=interpret)
    res = ff_add_f32(prod, yn, use_triton_asm=use_triton_asm, interpret=interpret)
    return ff_where(pos, res, FltFlt(z, z))


def ff_sqrt_fast(
    a: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """7-op FMA-residual ``fltflt_sqrt_fast`` (~45-bit mantissa, ``fltflt.hpp:712-720``)."""
    a_hi = jnp.asarray(a[0], dtype=jnp.float32)
    z = jnp.zeros_like(a_hi)
    one = jnp.ones_like(a_hi)
    half = jnp.asarray(0.5, dtype=jnp.float32)
    pos = a_hi > z
    safe_hi = jnp.where(pos, a_hi, one)
    xn = jnp.where(pos, jax.lax.rsqrt(safe_hi), z)
    yn = fmul_rn(a_hi, xn, use_triton_asm=use_triton_asm, interpret=interpret)
    neg_yn = fneg_f32(yn, use_triton_asm=use_triton_asm, interpret=interpret)
    rem_hi = fmaf_rn(neg_yn, yn, a_hi, use_triton_asm=use_triton_asm, interpret=interpret)
    residual = fadd_rn(rem_hi, a[1], use_triton_asm=use_triton_asm, interpret=interpret)
    half_xn = fmul_rn(xn, half, use_triton_asm=use_triton_asm, interpret=interpret)
    corr = fmul_rn(half_xn, residual, use_triton_asm=use_triton_asm, interpret=interpret)
    res = fast_two_sum(yn, corr, use_triton_asm=use_triton_asm, interpret=interpret)
    return ff_where(pos, res, FltFlt(z, z))


def ff_norm3d(
    dx: FltFltLike,
    dy: FltFltLike,
    dz: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Single-normalization 3D Euclidean norm ``sqrt(dx^2 + dy^2 + dz^2)`` (``fltflt.hpp:727-756``)."""
    px = two_prod_fma(dx[0], dx[0], use_triton_asm=use_triton_asm, interpret=interpret)
    py = two_prod_fma(dy[0], dy[0], use_triton_asm=use_triton_asm, interpret=interpret)
    pz = two_prod_fma(dz[0], dz[0], use_triton_asm=use_triton_asm, interpret=interpret)
    s = two_sum(px.hi, py.hi, use_triton_asm=use_triton_asm, interpret=interpret)
    t = two_sum(s.hi, pz.hi, use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fadd_rn(t.lo, s.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fadd_rn(lo, px.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fadd_rn(lo, py.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fadd_rn(lo, pz.lo, use_triton_asm=use_triton_asm, interpret=interpret)
    dx2 = fadd_rn(dx[0], dx[0], use_triton_asm=use_triton_asm, interpret=interpret)
    dy2 = fadd_rn(dy[0], dy[0], use_triton_asm=use_triton_asm, interpret=interpret)
    dz2 = fadd_rn(dz[0], dz[0], use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fmaf_rn(dx2, dx[1], lo, use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fmaf_rn(dy2, dy[1], lo, use_triton_asm=use_triton_asm, interpret=interpret)
    lo = fmaf_rn(dz2, dz[1], lo, use_triton_asm=use_triton_asm, interpret=interpret)
    sum_sq = fast_two_sum(t.hi, lo, use_triton_asm=use_triton_asm, interpret=interpret)
    return ff_sqrt_fast(sum_sq, use_triton_asm=use_triton_asm, interpret=interpret)


def ff_floor(
    a: FltFltLike,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Branchless ``floor(a)`` preserving boundary crossings when ``a.hi`` is exact integer (``fltflt.hpp:675-688``)."""
    a_hi = jnp.asarray(a[0], dtype=jnp.float32)
    a_lo = jnp.asarray(a[1], dtype=jnp.float32)
    z = jnp.zeros_like(a_hi)
    neg_one = jnp.asarray(-1.0, dtype=jnp.float32)
    thresh = jnp.asarray(8388608.0, dtype=jnp.float32)
    small = jnp.abs(a_hi) < thresh
    hi_fl = jnp.floor(a_hi)
    nudge = (hi_fl == a_hi) & (a_lo < z)
    small_hi = jnp.where(
        nudge,
        fadd_rn(a_hi, neg_one, use_triton_asm=use_triton_asm, interpret=interpret),
        hi_fl,
    )
    large_pair = fast_two_sum(
        a_hi, jnp.floor(a_lo), use_triton_asm=use_triton_asm, interpret=interpret
    )
    return ff_where(small, FltFlt(small_hi, z), large_pair)


def ff_mod_f32(
    a: FltFltLike,
    box_len: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Periodic wrap ``a - floor(a / L) * L`` into ``[0, L)`` preserving ``(hi, lo)`` precision."""
    L = jnp.asarray(box_len, dtype=jnp.float32)
    z = jnp.zeros_like(L)
    one = jnp.ones_like(L)
    safe_L = jnp.where(L > z, L, one)
    q = jnp.floor(a[0] / safe_L)
    neg_q = fneg_f32(q, use_triton_asm=use_triton_asm, interpret=interpret)
    rem = ff_fma(
        ff_from_f32(neg_q),
        ff_from_f32(safe_L),
        a,
        use_triton_asm=use_triton_asm,
        interpret=interpret,
    )
    # Handle rare boundary rounding where rem < 0 or rem >= L
    below = (rem.hi < z) | ((rem.hi == z) & (rem.lo < z))
    rem = ff_where(
        below,
        ff_add_f32(rem, safe_L, use_triton_asm=use_triton_asm, interpret=interpret),
        rem,
    )
    above = (rem.hi > safe_L) | ((rem.hi == safe_L) & (rem.lo >= z))
    rem = ff_where(
        above,
        ff_sub_f32(rem, safe_L, use_triton_asm=use_triton_asm, interpret=interpret),
        rem,
    )
    return ff_where(L > z, rem, FltFlt(a[0], a[1]))


def ff_min_image_f32(
    d: FltFltLike,
    box_len: jax.Array | float,
    *,
    use_triton_asm: bool | None = None,
    interpret: bool = False,
) -> FltFlt:
    """Minimum-image displacement ``d - round(d / L) * L`` preserving ``(hi, lo)`` precision."""
    L = jnp.asarray(box_len, dtype=jnp.float32)
    z = jnp.zeros_like(L)
    one = jnp.ones_like(L)
    safe_L = jnp.where(L > z, L, one)
    n_img = jnp.rint(d[0] / safe_L)
    neg_n = fneg_f32(n_img, use_triton_asm=use_triton_asm, interpret=interpret)
    wrapped = ff_fma(
        ff_from_f32(neg_n),
        ff_from_f32(safe_L),
        d,
        use_triton_asm=use_triton_asm,
        interpret=interpret,
    )
    return ff_where(L > z, wrapped, FltFlt(d[0], d[1]))


__all__ = [
    "FltFlt",
    "FltFltLike",
    "fadd_rn",
    "fast_two_sum",
    "ff_add",
    "ff_add_f32",
    "ff_div",
    "ff_floor",
    "ff_fma",
    "ff_from_f32",
    "ff_from_float64",
    "ff_min_image_f32",
    "ff_mod_f32",
    "ff_mul",
    "ff_mul_f32",
    "ff_neg",
    "ff_norm3d",
    "ff_sqrt",
    "ff_sqrt_fast",
    "ff_sub",
    "ff_sub_f32",
    "ff_to_f32",
    "ff_to_float64",
    "ff_where",
    "fltflt_triton_asm_context",
    "fmaf_rn",
    "fmul_rn",
    "fneg_f32",
    "fsub_rn",
    "should_use_triton_asm",
    "two_prod_fma",
    "two_sum",
]
