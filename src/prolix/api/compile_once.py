"""JIT compile-once helpers for Claim 1 V4-HLO (#266)."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.extend.core as jax_core


def count_jit_compiles(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> int:
    """Return how many times ``fn`` was traced when wrapped in ``jax.jit``.

    Python-side counter increments only on cache miss (first compile).
    """
    trace_count = [0]

    def wrapped(*inner_args: Any, **inner_kwargs: Any) -> Any:
        trace_count[0] += 1
        return fn(*inner_args, **inner_kwargs)

    jitted = jax.jit(wrapped)
    jitted(*args, **kwargs)
    jitted(*args, **kwargs)
    return trace_count[0]


def _count_scan_eqns(jaxpr: jax_core.Jaxpr | jax_core.ClosedJaxpr) -> int:
    """Recursively count ``scan`` equations, descending into nested jaxprs.

    A substring count over ``str(jaxpr)`` also matches unrelated tokens (e.g.
    variable names containing "scan", or the primitive name appearing in a
    docstring-like repr fragment) and cannot distinguish one scan equation
    from a nested jaxpr param that merely mentions scan-adjacent names. Eqn
    params can hold a single ``Jaxpr``/``ClosedJaxpr`` (``scan``, ``pjit``) or
    a tuple of them (``cond`` branches), so both shapes are walked.
    """
    if isinstance(jaxpr, jax_core.ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    count = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "scan":
            count += 1
        for param_value in eqn.params.values():
            if isinstance(param_value, (jax_core.Jaxpr, jax_core.ClosedJaxpr)):
                count += _count_scan_eqns(param_value)
            elif isinstance(param_value, (list, tuple)):
                for item in param_value:
                    if isinstance(item, (jax_core.Jaxpr, jax_core.ClosedJaxpr)):
                        count += _count_scan_eqns(item)
    return count


def count_jaxpr_scans(fn: Callable[..., Any], *args: Any) -> int:
    """Count ``scan`` equations in the jaxpr of ``fn``, including nested ones."""
    jaxpr = jax.make_jaxpr(fn)(*args)
    return _count_scan_eqns(jaxpr)
