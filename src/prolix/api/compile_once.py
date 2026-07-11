"""JIT compile-once helpers for Claim 1 V4-HLO (#266)."""

from __future__ import annotations

from typing import Any, Callable

import jax


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


def count_jaxpr_scans(fn: Callable[..., Any], *args: Any) -> int:
    """Count ``scan`` primitives in the jaxpr of ``fn`` at abstract args."""
    jaxpr = jax.make_jaxpr(fn)(*args)
    return str(jaxpr).count("scan")
