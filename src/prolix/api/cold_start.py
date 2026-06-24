"""Claim 2 WB2 (#280): browser cold-start timing for export path.

Measures ``.wasm`` artifact load (disk read proxy for fetch) plus JAX compile
and first-run warm-up for the W4 export trajectory. ``iree-compile`` is build-time
and is excluded from the runtime cold-start budget.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp

from prolix.api.export_run import make_single_trajectory_fn

# Roadmap Claim 2 WB2 target (browser load + first MD warm-up).
WASM_COLD_START_MAX_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class ColdStartTimings:
    """Runtime cold-start breakdown in seconds (build steps excluded)."""

    jax_compile_seconds: float
    jax_warmup_seconds: float
    wasm_load_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.jax_compile_seconds
            + self.jax_warmup_seconds
            + self.wasm_load_seconds
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "jax_compile_seconds": self.jax_compile_seconds,
            "jax_warmup_seconds": self.jax_warmup_seconds,
            "wasm_load_seconds": self.wasm_load_seconds,
            "total_seconds": self.total_seconds,
        }


def measure_browser_cold_start(
    bundle: Any,
    *,
    n_steps: int = 100,
    dt: float = 0.5,
    kT: float = 0.596,
    seed: int = 280,
    wasm_path: str | pathlib.Path | None = None,
) -> ColdStartTimings:
    """Time wasm load + JAX compile/first run for the export trajectory fn."""
    seed_arr = jnp.array(seed, dtype=jnp.uint32)
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    kT_arr = jnp.asarray(kT, dtype=jnp.float32)

    trajectory_fn = make_single_trajectory_fn(bundle, n_steps=n_steps)

    compile_start = time.perf_counter()
    compiled = jax.jit(trajectory_fn).lower(seed_arr, dt_arr, kT_arr).compile()
    jax_compile_seconds = time.perf_counter() - compile_start

    warmup_start = time.perf_counter()
    result = compiled(seed_arr, dt_arr, kT_arr)
    jax.block_until_ready(result)
    jax_warmup_seconds = time.perf_counter() - warmup_start

    wasm_load_seconds = 0.0
    if wasm_path is not None:
        path = pathlib.Path(wasm_path)
        load_start = time.perf_counter()
        _ = path.read_bytes()
        wasm_load_seconds = time.perf_counter() - load_start

    return ColdStartTimings(
        jax_compile_seconds=jax_compile_seconds,
        jax_warmup_seconds=jax_warmup_seconds,
        wasm_load_seconds=wasm_load_seconds,
    )


def assert_cold_start_under_limit(
    timings: ColdStartTimings,
    *,
    max_seconds: float = WASM_COLD_START_MAX_SECONDS,
) -> ColdStartTimings:
    """Return timings or raise ``AssertionError`` if over the WB2 cap."""
    total = timings.total_seconds
    if total > max_seconds:
        detail = ", ".join(f"{k}={v:.3f}s" for k, v in timings.as_dict().items())
        raise AssertionError(
            f"WB2 cold-start {total:.3f}s exceeds {max_seconds:.1f}s cap ({detail})"
        )
    return timings
