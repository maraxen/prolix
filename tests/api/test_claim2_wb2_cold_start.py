"""Claim 2 WB2 (#280): cold-start gate — wasm load + warm-up < 5 s.

Runtime budget covers JAX compile/first run plus reading the bundled ``.wasm``
artifact (fetch proxy). ``iree-compile`` is build-time and excluded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prolix.api.cold_start import (
    WASM_COLD_START_MAX_SECONDS,
    assert_cold_start_under_limit,
    measure_browser_cold_start,
)
from prolix.api.export_run import make_single_trajectory_fn
from prolix import export

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle

WB2_N_STEPS = 100


@pytest.mark.fast
def test_wb2_jax_cold_start_under_5_seconds():
    """100-step export compile + first run stays under the WB2 cap (no wasm load)."""
    bundle = _make_bundle(10, seed=280)
    timings = measure_browser_cold_start(
        bundle,
        n_steps=WB2_N_STEPS,
        seed=280,
    )
    assert_cold_start_under_limit(timings, max_seconds=WASM_COLD_START_MAX_SECONDS)
    assert timings.jax_compile_seconds < WASM_COLD_START_MAX_SECONDS


@pytest.mark.wasm
@pytest.mark.fast
def test_wb2_wasm_load_plus_warmup_under_5_seconds(tmp_path):
    """Bundled wasm read + JAX warm-up together stay under WB2."""
    if export.find_iree_compile() is None:
        pytest.skip("iree-compile not found; install with: uv sync --extra wasm")

    bundle = _make_bundle(10, seed=280)
    seed = jnp.array(280, dtype=jnp.uint32)
    dt = jnp.asarray(0.5, dtype=jnp.float32)
    kT = jnp.asarray(0.596, dtype=jnp.float32)
    fn = make_single_trajectory_fn(bundle, n_steps=WB2_N_STEPS)
    wasm_path = tmp_path / "trajectory.wasm"
    export.compile_lowered_to_wasm(
        jax.jit(fn).lower(seed, dt, kT),
        wasm_path,
    )

    timings = measure_browser_cold_start(
        bundle,
        n_steps=WB2_N_STEPS,
        seed=280,
        wasm_path=wasm_path,
    )
    assert_cold_start_under_limit(timings, max_seconds=WASM_COLD_START_MAX_SECONDS)
    assert timings.wasm_load_seconds < 1.0
