"""Claim 2 W3 (#277): IREE-WASM compile gate — artifact < 50 MB.

Compiles EnsemblePlan export StableHLO (W1 path) to wasm32 via iree-compile.
Requires optional ``wasm`` extra: ``uv sync --extra wasm``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prolix import export
from prolix.api.export_run import make_single_trajectory_fn

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle


def _require_iree_compile() -> str:
    exe = export.find_iree_compile()
    if exe is None:
        pytest.skip(
            "iree-compile not found; install with: uv sync --extra wasm"
        )
    return exe


@pytest.mark.wasm
@pytest.mark.fast
def test_w3_ensemble_plan_iree_wasm_under_50mb(tmp_path):
    """StableHLO from B=1 export lowers to wasm32 under the 50 MB Claim 2 cap."""
    _require_iree_compile()

    bundle = _make_bundle(10, seed=277)
    fn = make_single_trajectory_fn(bundle, n_steps=2)
    seed = jnp.array(1, dtype=jnp.uint32)
    dt = jnp.asarray(0.5, dtype=jnp.float32)
    kT = jnp.asarray(0.596, dtype=jnp.float32)

    lowered = jax.jit(fn).lower(seed, dt, kT)
    wasm_path = tmp_path / "ensemble_traj.wasm"
    export.compile_lowered_to_wasm(lowered, wasm_path)

    assert wasm_path.is_file()
    size = export.assert_wasm_artifact_under_limit(wasm_path)
    assert size > 0
