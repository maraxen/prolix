"""V4-HLO (#266): compile-once assertion for hetero ensemble export/dispatch.

Primary: exactly one JIT compile per export fn (cache hit on second call).
Secondary (homo B=4 stacked path): jaxpr contains a single ``scan`` (batched).
Hetero varied ``n_atoms`` uses unrolled singles: 1 compile, B scans (until
dynamic trim supports stacked hetero vmap).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prolix.api.compile_once import count_jaxpr_scans, count_jit_compiles
from prolix.api.export_run import make_hetero_trajectory_fn

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle


def _md_scalars(dt: float = 0.5, kT: float = 0.596):
    return jnp.asarray(dt, dtype=jnp.float32), jnp.asarray(kT, dtype=jnp.float32)


def _hetero_b4_bundles():
    sizes = (5, 10, 20, 35)
    return [_make_bundle(n, seed=100 + i) for i, n in enumerate(sizes)], sizes


@pytest.mark.fast
def test_v4_hlo_homo_b4_stacked_single_compile_and_single_scan():
    """Homo B=4 (same n_atoms): one JIT compile and one batched scan in jaxpr."""
    bundles = [_make_bundle(10, seed=300 + i) for i in range(4)]
    n_steps = 4
    dt, kT = _md_scalars()
    seed_base = jnp.array(266, dtype=jnp.uint32)

    run_all = make_hetero_trajectory_fn(bundles, n_steps=n_steps)

    compiles = count_jit_compiles(run_all, seed_base, dt, kT)
    assert compiles == 1, f"V4-HLO homo: expected 1 compile, got {compiles}"

    scans = count_jaxpr_scans(run_all, seed_base, dt, kT)
    assert scans == 1, (
        f"V4-HLO homo: expected 1 scan (stacked vmap), got {scans}"
    )

    compiled = jax.jit(run_all).lower(seed_base, dt, kT).compile()
    exported = compiled(seed_base, dt, kT)
    assert len(exported) == 4
    for pos in exported:
        assert pos.shape == (n_steps, 10, 3)
        assert jnp.all(jnp.isfinite(pos))


@pytest.mark.fast
def test_v4_hlo_hetero_b4_export_single_compile():
    """Hetero B=4 varied n_atoms: one JIT compile (unrolled singles path)."""
    bundles, _ = _hetero_b4_bundles()
    n_steps = 4
    dt, kT = _md_scalars()
    seed_base = jnp.array(266, dtype=jnp.uint32)

    run_all = make_hetero_trajectory_fn(bundles, n_steps=n_steps)
    compiles = count_jit_compiles(run_all, seed_base, dt, kT)
    assert compiles == 1, f"V4-HLO hetero: expected 1 compile, got {compiles}"

    scans = count_jaxpr_scans(run_all, seed_base, dt, kT)
    assert scans == len(bundles), (
        f"V4-HLO hetero unrolled: expected {len(bundles)} scans, got {scans}"
    )
