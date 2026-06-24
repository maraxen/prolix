"""Claim 2 W1 (#275): single-system EnsemblePlan.run lowers to valid StableHLO.

Gate: jax.jit(trajectory_fn).lower() succeeds for B=1 bonded EnsemblePlan path
and HLO contains no custom_call ops (blocks portable export).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prolix.api.export_run import make_single_trajectory_fn

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle


def _hlo_text(lowered) -> str:
    try:
        return lowered.compiler_ir(dialect="hlo").as_hlo_text()
    except Exception:
        return lowered.as_text()


def _md_scalars(dt: float, kT: float, *, dtype=jnp.float32):
    return jnp.asarray(dt, dtype=dtype), jnp.asarray(kT, dtype=dtype)


@pytest.mark.fast
def test_w1_ensemble_plan_single_system_lower_succeeds():
    """B=1 EnsemblePlan trajectory lowers without trace errors."""
    bundle = _make_bundle(10, seed=275)
    n_steps = 4
    dt, kT = _md_scalars(0.5, 0.596)
    seed = jnp.array(275, dtype=jnp.uint32)

    trajectory_fn = make_single_trajectory_fn(bundle, n_steps=n_steps)
    lowered = jax.jit(trajectory_fn).lower(seed, dt, kT)
    assert lowered is not None
    compiled = lowered.compile()
    positions = compiled(seed, dt, kT)
    assert positions.shape == (n_steps, 10, 3)
    assert jnp.all(jnp.isfinite(positions))


@pytest.mark.fast
def test_w1_ensemble_plan_stablehlo_no_custom_call():
    """StableHLO/HLO must not contain custom_call (export blocker)."""
    bundle = _make_bundle(10, seed=275)
    dt, kT = _md_scalars(0.5, 0.596)
    seed = jnp.array(42, dtype=jnp.uint32)
    trajectory_fn = make_single_trajectory_fn(bundle, n_steps=4)
    lowered = jax.jit(trajectory_fn).lower(seed, dt, kT)
    hlo = _hlo_text(lowered)
    assert "custom_call" not in hlo, (
        "HLO contains custom_call — check settle integrator for debug side effects"
    )
    mlir = lowered.as_text()
    assert "module" in mlir


@pytest.mark.fast
def test_w1_dt_kT_runtime_without_recompile():
    """Same compiled fn accepts different dt/kT scalar arrays at runtime."""
    bundle = _make_bundle(10, seed=275)
    seed = jnp.array(7, dtype=jnp.uint32)
    fn = jax.jit(make_single_trajectory_fn(bundle, n_steps=3))
    dt_a, kT_a = _md_scalars(0.5, 0.596)
    dt_b, kT_b = _md_scalars(0.25, 0.596)
    pos_a = fn(seed, dt_a, kT_a)
    pos_b = fn(seed, dt_b, kT_b)
    assert pos_a.shape == pos_b.shape == (3, 10, 3)
    assert float(jnp.max(jnp.abs(pos_a - pos_b))) > 0.0
