"""Claim 2 W2 (#276): hetero EnsemblePlan(B=4) lowers to valid StableHLO."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prolix.api import EnsemblePlan
from prolix.api.export_run import make_hetero_trajectory_fn, make_single_trajectory_fn

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


def _hetero_b4_bundles():
    sizes = (5, 10, 20, 35)
    return [_make_bundle(n, seed=100 + i) for i, n in enumerate(sizes)], sizes


@pytest.mark.fast
def test_w2_hetero_b4_lower_succeeds():
    """B=4 mixed n_atoms: export fn lowers and matches EnsemblePlan.run parity."""
    bundles, sizes = _hetero_b4_bundles()
    n_steps = 4
    dt, kT = _md_scalars(0.5, 0.596)
    seed_base = jnp.array(276, dtype=jnp.uint32)

    run_all = make_hetero_trajectory_fn(bundles, n_steps=n_steps)
    lowered = jax.jit(run_all).lower(seed_base, dt, kT)
    compiled = lowered.compile()
    exported = compiled(seed_base, dt, kT)

    assert len(exported) == len(sizes)
    for i, (pos, n_atoms) in enumerate(zip(exported, sizes, strict=True)):
        assert pos.shape == (n_steps, n_atoms, 3)
        assert jnp.all(jnp.isfinite(pos))

    refs = EnsemblePlan.from_bundles(bundles).run(
        n_steps=n_steps, dt=dt, kT=kT, seed=int(seed_base)
    )
    assert isinstance(refs, list)
    for i, (got, ref) in enumerate(zip(exported, refs, strict=True)):
        rmsd = jnp.sqrt(jnp.mean((got - ref.positions) ** 2))
        # float32 + XR-VACUUM-DT jnp conversion; was 1e-10 pre-fs-default.
        assert rmsd < 1e-6, f"system {i}: RMSD={rmsd:.3e}"


@pytest.mark.fast
def test_w2_hetero_b4_stablehlo_no_custom_call():
    """Hetero B=4 export HLO must not contain custom_call ops."""
    bundles, _ = _hetero_b4_bundles()
    dt, kT = _md_scalars(0.5, 0.596)
    seed_base = jnp.array(42, dtype=jnp.uint32)
    run_all = make_hetero_trajectory_fn(bundles, n_steps=4)
    lowered = jax.jit(run_all).lower(seed_base, dt, kT)
    hlo = _hlo_text(lowered)
    assert "custom_call" not in hlo
    assert "module" in lowered.as_text()


@pytest.mark.fast
def test_w1_helper_reexport_still_works():
    bundle = _make_bundle(10, seed=275)
    dt, kT = _md_scalars(0.5, 0.596)
    seed = jnp.array(1, dtype=jnp.uint32)
    pos = jax.jit(make_single_trajectory_fn(bundle, n_steps=2))(seed, dt, kT)
    assert pos.shape == (2, 10, 3)
