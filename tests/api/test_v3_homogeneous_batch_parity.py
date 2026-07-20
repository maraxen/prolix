"""V3: Homogeneous (same-size) batch parity vs B independent runs (#264).

After #1842 multi-bundle dispatch, batched and independent paths must agree
when bundles share shape and per-system seeds match.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest


from prolix.api import EnsemblePlan

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle


def _independent_runs(bundles, *, n_steps, dt, kT, seed):
    return [
        EnsemblePlan.from_bundle(b).run(
            n_steps=n_steps, dt=dt, kT=kT, seed=seed + i
        )
        for i, b in enumerate(bundles)
    ]


def test_v3_homo_b4_parity_vs_independent_runs():
    """B=4 same-size bundles: multi-bundle run matches independent runs."""
    n_atoms = 10
    b = _make_bundle(n_atoms, seed=7)
    bundles = [b, b, b, b]
    n_steps = 8
    dt = 0.5
    kT = 0.596
    seed = 42

    batched = EnsemblePlan.from_bundles(bundles).run(
        n_steps=n_steps, dt=dt, kT=kT, seed=seed
    )
    assert isinstance(batched, list)
    assert len(batched) == 4

    refs = _independent_runs(bundles, n_steps=n_steps, dt=dt, kT=kT, seed=seed)

    for i, (got, ref) in enumerate(zip(batched, refs, strict=True)):
        assert got.n_steps == ref.n_steps == n_steps
        assert got.positions.shape == ref.positions.shape
        rmsd = jnp.sqrt(jnp.mean((got.positions - ref.positions) ** 2))
        # float32 noise floor (not 1e-10/1e-12 as for the x64-scoped v1 tests) --
        # residual ~1e-9 Å is ordinary float32 rounding, not a correctness bug.
        # See debt 841 (batched-vs-independent divergence, fixed) and debt
        # 832/835 (separate, still-open x64 dtype-mixing issue -- unrelated).
        assert rmsd < 1e-5, f"system {i}: RMSD={rmsd:.3e} Å"
