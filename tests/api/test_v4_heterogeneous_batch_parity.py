"""V4: Heterogeneous batch parity vs N independent runs (#265).

Headline Claim-1 gate: multi-bundle EnsemblePlan.run() must match
independent from_bundle runs per system when sizes differ within one batch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp

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


def test_v4_hetero_b4_parity_vs_independent_runs():
    """B=4 varied-size bundles: multi-bundle run matches independent runs."""
    sizes = (5, 10, 20, 35)
    bundles = [_make_bundle(n, seed=100 + i) for i, n in enumerate(sizes)]
    n_steps = 8
    dt = 0.5
    kT = 0.596
    seed = 42

    batched = EnsemblePlan.from_bundles(bundles).run(
        n_steps=n_steps, dt=dt, kT=kT, seed=seed
    )
    assert isinstance(batched, list)
    assert len(batched) == len(sizes)

    refs = _independent_runs(bundles, n_steps=n_steps, dt=dt, kT=kT, seed=seed)

    for i, (got, ref, n_atoms) in enumerate(
        zip(batched, refs, sizes, strict=True)
    ):
        assert got.n_steps == ref.n_steps == n_steps
        assert got.positions.shape == (n_steps, n_atoms, 3)
        assert ref.positions.shape == (n_steps, n_atoms, 3)
        rmsd = jnp.sqrt(jnp.mean((got.positions - ref.positions) ** 2))
        # float32 noise floor -- see test_v3_homogeneous_batch_parity.py's
        # comment (debt 841 fixed; residual is ordinary float32 rounding).
        assert rmsd < 1e-5, f"system {i} (n={n_atoms}): RMSD={rmsd:.3e} Å"
