"""V5: Observable parity single vs batched EnsemblePlan runs (#267)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp

from prolix.api import EnsemblePlan
from prolix.api.bundle_md import bonded_energy_fn_from_bundle
from prolix.api.observables import Energy, KineticEnergy

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle


def _observables_for(bundle):
    energy_fn = bonded_energy_fn_from_bundle(bundle)

    def _eval(positions, b):
        del b
        return energy_fn(positions)

    return {
        "ke": KineticEnergy(),
        "energy": Energy(energy_fn=_eval, bundle=bundle),
    }


def _independent_observable_runs(bundles, *, n_steps, dt, kT, seed):
    out = []
    for i, b in enumerate(bundles):
        traj = EnsemblePlan.from_bundle(b).run(
            n_steps=n_steps,
            dt=dt,
            kT=kT,
            seed=seed + i,
            observables=_observables_for(b),
        )
        out.append(traj)
    return out


def test_v5_observable_parity_b4_homo_vs_independent():
    """Final-step observables match between multi-bundle and independent runs."""
    n_atoms = 10
    b = _make_bundle(n_atoms, seed=3)
    bundles = [b, b, b, b]
    n_steps = 6
    dt = 0.5
    kT = 0.596
    seed = 99
    observables = _observables_for(b)

    batched = EnsemblePlan.from_bundles(bundles).run(
        n_steps=n_steps,
        dt=dt,
        kT=kT,
        seed=seed,
        observables=observables,
    )
    refs = _independent_observable_runs(
        bundles, n_steps=n_steps, dt=dt, kT=kT, seed=seed
    )

    assert isinstance(batched, list)
    for i, (got, ref) in enumerate(zip(batched, refs, strict=True)):
        for key in ("ke", "energy"):
            assert key in got.observable_values
            assert key in ref.observable_values
            diff = jnp.abs(got.observable_values[key] - ref.observable_values[key])
            # float32 noise floor -- see test_v3_homogeneous_batch_parity.py's
            # comment (debt 841 fixed; residual is ordinary float32 rounding).
            assert diff < 1e-4, f"system {i} {key}: delta={diff:.3e}"
