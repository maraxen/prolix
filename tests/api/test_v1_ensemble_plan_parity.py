"""V1: EnsemblePlan(B=1) parity vs settle_langevin.

Contract (roadmap): solvated AKE, 1k steps, RMSD < 1e-12 Å vs direct
settle_langevin reference.

Current status: EnsemblePlan.run() uses stub zero energy and simplified
mass/shift wiring (ensemble_plan.py). Smoke harness lands here; strict
solvated-AKE parity remains skipped until bundle-backed FF integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.api.ensemble_plan import EnsemblePlan
from prolix.physics import settle
from jax_md import space


def _minimal_water_bundle(n_waters: int = 1):
    """Tiny TIP3P-like bundle for fast parity harness."""
    from unittest.mock import MagicMock

    n_atoms = 3 * n_waters
    positions = jnp.array(
        [[i * 3.0, 0.0, 0.0] for i in range(n_atoms)], dtype=jnp.float64
    )
    water_indices = jnp.array(
        [[3 * w, 3 * w + 1, 3 * w + 2] for w in range(n_waters)], dtype=jnp.int32
    )
    bundle = MagicMock()
    bundle.n_atoms = n_atoms
    bundle.n_waters = n_waters
    bundle.positions = positions
    bundle.water_indices = water_indices
    return bundle, positions


def _reference_settle_langevin(
    positions_init: jnp.ndarray,
    water_indices: jnp.ndarray,
    *,
    n_steps: int,
    dt: float,
    kT: float,
    seed: int,
) -> jnp.ndarray:
    """Direct settle_langevin trajectory (reference path)."""
    force_fn = lambda r, **kw: jnp.zeros_like(r)
    _, shift_fn = space.free()
    mass = jnp.ones(positions_init.shape[0], dtype=jnp.float64)
    init_fn, apply_fn = settle.settle_langevin(
        force_fn,
        shift_fn,
        dt=dt,
        kT=kT,
        gamma=10.0,
        mass=mass,
        water_indices=water_indices,
        project_ou_momentum_rigid=True,
    )
    key = jax.random.PRNGKey(seed)
    state = init_fn(key, positions_init)
    traj = []
    for _ in range(n_steps):
        state = apply_fn(state, kT=kT, dt=dt)
        traj.append(state.position)
    return jnp.stack(traj)


def test_v1_harness_runs_and_returns_trajectory():
    """V1 smoke: EnsemblePlan.from_bundle().run() completes with finite positions."""
    bundle, _ = _minimal_water_bundle(n_waters=1)
    ep = EnsemblePlan.from_bundle(bundle)
    traj = ep.run(n_steps=5, dt=0.5, kT=0.596, seed=42)
    assert traj.n_steps == 5
    assert traj.positions.shape == (5, 3, 3)
    assert jnp.all(jnp.isfinite(traj.positions))


@pytest.mark.skip(
    reason="V1 strict parity (solvated AKE, 1k steps, RMSD<1e-12) blocked on "
    "bundle FF wiring in EnsemblePlan.run()",
)
def test_v1_solvated_ake_parity_vs_settle_langevin():
    """Preregistered V1 gate — enable when run() uses real bundle energy."""
    jax.config.update("jax_enable_x64", True)
    n_steps = 1000
    dt = 1.0
    kT = 0.596
    seed = 0

    bundle, pos_init = _minimal_water_bundle(n_waters=1)
    ref = _reference_settle_langevin(
        pos_init,
        bundle.water_indices,
        n_steps=n_steps,
        dt=dt,
        kT=kT,
        seed=seed,
    )
    ep = EnsemblePlan.from_bundle(bundle)
    traj = ep.run(n_steps=n_steps, dt=dt, kT=kT, seed=seed)
    rmsd = jnp.sqrt(jnp.mean((traj.positions - ref) ** 2))
    assert rmsd < 1e-12, f"V1 parity failed: RMSD={rmsd:.3e} Å"
