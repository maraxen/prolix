"""Export-shaped EnsemblePlan runners for StableHLO / jax.jit lowering.

These helpers fix ``MolecularBundle``(s) and integration hyperparameters in a
closure so ``jax.jit(fn).lower(...)`` traces a pure trajectory computation.
Hetero batches unroll per-system scans (different ``n_atoms``); homo batches
can use the stacked vmap path separately (#2645).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from prolix.api.ensemble_plan import EnsemblePlan


def make_single_trajectory_fn(
    bundle: Any,
    *,
    n_steps: int,
    dt: float,
    kT: float,
):
    """B=1 trajectory: returns ``(n_steps, n_atoms, 3)`` positions."""

    def trajectory(seed: jnp.ndarray) -> jnp.ndarray:
        traj = EnsemblePlan.from_bundle(bundle).run(
            n_steps=n_steps,
            dt=dt,
            kT=kT,
            seed=seed,
        )
        return traj.positions

    return trajectory


def make_hetero_trajectory_fn(
    bundles: list[Any],
    *,
    n_steps: int,
    dt: float,
    kT: float,
):
    """B>1 hetero export: fixed bundle list in closure, tuple of trajectories out.

    Systems are unrolled at trace time (constant ``len(bundles)``). Each element
    has shape ``(n_steps, n_atoms_i, 3)`` — ragged lengths are not stacked.
    """

    singles = [
        make_single_trajectory_fn(b, n_steps=n_steps, dt=dt, kT=kT) for b in bundles
    ]
    n_systems = len(singles)

    def run_all(seed_base: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        return tuple(
            singles[i](seed_base + jnp.uint32(i)) for i in range(n_systems)
        )

    return run_all
