"""Export-shaped EnsemblePlan runners for StableHLO / jax.jit lowering.

``dt`` and ``kT`` are JAX scalar **arguments** to the returned callables so one
compiled program can sweep thermodynamic parameters without recompilation.
``n_steps`` remains host-static (``lax.scan`` length). Bundles are fixed in the
closure.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from prolix.api.ensemble_plan import EnsemblePlan


def make_single_trajectory_fn(
    bundle: Any,
    *,
    n_steps: int,
):
    """B=1 trajectory: ``(seed, dt, kT) -> (n_steps, n_atoms, 3)`` positions."""

    def trajectory(
        seed: jnp.ndarray,
        dt: jnp.ndarray,
        kT: jnp.ndarray,
    ) -> jnp.ndarray:
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
):
    """B>1 hetero export: ``(seed, dt, kT) -> tuple of position trajectories."""

    singles = [make_single_trajectory_fn(b, n_steps=n_steps) for b in bundles]
    n_systems = len(singles)

    def run_all(
        seed_base: jnp.ndarray,
        dt: jnp.ndarray,
        kT: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        return tuple(
            singles[i](seed_base + jnp.uint32(i), dt, kT)
            for i in range(n_systems)
        )

    return run_all
