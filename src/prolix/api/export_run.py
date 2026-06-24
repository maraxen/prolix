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


def make_smoke_diagnostics_fn(
    bundle: Any,
    *,
    n_steps: int,
):
    """B=1 smoke export: ``(seed, dt, kT) -> (positions, temperatures, energies)``.

    Per-step kinetic temperature and kinetic energy for browser trace rendering
    (Claim 2 W4). Bonded potential is omitted here because ``bonded_energy_fn``
    is not yet JIT-safe on the one-water fixture; KE is the thermostat-coupled
    quantity. ``dof`` is fixed at factory time from ``n_atoms``.
    """

    import jax
    import jax.numpy as jnp

    from prolix.api.bundle_md import (
        active_positions,
        as_integration_scalars,
        bonded_energy_fn_from_bundle,
        displacement_fn_for_bundle,
        masses_for_bundle,
        trim_trajectory_positions,
        water_indices_for_integration,
    )
    from prolix.physics.settle import settle_langevin
    from prolix.simulate import BOLTZMANN_KCAL

    n_atoms = int(jnp.asarray(bundle.n_atoms))
    dof = max(1, 3 * n_atoms - 6)
    energy_fn = bonded_energy_fn_from_bundle(bundle)
    _, shift_fn = displacement_fn_for_bundle(bundle)
    masses = masses_for_bundle(bundle)
    water_indices = water_indices_for_integration(bundle)

    def diagnostics(
        seed: jnp.ndarray,
        dt: jnp.ndarray,
        kT: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dt, kT = as_integration_scalars(dt, kT, dtype=bundle.positions.dtype)
        init_fn, apply_fn = settle_langevin(
            energy_fn,
            shift_fn,
            dt=1.0,
            kT=1.0,
            gamma=10.0,
            mass=masses,
            water_indices=water_indices,
            project_ou_momentum_rigid=True,
        )
        key = jax.random.PRNGKey(jnp.asarray(seed, dtype=jnp.uint32))
        state = init_fn(key, active_positions(bundle), kT=kT)

        def step_fn(carry: Any, _: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            new_state = apply_fn(carry, kT=kT, dt=dt)
            mom = new_state.momentum
            mass = new_state.mass
            mass_x = mass[:, None] if mass.ndim == 1 else mass
            ke = jnp.sum(mom**2 / (2.0 * mass_x))
            temp = (2.0 * ke) / (dof * BOLTZMANN_KCAL)
            return new_state, (new_state.position, temp, ke)

        _, (positions, temperatures, energies) = jax.lax.scan(
            step_fn, state, None, length=n_steps
        )
        positions = trim_trajectory_positions(positions, bundle)
        return positions, temperatures, energies

    return diagnostics
