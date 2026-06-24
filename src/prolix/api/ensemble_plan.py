"""EnsemblePlan: high-level API for batch molecular dynamics simulations.

Multi-bundle dispatch uses xtrax.tiling via EnsembleMDPlanner (#1842).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prolix.api.observables import Trajectory


class EnsemblePlan:
    """Orchestrates batch MD simulations over multiple MolecularBundle instances.

    Single-bundle runs return a Trajectory. Multi-bundle runs return a list of
    Trajectory objects, one per bundle, dispatched according to batch_plan
    (vmap vs safe_map chunk size on the N_MOLS axis).

    Args:
        bundles: List of MolecularBundle instances to simulate in parallel.
        planner: Optional planner with plan(bundles). When omitted and
                 len(bundles) > 1, EnsembleMDPlanner is used automatically.

    Attributes:
        bundles: The input bundle list.
        batch_plan: Result of planner.plan(bundles), or None for single-bundle
                    runs without an explicit planner.
    """

    def __init__(self, bundles: list[Any], planner: Any = None) -> None:
        self.bundles = bundles
        if planner is None and len(bundles) > 1:
            from prolix.api.ensemble_planner import EnsembleMDPlanner

            planner = EnsembleMDPlanner()
        if planner is not None:
            self.batch_plan = planner.plan(bundles)
        else:
            self.batch_plan = None

    @classmethod
    def from_bundle(cls, bundle: Any, planner: Any = None) -> EnsemblePlan:
        return cls([bundle], planner)

    @classmethod
    def from_bundles(
        cls, bundles: list[Any], planner: Any = None
    ) -> EnsemblePlan:
        return cls(bundles, planner)

    def _systems_chunk_size(self) -> int:
        """Chunk size for N_MOLS dispatch (0 = vmap intent)."""
        if self.batch_plan is None:
            return 1
        for name in ("n_mols", "n_systems"):
            try:
                return self.batch_plan.decision_for(name).batch_size
            except KeyError:
                continue
        return 1

    def run(
        self,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any = 0,
        observables: dict[str, Any] | None = None,
    ) -> Trajectory | list[Trajectory]:
        """Run MD simulation over all bundles.

        Args:
            n_steps: Number of MD steps (host-static; ``lax.scan`` length).
            dt: Timestep in AKMA units (``jnp.ndarray`` scalar or float).
            kT: Thermal energy (``jnp.ndarray`` scalar or float).
            seed: PRNG seed for thermostat noise (int or uint32 array).
            observables: Optional name → Observable map; final-step values
                are stored in Trajectory.observable_values.

        Returns:
            Trajectory for a single bundle, or list[Trajectory] when
            len(bundles) > 1.
        """
        if not self.bundles:
            raise ValueError("EnsemblePlan requires at least one bundle")

        if len(self.bundles) == 1:
            return self._run_single(
                self.bundles[0], n_steps, dt, kT, seed, observables=observables
            )

        from prolix.api.bundle_stack import can_jit_vmap_n_mols

        if can_jit_vmap_n_mols(self.bundles):
            return self._run_stacked_dispatch(
                n_steps, dt, kT, seed, observables=observables
            )

        chunk = self._systems_chunk_size()
        chunk = len(self.bundles) if chunk == 0 else max(1, chunk)
        trajectories: list[Trajectory] = []
        for i in range(0, len(self.bundles), chunk):
            for bundle in self.bundles[i : i + chunk]:
                obs = observables
                if observables is not None and len(self.bundles) > 1:
                    obs = _observables_for_bundle(bundle, observables)
                trajectories.append(
                    self._run_single(
                        bundle,
                        n_steps,
                        dt,
                        kT,
                        seed + len(trajectories),
                        observables=obs,
                    )
                )
        return trajectories

    def _run_stacked_dispatch(
        self,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None = None,
    ) -> list[Trajectory]:
        """JIT vmap / safe_map over stack-compatible bundles (#2645)."""
        import jax.numpy as jnp

        from prolix.api.bundle_stack import (
            stack_molecular_bundles,
            unstack_trajectories,
        )
        from prolix.api.ensemble_dispatch import dispatch_n_mols

        stacked = stack_molecular_bundles(self.bundles)
        n_systems = len(self.bundles)
        seeds = jnp.arange(n_systems, dtype=jnp.int32) + seed
        # Host-static: identical for every bundle in a jit-vmap batch (see can_jit_vmap_n_mols).
        integration_prefix = int(jnp.asarray(self.bundles[0].n_atoms))

        def run_one(bundle, seed_i: jnp.ndarray) -> Trajectory:
            obs = observables
            if observables is not None:
                obs = _observables_for_bundle(bundle, observables)
            return self._run_single(
                bundle,
                n_steps,
                dt,
                kT,
                seed_i,
                observables=obs,
                integration_prefix=integration_prefix,
                trim_output=False,
            )

        batched = dispatch_n_mols(
            self.batch_plan,
            n_systems,
            run_one,
            stacked,
            seeds,
        )
        return unstack_trajectories(batched, self.bundles)

    def _run_single(
        self,
        bundle: Any,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None = None,
        *,
        integration_prefix: int | None = None,
        trim_output: bool = True,
    ) -> Trajectory:
        import jax
        import jax.numpy as jnp

        from prolix.api.bundle_md import (
            active_positions,
            as_integration_scalars,
            energy_fn_from_bundle,
            displacement_fn_for_bundle,
            masses_for_bundle,
            positions_with_prefix,
            trim_trajectory_positions,
            unit_masses,
            water_indices_for_integration,
        )
        from prolix.api.observables import Trajectory
        from prolix.physics.settle import settle_langevin

        energy_fn = energy_fn_from_bundle(bundle)
        _displacement_fn, shift_fn = displacement_fn_for_bundle(bundle)
        dt, kT = as_integration_scalars(dt, kT, dtype=bundle.positions.dtype)

        if integration_prefix is not None:
            positions_init = positions_with_prefix(bundle, integration_prefix)
            masses = unit_masses(integration_prefix, bundle.positions.dtype)
            water_indices = None
        else:
            positions_init = active_positions(bundle)
            masses = masses_for_bundle(bundle)
            water_indices = water_indices_for_integration(bundle)

        # Factory defaults are placeholders; init/apply always receive traced dt/kT.
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
        state = init_fn(key, positions_init, kT=kT)

        def step_fn(carry: Any, _: Any) -> tuple[Any, Any]:
            new_state = apply_fn(carry, kT=kT, dt=dt)
            return new_state, new_state.position

        state, positions_array = jax.lax.scan(step_fn, state, None, length=n_steps)
        if trim_output:
            positions_array = trim_trajectory_positions(positions_array, bundle)

        observable_values: dict[str, Any] = {}
        if observables:
            for name, observable in observables.items():
                observable_values[name] = observable.compute(state)

        return Trajectory(
            positions=positions_array,
            observable_values=observable_values,
            n_steps=n_steps,
        )


def _observables_for_bundle(
    bundle: Any, observables: dict[str, Any]
) -> dict[str, Any]:
    """Re-bind bundle-scoped observables (e.g. Energy) for multi-bundle runs."""
    from prolix.api.observables import Energy

    rebound: dict[str, Any] = {}
    for name, observable in observables.items():
        if isinstance(observable, Energy):
            rebound[name] = Energy(
                energy_fn=observable.energy_fn,
                bundle=bundle,
            )
        else:
            rebound[name] = observable
    return rebound
