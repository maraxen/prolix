"""EnsemblePlan: high-level API for batch molecular dynamics simulations.

Multi-bundle dispatch uses xtrax.tiling via EnsembleMDPlanner (#1842).

Run modes (B1-INFER / B1-XTRAX-WIRE):
  - ``trajectory`` (default): ``lax.scan`` via ``dispatch_n_steps`` — stacks
    ``(n_steps, N, 3)``; AD-compatible pathological / baseline path.
  - ``inference``: ``lax.while_loop`` via ``dispatch_n_steps_inference`` —
    carry-only (final frame); not reverse-mode AD safe.

N_MOLS (using-xtrax):
  - Host ``partition_bundles_by_shape`` → K shape classes (Python over K only).
  - Within class: Vmap/SafeMap over replicas with distinct seeds.
  - DedupGather only for topology-keyed bodies (see ``ensemble_dedup``), never
    seeded Langevin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from prolix.api.observables import Trajectory

RunMode = Literal["trajectory", "inference"]


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
        *,
        xtc_path: str | Any | None = None,
        dt_unit: str = "fs",
        gamma: float = 10.0,
        run_mode: RunMode = "trajectory",
        save_every: int | None = None,
    ) -> Trajectory | list[Trajectory]:
        """Run MD simulation over all bundles.

        Args:
            n_steps: Number of MD steps (host-static).
            dt: Timestep. By default (**``dt_unit='fs'``**) this is **femtoseconds**
                and is converted to AKMA via ``dt_fs_to_akma`` (XR-VACUUM-DT).
                Pass ``dt_unit='akma'`` if ``dt`` is already in AKMA time units.
            kT: Thermal energy in kcal/mol (``jnp.ndarray`` scalar or float).
            seed: PRNG seed for thermostat noise (int or uint32 array).
            observables: Optional name → Observable map; final-step values
                are stored in Trajectory.observable_values.
            xtc_path: Optional path for MD traj XTC output (XR-SINK-XTC).
                Single-bundle only. Trajectory mode: post-run flush.
                Inference mode: stream frames every ``save_every`` steps
                (default: every step when ``xtc_path`` set).
            dt_unit: ``'fs'`` (default) or ``'akma'``.
            gamma: Langevin friction in **ps⁻¹** (converted via ``gamma_ps_to_akma``).
            run_mode: ``'trajectory'`` (scan + full traj stack, default) or
                ``'inference'`` (while_loop carry-only; not AD-safe).
            save_every: Inference XTC stride (steps). Ignored in trajectory mode
                unless ``xtc_path`` is set (then full traj is flushed once).

        Returns:
            Trajectory for a single bundle, or list[Trajectory] when
            len(bundles) > 1. Inference mode returns positions with shape
            ``(1, n_atoms, 3)`` (final frame only).
        """
        if not self.bundles:
            raise ValueError("EnsemblePlan requires at least one bundle")
        if dt_unit not in ("fs", "akma"):
            raise ValueError(f"dt_unit must be 'fs' or 'akma', got {dt_unit!r}")
        if run_mode not in ("trajectory", "inference"):
            raise ValueError(
                f"run_mode must be 'trajectory' or 'inference', got {run_mode!r}"
            )

        if xtc_path is not None and len(self.bundles) != 1:
            raise ValueError(
                "xtc_path is supported for single-bundle EnsemblePlan runs only"
            )

        if run_mode == "inference":
            return self._run_inference(
                n_steps,
                dt,
                kT,
                seed,
                observables=observables,
                xtc_path=xtc_path,
                dt_unit=dt_unit,
                gamma=gamma,
                save_every=save_every,
            )

        return self._run_trajectory(
            n_steps,
            dt,
            kT,
            seed,
            observables=observables,
            xtc_path=xtc_path,
            dt_unit=dt_unit,
            gamma=gamma,
        )

    def _run_trajectory(
        self,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None,
        *,
        xtc_path: str | Any | None,
        dt_unit: str,
        gamma: float,
    ) -> Trajectory | list[Trajectory]:
        """Pathological / baseline path: scan + full trajectory stack."""
        if len(self.bundles) == 1:
            traj = self._run_single(
                self.bundles[0],
                n_steps,
                dt,
                kT,
                seed,
                observables=observables,
                dt_unit=dt_unit,
                gamma=gamma,
            )
            if xtc_path is not None:
                from prolix.api.xtc_sink import write_positions_xtc

                write_positions_xtc(xtc_path, traj.positions)
            return traj

        return self._run_grouped(
            n_steps,
            dt,
            kT,
            seed,
            observables=observables,
            dt_unit=dt_unit,
            gamma=gamma,
            run_mode="trajectory",
        )

    def _run_inference(
        self,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None,
        *,
        xtc_path: str | Any | None,
        dt_unit: str,
        gamma: float,
        save_every: int | None,
    ) -> Trajectory | list[Trajectory]:
        """Inference path: while_loop carry-only; group-by-shape for multi-bundle."""
        if len(self.bundles) == 1:
            return self._run_single_inference(
                self.bundles[0],
                n_steps,
                dt,
                kT,
                seed,
                observables=observables,
                dt_unit=dt_unit,
                gamma=gamma,
                xtc_path=xtc_path,
                save_every=save_every,
            )

        return self._run_grouped(
            n_steps,
            dt,
            kT,
            seed,
            observables=observables,
            dt_unit=dt_unit,
            gamma=gamma,
            run_mode="inference",
        )

    def _run_grouped(
        self,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None,
        *,
        dt_unit: str,
        gamma: float,
        run_mode: RunMode,
    ) -> list[Trajectory]:
        """Host-partition by shape_spec; Vmap/SafeMap within each class (xtrax).

        Python iterates only over distinct shape classes (K≪N). Within a
        stackable class there is no per-system Python loop. DedupGather is
        **not** used for seeded Langevin (would scatter identical trajs).
        """
        from prolix.api.bundle_stack import can_jit_vmap_n_mols
        from prolix.api.ensemble_dedup import partition_bundles_by_shape
        from prolix.api.ensemble_planner import EnsembleMDPlanner

        groups = partition_bundles_by_shape(self.bundles)
        out: list[Trajectory | None] = [None] * len(self.bundles)
        planner = EnsembleMDPlanner()

        for indices in groups:
            group = [self.bundles[i] for i in indices]
            base_seed = int(seed) + indices[0]
            group_plan = planner.plan(group)

            if len(group) == 1:
                bundle = group[0]
                obs = observables
                if observables is not None:
                    obs = _observables_for_bundle(bundle, observables)
                if run_mode == "inference":
                    traj = self._run_single_inference(
                        bundle,
                        n_steps,
                        dt,
                        kT,
                        base_seed,
                        observables=obs,
                        dt_unit=dt_unit,
                        gamma=gamma,
                    )
                else:
                    traj = self._run_single(
                        bundle,
                        n_steps,
                        dt,
                        kT,
                        base_seed,
                        observables=obs,
                        dt_unit=dt_unit,
                        gamma=gamma,
                    )
                trajs = [traj]
            elif can_jit_vmap_n_mols(group):
                trajs = self._run_stacked_dispatch(
                    n_steps,
                    dt,
                    kT,
                    base_seed,
                    observables=observables,
                    dt_unit=dt_unit,
                    gamma=gamma,
                    run_mode=run_mode,
                    bundles_override=group,
                    plan_override=group_plan,
                )
            else:
                # Defensive: same shape_spec should always stack; fall back
                # without inventing DedupGather of seeded MD.
                trajs = []
                for j, bundle in enumerate(group):
                    obs = observables
                    if observables is not None:
                        obs = _observables_for_bundle(bundle, observables)
                    if run_mode == "inference":
                        trajs.append(
                            self._run_single_inference(
                                bundle,
                                n_steps,
                                dt,
                                kT,
                                base_seed + j,
                                observables=obs,
                                dt_unit=dt_unit,
                                gamma=gamma,
                            )
                        )
                    else:
                        trajs.append(
                            self._run_single(
                                bundle,
                                n_steps,
                                dt,
                                kT,
                                base_seed + j,
                                observables=obs,
                                dt_unit=dt_unit,
                                gamma=gamma,
                            )
                        )
            for idx, traj in zip(indices, trajs, strict=True):
                out[idx] = traj

        assert all(t is not None for t in out)
        return out  # type: ignore[return-value]

    def _run_stacked_dispatch(
        self,
        n_steps: int,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None = None,
        *,
        dt_unit: str = "fs",
        gamma: float = 10.0,
        run_mode: RunMode = "trajectory",
        bundles_override: list[Any] | None = None,
        plan_override: Any | None = None,
    ) -> list[Trajectory]:
        """JIT vmap / safe_map over stack-compatible bundles (#2645)."""
        import jax.numpy as jnp

        from prolix.api.bundle_stack import (
            integration_prefix_for_bundles,
            stack_molecular_bundles,
            unstack_trajectories,
        )
        from prolix.api.ensemble_dispatch import dispatch_n_mols
        from prolix.api.ensemble_planner import EnsembleMDPlanner

        bundles = bundles_override if bundles_override is not None else self.bundles
        stacked = stack_molecular_bundles(bundles)
        n_systems = len(bundles)
        seeds = jnp.arange(n_systems, dtype=jnp.int32) + seed
        integration_prefix = integration_prefix_for_bundles(bundles)
        if plan_override is not None:
            plan = plan_override
        elif bundles_override is not None:
            plan = EnsembleMDPlanner().plan(bundles)
        else:
            plan = self.batch_plan

        def run_one(bundle, seed_i: jnp.ndarray) -> Trajectory:
            obs = observables
            if observables is not None:
                obs = _observables_for_bundle(bundle, observables)
            if run_mode == "inference":
                return self._run_single_inference(
                    bundle,
                    n_steps,
                    dt,
                    kT,
                    seed_i,
                    observables=obs,
                    integration_prefix=integration_prefix,
                    trim_output=False,
                    dt_unit=dt_unit,
                    gamma=gamma,
                )
            return self._run_single(
                bundle,
                n_steps,
                dt,
                kT,
                seed_i,
                observables=obs,
                integration_prefix=integration_prefix,
                trim_output=False,
                dt_unit=dt_unit,
                gamma=gamma,
            )

        import jax

        if run_mode == "inference":
            # One XLA program: vmap × while_loop (step axis inside jit boundary).
            @jax.jit
            def _batched(sb, sd):
                return dispatch_n_mols(plan, n_systems, run_one, sb, sd)

            batched = _batched(stacked, seeds)
        else:
            batched = dispatch_n_mols(
                plan,
                n_systems,
                run_one,
                stacked,
                seeds,
            )
        return unstack_trajectories(batched, bundles)

    def _setup_integrator(
        self,
        bundle: Any,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        *,
        integration_prefix: int | None,
        dt_unit: str,
        gamma: float,
    ) -> tuple[Any, Any, Any, Any]:
        """Shared settle_langevin init. Returns (state, apply_fn, dt, kT)."""
        import jax
        import jax.numpy as jnp

        from prolix.api.bundle_md import (
            active_positions,
            as_integration_scalars,
            energy_fn_from_bundle,
            displacement_fn_for_bundle,
            masses_for_bundle,
            masses_with_prefix,
            positions_with_prefix,
            water_indices_for_integration,
        )
        from prolix.physics.kups_adapter import AKMA_TIME_UNIT_FS, gamma_ps_to_akma
        from prolix.physics.settle import settle_langevin

        energy_fn = energy_fn_from_bundle(bundle)
        _displacement_fn, shift_fn = displacement_fn_for_bundle(bundle)

        dt_arr = jnp.asarray(dt)
        if dt_unit == "fs":
            dt_akma = dt_arr / AKMA_TIME_UNIT_FS
        else:
            dt_akma = dt_arr
        gamma_akma = gamma_ps_to_akma(float(gamma))

        dt_s, kT_s = as_integration_scalars(
            dt_akma, kT, dtype=bundle.positions.dtype
        )

        if integration_prefix is not None:
            positions_init = positions_with_prefix(bundle, integration_prefix)
            masses = masses_with_prefix(bundle, integration_prefix)
            # SETTLE water indices: host int() is unsafe under vmap; stacked
            # path leaves SETTLE off (None). Single-system path uses the
            # non-prefix branch below.
            water_indices = None
        else:
            positions_init = active_positions(bundle)
            masses = masses_for_bundle(bundle)
            water_indices = water_indices_for_integration(bundle)

        init_fn, apply_fn = settle_langevin(
            energy_fn,
            shift_fn,
            dt=1.0,
            kT=1.0,
            gamma=gamma_akma,
            mass=masses,
            water_indices=water_indices,
            project_ou_momentum_rigid=True,
        )

        key = jax.random.PRNGKey(jnp.asarray(seed, dtype=jnp.uint32))
        state = init_fn(key, positions_init, kT=kT_s)
        return state, apply_fn, dt_s, kT_s

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
        dt_unit: str = "fs",
        gamma: float = 10.0,
    ) -> Trajectory:
        from prolix.api.bundle_md import trim_trajectory_positions
        from prolix.api.ensemble_dispatch import dispatch_n_steps
        from prolix.api.observables import Trajectory

        state, apply_fn, dt_s, kT_s = self._setup_integrator(
            bundle,
            dt,
            kT,
            seed,
            integration_prefix=integration_prefix,
            dt_unit=dt_unit,
            gamma=gamma,
        )

        def step_fn(carry: Any, _: Any) -> tuple[Any, Any]:
            new_state = apply_fn(carry, kT=kT_s, dt=dt_s)
            return new_state, new_state.position

        state, positions_array = dispatch_n_steps(step_fn, state, int(n_steps))
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

    def _run_single_inference(
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
        dt_unit: str = "fs",
        gamma: float = 10.0,
        xtc_path: str | Any | None = None,
        save_every: int | None = None,
    ) -> Trajectory:
        """Carry-only MD via while_loop; Trajectory holds final frame only."""
        import jax.numpy as jnp

        from prolix.api.bundle_md import trim_trajectory_positions
        from prolix.api.ensemble_dispatch import dispatch_n_steps_inference
        from prolix.api.observables import Trajectory

        state, apply_fn, dt_s, kT_s = self._setup_integrator(
            bundle,
            dt,
            kT,
            seed,
            integration_prefix=integration_prefix,
            dt_unit=dt_unit,
            gamma=gamma,
        )

        sink_cm = None
        on_step = None
        if xtc_path is not None:
            import numpy as np

            from prolix.api.xtc_sink import XtcFrameSink

            stride = 1 if save_every is None else max(1, int(save_every))
            n_write = int(jnp.asarray(bundle.n_atoms))
            sink_cm = XtcFrameSink(path=xtc_path)
            sink = sink_cm.__enter__()

            def on_step(step_i, positions):
                # Host side-effect: positions are (N, 3) from io_callback.
                i = int(np.asarray(step_i))
                if i % stride == 0:
                    pos = np.asarray(positions)[:n_write]
                    sink(pos)

        def step_fn(carry: Any) -> Any:
            return apply_fn(carry, kT=kT_s, dt=dt_s)

        try:
            state = dispatch_n_steps_inference(
                step_fn, state, int(n_steps), on_step=on_step
            )
        finally:
            if sink_cm is not None:
                sink_cm.__exit__(None, None, None)

        # Final frame only: (1, n_atoms, 3)
        final_pos = state.position
        if trim_output and integration_prefix is None:
            # trim expects (steps, atoms, 3)
            stacked = final_pos[None, ...]
            stacked = trim_trajectory_positions(stacked, bundle)
            positions_array = stacked
        else:
            n_active = (
                integration_prefix
                if integration_prefix is not None
                else int(jnp.asarray(bundle.n_atoms))
            )
            positions_array = final_pos[None, :n_active, :]

        observable_values: dict[str, Any] = {}
        if observables:
            for name, observable in observables.items():
                observable_values[name] = observable.compute(state)

        return Trajectory(
            positions=positions_array,
            observable_values=observable_values,
            n_steps=n_steps,
        )


def _group_indices_by_shape_spec(bundles: list[Any]) -> list[list[int]]:
    """Deprecated alias — use ``partition_bundles_by_shape``."""
    from prolix.api.ensemble_dedup import partition_bundles_by_shape

    return partition_bundles_by_shape(bundles)


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
