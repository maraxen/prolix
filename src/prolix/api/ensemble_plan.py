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

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from prolix.api.observables import Trajectory

RunMode = Literal["trajectory", "inference"]


class _NLDispatchCarry(NamedTuple):
    """``dispatch_n_steps_inference`` carry for the neighbor-list-aware step (debt 760).

    ``did_overflow`` is OR-accumulated across the loop (never branched on
    in-loop -- ``lax.while_loop`` can't do a host reallocation mid-loop) and
    host-checked once after the loop returns; see ``_run_single_inference``.
    """

    langevin_state: Any
    neighbor: Any
    did_overflow: Any


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
        use_neighbor_list: bool = False,
        nl_update_every: int = 20,
        use_flash_forces: bool = False,
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
                (default: every step when ``xtc_path`` set). Not yet supported
                together with ``use_neighbor_list=True`` (raises).
            dt_unit: ``'fs'`` (default) or ``'akma'``.
            gamma: Langevin friction in **ps⁻¹** (converted via ``gamma_ps_to_akma``).
            run_mode: ``'trajectory'`` (scan + full traj stack, default) or
                ``'inference'`` (while_loop carry-only; not AD-safe).
            save_every: Inference XTC stride (steps). Ignored in trajectory mode
                unless ``xtc_path`` is set (then full traj is flushed once).
            use_neighbor_list: debt 760/802 — compute direct-space LJ/Coulomb
                via O(N*K) neighbor-list kernels instead of the dense O(N²)
                path. ``run_mode='inference'`` only (raises otherwise if
                True); periodic (``box_size``) bundles only. Works for both
                single-bundle and multi-bundle (vmapped/stacked) dispatch --
                stacked bundles must share identical ``box_size``/
                ``cutoff_distance`` (one shared ``NeighborList`` seed is
                built and reused across the batch, debt 802). The rare
                "defensive fallback" per-bundle loop inside a shape-class
                group (should not normally trigger) does not support NL and
                raises clearly rather than silently degrading compile-sharing.
                Default False preserves existing dense-path behavior exactly.
            nl_update_every: Steps between neighbor-list rebuilds when
                ``use_neighbor_list=True`` (ignored otherwise).
            use_flash_forces: debt 761 — compute forces via
                ``force_fn_from_bundle`` (FlashMD's tiled/checkpointed
                ``flash_explicit_forces`` kernel) instead of autodiff through
                ``energy_fn_from_bundle``'s dense O(N²) energy. Verified to
                float32 precision against the autodiff path on real periodic
                PME-solvated bundles (1VII/2GB1); raises for implicit-solvent
                or non-periodic bundles (unverified paths — see
                ``force_fn_from_bundle``'s docstring) and when combined with
                ``use_neighbor_list=True`` (``single_padded_force`` has no
                neighbor-list branch yet). Works in both ``run_mode``s and for
                both single-bundle and multi-bundle (stacked) dispatch.
                Default False preserves existing autodiff-path behavior
                exactly.

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

        if use_neighbor_list and run_mode != "inference":
            raise ValueError(
                "use_neighbor_list=True requires run_mode='inference' (debt 760/802 "
                "scope; see plan doc)"
            )
        if use_neighbor_list and xtc_path is not None:
            raise ValueError(
                "use_neighbor_list=True does not yet support xtc_path streaming"
            )
        if use_flash_forces and use_neighbor_list:
            raise ValueError(
                "use_flash_forces=True cannot be combined with "
                "use_neighbor_list=True -- single_padded_force (debt 761's "
                "FlashMD path) has no neighbor-list branch yet."
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
                use_neighbor_list=use_neighbor_list,
                nl_update_every=nl_update_every,
                use_flash_forces=use_flash_forces,
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
            use_flash_forces=use_flash_forces,
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
        use_flash_forces: bool = False,
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
                use_flash_forces=use_flash_forces,
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
            use_flash_forces=use_flash_forces,
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
        use_neighbor_list: bool = False,
        nl_update_every: int = 20,
        use_flash_forces: bool = False,
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
                use_neighbor_list=use_neighbor_list,
                nl_update_every=nl_update_every,
                use_flash_forces=use_flash_forces,
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
            use_neighbor_list=use_neighbor_list,
            nl_update_every=nl_update_every,
            use_flash_forces=use_flash_forces,
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
        use_neighbor_list: bool = False,
        nl_update_every: int = 20,
        use_flash_forces: bool = False,
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
                        use_neighbor_list=use_neighbor_list,
                        nl_update_every=nl_update_every,
                        use_flash_forces=use_flash_forces,
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
                        use_flash_forces=use_flash_forces,
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
                    use_neighbor_list=use_neighbor_list,
                    nl_update_every=nl_update_every,
                    use_flash_forces=use_flash_forces,
                )
            else:
                # Defensive: same shape_spec should always stack; fall back
                # without inventing DedupGather of seeded MD. NL is not
                # extended to this rare fallback path (each bundle would
                # independently allocate/compile its own NL, defeating the
                # compile-sharing this whole path exists to preserve) --
                # raise clearly rather than silently ignoring the request.
                if use_neighbor_list:
                    raise RuntimeError(
                        "use_neighbor_list=True: this shape-class group failed "
                        "can_jit_vmap_n_mols's stackability check and fell back to "
                        "the defensive per-bundle loop -- NL is not supported there "
                        "(would defeat the compile-sharing this path exists for). "
                        "This indicates a real shape_spec/array-shape mismatch "
                        "within a group partition_bundles_by_shape considered "
                        "identical -- investigate before retrying."
                    )
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
                                use_flash_forces=use_flash_forces,
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
                                use_flash_forces=use_flash_forces,
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
        use_neighbor_list: bool = False,
        nl_update_every: int = 20,
        use_flash_forces: bool = False,
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

        if use_neighbor_list and run_mode != "inference":
            raise ValueError("use_neighbor_list=True requires run_mode='inference'")

        def run_one(bundle, seed_i: jnp.ndarray, neighbor: Any = None) -> Trajectory:
            obs = observables
            if observables is not None:
                obs = _observables_for_bundle(bundle, observables, atom_mask=bundle.atom_mask)
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
                    use_neighbor_list=use_neighbor_list,
                    nl_update_every=nl_update_every,
                    initial_neighbor=neighbor,
                    use_flash_forces=use_flash_forces,
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
                use_flash_forces=use_flash_forces,
            )

        import jax

        stacked_neighbor = None
        neighbor_fn = None
        if use_neighbor_list:
            stacked_neighbor, neighbor_fn = self._build_stacked_neighbor_seed(
                bundles, integration_prefix
            )

        if run_mode == "inference":
            # One XLA program: vmap × while_loop (step axis inside jit boundary).
            @jax.jit
            def _batched(sb, sd, snb):
                return dispatch_n_mols(plan, n_systems, run_one, sb, sd, extra_stacked=snb)

            batched = _batched(stacked, seeds, stacked_neighbor)

            if use_neighbor_list:
                overflow_arr = batched.observable_values.pop("_nl_did_overflow", None)
                if overflow_arr is not None and bool(jnp.any(overflow_arr)):
                    bumped_capacity = int(0.5 * stacked_neighbor.idx.shape[-1])
                    retried_neighbor = self._reallocate_stacked_neighbor_seed(
                        neighbor_fn, bundles, integration_prefix, extra_capacity=bumped_capacity
                    )
                    batched = _batched(stacked, seeds, retried_neighbor)
                    overflow_arr = batched.observable_values.pop("_nl_did_overflow", None)
                    if overflow_arr is not None and bool(jnp.any(overflow_arr)):
                        raise RuntimeError(
                            "Stacked neighbor list overflowed even after reallocating "
                            f"with +{bumped_capacity} extra capacity -- capacity formula "
                            "(compute_nl_capacity) likely needs a larger safety_factor "
                            "for this shape class."
                        )
        else:
            batched = dispatch_n_mols(
                plan,
                n_systems,
                run_one,
                stacked,
                seeds,
            )
        return unstack_trajectories(batched, bundles)

    @staticmethod
    def _build_stacked_neighbor_seed(
        bundles: list[Any], integration_prefix: int
    ) -> tuple[Any, Any]:
        """debt 802: one shared, vmap-safe ``NeighborList`` per bundle, pre-stacked.

        ``neighbor_fn.allocate()`` is host-only and cannot run on a traced
        position array, so this must happen *before* any bundle enters the
        vmapped/jitted dispatch. Each bundle's per-replica ``NeighborList`` is
        derived via ``seed.update(positions)`` rather than an independent
        ``.allocate()`` call — ``.allocate()`` computes a fresh, data-dependent
        capacity every time (confirmed empirically: two different position
        arrays of the same size/box/cutoff allocate to *different* capacities
        and are therefore not stackable), while ``.update()`` reuses `self`'s
        static fields (capacity, cell-list closures) applied to new positions,
        so every bundle's result shares identical pytree metadata with the
        seed and with each other — directly stackable via
        ``jax.tree.map(jnp.stack)`` and safe to slice per-replica under vmap.
        """
        import jax
        import jax.numpy as jnp

        from prolix.api.bundle_md import (
            _host_float,
            displacement_fn_for_bundle,
            positions_with_prefix,
        )
        from prolix.physics import neighbor_list as nl_mod

        ref = bundles[0]
        box_vec = jnp.diag(ref.box)
        cutoff = _host_float(ref.cutoff_distance, 9.0)
        for b in bundles[1:]:
            if not jnp.array_equal(jnp.diag(b.box), box_vec) or _host_float(
                b.cutoff_distance, 9.0
            ) != cutoff:
                raise ValueError(
                    "use_neighbor_list=True requires every stacked bundle to share "
                    "the same box_size and cutoff_distance (decision D1) -- got a "
                    "mismatch, so one shared NeighborList seed cannot be built."
                )

        displacement_fn, _shift_fn = displacement_fn_for_bundle(ref)
        neighbor_fn = nl_mod.make_neighbor_list_fn(displacement_fn, box_vec, cutoff)

        target_capacity = nl_mod.compute_nl_capacity(integration_prefix, box_vec, cutoff)
        seed_positions = positions_with_prefix(ref, integration_prefix)
        seed = neighbor_fn.allocate(seed_positions)
        extra_needed = max(0, target_capacity - seed.idx.shape[-1])
        if extra_needed > 0:
            seed = neighbor_fn.allocate(seed_positions, extra_capacity=extra_needed)

        per_bundle = [
            seed.update(positions_with_prefix(b, integration_prefix)) for b in bundles
        ]
        stacked_neighbor = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_bundle)
        return stacked_neighbor, neighbor_fn

    @staticmethod
    def _reallocate_stacked_neighbor_seed(
        neighbor_fn: Any, bundles: list[Any], integration_prefix: int, *, extra_capacity: int
    ) -> Any:
        """Host-side reallocate-and-restack after an overflow (debt 802)."""
        import jax
        import jax.numpy as jnp

        from prolix.api.bundle_md import positions_with_prefix

        ref = bundles[0]
        seed_positions = positions_with_prefix(ref, integration_prefix)
        seed = neighbor_fn.allocate(seed_positions, extra_capacity=extra_capacity)
        per_bundle = [
            seed.update(positions_with_prefix(b, integration_prefix)) for b in bundles
        ]
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_bundle)

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
        use_flash_forces: bool = False,
    ) -> tuple[Any, Any, Any, Any]:
        """Shared settle_langevin init. Returns (state, apply_fn, dt, kT)."""
        import jax
        import jax.numpy as jnp

        from prolix.api.bundle_md import (
            active_positions,
            as_integration_scalars,
            energy_fn_from_bundle,
            force_fn_from_bundle,
            displacement_fn_for_bundle,
            masses_for_bundle,
            masses_with_prefix,
            positions_with_prefix,
            water_indices_for_integration,
        )
        from prolix.physics.kups_adapter import AKMA_TIME_UNIT_FS, gamma_ps_to_akma
        from prolix.physics.settle import settle_langevin

        # debt 761: force_fn_from_bundle returns an (N, 3) force array rather
        # than a scalar energy -- settle_langevin's energy_or_force_fn
        # auto-detects which via jax.eval_shape (see
        # _make_settle_compatible_force_fn/make_force_fn_like_canonicalize),
        # so no other change is needed here for the integrator itself to
        # skip the dense-energy autodiff pass.
        energy_or_force_fn = (
            force_fn_from_bundle(bundle) if use_flash_forces else energy_fn_from_bundle(bundle)
        )
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
            # bundle.water_indices is already bucket-sized (static shape from
            # shape_spec.water_bucket_idx) — no host int() needed, unlike
            # water_indices_for_integration below which slices to the real
            # (traced) count and is unsafe under vmap. water_mask marks real
            # vs. padding rows so settle_langevin can redirect padding
            # scatters away from real atoms. See B1-SETTLE-STACK
            # (.praxia/docs/specs/260715_b1-settle-stack.md).
            #
            # WATER_BUCKETS has no zero-size entry (smallest is 8), so a
            # bundle with zero real water molecules still gets
            # water_indices.shape[0] == 8 (all masked-off placeholder rows).
            # settle_langevin's "no water -> plain Langevin" fallback keys
            # off water_indices.shape[0] == 0, which can therefore never
            # trigger here -- silently running the SETTLE rigid-body
            # integrator (different PRNG-consumption pattern) instead of
            # plain Langevin for non-water systems (debt 841). shape_spec is
            # this bundle's static field (host-known, vmap-safe even when
            # this runs under vmap), so checking it here is safe.
            if bundle.shape_spec.has_real_water:
                water_indices = bundle.water_indices
                water_mask = bundle.water_mask
            else:
                water_indices = None
                water_mask = None
            # Padding atoms carry unit mass (masses_with_prefix), not zero, so
            # they draw genuine nonzero initial momentum -- atom_mask lets the
            # no-water fallback re-center momentum using only real atoms
            # (debt 841; see settle_langevin's atom_mask docstring).
            atom_mask = bundle.atom_mask[:integration_prefix]
        else:
            positions_init = active_positions(bundle)
            masses = masses_for_bundle(bundle)
            water_indices = water_indices_for_integration(bundle)
            water_mask = None
            atom_mask = None

        init_fn, apply_fn = settle_langevin(
            energy_or_force_fn,
            shift_fn,
            dt=1.0,
            kT=1.0,
            gamma=gamma_akma,
            mass=masses,
            water_indices=water_indices,
            project_ou_momentum_rigid=True,
            water_mask=water_mask,
            atom_mask=atom_mask,
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
        use_flash_forces: bool = False,
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
            use_flash_forces=use_flash_forces,
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
        use_neighbor_list: bool = False,
        nl_update_every: int = 20,
        initial_neighbor: Any = None,
        use_flash_forces: bool = False,
    ) -> Trajectory:
        """Carry-only MD via while_loop; Trajectory holds final frame only.

        Args:
            initial_neighbor: debt 802 -- a pre-allocated ``jax_md.NeighborList``
                supplied by ``_run_stacked_dispatch`` when this call happens
                *inside* a vmapped/jitted context (``neighbor_fn.allocate()``
                is host-only and cannot run on a traced position array, so
                the stacked path must allocate outside and pass the result
                in). When given, internal allocation and the
                overflow-then-reallocate retry are both skipped -- overflow
                is instead surfaced via
                ``observable_values["_nl_did_overflow"]`` for the caller to
                check and retry the whole stack if needed. ``None`` (default)
                preserves the original single-bundle behavior exactly
                (internal allocation + retry, as verified in Phase 6 steps 4-5).
        """
        import jax
        import jax.numpy as jnp

        from prolix.api.bundle_md import trim_trajectory_positions
        from prolix.api.ensemble_dispatch import dispatch_n_steps_inference
        from prolix.api.observables import Trajectory

        if use_neighbor_list:
            if xtc_path is not None:
                raise ValueError(
                    "use_neighbor_list=True does not yet support xtc_path streaming"
                )
            if not bundle.shape_spec.has_pbc:
                raise ValueError(
                    "use_neighbor_list=True requires a periodic bundle (box_size)"
                )
            if integration_prefix is None:
                # Ghost atoms (debt 772's uniform lattice) must be present for
                # the shape-class NL capacity formula (compute_nl_capacity) to
                # apply -- default to the bundle's full padded/bucket size
                # rather than the real-atom-only slice active_positions()
                # would otherwise give _setup_integrator.
                integration_prefix = int(bundle.positions.shape[0])

        state, apply_fn, dt_s, kT_s = self._setup_integrator(
            bundle,
            dt,
            kT,
            seed,
            integration_prefix=integration_prefix,
            dt_unit=dt_unit,
            gamma=gamma,
            use_flash_forces=use_flash_forces,
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

        nl_overflow_for_caller = None
        if use_neighbor_list:
            import equinox as eqx

            ghost_target_positions = state.positions
            pad_mask_3d = bundle.atom_mask[:, None]
            update_every = int(nl_update_every)

            def _nl_step_fn(carry: _NLDispatchCarry, step_i: Any) -> _NLDispatchCarry:
                langevin_state, neighbor, did_overflow = carry
                should_update = (step_i % update_every) == 0
                new_neighbor = jax.lax.cond(
                    should_update,
                    lambda n: n.update(langevin_state.positions),
                    lambda n: n,
                    neighbor,
                )
                did_overflow = did_overflow | new_neighbor.did_buffer_overflow.astype(bool)
                new_langevin_state = apply_fn(
                    langevin_state, neighbor=new_neighbor, kT=kT_s, dt=dt_s
                )
                # Ghost-position/momentum pinning (debt 772's deferred piece):
                # settle.py's O-step has no atom_mask awareness, so ghosts
                # would otherwise random-walk off their fixed NL-capacity
                # lattice. Masking happens here, at the dispatch layer,
                # outside settle.py's validated internals -- matches the
                # existing batched_simulate.py precedent (equilibrate_single).
                # NVTLangevinState (typing.py:611) has no .replace() of its
                # own (that method belongs to the earlier IntegratorState
                # class, typing.py:511) -- reconstruct directly via eqx.tree_at.
                new_langevin_state = eqx.tree_at(
                    lambda s: (s.positions, s.momentum),
                    new_langevin_state,
                    (
                        jnp.where(
                            pad_mask_3d, new_langevin_state.positions, ghost_target_positions
                        ),
                        jnp.where(pad_mask_3d, new_langevin_state.momentum, 0.0),
                    ),
                )
                return _NLDispatchCarry(new_langevin_state, new_neighbor, did_overflow)

            if initial_neighbor is not None:
                # debt 802 (stacked/vmapped path): allocation and retry both
                # happen host-side in _run_stacked_dispatch, outside this
                # (potentially traced/vmapped) call -- neighbor_fn.allocate()
                # cannot run on a tracer. Surface overflow to the caller
                # instead of host-checking/retrying here.
                nbr0 = initial_neighbor
                final_carry = dispatch_n_steps_inference(
                    _nl_step_fn, _NLDispatchCarry(state, nbr0, jnp.array(False)), int(n_steps)
                )
                nl_overflow_for_caller = final_carry.did_overflow
            else:
                from prolix.api.bundle_md import _host_float, displacement_fn_for_bundle
                from prolix.physics import neighbor_list as nl_mod

                displacement_fn, _shift_fn = displacement_fn_for_bundle(bundle)
                box_vec = jnp.diag(bundle.box)
                cutoff = _host_float(bundle.cutoff_distance, 9.0)
                neighbor_fn = nl_mod.make_neighbor_list_fn(displacement_fn, box_vec, cutoff)

                nbr0 = neighbor_fn.allocate(state.positions)
                init_carry = _NLDispatchCarry(state, nbr0, jnp.array(False))
                final_carry = dispatch_n_steps_inference(
                    _nl_step_fn, init_carry, int(n_steps)
                )
                if bool(final_carry.did_overflow):
                    bumped_capacity = int(0.5 * nbr0.idx.shape[1])
                    retried_neighbor = neighbor_fn.allocate(
                        final_carry.langevin_state.positions,
                        extra_capacity=bumped_capacity,
                    )
                    retry_carry = _NLDispatchCarry(
                        state, retried_neighbor, jnp.array(False)
                    )
                    final_carry = dispatch_n_steps_inference(
                        _nl_step_fn, retry_carry, int(n_steps)
                    )
                    if bool(final_carry.did_overflow):
                        raise RuntimeError(
                            "Neighbor list overflowed even after reallocating with "
                            f"+{bumped_capacity} extra capacity -- capacity formula "
                            "(compute_nl_capacity) likely needs a larger safety_factor "
                            "for this system."
                        )
            state = final_carry.langevin_state
        else:

            def step_fn(carry: Any, _step_i: Any) -> Any:
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
        if nl_overflow_for_caller is not None:
            observable_values["_nl_did_overflow"] = nl_overflow_for_caller

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
    bundle: Any, observables: dict[str, Any], *, atom_mask: Any | None = None
) -> dict[str, Any]:
    """Re-bind bundle-scoped observables (e.g. Energy) for multi-bundle runs.

    Args:
        atom_mask: Bucket-sized (N,) bool, True for real (non-padding) atoms,
            when this bundle's integration state will be padded (the
            stacked/vmapped dispatch path). `None` (default) for unpadded
            dispatch (state already contains only real atoms). Threaded into
            KineticEnergy/Temperature/Pressure, whose momentum**2/(2*mass)
            sums are otherwise polluted by padding atoms (unit mass, nonzero
            momentum after the first integration step) -- debt 841.
    """
    import equinox as eqx

    from prolix.api.observables import Energy, KineticEnergy, Pressure, Temperature

    rebound: dict[str, Any] = {}
    for name, observable in observables.items():
        if isinstance(observable, Energy):
            rebound[name] = Energy(
                energy_fn=observable.energy_fn,
                bundle=bundle,
            )
        elif atom_mask is not None and isinstance(
            observable, (KineticEnergy, Temperature, Pressure)
        ):
            rebound[name] = eqx.tree_at(
                lambda o: o.atom_mask, observable, atom_mask, is_leaf=lambda x: x is None
            )
        else:
            rebound[name] = observable
    return rebound
