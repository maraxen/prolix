"""EnsemblePlan: high-level API for batch molecular dynamics simulations.

Multi-bundle dispatch uses xtrax.tiling via EnsembleMDPlanner (#1842).

Run modes (B1-INFER / B1-XTRAX-WIRE):
  - ``trajectory`` (default): ``lax.scan`` via ``dispatch_n_steps`` — stacks
    ``(n_steps, N, 3)``; AD-compatible pathological / baseline path.
  - ``inference``: ``lax.while_loop`` via ``dispatch_n_steps_inference`` —
    carry-only (final frame); not reverse-mode AD safe.

N_MOLS (using-xtrax):
  - Host ``partition_bundles_by_shape`` → K shape classes (Python over K only).
  - Within class: Vmap/ChunkedMap over replicas with distinct seeds.
  - DedupGather only for topology-keyed bodies (see ``ensemble_dedup``), never
    seeded Langevin.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from prolix.api.bundle_md import (
    _host_float,
    active_positions,
    as_integration_scalars,
    displacement_fn_for_bundle,
    energy_fn_from_bundle,
    force_fn_from_bundle,
    masses_for_bundle,
    masses_with_prefix,
    pbc_box_for_bundle,
    positions_with_prefix,
    trim_trajectory_positions,
    water_indices_for_integration,
)
from prolix.api.bundle_stack import (
    can_jit_vmap_n_mols,
    integration_prefix_for_bundles,
    stack_molecular_bundles,
    unstack_trajectories,
)
from prolix.api.ensemble_dedup import partition_bundles_by_shape
from prolix.api.ensemble_dispatch import (
    dispatch_n_mols,
    dispatch_n_steps,
    dispatch_n_steps_inference,
)
from prolix.api.ensemble_planner import EnsembleMDPlanner
from prolix.api.observables import Energy, KineticEnergy, Pressure, Temperature, Trajectory
from prolix.api.xtc_sink import XtcFrameSink, write_positions_xtc
from prolix.physics import neighbor_list as nl_mod
from prolix.physics.kups_adapter import AKMA_TIME_UNIT_FS, gamma_ps_to_akma
from prolix.physics.nonbonded_config import (
    engine_flags,
    lj_switch_width_from_config,
    resolve_nonbonded_config,
)
from prolix.physics.settle import settle_langevin

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
    last_update_positions: Any


def _integrator_positions(state: Any) -> Any:
    """Prolix ``NVTLangevinState.positions`` or jax_md ``NVTLangevinState.position``."""
    return getattr(state, "positions", state.position)


def _bind_neighbor(energy_or_force_fn: Any, neighbor: Any) -> Any:
    """Default ``neighbor=`` on every energy/force call (init probe + first force).

    ``settle_langevin`` / jax_md ``nvt_langevin`` construct canonicalize_force
    and call init without a neighbor. Closing over the allocated list makes
    init and stepping share the NL backend (DHFR jobs 20520003 / 20528981).
    Explicit ``neighbor=`` on apply still wins via kwargs.
    """

    def _bound(positions: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("neighbor", neighbor)
        return energy_or_force_fn(positions, **kwargs)

    return _bound


#: Bounded geometric growth for host-side NL reallocation after an overflow.
#: Each attempt roughly doubles total capacity, so 3 attempts cover an 8x
#: underestimate before giving up with a diagnostic.
_NL_MAX_REALLOC_ATTEMPTS = 3

#: Cache for single-bundle inference runners keyed by bundle identity + runtime knobs.
_SINGLE_INFERENCE_CACHE: dict[Any, dict[str, Any]] = {}

#: Cache for multi-bundle stacked inference runners keyed by bundle identities + runtime knobs.
_STACKED_INFERENCE_CACHE: dict[Any, dict[str, Any]] = {}


def _nl_target_capacity(
    nl_mod: Any, n_atoms: int, box_vec: Any, cutoff: float, skin: float
) -> int:
    """Recommended NL capacity for a system, or 0 when the box has no usable volume.

    Single source of truth for both NL construction sites, so the stacked and
    single-bundle paths cannot size capacity by different rules. The single-bundle
    path used to skip capacity planning entirely and take jax_md's data-dependent
    ``allocate()`` sizing; at DHFR scale (23,558 atoms, 9 A cutoff) that came out
    ~30% below this recommendation and overflowed mid-scan (job 21835473).

    ``compute_nl_capacity``'s uniform-density estimate is only meaningful for a
    periodic box. Free-space bundles carry a nominal or degenerate box where the
    estimate is either an error (zero volume) or vanishingly small, so return 0
    there and leave jax_md's own sizing in charge. Callers only ever top capacity
    *up*, so a 0 here can never shrink an allocation.
    """
    import numpy as np

    volume = float(np.prod(np.asarray(box_vec, dtype=np.float64)))
    if volume <= 0.0:
        return 0
    return int(nl_mod.compute_nl_capacity(n_atoms, box_vec, cutoff, dr_threshold=skin))


def _encode_bundle_nl(nbr: Any, bundle: Any) -> Any:
    """Pre-encode 1-2/1-3/1-4 exclusion bits and filter ghost atoms in ``nbr.idx`` (O1.1 / v2.4, idempotent)."""
    if nbr is None:
        return None
    upd_fn = getattr(nbr, "update_fn", None)
    pre_enc = bool(getattr(upd_fn, "_pre_encoded_exclusions", False))
    ghosts_filt = bool(getattr(upd_fn, "_ghosts_filtered", False))
    n_rows = int(nbr.idx.shape[0])
    atom_mask = getattr(bundle, "atom_mask", None)
    mask_slice = atom_mask[:n_rows] if atom_mask is not None else None
    if pre_enc and (ghosts_filt or mask_slice is None):
        return nbr
    excl_idx = getattr(bundle, "excl_dense_indices", None)
    excl_sv = getattr(bundle, "excl_dense_scales_vdw", None)
    excl_se = getattr(bundle, "excl_dense_scales_elec", None)
    if pre_enc or excl_idx is None or excl_sv is None or excl_se is None:
        if mask_slice is None or ghosts_filt:
            return nbr
        import jax.numpy as jnp

        ei_arg = jnp.zeros((n_rows, 0), dtype=jnp.int32)
        ev_arg = jnp.zeros((n_rows, 0), dtype=jnp.float32)
        ee_arg = jnp.zeros((n_rows, 0), dtype=jnp.float32)
    else:
        ei_arg = excl_idx[:n_rows]
        ev_arg = excl_sv[:n_rows]
        ee_arg = excl_se[:n_rows]
    encoded_idx = nl_mod.encode_nl_exclusions(
        nbr.idx,
        ei_arg,
        ev_arg,
        ee_arg,
        n_atoms=n_rows,
        atom_mask=mask_slice,
    )
    return dataclasses.replace(nbr, idx=encoded_idx)


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

    def __init__(
        self,
        bundles: list[Any] | Any,
        planner: Any = None,
        *,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> None:
        if not isinstance(bundles, (list, tuple)):
            bundles = [bundles]
        self.bundles = bundles
        self.use_pallas = use_pallas
        self.use_pallas_pme = use_pallas_pme
        if planner is None and len(bundles) > 1:

            planner = EnsembleMDPlanner()
        if planner is not None:
            self.batch_plan = planner.plan(bundles)
        else:
            self.batch_plan = None

    @classmethod
    def from_bundle(
        cls,
        bundle: Any,
        planner: Any = None,
        *,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> EnsemblePlan:
        return cls(
            [bundle],
            planner,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
        )

    @classmethod
    def from_bundles(
        cls,
        bundles: list[Any],
        planner: Any = None,
        *,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> EnsemblePlan:
        return cls(
            bundles,
            planner,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
        )

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

    def _resolve_nonbonded_defaults(
        self,
        run_mode: RunMode,
        use_neighbor_list: bool | None,
        use_flash_forces: bool | None,
    ) -> tuple[bool, bool]:
        """Production defaults: NL inference+PBC, Flash trajectory+PBC, dense opt-in.

        Passing both flags False is the explicit dense autodiff path. Omitting
        both (None) never silently selects dense on periodic explicit systems.
        """
        if use_neighbor_list is not None or use_flash_forces is not None:
            return bool(use_neighbor_list), bool(use_flash_forces)
        periodic = all(
            getattr(getattr(b, "shape_spec", None), "has_pbc", False) for b in self.bundles
        )
        implicit = any(
            getattr(getattr(b, "shape_spec", None), "has_implicit_solvent", False)
            for b in self.bundles
        )
        if implicit or not periodic:
            return False, False
        if run_mode == "inference":
            return True, False
        return False, True

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
        use_neighbor_list: bool | None = None,
        nl_update_every: int = 20,
        use_flash_forces: bool | None = None,
        nonbonded: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
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
                ``None`` (default) selects production defaults: NL for
                inference + periodic explicit, Flash for trajectory +
                periodic, dense autodiff only when neither kernel applies
                (vacuum/implicit) or when both flags are passed ``False``.
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
                ``None`` (default) participates in the production default
                resolution above. Pass ``False`` together with
                ``use_neighbor_list=False`` to opt into dense autodiff.
            nonbonded: Optional ``NonbondedConfig``. Mutually exclusive with
                passing ``use_neighbor_list`` / ``use_flash_forces`` (legacy
                flags). When omitted, production defaults are unchanged.
            use_pallas: Optional override for Pallas kernels (bonded megakernel +
                direct-space nonbonded). Defaults to ``self.use_pallas``.
            use_pallas_pme: Optional override for Pallas PME charge spread/gather.
                Defaults to ``self.use_pallas_pme`` (which falls back to ``use_pallas``).

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

        eff_use_pallas = (
            use_pallas if use_pallas is not None else getattr(self, "use_pallas", None)
        )
        eff_use_pallas_pme = (
            use_pallas_pme
            if use_pallas_pme is not None
            else getattr(self, "use_pallas_pme", None)
        )

        used_legacy = use_neighbor_list is not None or use_flash_forces is not None
        cutoff0 = _host_float(getattr(self.bundles[0], "cutoff_distance", 9.0), 9.0)
        if nonbonded is None:
            use_neighbor_list, use_flash_forces = self._resolve_nonbonded_defaults(
                run_mode, use_neighbor_list, use_flash_forces
            )
            nb_cfg = resolve_nonbonded_config(
                engine_nl=bool(use_neighbor_list),
                engine_flash=bool(use_flash_forces),
                cutoff=cutoff0,
                nl_update_every=nl_update_every,
                nonbonded=None,
                used_legacy_flags=False,
            )
        else:
            nb_cfg = resolve_nonbonded_config(
                engine_nl=False,
                engine_flash=False,
                cutoff=cutoff0,
                nl_update_every=nl_update_every,
                nonbonded=nonbonded,
                used_legacy_flags=used_legacy,
            )
            use_neighbor_list, use_flash_forces = engine_flags(nb_cfg)
            nlc = nb_cfg.neighbor_list
            if nlc is not None and nlc.nl_update_every is not None:
                nl_update_every = int(nlc.nl_update_every)

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
                nb_cfg=nb_cfg,
                use_pallas=eff_use_pallas,
                use_pallas_pme=eff_use_pallas_pme,
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
            nb_cfg=nb_cfg,
            use_pallas=eff_use_pallas,
            use_pallas_pme=eff_use_pallas_pme,
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
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
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
                nb_cfg=nb_cfg,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
            )
            if xtc_path is not None:

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
            nb_cfg=nb_cfg,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
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
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
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
                nb_cfg=nb_cfg,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
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
            nb_cfg=nb_cfg,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
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
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> list[Trajectory]:
        """Host-partition by shape_spec; Vmap/ChunkedMap within each class (xtrax).
        
        Python iterates only over distinct shape classes (K≪N). Within a
        stackable class there is no per-system Python loop. DedupGather is
        **not** used for seeded Langevin (would scatter identical trajs).
        """

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
                        nb_cfg=nb_cfg,
                        use_pallas=use_pallas,
                        use_pallas_pme=use_pallas_pme,
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
                        nb_cfg=nb_cfg,
                        use_pallas=use_pallas,
                        use_pallas_pme=use_pallas_pme,
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
                    nb_cfg=nb_cfg,
                    use_pallas=use_pallas,
                    use_pallas_pme=use_pallas_pme,
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
                                nb_cfg=nb_cfg,
                                use_pallas=use_pallas,
                                use_pallas_pme=use_pallas_pme,
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
                                nb_cfg=nb_cfg,
                                use_pallas=use_pallas,
                                use_pallas_pme=use_pallas_pme,
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
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> list[Trajectory]:
        """JIT vmap / safe_map over stack-compatible bundles (#2645)."""
        import jax.numpy as jnp

        bundles = bundles_override if bundles_override is not None else self.bundles
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

        import jax

        if not isinstance(n_steps, jax.core.Tracer) and int(n_steps) < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        import os
        import sys

        import numpy as np

        b0_box = bundles[0].box
        host_box_vec_0 = (
            np.diag(np.asarray(b0_box, dtype=np.float64))
            if b0_box is not None and not isinstance(b0_box, jax.core.Tracer) and b0_box.ndim == 2
            else (
                np.asarray(b0_box, dtype=np.float64)
                if b0_box is not None and not isinstance(b0_box, jax.core.Tracer)
                else None
            )
        )

        def _make_run_one(n_steps_arg: Any, host_box_vec: Any = None):
            def run_one(bundle, seed_i: jnp.ndarray, neighbor: Any = None) -> Trajectory:
                obs = observables
                if observables is not None:
                    obs = _observables_for_bundle(bundle, observables, atom_mask=bundle.atom_mask)
                if run_mode == "inference":
                    return self._run_single_inference(
                        bundle,
                        n_steps_arg,
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
                        nb_cfg=nb_cfg,
                        use_pallas=use_pallas,
                        use_pallas_pme=use_pallas_pme,
                        host_box_vec=host_box_vec,
                    )
                return self._run_single(
                    bundle,
                    n_steps_arg,
                    dt,
                    kT,
                    seed_i,
                    observables=obs,
                    integration_prefix=integration_prefix,
                    trim_output=False,
                    dt_unit=dt_unit,
                    gamma=gamma,
                    use_flash_forces=use_flash_forces,
                    nb_cfg=nb_cfg,
                    use_pallas=use_pallas,
                    use_pallas_pme=use_pallas_pme,
                    host_box_vec=host_box_vec,
                )

            return run_one

        if run_mode == "inference":

            can_cache_stacked = (
                observables is None
                and not isinstance(n_steps, jax.core.Tracer)
                and not isinstance(dt, jax.core.Tracer)
                and not isinstance(kT, jax.core.Tracer)
                and not isinstance(seed, jax.core.Tracer)
            )
            stacked_cache_key = None
            cached_stacked = None
            _make_stacked_cache_key = None
            if can_cache_stacked:

                def _make_stacked_cache_key() -> tuple[Any, ...]:
                    _bmd = sys.modules.get("prolix.api.bundle_md")
                    _be = sys.modules.get("prolix.batched_energy")
                    _pmep = sys.modules.get("prolix.physics.pme_pallas")
                    _use_pmep = bool(
                        use_pallas_pme
                        if use_pallas_pme is not None
                        else (
                            use_pallas
                            if use_pallas is not None
                            else os.environ.get("PROLIX_USE_PALLAS", "") == "1"
                        )
                    )
                    return (
                        tuple(id(b) for b in bundles),
                        integration_prefix,
                        getattr(plan, "n_mols_strategy", None),
                        float(_host_float(dt, 1.0)),
                        dt_unit,
                        float(_host_float(kT, 0.6)),
                        float(gamma),
                        bool(use_neighbor_list),
                        int(nl_update_every),
                        bool(use_flash_forces),
                        nb_cfg,
                        use_pallas,
                        use_pallas_pme,
                        bool(jax.config.jax_enable_x64),
                        os.environ.get("PROLIX_USE_PALLAS", ""),
                        os.environ.get("PROLIX_PME_GRID_POINTS", ""),
                        getattr(_bmd, "single_padded_energy", None),
                        getattr(_bmd, "pallas_bonded_all_fused", None),
                        getattr(_be, "pallas_bonded_all_fused", None),
                        getattr(_be, "pallas_nonbonded_nl", None),
                        getattr(_pmep, "_spread_charges_pallas_single", None)
                        if _use_pmep
                        else None,
                        getattr(_pmep, "_gather_pallas_single", None)
                        if _use_pmep
                        else None,
                        nl_mod.make_neighbor_list_fn,
                        _nl_target_capacity,
                    )

                stacked_cache_key = _make_stacked_cache_key()
                cand_s = _STACKED_INFERENCE_CACHE.get(stacked_cache_key)
                if (
                    cand_s is not None
                    and len(cand_s.get("bundles", ())) == len(bundles)
                    and all(a is b for a, b in zip(cand_s["bundles"], bundles))
                ):
                    cached_stacked = cand_s

            if cached_stacked is not None:
                stacked = cached_stacked["stacked"]
                stacked_neighbor = cached_stacked["stacked_neighbor"]
                neighbor_fn = cached_stacked["neighbor_fn"]
                _batched = cached_stacked["batched_fn"]
            else:
                stacked = stack_molecular_bundles(bundles)
                stacked_neighbor = None
                neighbor_fn = None
                if use_neighbor_list:
                    stacked_neighbor, neighbor_fn = self._build_stacked_neighbor_seed(
                        bundles, integration_prefix, nb_cfg=nb_cfg
                    )

                @jax.jit
                def _batched(sb, sd, snb, n_steps_in):
                    return dispatch_n_mols(
                        plan,
                        n_systems,
                        _make_run_one(n_steps_in, host_box_vec=host_box_vec_0),
                        sb,
                        sd,
                        extra_stacked=snb,
                    )

            n_steps_arr = jnp.asarray(n_steps, dtype=jnp.int32)
            batched = _batched(stacked, seeds, stacked_neighbor, n_steps_arr)

            if can_cache_stacked and cached_stacked is None and _make_stacked_cache_key is not None:
                stacked_cache_key = _make_stacked_cache_key()
                if len(_STACKED_INFERENCE_CACHE) >= 32:
                    _STACKED_INFERENCE_CACHE.pop(next(iter(_STACKED_INFERENCE_CACHE)))
                _STACKED_INFERENCE_CACHE[stacked_cache_key] = {
                    "bundles": list(bundles),
                    "stacked": stacked,
                    "stacked_neighbor": stacked_neighbor,
                    "neighbor_fn": neighbor_fn,
                    "batched_fn": _batched,
                }

            if use_neighbor_list:
                overflow_arr = batched.observable_values.pop("_nl_did_overflow", None)
                if overflow_arr is not None and bool(jnp.any(overflow_arr)):
                    bumped_capacity = int(0.5 * stacked_neighbor.idx.shape[-1])
                    retried_neighbor = self._reallocate_stacked_neighbor_seed(
                        neighbor_fn, bundles, integration_prefix, extra_capacity=bumped_capacity
                    )
                    if cached_stacked is not None:
                        cached_stacked["stacked_neighbor"] = retried_neighbor
                    elif can_cache_stacked and stacked_cache_key in _STACKED_INFERENCE_CACHE:
                        _STACKED_INFERENCE_CACHE[stacked_cache_key]["stacked_neighbor"] = (
                            retried_neighbor
                        )
                    batched = _batched(stacked, seeds, retried_neighbor, n_steps_arr)
                    overflow_arr = batched.observable_values.pop("_nl_did_overflow", None)
                    if overflow_arr is not None and bool(jnp.any(overflow_arr)):
                        raise RuntimeError(
                            "Stacked neighbor list overflowed even after reallocating "
                            f"with +{bumped_capacity} extra capacity -- capacity formula "
                            "(compute_nl_capacity) likely needs a larger safety_factor "
                            "for this shape class."
                        )
            if not isinstance(n_steps, jax.core.Tracer):
                import dataclasses

                batched = dataclasses.replace(batched, n_steps=int(n_steps))
        else:
            stacked = stack_molecular_bundles(bundles)
            batched = dispatch_n_mols(
                plan,
                n_systems,
                _make_run_one(n_steps, host_box_vec=host_box_vec_0),
                stacked,
                seeds,
            )
        return unstack_trajectories(batched, bundles)

    @staticmethod
    def _build_stacked_neighbor_seed(
        bundles: list[Any], integration_prefix: int, nb_cfg: Any = None
    ) -> tuple[Any, Any]:
        """debt 802: one shared, vmap-safe ``NeighborList`` per bundle, pre-stacked."""
        import dataclasses

        import jax
        import jax.numpy as jnp

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
        nlc = getattr(nb_cfg, "neighbor_list", None) if nb_cfg is not None else None
        skin = float(nlc.skin) if nlc is not None else 0.5
        cap_mult = float(nlc.capacity_multiplier) if nlc is not None else 1.25
        disable_cl = bool(nlc.disable_cell_list) if nlc is not None else False
        seed_positions_0 = positions_with_prefix(ref, integration_prefix)
        n_seed_rows = int(seed_positions_0.shape[0])
        union_mask = ref.atom_mask[:n_seed_rows]
        densest_bundle = ref
        max_real = int(jnp.sum(union_mask))
        for b in bundles[1:]:
            b_mask = b.atom_mask[:n_seed_rows]
            union_mask = union_mask | b_mask
            b_real = int(jnp.sum(b_mask))
            if b_real > max_real:
                max_real = b_real
                densest_bundle = b

        densest_positions = positions_with_prefix(densest_bundle, integration_prefix)
        neighbor_fn = nl_mod.make_neighbor_list_fn(
            displacement_fn,
            box_vec,
            cutoff,
            capacity_multiplier=cap_mult,
            disable_cell_list=disable_cl,
            dr_threshold=skin,
            atom_mask=union_mask,
        )

        seed = neighbor_fn.allocate(densest_positions)
        target_capacity = _nl_target_capacity(
            nl_mod, n_seed_rows, box_vec, cutoff, skin
        )
        cap_limit = max(0, n_seed_rows - 1)
        if getattr(seed, "cell_list_capacity", None) is not None:
            cap_limit = min(cap_limit, 27 * int(seed.cell_list_capacity))
        target_capacity = min(target_capacity, cap_limit)
        extra_needed = max(0, target_capacity - int(seed.idx.shape[-1]))
        if extra_needed > 0:
            pad_cols = jnp.full(
                (n_seed_rows, extra_needed),
                n_seed_rows,
                dtype=seed.idx.dtype,
            )
            seed = dataclasses.replace(
                seed,
                idx=jnp.concatenate([seed.idx, pad_cols], axis=-1),
                max_occupancy=target_capacity,
            )
        seed_invalidated = dataclasses.replace(
            seed,
            reference_position=seed.reference_position
            + jnp.asarray(0.5, dtype=seed.reference_position.dtype)
            * jnp.asarray(box_vec, dtype=seed.reference_position.dtype),
        )

        densest_nl = _encode_bundle_nl(seed, densest_bundle)
        per_bundle = []
        for b in bundles:
            if b is densest_bundle or (
                b.positions is densest_bundle.positions
                and b.atom_mask is densest_bundle.atom_mask
                and getattr(b, "excl_dense_indices", None)
                is getattr(densest_bundle, "excl_dense_indices", None)
            ):
                per_bundle.append(densest_nl)
            else:
                b_pos = positions_with_prefix(b, integration_prefix)
                b_updated = seed_invalidated.update(b_pos)
                per_bundle.append(_encode_bundle_nl(b_updated, b))
        stacked_neighbor = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_bundle)
        return stacked_neighbor, neighbor_fn

    @staticmethod
    def _reallocate_stacked_neighbor_seed(
        neighbor_fn: Any, bundles: list[Any], integration_prefix: int, *, extra_capacity: int
    ) -> Any:
        """Host-side reallocate-and-restack after an overflow (debt 802)."""
        import dataclasses

        import jax
        import jax.numpy as jnp

        ref = bundles[0]
        box_vec = jnp.diag(ref.box)
        seed_positions = positions_with_prefix(ref, integration_prefix)
        seed = neighbor_fn.allocate(seed_positions, extra_capacity=extra_capacity)
        seed_invalidated = dataclasses.replace(
            seed,
            reference_position=seed.reference_position
            + jnp.asarray(0.5, dtype=seed.reference_position.dtype)
            * jnp.asarray(box_vec, dtype=seed.reference_position.dtype),
        )
        first_nl = _encode_bundle_nl(seed_invalidated.update(seed_positions), ref)
        per_bundle = [first_nl]
        for b in bundles[1:]:
            if b is ref or (
                b.positions is ref.positions
                and b.atom_mask is ref.atom_mask
                and getattr(b, "excl_dense_indices", None)
                is getattr(ref, "excl_dense_indices", None)
            ):
                per_bundle.append(first_nl)
            else:
                per_bundle.append(
                    _encode_bundle_nl(
                        seed_invalidated.update(positions_with_prefix(b, integration_prefix)),
                        b,
                    )
                )
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
        initial_neighbor: Any = None,
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
        excl_mode: str = "table",
        host_box_vec: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Shared settle_langevin init. Returns (state, apply_fn, dt, kT)."""
        import dataclasses

        import jax
        import jax.numpy as jnp
        import numpy as np

        bundle_for_energy = bundle
        if (
            host_box_vec is not None
            and getattr(bundle, "box", None) is not None
            and isinstance(bundle.box, jax.core.Tracer)
        ):
            hb = np.asarray(host_box_vec, dtype=np.float64)
            box_concrete = (
                np.diag(hb) if (bundle.box.ndim == 2 and hb.ndim == 1) else hb
            )
            bundle_for_energy = dataclasses.replace(bundle, box=box_concrete)

        # debt 761: force_fn_from_bundle returns an (N, 3) force array rather
        # than a scalar energy -- settle_langevin's energy_or_force_fn
        # auto-detects which via jax.eval_shape (see
        # _make_settle_compatible_force_fn/make_force_fn_like_canonicalize),
        # so no other change is needed here for the integrator itself to
        # skip the dense-energy autodiff pass.
        tile = int(getattr(nb_cfg, "flash_tile_size", 256) or 256) if nb_cfg is not None else 256
        remat = bool(getattr(nb_cfg, "flash_remat", True)) if nb_cfg is not None else True
        energy_or_force_fn = (
            force_fn_from_bundle(
                bundle_for_energy,
                flash_tile_size=tile,
                flash_remat=remat,
                use_pallas=use_pallas,
            )
            if use_flash_forces
            else energy_fn_from_bundle(
                bundle_for_energy,
                lj_switch_width=lj_switch_width_from_config(nb_cfg),
                # When a neighbor list is bound below, single_padded_energy takes
                # its `neighbor is not None` branch, which reads only the sparse
                # excl_indices/excl_scales_*. The dense (N, N) exclusion matrices
                # -- ~4.4 GB at DHFR scale, rebuilt every step -- are unreachable
                # on that path, so skipping their construction cannot change a
                # number. This is what made the NL path OOM at init (job 20528981)
                # while flash, which already opted out, did not.
                build_dense_exclusions=initial_neighbor is None,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
                excl_mode=excl_mode,
            )
        )
        if initial_neighbor is not None:
            energy_or_force_fn = _bind_neighbor(energy_or_force_fn, initial_neighbor)
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
            # #4971: this MUST come from the same predicate as the shift_fn above.
            # settle_langevin's `box` defaults to None, and every PBC protection in
            # settle.py is gated on it being non-None -- the per-atom unwrap, both
            # minimum-image corrections on the SETTLE displacement, and the
            # post-solve re-wrap. Omitting it while shift_fn wraps atoms into
            # [0, box) leaves water molecules straddling a face permanently split
            # between two images, and SETTLE then fits a rigid body to an object up
            # to a box across and launches it.
            box=pbc_box_for_bundle(bundle),
            # use_flash_forces=True means energy_or_force_fn is
            # force_fn_from_bundle's already-force-shaped output -- tell
            # settle_langevin so it skips jax_md.quantity.canonicalize_force's
            # unprotected internal eval_shape probe (crashes on PME grid-sizing
            # code touching a bundle's static box_size field; see settle.py's
            # _make_settle_compatible_force_fn docstring for the full mechanism).
            is_force_shaped=use_flash_forces,
        )

        key = jax.random.PRNGKey(jnp.asarray(seed, dtype=jnp.uint32))
        init_kwargs: dict[str, Any] = {"kT": kT_s}
        if initial_neighbor is not None:
            init_kwargs["neighbor"] = initial_neighbor
        state = init_fn(key, positions_init, **init_kwargs)
        return state, apply_fn, dt_s, kT_s

    def _build_fast_reseed_fn(
        self,
        bundle: Any,
        state0: Any,
        *,
        integration_prefix: int | None,
        gamma: float,
    ) -> Any:
        """Build a jitted re-seeding init_fn that reuses ``state0.force`` without re-evaluating forces."""
        import jax

        _displacement_fn, shift_fn = displacement_fn_for_bundle(bundle)
        gamma_akma = gamma_ps_to_akma(float(gamma))
        if integration_prefix is not None:
            positions_init = positions_with_prefix(bundle, integration_prefix)
            masses = masses_with_prefix(bundle, integration_prefix)
            if bundle.shape_spec.has_real_water:
                water_indices = bundle.water_indices
                water_mask = bundle.water_mask
            else:
                water_indices = None
                water_mask = None
            atom_mask = bundle.atom_mask[:integration_prefix]
        else:
            positions_init = active_positions(bundle)
            masses = masses_for_bundle(bundle)
            water_indices = water_indices_for_integration(bundle)
            water_mask = None
            atom_mask = None

        force0 = state0.force
        fast_init_fn, _ = settle_langevin(
            lambda _r, **_kw: force0,
            shift_fn,
            dt=1.0,
            kT=1.0,
            gamma=gamma_akma,
            mass=masses,
            water_indices=water_indices,
            project_ou_momentum_rigid=True,
            water_mask=water_mask,
            atom_mask=atom_mask,
            box=pbc_box_for_bundle(bundle),
            is_force_shaped=True,
        )

        @jax.jit
        def _reseed(seed_u32: Any, kT_val: Any) -> Any:
            k = jax.random.PRNGKey(seed_u32)
            return fast_init_fn(k, positions_init, kT=kT_val)

        return _reseed

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
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
        host_box_vec: Any = None,
    ) -> Trajectory:

        state, apply_fn, dt_s, kT_s = self._setup_integrator(
            bundle,
            dt,
            kT,
            seed,
            integration_prefix=integration_prefix,
            dt_unit=dt_unit,
            gamma=gamma,
            use_flash_forces=use_flash_forces,
            nb_cfg=nb_cfg,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
            host_box_vec=host_box_vec,
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
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
        host_box_vec: Any = None,
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
        import dataclasses
        import os

        import jax
        import jax.numpy as jnp

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

        can_cache = (
            xtc_path is None
            and initial_neighbor is None
            and not isinstance(dt, jax.core.Tracer)
            and not isinstance(kT, jax.core.Tracer)
            and not isinstance(seed, jax.core.Tracer)
            and not any(
                isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(bundle)
            )
        )
        cache_key = None
        cached_entry = None
        if can_cache:
            import sys

            def _make_cache_key() -> tuple[Any, ...]:
                _bmd = sys.modules.get("prolix.api.bundle_md")
                _be = sys.modules.get("prolix.batched_energy")
                _pmep = sys.modules.get("prolix.physics.pme_pallas")
                _use_pmep = bool(
                    use_pallas_pme
                    if use_pallas_pme is not None
                    else (
                        use_pallas
                        if use_pallas is not None
                        else os.environ.get("PROLIX_USE_PALLAS", "") == "1"
                    )
                )
                return (
                    id(bundle),
                    integration_prefix,
                    float(_host_float(dt, 1.0)),
                    dt_unit,
                    float(_host_float(kT, 0.6)),
                    float(gamma),
                    bool(use_neighbor_list),
                    int(nl_update_every),
                    bool(use_flash_forces),
                    nb_cfg,
                    use_pallas,
                    use_pallas_pme,
                    bool(jax.config.jax_enable_x64),
                    os.environ.get("PROLIX_USE_PALLAS", ""),
                    os.environ.get("PROLIX_PME_GRID_POINTS", ""),
                    getattr(_bmd, "single_padded_energy", None),
                    getattr(_bmd, "pallas_bonded_all_fused", None),
                    getattr(_be, "pallas_bonded_all_fused", None),
                    getattr(_be, "pallas_nonbonded_nl", None),
                    getattr(_pmep, "_spread_charges_pallas_single", None)
                    if _use_pmep
                    else None,
                    getattr(_pmep, "_gather_pallas_single", None)
                    if _use_pmep
                    else None,
                    nl_mod.make_neighbor_list_fn,
                    _nl_target_capacity,
                )

            cache_key = _make_cache_key()
            cand = _SINGLE_INFERENCE_CACHE.get(cache_key)
            if cand is not None and cand.get("bundle") is bundle:
                cached_entry = cand

        neighbor_fn = None
        target_capacity = 0
        cutoff = _host_float(getattr(bundle, "cutoff_distance", 9.0), 9.0)
        nlc_top = getattr(nb_cfg, "neighbor_list", None) if nb_cfg is not None else None
        skin = float(nlc_top.skin) if nlc_top is not None else 0.5
        nbr0 = initial_neighbor

        if cached_entry is not None:
            nbr0 = cached_entry["nbr0"]
            neighbor_fn = cached_entry["neighbor_fn"]
            target_capacity = cached_entry["target_capacity"]
            cutoff = cached_entry["cutoff"]
            skin = cached_entry["skin"]
            apply_fn = cached_entry["apply_fn"]
            dt_s = cached_entry["dt_s"]
            kT_s = cached_entry["kT_s"]
            state = cached_entry["fast_reseed_fn"](
                jnp.asarray(seed, dtype=jnp.uint32), kT_s
            )
        else:
            if use_neighbor_list and nbr0 is None:
                displacement_fn, _shift_fn = displacement_fn_for_bundle(bundle)
                box_vec = jnp.diag(bundle.box)
                cap_mult = float(nlc_top.capacity_multiplier) if nlc_top is not None else 1.25
                disable_cl = bool(nlc_top.disable_cell_list) if nlc_top is not None else False
                seed_positions = positions_with_prefix(bundle, integration_prefix)
                n_seed_rows = int(seed_positions.shape[0])
                excl_idx_0 = getattr(bundle, "excl_dense_indices", None)
                excl_sv_0 = getattr(bundle, "excl_dense_scales_vdw", None)
                excl_se_0 = getattr(bundle, "excl_dense_scales_elec", None)
                neighbor_fn = nl_mod.make_neighbor_list_fn(
                    displacement_fn,
                    box_vec,
                    cutoff,
                    capacity_multiplier=cap_mult,
                    disable_cell_list=disable_cl,
                    dr_threshold=skin,
                    excl_indices=excl_idx_0[:n_seed_rows] if excl_idx_0 is not None else None,
                    excl_scales_vdw=excl_sv_0[:n_seed_rows] if excl_sv_0 is not None else None,
                    excl_scales_elec=excl_se_0[:n_seed_rows] if excl_se_0 is not None else None,
                    use_pallas_nl=use_pallas,
                    atom_mask=bundle.atom_mask[:n_seed_rows],
                )
                nbr0 = neighbor_fn.allocate(seed_positions)
                target_capacity = _nl_target_capacity(
                    nl_mod, n_seed_rows, box_vec, cutoff, skin
                )
                cap_limit = max(0, n_seed_rows - 1)
                if getattr(nbr0, "cell_list_capacity", None) is not None:
                    cap_limit = min(cap_limit, 27 * int(nbr0.cell_list_capacity))
                target_capacity = min(target_capacity, cap_limit)
                extra_needed = max(0, target_capacity - int(nbr0.idx.shape[-1]))
                if extra_needed > 0:
                    pad_cols = jnp.full(
                        (n_seed_rows, extra_needed),
                        n_seed_rows,
                        dtype=nbr0.idx.dtype,
                    )
                    nbr0 = dataclasses.replace(
                        nbr0,
                        idx=jnp.concatenate([nbr0.idx, pad_cols], axis=-1),
                        max_occupancy=target_capacity,
                    )
                nbr0 = _encode_bundle_nl(nbr0, bundle)

            eff_excl_mode = (
                "pre_encoded"
                if (nbr0 is not None and getattr(bundle, "excl_dense_indices", None) is not None)
                else "table"
            )

            state, apply_fn, dt_s, kT_s = self._setup_integrator(
                bundle,
                dt,
                kT,
                seed,
                integration_prefix=integration_prefix,
                dt_unit=dt_unit,
                gamma=gamma,
                use_flash_forces=use_flash_forces,
                initial_neighbor=nbr0,
                nb_cfg=nb_cfg,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
                excl_mode=eff_excl_mode,
                host_box_vec=host_box_vec,
            )

        sink_cm = None
        on_step = None
        if xtc_path is not None:
            import numpy as np

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
        nl_retry_count = 0
        nl_capacity_for_caller = (
            int(nbr0.idx.shape[-1]) if nbr0 is not None and use_neighbor_list else None
        )
        if use_neighbor_list:
            if cached_entry is not None and cached_entry.get("nl_step_fn") is not None:
                _nl_step_fn = cached_entry["nl_step_fn"]
            else:
                import equinox as eqx

                ghost_target_positions = _integrator_positions(state)
                n_nl_rows = int(ghost_target_positions.shape[0])
                pad_mask_1d = bundle.atom_mask[:n_nl_rows]
                pad_mask_3d = pad_mask_1d[:, None]
                _nl_disp_fn, _ = displacement_fn_for_bundle(bundle)
                nlc = getattr(nb_cfg, "neighbor_list", None) if nb_cfg is not None else None
                skin = float(nlc.skin) if nlc is not None else 0.5
                stride = int(nl_update_every)
                if nlc is not None and nlc.nl_update_every is None:
                    stride = 0
                thresh_sq = (skin * 0.5) ** 2
                box_vec_nl = jnp.diag(bundle.box) if bundle.box.ndim == 2 else bundle.box
                r_total_cutoff = float(_host_float(bundle.cutoff_distance, 9.0)) + skin
                excl_idx_nl = getattr(bundle, "excl_dense_indices", None)
                excl_sv_nl = getattr(bundle, "excl_dense_scales_vdw", None)
                excl_se_nl = getattr(bundle, "excl_dense_scales_elec", None)
                has_excl_nl = (
                    excl_idx_nl is not None
                    and excl_sv_nl is not None
                    and excl_se_nl is not None
                )
                if has_excl_nl:
                    excl_idx_nl = excl_idx_nl[:n_nl_rows]
                    excl_sv_nl = excl_sv_nl[:n_nl_rows]
                    excl_se_nl = excl_se_nl[:n_nl_rows]
                else:
                    excl_idx_nl = jnp.full((n_nl_rows, 1), -1, dtype=jnp.int32)
                    excl_sv_nl = jnp.ones((n_nl_rows, 1), dtype=jnp.float32)
                    excl_se_nl = jnp.ones((n_nl_rows, 1), dtype=jnp.float32)

                cell_cap_i = (
                    int(nbr0.cell_list_capacity)
                    if getattr(nbr0, "cell_list_capacity", None) is not None
                    else 64
                )
                cell_size_f = (
                    float(nbr0.cell_size)
                    if getattr(nbr0, "cell_size", None) is not None
                    else r_total_cutoff
                )
                can_pallas_nl = (
                    getattr(nbr0, "cell_list_fn", None) is not None
                    and getattr(nbr0, "cell_list_capacity", None) is not None
                    and nl_mod._should_use_pallas_nl_default(use_pallas)
                )
                import numpy as np

                if host_box_vec is not None:
                    box_host_nl = host_box_vec
                elif not isinstance(box_vec_nl, jax.core.Tracer):
                    box_host_nl = box_vec_nl
                else:
                    b0_box_np = np.asarray(self.bundles[0].box, dtype=np.float64)
                    box_host_nl = np.diag(b0_box_np) if b0_box_np.ndim == 2 else b0_box_np
                grid_dims_nl = (
                    nl_mod._compute_static_grid_dims(box_host_nl, cell_size_f)
                    if can_pallas_nl
                    else None
                )

                def _nl_step_fn(carry: _NLDispatchCarry, step_i: Any) -> _NLDispatchCarry:
                    langevin_state, neighbor, did_overflow, last_pos = carry
                    pos = _integrator_positions(langevin_state)
                    d_pos = jnp.where(pad_mask_3d, jax.vmap(_nl_disp_fn)(pos, last_pos), 0.0)
                    max_dr_sq = jnp.max(jnp.sum(d_pos * d_pos, axis=-1))
                    verlet = max_dr_sq > thresh_sq
                    if stride > 0:
                        by_stride = (step_i > 0) & ((step_i % stride) == 0)
                        should_update = verlet | by_stride
                    else:
                        should_update = verlet

                    max_occ_i = int(neighbor.idx.shape[-1])

                    @jax.custom_batching.custom_vmap
                    def _maybe_rebuild_nl(
                        su_in,
                        pos_in,
                        lp_in,
                        idx_in,
                        ref_in,
                        err_in,
                        bx_in,
                        ei_in,
                        ev_in,
                        ee_in,
                        am_in,
                    ):
                        def _do_one(operands):
                            p_o, _lp_o, i_o, r_o, e_o, bx_o, ei_o, ev_o, ee_o, am_o = operands
                            if can_pallas_nl:
                                p_w = jnp.mod(p_o, bx_o)
                                idx_new, occ_new = nl_mod.pallas_prune_and_encode_cell_list(
                                    p_w,
                                    bx_o,
                                    ei_o,
                                    ev_o,
                                    ee_o,
                                    cell_size=cell_size_f,
                                    cell_capacity=cell_cap_i,
                                    max_occupancy=max_occ_i,
                                    r_cutoff=r_total_cutoff,
                                    encode_exclusions=has_excl_nl,
                                    interpret=(jax.default_backend() == "cpu"),
                                    grid_dims=grid_dims_nl,
                                    atom_mask=am_o,
                                )
                                overflow_bit = jnp.uint8(
                                    nl_mod.partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW
                                )
                                err_new = jnp.where(occ_new > max_occ_i, e_o | overflow_bit, e_o)
                                return idx_new, p_w.astype(r_o.dtype), err_new, p_o
                            forced_ref = r_o + jnp.asarray(0.5, dtype=r_o.dtype) * bx_o.astype(r_o.dtype)
                            n_tmp = dataclasses.replace(
                                neighbor,
                                idx=i_o,
                                reference_position=forced_ref,
                                error=nl_mod.partition.PartitionError(e_o),
                            )
                            n_upd = n_tmp.update(p_o)
                            upd_fn = getattr(n_upd, "update_fn", None)
                            pre_enc = bool(getattr(upd_fn, "_pre_encoded_exclusions", False))
                            ghosts_filt = bool(getattr(upd_fn, "_ghosts_filtered", False))
                            if (has_excl_nl and not pre_enc) or not ghosts_filt:
                                idx_enc = nl_mod.encode_nl_exclusions(
                                    n_upd.idx,
                                    ei_o if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.int32),
                                    ev_o if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.float32),
                                    ee_o if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.float32),
                                    n_atoms=n_nl_rows,
                                    atom_mask=am_o,
                                )
                            else:
                                idx_enc = n_upd.idx
                            return idx_enc, n_upd.reference_position, n_upd.error.code, p_o

                        def _keep_one(operands):
                            _p_o, lp_o, i_o, r_o, e_o, _bx_o, _ei_o, _ev_o, _ee_o, _am_o = operands
                            return i_o, r_o, e_o, lp_o

                        return jax.lax.cond(
                            su_in,
                            _do_one,
                            _keep_one,
                            (pos_in, lp_in, idx_in, ref_in, err_in, bx_in, ei_in, ev_in, ee_in, am_in),
                        )

                    @_maybe_rebuild_nl.def_vmap
                    def _maybe_rebuild_nl_vmap(
                        axis_size,
                        in_batched,
                        su_in,
                        pos_in,
                        lp_in,
                        idx_in,
                        ref_in,
                        err_in,
                        bx_in,
                        ei_in,
                        ev_in,
                        ee_in,
                        am_in,
                    ):
                        B = int(axis_size)
                        su_b = su_in if in_batched[0] else jnp.broadcast_to(su_in[None, ...], (B,))
                        pos_b = pos_in if in_batched[1] else jnp.broadcast_to(pos_in[None, ...], (B, *pos_in.shape))
                        lp_b = lp_in if in_batched[2] else jnp.broadcast_to(lp_in[None, ...], (B, *lp_in.shape))
                        idx_b = idx_in if in_batched[3] else jnp.broadcast_to(idx_in[None, ...], (B, *idx_in.shape))
                        ref_b = ref_in if in_batched[4] else jnp.broadcast_to(ref_in[None, ...], (B, *ref_in.shape))
                        err_b = err_in if in_batched[5] else jnp.broadcast_to(err_in[None, ...], (B, *err_in.shape))
                        bx_b = bx_in if in_batched[6] else jnp.broadcast_to(bx_in[None, ...], (B, *bx_in.shape))
                        ei_b = ei_in if in_batched[7] else ei_in[None, ...]
                        ev_b = ev_in if in_batched[8] else ev_in[None, ...]
                        ee_b = ee_in if in_batched[9] else ee_in[None, ...]
                        am_b = am_in if in_batched[10] else am_in[None, ...]

                        any_upd = jnp.any(su_b)

                        def _do_batch(operands):
                            p_b, _lp_b, i_b, r_b, e_b, b_b, ei_op, ev_op, ee_op, am_op = operands
                            if can_pallas_nl:
                                p_w_b = jnp.mod(p_b, b_b[:, None, :])
                                idx_new_b, occ_new_b = nl_mod._pallas_prune_and_encode_cell_list_batch(
                                    B,
                                    (True, True, in_batched[7], in_batched[8], in_batched[9]),
                                    p_w_b,
                                    b_b,
                                    ei_op if in_batched[7] else ei_in,
                                    ev_op if in_batched[8] else ev_in,
                                    ee_op if in_batched[9] else ee_in,
                                    cell_size=cell_size_f,
                                    cell_capacity=cell_cap_i,
                                    max_occupancy=max_occ_i,
                                    r_cutoff=r_total_cutoff,
                                    encode_exclusions=has_excl_nl,
                                    interpret=(jax.default_backend() == "cpu"),
                                    grid_dims=grid_dims_nl,
                                    atom_mask=am_op if in_batched[10] else am_in,
                                    mask_batched=bool(in_batched[10]),
                                )
                                overflow_bit = jnp.uint8(
                                    nl_mod.partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW
                                )
                                err_new_b = jnp.where(occ_new_b > max_occ_i, e_b | overflow_bit, e_b)
                                return idx_new_b, p_w_b.astype(r_b.dtype), err_new_b, p_b

                            ei_full = jnp.broadcast_to(ei_op, (B, *ei_op.shape[1:]))
                            ev_full = jnp.broadcast_to(ev_op, (B, *ev_op.shape[1:]))
                            ee_full = jnp.broadcast_to(ee_op, (B, *ee_op.shape[1:]))
                            am_full = jnp.broadcast_to(am_op, (B, *am_op.shape[1:]))

                            def _one_fallback(p_1, r_1, i_1, e_1, bx_1, ei_1, ev_1, ee_1, am_1):
                                forced_ref = r_1 + jnp.asarray(0.5, dtype=r_1.dtype) * bx_1.astype(r_1.dtype)
                                n_tmp = dataclasses.replace(
                                    neighbor,
                                    idx=i_1,
                                    reference_position=forced_ref,
                                    error=nl_mod.partition.PartitionError(e_1),
                                )
                                n_upd = n_tmp.update(p_1)
                                upd_fn = getattr(n_upd, "update_fn", None)
                                pre_enc = bool(getattr(upd_fn, "_pre_encoded_exclusions", False))
                                ghosts_filt = bool(getattr(upd_fn, "_ghosts_filtered", False))
                                if (has_excl_nl and not pre_enc) or not ghosts_filt:
                                    idx_enc = nl_mod.encode_nl_exclusions(
                                        n_upd.idx,
                                        ei_1 if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.int32),
                                        ev_1 if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.float32),
                                        ee_1 if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.float32),
                                        n_atoms=n_nl_rows,
                                        atom_mask=am_1,
                                    )
                                else:
                                    idx_enc = n_upd.idx
                                return idx_enc, n_upd.reference_position, n_upd.error.code, p_1

                            return jax.vmap(_one_fallback)(
                                p_b, r_b, i_b, e_b, b_b, ei_full, ev_full, ee_full, am_full
                            )

                        def _keep_batch(operands):
                            _p_b, l_b, i_b, r_b, e_b, _b_b, _ei_op, _ev_op, _ee_op, _am_op = operands
                            return i_b, r_b, e_b, l_b

                        out = jax.lax.cond(
                            any_upd,
                            _do_batch,
                            _keep_batch,
                            (pos_b, lp_b, idx_b, ref_b, err_b, bx_b, ei_b, ev_b, ee_b, am_b),
                        )
                        return out, (True, True, True, True)

                    new_idx, new_ref, new_err_code, new_last = _maybe_rebuild_nl(
                        should_update,
                        pos,
                        last_pos,
                        neighbor.idx,
                        neighbor.reference_position,
                        neighbor.error.code,
                        box_vec_nl,
                        excl_idx_nl,
                        excl_sv_nl,
                        excl_se_nl,
                        pad_mask_1d,
                    )
                    new_neighbor = dataclasses.replace(
                        neighbor,
                        idx=new_idx,
                        reference_position=new_ref,
                        error=nl_mod.partition.PartitionError(new_err_code),
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
                    new_R = jnp.where(
                        pad_mask_3d,
                        _integrator_positions(new_langevin_state),
                        pos,
                    )
                    new_P = jnp.where(pad_mask_3d, new_langevin_state.momentum, 0.0)
                    if hasattr(new_langevin_state, "positions"):
                        new_langevin_state = eqx.tree_at(
                            lambda s: (s.positions, s.momentum),
                            new_langevin_state,
                            (new_R, new_P),
                        )
                    else:
                        new_langevin_state = new_langevin_state.set(
                            position=new_R, momentum=new_P
                        )
                    return _NLDispatchCarry(
                        new_langevin_state, new_neighbor, did_overflow, new_last
                    )

            if can_cache and cached_entry is None and cache_key is not None:
                fast_reseed_fn = self._build_fast_reseed_fn(
                    bundle, state, integration_prefix=integration_prefix, gamma=gamma
                )
                # Warm up fast_reseed_fn during the initial call and use its output
                # so Call 1 and subsequent cached calls pass identical state leaf
                # metadata (dtype/weak_type/sharding) into jitted_while.
                state = fast_reseed_fn(jnp.asarray(seed, dtype=jnp.uint32), kT_s)
                if len(_SINGLE_INFERENCE_CACHE) >= 8:
                    _SINGLE_INFERENCE_CACHE.pop(next(iter(_SINGLE_INFERENCE_CACHE)))
                cached_entry = {
                    "bundle": bundle,
                    "nbr0": nbr0,
                    "neighbor_fn": neighbor_fn,
                    "target_capacity": target_capacity,
                    "cutoff": cutoff,
                    "skin": skin,
                    "apply_fn": apply_fn,
                    "dt_s": dt_s,
                    "kT_s": kT_s,
                    "fast_reseed_fn": fast_reseed_fn,
                    "nl_step_fn": _nl_step_fn,
                    "step_fn": None,
                }
                cache_key = _make_cache_key()
                _SINGLE_INFERENCE_CACHE[cache_key] = cached_entry

            init_pos = _integrator_positions(state)
            init_carry = _NLDispatchCarry(state, nbr0, jnp.array(False), init_pos)
            final_carry = dispatch_n_steps_inference(
                _nl_step_fn, init_carry, n_steps
            )
            if initial_neighbor is not None:
                # debt 802 (stacked/vmapped path): overflow retry is host-side
                # in _run_stacked_dispatch -- neighbor_fn.allocate() cannot run
                # on a tracer.
                nl_overflow_for_caller = final_carry.did_overflow
            elif neighbor_fn is not None and bool(final_carry.did_overflow):
                # Size every retry from `state` -- the configuration the scan
                # actually restarts from. The old code sized from the overflowed
                # trajectory's *final* positions, which were produced by a run with
                # a known-broken neighbor list, and then restarted from `state`
                # anyway: capacity fitted to coordinates it would never see.
                #
                # Grow geometrically over a bounded number of attempts rather than
                # making one fixed +50% bump and giving up. At DHFR scale that
                # single bump added 271 to an already-undersized 542 and still
                # overflowed (job 21835473).
                retry_positions = _integrator_positions(state)
                base_capacity = int(nbr0.idx.shape[-1])
                retried_neighbor = nbr0
                for attempt in range(1, _NL_MAX_REALLOC_ATTEMPTS + 1):
                    # total capacity ~= base * 2**attempt
                    bumped_capacity = base_capacity * (2**attempt - 1)
                    retried_neighbor = _encode_bundle_nl(
                        neighbor_fn.allocate(
                            retry_positions, extra_capacity=bumped_capacity
                        ),
                        bundle,
                    )
                    retry_carry = _NLDispatchCarry(
                        state, retried_neighbor, jnp.array(False), retry_positions
                    )
                    final_carry = dispatch_n_steps_inference(
                        _nl_step_fn, retry_carry, n_steps
                    )
                    nl_retry_count = attempt
                    if not bool(final_carry.did_overflow):
                        break
                else:
                    raise RuntimeError(
                        "Neighbor list overflowed on the single-bundle inference "
                        f"path after {_NL_MAX_REALLOC_ATTEMPTS} reallocation "
                        f"attempts (capacity {base_capacity} -> "
                        f"{int(retried_neighbor.idx.shape[-1])}). "
                        f"compute_nl_capacity recommended {target_capacity} for "
                        f"this system ({int(retry_positions.shape[0])} atoms, "
                        f"cutoff={cutoff}, skin={skin}). If that recommendation is "
                        "itself too low, raise safety_factor in "
                        "prolix/physics/neighbor_list.py::compute_nl_capacity; if "
                        "it looks right but the allocated capacity is well below "
                        "it, the capacity pre-planning block in "
                        "_run_single_inference did not run."
                    )
                nl_capacity_for_caller = int(retried_neighbor.idx.shape[-1])
                if cached_entry is not None:
                    cached_entry["nbr0"] = retried_neighbor
            state = final_carry.langevin_state
        else:
            if cached_entry is not None and cached_entry.get("step_fn") is not None:
                step_fn = cached_entry["step_fn"]
            else:

                def step_fn(carry: Any, _step_i: Any) -> Any:
                    return apply_fn(carry, kT=kT_s, dt=dt_s)

                if can_cache and cached_entry is None and cache_key is not None:
                    fast_reseed_fn = self._build_fast_reseed_fn(
                        bundle, state, integration_prefix=integration_prefix, gamma=gamma
                    )
                    state = fast_reseed_fn(jnp.asarray(seed, dtype=jnp.uint32), kT_s)
                    if len(_SINGLE_INFERENCE_CACHE) >= 8:
                        _SINGLE_INFERENCE_CACHE.pop(next(iter(_SINGLE_INFERENCE_CACHE)))
                    cached_entry = {
                        "bundle": bundle,
                        "nbr0": None,
                        "neighbor_fn": None,
                        "target_capacity": 0,
                        "cutoff": cutoff,
                        "skin": skin,
                        "apply_fn": apply_fn,
                        "dt_s": dt_s,
                        "kT_s": kT_s,
                        "fast_reseed_fn": fast_reseed_fn,
                        "nl_step_fn": None,
                        "step_fn": step_fn,
                    }
                    cache_key = _make_cache_key()
                    _SINGLE_INFERENCE_CACHE[cache_key] = cached_entry

            try:
                state = dispatch_n_steps_inference(
                    step_fn, state, n_steps, on_step=on_step
                )
            finally:
                if sink_cm is not None:
                    sink_cm.__exit__(None, None, None)

        # Final frame only: (1, n_atoms, 3)
        final_pos = state.position
        if trim_output:
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
        if nl_capacity_for_caller is not None:
            # Capacity actually in force, and how many host-side reallocations it
            # took to get there. A benchmark that times plan.run() must treat a
            # non-zero retry count as a failed measurement: the overflow is only
            # detected after the full scan completes, so the timed region contains
            # a discarded scan plus the retry (roughly 2x the true per-step cost).
            observable_values["_nl_capacity"] = nl_capacity_for_caller
            observable_values["_nl_retry_count"] = nl_retry_count

        return Trajectory(
            positions=positions_array,
            observable_values=observable_values,
            n_steps=0 if isinstance(n_steps, jax.core.Tracer) else int(n_steps),
        )


def _group_indices_by_shape_spec(bundles: list[Any]) -> list[list[int]]:
    """Deprecated alias — use ``partition_bundles_by_shape``."""

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
