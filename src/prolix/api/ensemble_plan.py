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
    _JITTED_WHILE_CACHE,
    dispatch_n_mols,
    dispatch_n_steps,
    dispatch_n_steps_inference,
)
from prolix.api.ensemble_planner import EnsembleMDPlanner
from prolix.api.observables import Energy, KineticEnergy, Pressure, Temperature, Trajectory
from prolix.api.xtc_sink import XtcFrameSink, write_positions_xtc
from prolix.pallas_kernels import _infer_first_pure_water_block
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

#: Cache for compiled (_eval_force0_jit, _batched) functions keyed by bucket shape_spec + leaf shapes/dtypes + runtime knobs.
_STACKED_JIT_FN_CACHE: dict[Any, tuple[Any, Any]] = {}

#: Cache for canonical (neighbor_fn, seed_template) keyed by bucket size + box + cutoff/skin so stacked NeighborList shares PyTreeDef across batches.
_STACKED_NL_SEED_CACHE: dict[Any, tuple[Any, Any]] = {}


import contextlib as _contextlib


@_contextlib.contextmanager
def _scoped_active_atom_env(bundles: list[Any], n_padded: int | None):
    """Auto-clamp PROLIX_ACTIVE_ATOMS / PROLIX_ACTIVE_WATERS from bundle.n_atoms or concrete atom_mask."""
    import os

    import jax
    import numpy as np

    saved_a = os.environ.get("PROLIX_ACTIVE_ATOMS")
    saved_w = os.environ.get("PROLIX_ACTIVE_WATERS")
    did_set_a = False
    did_set_w = False
    try:
        if (
            n_padded is not None
            and int(n_padded) >= 64
            and not os.environ.get("PROLIX_ACTIVE_ATOMS", "").strip()
            and bundles
            and not any(isinstance(getattr(b, "n_atoms", None), jax.core.Tracer) for b in bundles)
        ):
            n_pad_i = int(n_padded)
            real_counts = []
            for b in bundles:
                na = int(b.n_atoms if isinstance(b.n_atoms, int) else int(np.asarray(b.n_atoms)))
                am = getattr(b, "atom_mask", None)
                if na >= n_pad_i and am is not None and not isinstance(am, jax.core.Tracer):
                    am_np = np.asarray(am, dtype=bool)
                    nz = np.nonzero(am_np)[0]
                    if nz.size > 0:
                        na = min(na, int(nz[-1]) + 1)
                real_counts.append(na)
            max_real_a = max(real_counts)
            is_implicit = getattr(bundles[0], "box", None) is None
            if (is_implicit and 0 < max_real_a < n_pad_i) or (
                n_pad_i >= 4096 and 0 < max_real_a <= int(0.75 * n_pad_i)
            ):
                gran = 32 if is_implicit else 1024
                clamped_a = min(n_pad_i, max(gran, ((max_real_a + gran - 1) // gran) * gran))
                if clamped_a < n_pad_i:
                    os.environ["PROLIX_ACTIVE_ATOMS"] = str(clamped_a)
                    did_set_a = True
                if not is_implicit and not os.environ.get("PROLIX_ACTIVE_WATERS", "").strip():
                    w_counts = []
                    w_pads = []
                    for b in bundles:
                        wm = getattr(b, "water_mask", None)
                        wi = getattr(b, "water_indices", None)
                        if wm is not None and wi is not None and not isinstance(wm, jax.core.Tracer):
                            w_counts.append(int(np.sum(np.asarray(wm, dtype=np.int32))))
                            w_pads.append(int(wi.shape[0]))
                    if w_counts and w_pads:
                        max_real_w = max(w_counts)
                        max_pad_w = max(w_pads)
                        if 0 < max_real_w < max_pad_w:
                            clamped_w = min(max_pad_w, max(512, ((max_real_w + 511) // 512) * 512))
                            os.environ["PROLIX_ACTIVE_WATERS"] = str(clamped_w)
                            did_set_w = True
        yield
    finally:
        if did_set_a:
            if saved_a is None:
                os.environ.pop("PROLIX_ACTIVE_ATOMS", None)
            else:
                os.environ["PROLIX_ACTIVE_ATOMS"] = saved_a
        if did_set_w:
            if saved_w is None:
                os.environ.pop("PROLIX_ACTIVE_WATERS", None)
            else:
                os.environ["PROLIX_ACTIVE_WATERS"] = saved_w


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
        if getattr(nbr, "max_k_tiles", None) is None:
            return dataclasses.replace(nbr, max_k_tiles=nl_mod._max_k_tiles_from_idx(nbr.idx))
        return nbr
    excl_idx = getattr(bundle, "excl_dense_indices", None)
    excl_sv = getattr(bundle, "excl_dense_scales_vdw", None)
    excl_se = getattr(bundle, "excl_dense_scales_elec", None)
    if pre_enc or excl_idx is None or excl_sv is None or excl_se is None:
        if mask_slice is None or ghosts_filt:
            if getattr(nbr, "max_k_tiles", None) is None:
                return dataclasses.replace(nbr, max_k_tiles=nl_mod._max_k_tiles_from_idx(nbr.idx))
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
    return dataclasses.replace(
        nbr,
        idx=encoded_idx,
        max_k_tiles=nl_mod._max_k_tiles_from_idx(encoded_idx),
    )


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
            with _scoped_active_atom_env(self.bundles, int(self.bundles[0].positions.shape[0])):
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

            with _scoped_active_atom_env(group, int(group[0].positions.shape[0])):
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
                        os.environ.get("PROLIX_ACTIVE_ATOMS", ""),
                        os.environ.get("PROLIX_ACTIVE_WATERS", ""),
                        os.environ.get("PROLIX_PME_EVERY", ""),
                        os.environ.get("PROLIX_NL_CHECK_EVERY", ""),
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

            n_reals_cached = None
            if cached_stacked is not None:
                stacked = cached_stacked["stacked"]
                stacked_neighbor = cached_stacked["stacked_neighbor"]
                stacked_force0 = cached_stacked.get("stacked_force0")
                neighbor_fn = cached_stacked["neighbor_fn"]
                _batched = cached_stacked["batched_fn"]
                n_reals_cached = cached_stacked.get("n_reals")
            else:
                stacked = stack_molecular_bundles(bundles)
                n_reals_cached = [
                    b.n_atoms if isinstance(b.n_atoms, int) else int(jnp.asarray(b.n_atoms))
                    for b in bundles
                ]
                stacked_neighbor = None
                neighbor_fn = None
                if use_neighbor_list:
                    stacked_neighbor, neighbor_fn = self._build_stacked_neighbor_seed(
                        bundles,
                        integration_prefix,
                        nb_cfg=nb_cfg,
                        use_pallas=use_pallas,
                    )

                _eval_force0_jit = None
                _batched = None
                jit_fn_key = None
                if can_cache_stacked and _make_stacked_cache_key is not None:
                    jit_fn_key = (
                        n_systems,
                        jax.tree_util.tree_structure(stacked),
                        tuple(
                            (x.shape, x.dtype)
                            for x in jax.tree_util.tree_leaves(stacked)
                            if hasattr(x, "shape")
                        ),
                        jax.tree_util.tree_structure(stacked_neighbor)
                        if stacked_neighbor is not None
                        else None,
                        tuple(
                            (x.shape, x.dtype)
                            for x in jax.tree_util.tree_leaves(stacked_neighbor)
                            if hasattr(x, "shape")
                        )
                        if stacked_neighbor is not None
                        else None,
                        tuple(np.asarray(host_box_vec_0).reshape(-1).tolist())
                        if host_box_vec_0 is not None
                        else None,
                        float(_host_float(getattr(bundles[0], "cutoff_distance", 9.0), 9.0)),
                        float(_host_float(getattr(bundles[0], "pme_alpha", 0.34), 0.34)),
                        _make_stacked_cache_key()[1:],
                    )
                    cached_jit_pair = _STACKED_JIT_FN_CACHE.get(jit_fn_key)
                    if cached_jit_pair is not None:
                        _eval_force0_jit, _batched = cached_jit_pair

                if _batched is None:
                    if can_cache_stacked and use_neighbor_list:
                        def _eval_force0_one(bundle_i, _seed_i, nbr_i=None):
                            eff_excl = (
                                "pre_encoded"
                                if (
                                    nbr_i is not None
                                    and getattr(bundle_i, "excl_dense_indices", None) is not None
                                )
                                else "table"
                            )
                            st0, _, _, _ = self._setup_integrator(
                                bundle_i,
                                dt,
                                kT,
                                0,
                                integration_prefix=integration_prefix,
                                dt_unit=dt_unit,
                                gamma=gamma,
                                use_flash_forces=use_flash_forces,
                                initial_neighbor=nbr_i,
                                nb_cfg=nb_cfg,
                                use_pallas=use_pallas,
                                use_pallas_pme=use_pallas_pme,
                                excl_mode=eff_excl,
                                host_box_vec=host_box_vec_0,
                            )
                            return st0.force

                        _eval_force0_jit = jax.jit(
                            lambda sb_in, snb_in: dispatch_n_mols(
                                plan,
                                n_systems,
                                _eval_force0_one,
                                sb_in,
                                jnp.zeros(n_systems, dtype=jnp.int32),
                                extra_stacked=snb_in,
                            )
                        )

                    @jax.jit
                    def _batched(sb, sd, snb, sf0, n_steps_in):
                        def _run_one_with_f0(bundle, seed_i, extra=None):
                            if sf0 is not None and extra is not None:
                                nbr_arg, f0_arg = extra
                            else:
                                nbr_arg, f0_arg = extra, None
                            obs = observables
                            if observables is not None:
                                obs = _observables_for_bundle(
                                    bundle, observables, atom_mask=bundle.atom_mask
                                )
                            return self._run_single_inference(
                                bundle,
                                n_steps_in,
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
                                initial_neighbor=nbr_arg,
                                use_flash_forces=use_flash_forces,
                                nb_cfg=nb_cfg,
                                use_pallas=use_pallas,
                                use_pallas_pme=use_pallas_pme,
                                host_box_vec=host_box_vec_0,
                                precomputed_force0=f0_arg,
                            )

                        extra_arg = (
                            (snb, sf0)
                            if (sf0 is not None and snb is not None)
                            else snb
                        )
                        return dispatch_n_mols(
                            plan,
                            n_systems,
                            _run_one_with_f0,
                            sb,
                            sd,
                            extra_stacked=extra_arg,
                        )

                    if jit_fn_key is not None:
                        if len(_STACKED_JIT_FN_CACHE) >= 32:
                            _STACKED_JIT_FN_CACHE.pop(next(iter(_STACKED_JIT_FN_CACHE)))
                        _STACKED_JIT_FN_CACHE[jit_fn_key] = (_eval_force0_jit, _batched)

                stacked_force0 = (
                    _eval_force0_jit(stacked, stacked_neighbor)
                    if (_eval_force0_jit is not None and use_neighbor_list)
                    else None
                )

            n_steps_arr = jnp.asarray(n_steps, dtype=jnp.int32)
            batched = _batched(stacked, seeds, stacked_neighbor, stacked_force0, n_steps_arr)

            if can_cache_stacked and cached_stacked is None and _make_stacked_cache_key is not None:
                stacked_cache_key = _make_stacked_cache_key()
                if len(_STACKED_INFERENCE_CACHE) >= 32:
                    _STACKED_INFERENCE_CACHE.pop(next(iter(_STACKED_INFERENCE_CACHE)))
                _STACKED_INFERENCE_CACHE[stacked_cache_key] = {
                    "bundles": list(bundles),
                    "stacked": stacked,
                    "stacked_neighbor": stacked_neighbor,
                    "stacked_force0": stacked_force0,
                    "neighbor_fn": neighbor_fn,
                    "batched_fn": _batched,
                    "n_reals": n_reals_cached,
                    "verified_no_overflow": False,
                }

            if use_neighbor_list:
                overflow_arr = batched.observable_values.pop("_nl_did_overflow", None)
                already_verified = (
                    cached_stacked is not None
                    and bool(cached_stacked.get("verified_no_overflow", False))
                )
                if not already_verified and overflow_arr is not None:
                    if bool(jnp.any(overflow_arr)):
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
                        batched = _batched(
                            stacked, seeds, retried_neighbor, stacked_force0, n_steps_arr
                        )
                        overflow_arr = batched.observable_values.pop("_nl_did_overflow", None)
                        if overflow_arr is not None and bool(jnp.any(overflow_arr)):
                            raise RuntimeError(
                                "Stacked neighbor list overflowed even after reallocating "
                                f"with +{bumped_capacity} extra capacity -- capacity formula "
                                "(compute_nl_capacity) likely needs a larger safety_factor "
                                "for this shape class."
                            )
                    if cached_stacked is not None:
                        cached_stacked["verified_no_overflow"] = True
                    elif can_cache_stacked and stacked_cache_key in _STACKED_INFERENCE_CACHE:
                        _STACKED_INFERENCE_CACHE[stacked_cache_key]["verified_no_overflow"] = True
            if not isinstance(n_steps, jax.core.Tracer):
                import dataclasses

                batched = dataclasses.replace(batched, n_steps=int(n_steps))
        else:
            n_reals_cached = None
            stacked = stack_molecular_bundles(bundles)
            batched = dispatch_n_mols(
                plan,
                n_systems,
                _make_run_one(n_steps, host_box_vec=host_box_vec_0),
                stacked,
                seeds,
            )
        return unstack_trajectories(batched, bundles, n_reals=n_reals_cached)

    @staticmethod
    def _build_stacked_neighbor_seed(
        bundles: list[Any],
        integration_prefix: int,
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
    ) -> tuple[Any, Any]:
        """debt 802: one shared, vmap-safe ``NeighborList`` per bundle, pre-stacked."""
        import dataclasses

        import jax
        import jax.numpy as jnp
        import numpy as np

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

        seed_cache_key = (
            n_seed_rows,
            seed_positions_0.dtype,
            tuple(np.asarray(box_vec, dtype=np.float64).reshape(-1).tolist()),
            float(cutoff),
            float(skin),
            float(cap_mult),
            bool(disable_cl),
            nl_mod.make_neighbor_list_fn,
            _nl_target_capacity,
        )
        cached_seed_pair = _STACKED_NL_SEED_CACHE.get(seed_cache_key)
        if cached_seed_pair is not None and nl_mod._should_use_pallas_nl_default(use_pallas):
            neighbor_fn, seed_template = cached_seed_pair
            seed = seed_template
        else:
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
            seed_template = _encode_bundle_nl(seed, densest_bundle)
            if (
                getattr(seed_template, "cell_list_fn", None) is not None
                and getattr(seed_template, "cell_list_capacity", None) is not None
                and nl_mod._should_use_pallas_nl_default(use_pallas)
            ):
                if len(_STACKED_NL_SEED_CACHE) >= 32:
                    _STACKED_NL_SEED_CACHE.pop(next(iter(_STACKED_NL_SEED_CACHE)))
                _STACKED_NL_SEED_CACHE[seed_cache_key] = (neighbor_fn, seed_template)
                seed = seed_template

        all_same_nl = all(
            b is densest_bundle
            or (
                b.positions is densest_bundle.positions
                and b.atom_mask is densest_bundle.atom_mask
                and getattr(b, "excl_dense_indices", None)
                is getattr(densest_bundle, "excl_dense_indices", None)
            )
            for b in bundles
        )
        can_pallas_batch_seed = (
            len(bundles) > 1
            and getattr(seed, "cell_list_fn", None) is not None
            and getattr(seed, "cell_list_capacity", None) is not None
            and nl_mod._should_use_pallas_nl_default(use_pallas)
        )
        if can_pallas_batch_seed:
            B_total = len(bundles)
            B_eval = 1 if all_same_nl else B_total
            eval_bundles = [densest_bundle] if all_same_nl else bundles
            pos_b_seed = jnp.stack(
                [positions_with_prefix(b, integration_prefix) for b in eval_bundles], axis=0
            )
            box_b_seed = jnp.broadcast_to(box_vec[None, :], (B_eval, 3))
            has_excl_seed = all(
                getattr(b, "excl_dense_indices", None) is not None
                and getattr(b, "excl_dense_scales_vdw", None) is not None
                and getattr(b, "excl_dense_scales_elec", None) is not None
                for b in eval_bundles
            )
            if has_excl_seed:
                ei_b_seed = jnp.stack([b.excl_dense_indices[:n_seed_rows] for b in eval_bundles], axis=0)
                ev_b_seed = jnp.stack(
                    [b.excl_dense_scales_vdw[:n_seed_rows] for b in eval_bundles], axis=0
                )
                ee_b_seed = jnp.stack(
                    [b.excl_dense_scales_elec[:n_seed_rows] for b in eval_bundles], axis=0
                )
            else:
                ei_b_seed = jnp.full((B_eval, n_seed_rows, 1), -1, dtype=jnp.int32)
                ev_b_seed = jnp.ones((B_eval, n_seed_rows, 1), dtype=jnp.float32)
                ee_b_seed = jnp.ones((B_eval, n_seed_rows, 1), dtype=jnp.float32)
            am_b_seed = jnp.stack([b.atom_mask[:n_seed_rows] for b in eval_bundles], axis=0)
            r_tot_seed = float(cutoff) + float(skin)
            cell_size_seed = (
                float(seed.cell_size)
                if getattr(seed, "cell_size", None) is not None
                else r_tot_seed
            )
            cell_cap_seed = int(seed.cell_list_capacity)
            max_occ_seed = int(seed.idx.shape[-1])
            grid_dims_seed = nl_mod._compute_static_grid_dims(box_vec, cell_size_seed)
            p_w_b_seed = jnp.mod(pos_b_seed, box_b_seed[:, None, :])
            idx_b_seed, occ_b_seed, mkt_b_seed = nl_mod._pallas_prune_and_encode_cell_list_batch(
                B_eval,
                (True, True, True, True, True),
                p_w_b_seed,
                box_b_seed,
                ei_b_seed,
                ev_b_seed,
                ee_b_seed,
                cell_size=cell_size_seed,
                cell_capacity=cell_cap_seed,
                max_occupancy=max_occ_seed,
                r_cutoff=r_tot_seed,
                encode_exclusions=has_excl_seed,
                interpret=(jax.default_backend() == "cpu"),
                grid_dims=grid_dims_seed,
                atom_mask=am_b_seed,
                mask_batched=True,
                return_max_k_tiles=True,
            )
            if all_same_nl and B_total > 1:
                idx_b_seed = jnp.broadcast_to(idx_b_seed, (B_total, n_seed_rows, max_occ_seed))
                occ_b_seed = jnp.broadcast_to(occ_b_seed, (B_total,))
                p_w_b_seed = jnp.broadcast_to(p_w_b_seed, (B_total, n_seed_rows, 3))
                mkt_b_seed = jnp.broadcast_to(mkt_b_seed, (B_total, *mkt_b_seed.shape[1:]))
            base_stacked = jax.tree.map(
                lambda *xs: jnp.stack(xs, axis=0), *([seed] * B_total)
            )
            overflow_bit_s = jnp.uint8(
                nl_mod.partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW
            )
            err_b_seed = jnp.where(
                occ_b_seed > max_occ_seed,
                base_stacked.error.code | overflow_bit_s,
                base_stacked.error.code,
            )
            stacked_neighbor = dataclasses.replace(
                base_stacked,
                idx=idx_b_seed,
                reference_position=p_w_b_seed.astype(base_stacked.reference_position.dtype),
                error=nl_mod.partition.PartitionError(err_b_seed),
                max_k_tiles=mkt_b_seed,
            )
            return stacked_neighbor, neighbor_fn

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
        precomputed_force0: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Shared settle_langevin init. Returns (state, apply_fn, dt, kT)."""
        import dataclasses

        import jax
        import jax.numpy as jnp
        import numpy as np

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

        bundle_for_energy = bundle
        if (
            host_box_vec is not None
            and getattr(bundle, "box", None) is not None
            and isinstance(bundle.box, jax.core.Tracer)
        ):
            hb = np.asarray(host_box_vec, dtype=np.dtype(bundle.positions.dtype))
            box_concrete = (
                np.diag(hb) if (bundle.box.ndim == 2 and hb.ndim == 1) else hb
            )
            bundle_for_energy = dataclasses.replace(bundle_for_energy, box=box_concrete)
        if int(positions_init.shape[0]) != int(bundle_for_energy.positions.shape[0]):
            bundle_for_energy = dataclasses.replace(
                bundle_for_energy, positions=positions_init
            )

        n_pad_host = (
            int(integration_prefix)
            if integration_prefix is not None
            else int(self.bundles[0].positions.shape[0])
        )
        host_pw_blk = 0
        for hb_item in self.bundles:
            if int(hb_item.positions.shape[0]) < n_pad_host:
                continue
            blk_i = _infer_first_pure_water_block(
                n_pad_host,
                (
                    (getattr(hb_item, "bond_idx", None), getattr(hb_item, "bond_mask", None), 2),
                    (getattr(hb_item, "urey_bradley_idx", None), getattr(hb_item, "urey_bradley_mask", None), 3),
                    (getattr(hb_item, "angle_idx", None), getattr(hb_item, "angle_mask", None), 3),
                    (getattr(hb_item, "dihedral_idx", None), getattr(hb_item, "dihedral_mask", None), 4),
                    (getattr(hb_item, "improper_idx", None), getattr(hb_item, "improper_mask", None), 4),
                    (getattr(hb_item, "exception_pairs", None), getattr(hb_item, "exception_mask", None), 2),
                    (getattr(hb_item, "cmap_torsion_idx", None), getattr(hb_item, "cmap_mask", None), 8),
                ),
            )
            host_pw_blk = max(host_pw_blk, blk_i)

        # debt 761: force_fn_from_bundle returns an (N, 3) force array rather
        # than a scalar energy -- settle_langevin's energy_or_force_fn
        # auto-detects which via jax.eval_shape (see
        # _make_settle_compatible_force_fn/make_force_fn_like_canonicalize),
        # so no other change is needed here for the integrator itself to
        # skip the dense-energy autodiff pass.
        import os

        env_up = os.environ.get("PROLIX_USE_PALLAS", "").strip().lower()
        pallas_enabled_here = (
            bool(use_pallas)
            if use_pallas is not None
            else (
                True
                if env_up in ("1", "true", "yes")
                else (
                    False
                    if env_up in ("0", "false", "no")
                    else (jax.default_backend() != "cpu")
                )
            )
        )
        can_pallas_gbsa = bool(
            pallas_enabled_here
            and bundle.shape_spec.has_implicit_solvent
            and not bundle.shape_spec.has_pbc
            and getattr(bundle, "excl_dense_indices", None) is not None
            and not use_flash_forces
            and initial_neighbor is None
            and lj_switch_width_from_config(nb_cfg) == 0.0
        )
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
                build_dense_exclusions=(initial_neighbor is None and not can_pallas_gbsa),
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
                excl_mode=excl_mode,
                first_pure_water_block=host_pw_blk,
                return_forces=bool(initial_neighbor is not None or can_pallas_gbsa),
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
            box=pbc_box_for_bundle(bundle),
            is_force_shaped=bool(use_flash_forces or initial_neighbor is not None or can_pallas_gbsa),
            use_pallas=use_pallas,
        )

        key = jax.random.PRNGKey(jnp.asarray(seed, dtype=jnp.uint32))
        init_kwargs: dict[str, Any] = {"kT": kT_s}
        if initial_neighbor is not None:
            init_kwargs["neighbor"] = initial_neighbor
        if precomputed_force0 is not None:
            init_kwargs["precomputed_force"] = precomputed_force0
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
        precomputed_force0: Any = None,
    ) -> Trajectory:
        """Carry-only MD via while_loop; Trajectory holds final frame only."""
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
                    os.environ.get("PROLIX_ACTIVE_ATOMS", ""),
                    os.environ.get("PROLIX_ACTIVE_WATERS", ""),
                    os.environ.get("PROLIX_PME_EVERY", ""),
                    os.environ.get("PROLIX_NL_CHECK_EVERY", ""),
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

        # MK-50: Single-JIT fast path for warm B=1 cached neighbor-list inference.
        # Executes fast_reseed_fn + while_loop + real-atom slice inside one @jax.jit
        # without synchronous mid-run host transfers (bool(did_overflow), int(n_atoms)).
        if (
            cached_entry is not None
            and not observables
            and use_neighbor_list
            and trim_output
            and not isinstance(n_steps, jax.core.Tracer)
            and cached_entry.get("fused_nl_run_fn") is not None
            and bool(cached_entry.get("verified_no_overflow", False))
        ):
            _JITTED_WHILE_CACHE[(cached_entry["nl_step_fn"], "fused_nl")] = cached_entry["fused_nl_run_fn"]
            positions_array, _ = cached_entry["fused_nl_run_fn"](
                jnp.asarray(seed, dtype=jnp.uint32),
                cached_entry["kT_s"],
                cached_entry["nbr0"],
                jnp.asarray(n_steps, dtype=jnp.int32),
            )
            return Trajectory(
                positions=positions_array,
                observable_values={
                    "_nl_capacity": cached_entry["nl_capacity"],
                    "_nl_retry_count": 0,
                },
                n_steps=int(n_steps),
            )

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
            if nbr0 is not None and getattr(nbr0, "max_k_tiles", None) is None:
                nbr0 = dataclasses.replace(
                    nbr0, max_k_tiles=nl_mod._max_k_tiles_from_idx(nbr0.idx)
                )

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
                precomputed_force0=precomputed_force0,
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
                if can_pallas_nl and has_excl_nl:
                    precomp_em_nl, precomp_ex_nl, precomp_ei32_nl, precomp_et32_nl = (
                        nl_mod.precompute_static_nl_exclusion_tables(
                            excl_idx_nl, excl_sv_nl, excl_se_nl, n_nl_rows
                        )
                    )
                else:
                    precomp_em_nl = jnp.zeros((1,), dtype=jnp.int32)
                    precomp_ex_nl = jnp.zeros((1,), dtype=jnp.int32)
                    precomp_ei32_nl = jnp.zeros((1, 32), dtype=jnp.int32)
                    precomp_et32_nl = jnp.zeros((1, 32), dtype=jnp.int32)

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

                @jax.custom_batching.custom_vmap
                def _any_replica_true(flag: Any) -> Any:
                    return jnp.asarray(flag, dtype=jnp.bool_)

                @_any_replica_true.def_vmap
                def _any_replica_true_vmap(axis_size: Any, in_batched: Any, flag: Any) -> Any:
                    del axis_size, in_batched
                    return jnp.any(flag), False

                n_act_su = nl_mod._resolve_nl_active_atoms(None, n_nl_rows)
                pad_mask_3d_su = pad_mask_3d[:n_act_su]
                nl_check_every = 1
                raw_ce = os.environ.get("PROLIX_NL_CHECK_EVERY", "").strip()
                if raw_ce:
                    try:
                        nl_check_every = max(1, int(raw_ce))
                    except ValueError:
                        nl_check_every = 1
                pme_every_val = 1
                raw_pe = os.environ.get("PROLIX_PME_EVERY", "").strip()
                if raw_pe:
                    try:
                        pme_every_val = max(1, int(raw_pe))
                    except ValueError:
                        pme_every_val = 1
                respa_2step_fn = getattr(apply_fn, "_respa_2step_fn", None)

                def _check_su(pos: Any, last_pos: Any, step_i: Any) -> Any:
                    def _eval_verlet(_):
                        d_pos = jnp.where(
                            pad_mask_3d_su,
                            jax.vmap(_nl_disp_fn)(pos[:n_act_su], last_pos[:n_act_su]),
                            0.0,
                        )
                        max_dr_sq = jnp.max(jnp.sum(d_pos * d_pos, axis=-1))
                        return max_dr_sq > thresh_sq

                    if nl_check_every > 1:
                        do_v = (step_i == 0) | ((step_i % nl_check_every) == 0)
                        verlet = jax.lax.cond(do_v, _eval_verlet, lambda _: jnp.bool_(False), operand=None)
                    else:
                        verlet = _eval_verlet(None)
                    if stride > 0:
                        by_stride = (step_i > 0) & ((step_i % stride) == 0)
                        return verlet | by_stride
                    return verlet

                def _integrate_step_only(langevin_state: Any, neighbor: Any) -> Any:
                    pos = _integrator_positions(langevin_state)
                    new_langevin_state = apply_fn(
                        langevin_state, neighbor=neighbor, kT=kT_s, dt=dt_s
                    )
                    if not getattr(apply_fn, "_pallas_settle_pins_ghosts", False):
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
                    return new_langevin_state

                def _integrate_2steps_respa(langevin_state: Any, neighbor: Any) -> Any:
                    pos = _integrator_positions(langevin_state)
                    new_langevin_state = respa_2step_fn(
                        langevin_state, neighbor=neighbor, kT=kT_s, dt=dt_s
                    )
                    if not getattr(apply_fn, "_pallas_settle_pins_ghosts", False):
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
                    return new_langevin_state

                @jax.custom_batching.custom_vmap
                def _rebuild_nl_unconditional(
                    su_in: Any,
                    pos_in: Any,
                    lp_in: Any,
                    idx_in: Any,
                    ref_in: Any,
                    err_in: Any,
                    mkt_in: Any,
                    bx_in: Any,
                    ei_in: Any,
                    ev_in: Any,
                    ee_in: Any,
                    am_in: Any,
                    pem_in: Any,
                    pex_in: Any,
                    pei32_in: Any,
                    pet32_in: Any,
                ) -> Any:
                    m_occ = int(idx_in.shape[-1])
                    if can_pallas_nl:
                        p_w = jnp.mod(pos_in, bx_in)
                        idx_new_b, occ_new_b, mkt_new_b = nl_mod._pallas_prune_and_encode_cell_list_batch(
                            1,
                            (True, True, False, False, False),
                            p_w[None, ...],
                            bx_in[None, ...],
                            ei_in,
                            ev_in,
                            ee_in,
                            cell_size=cell_size_f,
                            cell_capacity=cell_cap_i,
                            max_occupancy=m_occ,
                            r_cutoff=r_total_cutoff,
                            encode_exclusions=has_excl_nl,
                            interpret=(jax.default_backend() == "cpu"),
                            grid_dims=grid_dims_nl,
                            atom_mask=am_in,
                            mask_batched=False,
                            rebuild_mask=jnp.asarray(su_in, dtype=jnp.bool_).reshape(1),
                            old_idx=idx_in[None, ...],
                            precomputed_excl=(
                                (pem_in, pex_in, pei32_in, pet32_in) if has_excl_nl else None
                            ),
                            return_max_k_tiles=True,
                            old_max_k_tiles=mkt_in[None, ...],
                        )
                        idx_new = idx_new_b[0]
                        occ_new = occ_new_b[0]
                        mkt_new = mkt_new_b[0]
                        overflow_bit = jnp.uint8(
                            nl_mod.partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW
                        )
                        err_new = jnp.where(occ_new > m_occ, err_in | overflow_bit, err_in)
                        return idx_new, p_w.astype(ref_in.dtype), err_new, pos_in, mkt_new
                    del su_in, lp_in, mkt_in, pem_in, pex_in, pei32_in, pet32_in
                    forced_ref = ref_in + jnp.asarray(0.5, dtype=ref_in.dtype) * bx_in.astype(ref_in.dtype)
                    n_tmp = dataclasses.replace(
                        nbr0,
                        idx=idx_in,
                        reference_position=forced_ref,
                        error=nl_mod.partition.PartitionError(err_in),
                    )
                    n_upd = n_tmp.update(pos_in)
                    upd_fn = getattr(n_upd, "update_fn", None)
                    pre_enc = bool(getattr(upd_fn, "_pre_encoded_exclusions", False))
                    ghosts_filt = bool(getattr(upd_fn, "_ghosts_filtered", False))
                    if (has_excl_nl and not pre_enc) or not ghosts_filt:
                        idx_enc = nl_mod.encode_nl_exclusions(
                            n_upd.idx,
                            ei_in if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.int32),
                            ev_in if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.float32),
                            ee_in if (has_excl_nl and not pre_enc) else jnp.zeros((n_nl_rows, 0), dtype=jnp.float32),
                            n_atoms=n_nl_rows,
                            atom_mask=am_in,
                        )
                    else:
                        idx_enc = n_upd.idx
                    mkt_fb = nl_mod._max_k_tiles_from_idx(idx_enc)
                    return idx_enc, n_upd.reference_position, n_upd.error.code, pos_in, mkt_fb

                @_rebuild_nl_unconditional.def_vmap
                def _rebuild_nl_unconditional_vmap(
                    axis_size: Any,
                    in_batched: Any,
                    su_in: Any,
                    pos_in: Any,
                    lp_in: Any,
                    idx_in: Any,
                    ref_in: Any,
                    err_in: Any,
                    mkt_in: Any,
                    bx_in: Any,
                    ei_in: Any,
                    ev_in: Any,
                    ee_in: Any,
                    am_in: Any,
                    pem_in: Any,
                    pex_in: Any,
                    pei32_in: Any,
                    pet32_in: Any,
                ) -> Any:
                    B = int(axis_size)
                    su_b = su_in if in_batched[0] else jnp.broadcast_to(su_in[None, ...], (B,))
                    pos_b = pos_in if in_batched[1] else jnp.broadcast_to(pos_in[None, ...], (B, *pos_in.shape))
                    lp_b = lp_in if in_batched[2] else jnp.broadcast_to(lp_in[None, ...], (B, *lp_in.shape))
                    idx_b = idx_in if in_batched[3] else jnp.broadcast_to(idx_in[None, ...], (B, *idx_in.shape))
                    ref_b = ref_in if in_batched[4] else jnp.broadcast_to(ref_in[None, ...], (B, *ref_in.shape))
                    err_b = err_in if in_batched[5] else jnp.broadcast_to(err_in[None, ...], (B, *err_in.shape))
                    mkt_b = mkt_in if in_batched[6] else jnp.broadcast_to(mkt_in[None, ...], (B, *mkt_in.shape))
                    bx_b = bx_in if in_batched[7] else jnp.broadcast_to(bx_in[None, ...], (B, *bx_in.shape))
                    ei_b = ei_in if in_batched[8] else ei_in[None, ...]
                    ev_b = ev_in if in_batched[9] else ev_in[None, ...]
                    ee_b = ee_in if in_batched[10] else ee_in[None, ...]
                    am_b = am_in if in_batched[11] else am_in[None, ...]
                    m_occ = int(idx_b.shape[-1])

                    if can_pallas_nl:
                        p_w_b = jnp.mod(pos_b, bx_b[:, None, :])
                        excl_batched_vmap = bool(
                            in_batched[8] or in_batched[9] or in_batched[10] or in_batched[12] or in_batched[13] or in_batched[14] or in_batched[15]
                        )
                        if has_excl_nl:
                            pem_arg = (
                                (pem_in if in_batched[12] else jnp.broadcast_to(pem_in[None, ...], (B, *pem_in.shape)))
                                if excl_batched_vmap
                                else pem_in
                            )
                            pex_arg = (
                                (pex_in if in_batched[13] else jnp.broadcast_to(pex_in[None, ...], (B, *pex_in.shape)))
                                if excl_batched_vmap
                                else pex_in
                            )
                            pei32_arg = (
                                (pei32_in if in_batched[14] else jnp.broadcast_to(pei32_in[None, ...], (B, *pei32_in.shape)))
                                if excl_batched_vmap
                                else pei32_in
                            )
                            pet32_arg = (
                                (pet32_in if in_batched[15] else jnp.broadcast_to(pet32_in[None, ...], (B, *pet32_in.shape)))
                                if excl_batched_vmap
                                else pet32_in
                            )
                            precomp_arg = (pem_arg, pex_arg, pei32_arg, pet32_arg)
                        else:
                            precomp_arg = None
                        idx_new_b, occ_new_b, mkt_new_b = nl_mod._pallas_prune_and_encode_cell_list_batch(
                            B,
                            (True, True, excl_batched_vmap, excl_batched_vmap, excl_batched_vmap),
                            p_w_b,
                            bx_b,
                            ei_b if in_batched[8] else ei_in,
                            ev_b if in_batched[9] else ev_in,
                            ee_b if in_batched[10] else ee_in,
                            cell_size=cell_size_f,
                            cell_capacity=cell_cap_i,
                            max_occupancy=m_occ,
                            r_cutoff=r_total_cutoff,
                            encode_exclusions=has_excl_nl,
                            interpret=(jax.default_backend() == "cpu"),
                            grid_dims=grid_dims_nl,
                            atom_mask=am_b if in_batched[11] else am_in,
                            mask_batched=bool(in_batched[11]),
                            rebuild_mask=su_b,
                            old_idx=idx_b,
                            precomputed_excl=precomp_arg,
                            return_max_k_tiles=True,
                            old_max_k_tiles=mkt_b,
                        )
                        overflow_bit = jnp.uint8(
                            nl_mod.partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW
                        )
                        err_new_b = jnp.where(
                            su_b & (occ_new_b > m_occ),
                            err_b | overflow_bit,
                            err_b,
                        )
                        ref_new_b = jnp.where(
                            su_b[:, None, None],
                            p_w_b.astype(ref_b.dtype),
                            ref_b,
                        )
                        last_new_b = jnp.where(
                            su_b[:, None, None],
                            pos_b,
                            lp_b,
                        )
                        return (idx_new_b, ref_new_b, err_new_b, last_new_b, mkt_new_b), (True, True, True, True, True)

                    del pem_in, pex_in, pei32_in, pet32_in
                    ei_full = jnp.broadcast_to(ei_b, (B, *ei_b.shape[1:]))
                    ev_full = jnp.broadcast_to(ev_b, (B, *ev_b.shape[1:]))
                    ee_full = jnp.broadcast_to(ee_b, (B, *ee_b.shape[1:]))
                    am_full = jnp.broadcast_to(am_b, (B, *am_b.shape[1:]))

                    def _one_fallback(su_1, p_1, lp_1, r_1, i_1, e_1, mkt_1, bx_1, ei_1, ev_1, ee_1, am_1):
                        def _do_one(_):
                            forced_ref = r_1 + jnp.asarray(0.5, dtype=r_1.dtype) * bx_1.astype(r_1.dtype)
                            n_tmp = dataclasses.replace(
                                nbr0,
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
                            return idx_enc, n_upd.reference_position, n_upd.error.code, p_1, nl_mod._max_k_tiles_from_idx(idx_enc)

                        def _keep_one(_):
                            return i_1, r_1, e_1, lp_1, mkt_1

                        return jax.lax.cond(su_1, _do_one, _keep_one, operand=None)

                    out_fb = jax.lax.map(
                        lambda xs: _one_fallback(*xs),
                        (su_b, pos_b, lp_b, ref_b, idx_b, err_b, mkt_b, bx_b, ei_full, ev_full, ee_full, am_full),
                    )
                    return out_fb, (True, True, True, True, True)

                def _nl_step_fn(carry: _NLDispatchCarry, step_i: Any) -> _NLDispatchCarry:
                    langevin_state, neighbor, did_overflow, last_pos = carry
                    pos = _integrator_positions(langevin_state)
                    should_update = _check_su(pos, last_pos, step_i)

                    @jax.custom_batching.custom_vmap
                    def _maybe_rebuild_nl(
                        su_in,
                        pos_in,
                        lp_in,
                        idx_in,
                        ref_in,
                        err_in,
                        mkt_in,
                        bx_in,
                        ei_in,
                        ev_in,
                        ee_in,
                        am_in,
                        pem_in,
                        pex_in,
                        pei32_in,
                        pet32_in,
                    ):
                        def _do_one(operands):
                            p_o, lp_o, i_o, r_o, e_o, mkt_o, bx_o, ei_o, ev_o, ee_o, am_o, pem_o, pex_o, pei32_o, pet32_o = operands
                            return _rebuild_nl_unconditional(
                                jnp.bool_(True), p_o, lp_o, i_o, r_o, e_o, mkt_o, bx_o, ei_o, ev_o, ee_o, am_o, pem_o, pex_o, pei32_o, pet32_o
                            )

                        def _keep_one(operands):
                            _p_o, lp_o, i_o, r_o, e_o, mkt_o, _bx_o, _ei_o, _ev_o, _ee_o, _am_o, _pem_o, _pex_o, _pei32_o, _pet32_o = operands
                            return i_o, r_o, e_o, lp_o, mkt_o

                        return jax.lax.cond(
                            su_in,
                            _do_one,
                            _keep_one,
                            (pos_in, lp_in, idx_in, ref_in, err_in, mkt_in, bx_in, ei_in, ev_in, ee_in, am_in, pem_in, pex_in, pei32_in, pet32_in),
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
                        mkt_in,
                        bx_in,
                        ei_in,
                        ev_in,
                        ee_in,
                        am_in,
                        pem_in,
                        pex_in,
                        pei32_in,
                        pet32_in,
                    ):
                        B = int(axis_size)
                        su_b = su_in if in_batched[0] else jnp.broadcast_to(su_in[None, ...], (B,))
                        pos_b = pos_in if in_batched[1] else jnp.broadcast_to(pos_in[None, ...], (B, *pos_in.shape))
                        lp_b = lp_in if in_batched[2] else jnp.broadcast_to(lp_in[None, ...], (B, *lp_in.shape))
                        idx_b = idx_in if in_batched[3] else jnp.broadcast_to(idx_in[None, ...], (B, *idx_in.shape))
                        ref_b = ref_in if in_batched[4] else jnp.broadcast_to(ref_in[None, ...], (B, *ref_in.shape))
                        err_b = err_in if in_batched[5] else jnp.broadcast_to(err_in[None, ...], (B, *err_in.shape))
                        mkt_b = mkt_in if in_batched[6] else jnp.broadcast_to(mkt_in[None, ...], (B, *mkt_in.shape))
                        bx_b = bx_in if in_batched[7] else jnp.broadcast_to(bx_in[None, ...], (B, *bx_in.shape))
                        ei_b = ei_in if in_batched[8] else ei_in[None, ...]
                        ev_b = ev_in if in_batched[9] else ev_in[None, ...]
                        ee_b = ee_in if in_batched[10] else ee_in[None, ...]
                        am_b = am_in if in_batched[11] else am_in[None, ...]
                        pem_b = pem_in if in_batched[12] else pem_in[None, ...]
                        pex_b = pex_in if in_batched[13] else pex_in[None, ...]
                        pei32_b = pei32_in if in_batched[14] else pei32_in[None, ...]
                        pet32_b = pet32_in if in_batched[15] else pet32_in[None, ...]

                        any_upd = jnp.any(su_b)

                        def _do_batch(operands):
                            su_op, p_b, lp_op, i_b, r_b, e_b, m_b, b_b, ei_op, ev_op, ee_op, am_op, pem_op, pex_op, pei32_op, pet32_op = operands
                            out_b, _ = _rebuild_nl_unconditional_vmap(
                                B,
                                (
                                    True,
                                    True,
                                    True,
                                    True,
                                    True,
                                    True,
                                    True,
                                    True,
                                    in_batched[8],
                                    in_batched[9],
                                    in_batched[10],
                                    in_batched[11],
                                    in_batched[12],
                                    in_batched[13],
                                    in_batched[14],
                                    in_batched[15],
                                ),
                                su_op,
                                p_b,
                                lp_op,
                                i_b,
                                r_b,
                                e_b,
                                m_b,
                                b_b,
                                ei_op if in_batched[8] else ei_in,
                                ev_op if in_batched[9] else ev_in,
                                ee_op if in_batched[10] else ee_in,
                                am_op if in_batched[11] else am_in,
                                pem_op if in_batched[12] else pem_in,
                                pex_op if in_batched[13] else pex_in,
                                pei32_op if in_batched[14] else pei32_in,
                                pet32_op if in_batched[15] else pet32_in,
                            )
                            return out_b

                        def _keep_batch(operands):
                            _su_op, _p_b, l_b, i_b, r_b, e_b, m_b, _b_b, _ei_op, _ev_op, _ee_op, _am_op, _pem_op, _pex_op, _pei32_op, _pet32_op = operands
                            return i_b, r_b, e_b, l_b, m_b

                        out = jax.lax.cond(
                            any_upd,
                            _do_batch,
                            _keep_batch,
                            (su_b, pos_b, lp_b, idx_b, ref_b, err_b, mkt_b, bx_b, ei_b, ev_b, ee_b, am_b, pem_b, pex_b, pei32_b, pet32_b),
                        )
                        return out, (True, True, True, True, True)

                    mkt_cur = (
                        neighbor.max_k_tiles
                        if getattr(neighbor, "max_k_tiles", None) is not None
                        else nl_mod._max_k_tiles_from_idx(neighbor.idx)
                    )
                    new_idx, new_ref, new_err_code, new_last, new_mkt = _maybe_rebuild_nl(
                        should_update,
                        pos,
                        last_pos,
                        neighbor.idx,
                        neighbor.reference_position,
                        neighbor.error.code,
                        mkt_cur,
                        box_vec_nl,
                        excl_idx_nl,
                        excl_sv_nl,
                        excl_se_nl,
                        pad_mask_1d,
                        precomp_em_nl,
                        precomp_ex_nl,
                        precomp_ei32_nl,
                        precomp_et32_nl,
                    )
                    new_neighbor = dataclasses.replace(
                        neighbor,
                        idx=new_idx,
                        reference_position=new_ref,
                        error=nl_mod.partition.PartitionError(new_err_code),
                        max_k_tiles=new_mkt,
                    )
                    did_overflow = did_overflow | new_neighbor.did_buffer_overflow.astype(bool)
                    new_langevin_state = _integrate_step_only(langevin_state, new_neighbor)
                    return _NLDispatchCarry(
                        new_langevin_state, new_neighbor, did_overflow, new_last
                    )

                def _two_level_runner(
                    init_carry: _NLDispatchCarry, n_steps_in: Any, iterator: Any
                ) -> _NLDispatchCarry:
                    n_limit = jnp.asarray(n_steps_in, dtype=jnp.int32)
                    st0, nb0, ov0, lp0 = init_carry
                    has_nvt_fields = (
                        hasattr(st0, "mass")
                        and hasattr(st0, "position")
                        and hasattr(st0, "momentum")
                        and hasattr(st0, "force")
                        and hasattr(st0, "rng")
                        and hasattr(st0, "replace")
                    )
                    mass_fixed = st0.mass if has_nvt_fields else None

                    def _run_inner_segment(
                        s_start: Any,
                        st_start: Any,
                        nb_fixed: Any,
                        lp_fixed: Any,
                        any_u_start: Any,
                    ) -> tuple[Any, Any, Any]:
                        if has_nvt_fields:
                            if pme_every_val == 2 and respa_2step_fn is not None:
                                def _inner_cond_nvt_2(ic: Any) -> Any:
                                    s_i, _p_i, _m_i, _f_i, _r_i, any_u = ic
                                    return (s_i + jnp.int32(2) <= n_limit) & (~any_u)

                                def _inner_body_nvt_2(ic: Any) -> Any:
                                    s_i, p_i, m_i, f_i, r_i, _any_u = ic
                                    st_in = st_start.replace(
                                        position=p_i,
                                        momentum=m_i,
                                        force=f_i,
                                        mass=mass_fixed,
                                        rng=r_i,
                                    )
                                    st_next = _integrate_2steps_respa(st_in, nb_fixed)
                                    s_next = s_i + jnp.int32(2)
                                    pos_next = st_next.position
                                    su_next = _check_su(pos_next, lp_fixed, s_next)
                                    any_u_next = _any_replica_true(su_next)
                                    return (
                                        s_next,
                                        pos_next,
                                        st_next.momentum,
                                        st_next.force,
                                        st_next.rng,
                                        any_u_next,
                                    )

                                s_out, p_out, m_out, f_out, r_out, any_u_out = iterator(
                                    _inner_cond_nvt_2,
                                    _inner_body_nvt_2,
                                    (
                                        s_start,
                                        st_start.position,
                                        st_start.momentum,
                                        st_start.force,
                                        st_start.rng,
                                        any_u_start,
                                    ),
                                )
                            else:
                                s_out = s_start
                                p_out = st_start.position
                                m_out = st_start.momentum
                                f_out = st_start.force
                                r_out = st_start.rng
                                any_u_out = any_u_start

                            def _inner_cond_nvt(ic: Any) -> Any:
                                s_i, _p_i, _m_i, _f_i, _r_i, any_u = ic
                                return (s_i < n_limit) & (~any_u)

                            def _inner_body_nvt(ic: Any) -> Any:
                                s_i, p_i, m_i, f_i, r_i, _any_u = ic
                                st_in = st_start.replace(
                                    position=p_i,
                                    momentum=m_i,
                                    force=f_i,
                                    mass=mass_fixed,
                                    rng=r_i,
                                )
                                st_next = _integrate_step_only(st_in, nb_fixed)
                                s_next = s_i + jnp.int32(1)
                                pos_next = st_next.position
                                su_next = _check_su(pos_next, lp_fixed, s_next)
                                any_u_next = _any_replica_true(su_next)
                                return (
                                    s_next,
                                    pos_next,
                                    st_next.momentum,
                                    st_next.force,
                                    st_next.rng,
                                    any_u_next,
                                )

                            s_out, p_out, m_out, f_out, r_out, any_u_out = iterator(
                                _inner_cond_nvt,
                                _inner_body_nvt,
                                (
                                    s_out,
                                    p_out,
                                    m_out,
                                    f_out,
                                    r_out,
                                    any_u_out,
                                ),
                            )
                            st_out = st_start.replace(
                                position=p_out,
                                momentum=m_out,
                                force=f_out,
                                mass=mass_fixed,
                                rng=r_out,
                            )
                            return (s_out, st_out, any_u_out)

                        def _inner_cond(ic: Any) -> Any:
                            s_i, _st_i, any_u = ic
                            return (s_i < n_limit) & (~any_u)

                        def _inner_body(ic: Any) -> Any:
                            s_i, st_i, _any_u = ic
                            st_next = _integrate_step_only(st_i, nb_fixed)
                            s_next = s_i + jnp.int32(1)
                            pos_next = _integrator_positions(st_next)
                            su_next = _check_su(pos_next, lp_fixed, s_next)
                            any_u_next = _any_replica_true(su_next)
                            return (s_next, st_next, any_u_next)

                        return iterator(
                            _inner_cond, _inner_body, (s_start, st_start, any_u_start)
                        )

                    pos0 = _integrator_positions(st0)
                    su0 = _check_su(pos0, lp0, jnp.int32(0))
                    any_u0 = _any_replica_true(su0)
                    s1, st1, any_u1 = _run_inner_segment(
                        jnp.int32(0), st0, nb0, lp0, any_u0
                    )

                    def _outer_cond(oc: Any) -> Any:
                        s_i, _st_i, _nb_i, _ov_i, _lp_i, any_u = oc
                        return (s_i < n_limit) & any_u

                    def _outer_body(oc: Any) -> Any:
                        s_i, st_i, nb_i, ov_i, lp_i, _any_u = oc
                        pos_i = _integrator_positions(st_i)
                        su_i = _check_su(pos_i, lp_i, s_i)
                        mkt_i = (
                            nb_i.max_k_tiles
                            if getattr(nb_i, "max_k_tiles", None) is not None
                            else nl_mod._max_k_tiles_from_idx(nb_i.idx)
                        )
                        new_idx, new_ref, new_err, new_lp, new_mkt = _rebuild_nl_unconditional(
                            su_i,
                            pos_i,
                            lp_i,
                            nb_i.idx,
                            nb_i.reference_position,
                            nb_i.error.code,
                            mkt_i,
                            box_vec_nl,
                            excl_idx_nl,
                            excl_sv_nl,
                            excl_se_nl,
                            pad_mask_1d,
                            precomp_em_nl,
                            precomp_ex_nl,
                            precomp_ei32_nl,
                            precomp_et32_nl,
                        )
                        nb_new = dataclasses.replace(
                            nb_i,
                            idx=new_idx,
                            reference_position=new_ref,
                            error=nl_mod.partition.PartitionError(new_err),
                            max_k_tiles=new_mkt,
                        )
                        ov_new = ov_i | nb_new.did_buffer_overflow.astype(bool)
                        s_next, st_next, any_u_next = _run_inner_segment(
                            s_i, st_i, nb_new, new_lp, jnp.bool_(False)
                        )
                        return (s_next, st_next, nb_new, ov_new, new_lp, any_u_next)

                    _sf, st_f, nb_f, ov_f, lp_f, _auf = iterator(
                        _outer_cond, _outer_body, (s1, st1, nb0, ov0, lp0, any_u1)
                    )
                    return _NLDispatchCarry(st_f, nb_f, ov_f, lp_f)

                _nl_step_fn._two_level_runner = _two_level_runner  # type: ignore[attr-defined]


            if can_cache and cached_entry is None and cache_key is not None:
                fast_reseed_fn = self._build_fast_reseed_fn(
                    bundle, state, integration_prefix=integration_prefix, gamma=gamma
                )
                raw_n_b = bundle.n_atoms
                n_real_cached = raw_n_b if isinstance(raw_n_b, int) else int(jnp.asarray(raw_n_b))

                @jax.jit
                def _fused_nl_run(seed_u32: Any, kT_val: Any, nbr_in: Any, n_steps_i32: Any) -> Any:
                    st_reseeded = fast_reseed_fn(seed_u32, kT_val)
                    init_p = _integrator_positions(st_reseeded)
                    c0 = _NLDispatchCarry(st_reseeded, nbr_in, jnp.array(False), init_p)
                    fc = dispatch_n_steps_inference(_nl_step_fn, c0, n_steps_i32)
                    return fc.langevin_state.position[None, :n_real_cached, :], fc.did_overflow

                if len(_SINGLE_INFERENCE_CACHE) >= 8:
                    _SINGLE_INFERENCE_CACHE.pop(next(iter(_SINGLE_INFERENCE_CACHE)))
                cached_entry = {
                    "bundle": bundle,
                    "nbr0": nbr0,
                    "neighbor_fn": neighbor_fn,
                    "target_capacity": target_capacity,
                    "nl_capacity": int(nbr0.idx.shape[-1]),
                    "n_real": n_real_cached,
                    "cutoff": cutoff,
                    "skin": skin,
                    "apply_fn": apply_fn,
                    "dt_s": dt_s,
                    "kT_s": kT_s,
                    "fast_reseed_fn": fast_reseed_fn,
                    "fused_nl_run_fn": _fused_nl_run,
                    "verified_no_overflow": False,
                    "nl_step_fn": _nl_step_fn,
                    "step_fn": None,
                }
                cache_key = _make_cache_key()
                _SINGLE_INFERENCE_CACHE[cache_key] = cached_entry

                if (
                    not observables
                    and trim_output
                    and initial_neighbor is None
                    and not isinstance(n_steps, jax.core.Tracer)
                ):
                    if len(_JITTED_WHILE_CACHE) >= 128:
                        _JITTED_WHILE_CACHE.pop(next(iter(_JITTED_WHILE_CACHE)))
                    _JITTED_WHILE_CACHE[(_nl_step_fn, "fused_nl")] = _fused_nl_run
                    positions_array, did_overflow_1 = _fused_nl_run(
                        jnp.asarray(seed, dtype=jnp.uint32),
                        kT_s,
                        nbr0,
                        jnp.asarray(n_steps, dtype=jnp.int32),
                    )
                    if not bool(did_overflow_1):
                        cached_entry["verified_no_overflow"] = True
                        return Trajectory(
                            positions=positions_array,
                            observable_values={
                                "_nl_capacity": cached_entry["nl_capacity"],
                                "_nl_retry_count": 0,
                            },
                            n_steps=int(n_steps),
                        )

                state = fast_reseed_fn(jnp.asarray(seed, dtype=jnp.uint32), kT_s)

            init_pos = _integrator_positions(state)
            init_carry = _NLDispatchCarry(state, nbr0, jnp.array(False), init_pos)
            final_carry = dispatch_n_steps_inference(
                _nl_step_fn, init_carry, n_steps
            )
            if initial_neighbor is not None:
                nl_overflow_for_caller = final_carry.did_overflow
            elif neighbor_fn is not None and bool(final_carry.did_overflow):
                retry_positions = _integrator_positions(state)
                base_capacity = int(nbr0.idx.shape[-1])
                retried_neighbor = nbr0
                for attempt in range(1, _NL_MAX_REALLOC_ATTEMPTS + 1):
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
                    cached_entry["nl_capacity"] = nl_capacity_for_caller
                    cached_entry["verified_no_overflow"] = True
            elif cached_entry is not None:
                cached_entry["verified_no_overflow"] = True
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
                    raw_n_b = bundle.n_atoms
                    n_real_cached = (
                        raw_n_b if isinstance(raw_n_b, int) else int(jnp.asarray(raw_n_b))
                    )
                    state = fast_reseed_fn(jnp.asarray(seed, dtype=jnp.uint32), kT_s)
                    if len(_SINGLE_INFERENCE_CACHE) >= 8:
                        _SINGLE_INFERENCE_CACHE.pop(next(iter(_SINGLE_INFERENCE_CACHE)))
                    cached_entry = {
                        "bundle": bundle,
                        "nbr0": None,
                        "neighbor_fn": None,
                        "target_capacity": 0,
                        "nl_capacity": 0,
                        "n_real": n_real_cached,
                        "cutoff": cutoff,
                        "skin": skin,
                        "apply_fn": apply_fn,
                        "dt_s": dt_s,
                        "kT_s": kT_s,
                        "fast_reseed_fn": fast_reseed_fn,
                        "fused_nl_run_fn": None,
                        "verified_no_overflow": True,
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
        n_real_opt = cached_entry.get("n_real") if cached_entry is not None else None
        if trim_output:
            stacked = final_pos[None, ...]
            stacked = trim_trajectory_positions(stacked, bundle, n_real=n_real_opt)
            positions_array = stacked
        else:
            n_active = (
                integration_prefix
                if integration_prefix is not None
                else (n_real_opt if n_real_opt is not None else int(jnp.asarray(bundle.n_atoms)))
            )
            positions_array = final_pos[None, :n_active, :]

        observable_values: dict[str, Any] = {}
        if observables:
            for name, observable in observables.items():
                observable_values[name] = observable.compute(state)
        if nl_overflow_for_caller is not None:
            observable_values["_nl_did_overflow"] = nl_overflow_for_caller
        if nl_capacity_for_caller is not None:
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
