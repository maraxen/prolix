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

import contextlib as _contextlib
import dataclasses
import inspect
import math
import os
import sys
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad as jax_ad

from prolix.api import bundle_md as _bundle_md_mod
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
from prolix.api.observables import (
    KCAL_MOL_PER_A3_TO_BAR,
    Energy,
    KineticEnergy,
    Pressure,
    Temperature,
    Trajectory,
)
from prolix.api.xtc_sink import XtcFrameSink, write_positions_xtc
from prolix.constants import COULOMB_CONSTANT
from prolix.export import in_export_context
from prolix.knob_surface import KnobSurface  # noqa: TC001
from prolix.pallas_kernels import _infer_first_pure_water_block, pallas_bonded_all_fused
from prolix.physics import explicit_corrections
from prolix.physics import neighbor_list as nl_mod
from prolix.physics import pme_pallas as pme_pallas_mod
from prolix.physics.kups_adapter import AKMA_TIME_UNIT_FS, gamma_ps_to_akma
from prolix.physics.nonbonded_config import (
    engine_flags,
    lj_switch_width_from_config,
    resolve_nonbonded_config,
)
from prolix.physics.pallas_gbsa_tile import (
    build_gbsa_tile_pair_list,
    pallas_gbsa_tile_energy_and_forces,
)
from prolix.physics.pallas_ragged_queue import (
    PMETierMode,
    build_ragged_tile_pair_queue,
    pallas_nonbonded_tile_pairs_ragged,
    plan_pme_tiers,
    spme_forces_ragged_tiered,
)
from prolix.physics.pallas_tile_pairs import (
    DEFAULT_NPT_BOX_SAFETY,
    _pallas_bonded_and_tile_pairs_soa,
    estimate_max_tile_pairs,
    pallas_build_morton_tile_pairs_gpu,
    precompute_dedup_excl_edges,
    prepare_fused_bonded_buffers,
)
from prolix.physics.pme import (
    _cached_host_influence_function_np,
    _resolve_env_pme_grid_dims,
    compute_pme_grid_dims,
    influence_function,
    spme_self_energy,
)
from prolix.physics.pme_pallas import _gather_pallas_soa, _spread_charges_pallas_soa
from prolix.physics.settle import (
    _pallas_baoab_post_soa,
    _pallas_baoab_pre_force_batch,
    build_baoab_soa_static_plan,
    settle_langevin,
)
from prolix.types.bundles import (
    ANGLE_BUCKETS,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    CMAP_BUCKETS,
    CMAP_MAP_BUCKETS,
    CONSTRAINT_BUCKETS,
    DIHEDRAL_BUCKETS,
    EXCEPTION_BUCKETS,
    EXCL_BUCKETS,
    WATER_BUCKETS,
    MolecularShapeSpec,
)
from prolix.typing import NVTLangevinState

try:
    from prolix.openmm.converter import from_openmm_system as _omm_from_system
    from prolix.openmm.units import to_akma as _omm_to_akma

    _from_openmm_system: Any = _omm_from_system
    _to_akma: Any = _omm_to_akma
except ImportError:  # pragma: no cover - optional openmm dependency
    _from_openmm_system = None
    _to_akma = None

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

#: Cache for active atom/water counts and cutoff values per bundle identity to avoid synchronous GPU->CPU transfers on warm EnsemblePlan.run() calls.
_ACTIVE_ATOM_ENV_CACHE: dict[Any, tuple[int | None, int | None]] = {}
_BUNDLE_CUTOFF_CACHE: dict[int, float] = {}
_MORTON_SOA_CACHE: dict[Any, dict[str, Any]] = {}

_ORIG_MAKE_NL_FN = nl_mod.make_neighbor_list_fn
_ORIG_BMD_SINGLE_ENERGY = _bundle_md_mod.single_padded_energy
_ORIG_BMD_BONDED_FUSED = _bundle_md_mod.pallas_bonded_all_fused
_ORIG_SPREAD_SINGLE = pme_pallas_mod._spread_charges_pallas_single
_ORIG_GATHER_SINGLE = pme_pallas_mod._gather_pallas_single


def _pad_axis0_ep(arr: Any, target: int, fill_value: float | bool = 0) -> Any:
    if arr is None:
        return None
    cur = int(arr.shape[0])
    if cur == target:
        return arr
    if cur > target:
        return arr[:target]
    pad_spec = [(0, target - cur)] + [(0, 0)] * (arr.ndim - 1)
    return jnp.pad(arr, pad_spec, constant_values=fill_value)


def _pad_positions_to_bucket_ep(real_pos: Any, bucket: int, box_mat: Any) -> Any:
    n_real = int(real_pos.shape[0])
    n_ghost = bucket - n_real
    if n_ghost <= 0:
        return jnp.array(real_pos[:bucket], copy=True)
    box_np = None if box_mat is None else np.asarray(box_mat, dtype=np.float64)
    if box_np is not None and box_np.shape == (3, 3):
        box_np = np.diag(box_np)
    if box_np is None or box_np.shape != (3,) or np.any(box_np <= 0.0):
        ghost = jnp.zeros((n_ghost, 3), dtype=real_pos.dtype)
    else:
        m = max(int(np.ceil(n_ghost ** (1.0 / 3.0))), 1)
        axes = [np.linspace(0.0, box_np[d], num=m, endpoint=False) for d in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        ghost = jnp.asarray(grid[:n_ghost], dtype=real_pos.dtype)
    return jnp.concatenate([real_pos, ghost], axis=0)


def _union_shape_spec(specs: Sequence[MolecularShapeSpec]) -> MolecularShapeSpec:
    return MolecularShapeSpec(
        atom_bucket_idx=max(s.atom_bucket_idx for s in specs),
        bond_bucket_idx=max(s.bond_bucket_idx for s in specs),
        angle_bucket_idx=max(s.angle_bucket_idx for s in specs),
        dihedral_bucket_idx=max(s.dihedral_bucket_idx for s in specs),
        water_bucket_idx=max(s.water_bucket_idx for s in specs),
        excl_bucket_idx=max(s.excl_bucket_idx for s in specs),
        cmap_bucket_idx=max(s.cmap_bucket_idx for s in specs),
        exception_bucket_idx=max(s.exception_bucket_idx for s in specs),
        has_pbc=specs[0].has_pbc,
        has_implicit_solvent=specs[0].has_implicit_solvent,
        boundary_condition=specs[0].boundary_condition,
        has_real_water=any(s.has_real_water for s in specs),
        use_size_buckets=True,
        exact_n_atoms=0,
        exact_n_waters=0,
        cmap_map_bucket_idx=max(s.cmap_map_bucket_idx for s in specs),
        constraint_bucket_idx=max(s.constraint_bucket_idx for s in specs),
        has_real_constraints=any(s.has_real_constraints for s in specs),
        is_triclinic=specs[0].is_triclinic,
        has_real_cosolvents=any(s.has_real_cosolvents for s in specs),
    )


def _bucket_or_len(buckets: Sequence[int], idx: int) -> int:
    return int(buckets[idx]) if 0 <= int(idx) < len(buckets) else int(idx)


def _re_pad_bundle_to_spec(bundle: Any, target_spec: MolecularShapeSpec) -> Any:
    if bundle.shape_spec == target_spec:
        return bundle
    a = _bucket_or_len(ATOM_BUCKETS, target_spec.atom_bucket_idx)
    bb = _bucket_or_len(BOND_BUCKETS, target_spec.bond_bucket_idx)
    ab = _bucket_or_len(ANGLE_BUCKETS, target_spec.angle_bucket_idx)
    db = _bucket_or_len(DIHEDRAL_BUCKETS, target_spec.dihedral_bucket_idx)
    wb = _bucket_or_len(WATER_BUCKETS, target_spec.water_bucket_idx)
    eb = _bucket_or_len(EXCL_BUCKETS, target_spec.excl_bucket_idx)
    cb = _bucket_or_len(CMAP_BUCKETS, target_spec.cmap_bucket_idx)
    mb = _bucket_or_len(CMAP_MAP_BUCKETS, target_spec.cmap_map_bucket_idx)
    xb = _bucket_or_len(EXCEPTION_BUCKETS, target_spec.exception_bucket_idx)
    cons_b = _bucket_or_len(CONSTRAINT_BUCKETS, target_spec.constraint_bucket_idx)
    n_real = int(jnp.asarray(bundle.n_atoms))
    pos_padded = _pad_positions_to_bucket_ep(
        bundle.positions[:n_real], a, bundle.box if target_spec.has_pbc else None
    )
    if cons_b == 0:
        c_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
        c_lens = jnp.zeros((0,), dtype=jnp.float32)
        c_mask = jnp.zeros((0,), dtype=bool)
    else:
        cp_in, cl_in, cm_in = bundle.constraint_pairs, bundle.constraint_lengths, bundle.constraint_mask
        c_pairs = _pad_axis0_ep(cp_in, cons_b, 0) if cp_in is not None and cp_in.size > 0 else jnp.zeros((cons_b, 2), dtype=jnp.int32)
        c_lens = _pad_axis0_ep(cl_in, cons_b, 0.0) if cl_in is not None and cl_in.size > 0 else jnp.zeros((cons_b,), dtype=jnp.float32)
        c_mask = _pad_axis0_ep(cm_in, cons_b, False) if cm_in is not None and cm_in.size > 0 else jnp.zeros((cons_b,), dtype=bool)
    mol_id = bundle.molecule_id
    if mol_id is not None:
        cur_a = int(mol_id.shape[0])
        mol_id = jnp.concatenate([mol_id, jnp.arange(cur_a, a, dtype=jnp.int32) + jnp.max(mol_id) + 1], axis=0) if cur_a < a else mol_id[:a]
    excl_di = _pad_axis0_ep(bundle.excl_dense_indices, a, -1)
    excl_sv = _pad_axis0_ep(bundle.excl_dense_scales_vdw, a, 1.0)
    excl_se = _pad_axis0_ep(bundle.excl_dense_scales_elec, a, 1.0)
    if excl_di is None:
        excl_di = jnp.full((a, 32), -1, dtype=jnp.int32)
        excl_sv = jnp.ones((a, 32), dtype=jnp.float32)
        excl_se = jnp.ones((a, 32), dtype=jnp.float32)
    return dataclasses.replace(
        bundle,
        positions=pos_padded,
        masses=_pad_axis0_ep(bundle.masses, a, 1.0),
        charges=_pad_axis0_ep(bundle.charges, a, 0.0),
        sigmas=_pad_axis0_ep(bundle.sigmas, a, 0.0),
        epsilons=_pad_axis0_ep(bundle.epsilons, a, 0.0),
        radii=_pad_axis0_ep(bundle.radii, a, 0.0),
        scaled_radii=_pad_axis0_ep(bundle.scaled_radii, a, 0.0),
        atom_mask=_pad_axis0_ep(bundle.atom_mask, a, False),
        bond_idx=_pad_axis0_ep(bundle.bond_idx, bb, 0),
        bond_params=_pad_axis0_ep(bundle.bond_params, bb, 0.0),
        bond_mask=_pad_axis0_ep(bundle.bond_mask, bb, False),
        angle_idx=_pad_axis0_ep(bundle.angle_idx, ab, 0),
        angle_params=_pad_axis0_ep(bundle.angle_params, ab, 0.0),
        angle_mask=_pad_axis0_ep(bundle.angle_mask, ab, False),
        dihedral_idx=_pad_axis0_ep(bundle.dihedral_idx, db, 0),
        dihedral_params=_pad_axis0_ep(bundle.dihedral_params, db, 0.0),
        dihedral_mask=_pad_axis0_ep(bundle.dihedral_mask, db, False),
        improper_idx=_pad_axis0_ep(bundle.improper_idx, db, 0),
        improper_params=_pad_axis0_ep(bundle.improper_params, db, 0.0),
        improper_mask=_pad_axis0_ep(bundle.improper_mask, db, False),
        urey_bradley_idx=_pad_axis0_ep(bundle.urey_bradley_idx, ab, 0),
        urey_bradley_params=_pad_axis0_ep(bundle.urey_bradley_params, ab, 0.0),
        urey_bradley_mask=_pad_axis0_ep(bundle.urey_bradley_mask, ab, False),
        cmap_torsion_idx=_pad_axis0_ep(bundle.cmap_torsion_idx, cb, 0),
        cmap_energy_grids=_pad_axis0_ep(bundle.cmap_energy_grids, cb, 0.0),
        cmap_mask=_pad_axis0_ep(bundle.cmap_mask, cb, False),
        cmap_indices=_pad_axis0_ep(bundle.cmap_indices, cb, 0),
        cmap_coeffs=_pad_axis0_ep(bundle.cmap_coeffs, mb, 0.0),
        water_indices=_pad_axis0_ep(bundle.water_indices, wb, 0),
        water_mask=_pad_axis0_ep(bundle.water_mask, wb, False),
        constraint_pairs=c_pairs,
        constraint_lengths=c_lens,
        constraint_mask=c_mask,
        excl_indices=_pad_axis0_ep(bundle.excl_indices, eb, 0),
        excl_scales_vdw=_pad_axis0_ep(bundle.excl_scales_vdw, eb, 0.0),
        excl_scales_elec=_pad_axis0_ep(bundle.excl_scales_elec, eb, 0.0),
        excl_mask=_pad_axis0_ep(bundle.excl_mask, eb, False),
        excl_dense_indices=excl_di,
        excl_dense_scales_vdw=excl_sv,
        excl_dense_scales_elec=excl_se,
        exception_pairs=_pad_axis0_ep(bundle.exception_pairs, xb, 0),
        exception_sigmas=_pad_axis0_ep(bundle.exception_sigmas, xb, 0.0),
        exception_epsilons=_pad_axis0_ep(bundle.exception_epsilons, xb, 0.0),
        exception_chargeprods=_pad_axis0_ep(bundle.exception_chargeprods, xb, 0.0),
        exception_mask=_pad_axis0_ep(bundle.exception_mask, xb, False),
        molecule_id=mol_id,
        cosolvent_group_id=_pad_axis0_ep(bundle.cosolvent_group_id, a, 0),
        shape_spec=target_spec,
    )


def _align_bundles_for_stacking(bundles: Sequence[Any]) -> list[Any]:
    """Ensure every array field across ``bundles`` shares identical axis-0 shape for stacking."""
    if len(bundles) <= 1:
        return list(bundles)
    u_spec = _union_shape_spec([b.shape_spec for b in bundles])
    max_a = max(int(b.positions.shape[0]) for b in bundles)
    max_bb = max(int(b.bond_idx.shape[0]) for b in bundles)
    max_ub = max(int(b.urey_bradley_idx.shape[0]) for b in bundles)
    max_ab = max(int(b.angle_idx.shape[0]) for b in bundles)
    max_db = max(int(b.dihedral_idx.shape[0]) for b in bundles)
    max_ib = max(int(b.improper_idx.shape[0]) for b in bundles)
    max_wb = max(int(b.water_indices.shape[0]) if b.water_indices is not None else 0 for b in bundles)
    max_eb = max(int(b.excl_indices.shape[0]) for b in bundles)
    max_cb = max(int(b.cmap_torsion_idx.shape[0]) for b in bundles)
    max_mb = max(int(b.cmap_coeffs.shape[0]) for b in bundles)
    max_xb = max(int(b.exception_pairs.shape[0]) for b in bundles)
    max_cons = max(int(b.constraint_pairs.shape[0]) if b.constraint_pairs is not None else 0 for b in bundles)
    out_list: list[Any] = []
    for b in bundles:
        n_real = int(jnp.asarray(b.n_atoms))
        pos_p = (
            _pad_positions_to_bucket_ep(b.positions[:n_real], max_a, b.box if u_spec.has_pbc else None)
            if int(b.positions.shape[0]) != max_a
            else b.positions
        )
        cp_in, cl_in, cm_in = b.constraint_pairs, b.constraint_lengths, b.constraint_mask
        c_p = (
            jnp.zeros((0, 2), dtype=jnp.int32)
            if max_cons == 0
            else (_pad_axis0_ep(cp_in, max_cons, 0) if cp_in is not None and cp_in.size > 0 else jnp.zeros((max_cons, 2), dtype=jnp.int32))
        )
        c_l = (
            jnp.zeros((0,), dtype=jnp.float32)
            if max_cons == 0
            else (_pad_axis0_ep(cl_in, max_cons, 0.0) if cl_in is not None and cl_in.size > 0 else jnp.zeros((max_cons,), dtype=jnp.float32))
        )
        c_m = (
            jnp.zeros((0,), dtype=bool)
            if max_cons == 0
            else (_pad_axis0_ep(cm_in, max_cons, False) if cm_in is not None and cm_in.size > 0 else jnp.zeros((max_cons,), dtype=bool))
        )
        mol_id = b.molecule_id
        if mol_id is not None and int(mol_id.shape[0]) != max_a:
            cur_a = int(mol_id.shape[0])
            mol_id = jnp.concatenate([mol_id, jnp.arange(cur_a, max_a, dtype=jnp.int32) + jnp.max(mol_id) + 1], axis=0) if cur_a < max_a else mol_id[:max_a]
        out_list.append(
            dataclasses.replace(
                b,
                positions=pos_p,
                masses=_pad_axis0_ep(b.masses, max_a, 1.0),
                charges=_pad_axis0_ep(b.charges, max_a, 0.0),
                sigmas=_pad_axis0_ep(b.sigmas, max_a, 0.0),
                epsilons=_pad_axis0_ep(b.epsilons, max_a, 0.0),
                radii=_pad_axis0_ep(b.radii, max_a, 0.0),
                scaled_radii=_pad_axis0_ep(b.scaled_radii, max_a, 0.0),
                atom_mask=_pad_axis0_ep(b.atom_mask, max_a, False),
                bond_idx=_pad_axis0_ep(b.bond_idx, max_bb, 0),
                bond_params=_pad_axis0_ep(b.bond_params, max_bb, 0.0),
                bond_mask=_pad_axis0_ep(b.bond_mask, max_bb, False),
                angle_idx=_pad_axis0_ep(b.angle_idx, max_ab, 0),
                angle_params=_pad_axis0_ep(b.angle_params, max_ab, 0.0),
                angle_mask=_pad_axis0_ep(b.angle_mask, max_ab, False),
                dihedral_idx=_pad_axis0_ep(b.dihedral_idx, max_db, 0),
                dihedral_params=_pad_axis0_ep(b.dihedral_params, max_db, 0.0),
                dihedral_mask=_pad_axis0_ep(b.dihedral_mask, max_db, False),
                improper_idx=_pad_axis0_ep(b.improper_idx, max_ib, 0),
                improper_params=_pad_axis0_ep(b.improper_params, max_ib, 0.0),
                improper_mask=_pad_axis0_ep(b.improper_mask, max_ib, False),
                urey_bradley_idx=_pad_axis0_ep(b.urey_bradley_idx, max_ub, 0),
                urey_bradley_params=_pad_axis0_ep(b.urey_bradley_params, max_ub, 0.0),
                urey_bradley_mask=_pad_axis0_ep(b.urey_bradley_mask, max_ub, False),
                cmap_torsion_idx=_pad_axis0_ep(b.cmap_torsion_idx, max_cb, 0),
                cmap_energy_grids=_pad_axis0_ep(b.cmap_energy_grids, max_cb, 0.0),
                cmap_mask=_pad_axis0_ep(b.cmap_mask, max_cb, False),
                cmap_indices=_pad_axis0_ep(b.cmap_indices, max_cb, 0),
                cmap_coeffs=_pad_axis0_ep(b.cmap_coeffs, max_mb, 0.0),
                water_indices=_pad_axis0_ep(b.water_indices, max_wb, 0) if max_wb > 0 else jnp.zeros((0, 3), dtype=jnp.int32),
                water_mask=_pad_axis0_ep(b.water_mask, max_wb, False) if max_wb > 0 else jnp.zeros((0,), dtype=bool),
                constraint_pairs=c_p,
                constraint_lengths=c_l,
                constraint_mask=c_m,
                excl_indices=_pad_axis0_ep(b.excl_indices, max_eb, 0),
                excl_scales_vdw=_pad_axis0_ep(b.excl_scales_vdw, max_eb, 0.0),
                excl_scales_elec=_pad_axis0_ep(b.excl_scales_elec, max_eb, 0.0),
                excl_mask=_pad_axis0_ep(b.excl_mask, max_eb, False),
                excl_dense_indices=_pad_axis0_ep(b.excl_dense_indices, max_a, -1),
                excl_dense_scales_vdw=_pad_axis0_ep(b.excl_dense_scales_vdw, max_a, 1.0),
                excl_dense_scales_elec=_pad_axis0_ep(b.excl_dense_scales_elec, max_a, 1.0),
                exception_pairs=_pad_axis0_ep(b.exception_pairs, max_xb, 0),
                exception_sigmas=_pad_axis0_ep(b.exception_sigmas, max_xb, 0.0),
                exception_epsilons=_pad_axis0_ep(b.exception_epsilons, max_xb, 0.0),
                exception_chargeprods=_pad_axis0_ep(b.exception_chargeprods, max_xb, 0.0),
                exception_mask=_pad_axis0_ep(b.exception_mask, max_xb, False),
                molecule_id=mol_id,
                cosolvent_group_id=_pad_axis0_ep(b.cosolvent_group_id, max_a, 0),
                shape_spec=u_spec,
            )
        )
    return out_list


def _finite_nl_cutoff(cutoff_raw: float, box: Any = None, default: float = 9.0) -> float:
    """Ensure neighbor-list cutoff is finite, positive, and within half the minimum box height."""
    val = float(cutoff_raw)
    if not math.isfinite(val) or val <= 0.0:
        if box is not None and not isinstance(box, jax.core.Tracer):
            box_np = np.asarray(box, dtype=np.float64)
            diag_np = np.diag(box_np) if box_np.ndim == 2 else box_np
            min_h = float(np.min(diag_np))
            if math.isfinite(min_h) and min_h > 0.0:
                return min(float(default), 0.45 * min_h)
        return float(default)
    return val


def _cached_bundle_cutoff(bundle: Any, default: float = 9.0) -> float:
    bid = id(bundle)
    cached = _BUNDLE_CUTOFF_CACHE.get(bid)
    if cached is not None:
        return cached
    raw = getattr(bundle, "cutoff_distance", default)
    if isinstance(raw, jax.core.Tracer):
        return float(default)
    val = _finite_nl_cutoff(_host_float(raw, default), getattr(bundle, "box", None), default)
    if len(_BUNDLE_CUTOFF_CACHE) >= 256:
        _BUNDLE_CUTOFF_CACHE.pop(next(iter(_BUNDLE_CUTOFF_CACHE)))
    _BUNDLE_CUTOFF_CACHE[bid] = val
    return val


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
            env_key = (n_pad_i, tuple(id(b) for b in bundles))
            cached_aw = _ACTIVE_ATOM_ENV_CACHE.get(env_key)
            if cached_aw is not None:
                clamped_a_opt, clamped_w_opt = cached_aw
            else:
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
                clamped_a_opt = None
                clamped_w_opt = None
                if (is_implicit and 0 < max_real_a < n_pad_i) or (
                    n_pad_i >= 4096 and 0 < max_real_a <= int(0.75 * n_pad_i)
                ):
                    gran = 32 if is_implicit else 1024
                    clamped_a = min(n_pad_i, max(gran, ((max_real_a + gran - 1) // gran) * gran))
                    if clamped_a < n_pad_i:
                        clamped_a_opt = clamped_a
                    if not is_implicit:
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
                                clamped_w_opt = min(max_pad_w, max(512, ((max_real_w + 511) // 512) * 512))
                if len(_ACTIVE_ATOM_ENV_CACHE) >= 256:
                    _ACTIVE_ATOM_ENV_CACHE.pop(next(iter(_ACTIVE_ATOM_ENV_CACHE)))
                _ACTIVE_ATOM_ENV_CACHE[env_key] = (clamped_a_opt, clamped_w_opt)

            if clamped_a_opt is not None:
                os.environ["PROLIX_ACTIVE_ATOMS"] = str(clamped_a_opt)
                did_set_a = True
            if clamped_w_opt is not None and not os.environ.get("PROLIX_ACTIVE_WATERS", "").strip():
                os.environ["PROLIX_ACTIVE_WATERS"] = str(clamped_w_opt)
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


def _compute_morton_npt_vol_constants(
    *,
    box_b: np.ndarray | jax.Array,
    sig_b: jax.Array | np.ndarray,
    eps_b: jax.Array | np.ndarray,
    q_b: jax.Array | np.ndarray,
    am_b: jax.Array | np.ndarray,
    cutoff: float,
    lj_switch_width: float = 0.0,
    combining_rule: str = "lorentz_berthelot",
    use_dispersion_correction: bool = True,
    do_pme: bool = True,
    pme_alpha: float = 0.34,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute volume-invariant NPT barostat constants ``(c_tail_b, c_net_q_b)``.

    Both isotropic long-range LJ dispersion tail energy ``E_tail(V) = c_tail / V`` and
    SPME neutralizing background energy ``E_bg(V) = c_net_q / V`` scale strictly as ``1 / V``,
    so ``Delta(E_tail + E_bg) = (c_tail + c_net_q) * (1 / V_new - 1 / V_old)`` without O(N)
    recomputation on each barostat trial.
    """
    box_arr = np.asarray(box_b, dtype=np.float64)
    B = int(box_arr.shape[0])
    c_tail = np.zeros((B,), dtype=np.float64)
    c_net_q = np.zeros((B,), dtype=np.float64)

    sig_params = inspect.signature(explicit_corrections.lj_dispersion_tail_energy).parameters
    tail_kwargs: dict[str, Any] = {}
    if "lj_switch_width" in sig_params:
        tail_kwargs["lj_switch_width"] = float(lj_switch_width)
    if "combining_rule" in sig_params:
        tail_kwargs["combining_rule"] = str(combining_rule)

    alpha_f = float(pme_alpha)
    bg_prefactor = (
        -(math.pi * float(COULOMB_CONSTANT) / (2.0 * alpha_f * alpha_f))
        if (do_pme and alpha_f > 0.0)
        else 0.0
    )

    tail_dt = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    for b in range(B):
        bx_one = np.diag(box_arr[b]) if box_arr[b].ndim == 2 else box_arr[b]
        v0 = float(np.prod(bx_one))
        if use_dispersion_correction and v0 > 0.0:
            sig_j = jnp.asarray(sig_b[b], dtype=tail_dt)
            eps_j = jnp.asarray(eps_b[b], dtype=tail_dt)
            msk_b = jnp.asarray(np.asarray(am_b[b], dtype=bool))
            e_t0 = float(
                np.asarray(
                    explicit_corrections.lj_dispersion_tail_energy(
                        jnp.asarray(bx_one, dtype=tail_dt),
                        sig_j,
                        eps_j,
                        float(cutoff),
                        msk_b,
                        **tail_kwargs,
                    )
                )
            )
            c_tail[b] = e_t0 * v0
        if do_pme and alpha_f > 0.0:
            q_np = np.asarray(q_b[b], dtype=np.float64)
            m_np = np.asarray(am_b[b], dtype=bool)
            q_net = float(np.sum(np.where(m_np, q_np, 0.0)))
            c_net_q[b] = bg_prefactor * (q_net * q_net)

    return c_tail, c_net_q


def _compute_mc_barostat_dh(
    e_old: jax.Array,
    e_new: jax.Array,
    v_old: jax.Array,
    v_new: jax.Array,
    *,
    c_tail_b: jax.Array,
    c_net_q_b: jax.Array,
    p0_akma: float | jax.Array,
    n_mols: float | jax.Array,
    kT: jax.Array,
    dlnv: jax.Array | None = None,
) -> jax.Array:
    """Compute exact isotropic MC barostat acceptance exponent ``Delta H``."""
    if dlnv is None:
        dlnv = jnp.log(v_new / v_old)
    de_vol_corr = (c_tail_b + c_net_q_b) * (1.0 / v_new - 1.0 / v_old)
    out_dt = e_new.dtype
    return (
        (e_new - e_old)
        + de_vol_corr.astype(out_dt)
        + jnp.asarray(p0_akma, dtype=out_dt) * (v_new - v_old).astype(out_dt)
        - jnp.asarray(n_mols, dtype=out_dt) * kT.astype(out_dt) * dlnv.astype(out_dt)
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
        nonbonded_engine: str | None = None,
        precision: str | None = None,
        nb_k_splits: int | None = None,
        tp_group: int | None = None,
        cmm_frequency: int | None = None,
        deterministic_forces: bool | None = None,
        gb_use_neighbor_list: bool | None = None,
        pme_tier_mode: str | None = None,
        pressure_atm: float | None = None,
        barostat_interval: int | None = None,
        knob_surface: KnobSurface | None = None,
    ) -> None:
        if not isinstance(bundles, (list, tuple)):
            bundles = [bundles]
        self.bundles = list(bundles)
        self.knob_surface = knob_surface
        ks_pallas = knob_surface.pallas if knob_surface is not None else None
        ks_thermo = knob_surface.thermostat if knob_surface is not None else None
        ks_baro = knob_surface.barostat if knob_surface is not None else None

        self.use_pallas = (
            use_pallas
            if use_pallas is not None
            else (ks_pallas.use_pallas if ks_pallas is not None else None)
        )
        self.use_pallas_pme = (
            use_pallas_pme
            if use_pallas_pme is not None
            else (ks_pallas.use_pallas_pme if ks_pallas is not None else None)
        )
        self.nonbonded_engine = (
            str(nonbonded_engine)
            if nonbonded_engine is not None
            else (ks_pallas.nonbonded_engine if ks_pallas is not None else "auto")
        )
        self.precision = (
            str(precision)
            if precision is not None
            else (ks_pallas.precision if ks_pallas is not None else None)
        )
        self.nb_k_splits = (
            int(nb_k_splits)
            if nb_k_splits is not None
            else (ks_pallas.nb_k_splits if ks_pallas is not None else None)
        )
        self.tp_group = (
            int(tp_group)
            if tp_group is not None
            else (ks_pallas.tp_group if ks_pallas is not None else None)
        )
        self.cmm_frequency = (
            int(cmm_frequency)
            if cmm_frequency is not None
            else (ks_thermo.cmm_frequency if ks_thermo is not None else 0)
        )
        self.deterministic_forces = (
            bool(deterministic_forces)
            if deterministic_forces is not None
            else (ks_pallas.deterministic_forces if ks_pallas is not None else False)
        )
        self.gb_use_neighbor_list = (
            bool(gb_use_neighbor_list)
            if gb_use_neighbor_list is not None
            else (ks_pallas.gb_use_neighbor_list if ks_pallas is not None else False)
        )
        self.pme_tier_mode = str(pme_tier_mode) if pme_tier_mode is not None else "auto"
        self.pressure_atm = (
            float(pressure_atm)
            if pressure_atm is not None
            else (ks_baro.pressure_atm if ks_baro is not None else None)
        )
        self.barostat_interval = (
            int(barostat_interval)
            if barostat_interval is not None
            else (ks_baro.barostat_interval if ks_baro is not None else 25)
        )
        if planner is None and len(self.bundles) > 1:
            planner = EnsembleMDPlanner()
        if planner is not None:
            self.batch_plan = planner.plan(self.bundles)
        else:
            self.batch_plan = None

    def _default_v35_knobs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        knobs: dict[str, Any] = {
            "nonbonded_engine": getattr(self, "nonbonded_engine", "auto"),
            "precision": getattr(self, "precision", None),
            "nb_k_splits": getattr(self, "nb_k_splits", None),
            "tp_group": getattr(self, "tp_group", None),
            "cmm_frequency": int(getattr(self, "cmm_frequency", 0) or 0),
            "deterministic_forces": bool(getattr(self, "deterministic_forces", False)),
            "gb_use_neighbor_list": bool(getattr(self, "gb_use_neighbor_list", False)),
            "pme_tier_mode": str(getattr(self, "pme_tier_mode", "auto")),
            "pressure_atm": getattr(self, "pressure_atm", None),
            "barostat_interval": int(getattr(self, "barostat_interval", 25) or 25),
        }
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    knobs[k] = v
        return knobs

    def _current_v35_knobs(self) -> dict[str, Any]:
        active = getattr(self, "_active_v35_knobs", None)
        return dict(active) if active is not None else self._default_v35_knobs()

    def _can_use_morton_soa_runner(  # noqa: PLR0911
        self,
        bundles: Sequence[Any],
        *,
        use_neighbor_list: bool,
        use_flash_forces: bool,
        xtc_path: Any = None,
        initial_neighbor: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> bool:
        if not bundles or use_flash_forces or xtc_path is not None or initial_neighbor is not None:
            return False
        if in_export_context():
            return False
        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        if prec == "double":
            return False
        engine = str(knobs.get("nonbonded_engine", "auto")).strip().lower()
        if engine not in ("auto", "morton_tile_pairs"):
            return False
        if use_pallas is False or use_pallas_pme is False:
            return False
        env_up = os.environ.get("PROLIX_USE_PALLAS", "").strip().lower()
        if env_up in ("0", "false", "no"):
            return False
        if os.environ.get("PROLIX_PME_EVERY", "").strip() not in ("", "1"):
            return False
        if engine == "auto":
            if not use_neighbor_list or jax.default_backend() == "cpu":
                return False
            if self.knob_surface is not None and self.knob_surface.nonbonded is not None:
                comb_m = (
                    str(getattr(self.knob_surface.nonbonded, "lj_combining_rule", "lorentz_berthelot"))
                    .strip()
                    .lower()
                    .replace("-", "_")
                )
                if comb_m != "lorentz_berthelot":
                    return False
            cur_pme_pallas = sys.modules.get("prolix.physics.pme_pallas", pme_pallas_mod)
            if (
                nl_mod.make_neighbor_list_fn is not _ORIG_MAKE_NL_FN
                or _bundle_md_mod.single_padded_energy is not _ORIG_BMD_SINGLE_ENERGY
                or _bundle_md_mod.pallas_bonded_all_fused is not _ORIG_BMD_BONDED_FUSED
                or getattr(cur_pme_pallas._spread_charges_pallas_single, "__name__", "") != "_spread_charges_pallas_single"
                or getattr(cur_pme_pallas._gather_pallas_single, "__name__", "") != "_gather_pallas_single"
            ):
                return False
        n_reals: list[int] = []
        for b in bundles:
            sp = getattr(b, "shape_spec", None)
            if sp is None or not sp.has_pbc or sp.has_implicit_solvent:
                return False
            if getattr(sp, "is_triclinic", False) or getattr(sp, "has_real_cosolvents", False):
                return False
            if (
                isinstance(b.positions, jax.core.Tracer)
                or isinstance(b.charges, jax.core.Tracer)
                or getattr(b, "box", None) is None
                or isinstance(b.box, jax.core.Tracer)
                or getattr(b, "excl_dense_indices", None) is None
            ):
                return False
            bx_np = np.asarray(b.box, dtype=np.float64)
            if bx_np.ndim == 2:
                off = bx_np - np.diag(np.diag(bx_np))
                if np.max(np.abs(off)) > 1e-6 or np.any(np.diag(bx_np) <= 0.0):
                    return False
            elif bx_np.ndim != 1 or bx_np.shape[0] != 3 or np.any(bx_np <= 0.0):
                return False
            n_reals.append(int(np.asarray(b.n_atoms)))
        if len(bundles) > 1:
            if min(n_reals) * 1.5 < max(n_reals):
                return False
            if min(n_reals) != max(n_reals) and str(knobs.get("pme_tier_mode", "auto")).lower() != "auto":
                return False
            ei0 = bundles[0].excl_dense_indices
            ei0_np = np.asarray(ei0)
            bx0_np = np.diag(np.asarray(bundles[0].box, dtype=np.float64)) if bundles[0].box.ndim == 2 else np.asarray(bundles[0].box, dtype=np.float64)
            for b in bundles[1:]:
                if b.excl_dense_indices is not ei0 and not np.array_equal(np.asarray(b.excl_dense_indices), ei0_np):
                    return False
                bxi = np.diag(np.asarray(b.box, dtype=np.float64)) if b.box.ndim == 2 else np.asarray(b.box, dtype=np.float64)
                if not np.allclose(bxi, bx0_np, atol=1e-5):
                    return False
        return True

    def _can_use_ragged_mk68_runner(  # noqa: PLR0911
        self,
        bundles: Sequence[Any],
        *,
        use_neighbor_list: bool,
        use_flash_forces: bool,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> bool:
        if not bundles or use_flash_forces or in_export_context():
            return False
        knobs = self._current_v35_knobs()
        if str(knobs.get("precision") or "").strip().lower() == "double":
            return False
        engine = str(knobs.get("nonbonded_engine", "auto")).strip().lower()
        if engine not in ("auto", "ragged_tile_pairs"):
            return False
        if use_pallas is False or use_pallas_pme is False:
            return False
        if os.environ.get("PROLIX_USE_PALLAS", "").strip().lower() in ("0", "false", "no"):
            return False
        n_reals: list[int] = []
        for b in bundles:
            sp = getattr(b, "shape_spec", None)
            if sp is None or not sp.has_pbc or sp.has_implicit_solvent or getattr(sp, "is_triclinic", False):
                return False
            if isinstance(b.positions, jax.core.Tracer) or getattr(b, "box", None) is None or isinstance(b.box, jax.core.Tracer):
                return False
            if getattr(b, "excl_dense_indices", None) is None:
                return False
            n_reals.append(int(np.asarray(b.n_atoms)))
        if engine == "ragged_tile_pairs":
            return True
        if len(bundles) < 2 or not use_neighbor_list or jax.default_backend() == "cpu":
            return False
        if (
            nl_mod.make_neighbor_list_fn is not _ORIG_MAKE_NL_FN
            or _bundle_md_mod.single_padded_energy is not _ORIG_BMD_SINGLE_ENERGY
        ):
            return False
        bx0_np = (
            np.diag(np.asarray(bundles[0].box, dtype=np.float64))
            if bundles[0].box.ndim == 2
            else np.asarray(bundles[0].box, dtype=np.float64)
        )
        boxes_differ = any(
            not np.allclose(
                np.diag(np.asarray(b.box, dtype=np.float64)) if b.box.ndim == 2 else np.asarray(b.box, dtype=np.float64),
                bx0_np,
                atol=1e-5,
            )
            for b in bundles[1:]
        )
        return bool(
            boxes_differ
            or min(n_reals) * 1.5 < max(n_reals)
            or (min(n_reals) != max(n_reals) and str(knobs.get("pme_tier_mode", "auto")).lower() != "auto")
        )

    def _build_morton_soa_runner(
        self,
        bundles: Sequence[Any],
        *,
        integration_prefix: int | None,
        gamma: float,
        nl_update_every: int,
        nb_cfg: Any,
    ) -> dict[str, Any]:
        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        if prec == "mixed":
            if not bool(jax.config.read("jax_enable_x64")):
                jax.config.update("jax_enable_x64", True)
            state_dtype: Any = jnp.float64
            f64_acc = True
        elif prec == "single":
            state_dtype = jnp.float32
            f64_acc = False
        else:
            state_dtype = bundles[0].positions.dtype
            f64_acc = state_dtype == jnp.float64

        interpret = jax.default_backend() == "cpu"
        B = len(bundles)
        N = int(integration_prefix) if integration_prefix is not None else int(bundles[0].positions.shape[0])
        n_reals = [int(np.asarray(b.n_atoms)) for b in bundles]
        n_act = max(1, min(N, max(n_reals)))
        n_act_blocks = max(1, (n_act + 31) // 32)
        n_act_pad = n_act_blocks * 32
        n_waters_real = max(
            (int(np.sum(np.asarray(b.water_mask))) if b.shape_spec.has_real_water else 0)
            for b in bundles
        )

        pos_init_b = jnp.stack([positions_with_prefix(b, N).astype(state_dtype) for b in bundles], axis=0)
        mass_b = jnp.stack([masses_with_prefix(b, N).astype(state_dtype) for b in bundles], axis=0)
        am_b = jnp.stack([b.atom_mask[:N].astype(jnp.bool_) for b in bundles], axis=0)
        q_b = jnp.stack([b.charges[:N].astype(jnp.float32) for b in bundles], axis=0)
        sig_b = jnp.stack([b.sigmas[:N].astype(jnp.float32) for b in bundles], axis=0)
        eps_b = jnp.stack([b.epsilons[:N].astype(jnp.float32) for b in bundles], axis=0)

        if any(b.shape_spec.has_real_water for b in bundles):
            wi_b = jnp.stack([b.water_indices.astype(jnp.int32) for b in bundles], axis=0)
            wm_b = jnp.stack([b.water_mask.astype(jnp.int32) for b in bundles], axis=0)
        else:
            wi_b = jnp.zeros((B, 0, 3), dtype=jnp.int32)
            wm_b = jnp.zeros((B, 0), dtype=jnp.int32)

        has_cons = bool(
            any(getattr(b.shape_spec, "has_real_constraints", False) for b in bundles)
            and bundles[0].constraint_pairs is not None
            and int(bundles[0].constraint_pairs.shape[0]) > 0
        )
        cp_b = jnp.stack([b.constraint_pairs.astype(jnp.int32) for b in bundles], axis=0) if has_cons else None
        cl_b = jnp.stack([b.constraint_lengths.astype(state_dtype) for b in bundles], axis=0) if has_cons else None
        cm_b = jnp.stack([b.constraint_mask.astype(jnp.int32) for b in bundles], axis=0) if has_cons else None

        soa_plan = build_baoab_soa_static_plan(
            B, N, state_dtype, mass_b, wi_b, wm_b,
            atom_mask_b=am_b, constraint_pairs_b=cp_b, constraint_lengths_b=cl_b, constraint_mask_b=cm_b,
            n_active_atoms=n_act, n_active_waters=n_waters_real,
        )

        cutoff = _cached_bundle_cutoff(bundles[0], 9.0)
        nlc = getattr(nb_cfg, "neighbor_list", None) if nb_cfg is not None else None
        eff_skin = max(float(nlc.skin) if nlc is not None else 2.0, 2.0)
        thresh_sq = float((0.5 * eff_skin) ** 2)
        lj_sw = float(lj_switch_width_from_config(nb_cfg))
        tp_grp = knobs.get("tp_group")
        det_forces = bool(knobs.get("deterministic_forces", False))
        cmm_freq = int(knobs.get("cmm_frequency", 0) or 0)
        press_atm = knobs.get("pressure_atm")
        baro_int = max(1, int(knobs.get("barostat_interval", 25) or 25))
        chunk_k = max(1, int(nl_update_every) if int(nl_update_every) > 0 else 20)
        if press_atm is not None:
            chunk_k = min(chunk_k, baro_int)

        box0_np = (
            np.diag(np.asarray(bundles[0].box, dtype=np.float64))
            if bundles[0].box.ndim == 2
            else np.asarray(bundles[0].box, dtype=np.float64)
        )
        host_min_box = float(np.min(box0_np))
        box_safety = DEFAULT_NPT_BOX_SAFETY if press_atm is not None else 1.0
        eff_host_min_box = host_min_box / box_safety
        box_init_b = jnp.broadcast_to(jnp.asarray(box0_np, dtype=state_dtype)[None, :], (B, 3))
        max_pairs = estimate_max_tile_pairs(
            n_act_blocks, cutoff=cutoff, skin=eff_skin, box_safety=box_safety
        )
        dedup_excl = precompute_dedup_excl_edges(
            bundles[0].excl_dense_indices,
            bundles[0].excl_dense_scales_vdw,
            bundles[0].excl_dense_scales_elec,
            n_act,
        )

        def _stack_or_dedup(attr: str) -> Any:
            v0 = getattr(bundles[0], attr, None)
            if v0 is None:
                return None
            if B == 1:
                return v0
            v0_np = np.asarray(v0)
            for b_o in bundles[1:]:
                vb = getattr(b_o, attr, None)
                if vb is not v0 and (vb is None or not np.array_equal(v0_np, np.asarray(vb))):
                    return jnp.stack([getattr(bx, attr) for bx in bundles], axis=0)
            return v0

        imp_harm = bool(bundles[0].improper_params is not None and bundles[0].improper_params.shape[-1] == 2)
        bonded_bufs = prepare_fused_bonded_buffers(
            B,
            bonds=_stack_or_dedup("bond_idx"),
            bond_params=_stack_or_dedup("bond_params"),
            bond_mask=_stack_or_dedup("bond_mask"),
            urey_bradley_bonds=_stack_or_dedup("urey_bradley_idx"),
            urey_bradley_params=_stack_or_dedup("urey_bradley_params"),
            urey_bradley_mask=_stack_or_dedup("urey_bradley_mask"),
            angles=_stack_or_dedup("angle_idx"),
            angle_params=_stack_or_dedup("angle_params"),
            angle_mask=_stack_or_dedup("angle_mask"),
            dihedrals=_stack_or_dedup("dihedral_idx"),
            dihedral_params=_stack_or_dedup("dihedral_params"),
            dihedral_mask=_stack_or_dedup("dihedral_mask"),
            impropers=_stack_or_dedup("improper_idx"),
            improper_params=_stack_or_dedup("improper_params"),
            improper_mask=_stack_or_dedup("improper_mask"),
            improper_is_harmonic=imp_harm,
            exception_pairs=_stack_or_dedup("exception_pairs"),
            exception_sigmas=_stack_or_dedup("exception_sigmas"),
            exception_epsilons=_stack_or_dedup("exception_epsilons"),
            exception_chargeprods=_stack_or_dedup("exception_chargeprods"),
            exception_mask=_stack_or_dedup("exception_mask"),
            cmap_torsions=_stack_or_dedup("cmap_torsion_idx"),
            cmap_indices=_stack_or_dedup("cmap_indices"),
            cmap_coeffs=_stack_or_dedup("cmap_coeffs"),
            cmap_mask=_stack_or_dedup("cmap_mask"),
        )
        ks_nb = self.knob_surface.nonbonded if self.knob_surface is not None else None
        pme_alpha_raw = (
            float(ks_nb.pme_alpha)
            if ks_nb is not None
            else float(_host_float(getattr(bundles[0], "pme_alpha", 0.34), 0.34))
        )
        elec_method_raw = (
            str(ks_nb.nonbonded_method)
            if ks_nb is not None
            else str(
                getattr(bundles[0], "electrostatic_method", "pme" if pme_alpha_raw > 0.0 else "cutoff")
            )
        ).strip().lower().replace("-", "_")
        if elec_method_raw in ("reaction_field", "rf", "cutoff_periodic", "cutoffperiodic"):
            do_pme = False
            pme_alpha = 0.0
            rf_diel: float | None = float(
                ks_nb.reaction_field_dielectric
                if ks_nb is not None
                else getattr(bundles[0], "reaction_field_dielectric", 78.3)
            )
        elif elec_method_raw in (
            "cutoff",
            "no_cutoff",
            "nocutoff",
            "cutoff_non_periodic",
            "cutoffnonperiodic",
            "coulomb",
        ):
            do_pme = False
            pme_alpha = 0.0
            rf_diel = None
        else:
            pme_alpha = pme_alpha_raw
            do_pme = pme_alpha > 0.0
            rf_diel = None
        use_disp = bool(
            ks_nb.dispersion_correction
            if ks_nb is not None
            else getattr(bundles[0], "use_dispersion_correction", True)
        )
        comb_rule = str(
            getattr(ks_nb, "lj_combining_rule", "lorentz_berthelot")
            if ks_nb is not None
            else getattr(bundles[0], "lj_combining_rule", "lorentz_berthelot")
        )
        if do_pme:
            gd_env = _resolve_env_pme_grid_dims(n_atoms=N)
            if gd_env is not None:
                grid_dims = (int(gd_env[0]), int(gd_env[1]), int(gd_env[2]))
            else:
                gp = int(_host_float(getattr(bundles[0], "pme_grid_points", 64), 64))
                grid_dims = compute_pme_grid_dims(
                    jnp.asarray(box0_np, dtype=jnp.float32), float(np.mean(box0_np)) / float(max(gp, 1))
                )
            G_init = jnp.broadcast_to(
                jnp.asarray(
                    _cached_host_influence_function_np(
                        grid_dims,
                        tuple(float(x) for x in box0_np),
                        (3,),
                        "float32",
                        pme_alpha,
                        4,
                    ),
                    dtype=jnp.float32,
                )[None, ...],
                (B, grid_dims[0], grid_dims[1], grid_dims[2] // 2 + 1),
            )
            n_grid_f = float(grid_dims[0] * grid_dims[1] * grid_dims[2])
            pme_force_scale = -float(COULOMB_CONSTANT) * n_grid_f
            e_self_b = spme_self_energy(q_b, am_b, alpha=pme_alpha).astype(jnp.float32)
        else:
            grid_dims = (8, 8, 8)
            G_init = jnp.zeros((B, 1, 1, 1), dtype=jnp.float32)
            n_grid_f = 1.0
            pme_force_scale = 0.0
            e_self_b = jnp.zeros((B,), dtype=jnp.float32)

        gamma_akma = float(gamma_ps_to_akma(float(gamma)))
        lam_flat = jnp.ones((1,), dtype=jnp.float32)
        in_b_pre = (True,) * (12 + (3 if has_cons else 0))

        # Molecular anchor map for rigid-safe isotropic MC barostat (MK-74)
        mol_id_0 = getattr(bundles[0], "molecule_id", None)
        if mol_id_0 is not None and int(mol_id_0.shape[0]) >= N:
            mid_np = np.asarray(mol_id_0[:N], dtype=np.int32)
            first_atom_by_mol: dict[int, int] = {}
            anchor_np = np.arange(N, dtype=np.int32)
            for ai in range(n_act):
                m_val = int(mid_np[ai])
                if m_val not in first_atom_by_mol:
                    first_atom_by_mol[m_val] = ai
                anchor_np[ai] = first_atom_by_mol[m_val]
            n_mols_baro = max(1, len(first_atom_by_mol))
        else:
            anchor_np = np.arange(N, dtype=np.int32)
            if wi_b.shape[1] > 0:
                w0_np = np.asarray(wi_b[0])
                wm0_np = np.asarray(wm_b[0])
                for wi_k in range(w0_np.shape[0]):
                    if wm0_np[wi_k] > 0:
                        o_i, h1_i, h2_i = int(w0_np[wi_k, 0]), int(w0_np[wi_k, 1]), int(w0_np[wi_k, 2])
                        anchor_np[h1_i] = o_i
                        anchor_np[h2_i] = o_i
            n_mols_baro = max(1, len(set(anchor_np[:n_act].tolist())))
        anchor_idx_j = jnp.asarray(anchor_np, dtype=jnp.int32)

        def _inv_for_n(inv_pad: Any) -> Any:
            pw = int(inv_pad.shape[1])
            return inv_pad[:, :N] if pw >= N else jnp.pad(inv_pad, ((0, 0), (0, N - pw)))

        def _eval_morton_forces(
            px_m: Any, py_m: Any, pz_m: Any, tp_st: Any, box_b_cur: Any, G_b_cur: Any, compute_energy: bool = False
        ) -> tuple[Any, Any, Any, Any]:
            box_f32_flat = box_b_cur.astype(jnp.float32).reshape(B * 3)
            e_dir, fx_d, fy_d, fz_d = _pallas_bonded_and_tile_pairs_soa(
                B,
                px_m,
                py_m,
                pz_m,
                tp_st.q_m.reshape(B * n_act_pad),
                tp_st.sig_m.reshape(B * n_act_pad),
                tp_st.eps_m.reshape(B * n_act_pad),
                tp_st.mask_m.reshape(B * n_act_pad),
                tp_st.inv_perm.reshape(B * n_act_pad),
                tp_st.pair_ti.reshape(B * max_pairs),
                tp_st.pair_tj.reshape(B * max_pairs),
                tp_st.excl_full_bits.reshape(B * max_pairs * 32),
                tp_st.excl_14_bits.reshape(B * max_pairs * 32),
                tp_st.n_valid_pairs.reshape(B),
                box_f32_flat,
                lam_flat,
                bonded_bufs,
                n_act=n_act,
                n_act_blocks=n_act_blocks,
                max_pairs=max_pairs,
                pme_alpha=(pme_alpha if do_pme else 0.0),
                cutoff=cutoff,
                lj_switch_width=lj_sw,
                include_pme_excl_erf=do_pme,
                scale_14_vdw=dedup_excl.scale_14_vdw,
                scale_14_elec=dedup_excl.scale_14_elec,
                rf_dielectric=rf_diel,
                tp_group=tp_grp,
                compute_energy=compute_energy,
                deterministic_forces=det_forces,
                interpret=interpret,
            )
            if not do_pme:
                return e_dir, fx_d, fy_d, fz_d
            q_mf = tp_st.q_m.reshape(B * n_act_pad)
            m_mf = tp_st.mask_m.reshape(B * n_act_pad)
            mesh = _spread_charges_pallas_soa(
                B, (True, True, True, True), px_m, py_m, pz_m, q_mf, m_mf,
                box_f32_flat, grid_dims, n_act_pad, n_act_pad, interpret,
            )
            q_hat = jnp.fft.rfftn(mesh, s=grid_dims, axes=(-3, -2, -1))
            theta = jnp.fft.irfftn(q_hat * G_b_cur, s=grid_dims, axes=(-3, -2, -1)).astype(jnp.float32)
            fx_tot, fy_tot, fz_tot, _ = _gather_pallas_soa(
                B, (True, True, True, True, True), px_m, py_m, pz_m, q_mf, m_mf,
                box_f32_flat, theta.reshape(-1), grid_dims, n_act_pad, n_act_pad, n_act_blocks, interpret,
                force_scale=pme_force_scale, force_accum=(fx_d, fy_d, fz_d),
            )
            if compute_energy:
                e_rec = 0.5 * float(COULOMB_CONSTANT) * jnp.sum(mesh * theta, axis=(-3, -2, -1)) * n_grid_f + e_self_b
                e_tot = e_dir + e_rec
            else:
                e_tot = e_dir
            return e_tot, fx_tot, fy_tot, fz_tot

        tp0 = pallas_build_morton_tile_pairs_gpu(
            pos_init_b, q_b, sig_b, eps_b, am_b, box_init_b, dedup_excl,
            n_act=n_act, max_pairs=max_pairs, cutoff=cutoff, skin=eff_skin,
            host_min_box=eff_host_min_box, interpret=interpret,
        )
        pad_tail = n_act_pad - n_act
        pos_act0 = pos_init_b[:, :n_act, :].astype(jnp.float32)
        if pad_tail > 0:
            pos_act0 = jnp.pad(pos_act0, ((0, 0), (0, pad_tail), (0, 0)))
        pos_m0 = jnp.take_along_axis(pos_act0, tp0.perm[:, :, None], axis=1)
        _, fx_m0, fy_m0, fz_m0 = _eval_morton_forces(
            pos_m0[:, :, 0].reshape(B * n_act_pad),
            pos_m0[:, :, 1].reshape(B * n_act_pad),
            pos_m0[:, :, 2].reshape(B * n_act_pad),
            tp0,
            box_init_b,
            G_init,
            compute_energy=False,
        )
        p0_akma = float(press_atm) * 1.01325 / KCAL_MOL_PER_A3_TO_BAR if press_atm is not None else 0.0
        if press_atm is not None:
            c_tail_np, c_net_q_np = _compute_morton_npt_vol_constants(
                box_b=np.broadcast_to(box0_np[None, :], (B, 3)),
                sig_b=sig_b,
                eps_b=eps_b,
                q_b=q_b,
                am_b=am_b,
                cutoff=cutoff,
                lj_switch_width=lj_sw,
                combining_rule=comb_rule,
                use_dispersion_correction=use_disp,
                do_pme=do_pme,
                pme_alpha=pme_alpha,
            )
        else:
            c_tail_np = np.zeros((B,), dtype=np.float64)
            c_net_q_np = np.zeros((B,), dtype=np.float64)
        c_tail_j = jnp.asarray(c_tail_np, dtype=state_dtype)
        c_net_q_j = jnp.asarray(c_net_q_np, dtype=state_dtype)

        @jax.jit
        def _jit_run(
            seeds_u32: Any, n_steps_i32: Any, dt_s_in: Any, kT_s_in: Any,
            tp_in: Any, fx0_in: Any, fy0_in: Any, fz0_in: Any, box_in: Any, G_in: Any,
        ) -> tuple[Any, Any, Any]:
            dt_b = jnp.broadcast_to(jnp.asarray(dt_s_in, dtype=state_dtype), (B,))
            kT_b = jnp.broadcast_to(jnp.asarray(kT_s_in, dtype=state_dtype), (B,))
            keys_b = jax.vmap(jax.random.PRNGKey)(seeds_u32.astype(jnp.uint32))
            z_mb = jax.vmap(lambda k: jax.random.normal(k, (N, 3), dtype=jnp.float32))(keys_b).astype(state_dtype)
            mom0_b = jnp.where(
                am_b[:, :, None], z_mb * jnp.sqrt(mass_b[:, :, None] * kT_b[:, None, None]),
                jnp.asarray(0.0, dtype=state_dtype),
            )
            px0 = pos_init_b[:, :, 0].reshape(B * N)
            py0 = pos_init_b[:, :, 1].reshape(B * N)
            pz0 = pos_init_b[:, :, 2].reshape(B * N)
            mx0, my0, mz0 = _pallas_baoab_post_soa(
                px0, py0, pz0,
                mom0_b[:, :, 0].reshape(B * N), mom0_b[:, :, 1].reshape(B * N), mom0_b[:, :, 2].reshape(B * N),
                fx0_in, fy0_in, fz0_in, _inv_for_n(tp_in.inv_perm), n_act_pad,
                jnp.zeros((B,), dtype=state_dtype), 15.999, 1.008, interpret, soa_plan, f64_accumulators=f64_acc,
            )

            def _chunk_cond(oc: Any) -> Any:
                return oc[0] < n_steps_i32

            def _chunk_body(oc: Any) -> Any:
                step_i, px, py, pz, mx, my, mz, fx_m, fy_m, fz_m, tp_c, box_c, G_c = oc
                pos_cur_act = jnp.stack(
                    [px.reshape(B, N)[:, :n_act], py.reshape(B, N)[:, :n_act], pz.reshape(B, N)[:, :n_act]], axis=-1
                ).astype(jnp.float32)
                bx_f = box_c.astype(jnp.float32)[:, None, :]
                dr = pos_cur_act - tp_c.ref_positions[:, :n_act, :].astype(jnp.float32)
                dr = dr - bx_f * jnp.round(dr / bx_f)
                dr_sq = jnp.where(am_b[:, :n_act], jnp.sum(dr * dr, axis=-1), 0.0)
                need_reb = (step_i > 0) & jnp.any(jnp.max(dr_sq, axis=-1) > thresh_sq)

                def _do_rebuild(ops: Any) -> Any:
                    px_r, py_r, pz_r, fx_r, fy_r, fz_r, tp_old, bx_r = ops
                    pos_full = jnp.stack([px_r.reshape(B, N), py_r.reshape(B, N), pz_r.reshape(B, N)], axis=-1)
                    tp_new = pallas_build_morton_tile_pairs_gpu(
                        pos_full, q_b, sig_b, eps_b, am_b, bx_r, dedup_excl,
                        n_act=n_act, max_pairs=max_pairs, cutoff=cutoff, skin=eff_skin,
                        host_min_box=eff_host_min_box, interpret=interpret,
                    )
                    remap = jnp.take_along_axis(tp_old.inv_perm, tp_new.perm, axis=1)
                    fx_n = jnp.take_along_axis(fx_r.reshape(B, n_act_pad), remap, axis=1).reshape(B * n_act_pad)
                    fy_n = jnp.take_along_axis(fy_r.reshape(B, n_act_pad), remap, axis=1).reshape(B * n_act_pad)
                    fz_n = jnp.take_along_axis(fz_r.reshape(B, n_act_pad), remap, axis=1).reshape(B * n_act_pad)
                    return fx_n, fy_n, fz_n, tp_new

                fx_m, fy_m, fz_m, tp_c = jax.lax.cond(
                    need_reb, _do_rebuild, lambda ops: (ops[3], ops[4], ops[5], ops[6]),
                    (px, py, pz, fx_m, fy_m, fz_m, tp_c, box_c),
                )
                k_cur = jnp.minimum(jnp.int32(chunk_k), n_steps_i32 - step_i)
                if press_atm is not None:
                    next_baro = ((step_i // jnp.int32(baro_int)) + jnp.int32(1)) * jnp.int32(baro_int)
                    k_cur = jnp.minimum(k_cur, next_baro - step_i)
                step_next = step_i + k_cur
                inv_n = _inv_for_n(tp_c.inv_perm)

                def _inner_step(s_rel: Any, ic: Any) -> Any:
                    px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s = ic
                    seeds_s = jax.vmap(lambda k: jax.random.fold_in(k, step_i + s_rel))(keys_b)
                    px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, px_ms, py_ms, pz_ms = _pallas_baoab_pre_force_batch(
                        B, in_b_pre, pos_init_b, pos_init_b, pos_init_b, mass_b, seeds_s,
                        wi_b, wm_b, box_c, dt_b, kT_b, gamma_akma, 0.9572, 1.5139, 15.999, 1.008,
                        True, True, interpret, atom_mask=am_b, constraint_pairs=cp_b,
                        constraint_lengths=cl_b, constraint_mask=cm_b, n_active_atoms=n_act,
                        n_active_waters=n_waters_real, f64_accumulators=f64_acc, fuse_prev_post=True,
                        morton_inv_perm=inv_n, morton_pad=n_act_pad,
                        prebuilt_soa=(px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s, soa_plan),
                    )
                    _, fx_s, fy_s, fz_s = _eval_morton_forces(px_ms, py_ms, pz_ms, tp_c, box_c, G_c, False)
                    return (px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s)

                seeds_0 = jax.vmap(lambda k: jax.random.fold_in(k, step_i))(keys_b)
                px, py, pz, mxh, myh, mzh, px_m, py_m, pz_m = _pallas_baoab_pre_force_batch(
                    B, in_b_pre, pos_init_b, pos_init_b, pos_init_b, mass_b, seeds_0,
                    wi_b, wm_b, box_c, dt_b, kT_b, gamma_akma, 0.9572, 1.5139, 15.999, 1.008,
                    True, True, interpret, atom_mask=am_b, constraint_pairs=cp_b,
                    constraint_lengths=cl_b, constraint_mask=cm_b, n_active_atoms=n_act,
                    n_active_waters=n_waters_real, f64_accumulators=f64_acc, fuse_prev_post=False,
                    morton_inv_perm=inv_n, morton_pad=n_act_pad,
                    prebuilt_soa=(px, py, pz, mx, my, mz, fx_m, fy_m, fz_m, soa_plan),
                )
                if press_atm is None:
                    _, fx_m, fy_m, fz_m = _eval_morton_forces(px_m, py_m, pz_m, tp_c, box_c, G_c, False)
                    px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m = jax.lax.fori_loop(
                        jnp.int32(1), k_cur, _inner_step, (px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m)
                    )
                    e_old = jnp.zeros((B,), dtype=jnp.float32)
                    do_baro = jnp.bool_(False)
                else:
                    do_baro = ((step_next // jnp.int32(baro_int)) > (step_i // jnp.int32(baro_int)))

                    def _eval_forces_cond_e(
                        px_ce: Any, py_ce: Any, pz_ce: Any, want_e: Any
                    ) -> tuple[Any, Any, Any, Any]:
                        return jax.lax.cond(
                            want_e,
                            lambda _: _eval_morton_forces(px_ce, py_ce, pz_ce, tp_c, box_c, G_c, True),
                            lambda _: _eval_morton_forces(px_ce, py_ce, pz_ce, tp_c, box_c, G_c, False),
                            None,
                        )

                    e_old, fx_m, fy_m, fz_m = _eval_forces_cond_e(
                        px_m, py_m, pz_m, do_baro & (k_cur == jnp.int32(1))
                    )
                    px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m = jax.lax.fori_loop(
                        jnp.int32(1),
                        k_cur - jnp.int32(1),
                        _inner_step,
                        (px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m),
                    )

                    def _final_md_step(st_last: Any) -> Any:
                        _, px_l, py_l, pz_l, mxh_l, myh_l, mzh_l, fx_l, fy_l, fz_l = st_last
                        seeds_l = jax.vmap(lambda k: jax.random.fold_in(k, step_i + k_cur - jnp.int32(1)))(keys_b)
                        px_l, py_l, pz_l, mxh_l, myh_l, mzh_l, px_ml, py_ml, pz_ml = _pallas_baoab_pre_force_batch(
                            B, in_b_pre, pos_init_b, pos_init_b, pos_init_b, mass_b, seeds_l,
                            wi_b, wm_b, box_c, dt_b, kT_b, gamma_akma, 0.9572, 1.5139, 15.999, 1.008,
                            True, True, interpret, atom_mask=am_b, constraint_pairs=cp_b,
                            constraint_lengths=cl_b, constraint_mask=cm_b, n_active_atoms=n_act,
                            n_active_waters=n_waters_real, f64_accumulators=f64_acc, fuse_prev_post=True,
                            morton_inv_perm=inv_n, morton_pad=n_act_pad,
                            prebuilt_soa=(px_l, py_l, pz_l, mxh_l, myh_l, mzh_l, fx_l, fy_l, fz_l, soa_plan),
                        )
                        e_l, fx_l, fy_l, fz_l = _eval_forces_cond_e(px_ml, py_ml, pz_ml, do_baro)
                        return (e_l, px_l, py_l, pz_l, mxh_l, myh_l, mzh_l, fx_l, fy_l, fz_l)

                    e_old, px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m = jax.lax.cond(
                        k_cur > jnp.int32(1),
                        _final_md_step,
                        lambda st_last: st_last,
                        (e_old, px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m),
                    )

                mx, my, mz = _pallas_baoab_post_soa(
                    px, py, pz, mxh, myh, mzh, fx_m, fy_m, fz_m,
                    inv_n, n_act_pad, dt_b, 15.999, 1.008, interpret, soa_plan, f64_accumulators=f64_acc,
                )
                if cmm_freq > 0:
                    do_cmm = ((step_next // jnp.int32(max(1, cmm_freq))) > (step_i // jnp.int32(max(1, cmm_freq))))
                    def _rm_cmm(m_xyz: Any) -> Any:
                        mx_i, my_i, mz_i = m_xyz[0].reshape(B, N), m_xyz[1].reshape(B, N), m_xyz[2].reshape(B, N)
                        m_real = jnp.where(am_b, mass_b, jnp.asarray(0.0, dtype=state_dtype))
                        m_tot = jnp.maximum(jnp.sum(m_real, axis=1, keepdims=True), jnp.asarray(1e-12, dtype=state_dtype))
                        vx_cm = jnp.sum(jnp.where(am_b, mx_i, 0.0), axis=1, keepdims=True) / m_tot
                        vy_cm = jnp.sum(jnp.where(am_b, my_i, 0.0), axis=1, keepdims=True) / m_tot
                        vz_cm = jnp.sum(jnp.where(am_b, mz_i, 0.0), axis=1, keepdims=True) / m_tot
                        return (
                            jnp.where(am_b, mx_i - m_real * vx_cm, 0.0).reshape(B * N),
                            jnp.where(am_b, my_i - m_real * vy_cm, 0.0).reshape(B * N),
                            jnp.where(am_b, mz_i - m_real * vz_cm, 0.0).reshape(B * N),
                        )
                    mx, my, mz = jax.lax.cond(do_cmm, _rm_cmm, lambda m_xyz: m_xyz, (mx, my, mz))

                if press_atm is not None:
                    def _mc_baro(b_ops: Any) -> Any:
                        e_cur, px_b, py_b, pz_b, fx_b, fy_b, fz_b, tp_b, bx_b, G_b = b_ops
                        pos_cur = jnp.stack([px_b.reshape(B, N), py_b.reshape(B, N), pz_b.reshape(B, N)], axis=-1)
                        b_keys = jax.vmap(lambda k: jax.random.fold_in(k, step_next + jnp.int32(999983)))(keys_b)
                        u_pair = jax.vmap(lambda k: jax.random.uniform(k, (2,), dtype=jnp.float32))(b_keys)
                        dlnv = (u_pair[:, 0] - 0.5) * 0.004
                        dlnv_st = dlnv.astype(state_dtype)
                        mu = jnp.exp(dlnv_st / 3.0)
                        bx_prop = bx_b * mu[:, None]
                        anc = pos_cur[:, anchor_idx_j, :]
                        intra = pos_cur - anc
                        intra = intra - bx_b[:, None, :] * jnp.round(intra / bx_b[:, None, :])
                        pos_prop = jnp.where(
                            am_b[:, :, None],
                            jnp.mod(anc * mu[:, None, None] + intra, bx_prop[:, None, :]),
                            pos_cur,
                        )
                        G_prop = (
                            jax.vmap(lambda b_one: influence_function(grid_dims, b_one, pme_alpha, 4))(
                                bx_prop.astype(jnp.float32)
                            ).astype(jnp.float32)
                            if do_pme
                            else G_b
                        )
                        tp_prop = pallas_build_morton_tile_pairs_gpu(
                            pos_prop, q_b, sig_b, eps_b, am_b, bx_prop, dedup_excl,
                            n_act=n_act, max_pairs=max_pairs, cutoff=cutoff, skin=eff_skin,
                            host_min_box=eff_host_min_box, interpret=interpret,
                        )
                        p_ap = pos_prop[:, :n_act, :].astype(jnp.float32)
                        if pad_tail > 0:
                            p_ap = jnp.pad(p_ap, ((0, 0), (0, pad_tail), (0, 0)))
                        pm_prop = jnp.take_along_axis(p_ap, tp_prop.perm[:, :, None], axis=1)
                        e_new, fx_p, fy_p, fz_p = _eval_morton_forces(
                            pm_prop[:, :, 0].reshape(B * n_act_pad),
                            pm_prop[:, :, 1].reshape(B * n_act_pad),
                            pm_prop[:, :, 2].reshape(B * n_act_pad),
                            tp_prop, bx_prop, G_prop, True,
                        )
                        v_old = jnp.prod(bx_b, axis=-1)
                        v_new = jnp.prod(bx_prop, axis=-1)
                        kT_f = kT_b.astype(jnp.float32)
                        dh = _compute_mc_barostat_dh(
                            e_cur,
                            e_new,
                            v_old,
                            v_new,
                            c_tail_b=c_tail_j,
                            c_net_q_b=c_net_q_j,
                            p0_akma=p0_akma,
                            n_mols=n_mols_baro,
                            kT=kT_f,
                            dlnv=dlnv_st,
                        )
                        acc = u_pair[:, 1] < jnp.exp(jnp.clip(-dh / jnp.maximum(kT_f, 1e-6), -50.0, 50.0))
                        pos_sel = jnp.where(acc[:, None, None], pos_prop, pos_cur)
                        bx_sel = jnp.where(acc[:, None], bx_prop, bx_b)
                        G_sel = jnp.where(acc[:, None, None, None], G_prop, G_b)
                        tp_sel = jax.tree.map(
                            lambda np_v, op_v: jnp.where(acc.reshape((B,) + (1,) * (np_v.ndim - 1)), np_v, op_v),
                            tp_prop, tp_b,
                        )
                        fx_sel = jnp.where(acc[:, None], fx_p.reshape(B, n_act_pad), fx_b.reshape(B, n_act_pad)).reshape(B * n_act_pad)
                        fy_sel = jnp.where(acc[:, None], fy_p.reshape(B, n_act_pad), fy_b.reshape(B, n_act_pad)).reshape(B * n_act_pad)
                        fz_sel = jnp.where(acc[:, None], fz_p.reshape(B, n_act_pad), fz_b.reshape(B, n_act_pad)).reshape(B * n_act_pad)
                        return (
                            pos_sel[:, :, 0].reshape(B * N), pos_sel[:, :, 1].reshape(B * N), pos_sel[:, :, 2].reshape(B * N),
                            fx_sel, fy_sel, fz_sel, tp_sel, bx_sel, G_sel,
                        )
                    px, py, pz, fx_m, fy_m, fz_m, tp_c, box_c, G_c = jax.lax.cond(
                        do_baro,
                        _mc_baro,
                        lambda b_ops: b_ops[1:],
                        (e_old, px, py, pz, fx_m, fy_m, fz_m, tp_c, box_c, G_c),
                    )
                return (step_next, px, py, pz, mx, my, mz, fx_m, fy_m, fz_m, tp_c, box_c, G_c)

            _, pxf, pyf, pzf, mxf, myf, mzf, _, _, _, _, boxf, _ = jax.lax.while_loop(
                _chunk_cond, _chunk_body,
                (jnp.int32(0), px0, py0, pz0, mx0, my0, mz0, fx0_in, fy0_in, fz0_in, tp_in, box_in, G_in),
            )
            pos_out = jnp.stack([pxf.reshape(B, N), pyf.reshape(B, N), pzf.reshape(B, N)], axis=-1)
            mom_out = jnp.stack([mxf.reshape(B, N), myf.reshape(B, N), mzf.reshape(B, N)], axis=-1)
            return pos_out, mom_out, boxf

        return {
            "jit_run": _jit_run,
            "tp0": tp0,
            "fx_m0": fx_m0,
            "fy_m0": fy_m0,
            "fz_m0": fz_m0,
            "box_init_b": box_init_b,
            "G_init": G_init,
            "mass_b": mass_b,
            "am_b": am_b,
            "n_reals": n_reals,
            "N": N,
            "state_dtype": state_dtype,
            "eff_skin": eff_skin,
            "box_safety": box_safety,
            "max_pairs": max_pairs,
            "c_tail_b": c_tail_np,
            "c_net_q_b": c_net_q_np,
            "rf_dielectric": rf_diel,
            "deterministic_forces": det_forces,
        }

    def _run_morton_soa_batch(
        self,
        bundles: list[Any],
        n_steps: int | Any,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None = None,
        *,
        integration_prefix: int | None = None,
        trim_output: bool = True,
        dt_unit: str = "fs",
        gamma: float = 10.0,
        nl_update_every: int = 20,
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
    ) -> list[Trajectory]:
        B = len(bundles)
        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        if prec in ("mixed", "double") and not bool(jax.config.read("jax_enable_x64")):
            jax.config.update("jax_enable_x64", True)

        can_cache = (
            observables is None
            and not isinstance(n_steps, jax.core.Tracer)
            and not isinstance(dt, jax.core.Tracer)
            and not isinstance(kT, jax.core.Tracer)
            and not isinstance(seed, jax.core.Tracer)
        )
        cache_key = None
        entry = None
        if can_cache:
            try:
                ks_hash: Any = hash(self.knob_surface)
            except TypeError:
                ks_hash = id(self.knob_surface)
            cache_key = (
                "morton_soa_v36",
                tuple(id(b) for b in bundles),
                integration_prefix,
                float(gamma),
                int(nl_update_every),
                nb_cfg,
                use_pallas,
                bool(jax.config.read("jax_enable_x64")),
                os.environ.get("PROLIX_PME_GRID_POINTS", ""),
                os.environ.get("PROLIX_ACTIVE_ATOMS", ""),
                os.environ.get("PROLIX_ACTIVE_WATERS", ""),
                tuple(sorted(knobs.items())),
                ks_hash,
            )
            if B == 1:
                cand = _SINGLE_INFERENCE_CACHE.get(cache_key)
                if cand is not None and cand.get("bundle") is bundles[0]:
                    entry = cand
            else:
                cand = _STACKED_INFERENCE_CACHE.get(cache_key)
                if (
                    cand is not None
                    and len(cand.get("bundles", ())) == B
                    and all(a is b for a, b in zip(cand["bundles"], bundles, strict=True))
                ):
                    entry = cand

        if entry is None:
            entry = self._build_morton_soa_runner(
                bundles,
                integration_prefix=integration_prefix,
                gamma=float(gamma),
                nl_update_every=nl_update_every,
                nb_cfg=nb_cfg,
            )
            if can_cache and cache_key is not None:
                if B == 1:
                    entry["bundle"] = bundles[0]
                    if len(_SINGLE_INFERENCE_CACHE) >= 16:
                        _SINGLE_INFERENCE_CACHE.pop(next(iter(_SINGLE_INFERENCE_CACHE)))
                    _SINGLE_INFERENCE_CACHE[cache_key] = entry
                else:
                    entry["bundles"] = tuple(bundles)
                    if len(_STACKED_INFERENCE_CACHE) >= 16:
                        _STACKED_INFERENCE_CACHE.pop(next(iter(_STACKED_INFERENCE_CACHE)))
                    _STACKED_INFERENCE_CACHE[cache_key] = entry
                    _STACKED_JIT_FN_CACHE[cache_key] = (None, entry["jit_run"])

        state_dtype = entry["state_dtype"]
        dt_arr = jnp.asarray(dt, dtype=state_dtype)
        dt_akma = dt_arr / AKMA_TIME_UNIT_FS if dt_unit == "fs" else dt_arr
        kT_arr = jnp.asarray(kT, dtype=state_dtype)
        seeds_u32 = jnp.arange(B, dtype=jnp.uint32) + jnp.asarray(seed, dtype=jnp.uint32)
        n_steps_i32 = jnp.asarray(n_steps, dtype=jnp.int32)

        pos_out, mom_out, _box_out = entry["jit_run"](
            seeds_u32,
            n_steps_i32,
            dt_akma,
            kT_arr,
            entry["tp0"],
            entry["fx_m0"],
            entry["fy_m0"],
            entry["fz_m0"],
            entry["box_init_b"],
            entry["G_init"],
        )
        if B == 1 and can_cache:
            if len(_JITTED_WHILE_CACHE) >= 128:
                _JITTED_WHILE_CACHE.pop(next(iter(_JITTED_WHILE_CACHE)))
            _JITTED_WHILE_CACHE[(entry["jit_run"], "morton_soa")] = entry["jit_run"]

        N = int(entry["N"])
        n_reals = entry["n_reals"]
        n_steps_ret = 0 if isinstance(n_steps, jax.core.Tracer) else int(n_steps)
        trajs: list[Trajectory] = []
        for b_idx, bun in enumerate(bundles):
            n_r = int(n_reals[b_idx])
            n_out = n_r if trim_output else (int(integration_prefix) if integration_prefix is not None else n_r)
            obs_vals: dict[str, Any] = {}
            if knobs.get("pressure_atm") is not None:
                obs_vals["box"] = _box_out[b_idx]
            if observables:
                obs_b = _observables_for_bundle(bun, observables, atom_mask=bun.atom_mask[:N])
                st_b = NVTLangevinState(
                    pos_out[b_idx],
                    mom_out[b_idx],
                    jnp.zeros_like(pos_out[b_idx]),
                    entry["mass_b"][b_idx, :, None],
                    jax.random.PRNGKey(seeds_u32[b_idx]),
                )
                for oname, oobj in obs_b.items():
                    obs_vals[oname] = oobj.compute(st_b)
            trajs.append(
                cast("Any", Trajectory)(
                    positions=pos_out[b_idx, :n_out, :][None, ...],
                    observable_values=obs_vals,
                    n_steps=n_steps_ret,
                )
            )
        return trajs

    def _build_ragged_mk68_runner(
        self,
        bundles: list[Any],
        *,
        integration_prefix: int | None,
        gamma: float,
        nl_update_every: int,
        nb_cfg: Any,
        use_pallas: bool | None,
        use_pallas_pme: bool | None,
    ) -> dict[str, Any]:
        bundles = _align_bundles_for_stacking(bundles)
        B = len(bundles)
        N = int(integration_prefix) if integration_prefix is not None else int(bundles[0].positions.shape[0])
        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        state_dtype: Any = (
            jnp.float64
            if prec in ("mixed", "double")
            else (jnp.float32 if prec == "single" else bundles[0].positions.dtype)
        )
        f64_acc = state_dtype == jnp.float64
        if prec in ("mixed", "double") and not bool(jax.config.read("jax_enable_x64")):
            jax.config.update("jax_enable_x64", True)

        _raw_tier_mode = str(knobs.get("pme_tier_mode") or "auto").strip().lower()
        pme_tier_mode_val = cast(
            "PMETierMode",
            _raw_tier_mode if _raw_tier_mode in ("auto", "grouped", "uniform") else "auto",
        )
        tp_grp_knob = knobs.get("tp_group")
        cutoff = _finite_nl_cutoff(
            float(_host_float(getattr(bundles[0], "cutoff_distance", 9.0), 9.0)),
            getattr(bundles[0], "box", None),
            9.0,
        )
        pme_alpha = float(_host_float(getattr(bundles[0], "pme_alpha", 0.34), 0.34))
        do_pme = pme_alpha > 0.0
        lj_sw = float(lj_switch_width_from_config(nb_cfg))
        nlc = getattr(nb_cfg, "neighbor_list", None) if nb_cfg is not None else None
        skin = float(nlc.skin) if nlc is not None else 0.5
        chunk_len = max(1, int(nl_update_every))

        n_reals = [
            b.n_atoms if isinstance(b.n_atoms, int) else int(np.asarray(b.n_atoms))
            for b in bundles
        ]
        pos_init_b = jnp.stack([positions_with_prefix(b, N).astype(state_dtype) for b in bundles], axis=0)
        mass_b = jnp.stack([masses_with_prefix(b, N).astype(state_dtype) for b in bundles], axis=0)
        am_b = jnp.stack([jnp.asarray(b.atom_mask[:N], dtype=jnp.int32) for b in bundles], axis=0)
        box_init_b = jnp.stack(
            [(jnp.diag(b.box) if b.box.ndim == 2 else b.box).astype(state_dtype) for b in bundles],
            axis=0,
        )
        q_b = jnp.stack([jnp.asarray(b.charges[:N], dtype=jnp.float32) for b in bundles], axis=0)
        sig_b = jnp.stack([jnp.asarray(b.sigmas[:N], dtype=jnp.float32) for b in bundles], axis=0)
        eps_b = jnp.stack([jnp.asarray(b.epsilons[:N], dtype=jnp.float32) for b in bundles], axis=0)

        m_caps = [
            estimate_max_tile_pairs(max(1, (max(1, n_reals[i]) + 31) // 32), cutoff, skin)
            for i in range(B)
        ]
        m_max_all = max(*m_caps, 1)
        pair_queue = build_ragged_tile_pair_queue(
            m_caps, tp_group=tp_grp_knob, max_tile_pairs=m_max_all
        )
        dedup_excl_list = [
            precompute_dedup_excl_edges(
                b.excl_dense_indices[:N] if getattr(b, "excl_dense_indices", None) is not None else None,
                b.excl_dense_scales_vdw[:N] if getattr(b, "excl_dense_scales_vdw", None) is not None else None,
                b.excl_dense_scales_elec[:N] if getattr(b, "excl_dense_scales_elec", None) is not None else None,
                N,
            )
            for b in bundles
        ]
        host_min_boxes = [
            float(np.min(np.diag(np.asarray(b.box, dtype=np.float64)) if b.box.ndim == 2 else np.asarray(b.box, dtype=np.float64)))
            for b in bundles
        ]
        tier_plan = (
            plan_pme_tiers(
                np.asarray(box_init_b, dtype=np.float64),
                cutoff=cutoff,
                alpha=pme_alpha,
                n_real_per_replica=n_reals,
                padded_atoms=N,
                atom_mask=am_b,
                pme_tier_mode=pme_tier_mode_val,
            )
            if do_pme
            else None
        )
        stacked_bun = stack_molecular_bundles(bundles)
        eff_pallas_pme = bool(
            use_pallas_pme if use_pallas_pme is not None else (use_pallas if use_pallas is not None else True)
        )
        interpret = jax.default_backend() == "cpu"
        am_bool_b = am_b.astype(jnp.bool_)
        n_act_pad = max(1, (N + 31) // 32) * 32

        def _build_tp_all(pos_f32_b: Any, box_f32_b: Any):
            perms, invs, tps, efs, e14s, qms, sms, ems, mms, refs = [], [], [], [], [], [], [], [], [], []
            for i in range(B):
                tpi = pallas_build_morton_tile_pairs_gpu(
                    pos_f32_b[i : i + 1],
                    q_b[i : i + 1],
                    sig_b[i : i + 1],
                    eps_b[i : i + 1],
                    am_bool_b[i : i + 1],
                    box_f32_b[i : i + 1],
                    dedup_excl_list[i],
                    n_act=N,
                    max_pairs=m_caps[i],
                    cutoff=cutoff,
                    skin=skin,
                    host_min_box=host_min_boxes[i],
                    interpret=interpret,
                )
                pm_i = tpi.perm[0]
                iv_i = jnp.clip(tpi.inv_perm[0, :N], 0, int(pm_i.shape[0]) - 1)
                tp_raw = jnp.stack([tpi.pair_ti[0], tpi.pair_tj[0]], axis=-1)
                pad_m = m_max_all - int(tp_raw.shape[0])
                tp_arr = jnp.pad(tp_raw, ((0, pad_m), (0, 0)), constant_values=-1) if pad_m > 0 else tp_raw
                ef_arr = (
                    jnp.pad(tpi.excl_full_bits[0], ((0, pad_m), (0, 0)), constant_values=0)
                    if pad_m > 0
                    else tpi.excl_full_bits[0]
                )
                e14_arr = (
                    jnp.pad(tpi.excl_14_bits[0], ((0, pad_m), (0, 0)), constant_values=0)
                    if pad_m > 0
                    else tpi.excl_14_bits[0]
                )
                perms.append(pm_i)
                invs.append(iv_i)
                tps.append(tp_arr)
                efs.append(ef_arr)
                e14s.append(e14_arr)
                qms.append(tpi.q_m[0])
                sms.append(tpi.sig_m[0])
                ems.append(tpi.eps_m[0])
                mms.append(tpi.mask_m[0])
                refs.append(tpi.ref_positions[0, :N, :].astype(jnp.float32))
            return (
                jnp.stack(perms, axis=0),
                jnp.stack(invs, axis=0),
                jnp.stack(tps, axis=0),
                jnp.stack(efs, axis=0),
                jnp.stack(e14s, axis=0),
                jnp.stack(qms, axis=0),
                jnp.stack(sms, axis=0),
                jnp.stack(ems, axis=0),
                jnp.stack(mms, axis=0),
                jnp.stack(refs, axis=0),
            )

        def _bonded_one(pos_1: Any, sb_1: Any, bx_1: Any) -> Any:
            dp_1 = sb_1.dihedral_params
            flat_d = dp_1.reshape(-1, dp_1.shape[-1]) if dp_1.ndim == 3 else dp_1[..., :3]
            ip_1 = sb_1.improper_params
            flat_i = ip_1.reshape(-1, ip_1.shape[-1]) if ip_1.ndim == 3 else ip_1[..., :3]
            ub_i = (
                sb_1.urey_bradley_idx[:, ::2]
                if sb_1.urey_bradley_idx.shape[-1] == 3
                else sb_1.urey_bradley_idx[:, :2]
            )
            _, fb = pallas_bonded_all_fused(
                pos_1,
                bonds=sb_1.bond_idx,
                bond_params=sb_1.bond_params,
                bond_mask=sb_1.bond_mask,
                urey_bradley_bonds=ub_i,
                urey_bradley_params=sb_1.urey_bradley_params,
                urey_bradley_mask=sb_1.urey_bradley_mask,
                angles=sb_1.angle_idx,
                angle_params=sb_1.angle_params,
                angle_mask=sb_1.angle_mask,
                dihedrals=sb_1.dihedral_idx,
                dihedral_params=flat_d,
                dihedral_mask=sb_1.dihedral_mask,
                impropers=sb_1.improper_idx,
                improper_params=flat_i,
                improper_mask=sb_1.improper_mask,
                improper_is_harmonic=(flat_i.shape[-1] == 2),
                exception_pairs=sb_1.exception_pairs,
                exception_sigmas=sb_1.exception_sigmas,
                exception_epsilons=sb_1.exception_epsilons,
                exception_chargeprods=sb_1.exception_chargeprods,
                exception_mask=sb_1.exception_mask,
                cmap_torsions=sb_1.cmap_torsion_idx,
                cmap_indices=sb_1.cmap_indices,
                cmap_mask=sb_1.cmap_mask,
                cmap_coeffs=sb_1.cmap_coeffs,
                box=bx_1,
                dist_eps=1e-10,
                b0_is_ji=True,
                cmap_first_is_psi=True,
                return_forces=True,
                interpret=interpret,
            )
            return fb

        pad_tail = n_act_pad - N
        s14_v = float(dedup_excl_list[0].scale_14_vdw)
        s14_e = float(dedup_excl_list[0].scale_14_elec)

        def _eval_ragged_forces(pos_cur_b: Any, tp_tuple: Any, box_cur_b: Any) -> Any:
            perm_c, inv_c, tp_c, ef_c, e14_c, qm_c, sm_c, em_c, mm_c, _ref_c = tp_tuple
            pos_f32 = pos_cur_b.astype(jnp.float32)
            bx_f32 = box_cur_b.astype(jnp.float32)
            f_bond = jax.vmap(_bonded_one)(pos_f32, stacked_bun, bx_f32)
            pos_pad = jnp.pad(pos_f32, ((0, 0), (0, pad_tail), (0, 0))) if pad_tail > 0 else pos_f32
            pos_m = jnp.take_along_axis(pos_pad, perm_c[:, :, None], axis=1)
            _, f_dir_m = pallas_nonbonded_tile_pairs_ragged(
                pos_m,
                qm_c,
                sm_c,
                em_c,
                mm_c,
                tp_c,
                ef_c,
                e14_c,
                pair_queue=pair_queue,
                box=bx_f32,
                cutoff=cutoff,
                pme_alpha=(pme_alpha if do_pme else 0.0),
                lj_switch_width=lj_sw,
                scale_14_vdw=s14_v,
                scale_14_elec=s14_e,
                include_pme_excl_erf=do_pme,
                tp_group=pair_queue.tile_size,
                return_forces=True,
                interpret=interpret,
            )
            f_dir = jnp.take_along_axis(f_dir_m, inv_c[:, :, None], axis=1)
            if do_pme and tier_plan is not None:
                f_rec = spme_forces_ragged_tiered(
                    pos_f32,
                    q_b,
                    am_b,
                    bx_f32,
                    tier_plan=tier_plan,
                    use_pallas_pme=eff_pallas_pme,
                    interpret=interpret,
                )
                f_tot = f_bond + f_dir + f_rec
            else:
                f_tot = f_bond + f_dir
            return jnp.where(am_bool_b[:, :, None], f_tot, 0.0)

        if any(b.shape_spec.has_real_water for b in bundles):
            wi_b = jnp.stack([b.water_indices.astype(jnp.int32) for b in bundles], axis=0)
            wm_b = jnp.stack([b.water_mask.astype(jnp.int32) for b in bundles], axis=0)
        else:
            wi_b = jnp.zeros((B, 0, 3), dtype=jnp.int32)
            wm_b = jnp.zeros((B, 0), dtype=jnp.int32)
        has_cons = bool(
            any(getattr(b.shape_spec, "has_real_constraints", False) for b in bundles)
            and bundles[0].constraint_pairs is not None
            and int(bundles[0].constraint_pairs.shape[0]) > 0
        )
        cp_b = jnp.stack([b.constraint_pairs.astype(jnp.int32) for b in bundles], axis=0) if has_cons else None
        cl_b = jnp.stack([b.constraint_lengths.astype(state_dtype) for b in bundles], axis=0) if has_cons else None
        cm_b = jnp.stack([b.constraint_mask.astype(jnp.int32) for b in bundles], axis=0) if has_cons else None
        n_act = max(1, min(N, max(n_reals)))
        n_waters_real = max(
            (int(np.sum(np.asarray(b.water_mask))) if b.shape_spec.has_real_water else 0)
            for b in bundles
        )
        soa_plan = build_baoab_soa_static_plan(
            B, N, state_dtype, mass_b, wi_b, wm_b,
            atom_mask_b=am_bool_b, constraint_pairs_b=cp_b, constraint_lengths_b=cl_b, constraint_mask_b=cm_b,
            n_active_atoms=n_act, n_active_waters=n_waters_real,
        )
        gamma_akma = float(gamma_ps_to_akma(float(gamma)))
        in_b_pre = (True,) * (12 + (3 if has_cons else 0))
        ident_inv = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[None, :], (B, N))
        thresh_sq = float((0.5 * skin) ** 2)

        tp0_tuple = _build_tp_all(pos_init_b.astype(jnp.float32), box_init_b.astype(jnp.float32))
        f0_b = _eval_ragged_forces(pos_init_b, tp0_tuple, box_init_b)

        @jax.jit
        def _jit_ragged(
            seeds_u32: Any, n_steps_i32: Any, dt_s_in: Any, kT_s_in: Any,
            tp_in: Any, f0_in: Any, box_in: Any,
        ) -> tuple[Any, Any]:
            dt_b = jnp.broadcast_to(jnp.asarray(dt_s_in, dtype=state_dtype), (B,))
            kT_b = jnp.broadcast_to(jnp.asarray(kT_s_in, dtype=state_dtype), (B,))
            keys_b = jax.vmap(jax.random.PRNGKey)(seeds_u32.astype(jnp.uint32))
            z_mb = jax.vmap(lambda k: jax.random.normal(k, (N, 3), dtype=jnp.float32))(keys_b).astype(state_dtype)
            mom0_b = jnp.where(
                am_bool_b[:, :, None],
                z_mb * jnp.sqrt(mass_b[:, :, None] * kT_b[:, None, None]),
                jnp.asarray(0.0, dtype=state_dtype),
            )
            px0 = pos_init_b[:, :, 0].reshape(B * N)
            py0 = pos_init_b[:, :, 1].reshape(B * N)
            pz0 = pos_init_b[:, :, 2].reshape(B * N)
            fx0 = f0_in[:, :, 0].reshape(B * N)
            fy0 = f0_in[:, :, 1].reshape(B * N)
            fz0 = f0_in[:, :, 2].reshape(B * N)
            mx0, my0, mz0 = _pallas_baoab_post_soa(
                px0, py0, pz0,
                mom0_b[:, :, 0].reshape(B * N), mom0_b[:, :, 1].reshape(B * N), mom0_b[:, :, 2].reshape(B * N),
                fx0, fy0, fz0, ident_inv, N,
                jnp.zeros((B,), dtype=state_dtype), 15.999, 1.008, interpret, soa_plan, f64_accumulators=f64_acc,
            )

            def _chunk_cond(oc: Any) -> Any:
                return oc[0] < n_steps_i32

            def _chunk_body(oc: Any) -> Any:
                step_i, px, py, pz, mx, my, mz, fx, fy, fz, tp_c = oc
                pos_cur = jnp.stack([px.reshape(B, N), py.reshape(B, N), pz.reshape(B, N)], axis=-1).astype(jnp.float32)
                bx_f = box_in.astype(jnp.float32)[:, None, :]
                dr = pos_cur - tp_c[9]
                dr = dr - bx_f * jnp.round(dr / bx_f)
                dr_sq = jnp.where(am_bool_b, jnp.sum(dr * dr, axis=-1), 0.0)
                need_reb = (step_i > 0) & jnp.any(jnp.max(dr_sq, axis=-1) > thresh_sq)
                tp_c = jax.lax.cond(
                    need_reb,
                    lambda p_in: _build_tp_all(p_in, box_in.astype(jnp.float32)),
                    lambda _p_in: tp_c,
                    pos_cur,
                )
                k_cur = jnp.minimum(jnp.int32(chunk_len), n_steps_i32 - step_i)
                seeds_0 = jax.vmap(lambda k: jax.random.fold_in(k, step_i))(keys_b)
                px, py, pz, mxh, myh, mzh, px_f32, py_f32, pz_f32 = _pallas_baoab_pre_force_batch(
                    B, in_b_pre, pos_init_b, pos_init_b, pos_init_b, mass_b, seeds_0,
                    wi_b, wm_b, box_in, dt_b, kT_b, gamma_akma, 0.9572, 1.5139, 15.999, 1.008,
                    True, True, interpret, atom_mask=am_bool_b, constraint_pairs=cp_b,
                    constraint_lengths=cl_b, constraint_mask=cm_b, n_active_atoms=n_act,
                    n_active_waters=n_waters_real, f64_accumulators=f64_acc, fuse_prev_post=False,
                    morton_inv_perm=ident_inv, morton_pad=N,
                    prebuilt_soa=(px, py, pz, mx, my, mz, fx, fy, fz, soa_plan),
                )
                f_cur = _eval_ragged_forces(
                    jnp.stack([px_f32.reshape(B, N), py_f32.reshape(B, N), pz_f32.reshape(B, N)], axis=-1),
                    tp_c, box_in,
                )
                fx, fy, fz = f_cur[:, :, 0].reshape(B * N), f_cur[:, :, 1].reshape(B * N), f_cur[:, :, 2].reshape(B * N)

                def _inner_s(s_rel: Any, ic: Any) -> Any:
                    px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s = ic
                    seeds_s = jax.vmap(lambda k: jax.random.fold_in(k, step_i + s_rel))(keys_b)
                    px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, pxf_s, pyf_s, pzf_s = _pallas_baoab_pre_force_batch(
                        B, in_b_pre, pos_init_b, pos_init_b, pos_init_b, mass_b, seeds_s,
                        wi_b, wm_b, box_in, dt_b, kT_b, gamma_akma, 0.9572, 1.5139, 15.999, 1.008,
                        True, True, interpret, atom_mask=am_bool_b, constraint_pairs=cp_b,
                        constraint_lengths=cl_b, constraint_mask=cm_b, n_active_atoms=n_act,
                        n_active_waters=n_waters_real, f64_accumulators=f64_acc, fuse_prev_post=True,
                        morton_inv_perm=ident_inv, morton_pad=N,
                        prebuilt_soa=(px_s, py_s, pz_s, mxh_s, myh_s, mzh_s, fx_s, fy_s, fz_s, soa_plan),
                    )
                    fs = _eval_ragged_forces(
                        jnp.stack([pxf_s.reshape(B, N), pyf_s.reshape(B, N), pzf_s.reshape(B, N)], axis=-1),
                        tp_c, box_in,
                    )
                    return (
                        px_s, py_s, pz_s, mxh_s, myh_s, mzh_s,
                        fs[:, :, 0].reshape(B * N), fs[:, :, 1].reshape(B * N), fs[:, :, 2].reshape(B * N),
                    )

                px, py, pz, mxh, myh, mzh, fx, fy, fz = jax.lax.fori_loop(
                    jnp.int32(1), k_cur, _inner_s, (px, py, pz, mxh, myh, mzh, fx, fy, fz)
                )
                mx, my, mz = _pallas_baoab_post_soa(
                    px, py, pz, mxh, myh, mzh, fx, fy, fz,
                    ident_inv, N, dt_b, 15.999, 1.008, interpret, soa_plan, f64_accumulators=f64_acc,
                )
                return (step_i + k_cur, px, py, pz, mx, my, mz, fx, fy, fz, tp_c)

            _, pxf, pyf, pzf, mxf, myf, mzf, _, _, _, _ = jax.lax.while_loop(
                _chunk_cond, _chunk_body, (jnp.int32(0), px0, py0, pz0, mx0, my0, mz0, fx0, fy0, fz0, tp_in)
            )
            return (
                jnp.stack([pxf.reshape(B, N), pyf.reshape(B, N), pzf.reshape(B, N)], axis=-1),
                jnp.stack([mxf.reshape(B, N), myf.reshape(B, N), mzf.reshape(B, N)], axis=-1),
            )

        return {
            "jit_run": _jit_ragged,
            "tp0": tp0_tuple,
            "f0": f0_b,
            "box_init_b": box_init_b,
            "mass_b": mass_b,
            "n_reals": n_reals,
            "N": N,
            "state_dtype": state_dtype,
        }

    def _run_ragged_mk68_batch(
        self,
        bundles: list[Any],
        n_steps: int | Any,
        dt: float | Any,
        kT: float | Any,
        seed: int | Any,
        observables: dict[str, Any] | None = None,
        *,
        integration_prefix: int | None = None,
        trim_output: bool = True,
        dt_unit: str = "fs",
        gamma: float = 10.0,
        nl_update_every: int = 20,
        nb_cfg: Any = None,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
    ) -> list[Trajectory]:
        B = len(bundles)
        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        if prec in ("mixed", "double") and not bool(jax.config.read("jax_enable_x64")):
            jax.config.update("jax_enable_x64", True)

        can_cache = (
            observables is None
            and not isinstance(n_steps, jax.core.Tracer)
            and not isinstance(dt, jax.core.Tracer)
            and not isinstance(kT, jax.core.Tracer)
            and not isinstance(seed, jax.core.Tracer)
        )
        cache_key = None
        entry = None
        if can_cache:
            cache_key = (
                "ragged_mk68_v35",
                tuple(id(b) for b in bundles),
                integration_prefix,
                float(gamma),
                int(nl_update_every),
                nb_cfg,
                use_pallas,
                use_pallas_pme,
                bool(jax.config.read("jax_enable_x64")),
                os.environ.get("PROLIX_PME_GRID_POINTS", ""),
                tuple(sorted(knobs.items())),
            )
            cand = _STACKED_INFERENCE_CACHE.get(cache_key)
            if (
                cand is not None
                and len(cand.get("bundles", ())) == B
                and all(a is b for a, b in zip(cand["bundles"], bundles, strict=True))
            ):
                entry = cand

        if entry is None:
            entry = self._build_ragged_mk68_runner(
                bundles,
                integration_prefix=integration_prefix,
                gamma=float(gamma),
                nl_update_every=nl_update_every,
                nb_cfg=nb_cfg,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
            )
            if can_cache and cache_key is not None:
                entry["bundles"] = tuple(bundles)
                if len(_STACKED_INFERENCE_CACHE) >= 16:
                    _STACKED_INFERENCE_CACHE.pop(next(iter(_STACKED_INFERENCE_CACHE)))
                _STACKED_INFERENCE_CACHE[cache_key] = entry
                _STACKED_JIT_FN_CACHE[cache_key] = (None, entry["jit_run"])

        state_dtype = entry["state_dtype"]
        dt_arr = jnp.asarray(dt, dtype=state_dtype)
        dt_akma = dt_arr / AKMA_TIME_UNIT_FS if dt_unit == "fs" else dt_arr
        kT_arr = jnp.asarray(kT, dtype=state_dtype)
        seeds_u32 = jnp.arange(B, dtype=jnp.uint32) + jnp.asarray(seed, dtype=jnp.uint32)
        n_steps_i32 = jnp.asarray(n_steps, dtype=jnp.int32)

        pos_out, mom_out = entry["jit_run"](
            seeds_u32,
            n_steps_i32,
            dt_akma,
            kT_arr,
            entry["tp0"],
            entry["f0"],
            entry["box_init_b"],
        )

        N = int(entry["N"])
        n_reals = entry["n_reals"]
        n_steps_ret = 0 if isinstance(n_steps, jax.core.Tracer) else int(n_steps)
        trajs: list[Trajectory] = []
        for b_idx, bun in enumerate(bundles):
            n_r = int(n_reals[b_idx])
            n_out = n_r if trim_output else (int(integration_prefix) if integration_prefix is not None else n_r)
            obs_vals: dict[str, Any] = {}
            if observables:
                obs_b = _observables_for_bundle(bun, observables, atom_mask=bun.atom_mask[:N])
                st_b = NVTLangevinState(
                    pos_out[b_idx],
                    mom_out[b_idx],
                    jnp.zeros_like(pos_out[b_idx]),
                    entry["mass_b"][b_idx, :, None],
                    jax.random.PRNGKey(seeds_u32[b_idx]),
                )
                for oname, oobj in obs_b.items():
                    obs_vals[oname] = oobj.compute(st_b)
            trajs.append(
                cast("Any", Trajectory)(
                    positions=pos_out[b_idx, :n_out, :][None, ...],
                    observable_values=obs_vals,
                    n_steps=n_steps_ret,
                )
            )
        return trajs

    @classmethod
    def from_bundle(
        cls,
        bundle: Any,
        planner: Any = None,
        *,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
        nonbonded_engine: str | None = None,
        precision: str | None = None,
        nb_k_splits: int | None = None,
        tp_group: int | None = None,
        cmm_frequency: int | None = None,
        deterministic_forces: bool | None = None,
        gb_use_neighbor_list: bool | None = None,
        pme_tier_mode: str | None = None,
        pressure_atm: float | None = None,
        barostat_interval: int | None = None,
        knob_surface: KnobSurface | None = None,
    ) -> EnsemblePlan:
        return cls(
            [bundle],
            planner,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
            nonbonded_engine=nonbonded_engine,
            precision=precision,
            nb_k_splits=nb_k_splits,
            tp_group=tp_group,
            cmm_frequency=cmm_frequency,
            deterministic_forces=deterministic_forces,
            gb_use_neighbor_list=gb_use_neighbor_list,
            pme_tier_mode=pme_tier_mode,
            pressure_atm=pressure_atm,
            barostat_interval=barostat_interval,
            knob_surface=knob_surface,
        )

    @classmethod
    def from_bundles(
        cls,
        bundles: list[Any],
        planner: Any = None,
        *,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
        nonbonded_engine: str | None = None,
        precision: str | None = None,
        nb_k_splits: int | None = None,
        tp_group: int | None = None,
        cmm_frequency: int | None = None,
        deterministic_forces: bool | None = None,
        gb_use_neighbor_list: bool | None = None,
        pme_tier_mode: str | None = None,
        pressure_atm: float | None = None,
        barostat_interval: int | None = None,
        knob_surface: KnobSurface | None = None,
    ) -> EnsemblePlan:
        return cls(
            bundles,
            planner,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
            nonbonded_engine=nonbonded_engine,
            precision=precision,
            nb_k_splits=nb_k_splits,
            tp_group=tp_group,
            cmm_frequency=cmm_frequency,
            deterministic_forces=deterministic_forces,
            gb_use_neighbor_list=gb_use_neighbor_list,
            pme_tier_mode=pme_tier_mode,
            pressure_atm=pressure_atm,
            barostat_interval=barostat_interval,
            knob_surface=knob_surface,
        )

    @classmethod
    def from_openmm_systems(  # noqa: PLR0913
        cls,
        systems: Sequence[Any],
        positions_list: Sequence[Any] | None = None,
        *,
        planner: Any = None,
        integrator: Any = None,
        config: Any = None,
        align_buckets: bool = False,
        use_pallas: bool | None = None,
        use_pallas_pme: bool | None = None,
        nonbonded_engine: str | None = None,
        precision: str | None = None,
        nb_k_splits: int | None = None,
        tp_group: int | None = None,
        cmm_frequency: int | None = None,
        deterministic_forces: bool | None = None,
        gb_use_neighbor_list: bool | None = None,
        pme_tier_mode: str | None = None,
        pressure_atm: float | None = None,
        barostat_interval: int | None = None,
        knob_surface: KnobSurface | None = None,
    ) -> EnsemblePlan:
        """Construct an ``EnsemblePlan`` directly from a sequence of OpenMM ``System``s."""
        if _from_openmm_system is None or _to_akma is None:
            raise ImportError("openmm is required for EnsemblePlan.from_openmm_systems")
        if not systems:
            raise ValueError("from_openmm_systems requires at least one OpenMM System")
        if positions_list is not None and len(positions_list) != len(systems):
            raise ValueError(
                f"Expected {len(systems)} position arrays, got {len(positions_list)}"
            )

        converted_by_id: dict[int, Any] = {}
        bundles: list[Any] = []
        for idx, sys_obj in enumerate(systems):
            pos_raw = positions_list[idx] if positions_list is not None else None
            sys_key = id(sys_obj) if not isinstance(sys_obj, str) else None
            if sys_key is not None and sys_key in converted_by_id:
                base_bundle = converted_by_id[sys_key]
                if pos_raw is not None:
                    pos_akma = np.asarray(_to_akma(pos_raw, "length"), dtype=np.float64)
                    n_real = int(pos_akma.shape[0])
                    new_pos = jnp.asarray(base_bundle.positions).at[:n_real].set(
                        jnp.asarray(pos_akma, dtype=base_bundle.positions.dtype)
                    )
                    bundles.append(dataclasses.replace(base_bundle, positions=new_pos))
                else:
                    bundles.append(base_bundle)
            else:
                conv = _from_openmm_system(
                    sys_obj, integrator=integrator, config=config, positions=pos_raw
                )
                if sys_key is not None:
                    converted_by_id[sys_key] = conv.bundle
                bundles.append(conv.bundle)

        if align_buckets and len(bundles) > 1:
            specs = [b.shape_spec for b in bundles]
            if len(set(specs)) > 1:
                union_spec = _union_shape_spec(specs)
                bundles = [_re_pad_bundle_to_spec(b, union_spec) for b in bundles]

        return cls.from_bundles(
            bundles,
            planner,
            use_pallas=use_pallas,
            use_pallas_pme=use_pallas_pme,
            nonbonded_engine=nonbonded_engine,
            precision=precision,
            nb_k_splits=nb_k_splits,
            tp_group=tp_group,
            cmm_frequency=cmm_frequency,
            deterministic_forces=deterministic_forces,
            gb_use_neighbor_list=gb_use_neighbor_list,
            pme_tier_mode=pme_tier_mode,
            pressure_atm=pressure_atm,
            barostat_interval=barostat_interval,
            knob_surface=knob_surface,
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

    def run(  # noqa: PLR0913
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
        nonbonded_engine: str | None = None,
        precision: str | None = None,
        nb_k_splits: int | None = None,
        tp_group: int | None = None,
        cmm_frequency: int | None = None,
        deterministic_forces: bool | None = None,
        gb_use_neighbor_list: bool | None = None,
        pme_tier_mode: str | None = None,
        pressure_atm: float | None = None,
        target_pressure: float | None = None,
        barostat_interval: int | None = None,
        barostat_every: int | None = None,
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
        eff_baro_int = barostat_interval if barostat_interval is not None else barostat_every
        eff_press = pressure_atm if pressure_atm is not None else target_pressure
        active_knobs = self._default_v35_knobs(
            {
                "nonbonded_engine": nonbonded_engine,
                "precision": precision,
                "nb_k_splits": nb_k_splits,
                "tp_group": tp_group,
                "cmm_frequency": cmm_frequency,
                "deterministic_forces": deterministic_forces,
                "gb_use_neighbor_list": gb_use_neighbor_list,
                "pme_tier_mode": pme_tier_mode,
                "pressure_atm": eff_press,
                "barostat_interval": eff_baro_int,
            }
        )
        eff_engine = str(active_knobs["nonbonded_engine"])
        if eff_engine == "dense" and use_neighbor_list is None and use_flash_forces is None:
            use_neighbor_list = False
            use_flash_forces = False
        elif eff_engine == "flash" and use_neighbor_list is None and use_flash_forces is None:
            use_neighbor_list = False
            use_flash_forces = True
        elif (
            eff_engine in ("morton_tile_pairs", "ragged_tile_pairs", "neighbor_list")
            and use_neighbor_list is None
            and use_flash_forces is None
        ):
            use_neighbor_list = True
            use_flash_forces = False

        used_legacy = use_neighbor_list is not None or use_flash_forces is not None
        cutoff0 = _cached_bundle_cutoff(self.bundles[0], 9.0)
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

        prev_knobs = getattr(self, "_active_v35_knobs", None)
        self._active_v35_knobs = active_knobs
        try:
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
        finally:
            self._active_v35_knobs = prev_knobs

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
        if (
            len(groups) > 1
            and run_mode == "inference"
            and self._can_use_ragged_mk68_runner(
                self.bundles,
                use_neighbor_list=use_neighbor_list,
                use_flash_forces=use_flash_forces,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
            )
        ):
            aligned = _align_bundles_for_stacking(self.bundles)
            if can_jit_vmap_n_mols(aligned):
                with _scoped_active_atom_env(aligned, int(aligned[0].positions.shape[0])):
                    return self._run_ragged_mk68_batch(
                        aligned,
                        n_steps,
                        dt,
                        kT,
                        seed,
                        observables=observables,
                        integration_prefix=int(aligned[0].positions.shape[0]),
                        trim_output=True,
                        dt_unit=dt_unit,
                        gamma=gamma,
                        nl_update_every=nl_update_every,
                        nb_cfg=nb_cfg,
                        use_pallas=use_pallas,
                        use_pallas_pme=use_pallas_pme,
                    )

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

        if any(t is None for t in out):
            raise RuntimeError("Missing trajectory slot in partitioned EnsemblePlan.run")
        return [t for t in out if t is not None]

    def _run_stacked_dispatch(  # noqa: PLR0913
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
            if self._can_use_morton_soa_runner(
                bundles,
                use_neighbor_list=use_neighbor_list,
                use_flash_forces=use_flash_forces,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
            ):
                return self._run_morton_soa_batch(
                    bundles,
                    n_steps,
                    dt,
                    kT,
                    seed,
                    observables=observables,
                    integration_prefix=integration_prefix,
                    trim_output=True,
                    dt_unit=dt_unit,
                    gamma=gamma,
                    nl_update_every=nl_update_every,
                    nb_cfg=nb_cfg,
                    use_pallas=use_pallas,
                )
            if self._can_use_ragged_mk68_runner(
                bundles,
                use_neighbor_list=use_neighbor_list,
                use_flash_forces=use_flash_forces,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
            ):
                return self._run_ragged_mk68_batch(
                    bundles,
                    n_steps,
                    dt,
                    kT,
                    seed,
                    observables=observables,
                    integration_prefix=integration_prefix,
                    trim_output=True,
                    dt_unit=dt_unit,
                    gamma=gamma,
                    nl_update_every=nl_update_every,
                    nb_cfg=nb_cfg,
                    use_pallas=use_pallas,
                    use_pallas_pme=use_pallas_pme,
                )

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
                        bool(jax.config.read("jax_enable_x64")),
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
                        tuple(sorted(self._current_v35_knobs().items())),
                    )

                stacked_cache_key = _make_stacked_cache_key()
                cand_s = _STACKED_INFERENCE_CACHE.get(stacked_cache_key)
                if (
                    cand_s is not None
                    and len(cand_s.get("bundles", ())) == len(bundles)
                    and all(a is b for a, b in zip(cand_s["bundles"], bundles, strict=True))
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

                # MK-64: Detect shared topology index/mask arrays across congeneric replicas
                # so jax.vmap passes in_batched=False (stride=0) to the Pallas megakernel.
                _shared_top_candidates = (
                    "bond_idx",
                    "bond_mask",
                    "angle_idx",
                    "angle_mask",
                    "dihedral_idx",
                    "dihedral_mask",
                    "improper_idx",
                    "improper_mask",
                    "exception_pairs",
                    "exception_mask",
                    "cmap_torsion_idx",
                    "cmap_indices",
                    "cmap_coeffs",
                    "cmap_mask",
                )
                shared_top_fields: tuple[str, ...] = ()
                if n_systems > 1 and can_cache_stacked:
                    _shared_list: list[str] = []
                    for _tf in _shared_top_candidates:
                        _v0 = getattr(bundles[0], _tf, None)
                        if _v0 is None or isinstance(_v0, jax.core.Tracer):
                            continue
                        _v0_np = np.asarray(_v0)
                        _all_eq = True
                        for _b_other in bundles[1:]:
                            _vb = getattr(_b_other, _tf, None)
                            if _vb is _v0:
                                continue
                            if _vb is None or isinstance(_vb, jax.core.Tracer):
                                _all_eq = False
                                break
                            if not np.array_equal(_v0_np, np.asarray(_vb)):
                                _all_eq = False
                                break
                        if _all_eq:
                            _shared_list.append(_tf)
                    shared_top_fields = tuple(_shared_list)

                host_cutoff_0 = float(_host_float(getattr(bundles[0], "cutoff_distance", 9.0), 9.0))
                host_pme_alpha_0 = float(_host_float(getattr(bundles[0], "pme_alpha", 0.34), 0.34))
                n_pad_host_stacked = (
                    int(integration_prefix)
                    if integration_prefix is not None
                    else int(bundles[0].positions.shape[0])
                )
                host_pw_blk_stacked = 0
                for hb_item in bundles:
                    if int(hb_item.positions.shape[0]) < n_pad_host_stacked:
                        continue
                    blk_i = _infer_first_pure_water_block(
                        n_pad_host_stacked,
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
                    host_pw_blk_stacked = max(host_pw_blk_stacked, blk_i)
                _eval_force0_jit = None
                _batched = None
                jit_fn_key = None
                if can_cache_stacked and _make_stacked_cache_key is not None:
                    jit_fn_key = (
                        n_systems,
                        shared_top_fields,
                        host_pw_blk_stacked,
                        tuple(observables) if observables is not None else None,
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
                        host_cutoff_0,
                        host_pme_alpha_0,
                        _make_stacked_cache_key()[1:],
                    )
                    cached_jit_pair = _STACKED_JIT_FN_CACHE.get(jit_fn_key)
                    if cached_jit_pair is not None:
                        _eval_force0_jit, _batched = cached_jit_pair

                if _batched is None:
                    import dataclasses as _dc_ep

                    if can_cache_stacked and use_neighbor_list:
                        def _make_eval_force0(sb_in, snb_in):
                            _shared_0 = (
                                {_tf: getattr(sb_in, _tf)[0] for _tf in shared_top_fields}
                                if shared_top_fields
                                else {}
                            )

                            def _eval_force0_one(bundle_i, _seed_i, nbr_i=None):
                                bundle_i = _dc_ep.replace(
                                    bundle_i,
                                    cutoff_distance=host_cutoff_0,
                                    pme_alpha=host_pme_alpha_0,
                                    **_shared_0,
                                )
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

                            return dispatch_n_mols(
                                plan,
                                n_systems,
                                _eval_force0_one,
                                sb_in,
                                jnp.zeros(n_systems, dtype=jnp.int32),
                                extra_stacked=snb_in,
                            )

                        _eval_force0_jit = jax.jit(_make_eval_force0)

                    @jax.jit
                    def _batched(sb, sd, snb, sf0, n_steps_in):
                        _shared_0 = (
                            {_tf: getattr(sb, _tf)[0] for _tf in shared_top_fields}
                            if shared_top_fields
                            else {}
                        )

                        def _run_one_with_f0(bundle, seed_i, extra=None):
                            bundle = _dc_ep.replace(
                                bundle,
                                cutoff_distance=host_cutoff_0,
                                pme_alpha=host_pme_alpha_0,
                                **_shared_0,
                            )
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
                if not already_verified and overflow_arr is not None and stacked_neighbor is not None:
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
        cutoff = _finite_nl_cutoff(cutoff, box_vec, 9.0)

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

            seed = cast("Any", neighbor_fn).allocate(densest_positions)
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
            _nl_seed_res = nl_mod._pallas_prune_and_encode_cell_list_batch(
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
            idx_b_seed, occ_b_seed, mkt_b_seed = cast("tuple[Any, Any, Any]", _nl_seed_res)
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
                error=cast("Any", nl_mod.partition.PartitionError)(err_b_seed),
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

        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        if prec in ("mixed", "double"):
            if not bool(jax.config.read("jax_enable_x64")):
                jax.config.update("jax_enable_x64", True)
            positions_init = positions_init.astype(jnp.float64)
            masses = masses.astype(jnp.float64)
        elif prec == "single":
            positions_init = positions_init.astype(jnp.float32)
            masses = masses.astype(jnp.float32)
        f64_acc = prec in ("mixed", "double")
        constraints_arg: Any = (
            (
                bundle.constraint_pairs,
                bundle.constraint_lengths.astype(positions_init.dtype),
                bundle.constraint_mask,
            )
            if (
                bool(getattr(bundle.shape_spec, "has_real_constraints", False))
                and getattr(bundle, "constraint_pairs", None) is not None
                and int(bundle.constraint_pairs.shape[-2]) > 0
            )
            else None
        )

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

        _has_ad_tracer = any(
            isinstance(x, jax_ad.JVPTracer)
            for x in (
                dt,
                kT,
                gamma,
                positions_init,
                masses,
                getattr(bundle, "charges", None),
                getattr(bundle, "sigmas", None),
                getattr(bundle, "epsilons", None),
            )
        ) or any(
            isinstance(x, jax.core.Tracer) and type(x).__name__ != "BatchTracer"
            for x in (dt, kT, gamma)
        )
        eff_pallas_here = False if _has_ad_tracer else use_pallas
        eff_pallas_pme_here = False if _has_ad_tracer else use_pallas_pme

        env_up = os.environ.get("PROLIX_USE_PALLAS", "").strip().lower()
        pallas_enabled_here = (
            False
            if _has_ad_tracer
            else (
                bool(eff_pallas_here)
                if eff_pallas_here is not None
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
        nb_engine = str(knobs.get("nonbonded_engine", "auto")).strip().lower()
        n_gb_rows = int(positions_init.shape[0])
        can_use_gbsa_tile = bool(
            can_pallas_gbsa
            and nb_engine in ("auto", "gbsa_tile")
            and prec != "double"
            and not in_export_context()
            and n_gb_rows <= 8192
            and _bundle_md_mod.single_padded_energy is _ORIG_BMD_SINGLE_ENERGY
            and _bundle_md_mod.pallas_bonded_all_fused is _ORIG_BMD_BONDED_FUSED
            and (
                not isinstance(bundle.positions, jax.core.Tracer)
                or all(
                    b.excl_dense_indices is self.bundles[0].excl_dense_indices
                    or np.array_equal(
                        np.asarray(b.excl_dense_indices[:n_gb_rows]),
                        np.asarray(self.bundles[0].excl_dense_indices[:n_gb_rows]),
                    )
                    for b in self.bundles[1:]
                )
            )
        )
        if can_use_gbsa_tile:
            hb_gb = self.bundles[0] if isinstance(bundle.positions, jax.core.Tracer) else bundle
            gb_cutoff: float | None = None
            if bool(knobs.get("gb_use_neighbor_list", False)):
                c_val = float(_host_float(getattr(hb_gb, "cutoff_distance", 0.0), 0.0))
                gb_cutoff = c_val if c_val > 0.0 else 16.0
            gb_cache = getattr(self, "_gbsa_tl_cache", None)
            if gb_cache is None:
                gb_cache = {}
                self._gbsa_tl_cache = gb_cache
            gb_key = (id(hb_gb.excl_dense_indices), n_gb_rows, gb_cutoff)
            gbsa_tl = gb_cache.get(gb_key)
            if gbsa_tl is None:
                with jax.ensure_compile_time_eval():
                    gbsa_tl = build_gbsa_tile_pair_list(
                        np.asarray(hb_gb.positions[:n_gb_rows], dtype=np.float32),
                        atom_mask=np.asarray(hb_gb.atom_mask[:n_gb_rows], dtype=np.int32),
                        excl_indices=np.asarray(hb_gb.excl_dense_indices[:n_gb_rows], dtype=np.int32),
                        excl_scales_vdw=np.asarray(hb_gb.excl_dense_scales_vdw[:n_gb_rows], dtype=np.float32),
                        excl_scales_elec=np.asarray(hb_gb.excl_dense_scales_elec[:n_gb_rows], dtype=np.float32),
                        cutoff=gb_cutoff,
                        born_cutoff=gb_cutoff,
                        use_morton=(gb_cutoff is not None),
                    )
                gb_cache[gb_key] = gbsa_tl
            q_gb = bundle.charges[:n_gb_rows].astype(jnp.float32)
            rad_gb = bundle.radii[:n_gb_rows].astype(jnp.float32)
            sr_gb = bundle.scaled_radii[:n_gb_rows].astype(jnp.float32)
            sig_gb = bundle.sigmas[:n_gb_rows].astype(jnp.float32)
            eps_gb = bundle.epsilons[:n_gb_rows].astype(jnp.float32)
            am_gb = bundle.atom_mask[:n_gb_rows].astype(jnp.int32)
            dp_gb = bundle.dihedral_params
            flat_d_gb = dp_gb.reshape(-1, dp_gb.shape[-1]) if dp_gb.ndim == 3 else dp_gb[..., :3]
            ip_gb = bundle.improper_params
            flat_i_gb = ip_gb.reshape(-1, ip_gb.shape[-1]) if ip_gb.ndim == 3 else ip_gb[..., :3]
            ub_i_gb = (
                bundle.urey_bradley_idx[:, ::2]
                if bundle.urey_bradley_idx.shape[-1] == 3
                else bundle.urey_bradley_idx[:, :2]
            )
            interpret_gb = jax.default_backend() == "cpu"

            def _gbsa_tile_force_fn(pos_in: Any, **_kw: Any) -> Any:
                p_f32 = pos_in.astype(jnp.float32)
                _, f_b = pallas_bonded_all_fused(
                    p_f32,
                    bonds=bundle.bond_idx,
                    bond_params=bundle.bond_params.astype(jnp.float32),
                    bond_mask=bundle.bond_mask,
                    urey_bradley_bonds=ub_i_gb,
                    urey_bradley_params=bundle.urey_bradley_params.astype(jnp.float32),
                    urey_bradley_mask=bundle.urey_bradley_mask,
                    angles=bundle.angle_idx,
                    angle_params=bundle.angle_params.astype(jnp.float32),
                    angle_mask=bundle.angle_mask,
                    dihedrals=bundle.dihedral_idx,
                    dihedral_params=flat_d_gb.astype(jnp.float32),
                    dihedral_mask=bundle.dihedral_mask,
                    impropers=bundle.improper_idx,
                    improper_params=flat_i_gb.astype(jnp.float32),
                    improper_mask=bundle.improper_mask,
                    improper_is_harmonic=(flat_i_gb.shape[-1] == 2),
                    exception_pairs=bundle.exception_pairs,
                    exception_sigmas=bundle.exception_sigmas.astype(jnp.float32),
                    exception_epsilons=bundle.exception_epsilons.astype(jnp.float32),
                    exception_chargeprods=bundle.exception_chargeprods.astype(jnp.float32),
                    exception_mask=bundle.exception_mask,
                    cmap_torsions=bundle.cmap_torsion_idx,
                    cmap_indices=bundle.cmap_indices,
                    cmap_mask=bundle.cmap_mask,
                    cmap_coeffs=bundle.cmap_coeffs.astype(jnp.float32),
                    box=None,
                    dist_eps=1e-10,
                    b0_is_ji=True,
                    cmap_first_is_psi=True,
                    return_forces=True,
                    interpret=interpret_gb,
                )
                _, f_gb = pallas_gbsa_tile_energy_and_forces(
                    p_f32,
                    q_gb,
                    rad_gb,
                    sr_gb,
                    sig_gb,
                    eps_gb,
                    gbsa_tl,
                    atom_mask=am_gb,
                    cutoff=gb_cutoff,
                    born_cutoff=gb_cutoff,
                    interpret=interpret_gb,
                )
                return (f_b + f_gb).astype(pos_in.dtype)

            energy_or_force_fn: Any = _gbsa_tile_force_fn
        else:
            energy_or_force_fn = (
                force_fn_from_bundle(
                    bundle_for_energy,
                    flash_tile_size=tile,
                    flash_remat=remat,
                    use_pallas=eff_pallas_here,
                )
                if use_flash_forces
                else energy_fn_from_bundle(
                    bundle_for_energy,
                    lj_switch_width=lj_switch_width_from_config(nb_cfg),
                    build_dense_exclusions=(initial_neighbor is None and not can_pallas_gbsa),
                    use_pallas=eff_pallas_here,
                    use_pallas_pme=eff_pallas_pme_here,
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
            dt_akma, kT, dtype=positions_init.dtype
        )
        cmm_freq_int = int(knobs.get("cmm_frequency", 0) or 0)

        init_fn, apply_fn = settle_langevin(
            energy_or_force_fn,
            shift_fn,
            dt=1.0,
            kT=1.0,
            gamma=gamma_akma,
            mass=masses,
            water_indices=water_indices,
            constraints=cast("Any", constraints_arg),
            remove_linear_com_momentum=(cmm_freq_int == 1),
            project_ou_momentum_rigid=True,
            water_mask=water_mask,
            atom_mask=atom_mask,
            box=pbc_box_for_bundle(bundle),
            is_force_shaped=bool(use_flash_forces or initial_neighbor is not None or can_pallas_gbsa),
            use_pallas=(False if _has_ad_tracer else (True if can_use_gbsa_tile else eff_pallas_here)),
            f64_accumulators=f64_acc,
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

        knobs = self._current_v35_knobs()
        prec = str(knobs.get("precision") or "").strip().lower()
        if prec in ("mixed", "double"):
            if not bool(jax.config.read("jax_enable_x64")):
                jax.config.update("jax_enable_x64", True)
            positions_init = positions_init.astype(jnp.float64)
            masses = masses.astype(jnp.float64)
        elif prec == "single":
            positions_init = positions_init.astype(jnp.float32)
            masses = masses.astype(jnp.float32)
        f64_acc = prec in ("mixed", "double")
        constraints_arg: Any = (
            (
                bundle.constraint_pairs,
                bundle.constraint_lengths.astype(positions_init.dtype),
                bundle.constraint_mask,
            )
            if (
                bool(getattr(bundle.shape_spec, "has_real_constraints", False))
                and getattr(bundle, "constraint_pairs", None) is not None
                and int(bundle.constraint_pairs.shape[-2]) > 0
            )
            else None
        )

        force0 = state0.force
        fast_init_fn, _ = settle_langevin(
            lambda _r, **_kw: force0,
            shift_fn,
            dt=1.0,
            kT=1.0,
            gamma=gamma_akma,
            mass=masses,
            water_indices=water_indices,
            constraints=cast("Any", constraints_arg),
            project_ou_momentum_rigid=True,
            water_mask=water_mask,
            atom_mask=atom_mask,
            box=pbc_box_for_bundle(bundle),
            is_force_shaped=True,
            f64_accumulators=f64_acc,
        )

        @jax.jit
        def _reseed(seed_u32: Any, kT_val: Any) -> Any:
            k = jax.random.PRNGKey(seed_u32)
            st = fast_init_fn(k, positions_init, kT=kT_val)
            if hasattr(st, "force") and hasattr(state0, "force") and st.force.dtype != state0.force.dtype:
                st = dataclasses.replace(st, force=st.force.astype(state0.force.dtype))
            return st

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
            new_state = apply_fn(carry, kT=kT_s, dt=dt_s, use_pallas_settle=False)
            return new_state, new_state.position

        state, positions_array = dispatch_n_steps(step_fn, state, int(n_steps))
        if trim_output:
            positions_array = trim_trajectory_positions(positions_array, bundle)

        observable_values: dict[str, Any] = {}
        if observables:
            for name, observable in observables.items():
                observable_values[name] = observable.compute(state)

        return cast("Any", Trajectory)(
            positions=positions_array,
            observable_values=observable_values,
            n_steps=n_steps,
        )

    def _run_single_inference(  # noqa: PLR0913
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

        if (
            (isinstance(n_steps, int) or not isinstance(n_steps, jax.core.Tracer))
            and int(n_steps) >= 1
            and self._can_use_morton_soa_runner(
                [bundle],
                use_neighbor_list=use_neighbor_list,
                use_flash_forces=use_flash_forces,
                xtc_path=xtc_path,
                initial_neighbor=initial_neighbor,
                use_pallas=use_pallas,
                use_pallas_pme=use_pallas_pme,
            )
        ):
            return self._run_morton_soa_batch(
                [bundle],
                n_steps,
                dt,
                kT,
                seed,
                observables=observables,
                integration_prefix=integration_prefix,
                trim_output=trim_output,
                dt_unit=dt_unit,
                gamma=gamma,
                nl_update_every=nl_update_every,
                nb_cfg=nb_cfg,
                use_pallas=use_pallas,
            )[0]

        can_cache = (
            xtc_path is None
            and initial_neighbor is None
            and not isinstance(dt, jax.core.Tracer)
            and not isinstance(kT, jax.core.Tracer)
            and not isinstance(seed, jax.core.Tracer)
            and not isinstance(bundle.positions, jax.core.Tracer)
            and not isinstance(bundle.charges, jax.core.Tracer)
        )
        cache_key = None
        cached_entry = None
        if can_cache:
            import importlib
            import sys

            def _make_cache_key() -> tuple[Any, ...]:
                _bmd = sys.modules.get("prolix.api.bundle_md")
                _be = sys.modules.get("prolix.batched_energy")
                _use_pmep = bool(
                    use_pallas_pme
                    if use_pallas_pme is not None
                    else (
                        use_pallas
                        if use_pallas is not None
                        else os.environ.get("PROLIX_USE_PALLAS", "") == "1"
                    )
                )
                if _use_pmep and "prolix.physics.pme_pallas" not in sys.modules:
                    importlib.import_module("prolix.physics.pme_pallas")
                _pmep = sys.modules.get("prolix.physics.pme_pallas", pme_pallas_mod)
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
                    bool(jax.config.read("jax_enable_x64")),
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
                    tuple(sorted(self._current_v35_knobs().items())),
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
            _JITTED_WHILE_CACHE[(cached_entry["nl_step_fn"], "fused_nl")] = cached_entry[
                "fused_nl_jit_fn"
            ]
            positions_array, _ = cached_entry["fused_nl_run_fn"](
                jnp.asarray(seed, dtype=jnp.uint32),
                cached_entry["kT_s"],
                cached_entry["nbr0"],
                jnp.asarray(n_steps, dtype=jnp.int32),
            )
            return cast("Any", Trajectory)(
                positions=positions_array,
                observable_values=cast(
                    "dict[str, Any]",
                    {
                        "_nl_capacity": cached_entry["nl_capacity"],
                        "_nl_retry_count": 0,
                    },
                ),
                n_steps=int(n_steps),
            )

        neighbor_fn = None
        target_capacity = 0
        cutoff = _finite_nl_cutoff(_cached_bundle_cutoff(bundle, 9.0), getattr(bundle, "box", None), 9.0)
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
                seed_positions = (
                    positions_with_prefix(bundle, integration_prefix)
                    if integration_prefix is not None
                    else bundle.positions
                )
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
                nbr0 = cast("Any", neighbor_fn).allocate(seed_positions)
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
            sink_cm = XtcFrameSink(path=Path(xtc_path))
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
                r_total_cutoff = float(cutoff) + skin
                excl_idx_nl = getattr(bundle, "excl_dense_indices", None)
                excl_sv_nl = getattr(bundle, "excl_dense_scales_vdw", None)
                excl_se_nl = getattr(bundle, "excl_dense_scales_elec", None)
                if (
                    excl_idx_nl is not None
                    and excl_sv_nl is not None
                    and excl_se_nl is not None
                ):
                    has_excl_nl = True
                    excl_idx_nl = excl_idx_nl[:n_nl_rows]
                    excl_sv_nl = excl_sv_nl[:n_nl_rows]
                    excl_se_nl = excl_se_nl[:n_nl_rows]
                else:
                    has_excl_nl = False
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
                    new_langevin_state = cast("Any", respa_2step_fn)(
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
                def _rebuild_nl_unconditional(  # noqa: PLR0913
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
                        _nl_u_res = nl_mod._pallas_prune_and_encode_cell_list_batch(
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
                        idx_new_b, occ_new_b, mkt_new_b = cast("tuple[Any, Any, Any]", _nl_u_res)
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
                        error=cast("Any", nl_mod.partition.PartitionError)(err_in),
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
                def _rebuild_nl_unconditional_vmap(  # noqa: PLR0913
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
                        _nl_v_res = nl_mod._pallas_prune_and_encode_cell_list_batch(
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
                        idx_new_b, occ_new_b, mkt_new_b = cast("tuple[Any, Any, Any]", _nl_v_res)
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
                                error=cast("Any", nl_mod.partition.PartitionError)(e_1),
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
                    def _maybe_rebuild_nl(  # noqa: PLR0913
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
                    def _maybe_rebuild_nl_vmap(  # noqa: PLR0913
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
                        error=cast("Any", nl_mod.partition.PartitionError)(new_err_code),
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
                                    pos_next = st_next.position.astype(p_i.dtype)
                                    su_next = _check_su(pos_next, lp_fixed, s_next)
                                    any_u_next = _any_replica_true(su_next)
                                    return (
                                        s_next,
                                        pos_next,
                                        st_next.momentum.astype(m_i.dtype),
                                        st_next.force.astype(f_i.dtype),
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
                                pos_next = st_next.position.astype(p_i.dtype)
                                su_next = _check_su(pos_next, lp_fixed, s_next)
                                any_u_next = _any_replica_true(su_next)
                                return (
                                    s_next,
                                    pos_next,
                                    st_next.momentum.astype(m_i.dtype),
                                    st_next.force.astype(f_i.dtype),
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
                            error=cast("Any", nl_mod.partition.PartitionError)(new_err),
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

                cast("Any", _nl_step_fn)._two_level_runner = _two_level_runner


            if can_cache and cached_entry is None and cache_key is not None:
                fast_reseed_fn = self._build_fast_reseed_fn(
                    bundle, state, integration_prefix=integration_prefix, gamma=gamma
                )
                raw_n_b = bundle.n_atoms
                n_real_cached = raw_n_b if isinstance(raw_n_b, int) else int(jnp.asarray(raw_n_b))
                kT_dtype_cached = _integrator_positions(state).dtype
                nbr_template = nbr0

                def _fused_nl_body(
                    seed_u32: Any,
                    kT_arr: Any,
                    idx_in: Any,
                    ref_in: Any,
                    err_in: Any,
                    mkt_in: Any,
                    n_steps_i32: Any,
                    _tag: int = 0,
                ) -> Any:
                    del _tag
                    st_reseeded = fast_reseed_fn(seed_u32, kT_arr)
                    init_p = _integrator_positions(st_reseeded)
                    nbr_rebuilt = dataclasses.replace(
                        nbr_template,
                        idx=idx_in,
                        reference_position=ref_in,
                        error=cast("Any", nl_mod.partition.PartitionError)(err_in),
                        max_k_tiles=mkt_in,
                    )
                    c0 = _NLDispatchCarry(st_reseeded, nbr_rebuilt, jnp.array(False), init_p)
                    fc = dispatch_n_steps_inference(_nl_step_fn, c0, n_steps_i32)
                    return fc.langevin_state.position[None, :n_real_cached, :], fc.did_overflow

                _fused_nl_jit = jax.jit(_fused_nl_body, static_argnums=(7,))

                def _fused_nl_run(seed_u32: Any, kT_val: Any, nbr_in: Any, n_steps_i32: Any) -> Any:
                    idx_arr = jnp.asarray(nbr_in.idx)
                    ref_arr = jnp.asarray(nbr_in.reference_position)
                    err_arr = jnp.asarray(nbr_in.error.code)
                    mkt_raw = getattr(nbr_in, "max_k_tiles", None)
                    mkt_arr = (
                        jnp.asarray(mkt_raw)
                        if mkt_raw is not None
                        else nl_mod._max_k_tiles_from_idx(idx_arr)
                    )
                    s_arr = jnp.asarray(seed_u32, dtype=jnp.uint32)
                    k_arr = jnp.asarray(kT_val, dtype=kT_dtype_cached)
                    ns_arr = jnp.asarray(n_steps_i32, dtype=jnp.int32)
                    dev = getattr(idx_arr, "device", None)
                    if getattr(idx_arr, "_committed", False) and isinstance(dev, jax.Device):
                        s_arr = jax.device_put(s_arr, dev)
                        k_arr = jax.device_put(k_arr, dev)
                        ref_arr = jax.device_put(ref_arr, dev)
                        err_arr = jax.device_put(err_arr, dev)
                        mkt_arr = jax.device_put(mkt_arr, dev)
                        ns_arr = jax.device_put(ns_arr, dev)
                    return _fused_nl_jit(
                        s_arr, k_arr, idx_arr, ref_arr, err_arr, mkt_arr, ns_arr, 0
                    )

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
                    "fused_nl_jit_fn": _fused_nl_jit,
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
                    positions_array, did_overflow_1 = _fused_nl_run(
                        jnp.asarray(seed, dtype=jnp.uint32),
                        kT_s,
                        nbr0,
                        jnp.asarray(n_steps, dtype=jnp.int32),
                    )
                    if not bool(did_overflow_1):
                        if len(_JITTED_WHILE_CACHE) >= 128:
                            _JITTED_WHILE_CACHE.pop(next(iter(_JITTED_WHILE_CACHE)))
                        _JITTED_WHILE_CACHE[(_nl_step_fn, "fused_nl")] = _fused_nl_jit
                        cached_entry["verified_no_overflow"] = True
                        return cast("Any", Trajectory)(
                            positions=positions_array,
                            observable_values=cast(
                                "dict[str, Any]",
                                {
                                    "_nl_capacity": cached_entry["nl_capacity"],
                                    "_nl_retry_count": 0,
                                },
                            ),
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
                        cast("Any", neighbor_fn).allocate(
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
                    st_next = apply_fn(carry, kT=kT_s, dt=dt_s)
                    if hasattr(st_next, "force") and hasattr(carry, "force") and st_next.force.dtype != carry.force.dtype:
                        st_next = dataclasses.replace(st_next, force=st_next.force.astype(carry.force.dtype))
                    return st_next

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
        _raw_n_real = cached_entry.get("n_real") if cached_entry is not None else None
        n_real_opt: int | None = int(cast("Any", _raw_n_real)) if _raw_n_real is not None else None
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

        return cast("Any", Trajectory)(
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
