"""1D prefix-sum ragged CTA work queue & cost-modeled multi-tier PME engine (MK-68).

Eliminates ghost-CTA launch waste and redundant uniform FFT grid volume on
cross-size ragged molecular batches (e.g. ``1UAO`` 2k alongside ``DHFR`` 24k):

1. ``RaggedWorkQueue``: Flattens 2D ``(B, N_blocks_max)`` ragged grids into a
   compact 1D CTA work queue ``total_blocks = sum_b ceil(N_real[b] / tile_size)``
   with precomputed host LUTs ``(cta_to_replica, cta_to_local_block)`` so ghost
   CTAs on smaller replicas are never launched.
2. ``PMETierPlan`` & ``plan_pme_tiers``: Groups replicas into snapped 5-smooth
   3D FFT resolution tiers (e.g. ``32^3``, ``48^3``, ``60^3``) governed by an
   explicit dynamic-programming launch-vs-FFT cost model comparing saved FFT
   grid volume ``(B * G_max^3 - sum_t B_t * G_t^3)`` against extra cuFFT tier
   launch latency (``~15 us/tier``). Exposes ``pme_tier_mode="auto" | "grouped" | "uniform"``.
3. ``spread_charges_pallas_ragged``, ``gather_pallas_ragged``,
   ``spme_forces_ragged_tiered``, ``spme_energy_ragged_tiered``, and
   ``pallas_nonbonded_tile_pairs_ragged``: Pallas GPU kernels with ``custom_vjp``
   and ``prolix.export.in_export_context()`` pure-JAX fallbacks.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.scipy.special import erf, erfc

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.optimization import _lj_switch_weight_and_deriv
from prolix.physics.pallas_bonded import (
    _pl_atomic_add_1d,
    _pl_atomic_add_i32_1d,
    _reconstruct_hi_lo_i32_to_f32,
    _split_f32_to_hi_lo_i32,
    _triton_erf,
    _triton_erfc,
    _triton_gpu_available,
)
from prolix.physics.pbc import invert_box_lower_triangular
from prolix.physics.pme import (
    COULOMB_CONSTANT,
    _gather_reference,
    _is_perturbed,
    _spread_charges_reference,
    _unwrap_primal,
    compute_openmm_pme_params,
    influence_function,
    scale_influence_function_isotropic,
    snap_fft_dim,
    spme_reciprocal_energy,
    spme_self_energy,
)
from prolix.physics.pme_pallas import (
    BLOCK_N as _PME_BLOCK_N,
)
from prolix.physics.pme_pallas import (
    _bspline4_deriv_vec,
    _bspline4_vec,
    _select_stencil,
)

PMETierMode = Literal["auto", "grouped", "uniform"]

_NB_BLOCK_N = 32
_VALID_TIER_MODES: frozenset[str] = frozenset({"auto", "grouped", "uniform"})


def _resolve_interpret(interpret: bool | None) -> bool:
    if interpret is not None:
        return bool(interpret)
    if os.environ.get("PROLIX_PALLAS_INTERPRET", "").strip() == "1":
        return True
    return not _triton_gpu_available()


def _any_block_spec() -> Any:
    bs_cls: Any = pl.BlockSpec
    any_mem: Any = pl.ANY
    return bs_cls(memory_space=any_mem)


# =============================================================================
# 1. 1D Prefix-Sum Ragged CTA Work Queue (MK-68a)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RaggedWorkQueue:
    """Precomputed 1D prefix-sum CTA work queue for ragged multi-replica grids.

    Registered as a JAX Pytree: device LUT arrays pass dynamically across ``jax.jit``
    boundaries with zero host synchronization while ``total_blocks``, ``tile_size``,
    ``batch_size``, and ``n_real_tuple`` remain static concrete metadata for ``grid=(total_blocks,)``.
    """

    cta_to_replica: jax.Array
    cta_to_local_block: jax.Array
    n_real_per_replica: jax.Array
    blocks_per_replica: jax.Array
    block_offsets: jax.Array
    total_blocks: int
    tile_size: int
    batch_size: int
    n_real_tuple: tuple[int, ...]
    uniform_blocks_per_replica: int

    def __init__(
        self,
        cta_to_replica: jax.Array,
        cta_to_local_block: jax.Array,
        n_real_per_replica: jax.Array,
        blocks_per_replica: jax.Array,
        block_offsets: jax.Array,
        total_blocks: int,
        tile_size: int,
        batch_size: int,
        n_real_tuple: tuple[int, ...],
        uniform_blocks_per_replica: int,
    ) -> None:
        object.__setattr__(self, "cta_to_replica", cta_to_replica)
        object.__setattr__(self, "cta_to_local_block", cta_to_local_block)
        object.__setattr__(self, "n_real_per_replica", n_real_per_replica)
        object.__setattr__(self, "blocks_per_replica", blocks_per_replica)
        object.__setattr__(self, "block_offsets", block_offsets)
        object.__setattr__(self, "total_blocks", total_blocks)
        object.__setattr__(self, "tile_size", tile_size)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "n_real_tuple", n_real_tuple)
        object.__setattr__(self, "uniform_blocks_per_replica", uniform_blocks_per_replica)

    @property
    def uniform_total_blocks(self) -> int:
        return int(self.batch_size * self.uniform_blocks_per_replica)

    @property
    def ghost_cta_fraction(self) -> float:
        denom = self.uniform_total_blocks
        if denom <= 0:
            return 0.0
        return max(0.0, 1.0 - float(self.total_blocks) / float(denom))

    def tree_flatten(self):
        children = (
            self.cta_to_replica,
            self.cta_to_local_block,
            self.n_real_per_replica,
            self.blocks_per_replica,
            self.block_offsets,
        )
        aux_data = (
            self.total_blocks,
            self.tile_size,
            self.batch_size,
            self.n_real_tuple,
            self.uniform_blocks_per_replica,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        cta_rep, cta_blk, n_real, blks, offsets = children
        total_blocks, tile_size, batch_size, n_real_tuple, uniform_blks = aux_data
        return cls(
            cta_to_replica=cta_rep,
            cta_to_local_block=cta_blk,
            n_real_per_replica=n_real,
            blocks_per_replica=blks,
            block_offsets=offsets,
            total_blocks=total_blocks,
            tile_size=tile_size,
            batch_size=batch_size,
            n_real_tuple=n_real_tuple,
            uniform_blocks_per_replica=uniform_blks,
        )


jax.tree_util.register_pytree_node_class(RaggedWorkQueue)


def _infer_n_real_np(
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None,
    atom_mask: np.ndarray | jax.Array | None,
    batch_size_hint: int | None = None,
    padded_size_hint: int | None = None,
) -> np.ndarray:
    if n_real_per_replica is not None:
        arr = np.asarray(n_real_per_replica, dtype=np.int32).reshape(-1)
        return np.maximum(arr, 0)
    if atom_mask is not None and not isinstance(atom_mask, jax.core.Tracer):
        am_np = np.asarray(atom_mask, dtype=bool)
        if am_np.ndim == 1:
            am_np = am_np[None, :]
        B = am_np.shape[0]
        counts = np.zeros((B,), dtype=np.int32)
        for b in range(B):
            nz = np.nonzero(am_np[b])[0]
            counts[b] = int(nz[-1]) + 1 if nz.size > 0 else 0
        return counts
    if batch_size_hint is not None and padded_size_hint is not None:
        return np.full((int(batch_size_hint),), max(0, int(padded_size_hint)), dtype=np.int32)
    raise ValueError(
        "Either concrete n_real_per_replica or non-traced atom_mask must be supplied "
        "to build a static 1D RaggedWorkQueue."
    )


def build_ragged_work_queue(
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    *,
    tile_size: int = 32,
    padded_size: int | None = None,
    atom_mask: np.ndarray | jax.Array | None = None,
) -> RaggedWorkQueue:
    """Build a 1D prefix-sum CTA work queue ``total_blocks = sum_b ceil(N_real[b] / tile_size)``.

    Args:
        n_real_per_replica: Active count per replica ``(B,)`` (atoms or tile pairs).
        tile_size: Items processed per CTA (e.g. ``32`` atoms or ``tp_group`` tile pairs).
        padded_size: Optional padded bucket dimension ``N_pad`` used to report uniform CTA count.
        atom_mask: Optional ``(B, N)`` mask used to infer ``n_real_per_replica`` if omitted.

    Returns:
        ``RaggedWorkQueue`` with ``(cta_to_replica, cta_to_local_block)`` device int32 LUTs.
    """
    ts = max(1, int(tile_size))
    if padded_size is None and atom_mask is not None:
        padded_size = int(np.shape(atom_mask)[-1])
    n_real_np = _infer_n_real_np(
        n_real_per_replica,
        atom_mask,
        batch_size_hint=None,
        padded_size_hint=padded_size,
    )
    if padded_size is not None:
        n_real_np = np.minimum(n_real_np, max(0, int(padded_size)))

    B = int(n_real_np.shape[0])
    blocks_np = ((n_real_np + (ts - 1)) // ts).astype(np.int32)
    offsets_np = np.zeros((B + 1,), dtype=np.int32)
    if B > 0:
        np.cumsum(blocks_np, out=offsets_np[1:])
    total_blocks = int(offsets_np[-1])

    if total_blocks > 0:
        cta_rep_np = np.repeat(np.arange(B, dtype=np.int32), blocks_np)
        cta_blk_np = (
            np.arange(total_blocks, dtype=np.int32)
            - np.repeat(offsets_np[:-1], blocks_np)
        ).astype(np.int32)
    else:
        cta_rep_np = np.zeros((0,), dtype=np.int32)
        cta_blk_np = np.zeros((0,), dtype=np.int32)

    if padded_size is not None and int(padded_size) > 0:
        uniform_blks = (int(padded_size) + ts - 1) // ts
    elif B > 0:
        uniform_blks = int(np.max(blocks_np))
    else:
        uniform_blks = 0

    return RaggedWorkQueue(
        cta_to_replica=jnp.asarray(cta_rep_np, dtype=jnp.int32),
        cta_to_local_block=jnp.asarray(cta_blk_np, dtype=jnp.int32),
        n_real_per_replica=jnp.asarray(n_real_np, dtype=jnp.int32),
        blocks_per_replica=jnp.asarray(blocks_np, dtype=jnp.int32),
        block_offsets=jnp.asarray(offsets_np, dtype=jnp.int32),
        total_blocks=total_blocks,
        tile_size=ts,
        batch_size=B,
        n_real_tuple=tuple(int(x) for x in n_real_np.tolist()),
        uniform_blocks_per_replica=int(uniform_blks),
    )


def select_adaptive_tp_group(total_tile_pairs: int, tp_group: int | None = None) -> int:
    """Choose ``tp_group`` (tile pairs per CTA) to saturate SMs across small and large batches."""
    if tp_group is not None and int(tp_group) > 0:
        return int(tp_group)
    env_tpg = os.environ.get("PROLIX_TP_GROUP", "").strip()
    if env_tpg:
        return max(1, int(env_tpg))
    tot = max(0, int(total_tile_pairs))
    if tot < 512:
        return 1
    if tot < 1024:
        return 2
    if tot < 2048:
        return 4
    return 8


def build_ragged_tile_pair_queue(
    num_valid_pairs: Sequence[int] | np.ndarray | jax.Array,
    *,
    padded_pairs: int | None = None,
    tp_group: int | None = None,
    max_tile_pairs: int | None = None,
) -> RaggedWorkQueue:
    """Build a 1D CTA work queue over variable-length tile-pair counts ``num_valid_pairs[b]``."""
    eff_max = padded_pairs if padded_pairs is not None else max_tile_pairs
    nv_np = np.maximum(np.asarray(num_valid_pairs, dtype=np.int32).reshape(-1), 0)
    if eff_max is not None:
        nv_np = np.minimum(nv_np, max(0, int(eff_max)))
    tpg = select_adaptive_tp_group(int(np.sum(nv_np)), tp_group=tp_group)
    return build_ragged_work_queue(
        nv_np,
        tile_size=tpg,
        padded_size=eff_max,
    )


# =============================================================================
# 2. Cost-Modeled Multi-Resolution PME Tier Grouping (MK-68b)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PMETierGroup:
    """One resolution tier within a ``PMETierPlan``."""

    replica_indices: jax.Array
    ragged_queue: RaggedWorkQueue
    G_cached: jax.Array | None
    box_ref: jax.Array | None
    grid_dims: tuple[int, int, int]
    replica_indices_tuple: tuple[int, ...]
    n_slice: int

    def __init__(
        self,
        replica_indices: jax.Array,
        ragged_queue: RaggedWorkQueue,
        G_cached: jax.Array | None,
        box_ref: jax.Array | None,
        grid_dims: tuple[int, int, int],
        replica_indices_tuple: tuple[int, ...],
        n_slice: int,
    ) -> None:
        object.__setattr__(self, "replica_indices", replica_indices)
        object.__setattr__(self, "ragged_queue", ragged_queue)
        object.__setattr__(self, "G_cached", G_cached)
        object.__setattr__(self, "box_ref", box_ref)
        object.__setattr__(self, "grid_dims", grid_dims)
        object.__setattr__(self, "replica_indices_tuple", replica_indices_tuple)
        object.__setattr__(self, "n_slice", n_slice)

    @property
    def batch_size(self) -> int:
        return len(self.replica_indices_tuple)

    @property
    def grid_volume(self) -> int:
        return int(self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2])

    def tree_flatten(self):
        children = (
            self.replica_indices,
            self.ragged_queue,
            self.G_cached,
            self.box_ref,
        )
        aux_data = (
            self.grid_dims,
            self.replica_indices_tuple,
            self.n_slice,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        rep_idx, rq, g_cached, box_ref = children
        grid_dims, rep_tuple, n_slice = aux_data
        return cls(
            replica_indices=rep_idx,
            ragged_queue=rq,
            G_cached=g_cached,
            box_ref=box_ref,
            grid_dims=grid_dims,
            replica_indices_tuple=rep_tuple,
            n_slice=n_slice,
        )


jax.tree_util.register_pytree_node_class(PMETierGroup)


@dataclasses.dataclass(frozen=True)
class PMETierPlan:
    """Cost-modeled multi-resolution PME execution plan across ``B`` replicas."""

    tiers: tuple[PMETierGroup, ...]
    uniform_queue: RaggedWorkQueue
    mode: PMETierMode
    alpha: float
    order: int
    uniform_grid_dims: tuple[int, int, int]
    uniform_volume: int
    tiered_volume: int
    estimated_uniform_us: float
    estimated_tiered_us: float
    use_tiered: bool
    padded_atoms: int

    def __init__(
        self,
        tiers: tuple[PMETierGroup, ...],
        uniform_queue: RaggedWorkQueue,
        mode: PMETierMode,
        alpha: float,
        order: int,
        uniform_grid_dims: tuple[int, int, int],
        uniform_volume: int,
        tiered_volume: int,
        estimated_uniform_us: float,
        estimated_tiered_us: float,
        use_tiered: bool,
        padded_atoms: int,
    ) -> None:
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "uniform_queue", uniform_queue)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "uniform_grid_dims", uniform_grid_dims)
        object.__setattr__(self, "uniform_volume", uniform_volume)
        object.__setattr__(self, "tiered_volume", tiered_volume)
        object.__setattr__(self, "estimated_uniform_us", estimated_uniform_us)
        object.__setattr__(self, "estimated_tiered_us", estimated_tiered_us)
        object.__setattr__(self, "use_tiered", use_tiered)
        object.__setattr__(self, "padded_atoms", padded_atoms)

    def tree_flatten(self):
        children = (self.tiers, self.uniform_queue)
        aux_data = (
            self.mode,
            self.alpha,
            self.order,
            self.uniform_grid_dims,
            self.uniform_volume,
            self.tiered_volume,
            self.estimated_uniform_us,
            self.estimated_tiered_us,
            self.use_tiered,
            self.padded_atoms,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tiers, uniform_queue = children
        (
            mode,
            alpha,
            order,
            uniform_grid_dims,
            uniform_volume,
            tiered_volume,
            estimated_uniform_us,
            estimated_tiered_us,
            use_tiered,
            padded_atoms,
        ) = aux_data
        return cls(
            tiers=tiers,
            uniform_queue=uniform_queue,
            mode=mode,
            alpha=alpha,
            order=order,
            uniform_grid_dims=uniform_grid_dims,
            uniform_volume=uniform_volume,
            tiered_volume=tiered_volume,
            estimated_uniform_us=estimated_uniform_us,
            estimated_tiered_us=estimated_tiered_us,
            use_tiered=use_tiered,
            padded_atoms=padded_atoms,
        )


jax.tree_util.register_pytree_node_class(PMETierPlan)


def _snap_to_tier_dims(
    raw_dims: tuple[int, int, int],
    candidate_tier_dims: Sequence[int] | None,
) -> tuple[int, int, int]:
    snapped = (
        snap_fft_dim(raw_dims[0]),
        snap_fft_dim(raw_dims[1]),
        snap_fft_dim(raw_dims[2]),
    )
    if not candidate_tier_dims:
        return snapped
    cands = sorted(snap_fft_dim(int(c)) for c in candidate_tier_dims if int(c) > 0)
    if not cands:
        return snapped
    out: list[int] = []
    for d in snapped:
        chosen = None
        for c in cands:
            if c >= d:
                chosen = c
                break
        out.append(chosen if chosen is not None else snap_fft_dim(d))
    return (out[0], out[1], out[2])


def _is_lower_tri_3x3_arr(b_arr: np.ndarray | jax.Array) -> bool:
    """Return True if a concrete ``(3, 3)`` array has zeros above the main diagonal."""
    if b_arr.ndim != 2 or b_arr.shape != (3, 3):
        return False
    if not isinstance(b_arr, jax.core.Tracer):
        b_np = np.asarray(b_arr)
        return bool(b_np[0, 1] == 0.0 and b_np[0, 2] == 0.0 and b_np[1, 2] == 0.0)
    return False


def _has_nonzero_lower_tilt_np(b_np: np.ndarray) -> bool:
    return bool(
        np.any(b_np[..., 1, 0] != 0.0)
        or np.any(b_np[..., 2, 0] != 0.0)
        or np.any(b_np[..., 2, 1] != 0.0)
    )


def _infer_pme_is_triclinic(  # noqa: PLR0911
    box_size: np.ndarray | jax.Array,
    B: int,
    *,
    single_input: bool = False,
    is_triclinic: bool = False,
) -> bool:
    """Infer triclinic mode safely across ``B == 3`` ``(3, 3)`` orthorhombic batches."""
    if is_triclinic:
        return True
    b_arr = jnp.asarray(box_size)
    if b_arr.ndim == 3 and b_arr.shape[-2:] == (3, 3):
        if not isinstance(b_arr, jax.core.Tracer):
            return _has_nonzero_lower_tilt_np(np.asarray(b_arr))
        return True
    if b_arr.ndim == 2:
        if b_arr.shape[-1] == 9:
            if not isinstance(b_arr, jax.core.Tracer):
                return _has_nonzero_lower_tilt_np(np.asarray(b_arr).reshape(-1, 3, 3))
            return True
        if b_arr.shape[-1] == 6:
            if not isinstance(b_arr, jax.core.Tracer):
                return bool(np.any(np.asarray(b_arr)[:, 3:6] != 0.0))
            return True
        if b_arr.shape == (3, 3):
            if (not single_input) and B == 3 and not _is_lower_tri_3x3_arr(b_arr):
                return False
            if isinstance(b_arr, jax.core.Tracer):
                return single_input or B != 3
            if not _is_lower_tri_3x3_arr(b_arr):
                return False
            return _has_nonzero_lower_tilt_np(np.asarray(b_arr))
        return False
    if b_arr.ndim == 1:
        if b_arr.size == 6:
            if not isinstance(b_arr, jax.core.Tracer):
                return bool(np.any(np.asarray(b_arr)[3:6] != 0.0))
            return True
        if (single_input or B != 3) and b_arr.size == 9:
            if not isinstance(b_arr, jax.core.Tracer):
                return _has_nonzero_lower_tilt_np(np.asarray(b_arr).reshape(3, 3))
            return True
        if (not single_input) and B > 0 and b_arr.size == B * 9:
            if not isinstance(b_arr, jax.core.Tracer):
                return _has_nonzero_lower_tilt_np(np.asarray(b_arr).reshape(B, 3, 3))
            return True
    return False


def _extract_box_batch_for_tiers_np(  # noqa: PLR0911
    box_sizes: np.ndarray | jax.Array,
    *,
    is_triclinic: bool = False,
    batch_size_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract ``(lengths_np (B, 3), box_ref_np (B, 3) | (B, 3, 3), any_triclinic)`` on host."""
    b_np = np.asarray(box_sizes, dtype=np.float64)
    b_hint = int(batch_size_hint) if batch_size_hint is not None else None

    if b_np.ndim == 3 and b_np.shape[-2:] == (3, 3):
        lengths = np.diagonal(b_np, axis1=-2, axis2=-1)[:, :3]
        any_tri = bool(is_triclinic) or _has_nonzero_lower_tilt_np(b_np)
        return lengths, (b_np if any_tri else lengths), any_tri

    if b_np.ndim == 2:
        if b_np.shape[-1] == 9:
            mats = b_np.reshape(-1, 3, 3)
            lengths = np.diagonal(mats, axis1=-2, axis2=-1)[:, :3]
            any_tri = bool(is_triclinic) or _has_nonzero_lower_tilt_np(mats)
            return lengths, (mats if any_tri else lengths), any_tri
        if b_np.shape[-1] == 6:
            B = b_np.shape[0]
            lengths = b_np[:, :3]
            any_tri = bool(is_triclinic or np.any(b_np[:, 3:6] != 0.0))
            if not any_tri:
                return lengths, lengths, False
            mats = np.zeros((B, 3, 3), dtype=np.float64)
            mats[:, 0, 0] = b_np[:, 0]
            mats[:, 1, 1] = b_np[:, 1]
            mats[:, 2, 2] = b_np[:, 2]
            mats[:, 1, 0] = b_np[:, 3]
            mats[:, 2, 0] = b_np[:, 4]
            mats[:, 2, 1] = b_np[:, 5]
            return lengths, mats, True
        if b_np.shape == (3, 3):
            is_low = bool(b_np[0, 1] == 0.0 and b_np[0, 2] == 0.0 and b_np[1, 2] == 0.0)
            is_single_mat = bool(is_triclinic) or (is_low and b_hint != 3) or (is_low and _has_nonzero_lower_tilt_np(b_np))
            if is_single_mat:
                B_out = b_hint if (b_hint is not None and b_hint > 0) else 1
                diag = np.broadcast_to(np.diag(b_np)[None, :], (B_out, 3)).copy()
                any_tri = bool(is_triclinic) or _has_nonzero_lower_tilt_np(b_np)
                if any_tri:
                    mats = np.broadcast_to(b_np[None, :, :], (B_out, 3, 3)).copy()
                    return diag, mats, True
                return diag, diag, False
            lengths = b_np[:, :3]
            if is_triclinic:
                mats = np.zeros((3, 3, 3), dtype=np.float64)
                mats[:, 0, 0] = lengths[:, 0]
                mats[:, 1, 1] = lengths[:, 1]
                mats[:, 2, 2] = lengths[:, 2]
                return lengths, mats, True
            return lengths, lengths, False
        lengths = b_np[:, :3]
        if is_triclinic:
            B = lengths.shape[0]
            mats = np.zeros((B, 3, 3), dtype=np.float64)
            mats[:, 0, 0] = lengths[:, 0]
            mats[:, 1, 1] = lengths[:, 1]
            mats[:, 2, 2] = lengths[:, 2]
            return lengths, mats, True
        return lengths, lengths, False

    if b_np.ndim == 1:
        if b_np.size == 6:
            B_out = b_hint if (b_hint is not None and b_hint > 0) else 1
            lengths = np.broadcast_to(b_np[:3][None, :], (B_out, 3)).copy()
            any_tri = bool(is_triclinic or np.any(b_np[3:6] != 0.0))
            if not any_tri:
                return lengths, lengths, False
            mat1 = np.array(
                [[b_np[0], 0.0, 0.0], [b_np[3], b_np[1], 0.0], [b_np[4], b_np[5], b_np[2]]],
                dtype=np.float64,
            )
            return lengths, np.broadcast_to(mat1[None, :, :], (B_out, 3, 3)).copy(), True
        if b_hint is not None and b_hint > 0 and b_np.size == b_hint * 9:
            mats = b_np.reshape(b_hint, 3, 3)
            lengths = np.diagonal(mats, axis1=-2, axis2=-1)[:, :3]
            any_tri = bool(is_triclinic) or _has_nonzero_lower_tilt_np(mats)
            return lengths, (mats if any_tri else lengths), any_tri
        if b_np.size == 9 and (is_triclinic or b_hint != 3):
            mat1 = b_np.reshape(3, 3)
            B_out = b_hint if (b_hint is not None and b_hint > 0) else 1
            lengths = np.broadcast_to(np.diag(mat1)[None, :], (B_out, 3)).copy()
            any_tri = bool(is_triclinic) or _has_nonzero_lower_tilt_np(mat1)
            if any_tri:
                return lengths, np.broadcast_to(mat1[None, :, :], (B_out, 3, 3)).copy(), True
            return lengths, lengths, False
        if b_hint is not None and b_hint > 0 and b_np.size == b_hint * 3:
            lengths = b_np.reshape(b_hint, 3)
            return lengths, lengths, False
        if b_np.size >= 3 and (b_np.size % 9 == 0) and is_triclinic:
            mats = b_np.reshape(-1, 3, 3)
            lengths = np.diagonal(mats, axis1=-2, axis2=-1)[:, :3]
            return lengths, mats, True
        B_out = b_hint if (b_hint is not None and b_hint > 0) else 1
        lengths = np.broadcast_to(b_np[:3][None, :], (B_out, 3)).copy()
        return lengths, lengths, False

    raise ValueError(f"Unsupported box_sizes shape: {b_np.shape}")


def _extract_ortho_lengths_np(
    box_sizes: np.ndarray | jax.Array,
    *,
    is_triclinic: bool = False,
    batch_size_hint: int | None = None,
) -> np.ndarray:
    lengths, _, _ = _extract_box_batch_for_tiers_np(
        box_sizes, is_triclinic=is_triclinic, batch_size_hint=batch_size_hint
    )
    return lengths


def _solve_optimal_tier_partition(
    sorted_dims: list[tuple[int, int, int]],
    *,
    tier_launch_overhead_us: float,
    fft_points_per_us: float,
    max_tiers: int,
) -> tuple[list[tuple[int, int, tuple[int, int, int]]], float]:
    """Exact dynamic programming over contiguous sorted replica ranges ``[l..r]``.

    Minimizes ``sum_t (tier_launch_overhead_us + (r_t - l_t + 1) * V(l_t, r_t) / fft_points_per_us)``.
    """
    B = len(sorted_dims)
    if B == 0:
        return [], 0.0
    t_cap = max(1, min(int(max_tiers), B))
    launch_us = max(0.0, float(tier_launch_overhead_us))
    pts_per_us = max(1e-6, float(fft_points_per_us))

    span_dims: dict[tuple[int, int], tuple[int, int, int]] = {}
    span_cost: np.ndarray = np.zeros((B, B), dtype=np.float64)
    for l_idx in range(B):
        mx, my, mz = sorted_dims[l_idx]
        for r_idx in range(l_idx, B):
            dx, dy, dz = sorted_dims[r_idx]
            mx = max(mx, dx)
            my = max(my, dy)
            mz = max(mz, dz)
            span_dims[(l_idx, r_idx)] = (mx, my, mz)
            vol = float(mx * my * mz)
            span_cost[l_idx, r_idx] = launch_us + float(r_idx - l_idx + 1) * vol / pts_per_us

    inf = float("inf")
    dp = np.full((t_cap + 1, B + 1), inf, dtype=np.float64)
    prev = np.full((t_cap + 1, B + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for t in range(1, t_cap + 1):
        for i in range(t, B + 1):
            best_val = inf
            best_j = -1
            for j in range(t - 1, i):
                cand = dp[t - 1, j] + span_cost[j, i - 1]
                if cand < best_val:
                    best_val = cand
                    best_j = j
            dp[t, i] = best_val
            prev[t, i] = best_j

    best_t = 1
    best_cost = dp[1, B]
    # Strict '<' with tiny epsilon ensures ties prefer fewer kernel launches.
    for t in range(2, t_cap + 1):
        if dp[t, B] + 1e-9 < best_cost:
            best_cost = dp[t, B]
            best_t = t

    ranges_rev: list[tuple[int, int, tuple[int, int, int]]] = []
    cur_i = B
    for t in range(best_t, 0, -1):
        j = int(prev[t, cur_i])
        ranges_rev.append((j, cur_i - 1, span_dims[(j, cur_i - 1)]))
        cur_i = j
    ranges_rev.reverse()
    return ranges_rev, float(best_cost)


def plan_pme_tiers(  # noqa: PLR0913
    box_sizes: np.ndarray | jax.Array,
    *,
    cutoff: float = 9.0,
    ewald_error_tolerance: float = 5e-4,
    alpha: float | None = None,
    order: int = 4,
    grid_dims_per_replica: Sequence[tuple[int, int, int]] | None = None,
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    padded_atoms: int | None = None,
    atom_mask: np.ndarray | jax.Array | None = None,
    pme_tier_mode: PMETierMode = "auto",
    tier_launch_overhead_us: float = 15.0,
    fft_points_per_us: float = 18000.0,
    candidate_tier_dims: Sequence[int] | None = None,
    max_tiers: int = 4,
    cache_influence: bool = True,
    is_triclinic: bool = False,
) -> PMETierPlan:
    """Build a cost-modeled multi-resolution PME execution plan (MK-68).

    Args:
        box_sizes: Periodic box vectors ``(B, 3)``, ``(B, 3, 3)``, or ``(3,)`` in Å.
        cutoff: Direct-space PME cutoff in Å (used when inferring ``alpha`` & ``grid_dims``).
        ewald_error_tolerance: OpenMM Ewald tolerance (default ``5e-4``).
        alpha: Optional explicit Ewald ``alpha`` in Å^-1; defaults to OpenMM formula.
        order: B-spline interpolation order (``4``).
        grid_dims_per_replica: Optional explicit ``(Kx, Ky, Kz)`` per replica.
        n_real_per_replica: Active atom counts ``(B,)`` for 1D ragged spread/gather queues.
        padded_atoms: Padded atom dimension ``N_pad`` of the stacked batch.
        atom_mask: Optional ``(B, N)`` mask used to infer ``n_real_per_replica`` and ``padded_atoms``.
        pme_tier_mode: ``"auto"`` (DP launch-vs-FFT heuristic), ``"grouped"`` (group distinct
            grid tiers up to ``max_tiers``), or ``"uniform"`` (single ``G_max`` FFT batch).
        tier_launch_overhead_us: Extra cuFFT + spread/gather launch latency per tier (default ``15.0`` us).
        fft_points_per_us: Effective 3D FFT throughput in grid-points per microsecond (default ``18000.0``).
        candidate_tier_dims: Optional discrete tier buckets (e.g. ``(32, 48, 60)``); when ``None``,
            uses exact 5-smooth ``snap_fft_dim`` dimensions per replica.
        max_tiers: Maximum number of distinct FFT tiers to launch (default ``4``).
        cache_influence: Precompute static ``G_cached`` influence tensors per tier when ``box_sizes``
            is concrete.
    """
    mode_norm = str(pme_tier_mode).strip().lower()
    if mode_norm not in _VALID_TIER_MODES:
        raise ValueError(
            f"Invalid pme_tier_mode={pme_tier_mode!r}; expected one of {sorted(_VALID_TIER_MODES)}."
        )
    mode_typed = cast("PMETierMode", mode_norm)

    b_hint: int | None = None
    if grid_dims_per_replica is not None:
        b_hint = len(grid_dims_per_replica)
    elif n_real_per_replica is not None:
        b_hint = int(np.asarray(n_real_per_replica).reshape(-1).shape[0])
    elif atom_mask is not None and not isinstance(atom_mask, jax.core.Tracer):
        am_sh = np.shape(atom_mask)
        if len(am_sh) >= 2:
            b_hint = int(am_sh[0])

    has_concrete_box = not isinstance(box_sizes, jax.core.Tracer)
    if has_concrete_box:
        lengths_np, box_ref_np, _any_tri = _extract_box_batch_for_tiers_np(
            box_sizes, is_triclinic=is_triclinic, batch_size_hint=b_hint
        )
        B = int(lengths_np.shape[0])
    elif grid_dims_per_replica is not None:
        B = len(grid_dims_per_replica)
        lengths_np = np.ones((B, 3), dtype=np.float64)
        box_ref_np = lengths_np
    else:
        raise ValueError(
            "plan_pme_tiers requires concrete box_sizes or explicit grid_dims_per_replica."
        )

    if alpha is None:
        tol = max(float(ewald_error_tolerance), 1e-12)
        alpha_val = float(math.sqrt(-math.log(2.0 * tol)) / float(cutoff))
    else:
        alpha_val = float(alpha)

    if grid_dims_per_replica is not None:
        if len(grid_dims_per_replica) != B:
            raise ValueError(
                f"grid_dims_per_replica length ({len(grid_dims_per_replica)}) != batch size ({B})."
            )
        rep_dims = [
            _snap_to_tier_dims((int(g[0]), int(g[1]), int(g[2])), candidate_tier_dims)
            for g in grid_dims_per_replica
        ]
    else:
        rep_dims = []
        for b in range(B):
            _, gd = compute_openmm_pme_params(
                lengths_np[b],
                cutoff=float(cutoff),
                ewald_error_tolerance=float(ewald_error_tolerance),
            )
            rep_dims.append(_snap_to_tier_dims(gd, candidate_tier_dims))

    if padded_atoms is None and atom_mask is not None:
        padded_atoms = int(np.shape(atom_mask)[-1])
    if n_real_per_replica is None and atom_mask is None:
        eff_pad = int(padded_atoms) if padded_atoms is not None else 32
        n_real_np = np.full((B,), eff_pad, dtype=np.int32)
    else:
        n_real_np = _infer_n_real_np(
            n_real_per_replica,
            atom_mask,
            batch_size_hint=B,
            padded_size_hint=padded_atoms,
        )
    if padded_atoms is None:
        padded_atoms = int(np.max(n_real_np)) if B > 0 else 0
    padded_atoms_i = max(0, int(padded_atoms))
    n_real_np = np.minimum(n_real_np, padded_atoms_i) if padded_atoms_i > 0 else n_real_np

    uniform_queue = build_ragged_work_queue(
        n_real_np,
        tile_size=_PME_BLOCK_N,
        padded_size=padded_atoms_i,
    )

    if B == 0:
        return PMETierPlan(
            tiers=(),
            uniform_queue=uniform_queue,
            mode=mode_typed,
            alpha=alpha_val,
            order=int(order),
            uniform_grid_dims=(8, 8, 8),
            uniform_volume=0,
            tiered_volume=0,
            estimated_uniform_us=0.0,
            estimated_tiered_us=0.0,
            use_tiered=False,
            padded_atoms=padded_atoms_i,
        )

    u_kx = max(g[0] for g in rep_dims)
    u_ky = max(g[1] for g in rep_dims)
    u_kz = max(g[2] for g in rep_dims)
    uniform_dims = (u_kx, u_ky, u_kz)
    uniform_vol = int(B * u_kx * u_ky * u_kz)

    launch_us = max(0.0, float(tier_launch_overhead_us))
    pts_per_us = max(1e-6, float(fft_points_per_us))
    est_uniform_us = launch_us + float(uniform_vol) / pts_per_us

    sort_order = sorted(
        range(B),
        key=lambda b: (rep_dims[b][0] * rep_dims[b][1] * rep_dims[b][2], max(rep_dims[b]), rep_dims[b], b),
    )
    sorted_dims = [rep_dims[b] for b in sort_order]

    if mode_typed == "uniform":
        chosen_ranges = [(0, B - 1, uniform_dims)]
        est_tiered_us = est_uniform_us
    elif mode_typed == "grouped":
        # Group exact identical snapped dims first; if > max_tiers, merge via DP with zero launch penalty.
        exact_groups: list[tuple[int, int, tuple[int, int, int]]] = []
        start = 0
        for idx in range(1, B + 1):
            if idx == B or sorted_dims[idx] != sorted_dims[start]:
                exact_groups.append((start, idx - 1, sorted_dims[start]))
                start = idx
        if len(exact_groups) <= max(1, int(max_tiers)):
            chosen_ranges = exact_groups
            est_tiered_us = sum(
                launch_us + float((r - l + 1) * (gd[0] * gd[1] * gd[2])) / pts_per_us
                for l, r, gd in chosen_ranges
            )
        else:
            chosen_ranges, _ = _solve_optimal_tier_partition(
                sorted_dims,
                tier_launch_overhead_us=0.0,
                fft_points_per_us=pts_per_us,
                max_tiers=max_tiers,
            )
            est_tiered_us = sum(
                launch_us + float((r - l + 1) * (gd[0] * gd[1] * gd[2])) / pts_per_us
                for l, r, gd in chosen_ranges
            )
    else:
        # "auto": Exact DP balancing FFT grid volume savings against extra cuFFT tier launch overhead.
        dp_ranges, dp_cost = _solve_optimal_tier_partition(
            sorted_dims,
            tier_launch_overhead_us=launch_us,
            fft_points_per_us=pts_per_us,
            max_tiers=max_tiers,
        )
        if len(dp_ranges) > 1 and (dp_cost + 1e-6 < est_uniform_us):
            chosen_ranges = dp_ranges
            est_tiered_us = dp_cost
        else:
            chosen_ranges = [(0, B - 1, uniform_dims)]
            est_tiered_us = est_uniform_us

    tier_groups: list[PMETierGroup] = []
    tiered_vol = 0
    is_single_identity = (
        len(chosen_ranges) == 1 and [sort_order[k] for k in range(B)] == list(range(B))
    )

    for l_idx, r_idx, gd in chosen_ranges:
        if len(chosen_ranges) == 1 and is_single_identity:
            rep_tuple = tuple(range(B))
        else:
            rep_tuple = tuple(sorted(sort_order[k] for k in range(l_idx, r_idx + 1)))
        b_t = len(rep_tuple)
        tiered_vol += b_t * (gd[0] * gd[1] * gd[2])

        sub_n_real = n_real_np[list(rep_tuple)]
        max_real_t = int(np.max(sub_n_real)) if b_t > 0 else 0
        if len(chosen_ranges) == 1:
            n_slice_t = padded_atoms_i
        else:
            n_slice_t = min(
                padded_atoms_i,
                max(_PME_BLOCK_N, ((max_real_t + _PME_BLOCK_N - 1) // _PME_BLOCK_N) * _PME_BLOCK_N),
            ) if padded_atoms_i > 0 else 0

        rq_t = build_ragged_work_queue(
            sub_n_real,
            tile_size=_PME_BLOCK_N,
            padded_size=n_slice_t,
        )

        g_cached_arr: jax.Array | None = None
        box_ref_arr: jax.Array | None = None
        if cache_influence and has_concrete_box:
            sub_box = jnp.asarray(box_ref_np[list(rep_tuple)], dtype=jnp.float32)
            box_ref_arr = sub_box
            with jax.ensure_compile_time_eval():
                g_cached_arr = jax.vmap(
                    lambda b_elem, g_dims=gd: influence_function(
                        g_dims, b_elem, alpha_val, int(order)
                    )
                )(sub_box).astype(jnp.float32)

        tier_groups.append(
            PMETierGroup(
                replica_indices=jnp.asarray(rep_tuple, dtype=jnp.int32),
                ragged_queue=rq_t,
                G_cached=g_cached_arr,
                box_ref=box_ref_arr,
                grid_dims=gd,
                replica_indices_tuple=rep_tuple,
                n_slice=int(n_slice_t),
            )
        )

    return PMETierPlan(
        tiers=tuple(tier_groups),
        uniform_queue=uniform_queue,
        mode=mode_typed,
        alpha=alpha_val,
        order=int(order),
        uniform_grid_dims=uniform_dims,
        uniform_volume=uniform_vol,
        tiered_volume=int(tiered_vol),
        estimated_uniform_us=float(est_uniform_us),
        estimated_tiered_us=float(est_tiered_us),
        use_tiered=(len(tier_groups) > 1),
        padded_atoms=padded_atoms_i,
    )


# =============================================================================
# 3. 1D Ragged SPME Spread & Gather Pallas Kernels (MK-68a)
# =============================================================================


def _normalize_box6_batch(  # noqa: PLR0911
    box: jax.Array | None,
    B: int,
    dtype: Any,
    *,
    single_input: bool = False,
    is_triclinic: bool = False,
) -> tuple[jax.Array, bool]:
    """Normalize periodic box to ``(B_box, 6)`` ``[ax, by, cz, bx, cx, cy]`` (``B == 3`` safe)."""
    if box is None:
        return jnp.zeros((1, 6), dtype=dtype), False
    b_arr = jnp.asarray(box, dtype=dtype)
    if b_arr.ndim == 3 and b_arr.shape[-2:] == (3, 3):
        return (
            jnp.stack(
                [
                    b_arr[:, 0, 0],
                    b_arr[:, 1, 1],
                    b_arr[:, 2, 2],
                    b_arr[:, 1, 0],
                    b_arr[:, 2, 0],
                    b_arr[:, 2, 1],
                ],
                axis=-1,
            ),
            True,
        )
    if b_arr.ndim == 2:
        if b_arr.shape[-1] == 6:
            return b_arr, True
        if b_arr.shape[-1] == 9:
            return (
                jnp.stack(
                    [
                        b_arr[:, 0],
                        b_arr[:, 4],
                        b_arr[:, 8],
                        b_arr[:, 3],
                        b_arr[:, 6],
                        b_arr[:, 7],
                    ],
                    axis=-1,
                ),
                True,
            )
        if b_arr.shape == (3, 3):
            is_single_mat = (
                bool(single_input)
                or bool(is_triclinic)
                or (int(B) != 3)
                or _is_lower_tri_3x3_arr(b_arr)
            )
            if is_single_mat:
                row = jnp.stack(
                    [
                        b_arr[0, 0],
                        b_arr[1, 1],
                        b_arr[2, 2],
                        b_arr[1, 0],
                        b_arr[2, 0],
                        b_arr[2, 1],
                    ]
                )
                return row[None, :], True
        return (
            jnp.concatenate([b_arr[:, :3], jnp.zeros((b_arr.shape[0], 3), dtype=dtype)], axis=-1),
            True,
        )
    if b_arr.ndim == 1:
        if b_arr.size == 6:
            return b_arr[None, :], True
        if b_arr.size == 3:
            return jnp.concatenate([b_arr, jnp.zeros((3,), dtype=dtype)])[None, :], True
        if (not single_input) and B > 0 and b_arr.size == B * 9:
            b9 = b_arr.reshape(B, 9)
            return (
                jnp.stack([b9[:, 0], b9[:, 4], b9[:, 8], b9[:, 3], b9[:, 6], b9[:, 7]], axis=-1),
                True,
            )
        if b_arr.size == 9 and (single_input or is_triclinic or int(B) != 3):
            return (
                jnp.stack(
                    [b_arr[0], b_arr[4], b_arr[8], b_arr[3], b_arr[6], b_arr[7]]
                )[None, :],
                True,
            )
        if (not single_input) and B > 0 and b_arr.size == B * 3:
            b3 = b_arr.reshape(B, 3)
            return jnp.concatenate([b3, jnp.zeros((B, 3), dtype=dtype)], axis=-1), True
        return jnp.concatenate([b_arr[:3], jnp.zeros((3,), dtype=dtype)])[None, :], True
    raise ValueError(f"Unsupported periodic box shape: {b_arr.shape}")


def _box6_to_3x3_batch(box6: jax.Array, B: int) -> jax.Array:
    """Expand ``(B_box, 6)`` ``[ax, by, cz, bx, cx, cy]`` to ``(B, 3, 3)`` lower-triangular."""
    b6 = jnp.broadcast_to(box6, (B, 6))
    z = jnp.zeros((B,), dtype=b6.dtype)
    r0 = jnp.stack([b6[:, 0], z, z], axis=-1)
    r1 = jnp.stack([b6[:, 3], b6[:, 1], z], axis=-1)
    r2 = jnp.stack([b6[:, 4], b6[:, 5], b6[:, 2]], axis=-1)
    return jnp.stack([r0, r1, r2], axis=-2)


def _spread_reference_batched(
    pos_b: jax.Array,
    q_b: jax.Array,
    m_b: jax.Array,
    box_b_canon: jax.Array,
    grid_dims: tuple[int, int, int],
    order: int,
) -> jax.Array:
    B = int(pos_b.shape[0])
    b_q = int(q_b.shape[0]) > 1
    b_m = int(m_b.shape[0]) > 1
    b_bx = int(box_b_canon.shape[0]) > 1
    return jax.vmap(
        lambda b: _spread_charges_reference(
            pos_b[b],
            q_b[b] if b_q else q_b[0],
            m_b[b] if b_m else m_b[0],
            box_b_canon[b] if b_bx else box_b_canon[0],
            grid_dims=grid_dims,
            order=order,
        )
    )(jnp.arange(B, dtype=jnp.int32))


def _gather_reference_batched(
    pos_b: jax.Array,
    q_b: jax.Array,
    m_b: jax.Array,
    box_b_canon: jax.Array,
    th_b: jax.Array,
    grid_dims: tuple[int, int, int],
    order: int,
    force_scale: float,
    pot_scale: float,
) -> tuple[jax.Array, jax.Array]:
    B = int(pos_b.shape[0])
    b_q = int(q_b.shape[0]) > 1
    b_m = int(m_b.shape[0]) > 1
    b_bx = int(box_b_canon.shape[0]) > 1
    b_th = int(th_b.shape[0]) > 1
    return jax.vmap(
        lambda b: _gather_reference(
            pos_b[b],
            q_b[b] if b_q else q_b[0],
            m_b[b] if b_m else m_b[0],
            box_b_canon[b] if b_bx else box_b_canon[0],
            th_b[b] if b_th else th_b[0],
            grid_dims=grid_dims,
            order=order,
            force_scale=force_scale,
            pot_scale=pot_scale,
        )
    )(jnp.arange(B, dtype=jnp.int32))


def spread_charges_pallas_ragged(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    *,
    queue: RaggedWorkQueue | None = None,
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    order: int = 4,
    interpret: bool | None = None,
    is_triclinic: bool = False,
) -> jax.Array:
    """1D ragged CTA queue SPME B-spline charge spread (ortho & reduced triclinic, zero ghost CTAs)."""
    if order != 4:
        raise NotImplementedError("Pallas ragged SPME stencil is order-4 only.")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    single_input = positions.ndim == 2
    pos_b = positions[None, ...] if single_input else positions
    B, n_atoms, _ = pos_b.shape
    orig_dtype = positions.dtype
    Kx, Ky, Kz = grid_tuple
    mesh_stride = Kx * Ky * Kz

    if n_atoms == 0 or B == 0:
        out_empty = jnp.zeros((B, Kx, Ky, Kz), dtype=orig_dtype)
        return out_empty[0] if single_input else out_empty

    b_q = charges.ndim == 2 and not single_input
    b_m = atom_mask.ndim == 2 and not single_input
    q_b = charges if b_q else (charges[None, ...] if charges.ndim == 1 else charges[:1])
    m_b = atom_mask if b_m else (atom_mask[None, ...] if atom_mask.ndim == 1 else atom_mask[:1])

    eff_tri = _infer_pme_is_triclinic(
        box_size, B, single_input=single_input, is_triclinic=is_triclinic
    )
    box6_b, _ = _normalize_box6_batch(
        box_size, B, orig_dtype, single_input=single_input, is_triclinic=eff_tri
    )

    if in_export_context():
        box_canon = _box6_to_3x3_batch(box6_b, B) if eff_tri else box6_b[:, :3]
        out_ref = _spread_reference_batched(pos_b, q_b, m_b, box_canon, grid_tuple, order)
        return out_ref[0] if single_input else out_ref

    if queue is None:
        queue = build_ragged_work_queue(
            n_real_per_replica,
            tile_size=_PME_BLOCK_N,
            padded_size=n_atoms,
            atom_mask=m_b if b_m else jnp.broadcast_to(m_b, (B, n_atoms)),
        )
    if queue.tile_size != _PME_BLOCK_N:
        raise ValueError(f"Expected queue.tile_size == {_PME_BLOCK_N}, got {queue.tile_size}.")

    if queue.total_blocks == 0:
        out_zero = jnp.zeros((B, Kx, Ky, Kz), dtype=orig_dtype)
        return out_zero[0] if single_input else out_zero

    use_interpret = _resolve_interpret(interpret)
    dtype = jnp.float32
    pos_f = pos_b.astype(dtype)
    q_f = q_b.astype(dtype)
    m_i = jnp.asarray(m_b, dtype=jnp.int32)
    bx_f = (box6_b if eff_tri else box6_b[:, :3]).astype(dtype)
    bx_stride = 6 if eff_tri else 3

    if use_interpret:
        n_blocks_pad = max(1, (n_atoms + _PME_BLOCK_N - 1) // _PME_BLOCK_N)
        n_stride = n_blocks_pad * _PME_BLOCK_N
        pad = n_stride - n_atoms
        if pad > 0:
            pos_f = jnp.pad(pos_f, ((0, 0), (0, pad), (0, 0)))
            q_f = jnp.pad(q_f, ((0, 0), (0, pad)))
            m_i = jnp.pad(m_i, ((0, 0), (0, pad)))
    else:
        n_stride = n_atoms

    px_flat = jnp.asarray(pos_f[:, :, 0]).reshape(-1)
    py_flat = jnp.asarray(pos_f[:, :, 1]).reshape(-1)
    pz_flat = jnp.asarray(pos_f[:, :, 2]).reshape(-1)
    q_flat = q_f.reshape(-1)
    m_flat = m_i.reshape(-1)
    box_flat = bx_f.reshape(-1)
    mesh0_flat = jnp.zeros((B * mesh_stride,), dtype=dtype)

    B_q = int(q_f.shape[0])
    B_m = int(m_i.shape[0])
    B_box = int(bx_f.shape[0])

    def _spread_ragged_kernel(
        px_ref,
        py_ref,
        pz_ref,
        charges_ref,
        mask_ref,
        box_ref,
        cta_rep_ref,
        cta_blk_ref,
        n_real_ref,
        mesh_in_ref,
        mesh_out_ref,
    ):
        del mesh_in_ref
        cta_id = pl.program_id(0)
        b_id = _pl_load(cta_rep_ref, (jnp.full((1,), cta_id, dtype=jnp.int32),))[0]
        local_pid = _pl_load(cta_blk_ref, (jnp.full((1,), cta_id, dtype=jnp.int32),))[0]
        n_real_b = _pl_load(n_real_ref, (jnp.full((1,), b_id, dtype=jnp.int32),))[0]

        offs = local_pid * _PME_BLOCK_N + jnp.arange(_PME_BLOCK_N, dtype=jnp.int32)
        pos_base = b_id * n_stride
        q_base = (b_id if B_q > 1 else jnp.int32(0)) * n_stride
        m_base = (b_id if B_m > 1 else jnp.int32(0)) * n_stride
        box_base = (b_id if B_box > 1 else jnp.int32(0)) * bx_stride
        mesh_base = b_id * mesh_stride

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        in_bounds = (offs < n_real_b) & (offs < n_atoms)
        is_real = (_pl_load(mask_ref, (m_base + offs,), mask=in_bounds, other=0) > 0) & in_bounds
        qi = _pl_load(charges_ref, (q_base + offs,), mask=is_real, other=zero)
        active_q = is_real & (jnp.abs(qi) > zero)

        def _do_spread(_):
            px = _pl_load(px_ref, (pos_base + offs,), mask=active_q, other=zero)
            py = _pl_load(py_ref, (pos_base + offs,), mask=active_q, other=zero)
            pz = _pl_load(pz_ref, (pos_base + offs,), mask=active_q, other=zero)

            ax = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
            by = _pl_load(box_ref, (box_base + jnp.ones((1,), dtype=jnp.int32),))[0]
            cz = _pl_load(box_ref, (box_base + jnp.full((1,), 2, dtype=jnp.int32),))[0]
            kx = jnp.asarray(Kx, dtype=dtype)
            ky = jnp.asarray(Ky, dtype=dtype)
            kz = jnp.asarray(Kz, dtype=dtype)

            if eff_tri:
                bx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 3, dtype=jnp.int32),))[0]
                cx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 4, dtype=jnp.int32),))[0]
                cy_t = _pl_load(box_ref, (box_base + jnp.full((1,), 5, dtype=jnp.int32),))[0]
                b00 = one / ax
                b11 = one / by
                b22 = one / cz
                b10 = zero - bx_t * (b00 * b11)
                b21 = zero - cy_t * (b11 * b22)
                b20 = (bx_t * cy_t - by * cx_t) * (b00 * b11 * b22)
                sx_r = px * b00 + py * b10 + pz * b20
                sy_r = py * b11 + pz * b21
                sz_r = pz * b22
                frac_x = (sx_r - jnp.floor(sx_r)) * kx
                frac_y = (sy_r - jnp.floor(sy_r)) * ky
                frac_z = (sz_r - jnp.floor(sz_r)) * kz
            else:
                frac_x = px * (kx / ax)
                frac_y = py * (ky / by)
                frac_z = pz * (kz / cz)
            gx = jnp.floor(frac_x)
            gy = jnp.floor(frac_y)
            gz = jnp.floor(frac_z)
            ux = frac_x - gx
            uy = frac_y - gy
            uz = frac_z - gz

            wx = _bspline4_vec(ux)
            wy = _bspline4_vec(uy)
            wz = _bspline4_vec(uz)

            gxi = gx.astype(jnp.int32)
            gyi = gy.astype(jnp.int32)
            gzi = gz.astype(jnp.int32)

            s_yz = jnp.arange(16, dtype=jnp.int32)
            oy = (s_yz // 4)[None, :]
            oz = (s_yz % 4)[None, :]
            iy_16 = jnp.mod(gyi[:, None] + oy - 1, Ky)
            iz_16 = jnp.mod(gzi[:, None] + oz - 1, Kz)
            iyz_16 = mesh_base + iy_16 * Kz + iz_16

            wy_16 = _select_stencil(wy, oy)
            wz_16 = _select_stencil(wz, oz)
            q_wyz_16 = jnp.where(active_q[:, None], qi[:, None] * (wy_16 * wz_16), zero)
            mask_16 = jnp.broadcast_to(active_q[:, None], (_PME_BLOCK_N, 16)).reshape(_PME_BLOCK_N * 16)
            ky_kz = Ky * Kz

            for ox_i in range(4):
                ix_plane = jnp.mod(gxi + (ox_i - 1), Kx)[:, None] * ky_kz
                flat_idx_16 = (ix_plane + iyz_16).reshape(_PME_BLOCK_N * 16)
                w_16 = (wx[ox_i][:, None] * q_wyz_16).astype(dtype).reshape(_PME_BLOCK_N * 16)
                if use_interpret:
                    def _serial_atomic(k, _, idx_vec=flat_idx_16, w_vec=w_16):
                        plt.atomic_add(mesh_out_ref, (idx_vec[k],), w_vec[k])

                    jax.lax.fori_loop(0, _PME_BLOCK_N * 16, _serial_atomic, None)
                else:
                    plt.atomic_add(mesh_out_ref, (flat_idx_16,), w_16, mask=mask_16)

        jax.lax.cond(
            jnp.max(active_q.astype(jnp.int32)) > 0,
            _do_spread,
            lambda _: None,
            None,
        )

    any_spec = _any_block_spec()
    mesh_flat = pl.pallas_call(
        _spread_ragged_kernel,
        out_shape=jax.ShapeDtypeStruct((B * mesh_stride,), dtype),
        in_specs=[any_spec] * 10,
        out_specs=any_spec,
        grid=(queue.total_blocks,),
        interpret=use_interpret,
        compiler_params=resolve_pallas_compiler_params("pme", grid_blocks=queue.total_blocks)
        if not use_interpret
        else None,
        input_output_aliases={9: 0},
        name="pme_charge_spread_ragged_1d",
    )(
        px_flat,
        py_flat,
        pz_flat,
        q_flat,
        m_flat,
        box_flat,
        queue.cta_to_replica,
        queue.cta_to_local_block,
        queue.n_real_per_replica,
        mesh0_flat,
    )
    out = mesh_flat.reshape((B, Kx, Ky, Kz)).astype(orig_dtype)
    return out[0] if single_input else out


def gather_pallas_ragged(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    *,
    queue: RaggedWorkQueue | None = None,
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    order: int = 4,
    compute_dq: bool = True,
    force_scale: float = 1.0,
    pot_scale: float = 1.0,
    interpret: bool | None = None,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """1D ragged CTA queue SPME force & potential gather (ortho & reduced triclinic, zero ghost CTAs)."""
    if order != 4:
        raise NotImplementedError("Pallas ragged SPME stencil is order-4 only.")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    single_input = positions.ndim == 2
    pos_b = positions[None, ...] if single_input else positions
    B, n_atoms, _ = pos_b.shape
    orig_dtype = positions.dtype
    Kx, Ky, Kz = grid_tuple
    mesh_stride = Kx * Ky * Kz

    if n_atoms == 0 or B == 0:
        zf = jnp.zeros((B, n_atoms, 3), dtype=orig_dtype)
        zq = jnp.zeros((B, n_atoms), dtype=orig_dtype)
        return (zf[0], zq[0]) if single_input else (zf, zq)

    b_q = charges.ndim == 2 and not single_input
    b_m = atom_mask.ndim == 2 and not single_input
    b_mesh = theta_norm.ndim == 4 and not single_input

    q_b = charges if b_q else (charges[None, ...] if charges.ndim == 1 else charges[:1])
    m_b = atom_mask if b_m else (atom_mask[None, ...] if atom_mask.ndim == 1 else atom_mask[:1])
    th_b = theta_norm if b_mesh else theta_norm[None, ...]

    eff_tri = _infer_pme_is_triclinic(
        box_size, B, single_input=single_input, is_triclinic=is_triclinic
    )
    box6_b, _ = _normalize_box6_batch(
        box_size, B, orig_dtype, single_input=single_input, is_triclinic=eff_tri
    )

    if in_export_context():
        box_canon = _box6_to_3x3_batch(box6_b, B) if eff_tri else box6_b[:, :3]
        forces_ref, pots_ref = _gather_reference_batched(
            pos_b,
            q_b,
            m_b,
            box_canon,
            th_b.astype(orig_dtype),
            grid_tuple,
            order,
            float(force_scale),
            float(pot_scale),
        )
        if not compute_dq:
            pots_ref = jnp.zeros_like(pots_ref)
        return (forces_ref[0], pots_ref[0]) if single_input else (forces_ref, pots_ref)

    if queue is None:
        queue = build_ragged_work_queue(
            n_real_per_replica,
            tile_size=_PME_BLOCK_N,
            padded_size=n_atoms,
            atom_mask=m_b if b_m else jnp.broadcast_to(m_b, (B, n_atoms)),
        )
    if queue.tile_size != _PME_BLOCK_N:
        raise ValueError(f"Expected queue.tile_size == {_PME_BLOCK_N}, got {queue.tile_size}.")

    if queue.total_blocks == 0:
        zf = jnp.zeros((B, n_atoms, 3), dtype=orig_dtype)
        zq = jnp.zeros((B, n_atoms), dtype=orig_dtype)
        return (zf[0], zq[0]) if single_input else (zf, zq)

    use_interpret = _resolve_interpret(interpret)
    dtype = jnp.float32
    pos_f = pos_b.astype(dtype)
    q_f = q_b.astype(dtype)
    m_i = jnp.asarray(m_b, dtype=jnp.int32)
    bx_f = (box6_b if eff_tri else box6_b[:, :3]).astype(dtype)
    bx_stride = 6 if eff_tri else 3
    th_f = th_b.astype(dtype)

    if use_interpret:
        n_blocks_pad = max(1, (n_atoms + _PME_BLOCK_N - 1) // _PME_BLOCK_N)
        n_stride = n_blocks_pad * _PME_BLOCK_N
        pad = n_stride - n_atoms
        if pad > 0:
            pos_f = jnp.pad(pos_f, ((0, 0), (0, pad), (0, 0)))
            q_f = jnp.pad(q_f, ((0, 0), (0, pad)))
            m_i = jnp.pad(m_i, ((0, 0), (0, pad)))
    else:
        n_stride = n_atoms

    px_flat = jnp.asarray(pos_f[:, :, 0]).reshape(-1)
    py_flat = jnp.asarray(pos_f[:, :, 1]).reshape(-1)
    pz_flat = jnp.asarray(pos_f[:, :, 2]).reshape(-1)
    q_flat = q_f.reshape(-1)
    m_flat = m_i.reshape(-1)
    box_flat = bx_f.reshape(-1)
    theta_flat = th_f.reshape(-1)

    B_q = int(q_f.shape[0])
    B_m = int(m_i.shape[0])
    B_box = int(bx_f.shape[0])
    B_mesh = int(th_f.shape[0])

    z_fx = jnp.zeros((B * n_stride,), dtype=dtype)
    z_fy = jnp.zeros((B * n_stride,), dtype=dtype)
    z_fz = jnp.zeros((B * n_stride,), dtype=dtype)

    def _gather_ragged_kernel(
        px_ref,
        py_ref,
        pz_ref,
        charges_ref,
        mask_ref,
        box_ref,
        theta_flat_ref,
        cta_rep_ref,
        cta_blk_ref,
        n_real_ref,
        fx_in_ref,
        fy_in_ref,
        fz_in_ref,
        *rest_refs,
    ):
        del fx_in_ref, fy_in_ref, fz_in_ref
        if compute_dq:
            _dq_in_ref, fx_out_ref, fy_out_ref, fz_out_ref, dq_out_ref = rest_refs
            del _dq_in_ref
        else:
            fx_out_ref, fy_out_ref, fz_out_ref = rest_refs
            dq_out_ref = None

        cta_id = pl.program_id(0)
        b_id = _pl_load(cta_rep_ref, (jnp.full((1,), cta_id, dtype=jnp.int32),))[0]
        local_pid = _pl_load(cta_blk_ref, (jnp.full((1,), cta_id, dtype=jnp.int32),))[0]
        n_real_b = _pl_load(n_real_ref, (jnp.full((1,), b_id, dtype=jnp.int32),))[0]

        offs = local_pid * _PME_BLOCK_N + jnp.arange(_PME_BLOCK_N, dtype=jnp.int32)
        pos_base = b_id * n_stride
        q_base = (b_id if B_q > 1 else jnp.int32(0)) * n_stride
        m_base = (b_id if B_m > 1 else jnp.int32(0)) * n_stride
        box_base = (b_id if B_box > 1 else jnp.int32(0)) * bx_stride
        mesh_base = (b_id if B_mesh > 1 else jnp.int32(0)) * mesh_stride
        out_base = b_id * n_stride

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        in_bounds_act = (offs < n_real_b) & (offs < n_atoms)
        in_bounds_out = offs < n_stride
        is_real = (_pl_load(mask_ref, (m_base + offs,), mask=in_bounds_act, other=0) > 0) & in_bounds_act
        qi = _pl_load(charges_ref, (q_base + offs,), mask=is_real, other=zero)
        active_lane = is_real if compute_dq else (is_real & (jnp.abs(qi) > zero))
        zero_vec = (offs - offs).astype(dtype)

        def _do_gather(_):
            px = _pl_load(px_ref, (pos_base + offs,), mask=active_lane, other=zero)
            py = _pl_load(py_ref, (pos_base + offs,), mask=active_lane, other=zero)
            pz = _pl_load(pz_ref, (pos_base + offs,), mask=active_lane, other=zero)

            ax = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
            by = _pl_load(box_ref, (box_base + jnp.ones((1,), dtype=jnp.int32),))[0]
            cz = _pl_load(box_ref, (box_base + jnp.full((1,), 2, dtype=jnp.int32),))[0]
            kx = jnp.asarray(Kx, dtype=dtype)
            ky = jnp.asarray(Ky, dtype=dtype)
            kz = jnp.asarray(Kz, dtype=dtype)

            if eff_tri:
                bx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 3, dtype=jnp.int32),))[0]
                cx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 4, dtype=jnp.int32),))[0]
                cy_t = _pl_load(box_ref, (box_base + jnp.full((1,), 5, dtype=jnp.int32),))[0]
                b00 = one / ax
                b11 = one / by
                b22 = one / cz
                b10 = zero - bx_t * (b00 * b11)
                b21 = zero - cy_t * (b11 * b22)
                b20 = (bx_t * cy_t - by * cx_t) * (b00 * b11 * b22)
                sx_r = px * b00 + py * b10 + pz * b20
                sy_r = py * b11 + pz * b21
                sz_r = pz * b22
                frac_x = (sx_r - jnp.floor(sx_r)) * kx
                frac_y = (sy_r - jnp.floor(sy_r)) * ky
                frac_z = (sz_r - jnp.floor(sz_r)) * kz
            else:
                b00 = b11 = b22 = b10 = b21 = b20 = zero
                sx = kx / ax
                sy = ky / by
                sz = kz / cz
                frac_x = px * sx
                frac_y = py * sy
                frac_z = pz * sz
            gx = jnp.floor(frac_x)
            gy = jnp.floor(frac_y)
            gz = jnp.floor(frac_z)
            ux = frac_x - gx
            uy = frac_y - gy
            uz = frac_z - gz

            wx = _bspline4_vec(ux)
            wy = _bspline4_vec(uy)
            wz = _bspline4_vec(uz)
            dwx = _bspline4_deriv_vec(ux)
            dwy = _bspline4_deriv_vec(uy)
            dwz = _bspline4_deriv_vec(uz)

            gxi = gx.astype(jnp.int32)
            gyi = gy.astype(jnp.int32)
            gzi = gz.astype(jnp.int32)

            s_yz = jnp.arange(16, dtype=jnp.int32)
            oy = (s_yz // 4)[None, :]
            oz = (s_yz % 4)[None, :]
            iy_16 = jnp.mod(gyi[:, None] + oy - 1, Ky)
            iz_16 = jnp.mod(gzi[:, None] + oz - 1, Kz)
            iyz_16 = mesh_base + iy_16 * Kz + iz_16

            wy_16 = _select_stencil(wy, oy)
            wz_16 = _select_stencil(wz, oz)
            dwy_16 = _select_stencil(dwy, oy)
            dwz_16 = _select_stencil(dwz, oz)

            wyz_16 = wy_16 * wz_16
            dwyz_16 = dwy_16 * wz_16
            wdyz_16 = wy_16 * dwz_16

            mask_16 = jnp.broadcast_to(active_lane[:, None], (_PME_BLOCK_N, 16)).reshape(_PME_BLOCK_N * 16)
            ky_kz = Ky * Kz

            acc_dx = zero_vec
            acc_dy = zero_vec
            acc_dz = zero_vec
            acc_dq = zero_vec

            for ox_i in range(4):
                ix_plane = jnp.mod(gxi + (ox_i - 1), Kx)[:, None] * ky_kz
                flat_idx_16 = (ix_plane + iyz_16).reshape(_PME_BLOCK_N * 16)
                t_16 = _pl_load(
                    theta_flat_ref,
                    (flat_idx_16,),
                    mask=mask_16,
                    other=zero,
                ).reshape(_PME_BLOCK_N, 16)
                sum_wyz = jnp.sum(wyz_16 * t_16, axis=-1)
                sum_dwyz = jnp.sum(dwyz_16 * t_16, axis=-1)
                sum_wdyz = jnp.sum(wdyz_16 * t_16, axis=-1)
                wx_i = wx[ox_i]
                dwx_i = dwx[ox_i]
                acc_dx = acc_dx + dwx_i * sum_wyz
                acc_dy = acc_dy + wx_i * sum_dwyz
                acc_dz = acc_dz + wx_i * sum_wdyz
                if compute_dq:
                    acc_dq = acc_dq + wx_i * sum_wyz

            f_scale = jnp.asarray(force_scale, dtype=dtype)
            q_scale = jnp.asarray(pot_scale, dtype=dtype)
            dq_v = jnp.where(is_real, acc_dq * q_scale, zero).astype(dtype) if compute_dq else zero_vec
            q_f_lane = qi * f_scale
            if eff_tri:
                gx_f = q_f_lane * (kx * acc_dx)
                gy_f = q_f_lane * (ky * acc_dy)
                gz_f = q_f_lane * (kz * acc_dz)
                dpx_v = jnp.where(active_lane, gx_f * b00, zero).astype(dtype)
                dpy_v = jnp.where(active_lane, gx_f * b10 + gy_f * b11, zero).astype(dtype)
                dpz_v = jnp.where(active_lane, gx_f * b20 + gy_f * b21 + gz_f * b22, zero).astype(dtype)
            else:
                dpx_v = jnp.where(active_lane, q_f_lane * (kx / ax) * acc_dx, zero).astype(dtype)
                dpy_v = jnp.where(active_lane, q_f_lane * (ky / by) * acc_dy, zero).astype(dtype)
                dpz_v = jnp.where(active_lane, q_f_lane * (kz / cz) * acc_dz, zero).astype(dtype)
            return (dpx_v, dpy_v, dpz_v, dq_v)

        dpx, dpy, dpz, dq = jax.lax.cond(
            jnp.max(active_lane.astype(jnp.int32)) > 0,
            _do_gather,
            lambda z0: z0,
            (zero_vec, zero_vec, zero_vec, zero_vec),
        )

        if use_interpret:
            _pl_store(fx_out_ref, (out_base + offs,), dpx)
            _pl_store(fy_out_ref, (out_base + offs,), dpy)
            _pl_store(fz_out_ref, (out_base + offs,), dpz)
            if compute_dq and dq_out_ref is not None:
                _pl_store(dq_out_ref, (out_base + offs,), dq)
        else:
            _pl_store(fx_out_ref, (out_base + offs,), dpx, mask=in_bounds_out)
            _pl_store(fy_out_ref, (out_base + offs,), dpy, mask=in_bounds_out)
            _pl_store(fz_out_ref, (out_base + offs,), dpz, mask=in_bounds_out)
            if compute_dq and dq_out_ref is not None:
                _pl_store(dq_out_ref, (out_base + offs,), dq, mask=in_bounds_out)

    any_spec = _any_block_spec()
    if compute_dq:
        z_dq = jnp.zeros((B * n_stride,), dtype=dtype)
        fx_pad, fy_pad, fz_pad, dq_pad = pl.pallas_call(
            _gather_ragged_kernel,
            out_shape=[jax.ShapeDtypeStruct((B * n_stride,), dtype) for _ in range(4)],
            in_specs=[any_spec] * 14,
            out_specs=[any_spec] * 4,
            grid=(queue.total_blocks,),
            interpret=use_interpret,
            compiler_params=resolve_pallas_compiler_params("pme", grid_blocks=queue.total_blocks)
            if not use_interpret
            else None,
            input_output_aliases={10: 0, 11: 1, 12: 2, 13: 3},
            name="pme_bwd_gather_ragged_1d",
        )(
            px_flat,
            py_flat,
            pz_flat,
            q_flat,
            m_flat,
            box_flat,
            theta_flat,
            queue.cta_to_replica,
            queue.cta_to_local_block,
            queue.n_real_per_replica,
            z_fx,
            z_fy,
            z_fz,
            z_dq,
        )
        dq_out = dq_pad.reshape((B, n_stride))[:, :n_atoms].astype(orig_dtype)
    else:
        fx_pad, fy_pad, fz_pad = pl.pallas_call(
            _gather_ragged_kernel,
            out_shape=[jax.ShapeDtypeStruct((B * n_stride,), dtype) for _ in range(3)],
            in_specs=[any_spec] * 13,
            out_specs=[any_spec] * 3,
            grid=(queue.total_blocks,),
            interpret=use_interpret,
            compiler_params=resolve_pallas_compiler_params("pme", grid_blocks=queue.total_blocks)
            if not use_interpret
            else None,
            input_output_aliases={10: 0, 11: 1, 12: 2},
            name="pme_bwd_gather_ragged_1d_nodq",
        )(
            px_flat,
            py_flat,
            pz_flat,
            q_flat,
            m_flat,
            box_flat,
            theta_flat,
            queue.cta_to_replica,
            queue.cta_to_local_block,
            queue.n_real_per_replica,
            z_fx,
            z_fy,
            z_fz,
        )
        dq_out = jnp.zeros((B, n_atoms), dtype=orig_dtype)

    f_out = jnp.stack(
        [
            fx_pad.reshape((B, n_stride))[:, :n_atoms],
            fy_pad.reshape((B, n_stride))[:, :n_atoms],
            fz_pad.reshape((B, n_stride))[:, :n_atoms],
        ],
        axis=-1,
    ).astype(orig_dtype)
    if single_input:
        return f_out[0], dq_out[0]
    return f_out, dq_out


# =============================================================================
# 4. Multi-Resolution Tiered SPME Execution & Custom VJP (MK-68b)
# =============================================================================


def _resolve_tier_green(
    tier: PMETierGroup,
    box_t: jax.Array,
    alpha: float,
    order: int,
    work_dtype: Any,
) -> jax.Array:
    if box_t.ndim == 2 and box_t.shape[-1] == 3:
        if (
            tier.G_cached is not None
            and tier.box_ref is not None
            and tier.box_ref.ndim == 2
            and tier.box_ref.shape == box_t.shape
        ):
            return scale_influence_function_isotropic(
                tier.G_cached.astype(work_dtype),
                tier.box_ref.astype(work_dtype),
                box_t.astype(work_dtype),
                alpha=float(alpha),
                grid_dims=tier.grid_dims,
                order=int(order),
            )
        return jax.vmap(
            lambda b_vec: influence_function(tier.grid_dims, b_vec, alpha, order)
        )(box_t).astype(work_dtype)

    if box_t.ndim == 3 and box_t.shape[-2:] == (3, 3):
        if (
            work_dtype == jnp.float32
            and tier.G_cached is not None
            and tier.box_ref is not None
            and tier.box_ref.shape == box_t.shape
        ):
            ref_b = tier.box_ref.astype(box_t.dtype)
            g_cached = tier.G_cached.astype(work_dtype)
            return jax.lax.cond(
                jnp.all(jnp.abs(box_t - ref_b) <= jnp.asarray(1e-6, dtype=box_t.dtype)),
                lambda _: g_cached,
                lambda _: jax.vmap(
                    lambda b_mat: influence_function(tier.grid_dims, b_mat, alpha, order)
                )(box_t).astype(work_dtype),
                None,
            )
        return jax.vmap(
            lambda b_mat: influence_function(tier.grid_dims, b_mat, alpha, order)
        )(box_t).astype(work_dtype)

    return influence_function(tier.grid_dims, box_t, alpha, order).astype(work_dtype)


def _eval_single_tier_forward(
    pos_t: jax.Array,
    q_t: jax.Array,
    mask_t: jax.Array,
    box_t: jax.Array,
    tier: PMETierGroup,
    alpha: float,
    order: int,
    use_pallas_pme: bool,
    interpret: bool | None,
    *,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    orig_dtype = pos_t.dtype
    Kx, Ky, Kz = tier.grid_dims
    use_pl = bool(use_pallas_pme and (not in_export_context()))
    work_dtype = jnp.float32 if use_pl else orig_dtype

    pos_w = pos_t.astype(work_dtype)
    q_w = q_t.astype(work_dtype)
    box_w = jnp.asarray(box_t, dtype=work_dtype)

    if use_pl:
        Q = spread_charges_pallas_ragged(
            pos_w,
            q_w,
            mask_t,
            box_w,
            tier.grid_dims,
            queue=tier.ragged_queue,
            order=order,
            interpret=interpret,
            is_triclinic=is_triclinic,
        ).astype(jnp.float32)
    else:
        Q = _spread_reference_batched(
            pos_w,
            q_w,
            mask_t,
            box_w,
            tier.grid_dims,
            order,
        )

    Q_hat = jnp.fft.rfftn(Q, s=tier.grid_dims, axes=(-3, -2, -1))
    G = _resolve_tier_green(tier, box_w, alpha, order, Q.dtype)
    theta = jnp.fft.irfftn(Q_hat * G, s=tier.grid_dims, axes=(-3, -2, -1)).astype(Q.dtype)
    n_grid = jnp.asarray(float(Kx * Ky * Kz), dtype=Q.dtype)
    theta_norm = (theta * n_grid).astype(Q.dtype)
    e_recip = (
        jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=Q.dtype)
        * jnp.sum(Q * theta_norm, axis=(-3, -2, -1))
    ).astype(orig_dtype)
    e_self = spme_self_energy(q_t, mask_t, alpha).astype(orig_dtype)
    return e_recip + e_self, theta_norm, Q


def _eval_single_tier_gather(
    pos_t: jax.Array,
    q_t: jax.Array,
    mask_t: jax.Array,
    box_t: jax.Array,
    theta_norm: jax.Array,
    tier: PMETierGroup,
    order: int,
    use_pallas_pme: bool,
    compute_dq: bool,
    force_scale: float,
    pot_scale: float,
    interpret: bool | None,
    *,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    orig_dtype = pos_t.dtype
    use_pl = bool(use_pallas_pme and (not in_export_context()))
    if use_pl:
        f_t, dq_t = gather_pallas_ragged(
            pos_t.astype(jnp.float32),
            q_t.astype(jnp.float32),
            mask_t,
            jnp.asarray(box_t, dtype=jnp.float32),
            theta_norm.astype(jnp.float32),
            tier.grid_dims,
            queue=tier.ragged_queue,
            order=order,
            compute_dq=compute_dq,
            force_scale=force_scale,
            pot_scale=pot_scale,
            interpret=interpret,
            is_triclinic=is_triclinic,
        )
        return f_t.astype(orig_dtype), dq_t.astype(orig_dtype)
    f_ref, dq_ref = _gather_reference_batched(
        pos_t,
        q_t,
        mask_t,
        jnp.asarray(box_t, dtype=orig_dtype),
        theta_norm.astype(orig_dtype),
        tier.grid_dims,
        order,
        force_scale,
        pot_scale,
    )
    if not compute_dq:
        dq_ref = jnp.zeros_like(dq_ref)
    return f_ref.astype(orig_dtype), dq_ref.astype(orig_dtype)


def _broadcast_batch_inputs(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    *,
    is_triclinic: bool = False,
) -> tuple[bool, jax.Array, jax.Array, jax.Array, jax.Array, tuple[bool, bool, bool], bool]:
    single_input = positions.ndim == 2
    pos_b = positions[None, ...] if single_input else positions
    B, n_atoms, _ = pos_b.shape
    b_q = charges.ndim == 2 and not single_input
    b_m = atom_mask.ndim == 2 and not single_input
    eff_tri = _infer_pme_is_triclinic(
        box_size, B, single_input=single_input, is_triclinic=is_triclinic
    )
    box6_b, _ = _normalize_box6_batch(
        box_size, B, positions.dtype, single_input=single_input, is_triclinic=eff_tri
    )
    b_box = (not single_input) and (int(box6_b.shape[0]) == B) and (B > 1 or jnp.asarray(box_size).ndim >= 2)
    q_b = charges if b_q else jnp.broadcast_to(charges.reshape(1, n_atoms), (B, n_atoms))
    m_b = atom_mask if b_m else jnp.broadcast_to(atom_mask.reshape(1, n_atoms), (B, n_atoms))
    if eff_tri:
        bx_b = _box6_to_3x3_batch(box6_b, B)
    else:
        bx_b = jnp.broadcast_to(box6_b[:, :3], (B, 3))
    return single_input, pos_b, q_b, m_b, bx_b, (b_q, b_m, b_box), eff_tri


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _spme_energy_tiered_diff(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    tier_plan: PMETierPlan,
    use_pallas_pme: bool,
    interpret: bool | None,
    is_triclinic: bool,
) -> jax.Array:
    single_input, pos_b, q_b, m_b, bx_b, _, eff_tri = _broadcast_batch_inputs(
        positions, charges, atom_mask, box_size, is_triclinic=is_triclinic
    )
    B = int(pos_b.shape[0])
    orig_dtype = positions.dtype
    if B == 0 or pos_b.shape[1] == 0:
        z = jnp.zeros((B,), dtype=orig_dtype)
        return z[0] if single_input else z

    if len(tier_plan.tiers) == 1 and tier_plan.tiers[0].replica_indices_tuple == tuple(range(B)):
        t0 = tier_plan.tiers[0]
        n_sl = t0.n_slice
        e_vec, _, _ = _eval_single_tier_forward(
            pos_b[:, :n_sl, :],
            q_b[:, :n_sl],
            m_b[:, :n_sl],
            bx_b,
            t0,
            tier_plan.alpha,
            tier_plan.order,
            use_pallas_pme,
            interpret,
            is_triclinic=eff_tri,
        )
        return e_vec[0] if single_input else e_vec

    e_out = jnp.zeros((B,), dtype=orig_dtype)
    for tier in tier_plan.tiers:
        idx = tier.replica_indices
        n_sl = tier.n_slice
        e_t, _, _ = _eval_single_tier_forward(
            pos_b[idx, :n_sl, :],
            q_b[idx, :n_sl],
            m_b[idx, :n_sl],
            bx_b[idx],
            tier,
            tier_plan.alpha,
            tier_plan.order,
            use_pallas_pme,
            interpret,
            is_triclinic=eff_tri,
        )
        e_out = e_out.at[idx].set(e_t)
    return e_out[0] if single_input else e_out


def _spme_energy_tiered_fwd(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    tier_plan: PMETierPlan,
    use_pallas_pme: bool,
    interpret: bool | None,
    is_triclinic: bool,
):
    pos_v = _unwrap_primal(positions)
    q_v = _unwrap_primal(charges)
    m_v = _unwrap_primal(atom_mask)
    bx_v = _unwrap_primal(box_size)
    plan_v = jax.tree_util.tree_map(_unwrap_primal, tier_plan)
    q_pert = _is_perturbed(charges)
    bx_pert = _is_perturbed(box_size)

    single_input, pos_b, q_b, m_b, bx_b, batched_flags, eff_tri = _broadcast_batch_inputs(
        pos_v, q_v, m_v, bx_v, is_triclinic=is_triclinic
    )
    B = int(pos_b.shape[0])
    orig_dtype = pos_v.dtype

    if B == 0 or pos_b.shape[1] == 0:
        z = jnp.zeros((B,), dtype=orig_dtype)
        out = z[0] if single_input else z
        return out, (
            pos_v,
            q_v,
            m_v,
            bx_v,
            plan_v,
            (),
            single_input,
            batched_flags,
            q_pert,
            bx_pert,
            eff_tri,
        )

    if len(plan_v.tiers) == 1 and plan_v.tiers[0].replica_indices_tuple == tuple(range(B)):
        t0 = plan_v.tiers[0]
        n_sl = t0.n_slice
        e_vec, th0, _ = _eval_single_tier_forward(
            pos_b[:, :n_sl, :],
            q_b[:, :n_sl],
            m_b[:, :n_sl],
            bx_b,
            t0,
            plan_v.alpha,
            plan_v.order,
            use_pallas_pme,
            interpret,
            is_triclinic=eff_tri,
        )
        out = e_vec[0] if single_input else e_vec
        return out, (
            pos_v,
            q_v,
            m_v,
            bx_v,
            plan_v,
            (th0,),
            single_input,
            batched_flags,
            q_pert,
            bx_pert,
            eff_tri,
        )

    e_out = jnp.zeros((B,), dtype=orig_dtype)
    thetas: list[jax.Array] = []
    for tier in plan_v.tiers:
        idx = tier.replica_indices
        n_sl = tier.n_slice
        e_t, th_t, _ = _eval_single_tier_forward(
            pos_b[idx, :n_sl, :],
            q_b[idx, :n_sl],
            m_b[idx, :n_sl],
            bx_b[idx],
            tier,
            plan_v.alpha,
            plan_v.order,
            use_pallas_pme,
            interpret,
            is_triclinic=eff_tri,
        )
        e_out = e_out.at[idx].set(e_t)
        thetas.append(th_t)
    out = e_out[0] if single_input else e_out
    return out, (
        pos_v,
        q_v,
        m_v,
        bx_v,
        plan_v,
        tuple(thetas),
        single_input,
        batched_flags,
        q_pert,
        bx_pert,
        eff_tri,
    )


def _spme_energy_tiered_bwd(
    use_pallas_pme: bool, interpret: bool | None, is_triclinic: bool, residuals, g
):
    del is_triclinic
    (
        pos_v,
        q_v,
        m_v,
        bx_v,
        plan_v,
        thetas,
        single_input,
        batched_flags,
        q_pert,
        bx_pert,
        eff_tri,
    ) = residuals
    dtype = pos_v.dtype
    bx_arr = jnp.asarray(bx_v, dtype=dtype)
    zero_plan = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x, dtype=jax.dtypes.float0)
        if jnp.issubdtype(x.dtype, jnp.integer)
        else jnp.zeros_like(x),
        plan_v,
    )
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return (
            jnp.zeros_like(pos_v),
            jnp.zeros_like(q_v),
            jnp.zeros_like(m_v, dtype=jax.dtypes.float0)
            if jnp.issubdtype(m_v.dtype, jnp.bool_) or jnp.issubdtype(m_v.dtype, jnp.integer)
            else jnp.zeros_like(m_v),
            jnp.zeros_like(bx_arr),
            zero_plan,
        )

    _, pos_b, q_b, m_b, bx_b, (b_q, _b_m, _b_box), _ = _broadcast_batch_inputs(
        pos_v, q_v, m_v, bx_arr, is_triclinic=eff_tri
    )
    B, n_atoms, _ = pos_b.shape
    g_vec = jnp.asarray(g, dtype=dtype).reshape((1,) if single_input else (B,))

    dpos_b = jnp.zeros((B, n_atoms, 3), dtype=dtype)
    dq_b = jnp.zeros((B, n_atoms), dtype=q_v.dtype)

    if len(plan_v.tiers) == 1 and plan_v.tiers[0].replica_indices_tuple == tuple(range(B)):
        t0 = plan_v.tiers[0]
        n_sl = t0.n_slice
        f_t, pot_t = _eval_single_tier_gather(
            pos_b[:, :n_sl, :],
            q_b[:, :n_sl],
            m_b[:, :n_sl],
            bx_b,
            thetas[0],
            t0,
            plan_v.order,
            use_pallas_pme,
            q_pert,
            force_scale=float(COULOMB_CONSTANT),
            pot_scale=float(COULOMB_CONSTANT),
            interpret=interpret,
            is_triclinic=eff_tri,
        )
        dpos_sub = (f_t * g_vec[:, None, None]).astype(dtype)
        dpos_b = dpos_sub if n_sl == n_atoms else dpos_b.at[:, :n_sl, :].set(dpos_sub)
        if q_pert:
            self_coeff = -2.0 * plan_v.alpha / math.sqrt(math.pi) * COULOMB_CONSTANT
            dq_sub = (
                (pot_t + jnp.asarray(self_coeff, dtype=dtype) * q_b[:, :n_sl].astype(dtype))
                * m_b[:, :n_sl].astype(dtype)
                * g_vec[:, None]
            ).astype(q_v.dtype)
            dq_b = dq_sub if n_sl == n_atoms else dq_b.at[:, :n_sl].set(dq_sub)
    else:
        self_coeff = -2.0 * plan_v.alpha / math.sqrt(math.pi) * COULOMB_CONSTANT
        for tier, th_t in zip(plan_v.tiers, thetas, strict=True):
            idx = tier.replica_indices
            n_sl = tier.n_slice
            g_t = g_vec[idx]
            f_t, pot_t = _eval_single_tier_gather(
                pos_b[idx, :n_sl, :],
                q_b[idx, :n_sl],
                m_b[idx, :n_sl],
                bx_b[idx],
                th_t,
                tier,
                plan_v.order,
                use_pallas_pme,
                q_pert,
                force_scale=float(COULOMB_CONSTANT),
                pot_scale=float(COULOMB_CONSTANT),
                interpret=interpret,
                is_triclinic=eff_tri,
            )
            dpos_b = dpos_b.at[idx, :n_sl, :].set((f_t * g_t[:, None, None]).astype(dtype))
            if q_pert:
                dq_t = (
                    (pot_t + jnp.asarray(self_coeff, dtype=dtype) * q_b[idx, :n_sl].astype(dtype))
                    * m_b[idx, :n_sl].astype(dtype)
                    * g_t[:, None]
                ).astype(q_v.dtype)
                dq_b = dq_b.at[idx, :n_sl].set(dq_t)

    if bx_pert:
        def _box_energy_vec(b_raw: jax.Array) -> jax.Array:
            _, _, _, _, bx_cur, _, _ = _broadcast_batch_inputs(
                pos_v, q_v, m_v, b_raw, is_triclinic=eff_tri
            )
            if len(plan_v.tiers) == 1 and plan_v.tiers[0].replica_indices_tuple == tuple(range(B)):
                t0_b = plan_v.tiers[0]
                ns_b = t0_b.n_slice
                return jax.vmap(
                    lambda p1, q1, m1, b1: spme_reciprocal_energy(
                        p1, q1, m1, b1, t0_b.grid_dims, plan_v.alpha, plan_v.order, False
                    )
                )(pos_b[:, :ns_b, :], q_b[:, :ns_b], m_b[:, :ns_b], bx_cur)
            e_acc = jnp.zeros((B,), dtype=dtype)
            for tier_b in plan_v.tiers:
                idx_b = tier_b.replica_indices
                ns_b = tier_b.n_slice
                gd_b = tier_b.grid_dims
                e_sub = jax.vmap(
                    lambda p1, q1, m1, b1, gd=gd_b: spme_reciprocal_energy(
                        p1, q1, m1, b1, gd, plan_v.alpha, plan_v.order, False
                    )
                )(pos_b[idx_b, :ns_b, :], q_b[idx_b, :ns_b], m_b[idx_b, :ns_b], bx_cur[idx_b])
                e_acc = e_acc.at[idx_b].set(e_sub.astype(dtype))
            return e_acc

        _, vjp_box_all = jax.vjp(_box_energy_vec, bx_arr)
        (dbox_out_raw,) = vjp_box_all(g_vec)
        dbox_out = dbox_out_raw.astype(bx_arr.dtype)
    else:
        dbox_out = jnp.zeros_like(bx_arr)

    dpos_out = dpos_b[0] if single_input else dpos_b
    if single_input:
        dq_out = dq_b[0]
    elif not b_q:
        dq_out = jnp.sum(dq_b, axis=0).reshape(q_v.shape)
    else:
        dq_out = dq_b

    dmask_out = (
        jnp.zeros_like(m_v, dtype=jax.dtypes.float0)
        if jnp.issubdtype(m_v.dtype, jnp.bool_) or jnp.issubdtype(m_v.dtype, jnp.integer)
        else jnp.zeros_like(m_v)
    )
    return dpos_out, dq_out, dmask_out, dbox_out, zero_plan


_spme_energy_tiered_diff.defvjp(
    _spme_energy_tiered_fwd,
    _spme_energy_tiered_bwd,
    symbolic_zeros=True,
)


def spme_energy_ragged_tiered(  # noqa: PLR0913
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    *,
    tier_plan: PMETierPlan | None = None,
    alpha: float | None = None,
    cutoff: float = 9.0,
    ewald_error_tolerance: float = 5e-4,
    order: int = 4,
    pme_tier_mode: PMETierMode = "auto",
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    grid_dims_per_replica: Sequence[tuple[int, int, int]] | None = None,
    tier_launch_overhead_us: float = 15.0,
    fft_points_per_us: float = 18000.0,
    use_pallas_pme: bool = True,
    interpret: bool | None = None,
    is_triclinic: bool = False,
) -> jax.Array:
    """Compute SPME reciprocal + self energy ``(B,)`` using 1D ragged queues & multi-tier FFTs."""
    b_hint = 1 if positions.ndim == 2 else int(positions.shape[0])
    eff_tri = _infer_pme_is_triclinic(
        box_size, b_hint, single_input=(positions.ndim == 2), is_triclinic=is_triclinic
    )
    if tier_plan is None:
        tier_plan = plan_pme_tiers(
            box_size,
            cutoff=cutoff,
            ewald_error_tolerance=ewald_error_tolerance,
            alpha=alpha,
            order=order,
            grid_dims_per_replica=grid_dims_per_replica,
            n_real_per_replica=n_real_per_replica if n_real_per_replica is not None else ([int(positions.shape[0])] if positions.ndim == 2 else None),
            padded_atoms=int(positions.shape[-2]),
            atom_mask=atom_mask,
            pme_tier_mode=pme_tier_mode,
            tier_launch_overhead_us=tier_launch_overhead_us,
            fft_points_per_us=fft_points_per_us,
            is_triclinic=eff_tri,
        )
    return _spme_energy_tiered_diff(
        positions,
        charges,
        atom_mask,
        box_size,
        tier_plan,
        bool(use_pallas_pme),
        interpret,
        bool(eff_tri),
    )


def spme_forces_ragged_tiered(  # noqa: PLR0913
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    *,
    tier_plan: PMETierPlan | None = None,
    alpha: float | None = None,
    cutoff: float = 9.0,
    ewald_error_tolerance: float = 5e-4,
    order: int = 4,
    pme_tier_mode: PMETierMode = "auto",
    n_real_per_replica: Sequence[int] | np.ndarray | jax.Array | None = None,
    grid_dims_per_replica: Sequence[tuple[int, int, int]] | None = None,
    tier_launch_overhead_us: float = 15.0,
    fft_points_per_us: float = 18000.0,
    use_pallas_pme: bool = True,
    interpret: bool | None = None,
    is_triclinic: bool = False,
) -> jax.Array:
    """Direct SPME reciprocal force evaluation ``F = -dE/dr`` without scalar energy reductions."""
    if in_export_context():
        if positions.ndim == 2:
            return -jax.grad(
                lambda r: spme_energy_ragged_tiered(
                    r,
                    charges,
                    atom_mask,
                    box_size,
                    tier_plan=tier_plan,
                    alpha=alpha,
                    cutoff=cutoff,
                    ewald_error_tolerance=ewald_error_tolerance,
                    order=order,
                    pme_tier_mode=pme_tier_mode,
                    n_real_per_replica=n_real_per_replica,
                    grid_dims_per_replica=grid_dims_per_replica,
                    tier_launch_overhead_us=tier_launch_overhead_us,
                    fft_points_per_us=fft_points_per_us,
                    use_pallas_pme=False,
                    interpret=interpret,
                    is_triclinic=is_triclinic,
                )
            )(positions)
        return -jax.grad(
            lambda r: jnp.sum(
                spme_energy_ragged_tiered(
                    r,
                    charges,
                    atom_mask,
                    box_size,
                    tier_plan=tier_plan,
                    alpha=alpha,
                    cutoff=cutoff,
                    ewald_error_tolerance=ewald_error_tolerance,
                    order=order,
                    pme_tier_mode=pme_tier_mode,
                    n_real_per_replica=n_real_per_replica,
                    grid_dims_per_replica=grid_dims_per_replica,
                    tier_launch_overhead_us=tier_launch_overhead_us,
                    fft_points_per_us=fft_points_per_us,
                    use_pallas_pme=False,
                    interpret=interpret,
                    is_triclinic=is_triclinic,
                )
            )
        )(positions)

    single_input, pos_b, q_b, m_b, bx_b, _, eff_tri = _broadcast_batch_inputs(
        positions, charges, atom_mask, box_size, is_triclinic=is_triclinic
    )
    B, n_atoms, _ = pos_b.shape
    orig_dtype = positions.dtype
    if B == 0 or n_atoms == 0:
        zf = jnp.zeros((B, n_atoms, 3), dtype=orig_dtype)
        return zf[0] if single_input else zf

    if tier_plan is None:
        tier_plan = plan_pme_tiers(
            box_size,
            cutoff=cutoff,
            ewald_error_tolerance=ewald_error_tolerance,
            alpha=alpha,
            order=order,
            grid_dims_per_replica=grid_dims_per_replica,
            n_real_per_replica=n_real_per_replica if n_real_per_replica is not None else ([n_atoms] if single_input else None),
            padded_atoms=n_atoms,
            atom_mask=m_b,
            pme_tier_mode=pme_tier_mode,
            tier_launch_overhead_us=tier_launch_overhead_us,
            fft_points_per_us=fft_points_per_us,
            is_triclinic=eff_tri,
        )

    def _run_one_tier_forces(
        p_sub: jax.Array,
        q_sub: jax.Array,
        m_sub: jax.Array,
        b_sub: jax.Array,
        tier: PMETierGroup,
    ) -> jax.Array:
        Kx, Ky, Kz = tier.grid_dims
        use_pl = bool(use_pallas_pme)
        work_dtype = jnp.float32 if use_pl else orig_dtype
        p_w = p_sub.astype(work_dtype)
        q_w = q_sub.astype(work_dtype)
        b_w = jnp.asarray(b_sub, dtype=work_dtype)
        if use_pl:
            Q = spread_charges_pallas_ragged(
                p_w,
                q_w,
                m_sub,
                b_w,
                tier.grid_dims,
                queue=tier.ragged_queue,
                order=tier_plan.order,
                interpret=interpret,
                is_triclinic=eff_tri,
            ).astype(jnp.float32)
        else:
            Q = _spread_reference_batched(
                p_w,
                q_w,
                m_sub,
                b_w,
                tier.grid_dims,
                tier_plan.order,
            )
        Q_hat = jnp.fft.rfftn(Q, s=tier.grid_dims, axes=(-3, -2, -1))
        G = _resolve_tier_green(tier, b_w, tier_plan.alpha, tier_plan.order, Q.dtype)
        theta = jnp.fft.irfftn(Q_hat * G, s=tier.grid_dims, axes=(-3, -2, -1)).astype(Q.dtype)
        theta_norm = (theta * jnp.asarray(float(Kx * Ky * Kz), dtype=Q.dtype)).astype(Q.dtype)
        f_sub, _ = _eval_single_tier_gather(
            p_sub,
            q_sub,
            m_sub,
            b_sub,
            theta_norm,
            tier,
            tier_plan.order,
            use_pallas_pme,
            compute_dq=False,
            force_scale=-float(COULOMB_CONSTANT),
            pot_scale=1.0,
            interpret=interpret,
            is_triclinic=eff_tri,
        )
        return f_sub

    if len(tier_plan.tiers) == 1 and tier_plan.tiers[0].replica_indices_tuple == tuple(range(B)):
        t0 = tier_plan.tiers[0]
        n_sl = t0.n_slice
        f_sub = _run_one_tier_forces(
            pos_b[:, :n_sl, :],
            q_b[:, :n_sl],
            m_b[:, :n_sl],
            bx_b,
            t0,
        )
        f_out = (
            f_sub
            if n_sl == n_atoms
            else jnp.zeros((B, n_atoms, 3), dtype=orig_dtype).at[:, :n_sl, :].set(f_sub)
        )
        return f_out[0] if single_input else f_out

    f_out = jnp.zeros((B, n_atoms, 3), dtype=orig_dtype)
    for tier in tier_plan.tiers:
        idx = tier.replica_indices
        n_sl = tier.n_slice
        f_t = _run_one_tier_forces(
            pos_b[idx, :n_sl, :],
            q_b[idx, :n_sl],
            m_b[idx, :n_sl],
            bx_b[idx],
            tier,
        )
        f_out = f_out.at[idx, :n_sl, :].set(f_t)
    return f_out[0] if single_input else f_out


def spme_energy_and_forces_ragged_tiered(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    *,
    is_triclinic: bool = False,
    **kwargs,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(energies, forces)`` for multi-tier ragged SPME."""
    if positions.ndim == 2:
        e, g = jax.value_and_grad(
            lambda r: spme_energy_ragged_tiered(
                r, charges, atom_mask, box_size, is_triclinic=is_triclinic, **kwargs
            )
        )(positions)
        return e, -g

    def _sum_fn(r):
        e_vec = spme_energy_ragged_tiered(
            r, charges, atom_mask, box_size, is_triclinic=is_triclinic, **kwargs
        )
        return jnp.sum(e_vec), e_vec

    (_, e_vec), g = jax.value_and_grad(_sum_fn, has_aux=True)(positions)
    return e_vec, -g


# =============================================================================
# 5. 1D Ragged Tile-Pair Direct Nonbonded Engine (MK-68a + MK-73)
# =============================================================================


def _wrap_pos_box6_1(pos_1: jax.Array, bx_1: jax.Array) -> jax.Array:
    """Wrap ``(N, 3)`` Cartesian coords sequentially ``z -> y -> x`` into ``[0, ax) x [0, by) x [0, cz)``."""
    dtype = pos_1.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    ax, by, cz = bx_1[0], bx_1[1], bx_1[2]
    bx_t, cx_t, cy_t = bx_1[3], bx_1[4], bx_1[5]
    px, py, pz = pos_1[:, 0], pos_1[:, 1], pos_1[:, 2]

    scz = jnp.where(cz > zero, cz, one)
    nz = jnp.where(cz > zero, jnp.floor(pz / scz), zero)
    px = px - nz * cx_t
    py = py - nz * cy_t
    pz = jnp.where(cz > zero, pz - nz * cz, pz)

    sby = jnp.where(by > zero, by, one)
    ny = jnp.where(by > zero, jnp.floor(py / sby), zero)
    px = px - ny * bx_t
    py = jnp.where(by > zero, py - ny * by, py)

    sax = jnp.where(ax > zero, ax, one)
    nx = jnp.where(ax > zero, jnp.floor(px / sax), zero)
    px = jnp.where(ax > zero, px - nx * ax, px)
    return jnp.stack([px, py, pz], axis=-1)


def _wrap_pos_box6_batch(pos_b: jax.Array, bx_b: jax.Array) -> jax.Array:
    """Wrap ``(B, N, 3)`` Cartesian coords sequentially ``z -> y -> x`` into ``[0, ax) x [0, by) x [0, cz)``."""
    dtype = pos_b.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    ax = bx_b[:, 0:1]
    by = bx_b[:, 1:2]
    cz = bx_b[:, 2:3]
    bx_t = bx_b[:, 3:4]
    cx_t = bx_b[:, 4:5]
    cy_t = bx_b[:, 5:6]
    px, py, pz = pos_b[:, :, 0], pos_b[:, :, 1], pos_b[:, :, 2]

    scz = jnp.where(cz > zero, cz, one)
    nz = jnp.where(cz > zero, jnp.floor(pz / scz), zero)
    px = px - nz * cx_t
    py = py - nz * cy_t
    pz = jnp.where(cz > zero, pz - nz * cz, pz)

    sby = jnp.where(by > zero, by, one)
    ny = jnp.where(by > zero, jnp.floor(py / sby), zero)
    px = px - ny * bx_t
    py = jnp.where(by > zero, py - ny * by, py)

    sax = jnp.where(ax > zero, ax, one)
    nx = jnp.where(ax > zero, jnp.floor(px / sax), zero)
    px = jnp.where(ax > zero, px - nx * ax, px)
    return jnp.stack([px, py, pz], axis=-1)


def _spread_bits_10_rq(v: jax.Array) -> jax.Array:
    x = v & jnp.int32(0x3FF)
    x = (x | (x << 16)) & jnp.int32(0x030000FF)
    x = (x | (x << 8)) & jnp.int32(0x0300F00F)
    x = (x | (x << 4)) & jnp.int32(0x030C30C3)
    x = (x | (x << 2)) & jnp.int32(0x09249249)
    return x


@dataclasses.dataclass(frozen=True)
class RaggedMortonTilePairs:
    """Ragged cross-size, cross-shape 32x32 Morton tile-pair state + 1D CTA work queue."""

    perm: jax.Array
    inv_perm: jax.Array
    positions_sorted: jax.Array
    atom_mask_sorted: jax.Array
    tile_pairs: jax.Array
    excl_full_mask: jax.Array
    excl_14_mask: jax.Array
    num_valid_pairs: jax.Array
    pair_queue: RaggedWorkQueue
    box6: jax.Array
    scale_14_vdw: jax.Array
    scale_14_elec: jax.Array
    n_pad: int
    max_pairs: int
    is_triclinic: bool

    def reorder_atom_array(self, arr: jax.Array, *, pad_value: float | bool = 0.0) -> jax.Array:
        """Pad & gather ``(B, N, ...)`` or ``(N, ...)`` into Morton-sorted ``(B, n_pad, ...)``."""
        B = int(self.perm.shape[0])
        single = arr.ndim == 1 or (arr.ndim == 2 and int(arr.shape[0]) != B and int(arr.shape[-1]) == 3)
        a_b = arr[None, ...] if single else arr
        if int(a_b.shape[0]) == 1 and B > 1:
            a_b = jnp.broadcast_to(a_b, (B, *a_b.shape[1:]))
        n_in = int(a_b.shape[1])
        pad_n = self.n_pad - n_in
        if pad_n > 0:
            pad_spec = [(0, 0), (0, pad_n)] + [(0, 0)] * (a_b.ndim - 2)
            a_b = jnp.pad(a_b, pad_spec, constant_values=pad_value)
        elif pad_n < 0:
            a_b = a_b[:, : self.n_pad]
        b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]
        return a_b[b_idx, self.perm]

    def unsort_atom_array(self, arr_sorted: jax.Array, *, n_orig: int | None = None) -> jax.Array:
        """Restore Morton-sorted ``(B, n_pad, ...)`` back to original atom order ``(B, n_orig, ...)``."""
        B = int(self.inv_perm.shape[0])
        b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]
        restored = arr_sorted[b_idx, self.inv_perm]
        mask_orig = self.atom_mask_sorted[b_idx, self.inv_perm]
        m_exp = mask_orig.reshape((B, self.n_pad) + (1,) * (restored.ndim - 2))
        restored = jnp.where(m_exp, restored, jnp.zeros_like(restored))
        if n_orig is not None:
            return restored[:, : int(n_orig)]
        return restored

    def tree_flatten(self):
        children = (
            self.perm,
            self.inv_perm,
            self.positions_sorted,
            self.atom_mask_sorted,
            self.tile_pairs,
            self.excl_full_mask,
            self.excl_14_mask,
            self.num_valid_pairs,
            self.pair_queue,
            self.box6,
            self.scale_14_vdw,
            self.scale_14_elec,
        )
        aux = (self.n_pad, self.max_pairs, self.is_triclinic)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_pad, max_pairs, is_tri = aux
        return cls(*children, n_pad=n_pad, max_pairs=max_pairs, is_triclinic=is_tri)


jax.tree_util.register_pytree_node_class(RaggedMortonTilePairs)


def _min_periodic_subtile_gap_sq(
    c_i: jax.Array,
    h_i: jax.Array,
    c_j: jax.Array,
    h_j: jax.Array,
    bx_1: jax.Array,
    *,
    has_box: bool,
) -> jax.Array:
    """Minimum squared Euclidean AABB gap across ``4 x 4`` sub-tiles for all ``(T, T)`` tile pairs.

    ``c_i, h_i`` have shape ``(T, 4, 3)`` (center and half-extents of the 4 8-atom sub-tiles
    per 32-atom tile). When ``has_box`` is True, evaluates the exact minimum over all integer
    Bravais lattice shifts ``n_z c + n_y b + n_x a`` via 9 cascading ``(dnz, dny)`` shifts and
    analytic nearest ``dnx``, guaranteeing zero missed cutoff pairs on skewed dodecahedron and
    truncated-octahedron boundaries.
    """
    dtype = c_i.dtype
    zf = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)

    # Expand to (T, T, 4, 4)
    d0x = c_i[:, None, :, None, 0] - c_j[None, :, None, :, 0]
    d0y = c_i[:, None, :, None, 1] - c_j[None, :, None, :, 1]
    d0z = c_i[:, None, :, None, 2] - c_j[None, :, None, :, 2]
    hxs = h_i[:, None, :, None, 0] + h_j[None, :, None, :, 0]
    hys = h_i[:, None, :, None, 1] + h_j[None, :, None, :, 1]
    hzs = h_i[:, None, :, None, 2] + h_j[None, :, None, :, 2]

    if not has_box:
        gx = jnp.maximum(zf, jnp.abs(d0x) - hxs)
        gy = jnp.maximum(zf, jnp.abs(d0y) - hys)
        gz = jnp.maximum(zf, jnp.abs(d0z) - hzs)
        return jnp.min(gx * gx + gy * gy + gz * gz, axis=(-2, -1))

    ax, by, cz = bx_1[0], bx_1[1], bx_1[2]
    bx_t, cx_t, cy_t = bx_1[3], bx_1[4], bx_1[5]
    sax = jnp.where(ax > zf, ax, one)
    sby = jnp.where(by > zf, by, one)
    scz = jnp.where(cz > zf, cz, one)
    inv_ax = jnp.where(ax > zf, one / sax, zf)
    inv_by = jnp.where(by > zf, one / sby, zf)
    inv_cz = jnp.where(cz > zf, one / scz, zf)

    qz0 = zf - jnp.floor(d0z * inv_cz + half)
    min_d2 = jnp.full(d0x.shape, 1.0e30, dtype=dtype)
    for dnz_f in (-1.0, 0.0, 1.0):
        kz = qz0 + jnp.asarray(dnz_f, dtype=dtype)
        dz = jnp.where(cz > zf, d0z + kz * cz, d0z)
        gz = jnp.maximum(zf, jnp.abs(dz) - hzs)
        gz2 = gz * gz
        y1 = jnp.where(cz > zf, d0y + kz * cy_t, d0y)
        qy0 = zf - jnp.floor(y1 * inv_by + half)
        for dny_f in (-1.0, 0.0, 1.0):
            ky = qy0 + jnp.asarray(dny_f, dtype=dtype)
            dy = jnp.where(by > zf, y1 + ky * by, y1)
            gy = jnp.maximum(zf, jnp.abs(dy) - hys)
            x1 = y1 * zf + d0x + jnp.where(cz > zf, kz * cx_t, zf) + jnp.where(by > zf, ky * bx_t, zf)
            dx = jnp.where(ax > zf, x1 - sax * jnp.floor(x1 * inv_ax + half), x1)
            gx = jnp.maximum(zf, jnp.abs(dx) - hxs)
            min_d2 = jnp.minimum(min_d2, gx * gx + gy * gy + gz2)
    return jnp.min(min_d2, axis=(-2, -1))


def build_ragged_morton_tile_pairs(
    positions: jax.Array,
    atom_mask: jax.Array,
    box: jax.Array | None,
    *,
    cutoff: float = 9.0,
    skin: float = 2.0,
    excl_indices: np.ndarray | jax.Array | None = None,
    excl_scales_vdw: np.ndarray | jax.Array | None = None,
    excl_scales_elec: np.ndarray | jax.Array | None = None,
    max_pairs: int | None = None,
    tp_group: int | None = None,
    is_triclinic: bool = False,
) -> RaggedMortonTilePairs:
    """Build 32x32 fractional Morton-sorted tile pairs & 1D ragged CTA queue across cross-size/cross-shape batches.

    Supports arbitrary mixtures of orthorhombic, rhombic-dodecahedron, and truncated-octahedron
    replicas in a single ``(B, N_pad, 3)`` batch (``B == 3`` safe).
    """
    single_input = positions.ndim == 2
    pos_in = positions[None, ...] if single_input else positions
    B, n_in, _ = pos_in.shape
    m_in = atom_mask[None, ...] if atom_mask.ndim == 1 else atom_mask
    if int(m_in.shape[0]) == 1 and B > 1:
        m_in = jnp.broadcast_to(m_in, (B, n_in))
    m_bool_in = m_in > 0

    n_blocks = max(1, (max(n_in, 1) + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    n_pad = n_blocks * _NB_BLOCK_N
    pad_n = n_pad - n_in
    if pad_n > 0:
        pos_pad = jnp.pad(pos_in, ((0, 0), (0, pad_n), (0, 0)))
        mask_pad = jnp.pad(m_bool_in, ((0, 0), (0, pad_n)), constant_values=False)
    else:
        pos_pad = pos_in
        mask_pad = m_bool_in

    eff_tri = (
        _infer_pme_is_triclinic(box, B, single_input=single_input, is_triclinic=is_triclinic)
        if box is not None
        else False
    )
    box6_raw, has_box = _normalize_box6_batch(
        box, max(B, 1), jnp.float32, single_input=single_input, is_triclinic=eff_tri
    )
    box6_b = jnp.broadcast_to(box6_raw, (max(B, 1), 6))[:B]

    pos_f32 = pos_pad.astype(jnp.float32)
    if has_box and B > 0:
        h_mats = _box6_to_3x3_batch(box6_b, B)
        b_inv = jax.vmap(invert_box_lower_triangular)(h_mats)
        s_frac = jnp.matmul(pos_f32, b_inv, precision=jax.lax.Precision.HIGHEST)
        s_wrap = s_frac - jnp.floor(s_frac)
    elif B > 0:
        p_masked_min = jnp.where(mask_pad[:, :, None], pos_f32, jnp.asarray(1.0e30, dtype=jnp.float32))
        p_masked_max = jnp.where(mask_pad[:, :, None], pos_f32, jnp.asarray(-1.0e30, dtype=jnp.float32))
        mn = jnp.min(p_masked_min, axis=1, keepdims=True)
        mx = jnp.max(p_masked_max, axis=1, keepdims=True)
        span = jnp.maximum(mx - mn, jnp.asarray(1.0e-3, dtype=jnp.float32))
        s_wrap = jnp.clip((pos_f32 - mn) / span, 0.0, 0.999999)
    else:
        s_wrap = jnp.zeros_like(pos_f32)

    q_10 = jnp.clip(jnp.floor(s_wrap * 1024.0).astype(jnp.int32), 0, 1023)
    codes = (
        _spread_bits_10_rq(q_10[..., 0])
        | (_spread_bits_10_rq(q_10[..., 1]) << 1)
        | (_spread_bits_10_rq(q_10[..., 2]) << 2)
    )
    codes = jnp.where(mask_pad, codes, jnp.int32(0x7FFFFFFF))
    perm = jnp.argsort(codes, axis=-1).astype(jnp.int32)
    inv_perm = jnp.argsort(perm, axis=-1).astype(jnp.int32)

    b_gather = jnp.arange(B, dtype=jnp.int32)[:, None]
    pos_sorted = pos_pad[b_gather, perm]
    mask_sorted = mask_pad[b_gather, perm]

    upper_cap = n_blocks * (n_blocks + 1) // 2
    if max_pairs is not None and int(max_pairs) > 0:
        m_cap = int(max_pairs)
    else:
        r_eff_f = max(float(cutoff) + float(skin), 6.0)
        est_per = int(max(96.0, 0.5 * (4.189 * ((r_eff_f + 7.5) ** 3) / 320.0) * 2.8))
        m_cap = max(64, min(upper_cap, ((max(2048, n_blocks * est_per) + 63) // 64) * 64))

    r_eff_sq = float((float(cutoff) + float(skin)) ** 2)
    tri_ti, tri_tj = jnp.triu_indices(n_blocks)

    def _build_subtile_aabbs_1(pos_s1: jax.Array, msk_s1: jax.Array, bx_1: jax.Array):
        p_w = _wrap_pos_box6_1(pos_s1.astype(jnp.float32), bx_1) if has_box else pos_s1.astype(jnp.float32)
        p_sub = p_w.reshape(n_blocks, 4, 8, 3)
        m_sub = msk_s1.reshape(n_blocks, 4, 8)
        sub_has = jnp.any(m_sub, axis=-1)
        first_idx = jnp.argmax(m_sub.astype(jnp.int32), axis=-1)
        ref = jnp.take_along_axis(p_sub, first_idx[:, :, None, None], axis=2)[:, :, 0, :]
        dr = p_sub - ref[:, :, None, :]
        if has_box:
            ax, by, cz = bx_1[0], bx_1[1], bx_1[2]
            bx_t, cx_t, cy_t = bx_1[3], bx_1[4], bx_1[5]
            zf = jnp.asarray(0.0, dtype=jnp.float32)
            one = jnp.asarray(1.0, dtype=jnp.float32)
            half = jnp.asarray(0.5, dtype=jnp.float32)
            dx, dy, dz = dr[..., 0], dr[..., 1], dr[..., 2]
            sz = jnp.where(cz > zf, jnp.floor(dz / jnp.where(cz > zf, cz, one) + half), zf)
            dx = dx - sz * cx_t
            dy = dy - sz * cy_t
            dz = dz - sz * cz
            sy = jnp.where(by > zf, jnp.floor(dy / jnp.where(by > zf, by, one) + half), zf)
            dx = dx - sy * bx_t
            dy = dy - sy * by
            sx = jnp.where(ax > zf, jnp.floor(dx / jnp.where(ax > zf, ax, one) + half), zf)
            dx = dx - sx * ax
            p_unw = ref[:, :, None, :] + jnp.stack([dx, dy, dz], axis=-1)
        else:
            p_unw = p_sub
        p_lo = jnp.min(jnp.where(m_sub[..., None], p_unw, ref[:, :, None, :]), axis=2)
        p_hi = jnp.max(jnp.where(m_sub[..., None], p_unw, ref[:, :, None, :]), axis=2)
        centers = 0.5 * (p_lo + p_hi)
        halves = jnp.where(sub_has[..., None], 0.5 * (p_hi - p_lo), jnp.asarray(-1.0e10, dtype=jnp.float32))
        tile_has = jnp.any(sub_has, axis=-1)
        gap_sq = _min_periodic_subtile_gap_sq(centers, halves, centers, halves, bx_1, has_box=has_box)
        active_mat = tile_has[:, None] & tile_has[None, :] & (gap_sq <= jnp.asarray(r_eff_sq, dtype=jnp.float32))
        return active_mat[tri_ti, tri_tj]

    if B > 0:
        keep_ut_b = jax.vmap(_build_subtile_aabbs_1)(pos_sorted, mask_sorted, box6_b)
    else:
        keep_ut_b = jnp.zeros((0, upper_cap), dtype=jnp.bool_)

    # Host/concrete fast exclusion stamping + compaction when inputs are concrete
    is_traced = (
        isinstance(pos_sorted, jax.core.Tracer)
        or isinstance(mask_sorted, jax.core.Tracer)
        or isinstance(box6_b, jax.core.Tracer)
    )

    s14v_np = np.full((max(B, 1),), 0.5, dtype=np.float32)
    s14e_np = np.full((max(B, 1),), float(1.0 / 1.2), dtype=np.float32)

    if not is_traced and B > 0:
        keep_np = np.asarray(keep_ut_b, dtype=bool).copy()
        inv_perm_np = np.asarray(inv_perm, dtype=np.int32)
        mask_in_np = np.asarray(m_bool_in, dtype=bool)
        tri_ti_np = np.asarray(tri_ti, dtype=np.int32)
        tri_tj_np = np.asarray(tri_tj, dtype=np.int32)

        tp_np = np.full((B, m_cap, 2), -1, dtype=np.int32)
        ef_np = np.zeros((B, m_cap, _NB_BLOCK_N), dtype=np.uint32)
        e14_np = np.zeros((B, m_cap, _NB_BLOCK_N), dtype=np.uint32)
        n_val_np = np.zeros((B,), dtype=np.int32)

        excl_idx_np = np.asarray(excl_indices, dtype=np.int32) if excl_indices is not None else None
        excl_sv_np = np.asarray(excl_scales_vdw, dtype=np.float32) if excl_scales_vdw is not None else None
        excl_se_np = np.asarray(excl_scales_elec, dtype=np.float32) if excl_scales_elec is not None else None

        for b in range(B):
            edges_full: tuple[np.ndarray, np.ndarray] | None = None
            edges_14: tuple[np.ndarray, np.ndarray] | None = None
            if excl_idx_np is not None:
                idx_b = excl_idx_np[b if (excl_idx_np.ndim == 3 and excl_idx_np.shape[0] > 1) else 0] if excl_idx_np.ndim == 3 else excl_idx_np
                n_rows = min(n_in, idx_b.shape[0])
                idx_b = idx_b[:n_rows]
                if excl_sv_np is not None:
                    sv_b = (excl_sv_np[b if (excl_sv_np.ndim == 3 and excl_sv_np.shape[0] > 1) else 0] if excl_sv_np.ndim == 3 else excl_sv_np)[:n_rows]
                else:
                    sv_b = np.zeros_like(idx_b, dtype=np.float32)
                if excl_se_np is not None:
                    se_b = (excl_se_np[b if (excl_se_np.ndim == 3 and excl_se_np.shape[0] > 1) else 0] if excl_se_np.ndim == 3 else excl_se_np)[:n_rows]
                else:
                    se_b = np.zeros_like(idx_b, dtype=np.float32)
                rows = np.broadcast_to(np.arange(n_rows, dtype=np.int32)[:, None], idx_b.shape)
                v_ex = (idx_b >= 0) & (idx_b < n_in) & (idx_b != rows)
                if np.any(v_ex):
                    u_orig = rows[v_ex]
                    w_orig = idx_b[v_ex]
                    act_both = mask_in_np[b, u_orig] & mask_in_np[b, w_orig]
                    sv_v = sv_b[v_ex]
                    se_v = se_b[v_ex]
                    non_unit = act_both & ((sv_v < 1.0 - 1e-6) | (se_v < 1.0 - 1e-6))
                    if np.any(non_unit):
                        um = inv_perm_np[b, u_orig[non_unit]]
                        wm = inv_perm_np[b, w_orig[non_unit]]
                        sv_nu = sv_v[non_unit]
                        se_nu = se_v[non_unit]
                        a_m = np.minimum(um, wm)
                        c_m = np.maximum(um, wm)
                        keys = a_m.astype(np.int64) * np.int64(n_pad) + c_m.astype(np.int64)
                        _, u_idx = np.unique(keys, return_index=True)
                        a_u, c_u = a_m[u_idx], c_m[u_idx]
                        sv_u, se_u = sv_nu[u_idx], se_nu[u_idx]
                        is_f = (sv_u <= 1e-6) & (se_u <= 1e-6)
                        is_14 = ~is_f
                        if np.any(is_14):
                            s14v_np[b] = float(np.median(sv_u[is_14]))
                            s14e_np[b] = float(np.median(se_u[is_14]))
                        if np.any(is_f):
                            edges_full = (a_u[is_f], c_u[is_f])
                        if np.any(is_14):
                            edges_14 = (a_u[is_14], c_u[is_14])
                        # Ensure every tile pair carrying an excluded edge is retained
                        for e_tup in (edges_full, edges_14):
                            if e_tup is not None:
                                eti = e_tup[0] // _NB_BLOCK_N
                                etj = e_tup[1] // _NB_BLOCK_N
                                ut_pos = eti * n_blocks - (eti * (eti - 1)) // 2 + (etj - eti)
                                keep_np[b, ut_pos] = True

            sel = np.nonzero(keep_np[b])[0]
            cnt = min(int(sel.size), m_cap)
            n_val_np[b] = cnt
            if cnt > 0:
                sel_c = sel[:cnt]
                tp_np[b, :cnt, 0] = tri_ti_np[sel_c]
                tp_np[b, :cnt, 1] = tri_tj_np[sel_c]
                ut_to_slot = np.full((upper_cap,), -1, dtype=np.int32)
                ut_to_slot[sel_c] = np.arange(cnt, dtype=np.int32)
                for e_tup, target_arr in ((edges_full, ef_np), (edges_14, e14_np)):
                    if e_tup is not None:
                        ea, ec = e_tup
                        eti = ea // _NB_BLOCK_N
                        eli = ea % _NB_BLOCK_N
                        etj = ec // _NB_BLOCK_N
                        elj = ec % _NB_BLOCK_N
                        ut_pos = eti * n_blocks - (eti * (eti - 1)) // 2 + (etj - eti)
                        slots = ut_to_slot[ut_pos]
                        ok = slots >= 0
                        if np.any(ok):
                            bits = np.left_shift(np.uint32(1), elj[ok].astype(np.uint32))
                            np.bitwise_or.at(target_arr[b], (slots[ok], eli[ok]), bits)

        tile_pairs_arr = jnp.asarray(tp_np, dtype=jnp.int32)
        excl_full_arr = jax.lax.bitcast_convert_type(jnp.asarray(ef_np, dtype=jnp.uint32), jnp.int32)
        excl_14_arr = jax.lax.bitcast_convert_type(jnp.asarray(e14_np, dtype=jnp.uint32), jnp.int32)
        num_valid_arr = jnp.asarray(n_val_np, dtype=jnp.int32)
        pair_queue = build_ragged_tile_pair_queue(
            n_val_np,
            tp_group=tp_group,
            max_tile_pairs=m_cap,
        )
    else:
        # Traced fallback (e.g. inside jax.jit): pure-JAX compaction with static upper-bound queue
        m_cap = upper_cap if max_pairs is None else m_cap

        def _compact_traced_1(keep_1):
            n_v = jnp.minimum(jnp.sum(keep_1.astype(jnp.int32)), jnp.int32(m_cap))
            order_k = jnp.argsort(~keep_1)[:m_cap]
            valid_k = jnp.arange(m_cap, dtype=jnp.int32) < n_v
            ti_s = jnp.where(valid_k, tri_ti[order_k], jnp.int32(-1))
            tj_s = jnp.where(valid_k, tri_tj[order_k], jnp.int32(-1))
            return jnp.stack([ti_s, tj_s], axis=-1), n_v

        if B > 0:
            tile_pairs_arr, num_valid_arr = jax.vmap(_compact_traced_1)(keep_ut_b)
        else:
            tile_pairs_arr = jnp.full((0, m_cap, 2), -1, dtype=jnp.int32)
            num_valid_arr = jnp.zeros((0,), dtype=jnp.int32)
        excl_full_arr = jnp.zeros((B, m_cap, _NB_BLOCK_N), dtype=jnp.int32)
        excl_14_arr = jnp.zeros((B, m_cap, _NB_BLOCK_N), dtype=jnp.int32)
        if not isinstance(m_bool_in, jax.core.Tracer) and B > 0:
            n_real_est = _infer_n_real_np(None, m_bool_in)
            t_est = (n_real_est + _NB_BLOCK_N - 1) // _NB_BLOCK_N
            ub_pairs = np.minimum(t_est * (t_est + 1) // 2, m_cap).astype(np.int32)
        else:
            ub_pairs = np.full((B,), m_cap, dtype=np.int32)
        base_q = build_ragged_tile_pair_queue(ub_pairs, tp_group=tp_group, max_tile_pairs=m_cap)
        pair_queue = RaggedWorkQueue(
            cta_to_replica=base_q.cta_to_replica,
            cta_to_local_block=base_q.cta_to_local_block,
            n_real_per_replica=num_valid_arr,
            blocks_per_replica=base_q.blocks_per_replica,
            block_offsets=base_q.block_offsets,
            total_blocks=base_q.total_blocks,
            tile_size=base_q.tile_size,
            batch_size=base_q.batch_size,
            n_real_tuple=base_q.n_real_tuple,
            uniform_blocks_per_replica=base_q.uniform_blocks_per_replica,
        )

    return RaggedMortonTilePairs(
        perm=perm,
        inv_perm=inv_perm,
        positions_sorted=pos_sorted,
        atom_mask_sorted=mask_sorted,
        tile_pairs=tile_pairs_arr,
        excl_full_mask=excl_full_arr,
        excl_14_mask=excl_14_arr,
        num_valid_pairs=num_valid_arr,
        pair_queue=pair_queue,
        box6=box6_b,
        scale_14_vdw=jnp.asarray(s14v_np[:B], dtype=jnp.float32),
        scale_14_elec=jnp.asarray(s14e_np[:B], dtype=jnp.float32),
        n_pad=n_pad,
        max_pairs=m_cap,
        is_triclinic=bool(eff_tri),
    )


_VALID_LJ_COMBINING_RULES: frozenset[str] = frozenset({"lorentz_berthelot", "geometric"})


def _normalize_lj_combining_rule(lj_combining_rule: str) -> str:
    rule = str(lj_combining_rule).strip().lower().replace("-", "_")
    if rule not in _VALID_LJ_COMBINING_RULES:
        raise ValueError(
            f"Unsupported lj_combining_rule={lj_combining_rule!r}; "
            "expected 'lorentz_berthelot' or 'geometric'."
        )
    return rule


def _reaction_field_k_c(cutoff: float, rf_dielectric: float) -> tuple[float, float]:
    """Compute OpenMM reaction-field ``(k_rf, c_rf)`` safely including conductor ``rf_dielectric=inf``."""
    rc = float(cutoff)
    if rc <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff!r}")
    eps = float(rf_dielectric)
    if eps <= 0.0 or math.isnan(eps):
        raise ValueError(f"rf_dielectric must be positive, got {rf_dielectric!r}")
    inv_rc = 1.0 / rc
    inv_rc3 = inv_rc * inv_rc * inv_rc
    inv_eps = 1.0 / eps  # 0.0 when eps == +inf (conductor limit), avoiding inf/inf NaN
    denom = 2.0 + inv_eps
    k_rf = inv_rc3 * ((1.0 - inv_eps) / denom)
    c_rf = inv_rc * (3.0 / denom)
    return k_rf, c_rf


def _nonbonded_tile_pairs_ragged_reference(  # noqa: PLR0913
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_pairs: jax.Array,
    excl_full_mask: jax.Array,
    excl_14_mask: jax.Array,
    box6: jax.Array,
    lam_arr: jax.Array,
    scale_14_vdw: jax.Array,
    scale_14_elec: jax.Array,
    *,
    has_box: bool,
    cutoff: float,
    pme_alpha: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
    lj_combining_rule: str = "lorentz_berthelot",
    rf_dielectric: float | None = None,
    deterministic_forces: bool = False,
    n_valid_per_replica: jax.Array | None = None,
) -> jax.Array:
    """Pure-JAX tile-pair reference for StableHLO export (zero dense N^2 allocation)."""
    del deterministic_forces
    rule = _normalize_lj_combining_rule(lj_combining_rule)
    is_geometric = rule == "geometric"
    has_rf = (pme_alpha <= 0.0) and (rf_dielectric is not None)
    k_rf_f, c_rf_f = (
        _reaction_field_k_c(cutoff, rf_dielectric)
        if has_rf and rf_dielectric is not None
        else (0.0, 0.0)
    )

    B, N, _ = positions.shape
    M_pairs = int(tile_pairs.shape[1])
    dtype = positions.dtype
    if B == 0 or N == 0 or M_pairs == 0:
        return jnp.zeros((B,), dtype=dtype)

    n_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    n_pad = n_blocks * _NB_BLOCK_N
    pad_n = n_pad - N
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    four = jnp.asarray(4.0, dtype=dtype)

    def _one_replica(b_idx: jax.Array) -> jax.Array:
        pos_1 = positions[b_idx]
        q_1 = charges[b_idx if charges.shape[0] > 1 else 0]
        sig_1 = sigmas[b_idx if sigmas.shape[0] > 1 else 0]
        eps_1 = epsilons[b_idx if epsilons.shape[0] > 1 else 0]
        msk_1 = atom_mask[b_idx if atom_mask.shape[0] > 1 else 0].astype(jnp.bool_)
        tp_1 = tile_pairs[b_idx if tile_pairs.shape[0] > 1 else 0]
        ef_1 = excl_full_mask[b_idx if excl_full_mask.shape[0] > 1 else 0]
        e14_1 = excl_14_mask[b_idx if excl_14_mask.shape[0] > 1 else 0]
        bx_1 = box6[b_idx if box6.shape[0] > 1 else 0]
        lam_1 = lam_arr[b_idx if lam_arr.shape[0] > 1 else 0]
        s14_v = scale_14_vdw[b_idx if scale_14_vdw.shape[0] > 1 else 0]
        s14_e = scale_14_elec[b_idx if scale_14_elec.shape[0] > 1 else 0]
        if n_valid_per_replica is not None:
            n_val_1 = n_valid_per_replica[
                b_idx if n_valid_per_replica.shape[0] > 1 else 0
            ].astype(jnp.int32)
        else:
            n_val_1 = jnp.int32(M_pairs)

        if has_box:
            pos_1 = _wrap_pos_box6_1(pos_1, bx_1)

        if pad_n > 0:
            pos_1 = jnp.pad(pos_1, ((0, pad_n), (0, 0)))
            q_1 = jnp.pad(q_1, (0, pad_n))
            sig_1 = jnp.pad(sig_1, (0, pad_n), constant_values=1.0)
            eps_1 = jnp.pad(eps_1, (0, pad_n))
            msk_1 = jnp.pad(msk_1, (0, pad_n), constant_values=False)

        lane = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
        upper_tri = lane[:, None] < lane[None, :]
        shifts = lane[None, :]
        oml = one - lam_1
        oml_pow = one
        for _ in range(int(softcore_exponent)):
            oml_pow = oml_pow * oml
        soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * oml_pow
        cut_sq = jnp.asarray(cutoff * cutoff, dtype=dtype)

        def _pair_energy(p_idx, tp_entry, ef_bits, e14_bits):
            ti, tj = tp_entry[0], tp_entry[1]
            v_tp = (
                (p_idx < n_val_1)
                & (ti >= 0)
                & (tj >= 0)
                & (ti < n_blocks)
                & (tj < n_blocks)
            )
            s_ti = jnp.clip(ti, 0, n_blocks - 1)
            s_tj = jnp.clip(tj, 0, n_blocks - 1)
            oi = s_ti * _NB_BLOCK_N + lane
            oj = s_tj * _NB_BLOCK_N + lane

            pi = pos_1[oi]
            pj = pos_1[oj]
            mi = msk_1[oi] & (oi < N)
            mj = msk_1[oj] & (oj < N)
            is_full = ((ef_bits[:, None] >> shifts) & jnp.int32(1)) != 0
            is_14 = ((e14_bits[:, None] >> shifts) & jnp.int32(1)) != 0
            is_diag = s_ti == s_tj
            geom_ok = jnp.where(is_diag, upper_tri, True)
            valid = (
                v_tp
                & mi[:, None]
                & mj[None, :]
                & geom_ok
                & (True if include_pme_excl_erf else (~is_full))
            )

            dx = pi[:, None, 0] - pj[None, :, 0]
            dy = pi[:, None, 1] - pj[None, :, 1]
            dz = pi[:, None, 2] - pj[None, :, 2]
            if has_box:
                ax, by, cz, bx_t, cx_t, cy_t = bx_1[0], bx_1[1], bx_1[2], bx_1[3], bx_1[4], bx_1[5]
                sz = jnp.where(cz > zero, jnp.floor(dz / jnp.where(cz > zero, cz, one) + half), zero)
                dx = dx - sz * cx_t
                dy = dy - sz * cy_t
                dz = dz - sz * cz
                sy = jnp.where(by > zero, jnp.floor(dy / jnp.where(by > zero, by, one) + half), zero)
                dx = dx - sy * bx_t
                dy = dy - sy * by
                sx = jnp.where(ax > zero, jnp.floor(dx / jnp.where(ax > zero, ax, one) + half), zero)
                dx = dx - sx * ax

            raw_r2 = dx * dx + dy * dy + dz * dz
            act = valid & (raw_r2 < cut_sq)
            r2 = raw_r2 + jnp.asarray(1e-12, dtype=dtype)
            dist = jnp.sqrt(r2)
            inv_d = jnp.where(act, one / dist, zero)

            w_v = jnp.where(is_full, zero, jnp.where(is_14, s14_v, one))
            w_e = jnp.where(is_full, zero, jnp.where(is_14, s14_e, one))

            sig_i_m = sig_1[oi][:, None]
            sig_j_m = sig_1[oj][None, :]
            if is_geometric:
                sig_prod = sig_i_m * sig_j_m
                sig_pos = sig_prod > zero
                sig_ij = jnp.where(
                    sig_pos,
                    jnp.sqrt(jnp.where(sig_pos, sig_prod, one)),
                    zero,
                )
            else:
                sig_ij = half * (sig_i_m + sig_j_m)

            eps_prod = eps_1[oi][:, None] * eps_1[oj][None, :]
            eps_pos = eps_prod > zero
            eps_ij = jnp.where(
                eps_pos,
                jnp.sqrt(jnp.where(eps_pos, eps_prod, one)),
                zero,
            )

            s2 = sig_ij * sig_ij
            s6 = s2 * s2 * s2
            r2_lj = jnp.where(act, r2, one)
            r6 = r2_lj * r2_lj * r2_lj
            denom_lj = jnp.maximum(
                soft_term * s6 + r6,
                jnp.asarray(1e-6, dtype=dtype) * s6 + jnp.asarray(1e-30, dtype=dtype),
            )
            x_inv6 = jnp.where(act, s6 / denom_lj, zero)
            s_w, _ = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
            e_lj = w_v * four * eps_ij * lam_1 * (x_inv6 * x_inv6 - x_inv6) * s_w

            qq = jnp.asarray(coulomb_constant, dtype=dtype) * q_1[oi][:, None] * q_1[oj][None, :]
            if pme_alpha > 0.0:
                ar = jnp.asarray(pme_alpha, dtype=dtype) * dist
                if include_pme_excl_erf:
                    e_coul = qq * (w_e - erf(ar)) * inv_d
                else:
                    e_coul = qq * w_e * erfc(ar) * inv_d
            elif has_rf:
                k_rf_c = jnp.asarray(k_rf_f, dtype=dtype)
                c_rf_c = jnp.asarray(c_rf_f, dtype=dtype)
                e_coul = jnp.where(
                    dist >= jnp.asarray(1e-4, dtype=dtype),
                    qq * w_e * (inv_d + k_rf_c * r2 - c_rf_c),
                    zero,
                )
            else:
                e_coul = jnp.where(dist >= jnp.asarray(1e-4, dtype=dtype), qq * w_e * inv_d, zero)
            return jnp.sum(jnp.where(act, e_lj + e_coul, zero))

        p_indices = jnp.arange(M_pairs, dtype=jnp.int32)
        return jnp.sum(jax.vmap(_pair_energy)(p_indices, tp_1, ef_1, e14_1))

    return jax.vmap(_one_replica)(jnp.arange(B, dtype=jnp.int32))


def _nonbonded_tile_pairs_ragged_reference_det_forces(  # noqa: PLR0913
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_pairs: jax.Array,
    excl_full_mask: jax.Array,
    excl_14_mask: jax.Array,
    box6: jax.Array,
    lam_arr: jax.Array,
    scale_14_vdw: jax.Array,
    scale_14_elec: jax.Array,
    *,
    has_box: bool,
    cutoff: float,
    pme_alpha: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
    lj_combining_rule: str = "lorentz_berthelot",
    rf_dielectric: float | None = None,
    n_valid_per_replica: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX ``(hi_i32, lo_i32)`` ``2^20`` deterministic force reference for ``export_context()``."""
    rule = _normalize_lj_combining_rule(lj_combining_rule)
    is_geometric = rule == "geometric"
    has_rf = (pme_alpha <= 0.0) and (rf_dielectric is not None)
    k_rf_f, c_rf_f = (
        _reaction_field_k_c(cutoff, rf_dielectric)
        if has_rf and rf_dielectric is not None
        else (0.0, 0.0)
    )

    orig_dtype = positions.dtype
    B, N, _ = positions.shape
    M_pairs = int(tile_pairs.shape[1])
    if B == 0 or N == 0 or M_pairs == 0:
        return jnp.zeros((B,), dtype=orig_dtype), jnp.zeros((B, N, 3), dtype=orig_dtype)

    dtype = jnp.float32
    pos_f = positions.astype(dtype)
    q_f = charges.astype(dtype)
    sig_f = sigmas.astype(dtype)
    eps_f = epsilons.astype(dtype)
    bx_f = box6.astype(dtype)
    lam_f = lam_arr.astype(dtype)
    s14v_f = scale_14_vdw.astype(dtype)
    s14e_f = scale_14_elec.astype(dtype)

    n_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    n_pad = n_blocks * _NB_BLOCK_N
    pad_n = n_pad - N
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    four = jnp.asarray(4.0, dtype=dtype)

    def _one_rep_det(b_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        pos_1 = pos_f[b_idx]
        q_1 = q_f[b_idx if q_f.shape[0] > 1 else 0]
        sig_1 = sig_f[b_idx if sig_f.shape[0] > 1 else 0]
        eps_1 = eps_f[b_idx if eps_f.shape[0] > 1 else 0]
        msk_1 = atom_mask[b_idx if atom_mask.shape[0] > 1 else 0].astype(jnp.bool_)
        tp_1 = tile_pairs[b_idx if tile_pairs.shape[0] > 1 else 0].astype(jnp.int32)
        ef_1 = excl_full_mask[b_idx if excl_full_mask.shape[0] > 1 else 0].astype(jnp.int32)
        e14_1 = excl_14_mask[b_idx if excl_14_mask.shape[0] > 1 else 0].astype(jnp.int32)
        bx_1 = bx_f[b_idx if bx_f.shape[0] > 1 else 0]
        lam_1 = lam_f[b_idx if lam_f.shape[0] > 1 else 0]
        s14_v = s14v_f[b_idx if s14v_f.shape[0] > 1 else 0]
        s14_e = s14e_f[b_idx if s14e_f.shape[0] > 1 else 0]
        if n_valid_per_replica is not None:
            n_val_1 = n_valid_per_replica[
                b_idx if n_valid_per_replica.shape[0] > 1 else 0
            ].astype(jnp.int32)
        else:
            n_val_1 = jnp.int32(M_pairs)

        if has_box:
            pos_1 = _wrap_pos_box6_1(pos_1, bx_1)

        if pad_n > 0:
            pos_1 = jnp.pad(pos_1, ((0, pad_n), (0, 0)))
            q_1 = jnp.pad(q_1, (0, pad_n))
            sig_1 = jnp.pad(sig_1, (0, pad_n), constant_values=1.0)
            eps_1 = jnp.pad(eps_1, (0, pad_n))
            msk_1 = jnp.pad(msk_1, (0, pad_n), constant_values=False)

        lane = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
        upper_tri = lane[:, None] < lane[None, :]
        shifts = lane[None, :]
        oml = one - lam_1
        oml_pow = one
        for _ in range(int(softcore_exponent)):
            oml_pow = oml_pow * oml
        soft_term = jnp.asarray(softcore_alpha, dtype=dtype) * oml_pow
        cut_sq = jnp.asarray(cutoff * cutoff, dtype=dtype)
        coul_c = jnp.asarray(coulomb_constant, dtype=dtype)
        alpha_c = jnp.asarray(pme_alpha, dtype=dtype)
        two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

        def _pair_det(p_idx, tp_entry, ef_bits, e14_bits):
            ti, tj = tp_entry[0], tp_entry[1]
            v_tp = (
                (p_idx < n_val_1)
                & (ti >= 0)
                & (tj >= 0)
                & (ti < n_blocks)
                & (tj < n_blocks)
            )
            s_ti = jnp.clip(ti, 0, n_blocks - 1)
            s_tj = jnp.clip(tj, 0, n_blocks - 1)
            oi = s_ti * _NB_BLOCK_N + lane
            oj = s_tj * _NB_BLOCK_N + lane

            pi = pos_1[oi]
            pj = pos_1[oj]
            mi = v_tp & msk_1[oi] & (oi < N)
            mj = v_tp & msk_1[oj] & (oj < N)
            is_full = ((ef_bits[:, None] >> shifts) & jnp.int32(1)) != 0
            is_14 = ((e14_bits[:, None] >> shifts) & jnp.int32(1)) != 0
            is_diag = s_ti == s_tj
            geom_ok = jnp.where(is_diag, upper_tri, True)
            valid = (
                mi[:, None]
                & mj[None, :]
                & geom_ok
                & (True if include_pme_excl_erf else (~is_full))
            )

            dx = pi[:, None, 0] - pj[None, :, 0]
            dy = pi[:, None, 1] - pj[None, :, 1]
            dz = pi[:, None, 2] - pj[None, :, 2]
            if has_box:
                ax, by, cz, bx_t, cx_t, cy_t = bx_1[0], bx_1[1], bx_1[2], bx_1[3], bx_1[4], bx_1[5]
                sz = jnp.where(cz > zero, jnp.floor(dz / jnp.where(cz > zero, cz, one) + half), zero)
                dx = dx - sz * cx_t
                dy = dy - sz * cy_t
                dz = dz - sz * cz
                sy = jnp.where(by > zero, jnp.floor(dy / jnp.where(by > zero, by, one) + half), zero)
                dx = dx - sy * bx_t
                dy = dy - sy * by
                sx = jnp.where(ax > zero, jnp.floor(dx / jnp.where(ax > zero, ax, one) + half), zero)
                dx = dx - sx * ax

            raw_r2 = dx * dx + dy * dy + dz * dz
            act = valid & (raw_r2 < cut_sq)
            r2 = raw_r2 + jnp.asarray(1e-12, dtype=dtype)
            dist = jnp.sqrt(r2)
            inv_d = jnp.where(act, one / dist, zero)
            inv_r2 = inv_d * inv_d

            w_v = jnp.where(is_full, zero, jnp.where(is_14, s14_v, one))
            w_e = jnp.where(is_full, zero, jnp.where(is_14, s14_e, one))

            sig_i_m = sig_1[oi][:, None]
            sig_j_m = sig_1[oj][None, :]
            if is_geometric:
                sig_prod = sig_i_m * sig_j_m
                sig_pos = sig_prod > zero
                sig_ij = jnp.where(
                    sig_pos,
                    jnp.sqrt(jnp.where(sig_pos, sig_prod, one)),
                    zero,
                )
            else:
                sig_ij = half * (sig_i_m + sig_j_m)

            eps_prod = eps_1[oi][:, None] * eps_1[oj][None, :]
            eps_pos = eps_prod > zero
            eps_ij = jnp.where(
                eps_pos,
                jnp.sqrt(jnp.where(eps_pos, eps_prod, one)),
                zero,
            )

            s2 = sig_ij * sig_ij
            s6 = s2 * s2 * s2
            r2_lj = jnp.where(act, r2, one)
            r6 = r2_lj * r2_lj * r2_lj
            denom_lj = jnp.maximum(
                soft_term * s6 + r6,
                jnp.asarray(1e-6, dtype=dtype) * s6 + jnp.asarray(1e-30, dtype=dtype),
            )
            x_inv6 = jnp.where(act, s6 / denom_lj, zero)
            u_rat = jnp.where(act, r6 / denom_lj, zero)
            v_unsw = four * eps_ij * lam_1 * (x_inv6 * x_inv6 - x_inv6)
            dv_unsw_dr = (
                jnp.asarray(-24.0, dtype=dtype)
                * eps_ij
                * lam_1
                * (two * x_inv6 - one)
                * x_inv6
                * u_rat
                * inv_d
            )
            s_w, ds_dr = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
            e_lj = w_v * v_unsw * s_w
            de_lj_dr = w_v * (dv_unsw_dr * s_w + v_unsw * ds_dr)

            qq = coul_c * q_1[oi][:, None] * q_1[oj][None, :]
            if pme_alpha > 0.0:
                ar = alpha_c * dist
                exp_term = jnp.exp(zero - (ar * ar))
                if include_pme_excl_erf:
                    erf_v = _triton_erf(ar)
                    eff_w = w_e - erf_v
                else:
                    erfc_v = _triton_erfc(ar)
                    eff_w = w_e * erfc_v
                e_coul = qq * eff_w * inv_d
                de_coul_dr = qq * (
                    (zero - eff_w) * inv_r2
                    - (w_e if not include_pme_excl_erf else one)
                    * (alpha_c * two_over_sqrt_pi * exp_term)
                    * inv_d
                )
            elif has_rf:
                vc = dist >= jnp.asarray(1e-4, dtype=dtype)
                k_rf_c = jnp.asarray(k_rf_f, dtype=dtype)
                two_k_rf_c = jnp.asarray(2.0 * k_rf_f, dtype=dtype)
                c_rf_c = jnp.asarray(c_rf_f, dtype=dtype)
                e_coul = jnp.where(vc, qq * w_e * (inv_d + k_rf_c * r2 - c_rf_c), zero)
                de_coul_dr = jnp.where(vc, qq * w_e * (two_k_rf_c * dist - inv_r2), zero)
            else:
                vc = dist >= jnp.asarray(1e-4, dtype=dtype)
                e_coul = jnp.where(vc, qq * w_e * inv_d, zero)
                de_coul_dr = jnp.where(vc, (zero - qq) * w_e * inv_r2, zero)

            has_act = jnp.max(act.astype(jnp.int32)) > 0
            e_sum = jnp.where(has_act, jnp.sum(jnp.where(act, e_lj + e_coul, zero)), zero)
            f_sc = jnp.where(act, (zero - (de_lj_dr + de_coul_dr)) * inv_d, zero)
            fxm = f_sc * dx
            fym = f_sc * dy
            fzm = f_sc * dz
            fx_i = jnp.sum(fxm, axis=1)
            fy_i = jnp.sum(fym, axis=1)
            fz_i = jnp.sum(fzm, axis=1)
            fx_j = zero - jnp.sum(fxm, axis=0)
            fy_j = zero - jnp.sum(fym, axis=0)
            fz_j = zero - jnp.sum(fzm, axis=0)

            mi_eff = mi & has_act
            mj_eff = mj & has_act
            hxi, lxi = _split_f32_to_hi_lo_i32(fx_i, mi_eff)
            hyi, lyi = _split_f32_to_hi_lo_i32(fy_i, mi_eff)
            hzi, lzi = _split_f32_to_hi_lo_i32(fz_i, mi_eff)
            hxj, lxj = _split_f32_to_hi_lo_i32(fx_j, mj_eff)
            hyj, lyj = _split_f32_to_hi_lo_i32(fy_j, mj_eff)
            hzj, lzj = _split_f32_to_hi_lo_i32(fz_j, mj_eff)

            hxi_u = jnp.where(is_diag, hxi + hxj, hxi)
            lxi_u = jnp.where(is_diag, lxi + lxj, lxi)
            hyi_u = jnp.where(is_diag, hyi + hyj, hyi)
            lyi_u = jnp.where(is_diag, lyi + lyj, lyi)
            hzi_u = jnp.where(is_diag, hzi + hzj, hzi)
            lzi_u = jnp.where(is_diag, lzi + lzj, lzi)
            hxj_u = jnp.where(is_diag, jnp.int32(0), hxj)
            lxj_u = jnp.where(is_diag, jnp.int32(0), lxj)
            hyj_u = jnp.where(is_diag, jnp.int32(0), hyj)
            lyj_u = jnp.where(is_diag, jnp.int32(0), lyj)
            hzj_u = jnp.where(is_diag, jnp.int32(0), hzj)
            lzj_u = jnp.where(is_diag, jnp.int32(0), lzj)
            return (
                e_sum,
                oi,
                oj,
                hxi_u,
                lxi_u,
                hyi_u,
                lyi_u,
                hzi_u,
                lzi_u,
                hxj_u,
                lxj_u,
                hyj_u,
                lyj_u,
                hzj_u,
                lzj_u,
            )

        p_indices = jnp.arange(M_pairs, dtype=jnp.int32)
        (
            e_all,
            oi_all,
            oj_all,
            hxi_a,
            lxi_a,
            hyi_a,
            lyi_a,
            hzi_a,
            lzi_a,
            hxj_a,
            lxj_a,
            hyj_a,
            lyj_a,
            hzj_a,
            lzj_a,
        ) = jax.vmap(_pair_det)(p_indices, tp_1, ef_1, e14_1)

        oi_flat = oi_all.reshape(-1)
        oj_flat = oj_all.reshape(-1)
        zi_pad = jnp.zeros((n_pad,), dtype=jnp.int32)
        fx_hi = zi_pad.at[oi_flat].add(hxi_a.reshape(-1)).at[oj_flat].add(hxj_a.reshape(-1))
        fx_lo = zi_pad.at[oi_flat].add(lxi_a.reshape(-1)).at[oj_flat].add(lxj_a.reshape(-1))
        fy_hi = zi_pad.at[oi_flat].add(hyi_a.reshape(-1)).at[oj_flat].add(hyj_a.reshape(-1))
        fy_lo = zi_pad.at[oi_flat].add(lyi_a.reshape(-1)).at[oj_flat].add(lyj_a.reshape(-1))
        fz_hi = zi_pad.at[oi_flat].add(hzi_a.reshape(-1)).at[oj_flat].add(hzj_a.reshape(-1))
        fz_lo = zi_pad.at[oi_flat].add(lzi_a.reshape(-1)).at[oj_flat].add(lzj_a.reshape(-1))

        fx_rec = _reconstruct_hi_lo_i32_to_f32(fx_hi, fx_lo)[:N]
        fy_rec = _reconstruct_hi_lo_i32_to_f32(fy_hi, fy_lo)[:N]
        fz_rec = _reconstruct_hi_lo_i32_to_f32(fz_hi, fz_lo)[:N]
        f_rep = jnp.stack([fx_rec, fy_rec, fz_rec], axis=-1).astype(orig_dtype)
        return jnp.sum(e_all).astype(orig_dtype), f_rep

    return jax.vmap(_one_rep_det)(jnp.arange(B, dtype=jnp.int32))


def _pallas_nonbonded_tile_pairs_ragged_core(  # noqa: PLR0913
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_pairs: jax.Array,
    excl_full_mask: jax.Array,
    excl_14_mask: jax.Array,
    box6: jax.Array,
    lam_arr: jax.Array,
    scale_14_vdw: jax.Array,
    scale_14_elec: jax.Array,
    pair_queue: RaggedWorkQueue,
    *,
    has_box: bool,
    cutoff: float,
    pme_alpha: float,
    coulomb_constant: float,
    lj_switch_width: float,
    softcore_alpha: float,
    softcore_exponent: int,
    include_pme_excl_erf: bool,
    compute_grad_q: bool,
    interpret: bool,
    deterministic_forces: bool = False,
    lj_combining_rule: str = "lorentz_berthelot",
    rf_dielectric: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """1D ragged CTA work-queue Pallas kernel for 32x32 symmetric tile-pair nonbonded."""
    rule = _normalize_lj_combining_rule(lj_combining_rule)
    is_geometric = rule == "geometric"
    det_f = bool(deterministic_forces)
    has_rf = (pme_alpha <= 0.0) and (rf_dielectric is not None)
    k_rf_f, c_rf_f = (
        _reaction_field_k_c(cutoff, rf_dielectric)
        if has_rf and rf_dielectric is not None
        else (0.0, 0.0)
    )

    B, N, _ = positions.shape
    M_pairs = int(tile_pairs.shape[1])
    orig_dtype = positions.dtype

    if B == 0 or N == 0 or M_pairs == 0 or pair_queue.total_blocks == 0:
        return (
            jnp.zeros((B,), dtype=orig_dtype),
            jnp.zeros((B, N, 3), dtype=orig_dtype),
            jnp.zeros((B, N), dtype=charges.dtype),
        )

    dtype = jnp.float32 if (det_f or (not interpret and orig_dtype == jnp.float64)) else orig_dtype
    pos_b = positions.astype(dtype)
    q_b = charges.astype(dtype)
    sig_b = sigmas.astype(dtype)
    eps_b = epsilons.astype(dtype)
    bx_b = box6.astype(dtype)
    lam_b = lam_arr.astype(dtype)
    s14v_b = scale_14_vdw.astype(dtype)
    s14e_b = scale_14_elec.astype(dtype)

    n_blocks = max(1, (N + _NB_BLOCK_N - 1) // _NB_BLOCK_N)
    N_pad = n_blocks * _NB_BLOCK_N
    pad_n = N_pad - N

    if has_box:
        pos_wrapped = _wrap_pos_box6_batch(pos_b, bx_b)
    else:
        pos_wrapped = pos_b

    B_pos = int(pos_wrapped.shape[0])
    B_q = int(q_b.shape[0])
    B_sig = int(sig_b.shape[0])
    B_eps = int(eps_b.shape[0])
    B_mask = int(atom_mask.shape[0])
    B_tp = int(tile_pairs.shape[0])
    B_ef = int(excl_full_mask.shape[0])
    B_e14 = int(excl_14_mask.shape[0])
    B_box = int(bx_b.shape[0])
    B_lam = int(lam_b.shape[0])
    B_s14v = int(s14v_b.shape[0])
    B_s14e = int(s14e_b.shape[0])

    if pad_n > 0:
        pos_pad = jnp.pad(pos_wrapped, ((0, 0), (0, pad_n), (0, 0)))
        q_pad = jnp.pad(q_b, ((0, 0), (0, pad_n)))
        sig_pad = jnp.pad(sig_b, ((0, 0), (0, pad_n)), constant_values=1.0)
        eps_pad = jnp.pad(eps_b, ((0, 0), (0, pad_n)))
        mask_pad = jnp.pad(atom_mask.astype(jnp.int32), ((0, 0), (0, pad_n)), constant_values=0)
    else:
        pos_pad = pos_wrapped
        q_pad = q_b
        sig_pad = sig_b
        eps_pad = eps_b
        mask_pad = atom_mask.astype(jnp.int32)

    tp_group = pair_queue.tile_size
    n_pair_blocks_max = (M_pairs + tp_group - 1) // tp_group
    M_pairs_pad = n_pair_blocks_max * tp_group
    pad_tp = M_pairs_pad - M_pairs
    if pad_tp > 0:
        tp_pad = jnp.pad(tile_pairs.astype(jnp.int32), ((0, 0), (0, pad_tp), (0, 0)), constant_values=-1)
        ef_pad = jnp.pad(excl_full_mask.astype(jnp.int32), ((0, 0), (0, pad_tp), (0, 0)), constant_values=0)
        e14_pad = jnp.pad(excl_14_mask.astype(jnp.int32), ((0, 0), (0, pad_tp), (0, 0)), constant_values=0)
    else:
        tp_pad = tile_pairs.astype(jnp.int32)
        ef_pad = excl_full_mask.astype(jnp.int32)
        e14_pad = excl_14_mask.astype(jnp.int32)

    pos_x = jnp.asarray(pos_pad[:, :, 0].reshape(B_pos * N_pad))
    pos_y = jnp.asarray(pos_pad[:, :, 1].reshape(B_pos * N_pad))
    pos_z = jnp.asarray(pos_pad[:, :, 2].reshape(B_pos * N_pad))
    q_flat = jnp.asarray(q_pad.reshape(B_q * N_pad))
    sig_flat = jnp.asarray(sig_pad.reshape(B_sig * N_pad))
    eps_flat = jnp.asarray(eps_pad.reshape(B_eps * N_pad))
    mask_flat = jnp.asarray(mask_pad.reshape(B_mask * N_pad))
    tp_i_flat = jnp.asarray(tp_pad[:, :, 0].reshape(B_tp * M_pairs_pad))
    tp_j_flat = jnp.asarray(tp_pad[:, :, 1].reshape(B_tp * M_pairs_pad))
    ef_flat = jnp.asarray(ef_pad.reshape(B_ef * M_pairs_pad * _NB_BLOCK_N))
    e14_flat = jnp.asarray(e14_pad.reshape(B_e14 * M_pairs_pad * _NB_BLOCK_N))
    box_flat = jnp.asarray(bx_b.reshape(B_box * 6))
    lam_flat = jnp.asarray(lam_b.reshape(B_lam))
    s14v_flat = jnp.asarray(s14v_b.reshape(B_s14v))
    s14e_flat = jnp.asarray(s14e_b.reshape(B_s14e))

    z_e = jnp.zeros((B,), dtype=dtype)

    def _tp_ragged_kernel(  # noqa: PLR0913
        px_ref,
        py_ref,
        pz_ref,
        q_ref,
        sig_ref,
        eps_ref,
        mask_ref,
        tp_i_ref,
        tp_j_ref,
        ef_ref,
        e14_ref,
        box_ref,
        lam_ref,
        s14v_ref,
        s14e_ref,
        cta_rep_ref,
        cta_blk_ref,
        n_valid_ref,
        *mut_and_out_refs,
    ):
        n_mut = len(mut_and_out_refs) // 2
        out_refs = mut_and_out_refs[n_mut:]
        e_out_ref = out_refs[0]

        if det_f:
            if compute_grad_q:
                (
                    fx_hi_ref,
                    fy_hi_ref,
                    fz_hi_ref,
                    fx_lo_ref,
                    fy_lo_ref,
                    fz_lo_ref,
                    gq_hi_ref,
                    gq_lo_ref,
                ) = out_refs[1:]
            else:
                (
                    fx_hi_ref,
                    fy_hi_ref,
                    fz_hi_ref,
                    fx_lo_ref,
                    fy_lo_ref,
                    fz_lo_ref,
                ) = out_refs[1:]
                gq_hi_ref = None
                gq_lo_ref = None
            fx_out_ref = None
            fy_out_ref = None
            fz_out_ref = None
            gq_out_ref = None
        else:
            if compute_grad_q:
                fx_out_ref, fy_out_ref, fz_out_ref, gq_out_ref = out_refs[1:]
            else:
                fx_out_ref, fy_out_ref, fz_out_ref = out_refs[1:]
                gq_out_ref = None
            fx_hi_ref = None
            fy_hi_ref = None
            fz_hi_ref = None
            fx_lo_ref = None
            fy_lo_ref = None
            fz_lo_ref = None
            gq_hi_ref = None
            gq_lo_ref = None

        cta_id = pl.program_id(0)
        b_id = _pl_load(cta_rep_ref, (jnp.full((1,), cta_id, dtype=jnp.int32),))[0]
        local_pid = _pl_load(cta_blk_ref, (jnp.full((1,), cta_id, dtype=jnp.int32),))[0]
        n_valid_b = _pl_load(n_valid_ref, (jnp.full((1,), b_id, dtype=jnp.int32),))[0]

        base_pos = (b_id * N_pad) if B_pos > 1 else jnp.int32(0)
        base_q = (b_id * N_pad) if B_q > 1 else jnp.int32(0)
        base_sig = (b_id * N_pad) if B_sig > 1 else jnp.int32(0)
        base_eps = (b_id * N_pad) if B_eps > 1 else jnp.int32(0)
        base_mask = (b_id * N_pad) if B_mask > 1 else jnp.int32(0)
        base_tp = (b_id * M_pairs_pad) if B_tp > 1 else jnp.int32(0)
        base_ef = (b_id * (M_pairs_pad * _NB_BLOCK_N)) if B_ef > 1 else jnp.int32(0)
        base_e14 = (b_id * (M_pairs_pad * _NB_BLOCK_N)) if B_e14 > 1 else jnp.int32(0)
        base_box = (b_id * 6) if B_box > 1 else jnp.int32(0)
        base_lam = b_id if B_lam > 1 else jnp.int32(0)
        base_s14v = b_id if B_s14v > 1 else jnp.int32(0)
        base_s14e = b_id if B_s14e > 1 else jnp.int32(0)
        f_base = b_id * N_pad

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        half = jnp.asarray(0.5, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        four = jnp.asarray(4.0, dtype=dtype)

        ax = _pl_load(box_ref, (base_box + jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (base_box + jnp.ones((1,), dtype=jnp.int32),))[0]
        cz = _pl_load(box_ref, (base_box + jnp.full((1,), 2, dtype=jnp.int32),))[0]
        bx_t = _pl_load(box_ref, (base_box + jnp.full((1,), 3, dtype=jnp.int32),))[0]
        cx_t = _pl_load(box_ref, (base_box + jnp.full((1,), 4, dtype=jnp.int32),))[0]
        cy_t = _pl_load(box_ref, (base_box + jnp.full((1,), 5, dtype=jnp.int32),))[0]

        if has_box:
            ax_pos = ax > zero
            by_pos = by > zero
            cz_pos = cz > zero
            inv_ax = jnp.where(ax_pos, one / jnp.where(ax_pos, ax, one), zero)
            inv_by = jnp.where(by_pos, one / jnp.where(by_pos, by, one), zero)
            inv_cz = jnp.where(cz_pos, one / jnp.where(cz_pos, cz, one), zero)
        else:
            inv_ax = zero
            inv_by = zero
            inv_cz = zero

        lam = _pl_load(lam_ref, (base_lam + jnp.zeros((1,), dtype=jnp.int32),))[0]
        _oml_rq = one - lam
        _oml_rq_pow = one
        for _ in range(int(softcore_exponent)):
            _oml_rq_pow = _oml_rq_pow * _oml_rq
        soft_alpha_lj = jnp.asarray(softcore_alpha, dtype=dtype) * _oml_rq_pow
        s14_vdw = _pl_load(s14v_ref, (base_s14v + jnp.zeros((1,), dtype=jnp.int32),))[0]
        s14_elec = _pl_load(s14e_ref, (base_s14e + jnp.zeros((1,), dtype=jnp.int32),))[0]
        cutoff_val = jnp.asarray(cutoff, dtype=dtype)
        cutoff_sq = cutoff_val * cutoff_val
        coul_const = jnp.asarray(coulomb_constant, dtype=dtype)
        alpha_val = jnp.asarray(pme_alpha, dtype=dtype)
        two_over_sqrt_pi = jnp.asarray(1.1283791670955126, dtype=dtype)

        lane_32 = jnp.arange(_NB_BLOCK_N, dtype=jnp.int32)
        upper_tri_mask = lane_32[:, None] < lane_32[None, :]
        bit_shifts = lane_32[None, :]

        def _eval_one_pair(step_idx, e_acc):
            p_idx = local_pid * tp_group + step_idx
            ti = _pl_load(tp_i_ref, (base_tp + jnp.full((1,), p_idx, dtype=jnp.int32),))[0]
            tj = _pl_load(tp_j_ref, (base_tp + jnp.full((1,), p_idx, dtype=jnp.int32),))[0]
            valid_tp = (
                (p_idx < n_valid_b)
                & (p_idx < M_pairs)
                & (ti >= 0)
                & (tj >= 0)
                & (ti < n_blocks)
                & (tj < n_blocks)
            )

            def _do_tp(e_in):
                safe_ti = jnp.clip(ti, 0, n_blocks - 1)
                safe_tj = jnp.clip(tj, 0, n_blocks - 1)
                offs_i = safe_ti * _NB_BLOCK_N + lane_32
                offs_j = safe_tj * _NB_BLOCK_N + lane_32

                px_i = _pl_load(px_ref, (base_pos + offs_i,))
                py_i = _pl_load(py_ref, (base_pos + offs_i,))
                pz_i = _pl_load(pz_ref, (base_pos + offs_i,))
                mask_i = (_pl_load(mask_ref, (base_mask + offs_i,)) > 0) & (offs_i < N)

                px_j = _pl_load(px_ref, (base_pos + offs_j,))
                py_j = _pl_load(py_ref, (base_pos + offs_j,))
                pz_j = _pl_load(pz_ref, (base_pos + offs_j,))
                mask_j = (_pl_load(mask_ref, (base_mask + offs_j,)) > 0) & (offs_j < N)

                full_bits = _pl_load(ef_ref, (base_ef + p_idx * _NB_BLOCK_N + lane_32,))
                is_full_excl = ((full_bits[:, None] >> bit_shifts) & jnp.int32(1)) != 0

                is_diag = safe_ti == safe_tj
                pair_geom_valid = jnp.where(is_diag, upper_tri_mask, jnp.bool_(True))
                valid_pair = (
                    mask_i[:, None]
                    & mask_j[None, :]
                    & pair_geom_valid
                    & (True if include_pme_excl_erf else (~is_full_excl))
                )

                raw_dx = px_i[:, None] - px_j[None, :]
                raw_dy = py_i[:, None] - py_j[None, :]
                raw_dz = pz_i[:, None] - pz_j[None, :]
                if has_box:
                    sz = jnp.floor(raw_dz * inv_cz + half)
                    dx_z = raw_dx - sz * cx_t
                    dy_z = raw_dy - sz * cy_t
                    dz = raw_dz - sz * cz
                    sy = jnp.floor(dy_z * inv_by + half)
                    dx_y = dx_z - sy * bx_t
                    dy = dy_z - sy * by
                    sx = jnp.floor(dx_y * inv_ax + half)
                    dx = dx_y - sx * ax
                else:
                    dx, dy, dz = raw_dx, raw_dy, raw_dz

                raw_r_sq = dx * dx + dy * dy + dz * dz
                active = valid_pair & (raw_r_sq < cutoff_sq)

                def _eval_active_tp(e_act_in):
                    q_i = _pl_load(q_ref, (base_q + offs_i,))
                    sig_i = _pl_load(sig_ref, (base_sig + offs_i,))
                    eps_i = _pl_load(eps_ref, (base_eps + offs_i,))
                    q_j = _pl_load(q_ref, (base_q + offs_j,))
                    sig_j = _pl_load(sig_ref, (base_sig + offs_j,))
                    eps_j = _pl_load(eps_ref, (base_eps + offs_j,))
                    bits_14 = _pl_load(e14_ref, (base_e14 + p_idx * _NB_BLOCK_N + lane_32,))
                    is_14_excl = ((bits_14[:, None] >> bit_shifts) & jnp.int32(1)) != 0

                    r_sq = raw_r_sq + jnp.asarray(1e-12, dtype=dtype)
                    dist = jnp.sqrt(r_sq)
                    inv_dist = jnp.where(active, one / dist, zero)
                    inv_r_sq = inv_dist * inv_dist

                    w_vdw = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_vdw, one))
                    w_elec = jnp.where(is_full_excl, zero, jnp.where(is_14_excl, s14_elec, one))

                    if is_geometric:
                        sig_prod = sig_i[:, None] * sig_j[None, :]
                        sig_pos = sig_prod > zero
                        sig_ij = jnp.where(
                            sig_pos,
                            jnp.sqrt(jnp.where(sig_pos, sig_prod, one)),
                            zero,
                        )
                    else:
                        sig_ij = half * (sig_i[:, None] + sig_j[None, :])

                    eps_prod = eps_i[:, None] * eps_j[None, :]
                    eps_pos = eps_prod > zero
                    eps_ij = jnp.where(
                        eps_pos,
                        jnp.sqrt(jnp.where(eps_pos, eps_prod, one)),
                        zero,
                    )

                    s2_rq = sig_ij * sig_ij
                    s6_rq = s2_rq * s2_rq * s2_rq
                    r2_lj = jnp.where(active, r_sq, one)
                    r6_rq = r2_lj * r2_lj * r2_lj
                    denom_lj = jnp.maximum(
                        soft_alpha_lj * s6_rq + r6_rq,
                        jnp.asarray(1e-6, dtype=dtype) * s6_rq + jnp.asarray(1e-30, dtype=dtype),
                    )
                    x_inv6 = jnp.where(active, s6_rq / denom_lj, zero)
                    u_rat = jnp.where(active, r6_rq / denom_lj, zero)
                    v_unsw = four * eps_ij * lam * (x_inv6 * x_inv6 - x_inv6)
                    dv_unsw_dr = (
                        jnp.asarray(-24.0, dtype=dtype)
                        * eps_ij
                        * lam
                        * (two * x_inv6 - one)
                        * x_inv6
                        * u_rat
                        * inv_dist
                    )

                    s_w, ds_dr = _lj_switch_weight_and_deriv(dist, cutoff, lj_switch_width)
                    e_lj = w_vdw * v_unsw * s_w
                    de_lj_dr = w_vdw * (dv_unsw_dr * s_w + v_unsw * ds_dr)

                    qq = coul_const * q_i[:, None] * q_j[None, :]
                    if pme_alpha > 0.0:
                        alpha_r = alpha_val * dist
                        exp_term = jnp.exp(zero - (alpha_r * alpha_r))
                        if include_pme_excl_erf:
                            erf_v = erf(alpha_r) if (interpret and dtype == jnp.float64) else _triton_erf(alpha_r)
                            eff_w = w_elec - erf_v
                        else:
                            erfc_v = erfc(alpha_r) if (interpret and dtype == jnp.float64) else _triton_erfc(alpha_r)
                            eff_w = w_elec * erfc_v
                        e_coul = qq * eff_w * inv_dist
                        de_coul_dr = qq * (
                            (zero - eff_w) * inv_r_sq
                            - (w_elec if not include_pme_excl_erf else one)
                            * (alpha_val * two_over_sqrt_pi * exp_term)
                            * inv_dist
                        )
                        gq_coeff = jnp.where(active, coul_const * eff_w * inv_dist, zero)
                    elif has_rf:
                        valid_coul = dist >= jnp.asarray(1e-4, dtype=dtype)
                        k_rf_c = jnp.asarray(k_rf_f, dtype=dtype)
                        two_k_rf_c = jnp.asarray(2.0 * k_rf_f, dtype=dtype)
                        c_rf_c = jnp.asarray(c_rf_f, dtype=dtype)
                        rf_kernel = inv_dist + k_rf_c * r_sq - c_rf_c
                        e_coul = jnp.where(valid_coul, qq * w_elec * rf_kernel, zero)
                        de_coul_dr = jnp.where(
                            valid_coul,
                            qq * w_elec * (two_k_rf_c * dist - inv_r_sq),
                            zero,
                        )
                        gq_coeff = jnp.where(active & valid_coul, coul_const * w_elec * rf_kernel, zero)
                    else:
                        valid_coul = dist >= jnp.asarray(1e-4, dtype=dtype)
                        e_coul = jnp.where(valid_coul, qq * w_elec * inv_dist, zero)
                        de_coul_dr = jnp.where(valid_coul, (zero - qq) * w_elec * inv_r_sq, zero)
                        gq_coeff = jnp.where(active & valid_coul, coul_const * w_elec * inv_dist, zero)

                    e_pair = jnp.where(active, e_lj + e_coul, zero)
                    f_scalar = jnp.where(active, (zero - (de_lj_dr + de_coul_dr)) * inv_dist, zero)

                    fx_mat = f_scalar * dx
                    fy_mat = f_scalar * dy
                    fz_mat = f_scalar * dz

                    fx_i = jnp.sum(fx_mat, axis=1)
                    fy_i = jnp.sum(fy_mat, axis=1)
                    fz_i = jnp.sum(fz_mat, axis=1)
                    fx_j = zero - jnp.sum(fx_mat, axis=0)
                    fy_j = zero - jnp.sum(fy_mat, axis=0)
                    fz_j = zero - jnp.sum(fz_mat, axis=0)

                    if compute_grad_q:
                        gq_i = jnp.sum(gq_coeff * q_j[None, :], axis=1)
                        gq_j = jnp.sum(gq_coeff * q_i[:, None], axis=0)
                    else:
                        gq_i = zero
                        gq_j = zero

                    if det_f:
                        hxi, lxi = _split_f32_to_hi_lo_i32(fx_i, mask_i)
                        hyi, lyi = _split_f32_to_hi_lo_i32(fy_i, mask_i)
                        hzi, lzi = _split_f32_to_hi_lo_i32(fz_i, mask_i)
                        hxj, lxj = _split_f32_to_hi_lo_i32(fx_j, mask_j)
                        hyj, lyj = _split_f32_to_hi_lo_i32(fy_j, mask_j)
                        hzj, lzj = _split_f32_to_hi_lo_i32(fz_j, mask_j)
                        if compute_grad_q:
                            hgqi, lgqi = _split_f32_to_hi_lo_i32(gq_i, mask_i)
                            hgqj, lgqj = _split_f32_to_hi_lo_i32(gq_j, mask_j)
                        else:
                            hgqi = hxi
                            lgqi = lxi
                            hgqj = hxj
                            lgqj = lxj

                        def _accum_diag_det(_):
                            _pl_atomic_add_i32_1d(fx_hi_ref, f_base + offs_i, hxi + hxj, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fy_hi_ref, f_base + offs_i, hyi + hyj, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fz_hi_ref, f_base + offs_i, hzi + hzj, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fx_lo_ref, f_base + offs_i, lxi + lxj, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fy_lo_ref, f_base + offs_i, lyi + lyj, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fz_lo_ref, f_base + offs_i, lzi + lzj, mask_i, interpret)
                            if compute_grad_q and gq_hi_ref is not None and gq_lo_ref is not None:
                                _pl_atomic_add_i32_1d(gq_hi_ref, f_base + offs_i, hgqi + hgqj, mask_i, interpret)
                                _pl_atomic_add_i32_1d(gq_lo_ref, f_base + offs_i, lgqi + lgqj, mask_i, interpret)

                        def _accum_offdiag_det(_):
                            _pl_atomic_add_i32_1d(fx_hi_ref, f_base + offs_i, hxi, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fy_hi_ref, f_base + offs_i, hyi, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fz_hi_ref, f_base + offs_i, hzi, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fx_lo_ref, f_base + offs_i, lxi, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fy_lo_ref, f_base + offs_i, lyi, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fz_lo_ref, f_base + offs_i, lzi, mask_i, interpret)
                            _pl_atomic_add_i32_1d(fx_hi_ref, f_base + offs_j, hxj, mask_j, interpret)
                            _pl_atomic_add_i32_1d(fy_hi_ref, f_base + offs_j, hyj, mask_j, interpret)
                            _pl_atomic_add_i32_1d(fz_hi_ref, f_base + offs_j, hzj, mask_j, interpret)
                            _pl_atomic_add_i32_1d(fx_lo_ref, f_base + offs_j, lxj, mask_j, interpret)
                            _pl_atomic_add_i32_1d(fy_lo_ref, f_base + offs_j, lyj, mask_j, interpret)
                            _pl_atomic_add_i32_1d(fz_lo_ref, f_base + offs_j, lzj, mask_j, interpret)
                            if compute_grad_q and gq_hi_ref is not None and gq_lo_ref is not None:
                                _pl_atomic_add_i32_1d(gq_hi_ref, f_base + offs_i, hgqi, mask_i, interpret)
                                _pl_atomic_add_i32_1d(gq_lo_ref, f_base + offs_i, lgqi, mask_i, interpret)
                                _pl_atomic_add_i32_1d(gq_hi_ref, f_base + offs_j, hgqj, mask_j, interpret)
                                _pl_atomic_add_i32_1d(gq_lo_ref, f_base + offs_j, lgqj, mask_j, interpret)

                        jax.lax.cond(is_diag, _accum_diag_det, _accum_offdiag_det, None)
                    else:
                        def _accum_diag(_):
                            _pl_atomic_add_1d(fx_out_ref, f_base + offs_i, fx_i + fx_j, mask_i, interpret)
                            _pl_atomic_add_1d(fy_out_ref, f_base + offs_i, fy_i + fy_j, mask_i, interpret)
                            _pl_atomic_add_1d(fz_out_ref, f_base + offs_i, fz_i + fz_j, mask_i, interpret)
                            if compute_grad_q and gq_out_ref is not None:
                                _pl_atomic_add_1d(gq_out_ref, f_base + offs_i, gq_i + gq_j, mask_i, interpret)

                        def _accum_offdiag(_):
                            _pl_atomic_add_1d(fx_out_ref, f_base + offs_i, fx_i, mask_i, interpret)
                            _pl_atomic_add_1d(fy_out_ref, f_base + offs_i, fy_i, mask_i, interpret)
                            _pl_atomic_add_1d(fz_out_ref, f_base + offs_i, fz_i, mask_i, interpret)
                            _pl_atomic_add_1d(fx_out_ref, f_base + offs_j, fx_j, mask_j, interpret)
                            _pl_atomic_add_1d(fy_out_ref, f_base + offs_j, fy_j, mask_j, interpret)
                            _pl_atomic_add_1d(fz_out_ref, f_base + offs_j, fz_j, mask_j, interpret)
                            if compute_grad_q and gq_out_ref is not None:
                                _pl_atomic_add_1d(gq_out_ref, f_base + offs_i, gq_i, mask_i, interpret)
                                _pl_atomic_add_1d(gq_out_ref, f_base + offs_j, gq_j, mask_j, interpret)

                        jax.lax.cond(is_diag, _accum_diag, _accum_offdiag, None)
                    return e_act_in + jnp.sum(e_pair)

                return jax.lax.cond(
                    jnp.max(active.astype(jnp.int32)) > 0,
                    _eval_active_tp,
                    lambda e_act_in: e_act_in,
                    e_in,
                )

            return jax.lax.cond(valid_tp, _do_tp, lambda e_in: e_in, e_acc)

        zero_e = (cta_id - cta_id).astype(dtype)
        e_cta = jax.lax.fori_loop(jnp.int32(0), jnp.int32(tp_group), _eval_one_pair, zero_e)
        _pl_atomic_add_1d(
            e_out_ref,
            jnp.full((1,), b_id, dtype=jnp.int32),
            e_cta[None],
            jnp.ones((1,), dtype=jnp.bool_),
            interpret,
        )

    any_spec = _any_block_spec()
    compiler_params = (
        resolve_pallas_compiler_params("megakernel", grid_blocks=pair_queue.total_blocks)
        if not interpret
        else None
    )
    read_only_inputs = (
        pos_x,
        pos_y,
        pos_z,
        q_flat,
        sig_flat,
        eps_flat,
        mask_flat,
        tp_i_flat,
        tp_j_flat,
        ef_flat,
        e14_flat,
        box_flat,
        lam_flat,
        s14v_flat,
        s14e_flat,
        pair_queue.cta_to_replica,
        pair_queue.cta_to_local_block,
        pair_queue.n_real_per_replica,
    )

    if det_f:
        z_fx_hi = jnp.zeros((B * N_pad,), dtype=jnp.int32)
        z_fy_hi = jnp.zeros((B * N_pad,), dtype=jnp.int32)
        z_fz_hi = jnp.zeros((B * N_pad,), dtype=jnp.int32)
        z_fx_lo = jnp.zeros((B * N_pad,), dtype=jnp.int32)
        z_fy_lo = jnp.zeros((B * N_pad,), dtype=jnp.int32)
        z_fz_lo = jnp.zeros((B * N_pad,), dtype=jnp.int32)
        if compute_grad_q:
            z_gq_hi = jnp.zeros((B * N_pad,), dtype=jnp.int32)
            z_gq_lo = jnp.zeros((B * N_pad,), dtype=jnp.int32)
            (
                e_vec,
                fx_hi_pad,
                fy_hi_pad,
                fz_hi_pad,
                fx_lo_pad,
                fy_lo_pad,
                fz_lo_pad,
                gq_hi_pad,
                gq_lo_pad,
            ) = pl.pallas_call(
                _tp_ragged_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((B,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                ],
                in_specs=[any_spec] * 27,
                out_specs=[any_spec] * 9,
                grid=(pair_queue.total_blocks,),
                input_output_aliases={18 + k: k for k in range(9)},
                interpret=interpret,
                compiler_params=compiler_params,
                name="pallas_nonbonded_tile_pairs_ragged_1d_det",
            )(
                *read_only_inputs,
                z_e,
                z_fx_hi,
                z_fy_hi,
                z_fz_hi,
                z_fx_lo,
                z_fy_lo,
                z_fz_lo,
                z_gq_hi,
                z_gq_lo,
            )
            gq_pad = _reconstruct_hi_lo_i32_to_f32(gq_hi_pad, gq_lo_pad)
            gq_out = gq_pad.reshape((B, N_pad))[:, :N].astype(charges.dtype)
        else:
            (
                e_vec,
                fx_hi_pad,
                fy_hi_pad,
                fz_hi_pad,
                fx_lo_pad,
                fy_lo_pad,
                fz_lo_pad,
            ) = pl.pallas_call(
                _tp_ragged_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((B,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                    jax.ShapeDtypeStruct((B * N_pad,), jnp.int32),
                ],
                in_specs=[any_spec] * 25,
                out_specs=[any_spec] * 7,
                grid=(pair_queue.total_blocks,),
                input_output_aliases={18 + k: k for k in range(7)},
                interpret=interpret,
                compiler_params=compiler_params,
                name="pallas_nonbonded_tile_pairs_ragged_1d_det_nogq",
            )(
                *read_only_inputs,
                z_e,
                z_fx_hi,
                z_fy_hi,
                z_fz_hi,
                z_fx_lo,
                z_fy_lo,
                z_fz_lo,
            )
            gq_out = jnp.zeros((B, N), dtype=charges.dtype)

        fx_pad = _reconstruct_hi_lo_i32_to_f32(fx_hi_pad, fx_lo_pad)
        fy_pad = _reconstruct_hi_lo_i32_to_f32(fy_hi_pad, fy_lo_pad)
        fz_pad = _reconstruct_hi_lo_i32_to_f32(fz_hi_pad, fz_lo_pad)
    else:
        z_fx = jnp.zeros((B * N_pad,), dtype=dtype)
        z_fy = jnp.zeros((B * N_pad,), dtype=dtype)
        z_fz = jnp.zeros((B * N_pad,), dtype=dtype)
        if compute_grad_q:
            z_gq = jnp.zeros((B * N_pad,), dtype=dtype)
            e_vec, fx_pad, fy_pad, fz_pad, gq_pad = pl.pallas_call(
                _tp_ragged_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((B,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                ],
                in_specs=[any_spec] * 23,
                out_specs=[any_spec] * 5,
                grid=(pair_queue.total_blocks,),
                input_output_aliases={18: 0, 19: 1, 20: 2, 21: 3, 22: 4},
                interpret=interpret,
                compiler_params=compiler_params,
                name="pallas_nonbonded_tile_pairs_ragged_1d",
            )(
                *read_only_inputs,
                z_e,
                z_fx,
                z_fy,
                z_fz,
                z_gq,
            )
            gq_out = gq_pad.reshape((B, N_pad))[:, :N].astype(charges.dtype)
        else:
            e_vec, fx_pad, fy_pad, fz_pad = pl.pallas_call(
                _tp_ragged_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((B,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                    jax.ShapeDtypeStruct((B * N_pad,), dtype),
                ],
                in_specs=[any_spec] * 22,
                out_specs=[any_spec] * 4,
                grid=(pair_queue.total_blocks,),
                input_output_aliases={18: 0, 19: 1, 20: 2, 21: 3},
                interpret=interpret,
                compiler_params=compiler_params,
                name="pallas_nonbonded_tile_pairs_ragged_1d_nogq",
            )(
                *read_only_inputs,
                z_e,
                z_fx,
                z_fy,
                z_fz,
            )
            gq_out = jnp.zeros((B, N), dtype=charges.dtype)

    f_out = jnp.stack(
        [
            fx_pad.reshape((B, N_pad))[:, :N],
            fy_pad.reshape((B, N_pad))[:, :N],
            fz_pad.reshape((B, N_pad))[:, :N],
        ],
        axis=-1,
    ).astype(orig_dtype)
    return e_vec.astype(orig_dtype), f_out, gq_out


def pallas_nonbonded_tile_pairs_ragged(  # noqa: PLR0913
    positions: jax.Array,
    charges: jax.Array,
    sigmas: jax.Array,
    epsilons: jax.Array,
    atom_mask: jax.Array,
    tile_pairs: jax.Array,
    excl_full_mask: jax.Array,
    excl_14_mask: jax.Array,
    *,
    num_valid_pairs: Sequence[int] | np.ndarray | jax.Array | None = None,
    pair_queue: RaggedWorkQueue | None = None,
    box: jax.Array | None = None,
    cutoff: float = 9.0,
    pme_alpha: float = 0.0,
    rf_dielectric: float | None = None,
    coulomb_constant: float = COULOMB_CONSTANT,
    lj_combining_rule: str = "lorentz_berthelot",
    lj_switch_width: float = 0.0,
    soft_core_lambda: float | jax.Array = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    scale_14_vdw: float | jax.Array = 0.5,
    scale_14_elec: float | jax.Array = 0.8333333333333334,
    include_pme_excl_erf: bool = False,
    tp_group: int | None = None,
    deterministic_forces: bool = False,
    return_forces: bool = False,
    interpret: bool | None = None,
    is_triclinic: bool = False,
) -> Any:
    """Evaluate 32x32 symmetric tile-pair nonbonded over a 1D ragged CTA work queue."""
    rule_norm = _normalize_lj_combining_rule(lj_combining_rule)
    if (pme_alpha <= 0.0) and (rf_dielectric is not None):
        _reaction_field_k_c(cutoff, rf_dielectric)
    det_f = bool(deterministic_forces)

    single_input = positions.ndim == 2
    pos_b = positions[None, ...] if single_input else positions
    B, N, _ = pos_b.shape
    M_pairs = int(tile_pairs.shape[-2])

    q_b = charges[None, ...] if charges.ndim == 1 else charges
    sig_b = sigmas[None, ...] if sigmas.ndim == 1 else sigmas
    eps_b = epsilons[None, ...] if epsilons.ndim == 1 else epsilons
    m_b = atom_mask[None, ...] if atom_mask.ndim == 1 else atom_mask
    tp_b = tile_pairs[None, ...] if tile_pairs.ndim == 2 else tile_pairs
    ef_b = excl_full_mask[None, ...] if excl_full_mask.ndim == 2 else excl_full_mask
    e14_b = excl_14_mask[None, ...] if excl_14_mask.ndim == 2 else excl_14_mask

    eff_tri = (
        _infer_pme_is_triclinic(box, B, single_input=single_input, is_triclinic=is_triclinic)
        if box is not None
        else False
    )
    box6, has_box = _normalize_box6_batch(
        box, B, pos_b.dtype, single_input=single_input, is_triclinic=eff_tri
    )
    lam_arr = jnp.asarray(soft_core_lambda, dtype=pos_b.dtype).reshape(-1)
    s14v_arr = jnp.asarray(scale_14_vdw, dtype=pos_b.dtype).reshape(-1)
    s14e_arr = jnp.asarray(scale_14_elec, dtype=pos_b.dtype).reshape(-1)

    if pair_queue is not None:
        n_valid_ref_arr: jax.Array | None = jnp.asarray(
            pair_queue.n_real_per_replica, dtype=jnp.int32
        ).reshape(-1)
    elif num_valid_pairs is not None:
        n_valid_ref_arr = jnp.asarray(num_valid_pairs, dtype=jnp.int32).reshape(-1)
    else:
        n_valid_ref_arr = None

    if in_export_context():
        if return_forces:
            if det_f:
                e_vec, f_vec = _nonbonded_tile_pairs_ragged_reference_det_forces(
                    pos_b,
                    q_b,
                    sig_b,
                    eps_b,
                    m_b,
                    tp_b,
                    ef_b,
                    e14_b,
                    box6,
                    lam_arr,
                    s14v_arr,
                    s14e_arr,
                    has_box=has_box,
                    cutoff=float(cutoff),
                    pme_alpha=float(pme_alpha),
                    coulomb_constant=float(coulomb_constant),
                    lj_switch_width=float(lj_switch_width),
                    softcore_alpha=float(softcore_alpha),
                    softcore_exponent=int(softcore_exponent),
                    include_pme_excl_erf=bool(include_pme_excl_erf),
                    lj_combining_rule=rule_norm,
                    rf_dielectric=rf_dielectric,
                    n_valid_per_replica=n_valid_ref_arr,
                )
                return (e_vec[0], f_vec[0]) if single_input else (e_vec, f_vec)

            def _sum_ref(p_in):
                ev = _nonbonded_tile_pairs_ragged_reference(
                    p_in,
                    q_b,
                    sig_b,
                    eps_b,
                    m_b,
                    tp_b,
                    ef_b,
                    e14_b,
                    box6,
                    lam_arr,
                    s14v_arr,
                    s14e_arr,
                    has_box=has_box,
                    cutoff=float(cutoff),
                    pme_alpha=float(pme_alpha),
                    coulomb_constant=float(coulomb_constant),
                    lj_switch_width=float(lj_switch_width),
                    softcore_alpha=float(softcore_alpha),
                    softcore_exponent=int(softcore_exponent),
                    include_pme_excl_erf=bool(include_pme_excl_erf),
                    lj_combining_rule=rule_norm,
                    rf_dielectric=rf_dielectric,
                    deterministic_forces=det_f,
                    n_valid_per_replica=n_valid_ref_arr,
                )
                return jnp.sum(ev), ev

            (_, e_vec), g_pos = jax.value_and_grad(_sum_ref, has_aux=True)(pos_b)
            f_vec = -g_pos
            return (e_vec[0], f_vec[0]) if single_input else (e_vec, f_vec)
        e_vec = _nonbonded_tile_pairs_ragged_reference(
            pos_b,
            q_b,
            sig_b,
            eps_b,
            m_b,
            tp_b,
            ef_b,
            e14_b,
            box6,
            lam_arr,
            s14v_arr,
            s14e_arr,
            has_box=has_box,
            cutoff=float(cutoff),
            pme_alpha=float(pme_alpha),
            coulomb_constant=float(coulomb_constant),
            lj_switch_width=float(lj_switch_width),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            include_pme_excl_erf=bool(include_pme_excl_erf),
            lj_combining_rule=rule_norm,
            rf_dielectric=rf_dielectric,
            deterministic_forces=det_f,
            n_valid_per_replica=n_valid_ref_arr,
        )
        return e_vec[0] if single_input else e_vec

    if pair_queue is None:
        if num_valid_pairs is None:
            num_valid_pairs = np.full((B,), M_pairs, dtype=np.int32)
        pair_queue = build_ragged_tile_pair_queue(
            num_valid_pairs,
            tp_group=tp_group,
            max_tile_pairs=M_pairs,
        )
    eff_n_valid = pair_queue.n_real_per_replica

    use_interpret = _resolve_interpret(interpret)
    if return_forces:
        e_vec, f_vec, _ = _pallas_nonbonded_tile_pairs_ragged_core(
            pos_b,
            q_b,
            sig_b,
            eps_b,
            m_b,
            tp_b,
            ef_b,
            e14_b,
            box6,
            lam_arr,
            s14v_arr,
            s14e_arr,
            pair_queue,
            has_box=has_box,
            cutoff=float(cutoff),
            pme_alpha=float(pme_alpha),
            coulomb_constant=float(coulomb_constant),
            lj_switch_width=float(lj_switch_width),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            include_pme_excl_erf=bool(include_pme_excl_erf),
            compute_grad_q=False,
            interpret=use_interpret,
            deterministic_forces=det_f,
            lj_combining_rule=rule_norm,
            rf_dielectric=rf_dielectric,
        )
        return (e_vec[0], f_vec[0]) if single_input else (e_vec, f_vec)

    @functools.partial(jax.custom_vjp, nondiff_argnums=())
    def _diff_fn(p_in, q_in, sig_in, eps_in):
        ev, _, _ = _pallas_nonbonded_tile_pairs_ragged_core(
            p_in,
            q_in,
            sig_in,
            eps_in,
            m_b,
            tp_b,
            ef_b,
            e14_b,
            box6,
            lam_arr,
            s14v_arr,
            s14e_arr,
            pair_queue,
            has_box=has_box,
            cutoff=float(cutoff),
            pme_alpha=float(pme_alpha),
            coulomb_constant=float(coulomb_constant),
            lj_switch_width=float(lj_switch_width),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            include_pme_excl_erf=bool(include_pme_excl_erf),
            compute_grad_q=False,
            interpret=use_interpret,
            deterministic_forces=det_f,
            lj_combining_rule=rule_norm,
            rf_dielectric=rf_dielectric,
        )
        return ev

    def _fwd(p_in, q_in, sig_in, eps_in):
        pv = _unwrap_primal(p_in)
        qv = _unwrap_primal(q_in)
        sv = _unwrap_primal(sig_in)
        ev_p = _unwrap_primal(eps_in)
        q_p = _is_perturbed(q_in)
        s_p = _is_perturbed(sig_in)
        e_p = _is_perturbed(eps_in)
        e_out, f_out, gq_out = _pallas_nonbonded_tile_pairs_ragged_core(
            pv,
            qv,
            sv,
            ev_p,
            m_b,
            tp_b,
            ef_b,
            e14_b,
            box6,
            lam_arr,
            s14v_arr,
            s14e_arr,
            pair_queue,
            has_box=has_box,
            cutoff=float(cutoff),
            pme_alpha=float(pme_alpha),
            coulomb_constant=float(coulomb_constant),
            lj_switch_width=float(lj_switch_width),
            softcore_alpha=float(softcore_alpha),
            softcore_exponent=int(softcore_exponent),
            include_pme_excl_erf=bool(include_pme_excl_erf),
            compute_grad_q=q_p,
            interpret=use_interpret,
            deterministic_forces=det_f,
            lj_combining_rule=rule_norm,
            rf_dielectric=rf_dielectric,
        )
        return e_out, (pv, qv, sv, ev_p, f_out, gq_out, q_p, s_p, e_p)

    def _bwd(res, g_e):
        pv, qv, sv, ev_p, f_out, gq_out, q_p, s_p, e_p = res
        if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
            g_e, jax.custom_derivatives.SymbolicZero
        ):
            return (
                jnp.zeros_like(pv),
                jnp.zeros_like(qv),
                jnp.zeros_like(sv),
                jnp.zeros_like(ev_p),
            )
        g_vec = jnp.asarray(g_e, dtype=pv.dtype).reshape((B,))
        dp = (-f_out * g_vec[:, None, None]).astype(pv.dtype)
        if q_p:
            dq_full = gq_out.astype(pv.dtype) * g_vec[:, None]
            dq = (jnp.sum(dq_full, axis=0, keepdims=True) if qv.shape[0] == 1 and B > 1 else dq_full).astype(qv.dtype)
        else:
            dq = jnp.zeros_like(qv)

        if s_p or e_p:
            _, vjp_se = jax.vjp(
                lambda s_arg, e_arg: _nonbonded_tile_pairs_ragged_reference(
                    pv,
                    qv,
                    s_arg,
                    e_arg,
                    m_b,
                    tp_b,
                    ef_b,
                    e14_b,
                    box6,
                    lam_arr,
                    s14v_arr,
                    s14e_arr,
                    has_box=has_box,
                    cutoff=float(cutoff),
                    pme_alpha=float(pme_alpha),
                    coulomb_constant=float(coulomb_constant),
                    lj_switch_width=float(lj_switch_width),
                    softcore_alpha=float(softcore_alpha),
                    softcore_exponent=int(softcore_exponent),
                    include_pme_excl_erf=bool(include_pme_excl_erf),
                    lj_combining_rule=rule_norm,
                    rf_dielectric=rf_dielectric,
                    deterministic_forces=det_f,
                    n_valid_per_replica=eff_n_valid,
                ),
                sv,
                ev_p,
            )
            ds, de = vjp_se(g_vec)
            ds = ds.astype(sv.dtype) if s_p else jnp.zeros_like(sv)
            de = de.astype(ev_p.dtype) if e_p else jnp.zeros_like(ev_p)
        else:
            ds = jnp.zeros_like(sv)
            de = jnp.zeros_like(ev_p)
        return dp, dq, ds, de

    _diff_fn.defvjp(_fwd, _bwd, symbolic_zeros=True)
    e_vec = _diff_fn(pos_b, q_b, sig_b, eps_b)
    return e_vec[0] if single_input else e_vec


__all__ = [
    "PMETierGroup",
    "PMETierMode",
    "PMETierPlan",
    "RaggedMortonTilePairs",
    "RaggedWorkQueue",
    "build_ragged_morton_tile_pairs",
    "build_ragged_tile_pair_queue",
    "build_ragged_work_queue",
    "gather_pallas_ragged",
    "pallas_nonbonded_tile_pairs_ragged",
    "plan_pme_tiers",
    "select_adaptive_tp_group",
    "spme_energy_and_forces_ragged_tiered",
    "spme_energy_ragged_tiered",
    "spme_forces_ragged_tiered",
    "spread_charges_pallas_ragged",
]
