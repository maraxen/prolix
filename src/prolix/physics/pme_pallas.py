"""Vectorized Pallas SPME B-spline spread/gather (ADR-006, R4.1).

Opt-in kernels. Default production path remains the XLA stencil in ``pme.py``.
Do not autodiff through ``atomic_add``; pair with the existing custom VJP.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from prolix.export import resolve_pallas_compiler_params

BLOCK_N = 32


def _bspline4_vec(u: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Order-4 cardinal B-spline weights for vector ``u`` of shape ``(B,)``."""
    dtype = u.dtype
    c1 = jnp.asarray(1.0, dtype=dtype)
    c3 = jnp.asarray(3.0, dtype=dtype)
    c4 = jnp.asarray(4.0, dtype=dtype)
    c6 = jnp.asarray(6.0, dtype=dtype)
    inv6 = jnp.asarray(1.0 / 6.0, dtype=dtype)
    u2 = u * u
    u3 = u2 * u
    w0 = (c1 - c3 * u + c3 * u2 - u3) * inv6
    w1 = (c4 - c6 * u2 + c3 * u3) * inv6
    w2 = (c1 + c3 * u + c3 * u2 - c3 * u3) * inv6
    w3 = u3 * inv6
    return (w0, w1, w2, w3)


def _bspline4_deriv_vec(u: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Order-4 cardinal B-spline derivatives for vector ``u`` of shape ``(B,)``."""
    dtype = u.dtype
    cm3 = jnp.asarray(-3.0, dtype=dtype)
    cm12 = jnp.asarray(-12.0, dtype=dtype)
    c3 = jnp.asarray(3.0, dtype=dtype)
    c6 = jnp.asarray(6.0, dtype=dtype)
    c9 = jnp.asarray(9.0, dtype=dtype)
    inv6 = jnp.asarray(1.0 / 6.0, dtype=dtype)
    u2 = u * u
    dw0 = (cm3 + c6 * u - c3 * u2) * inv6
    dw1 = (cm12 * u + c9 * u2) * inv6
    dw2 = (c3 + c6 * u - c9 * u2) * inv6
    dw3 = (c3 * u2) * inv6
    return (dw0, dw1, dw2, dw3)


def _select_stencil(
    w_tuple: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    offset: jax.Array,
) -> jax.Array:
    """Select B-spline weight per stencil offset ``(1, 64)`` -> ``(BLOCK_N, 64)`` without ``jnp.stack``."""
    w0, w1, w2, w3 = w_tuple
    return (
        w0[:, None] * (offset == 0)
        + w1[:, None] * (offset == 1)
        + w2[:, None] * (offset == 2)
        + w3[:, None] * (offset == 3)
    )


def compute_pme_spatial_perm(
    positions: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    atom_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """MK-53: Compute spatial 3D grid-cell permutation ``(perm, inv_perm)`` for SPME locality.

    Sorting atoms by their coarse 3D PME grid cell before ``spread_charges_pallas`` /
    ``gather_pallas`` groups spatially adjacent atoms into the same 32-atom Pallas CTA,
    maximizing L2 cache hit rate on the ``64^3`` PME mesh and pushing ghost atoms (`~atom_mask`)
    to the trailing CTAs where they exit in O(1).
    """
    pos_single = positions[0] if positions.ndim == 3 else positions
    box_arr = jnp.asarray(box_size, dtype=pos_single.dtype)
    if box_arr.ndim == 3:
        box_1 = jnp.diag(box_arr[0])
    elif box_arr.ndim == 2:
        box_1 = jnp.diag(box_arr) if box_arr.shape == (3, 3) else box_arr[0, :3]
    else:
        box_1 = box_arr[:3]
    Kx, Ky, Kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    k_vec = jnp.asarray([Kx, Ky, Kz], dtype=pos_single.dtype)
    safe_box = jnp.where(box_1 > 0, box_1, jnp.ones_like(box_1))
    frac = jnp.mod(pos_single, safe_box[None, :]) * (k_vec / safe_box)[None, :]
    g_idx = jnp.clip(
        jnp.floor(frac).astype(jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.asarray([Kx - 1, Ky - 1, Kz - 1], dtype=jnp.int32),
    )
    # Sort by 4x4x4 sub-grid tiles so each 32-atom CTA lands in a compact L2-resident sub-volume
    cell_key = (
        (g_idx[:, 0] >> 2) * ((Ky // 4 + 1) * (Kz // 4 + 1))
        + (g_idx[:, 1] >> 2) * (Kz // 4 + 1)
        + (g_idx[:, 2] >> 2)
    ).astype(jnp.int32)
    if atom_mask is not None:
        m_single = atom_mask[0] if atom_mask.ndim == 2 else atom_mask
        sentinel = jnp.int32(Kx * Ky * Kz + 1)
        cell_key = jnp.where(jnp.asarray(m_single, dtype=jnp.bool_), cell_key, sentinel)
    perm = jnp.argsort(cell_key).astype(jnp.int32)
    inv_perm = jnp.argsort(perm).astype(jnp.int32)
    return perm, inv_perm


def _spread_charges_pallas_batch(
    B: int,
    in_batched: tuple[bool, bool, bool, bool],
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
) -> jax.Array:
    b_pos, b_q, b_m, b_box = in_batched
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    mesh_stride = Kx * Ky * Kz
    n_atoms = int(positions.shape[1] if b_pos else positions.shape[0])
    n_blocks = max(1, (n_atoms + BLOCK_N - 1) // BLOCK_N)
    n_pad = n_blocks * BLOCK_N
    pad = n_pad - n_atoms

    orig_dtype = positions.dtype
    dtype = jnp.float32
    pos_in = positions.astype(dtype)
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    box_in = jnp.asarray(box_size, dtype=dtype)

    if not b_pos:
        pos_in = pos_in[None, ...]
    if not b_q:
        q_in = q_in[None, ...]
    if not b_m:
        mask_in = mask_in[None, ...]
    if not b_box:
        box_in = box_in[None, ...]

    if pad > 0:
        pos_in = jnp.pad(pos_in, ((0, 0), (0, pad), (0, 0)), constant_values=0.0)
        q_in = jnp.pad(q_in, ((0, 0), (0, pad)), constant_values=0.0)
        mask_in = jnp.pad(mask_in, ((0, 0), (0, pad)), constant_values=0)

    pos_x = jnp.asarray(pos_in[:, :, 0], dtype=dtype).reshape(-1)
    pos_y = jnp.asarray(pos_in[:, :, 1], dtype=dtype).reshape(-1)
    pos_z = jnp.asarray(pos_in[:, :, 2], dtype=dtype).reshape(-1)
    q_flat = q_in.reshape(-1)
    mask_flat = mask_in.reshape(-1)
    box_flat = box_in.reshape(-1)
    mesh0_flat = jnp.zeros((B * mesh_stride,), dtype=dtype)

    def _spread_kernel(px_ref, py_ref, pz_ref, charges_ref, mask_ref, box_ref, mesh_in_ref, mesh_out_ref):
        del mesh_in_ref
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        offs = pid * BLOCK_N + jnp.arange(BLOCK_N, dtype=jnp.int32)

        pos_base = (b_id if b_pos else 0) * n_pad
        q_base = (b_id if b_q else 0) * n_pad
        m_base = (b_id if b_m else 0) * n_pad
        box_base = (b_id if b_box else 0) * 3
        mesh_base = b_id * mesh_stride

        zero = jnp.asarray(0.0, dtype=dtype)
        is_real = (_pl_load(mask_ref, (m_base + offs,)) > 0) & (offs < n_atoms)
        qi = _pl_load(charges_ref, (q_base + offs,), mask=is_real, other=zero)
        active_q = is_real & (jnp.abs(qi) > zero)

        def _do_spread(_):
            px = _pl_load(px_ref, (pos_base + offs,), mask=active_q, other=zero)
            py = _pl_load(py_ref, (pos_base + offs,), mask=active_q, other=zero)
            pz = _pl_load(pz_ref, (pos_base + offs,), mask=active_q, other=zero)

            bx = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
            by = _pl_load(box_ref, (box_base + jnp.ones((1,), dtype=jnp.int32),))[0]
            bz = _pl_load(box_ref, (box_base + jnp.full((1,), 2, dtype=jnp.int32),))[0]
            kx = jnp.asarray(Kx, dtype=dtype)
            ky = jnp.asarray(Ky, dtype=dtype)
            kz = jnp.asarray(Kz, dtype=dtype)

            frac_x = px * (kx / bx)
            frac_y = py * (ky / by)
            frac_z = pz * (kz / bz)
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

            # MK-49 / MK-53: Separable (BLOCK_N=32, 16) YZ-plane stencil reused across 4 X-planes
            s_yz = jnp.arange(16, dtype=jnp.int32)
            oy = (s_yz // 4)[None, :]
            oz = (s_yz % 4)[None, :]

            iy_16 = jnp.mod(gyi[:, None] + oy - 1, Ky)
            iz_16 = jnp.mod(gzi[:, None] + oz - 1, Kz)
            iyz_16 = mesh_base + iy_16 * Kz + iz_16

            wy_16 = _select_stencil(wy, oy)
            wz_16 = _select_stencil(wz, oz)
            q_wyz_16 = jnp.where(active_q[:, None], qi[:, None] * (wy_16 * wz_16), zero)
            mask_16 = jnp.broadcast_to(active_q[:, None], (BLOCK_N, 16)).reshape(BLOCK_N * 16)
            ky_kz = Ky * Kz

            for ox_i in range(4):
                ix_plane = jnp.mod(gxi + (ox_i - 1), Kx)[:, None] * ky_kz
                flat_idx_16 = (ix_plane + iyz_16).reshape(BLOCK_N * 16)
                w_16 = (wx[ox_i][:, None] * q_wyz_16).astype(dtype).reshape(BLOCK_N * 16)
                if interpret:
                    def _serial_atomic(k, _, idx_vec=flat_idx_16, w_vec=w_16):
                        plt.atomic_add(mesh_out_ref, (idx_vec[k],), w_vec[k])

                    jax.lax.fori_loop(0, BLOCK_N * 16, _serial_atomic, None)
                else:
                    plt.atomic_add(mesh_out_ref, (flat_idx_16,), w_16, mask=mask_16)

        jax.lax.cond(
            jnp.max(active_q.astype(jnp.int32)) > 0,
            _do_spread,
            lambda _: None,
            None,
        )

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    mesh_flat = pl.pallas_call(
        _spread_kernel,
        out_shape=jax.ShapeDtypeStruct((B * mesh_stride,), dtype),
        in_specs=[any_spec] * 7,
        out_specs=any_spec,
        grid=(B, n_blocks),
        interpret=interpret,
        compiler_params=resolve_pallas_compiler_params("pme", grid_blocks=B * n_blocks)
        if not interpret
        else None,
        input_output_aliases={6: 0},
        name="pme_charge_spread",
    )(pos_x, pos_y, pos_z, q_flat, mask_flat, box_flat, mesh0_flat)
    return mesh_flat.reshape((B, Kx, Ky, Kz)).astype(orig_dtype)


def _spread_charges_pallas_single(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
) -> jax.Array:
    out = _spread_charges_pallas_batch(
        1,
        (False, False, False, False),
        positions,
        charges,
        atom_mask,
        box_size,
        grid_dims,
        interpret,
    )
    return out[0]


def spread_charges_pallas(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    atom_perm: jax.Array | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Spread charges onto the PME mesh with a vectorized ``(BLOCK_N=32, 64)`` stencil."""
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    if atom_perm is None and os.environ.get("PROLIX_PME_SPATIAL_SORT", "").strip() == "1":
        atom_perm, _ = compute_pme_spatial_perm(
            positions, box_size, grid_tuple, atom_mask=atom_mask
        )
    if atom_perm is not None:
        perm = jnp.asarray(atom_perm, dtype=jnp.int32)
        if positions.ndim == 3:
            positions = positions[:, perm, :]
            if charges.ndim == 2:
                charges = charges[:, perm]
            else:
                charges = charges[perm]
            if atom_mask.ndim == 2:
                atom_mask = atom_mask[:, perm]
            else:
                atom_mask = atom_mask[perm]
        else:
            positions = positions[perm]
            charges = charges[perm]
            atom_mask = atom_mask[perm]

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, mask, box):
        return _spread_charges_pallas_single(pos, q, mask, box, grid_tuple, interpret)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, mask, box):
        out = _spread_charges_pallas_batch(
            int(axis_size),
            tuple(bool(b) for b in in_batched),
            pos,
            q,
            mask,
            box,
            grid_tuple,
            interpret,
        )
        return out, True

    if positions.ndim == 3:
        B = positions.shape[0]
        b_q = charges.ndim == 2
        b_m = atom_mask.ndim == 2
        b_box = jnp.asarray(box_size).ndim >= 2
        return _spread_charges_pallas_batch(
            int(B),
            (True, b_q, b_m, b_box),
            positions,
            charges,
            atom_mask,
            box_size,
            grid_tuple,
            interpret,
        )

    return _vmappable(positions, charges, atom_mask, box_size)


def _gather_pallas_batch(
    B: int,
    in_batched: tuple[bool, bool, bool, bool, bool],
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
    compute_dq: bool = True,
) -> tuple[jax.Array, jax.Array]:
    b_pos, b_q, b_m, b_box, b_mesh = in_batched
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    mesh_stride = Kx * Ky * Kz
    n_atoms = int(positions.shape[1] if b_pos else positions.shape[0])
    n_blocks = max(1, (n_atoms + BLOCK_N - 1) // BLOCK_N)
    n_pad = n_blocks * BLOCK_N
    pad = n_pad - n_atoms

    orig_dtype = positions.dtype
    dtype = jnp.float32
    pos_in = positions.astype(dtype)
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    box_in = jnp.asarray(box_size, dtype=dtype)
    theta_in = jnp.asarray(theta_norm, dtype=dtype)

    if not b_pos:
        pos_in = pos_in[None, ...]
    if not b_q:
        q_in = q_in[None, ...]
    if not b_m:
        mask_in = mask_in[None, ...]
    if not b_box:
        box_in = box_in[None, ...]
    if not b_mesh:
        theta_in = theta_in[None, ...]

    if pad > 0:
        pos_in = jnp.pad(pos_in, ((0, 0), (0, pad), (0, 0)), constant_values=0.0)
        q_in = jnp.pad(q_in, ((0, 0), (0, pad)), constant_values=0.0)
        mask_in = jnp.pad(mask_in, ((0, 0), (0, pad)), constant_values=0)

    pos_x = jnp.asarray(pos_in[:, :, 0], dtype=dtype).reshape(-1)
    pos_y = jnp.asarray(pos_in[:, :, 1], dtype=dtype).reshape(-1)
    pos_z = jnp.asarray(pos_in[:, :, 2], dtype=dtype).reshape(-1)
    q_flat = q_in.reshape(-1)
    mask_flat = mask_in.reshape(-1)
    box_flat = box_in.reshape(-1)
    theta_flat = theta_in.reshape(-1)

    def _gather_kernel(
        px_ref,
        py_ref,
        pz_ref,
        charges_ref,
        mask_ref,
        box_ref,
        theta_flat_ref,
        fx_out_ref,
        fy_out_ref,
        fz_out_ref,
        *dq_out_refs,
    ):
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        offs = pid * BLOCK_N + jnp.arange(BLOCK_N, dtype=jnp.int32)

        pos_base = (b_id if b_pos else 0) * n_pad
        q_base = (b_id if b_q else 0) * n_pad
        m_base = (b_id if b_m else 0) * n_pad
        box_base = (b_id if b_box else 0) * 3
        mesh_base = (b_id if b_mesh else 0) * mesh_stride
        out_base = b_id * n_pad

        zero = jnp.asarray(0.0, dtype=dtype)
        is_real = (_pl_load(mask_ref, (m_base + offs,)) > 0) & (offs < n_atoms)
        qi = _pl_load(charges_ref, (q_base + offs,), mask=is_real, other=zero)
        active_lane = is_real if compute_dq else (is_real & (jnp.abs(qi) > zero))
        zero_vec = (offs - offs).astype(dtype)

        def _do_gather(_):
            px = _pl_load(px_ref, (pos_base + offs,), mask=active_lane, other=zero)
            py = _pl_load(py_ref, (pos_base + offs,), mask=active_lane, other=zero)
            pz = _pl_load(pz_ref, (pos_base + offs,), mask=active_lane, other=zero)

            bx = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
            by = _pl_load(box_ref, (box_base + jnp.ones((1,), dtype=jnp.int32),))[0]
            bz = _pl_load(box_ref, (box_base + jnp.full((1,), 2, dtype=jnp.int32),))[0]
            kx = jnp.asarray(Kx, dtype=dtype)
            ky = jnp.asarray(Ky, dtype=dtype)
            kz = jnp.asarray(Kz, dtype=dtype)

            sx = kx / bx
            sy = ky / by
            sz = kz / bz
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

            # MK-49 / MK-53: Separable (BLOCK_N=32, 16) YZ-plane stencil reused across 4 X-planes
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

            mask_16 = jnp.broadcast_to(active_lane[:, None], (BLOCK_N, 16)).reshape(BLOCK_N * 16)
            ky_kz = Ky * Kz

            acc_dx = zero_vec
            acc_dy = zero_vec
            acc_dz = zero_vec
            acc_dq = zero_vec

            for ox_i in range(4):
                ix_plane = jnp.mod(gxi + (ox_i - 1), Kx)[:, None] * ky_kz
                flat_idx_16 = (ix_plane + iyz_16).reshape(BLOCK_N * 16)
                t_16 = _pl_load(
                    theta_flat_ref,
                    (flat_idx_16,),
                    mask=mask_16,
                    other=zero,
                ).reshape(BLOCK_N, 16)
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

            if compute_dq:
                dq_v = jnp.where(is_real, acc_dq, zero).astype(dtype)
            else:
                dq_v = zero_vec
            dpx_v = jnp.where(active_lane, qi * sx * acc_dx, zero).astype(dtype)
            dpy_v = jnp.where(active_lane, qi * sy * acc_dy, zero).astype(dtype)
            dpz_v = jnp.where(active_lane, qi * sz * acc_dz, zero).astype(dtype)
            return (dpx_v, dpy_v, dpz_v, dq_v)

        dpx, dpy, dpz, dq = jax.lax.cond(
            jnp.max(active_lane.astype(jnp.int32)) > 0,
            _do_gather,
            lambda z0: z0,
            (zero_vec, zero_vec, zero_vec, zero_vec),
        )

        _pl_store(fx_out_ref, (out_base + offs,), dpx)
        _pl_store(fy_out_ref, (out_base + offs,), dpy)
        _pl_store(fz_out_ref, (out_base + offs,), dpz)
        if compute_dq:
            _pl_store(dq_out_refs[0], (out_base + offs,), dq)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    if compute_dq:
        fx_pad, fy_pad, fz_pad, dq_pad = pl.pallas_call(
            _gather_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
            ],
            in_specs=[any_spec] * 7,
            out_specs=[any_spec] * 4,
            grid=(B, n_blocks),
            interpret=interpret,
            compiler_params=resolve_pallas_compiler_params("pme", grid_blocks=B * n_blocks)
            if not interpret
            else None,
            name="pme_bwd_gather",
        )(pos_x, pos_y, pos_z, q_flat, mask_flat, box_flat, theta_flat)
        dq_b = dq_pad.reshape((B, n_pad))[:, :n_atoms].astype(orig_dtype)
    else:
        fx_pad, fy_pad, fz_pad = pl.pallas_call(
            _gather_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
                jax.ShapeDtypeStruct((B * n_pad,), dtype),
            ],
            in_specs=[any_spec] * 7,
            out_specs=[any_spec] * 3,
            grid=(B, n_blocks),
            interpret=interpret,
            compiler_params=resolve_pallas_compiler_params("pme", grid_blocks=B * n_blocks)
            if not interpret
            else None,
            name="pme_bwd_gather",
        )(pos_x, pos_y, pos_z, q_flat, mask_flat, box_flat, theta_flat)
        dq_b = jnp.zeros((B, n_atoms), dtype=orig_dtype)
    fx_b = fx_pad.reshape((B, n_pad))[:, :n_atoms]
    fy_b = fy_pad.reshape((B, n_pad))[:, :n_atoms]
    fz_b = fz_pad.reshape((B, n_pad))[:, :n_atoms]
    dpos = jnp.stack([fx_b, fy_b, fz_b], axis=-1)
    return dpos.astype(orig_dtype), dq_b


def _gather_pallas_single(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
    compute_dq: bool = True,
) -> tuple[jax.Array, jax.Array]:
    dpos_b, dq_b = _gather_pallas_batch(
        1,
        (False, False, False, False, False),
        positions,
        charges,
        atom_mask,
        box_size,
        theta_norm,
        grid_dims,
        interpret,
        compute_dq=compute_dq,
    )
    return dpos_b[0], dq_b[0]


def gather_pallas(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    atom_perm: jax.Array | None = None,
    atom_inv_perm: jax.Array | None = None,
    interpret: bool = False,
    compute_dq: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Gather potential and dE/dr contributions from ``theta_norm``.

    Returns ``(forces, potentials)`` in the same convention as ``_spme_bwd``
    *before* multiplying by ``COULOMB_CONSTANT`` / ``g``.
    """
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    cdq = bool(compute_dq)

    inv_p = None
    if atom_perm is None and os.environ.get("PROLIX_PME_SPATIAL_SORT", "").strip() == "1":
        atom_perm, atom_inv_perm = compute_pme_spatial_perm(
            positions, box_size, grid_tuple, atom_mask=atom_mask
        )
    if atom_perm is not None:
        perm = jnp.asarray(atom_perm, dtype=jnp.int32)
        inv_p = (
            jnp.argsort(perm).astype(jnp.int32)
            if atom_inv_perm is None
            else jnp.asarray(atom_inv_perm, dtype=jnp.int32)
        )
        if positions.ndim == 3:
            positions = positions[:, perm, :]
            charges = charges[:, perm] if charges.ndim == 2 else charges[perm]
            atom_mask = atom_mask[:, perm] if atom_mask.ndim == 2 else atom_mask[perm]
        else:
            positions = positions[perm]
            charges = charges[perm]
            atom_mask = atom_mask[perm]

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, mask, box, dE_dmesh):
        return _gather_pallas_single(
            pos, q, mask, box, dE_dmesh, grid_tuple, interpret, compute_dq=cdq
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, mask, box, dE_dmesh):
        dpos_out, dq_out = _gather_pallas_batch(
            int(axis_size),
            tuple(bool(b) for b in in_batched),
            pos,
            q,
            mask,
            box,
            dE_dmesh,
            grid_tuple,
            interpret,
            compute_dq=cdq,
        )
        return (dpos_out, dq_out), (True, True)

    if positions.ndim == 3:
        B = int(positions.shape[0])
        b_q = charges.ndim == 2
        b_m = atom_mask.ndim == 2
        b_box = jnp.asarray(box_size).ndim >= 2
        b_mesh = theta_norm.ndim == 4
        dpos_s, dq_s = _gather_pallas_batch(
            B,
            (True, b_q, b_m, b_box, b_mesh),
            positions,
            charges,
            atom_mask,
            box_size,
            theta_norm,
            grid_tuple,
            interpret,
            compute_dq=cdq,
        )
        if inv_p is not None:
            return dpos_s[:, inv_p, :], dq_s[:, inv_p]
        return dpos_s, dq_s

    dpos_s, dq_s = _vmappable(positions, charges, atom_mask, box_size, theta_norm)
    if inv_p is not None:
        return dpos_s[inv_p], dq_s[inv_p]
    return dpos_s, dq_s


def make_spread_op(
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    context: object | None = None,
):
    """Build the SPME charge spread op for one mesh shape."""
    import functools

    import xtrax

    dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    def _spread_ref_one(pos_1, q_1, mask_1, box_1):
        dtype = pos_1.dtype
        K_arr = jnp.array(dims, dtype=dtype)
        box_arr = jnp.asarray(box_1, dtype=dtype)
        frac = (pos_1 / box_arr) * K_arr
        grid_idx = jnp.floor(frac).astype(jnp.int32)
        u = frac - grid_idx.astype(dtype)
        wx = _bspline4_vec(u[:, 0])
        wy = _bspline4_vec(u[:, 1])
        wz = _bspline4_vec(u[:, 2])
        s_idx = jnp.arange(64, dtype=jnp.int32)
        ox = (s_idx // 16)[None, :]
        oy = ((s_idx // 4) % 4)[None, :]
        oz = (s_idx % 4)[None, :]
        ix = (grid_idx[:, None, 0] + ox - 1) % dims[0]
        iy = (grid_idx[:, None, 1] + oy - 1) % dims[1]
        iz = (grid_idx[:, None, 2] + oz - 1) % dims[2]
        w = (_select_stencil(wx, ox) * _select_stencil(wy, oy) * _select_stencil(wz, oz)) * (
            q_1 * (mask_1 > 0).astype(q_1.dtype)
        ).astype(dtype)[:, None]
        return jnp.zeros(dims, dtype=dtype).at[ix, iy, iz].add(w)

    def _spread_ref(pos, q, mask, box):
        if pos.ndim == 3:
            B = pos.shape[0]
            b_q = q.ndim == 2
            b_m = mask.ndim == 2
            b_box = jnp.asarray(box).ndim >= 2
            return jax.vmap(
                lambda b: _spread_ref_one(
                    pos[b],
                    q[b] if b_q else q,
                    mask[b] if b_m else mask,
                    box[b] if b_box else box,
                )
            )(jnp.arange(B))
        return _spread_ref_one(pos, q, mask, box)

    return xtrax.op(
        _spread_ref,
        kernels={
            "gpu": functools.partial(spread_charges_pallas, grid_dims=dims, order=order, interpret=False),
            "cpu": functools.partial(spread_charges_pallas, grid_dims=dims, order=order, interpret=True),
        },
        name="pme_spread",
        context=context,
    )


def make_gather_op(
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    compute_dq: bool = True,
    context: object | None = None,
):
    """Build the SPME backward gather op for one mesh shape."""
    import functools

    import xtrax

    dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    cdq = bool(compute_dq)

    def _gather_ref_one(pos_1, q_1, mask_1, box_1, theta_1):
        dtype = pos_1.dtype
        K_arr = jnp.array(dims, dtype=dtype)
        box_arr = jnp.asarray(box_1, dtype=dtype)
        frac = (pos_1 / box_arr) * K_arr
        grid_idx = jnp.floor(frac).astype(jnp.int32)
        u = frac - grid_idx.astype(dtype)
        wx = _bspline4_vec(u[:, 0])
        wy = _bspline4_vec(u[:, 1])
        wz = _bspline4_vec(u[:, 2])
        dwx = _bspline4_deriv_vec(u[:, 0])
        dwy = _bspline4_deriv_vec(u[:, 1])
        dwz = _bspline4_deriv_vec(u[:, 2])
        s_idx = jnp.arange(64, dtype=jnp.int32)
        ox = (s_idx // 16)[None, :]
        oy = ((s_idx // 4) % 4)[None, :]
        oz = (s_idx % 4)[None, :]
        ix = (grid_idx[:, None, 0] + ox - 1) % dims[0]
        iy = (grid_idx[:, None, 1] + oy - 1) % dims[1]
        iz = (grid_idx[:, None, 2] + oz - 1) % dims[2]
        t_val = theta_1[ix, iy, iz]
        wx_s = _select_stencil(wx, ox)
        wy_s = _select_stencil(wy, oy)
        wz_s = _select_stencil(wz, oz)
        dwx_s = _select_stencil(dwx, ox)
        dwy_s = _select_stencil(dwy, oy)
        dwz_s = _select_stencil(dwz, oz)
        pot = (
            jnp.where(mask_1 > 0, jnp.sum(wx_s * wy_s * wz_s * t_val, axis=1), 0.0).astype(dtype)
            if cdq
            else jnp.zeros_like(q_1, dtype=dtype)
        )
        q_masked = (q_1 * (mask_1 > 0).astype(q_1.dtype)).astype(dtype)
        scale = K_arr / box_arr
        fx = q_masked * scale[0] * jnp.sum(dwx_s * wy_s * wz_s * t_val, axis=1)
        fy = q_masked * scale[1] * jnp.sum(wx_s * dwy_s * wz_s * t_val, axis=1)
        fz = q_masked * scale[2] * jnp.sum(wx_s * wy_s * dwz_s * t_val, axis=1)
        return jnp.stack([fx, fy, fz], axis=-1).astype(dtype), pot

    def _gather_ref(pos, q, mask, box, theta):
        if pos.ndim == 3:
            B = pos.shape[0]
            b_q = q.ndim == 2
            b_m = mask.ndim == 2
            b_box = jnp.asarray(box).ndim >= 2
            b_th = theta.ndim == 4
            return jax.vmap(
                lambda b: _gather_ref_one(
                    pos[b],
                    q[b] if b_q else q,
                    mask[b] if b_m else mask,
                    box[b] if b_box else box,
                    theta[b] if b_th else theta,
                )
            )(jnp.arange(B))
        return _gather_ref_one(pos, q, mask, box, theta)

    return xtrax.op(
        _gather_ref,
        kernels={
            "gpu": functools.partial(
                gather_pallas, grid_dims=dims, order=order, interpret=False, compute_dq=cdq
            ),
            "cpu": functools.partial(
                gather_pallas, grid_dims=dims, order=order, interpret=True, compute_dq=cdq
            ),
        },
        name="pme_gather",
        context=context,
    )

