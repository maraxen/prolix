"""Vectorized Pallas SPME B-spline spread/gather (ADR-006, R4.1).

Opt-in kernels. Default production path remains the XLA stencil in ``pme.py``.
Do not autodiff through ``atomic_add``; pair with the existing custom VJP.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
import xtrax
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from prolix.export import in_export_context, resolve_pallas_compiler_params
from prolix.physics.pallas_bonded import _triton_gpu_available
from prolix.physics.pbc import invert_box_lower_triangular

BLOCK_N = 32


def _pme_compiler_params(grid_blocks: int) -> plt.CompilerParams:
    """MK-72: Optimal BLOCK_N=32 Pallas SPME Triton launch parameters.

    Uses ``num_warps=2, num_stages=1`` by default on Ada/Hopper/Blackwell to avoid
    Hopper ``sm_90`` pointer-loop register doubling from ``num_stages=2`` while
    honoring explicit ``PROLIX_PALLAS_NUM_WARPS`` / ``PROLIX_PALLAS_NUM_STAGES``
    environment overrides.
    """
    env_w = os.environ.get("PROLIX_PALLAS_NUM_WARPS", "").strip()
    env_s = os.environ.get("PROLIX_PALLAS_NUM_STAGES", "").strip()
    if env_w or env_s:
        return resolve_pallas_compiler_params("pme", grid_blocks=grid_blocks)
    return plt.CompilerParams(num_warps=2, num_stages=1)


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


def _infer_is_triclinic_pme(
    box_size: jax.Array,
    B: int,
    is_triclinic: bool = False,
) -> bool:
    """Resolve ``is_triclinic`` while preserving ``B == 3`` ``(3, 3)`` orthorhombic disambiguation."""
    if is_triclinic:
        return True
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim >= 1 and int(box_arr.shape[-1]) == 9:
        return True
    if box_arr.ndim == 3 and box_arr.shape[-2:] == (3, 3):
        return True
    return bool(box_arr.ndim == 2 and box_arr.shape == (3, 3) and int(B) != 3)


def _extract_single_triclinic_box_mat(
    box_arr: jax.Array,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    """Extract a single ``(3, 3)`` lower-triangular box matrix from ``box_arr``."""
    b = jnp.asarray(box_arr, dtype=dtype)
    if b.ndim == 3 and b.shape[-2:] == (3, 3):
        return b[0]
    if b.ndim == 2 and b.shape == (3, 3):
        return b
    if b.ndim == 2 and b.shape[-1] == 9:
        return b[0, :9].reshape(3, 3)
    if b.ndim == 1 and b.shape[0] == 9:
        return b.reshape(3, 3)
    d = b[0, :3] if b.ndim >= 2 else b[:3]
    return jnp.diag(d)


def compute_pme_spatial_perm(
    positions: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    atom_mask: jax.Array | None = None,
    *,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """MK-53: Compute spatial 3D grid-cell permutation ``(perm, inv_perm)`` for SPME locality.

    Sorting atoms by their coarse 3D PME grid cell before ``spread_charges_pallas`` /
    ``gather_pallas`` groups spatially adjacent atoms into the same 32-atom Pallas CTA,
    maximizing L2 cache hit rate on the ``64^3`` PME mesh and pushing ghost atoms (`~atom_mask`)
    to the trailing CTAs where they exit in O(1).
    """
    B_in = int(positions.shape[0]) if positions.ndim == 3 else 1
    eff_tri = _infer_is_triclinic_pme(box_size, B_in, is_triclinic)
    pos_single = positions[0] if positions.ndim == 3 else positions
    box_arr = jnp.asarray(box_size, dtype=pos_single.dtype)
    Kx, Ky, Kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    k_vec = jnp.asarray([Kx, Ky, Kz], dtype=pos_single.dtype)
    if eff_tri:
        h_mat = _extract_single_triclinic_box_mat(box_arr, pos_single.dtype)
        b_inv = invert_box_lower_triangular(h_mat)
        s = jnp.matmul(pos_single, b_inv, precision=jax.lax.Precision.HIGHEST)
        frac = (s - jnp.floor(s)) * k_vec[None, :]
    else:
        if box_arr.ndim == 3:
            box_1 = jnp.diag(box_arr[0])
        elif box_arr.ndim == 2:
            box_1 = jnp.diag(box_arr) if (box_arr.shape == (3, 3) and B_in != 3) else box_arr[0, :3]
        else:
            box_1 = box_arr[:3]
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


def _resolve_pme_active_atoms(
    n_active_atoms: int | None,
    n_atoms: int,
    atom_mask: jax.Array | None = None,
) -> int:
    del atom_mask
    if n_active_atoms is not None:
        return min(n_atoms, max(1, int(n_active_atoms)))
    raw = os.environ.get("PROLIX_ACTIVE_ATOMS", "").strip()
    if raw:
        try:
            return min(n_atoms, max(1, int(raw)))
        except ValueError:
            pass
    return n_atoms


def _normalize_pme_box_batch(  # noqa: PLR0911
    box_size: jax.Array,
    b_box: bool,
    B: int,
    dtype: jax.typing.DTypeLike,
    *,
    is_triclinic: bool = False,
) -> jax.Array:
    """Normalize orthorhombic box to ``(1_or_B, 3)`` or triclinic box to row-major ``(1_or_B, 9)``."""
    box_arr = jnp.asarray(box_size, dtype=dtype)
    if is_triclinic:
        if box_arr.ndim == 3 and box_arr.shape[-2:] == (3, 3):
            flat = box_arr.reshape(box_arr.shape[0], 9)
            return jnp.broadcast_to(flat, (B, 9)) if b_box else flat[:1]
        if box_arr.ndim == 2 and box_arr.shape == (3, 3):
            flat1 = box_arr.reshape(1, 9)
            return jnp.broadcast_to(flat1, (B, 9)) if b_box else flat1
        if box_arr.ndim == 2 and box_arr.shape[-1] == 9:
            return jnp.broadcast_to(box_arr[:, :9], (B, 9)) if b_box else box_arr[:1, :9]
        if box_arr.ndim == 1 and box_arr.shape[0] == 9:
            flat1 = box_arr[None, :9]
            return jnp.broadcast_to(flat1, (B, 9)) if b_box else flat1
        if box_arr.ndim == 2:
            d_b = jnp.broadcast_to(box_arr[:, :3], (B, 3)) if b_box else box_arr[:1, :3]
            return jax.vmap(jnp.diag)(d_b).reshape(d_b.shape[0], 9)
        d_1 = jnp.diag(box_arr[:3]).reshape(1, 9)
        return jnp.broadcast_to(d_1, (B, 9)) if b_box else d_1

    if b_box:
        if box_arr.ndim == 3:
            out = jnp.diagonal(box_arr, axis1=-2, axis2=-1)
        elif box_arr.ndim == 1:
            out = jnp.broadcast_to(box_arr[None, :3], (B, 3))
        elif box_arr.shape == (3, 3):
            is_diag = (
                (box_arr[0, 1] == 0)
                & (box_arr[0, 2] == 0)
                & (box_arr[1, 0] == 0)
                & (box_arr[1, 2] == 0)
                & (box_arr[2, 0] == 0)
                & (box_arr[2, 1] == 0)
            )
            diag_b = jnp.broadcast_to(jnp.diag(box_arr)[None, :], (B, 3))
            out = jnp.where(is_diag, diag_b, box_arr[:, :3]) if B == 3 else diag_b
        else:
            out = box_arr[:, :3]
        return out
    if box_arr.ndim == 3:
        d1 = jnp.diagonal(box_arr[0], axis1=-2, axis2=-1)
    elif box_arr.ndim == 2:
        d1 = jnp.diag(box_arr) if box_arr.shape == (3, 3) else box_arr[0, :3]
    else:
        d1 = box_arr[:3]
    return d1[None, :]


def _spread_charges_pallas_soa(
    B: int,
    in_batched: tuple[bool, bool, bool, bool],
    pos_x: jax.Array,
    pos_y: jax.Array,
    pos_z: jax.Array,
    q_flat: jax.Array,
    mask_flat: jax.Array,
    box_flat: jax.Array,
    grid_dims: tuple[int, int, int],
    n_stride: int,
    n_act: int,
    interpret: bool = False,
    *,
    is_triclinic: bool = False,
) -> jax.Array:
    """Zero-glue SoA Pallas SPME charge spread with MK-70 8x8x8 register-brick coalescing."""
    interpret = bool(interpret) or in_export_context()
    b_pos, b_q, b_m, b_box = in_batched
    box_stride = 9 if is_triclinic else 3
    eff_b_box = bool(b_box) and int(box_flat.size) >= B * box_stride
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    mesh_stride = Kx * Ky * Kz
    n_blocks = max(1, (n_act + BLOCK_N - 1) // BLOCK_N)
    dtype = pos_x.dtype
    mesh0_flat = jnp.zeros((B * mesh_stride,), dtype=dtype)
    use_brick = (not interpret) and (Kx >= 8) and (Ky >= 8) and (Kz >= 8) and (
        os.environ.get("PROLIX_PME_DISABLE_BRICK", "").strip() != "1"
    )

    def _spread_kernel(px_ref, py_ref, pz_ref, charges_ref, mask_ref, box_ref, mesh_in_ref, mesh_out_ref):
        del mesh_in_ref
        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        local_idx = jnp.arange(BLOCK_N, dtype=jnp.int32)
        offs = pid * BLOCK_N + local_idx

        pos_base = (b_id if b_pos else 0) * n_stride
        q_base = (b_id if b_q else 0) * n_stride
        m_base = (b_id if b_m else 0) * n_stride
        box_base = (b_id if eff_b_box else 0) * box_stride
        mesh_base = b_id * mesh_stride

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        in_bounds = offs < n_act
        is_real = (_pl_load(mask_ref, (m_base + offs,), mask=in_bounds, other=0) > 0) & in_bounds
        qi = _pl_load(charges_ref, (q_base + offs,), mask=is_real, other=zero)
        active_q = is_real & (jnp.abs(qi) > zero)

        def _do_spread(_):
            px = _pl_load(px_ref, (pos_base + offs,), mask=active_q, other=zero)
            py = _pl_load(py_ref, (pos_base + offs,), mask=active_q, other=zero)
            pz = _pl_load(pz_ref, (pos_base + offs,), mask=active_q, other=zero)

            kx = jnp.asarray(Kx, dtype=dtype)
            ky = jnp.asarray(Ky, dtype=dtype)
            kz = jnp.asarray(Kz, dtype=dtype)

            if is_triclinic:
                eps_b = jnp.asarray(1e-6, dtype=dtype)
                ax = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
                bx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 3, dtype=jnp.int32),))[0]
                by = _pl_load(box_ref, (box_base + jnp.full((1,), 4, dtype=jnp.int32),))[0]
                cx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 6, dtype=jnp.int32),))[0]
                cy_t = _pl_load(box_ref, (box_base + jnp.full((1,), 7, dtype=jnp.int32),))[0]
                cz = _pl_load(box_ref, (box_base + jnp.full((1,), 8, dtype=jnp.int32),))[0]
                b00 = one / jnp.maximum(ax, eps_b)
                b11 = one / jnp.maximum(by, eps_b)
                b22 = one / jnp.maximum(cz, eps_b)
                b10 = (zero - bx_t) * b00 * b11
                b21 = (zero - cy_t) * b11 * b22
                b20 = (bx_t * cy_t - by * cx_t) * (b00 * b11 * b22)
                sx_raw = px * b00 + py * b10 + pz * b20
                sy_raw = py * b11 + pz * b21
                sz_raw = pz * b22
                frac_x = (sx_raw - jnp.floor(sx_raw)) * kx
                frac_y = (sy_raw - jnp.floor(sy_raw)) * ky
                frac_z = (sz_raw - jnp.floor(sz_raw)) * kz
            else:
                bx = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
                by = _pl_load(box_ref, (box_base + jnp.ones((1,), dtype=jnp.int32),))[0]
                bz = _pl_load(box_ref, (box_base + jnp.full((1,), 2, dtype=jnp.int32),))[0]
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
            ky_kz = Ky * Kz

            def _scatter_fallback(_):
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

            if use_brick:
                # MK-70: Coalesced 8x8x8 local register brick when all 32 Morton-sorted atoms'
                # 4x4x4 stencils lie within an 8x8x8 sub-brick (span <= 4 cells per axis).
                first_lane = jnp.min(jnp.where(active_q, local_idx, jnp.int32(BLOCK_N))).astype(jnp.int32)
                is_first = local_idx == first_lane
                g0_x = jnp.sum(jnp.where(is_first, gxi, jnp.int32(0)), dtype=jnp.int32).astype(jnp.int32)
                g0_y = jnp.sum(jnp.where(is_first, gyi, jnp.int32(0)), dtype=jnp.int32).astype(jnp.int32)
                g0_z = jnp.sum(jnp.where(is_first, gzi, jnp.int32(0)), dtype=jnp.int32).astype(jnp.int32)

                rel_x = jnp.where(
                    active_q,
                    jnp.mod(gxi - g0_x + jnp.int32(Kx // 2), jnp.int32(Kx)) - jnp.int32(Kx // 2),
                    jnp.int32(0),
                ).astype(jnp.int32)
                rel_y = jnp.where(
                    active_q,
                    jnp.mod(gyi - g0_y + jnp.int32(Ky // 2), jnp.int32(Ky)) - jnp.int32(Ky // 2),
                    jnp.int32(0),
                ).astype(jnp.int32)
                rel_z = jnp.where(
                    active_q,
                    jnp.mod(gzi - g0_z + jnp.int32(Kz // 2), jnp.int32(Kz)) - jnp.int32(Kz // 2),
                    jnp.int32(0),
                ).astype(jnp.int32)

                min_rx = jnp.min(rel_x).astype(jnp.int32)
                max_rx = jnp.max(rel_x).astype(jnp.int32)
                min_ry = jnp.min(rel_y).astype(jnp.int32)
                max_ry = jnp.max(rel_y).astype(jnp.int32)
                min_rz = jnp.min(rel_z).astype(jnp.int32)
                max_rz = jnp.max(rel_z).astype(jnp.int32)

                fits_brick = (
                    ((max_rx - min_rx) <= 4)
                    & ((max_ry - min_ry) <= 4)
                    & ((max_rz - min_rz) <= 4)
                )

                def _brick_spread(_):
                    dx = (rel_x - min_rx).astype(jnp.int32)
                    dy = (rel_y - min_ry).astype(jnp.int32)
                    dz = (rel_z - min_rz).astype(jnp.int32)
                    u8 = jnp.arange(8, dtype=jnp.int32)[None, :]

                    wy_8 = jnp.where(active_q[:, None], _select_stencil(wy, u8 - dy[:, None]), zero)
                    wz_8 = jnp.where(active_q[:, None], _select_stencil(wz, u8 - dz[:, None]), zero)
                    wyz_64 = (wy_8[:, :, None] * wz_8[:, None, :]).reshape(BLOCK_N, 64)

                    u8_1d = jnp.arange(8, dtype=jnp.int32)
                    iy_8 = jnp.mod(g0_y + min_ry - jnp.int32(1) + u8_1d, jnp.int32(Ky)).astype(jnp.int32)
                    iz_8 = jnp.mod(g0_z + min_rz - jnp.int32(1) + u8_1d, jnp.int32(Kz)).astype(jnp.int32)
                    iyz_64 = (
                        mesh_base + (iy_8[:, None] * jnp.int32(Kz) + iz_8[None, :]).reshape(64)
                    ).astype(jnp.int32)

                    for ux_i in range(8):
                        off_x = (jnp.int32(ux_i) - dx).astype(jnp.int32)
                        wx_u = (
                            wx[0] * (off_x == 0)
                            + wx[1] * (off_x == 1)
                            + wx[2] * (off_x == 2)
                            + wx[3] * (off_x == 3)
                        )
                        q_wx_u = jnp.where(active_q, qi * wx_u, zero)
                        brick_plane = jnp.sum(
                            q_wx_u[:, None] * wyz_64, axis=0
                        ).astype(dtype)
                        ix_u = jnp.mod(
                            g0_x + min_rx - jnp.int32(1) + jnp.int32(ux_i), jnp.int32(Kx)
                        ).astype(jnp.int32)
                        flat_idx_64 = (ix_u * jnp.int32(ky_kz) + iyz_64).astype(jnp.int32)
                        nz_mask = jnp.abs(brick_plane) > zero
                        if interpret:
                            def _serial_b(k, _, idx_v=flat_idx_64, p_v=brick_plane):
                                plt.atomic_add(mesh_out_ref, (idx_v[k],), p_v[k])

                            jax.lax.fori_loop(0, 64, _serial_b, None)
                        else:
                            plt.atomic_add(mesh_out_ref, (flat_idx_64,), brick_plane, mask=nz_mask)

                jax.lax.cond(fits_brick, _brick_spread, _scatter_fallback, None)
            else:
                _scatter_fallback(None)

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
        compiler_params=_pme_compiler_params(B * n_blocks) if not interpret else None,
        input_output_aliases={6: 0},
        name="pme_charge_spread",
    )(pos_x, pos_y, pos_z, q_flat, mask_flat, box_flat, mesh0_flat)
    return mesh_flat.reshape((B, Kx, Ky, Kz))


def _spread_pure_jax_one(
    pos_1: jax.Array,
    q_1: jax.Array,
    mask_1: jax.Array,
    box_1: jax.Array,
    dims: tuple[int, int, int],
    is_triclinic: bool,
) -> jax.Array:
    dtype = pos_1.dtype
    K_arr = jnp.array(dims, dtype=dtype)
    box_arr = jnp.asarray(box_1, dtype=dtype)
    if is_triclinic or (box_arr.ndim == 2 and box_arr.shape == (3, 3)) or box_arr.shape[-1] == 9:
        h_mat = _extract_single_triclinic_box_mat(box_arr, dtype)
        b_inv = invert_box_lower_triangular(h_mat)
        s = jnp.matmul(pos_1, b_inv, precision=jax.lax.Precision.HIGHEST)
        frac = (s - jnp.floor(s)) * K_arr
    else:
        b_vec = box_arr[:3]
        s = pos_1 / b_vec
        frac = (s - jnp.floor(s)) * K_arr
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


def _gather_pure_jax_one(
    pos_1: jax.Array,
    q_1: jax.Array,
    mask_1: jax.Array,
    box_1: jax.Array,
    theta_1: jax.Array,
    dims: tuple[int, int, int],
    is_triclinic: bool,
    compute_dq: bool,
    force_scale: float,
    pot_scale: float,
) -> tuple[jax.Array, jax.Array]:
    dtype = pos_1.dtype
    K_arr = jnp.array(dims, dtype=dtype)
    box_arr = jnp.asarray(box_1, dtype=dtype)
    use_tri = is_triclinic or (box_arr.ndim == 2 and box_arr.shape == (3, 3)) or box_arr.shape[-1] == 9
    if use_tri:
        h_mat = _extract_single_triclinic_box_mat(box_arr, dtype)
        b_inv = invert_box_lower_triangular(h_mat)
        s = jnp.matmul(pos_1, b_inv, precision=jax.lax.Precision.HIGHEST)
        frac = (s - jnp.floor(s)) * K_arr
    else:
        b_inv = None
        b_vec = box_arr[:3]
        s = pos_1 / b_vec
        frac = (s - jnp.floor(s)) * K_arr
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
        jnp.where(
            mask_1 > 0,
            jnp.sum(wx_s * wy_s * wz_s * t_val, axis=1) * jnp.asarray(pot_scale, dtype=dtype),
            0.0,
        ).astype(dtype)
        if compute_dq
        else jnp.zeros_like(q_1, dtype=dtype)
    )
    q_masked = (q_1 * (mask_1 > 0).astype(q_1.dtype)).astype(dtype) * jnp.asarray(force_scale, dtype=dtype)
    du_x = jnp.sum(dwx_s * wy_s * wz_s * t_val, axis=1)
    du_y = jnp.sum(wx_s * dwy_s * wz_s * t_val, axis=1)
    du_z = jnp.sum(wx_s * wy_s * dwz_s * t_val, axis=1)
    if use_tri and b_inv is not None:
        gs = jnp.stack([du_x * K_arr[0], du_y * K_arr[1], du_z * K_arr[2]], axis=-1)
        f_vec = q_masked[:, None] * jnp.matmul(gs, b_inv.T, precision=jax.lax.Precision.HIGHEST)
        return f_vec.astype(dtype), pot
    scale = K_arr / box_arr[:3]
    fx = q_masked * scale[0] * du_x
    fy = q_masked * scale[1] * du_y
    fz = q_masked * scale[2] * du_z
    return jnp.stack([fx, fy, fz], axis=-1).astype(dtype), pot


def _spread_charges_pallas_batch(
    B: int,
    in_batched: tuple[bool, bool, bool, bool],
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
    n_active_atoms: int | None = None,
    *,
    is_triclinic: bool = False,
) -> jax.Array:
    eff_tri = _infer_is_triclinic_pme(box_size, B, is_triclinic)
    interpret = bool(interpret) or in_export_context()
    b_pos, b_q, b_m, b_box = in_batched
    n_atoms = int(positions.shape[1] if b_pos else positions.shape[0])
    n_act = _resolve_pme_active_atoms(n_active_atoms, n_atoms, atom_mask)
    n_blocks = max(1, (n_act + BLOCK_N - 1) // BLOCK_N)
    n_pad = n_blocks * BLOCK_N
    pad = n_pad - n_act

    orig_dtype = positions.dtype
    dtype = orig_dtype if (interpret and orig_dtype == jnp.float64) else jnp.float32
    pos_in = positions.astype(dtype)
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    box_in = _normalize_pme_box_batch(box_size, b_box, B, dtype, is_triclinic=eff_tri)

    if not b_pos:
        pos_in = pos_in[None, ...]
    if not b_q:
        q_in = q_in[None, ...]
    if not b_m:
        mask_in = mask_in[None, ...]

    # MK-62: On GPU (interpret=False), pass unpadded (B, n_atoms) buffers directly using
    # masked pointer loads so zero global-memory slice/pad kernels execute per step.
    if interpret:
        if n_act < n_atoms:
            pos_in = pos_in[:, :n_act, :]
            q_in = q_in[:, :n_act]
            mask_in = mask_in[:, :n_act]
        if pad > 0:
            pos_in = jnp.pad(pos_in, ((0, 0), (0, pad), (0, 0)), constant_values=0.0)
            q_in = jnp.pad(q_in, ((0, 0), (0, pad)), constant_values=0.0)
            mask_in = jnp.pad(mask_in, ((0, 0), (0, pad)), constant_values=0)
        n_stride = n_pad
    else:
        n_stride = n_atoms

    pos_x = jnp.asarray(pos_in[:, :, 0], dtype=dtype).reshape(-1)
    pos_y = jnp.asarray(pos_in[:, :, 1], dtype=dtype).reshape(-1)
    pos_z = jnp.asarray(pos_in[:, :, 2], dtype=dtype).reshape(-1)
    q_flat = q_in.reshape(-1)
    mask_flat = mask_in.reshape(-1)
    box_flat = box_in.reshape(-1)

    mesh_f32 = _spread_charges_pallas_soa(
        B,
        in_batched,
        pos_x,
        pos_y,
        pos_z,
        q_flat,
        mask_flat,
        box_flat,
        grid_dims,
        n_stride,
        n_act,
        interpret,
        is_triclinic=eff_tri,
    )
    return mesh_f32.astype(orig_dtype)


def _spread_charges_pallas_single(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
    n_active_atoms: int | None = None,
    *,
    is_triclinic: bool = False,
) -> jax.Array:
    extra_kw: dict[str, int | bool] = {"is_triclinic": bool(is_triclinic)}
    if n_active_atoms is not None:
        extra_kw["n_active_atoms"] = n_active_atoms
    out = _spread_charges_pallas_batch(
        1,
        (False, False, False, False),
        positions,
        charges,
        atom_mask,
        box_size,
        grid_dims,
        interpret,
        **extra_kw,  # type: ignore[arg-type]
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
    interpret: bool | None = None,
    n_active_atoms: int | None = None,
    is_triclinic: bool = False,
) -> jax.Array:
    """Spread charges onto the PME mesh with a vectorized ``(BLOCK_N=32, 64)`` stencil."""
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    B_top = int(positions.shape[0]) if positions.ndim == 3 else 1
    eff_tri = _infer_is_triclinic_pme(box_size, B_top, is_triclinic)

    if in_export_context():
        if positions.ndim == 3:
            b_q_e = charges.ndim == 2
            b_m_e = atom_mask.ndim == 2
            box_norm_e = _normalize_pme_box_batch(
                box_size, jnp.asarray(box_size).ndim >= 2, B_top, positions.dtype, is_triclinic=eff_tri
            )
            b_bx_e = int(box_norm_e.shape[0]) == B_top
            return jax.vmap(
                lambda b: _spread_pure_jax_one(
                    positions[b],
                    charges[b] if b_q_e else charges,
                    atom_mask[b] if b_m_e else atom_mask,
                    box_norm_e[b] if b_bx_e else box_norm_e[0],
                    grid_tuple,
                    eff_tri,
                )
            )(jnp.arange(B_top))
        return _spread_pure_jax_one(positions, charges, atom_mask, box_size, grid_tuple, eff_tri)

    extra_kw: dict[str, int | bool] = {"is_triclinic": eff_tri}
    if n_active_atoms is not None:
        extra_kw["n_active_atoms"] = n_active_atoms
    if atom_perm is None and os.environ.get("PROLIX_PME_SPATIAL_SORT", "").strip() == "1":
        atom_perm, _ = compute_pme_spatial_perm(
            positions, box_size, grid_tuple, atom_mask=atom_mask, is_triclinic=eff_tri
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

    interp = (not _triton_gpu_available()) if interpret is None else bool(interpret)

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, mask, box):
        return _spread_charges_pallas_single(
            pos,
            q,
            mask,
            box,
            grid_tuple,
            interp,
            n_active_atoms=n_active_atoms,
            is_triclinic=eff_tri,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, mask, box):
        v_tri = _infer_is_triclinic_pme(box, int(axis_size), is_triclinic)
        v_kw: dict[str, int | bool] = {"is_triclinic": v_tri}
        if n_active_atoms is not None:
            v_kw["n_active_atoms"] = n_active_atoms
        out = _spread_charges_pallas_batch(
            int(axis_size),
            (bool(in_batched[0]), bool(in_batched[1]), bool(in_batched[2]), bool(in_batched[3])),
            pos,
            q,
            mask,
            box,
            grid_tuple,
            interp,
            **v_kw,  # type: ignore[arg-type]
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
            interp,
            **extra_kw,  # type: ignore[arg-type]
        )

    return _vmappable(positions, charges, atom_mask, box_size)


def _gather_pallas_soa(  # noqa: PLR0913
    B: int,
    in_batched: tuple[bool, bool, bool, bool, bool],
    pos_x: jax.Array,
    pos_y: jax.Array,
    pos_z: jax.Array,
    q_flat: jax.Array,
    mask_flat: jax.Array,
    box_flat: jax.Array,
    theta_flat: jax.Array,
    grid_dims: tuple[int, int, int],
    n_stride: int,
    n_act: int,
    n_blocks_out: int,
    interpret: bool = False,
    *,
    compute_dq: bool = False,
    force_scale: float = 1.0,
    pot_scale: float = 1.0,
    force_accum: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Zero-glue SoA Pallas SPME force/potential gather with optional in-place force accumulation."""
    interpret = bool(interpret) or in_export_context()
    b_pos, b_q, b_m, b_box, b_mesh = in_batched
    box_stride = 9 if is_triclinic else 3
    eff_b_box = bool(b_box) and int(box_flat.size) >= B * box_stride
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    mesh_stride = Kx * Ky * Kz
    dtype = pos_x.dtype
    has_accum = force_accum is not None

    def _gather_kernel(*refs):
        px_ref, py_ref, pz_ref, charges_ref, mask_ref, box_ref, theta_flat_ref = refs[:7]
        if has_accum:
            fx_in_ref, fy_in_ref, fz_in_ref = refs[7:10]
            out_refs = refs[10:]
        else:
            fx_in_ref = fy_in_ref = fz_in_ref = None
            out_refs = refs[7:]
        fx_out_ref, fy_out_ref, fz_out_ref = out_refs[:3]
        dq_out_refs = out_refs[3:]

        b_id = pl.program_id(0)
        pid = pl.program_id(1)
        offs = pid * BLOCK_N + jnp.arange(BLOCK_N, dtype=jnp.int32)

        pos_base = (b_id if b_pos else 0) * n_stride
        q_base = (b_id if b_q else 0) * n_stride
        m_base = (b_id if b_m else 0) * n_stride
        box_base = (b_id if eff_b_box else 0) * box_stride
        mesh_base = (b_id if b_mesh else 0) * mesh_stride
        out_base = b_id * n_stride

        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        in_bounds_act = offs < n_act
        in_bounds_out = offs < n_stride
        is_real = (_pl_load(mask_ref, (m_base + offs,), mask=in_bounds_act, other=0) > 0) & in_bounds_act
        qi = _pl_load(charges_ref, (q_base + offs,), mask=is_real, other=zero)
        active_lane = is_real if compute_dq else (is_real & (jnp.abs(qi) > zero))
        zero_vec = (offs - offs).astype(dtype)

        def _do_gather(_):
            px = _pl_load(px_ref, (pos_base + offs,), mask=active_lane, other=zero)
            py = _pl_load(py_ref, (pos_base + offs,), mask=active_lane, other=zero)
            pz = _pl_load(pz_ref, (pos_base + offs,), mask=active_lane, other=zero)

            kx = jnp.asarray(Kx, dtype=dtype)
            ky = jnp.asarray(Ky, dtype=dtype)
            kz = jnp.asarray(Kz, dtype=dtype)

            if is_triclinic:
                eps_b = jnp.asarray(1e-6, dtype=dtype)
                ax = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
                bx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 3, dtype=jnp.int32),))[0]
                by = _pl_load(box_ref, (box_base + jnp.full((1,), 4, dtype=jnp.int32),))[0]
                cx_t = _pl_load(box_ref, (box_base + jnp.full((1,), 6, dtype=jnp.int32),))[0]
                cy_t = _pl_load(box_ref, (box_base + jnp.full((1,), 7, dtype=jnp.int32),))[0]
                cz = _pl_load(box_ref, (box_base + jnp.full((1,), 8, dtype=jnp.int32),))[0]
                b00 = one / jnp.maximum(ax, eps_b)
                b11 = one / jnp.maximum(by, eps_b)
                b22 = one / jnp.maximum(cz, eps_b)
                b10 = (zero - bx_t) * b00 * b11
                b21 = (zero - cy_t) * b11 * b22
                b20 = (bx_t * cy_t - by * cx_t) * (b00 * b11 * b22)
                sx_raw = px * b00 + py * b10 + pz * b20
                sy_raw = py * b11 + pz * b21
                sz_raw = pz * b22
                frac_x = (sx_raw - jnp.floor(sx_raw)) * kx
                frac_y = (sy_raw - jnp.floor(sy_raw)) * ky
                frac_z = (sz_raw - jnp.floor(sz_raw)) * kz
                sx = sy = sz = zero
            else:
                bx = _pl_load(box_ref, (box_base + jnp.zeros((1,), dtype=jnp.int32),))[0]
                by = _pl_load(box_ref, (box_base + jnp.ones((1,), dtype=jnp.int32),))[0]
                bz = _pl_load(box_ref, (box_base + jnp.full((1,), 2, dtype=jnp.int32),))[0]
                b00 = b11 = b22 = b10 = b21 = b20 = zero
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

            # MK-62: Fuse force_scale and pot_scale directly into gather output registers
            f_scale = jnp.asarray(force_scale, dtype=dtype)
            q_scale = jnp.asarray(pot_scale, dtype=dtype)
            if compute_dq:
                dq_v = jnp.where(is_real, acc_dq * q_scale, zero).astype(dtype)
            else:
                dq_v = zero_vec
            q_f = qi * f_scale
            if is_triclinic:
                gs_x = kx * acc_dx
                gs_y = ky * acc_dy
                gs_z = kz * acc_dz
                dpx_v = jnp.where(active_lane, q_f * (b00 * gs_x), zero).astype(dtype)
                dpy_v = jnp.where(active_lane, q_f * (b10 * gs_x + b11 * gs_y), zero).astype(dtype)
                dpz_v = jnp.where(
                    active_lane, q_f * (b20 * gs_x + b21 * gs_y + b22 * gs_z), zero
                ).astype(dtype)
            else:
                dpx_v = jnp.where(active_lane, q_f * sx * acc_dx, zero).astype(dtype)
                dpy_v = jnp.where(active_lane, q_f * sy * acc_dy, zero).astype(dtype)
                dpz_v = jnp.where(active_lane, q_f * sz * acc_dz, zero).astype(dtype)
            return (dpx_v, dpy_v, dpz_v, dq_v)

        dpx, dpy, dpz, dq = jax.lax.cond(
            jnp.max(active_lane.astype(jnp.int32)) > 0,
            _do_gather,
            lambda z0: z0,
            (zero_vec, zero_vec, zero_vec, zero_vec),
        )

        if has_accum:
            if interpret:
                dpx = dpx + _pl_load(fx_in_ref, (out_base + offs,))
                dpy = dpy + _pl_load(fy_in_ref, (out_base + offs,))
                dpz = dpz + _pl_load(fz_in_ref, (out_base + offs,))
            else:
                dpx = dpx + _pl_load(fx_in_ref, (out_base + offs,), mask=in_bounds_out, other=zero)
                dpy = dpy + _pl_load(fy_in_ref, (out_base + offs,), mask=in_bounds_out, other=zero)
                dpz = dpz + _pl_load(fz_in_ref, (out_base + offs,), mask=in_bounds_out, other=zero)

        if interpret:
            _pl_store(fx_out_ref, (out_base + offs,), dpx)
            _pl_store(fy_out_ref, (out_base + offs,), dpy)
            _pl_store(fz_out_ref, (out_base + offs,), dpz)
            if compute_dq:
                _pl_store(dq_out_refs[0], (out_base + offs,), dq)
        else:
            _pl_store(fx_out_ref, (out_base + offs,), dpx, mask=in_bounds_out)
            _pl_store(fy_out_ref, (out_base + offs,), dpy, mask=in_bounds_out)
            _pl_store(fz_out_ref, (out_base + offs,), dpz, mask=in_bounds_out)
            if compute_dq:
                _pl_store(dq_out_refs[0], (out_base + offs,), dq, mask=in_bounds_out)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    in_args: list[jax.Array] = [pos_x, pos_y, pos_z, q_flat, mask_flat, box_flat, theta_flat]
    aliases: dict[int, int] = {}
    if force_accum is not None:
        in_args.extend([force_accum[0], force_accum[1], force_accum[2]])
        aliases = {7: 0, 8: 1, 9: 2}

    n_in = len(in_args)
    cp = _pme_compiler_params(B * n_blocks_out) if not interpret else None
    if compute_dq:
        fx_pad, fy_pad, fz_pad, dq_pad = pl.pallas_call(
            _gather_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((B * n_stride,), dtype),
                jax.ShapeDtypeStruct((B * n_stride,), dtype),
                jax.ShapeDtypeStruct((B * n_stride,), dtype),
                jax.ShapeDtypeStruct((B * n_stride,), dtype),
            ],
            in_specs=[any_spec] * n_in,
            out_specs=[any_spec] * 4,
            grid=(B, n_blocks_out),
            interpret=interpret,
            compiler_params=cp,
            input_output_aliases=aliases,
            name="pme_bwd_gather",
        )(*in_args)
        return fx_pad, fy_pad, fz_pad, dq_pad

    fx_pad, fy_pad, fz_pad = pl.pallas_call(
        _gather_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B * n_stride,), dtype),
            jax.ShapeDtypeStruct((B * n_stride,), dtype),
            jax.ShapeDtypeStruct((B * n_stride,), dtype),
        ],
        in_specs=[any_spec] * n_in,
        out_specs=[any_spec] * 3,
        grid=(B, n_blocks_out),
        interpret=interpret,
        compiler_params=cp,
        input_output_aliases=aliases,
        name="pme_bwd_gather",
    )(*in_args)
    return fx_pad, fy_pad, fz_pad, None


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
    n_active_atoms: int | None = None,
    force_scale: float = 1.0,
    pot_scale: float = 1.0,
    *,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    eff_tri = _infer_is_triclinic_pme(box_size, B, is_triclinic)
    interpret = bool(interpret) or in_export_context()
    b_pos, b_q, b_m, b_box, b_mesh = in_batched
    n_atoms = int(positions.shape[1] if b_pos else positions.shape[0])
    n_act = _resolve_pme_active_atoms(n_active_atoms, n_atoms)
    n_blocks = max(1, (n_act + BLOCK_N - 1) // BLOCK_N)
    n_pad = n_blocks * BLOCK_N
    pad = n_pad - n_act

    orig_dtype = positions.dtype
    dtype = orig_dtype if (interpret and orig_dtype == jnp.float64) else jnp.float32
    pos_in = positions.astype(dtype)
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    box_in = _normalize_pme_box_batch(box_size, b_box, B, dtype, is_triclinic=eff_tri)
    theta_in = jnp.asarray(theta_norm, dtype=dtype)

    if not b_pos:
        pos_in = pos_in[None, ...]
    if not b_q:
        q_in = q_in[None, ...]
    if not b_m:
        mask_in = mask_in[None, ...]
    if not b_mesh:
        theta_in = theta_in[None, ...]

    # MK-62: On GPU (interpret=False), eliminate all input/output slice & jnp.pad kernels
    # by reading/writing directly with masked pointer operations over n_stride = n_atoms.
    if interpret:
        if n_act < n_atoms:
            pos_in = pos_in[:, :n_act, :]
            q_in = q_in[:, :n_act]
            mask_in = mask_in[:, :n_act]
        if pad > 0:
            pos_in = jnp.pad(pos_in, ((0, 0), (0, pad), (0, 0)), constant_values=0.0)
            q_in = jnp.pad(q_in, ((0, 0), (0, pad)), constant_values=0.0)
            mask_in = jnp.pad(mask_in, ((0, 0), (0, pad)), constant_values=0)
        n_stride = n_pad
        n_blocks_out = n_blocks
    else:
        n_stride = n_atoms
        n_blocks_out = max(1, (n_atoms + BLOCK_N - 1) // BLOCK_N)

    pos_x = jnp.asarray(pos_in[:, :, 0], dtype=dtype).reshape(-1)
    pos_y = jnp.asarray(pos_in[:, :, 1], dtype=dtype).reshape(-1)
    pos_z = jnp.asarray(pos_in[:, :, 2], dtype=dtype).reshape(-1)
    q_flat = q_in.reshape(-1)
    mask_flat = mask_in.reshape(-1)
    box_flat = box_in.reshape(-1)
    theta_flat = theta_in.reshape(-1)

    fx_pad, fy_pad, fz_pad, dq_pad = _gather_pallas_soa(
        B,
        in_batched,
        pos_x,
        pos_y,
        pos_z,
        q_flat,
        mask_flat,
        box_flat,
        theta_flat,
        grid_dims,
        n_stride,
        n_act,
        n_blocks_out,
        interpret,
        compute_dq=compute_dq,
        force_scale=force_scale,
        pot_scale=pot_scale,
        is_triclinic=eff_tri,
    )
    if dq_pad is not None:
        dq_b = (
            dq_pad.reshape((B, n_pad))[:, :n_act].astype(orig_dtype)
            if interpret
            else dq_pad.reshape((B, n_atoms)).astype(orig_dtype)
        )
    else:
        dq_b = jnp.zeros((B, n_act if interpret else n_atoms), dtype=orig_dtype)
    if interpret:
        fx_b = fx_pad.reshape((B, n_pad))[:, :n_act]
        fy_b = fy_pad.reshape((B, n_pad))[:, :n_act]
        fz_b = fz_pad.reshape((B, n_pad))[:, :n_act]
        dpos = jnp.stack([fx_b, fy_b, fz_b], axis=-1).astype(orig_dtype)
        if n_act < n_atoms:
            tail = n_atoms - n_act
            dpos = jnp.pad(dpos, ((0, 0), (0, tail), (0, 0)), constant_values=0.0)
            dq_b = jnp.pad(dq_b, ((0, 0), (0, tail)), constant_values=0.0)
        return dpos, dq_b
    fx_b = fx_pad.reshape((B, n_atoms))
    fy_b = fy_pad.reshape((B, n_atoms))
    fz_b = fz_pad.reshape((B, n_atoms))
    dpos = jnp.stack([fx_b, fy_b, fz_b], axis=-1).astype(orig_dtype)
    return dpos, dq_b


def _gather_pallas_single(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
    compute_dq: bool = True,
    n_active_atoms: int | None = None,
    force_scale: float = 1.0,
    pot_scale: float = 1.0,
    *,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    extra_kw: dict[str, Any] = {"is_triclinic": bool(is_triclinic)}
    if n_active_atoms is not None:
        extra_kw["n_active_atoms"] = n_active_atoms
    if force_scale != 1.0:
        extra_kw["force_scale"] = force_scale
    if pot_scale != 1.0:
        extra_kw["pot_scale"] = pot_scale
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
        **extra_kw,
    )
    return dpos_b[0], dq_b[0]


def gather_pallas(  # noqa: PLR0913
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
    n_active_atoms: int | None = None,
    force_scale: float = 1.0,
    pot_scale: float = 1.0,
    force_accum: jax.Array | None = None,
    is_triclinic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Gather potential and dE/dr contributions from ``theta_norm``.

    Returns ``(forces, potentials)`` in the same convention as ``_spme_bwd``
    *before* multiplying by ``COULOMB_CONSTANT`` / ``g`` (unless ``force_scale`` /
    ``pot_scale`` are explicitly passed, as in MK-62 fused force evaluation).
    """
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    cdq = bool(compute_dq)
    f_scale = float(force_scale)
    q_scale = float(pot_scale)
    B_top = int(positions.shape[0]) if positions.ndim == 3 else 1
    eff_tri = _infer_is_triclinic_pme(box_size, B_top, is_triclinic)

    if in_export_context():
        if positions.ndim == 3:
            b_q_e = charges.ndim == 2
            b_m_e = atom_mask.ndim == 2
            b_th_e = theta_norm.ndim == 4
            box_norm_e = _normalize_pme_box_batch(
                box_size, jnp.asarray(box_size).ndim >= 2, B_top, positions.dtype, is_triclinic=eff_tri
            )
            b_bx_e = int(box_norm_e.shape[0]) == B_top
            dpos_e, dq_e = jax.vmap(
                lambda b: _gather_pure_jax_one(
                    positions[b],
                    charges[b] if b_q_e else charges,
                    atom_mask[b] if b_m_e else atom_mask,
                    box_norm_e[b] if b_bx_e else box_norm_e[0],
                    theta_norm[b] if b_th_e else theta_norm,
                    grid_tuple,
                    eff_tri,
                    cdq,
                    f_scale,
                    q_scale,
                )
            )(jnp.arange(B_top))
        else:
            dpos_e, dq_e = _gather_pure_jax_one(
                positions, charges, atom_mask, box_size, theta_norm, grid_tuple, eff_tri, cdq, f_scale, q_scale
            )
        if force_accum is not None:
            dpos_e = dpos_e + jnp.asarray(force_accum, dtype=dpos_e.dtype)
        return dpos_e, dq_e

    extra_kw: dict[str, Any] = {"is_triclinic": eff_tri}
    if n_active_atoms is not None:
        extra_kw["n_active_atoms"] = n_active_atoms
    if f_scale != 1.0:
        extra_kw["force_scale"] = f_scale
    if q_scale != 1.0:
        extra_kw["pot_scale"] = q_scale

    inv_p = None
    if atom_perm is None and os.environ.get("PROLIX_PME_SPATIAL_SORT", "").strip() == "1":
        atom_perm, atom_inv_perm = compute_pme_spatial_perm(
            positions, box_size, grid_tuple, atom_mask=atom_mask, is_triclinic=eff_tri
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
            pos,
            q,
            mask,
            box,
            dE_dmesh,
            grid_tuple,
            interpret,
            compute_dq=cdq,
            n_active_atoms=n_active_atoms,
            force_scale=f_scale,
            pot_scale=q_scale,
            is_triclinic=eff_tri,
        )

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, mask, box, dE_dmesh):
        v_tri = _infer_is_triclinic_pme(box, int(axis_size), is_triclinic)
        v_kw: dict[str, Any] = {"is_triclinic": v_tri}
        if n_active_atoms is not None:
            v_kw["n_active_atoms"] = n_active_atoms
        if f_scale != 1.0:
            v_kw["force_scale"] = f_scale
        if q_scale != 1.0:
            v_kw["pot_scale"] = q_scale
        dpos_out, dq_out = _gather_pallas_batch(
            int(axis_size),
            (
                bool(in_batched[0]),
                bool(in_batched[1]),
                bool(in_batched[2]),
                bool(in_batched[3]),
                bool(in_batched[4]),
            ),
            pos,
            q,
            mask,
            box,
            dE_dmesh,
            grid_tuple,
            interpret,
            compute_dq=cdq,
            **v_kw,
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
            **extra_kw,
        )
        if inv_p is not None:
            dpos_s, dq_s = dpos_s[:, inv_p, :], dq_s[:, inv_p]
        if force_accum is not None:
            dpos_s = dpos_s + jnp.asarray(force_accum, dtype=dpos_s.dtype)
        return dpos_s, dq_s

    dpos_s, dq_s = _vmappable(positions, charges, atom_mask, box_size, theta_norm)
    if inv_p is not None:
        dpos_s, dq_s = dpos_s[inv_p], dq_s[inv_p]
    if force_accum is not None:
        dpos_s = dpos_s + jnp.asarray(force_accum, dtype=dpos_s.dtype)
    return dpos_s, dq_s


def make_spread_op(
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    is_triclinic: bool = False,
    context: object | None = None,
):
    """Build the SPME charge spread op for one mesh shape."""
    dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    def _spread_ref(pos, q, mask, box):
        B_r = int(pos.shape[0]) if pos.ndim == 3 else 1
        eff_tri = _infer_is_triclinic_pme(box, B_r, is_triclinic)
        if pos.ndim == 3:
            b_q = q.ndim == 2
            b_m = mask.ndim == 2
            box_n = _normalize_pme_box_batch(
                box, jnp.asarray(box).ndim >= 2, B_r, pos.dtype, is_triclinic=eff_tri
            )
            b_box = int(box_n.shape[0]) == B_r
            return jax.vmap(
                lambda b: _spread_pure_jax_one(
                    pos[b],
                    q[b] if b_q else q,
                    mask[b] if b_m else mask,
                    box_n[b] if b_box else box_n[0],
                    dims,
                    eff_tri,
                )
            )(jnp.arange(B_r))
        return _spread_pure_jax_one(pos, q, mask, box, dims, eff_tri)

    _spread_gpu_pallas = functools.partial(
        spread_charges_pallas,
        grid_dims=dims,
        order=order,
        interpret=False,
        is_triclinic=is_triclinic,
    )

    def _spread_gpu(pos, q, mask, box):
        if not _triton_gpu_available():
            return _spread_ref(pos, q, mask, box)
        return _spread_gpu_pallas(pos, q, mask, box)

    return xtrax.op(
        _spread_ref,
        kernels={
            "gpu": _spread_gpu,
            "cpu": functools.partial(
                spread_charges_pallas,
                grid_dims=dims,
                order=order,
                interpret=True,
                is_triclinic=is_triclinic,
            ),
        },
        name="pme_spread",
        context=context,
    )


def make_gather_op(
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    compute_dq: bool = True,
    force_scale: float = 1.0,
    pot_scale: float = 1.0,
    is_triclinic: bool = False,
    context: object | None = None,
):
    """Build the SPME backward gather op for one mesh shape."""
    dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    cdq = bool(compute_dq)
    f_scale = float(force_scale)
    q_scale = float(pot_scale)

    def _gather_ref(pos, q, mask, box, theta):
        B_r = int(pos.shape[0]) if pos.ndim == 3 else 1
        eff_tri = _infer_is_triclinic_pme(box, B_r, is_triclinic)
        if pos.ndim == 3:
            b_q = q.ndim == 2
            b_m = mask.ndim == 2
            b_th = theta.ndim == 4
            box_n = _normalize_pme_box_batch(
                box, jnp.asarray(box).ndim >= 2, B_r, pos.dtype, is_triclinic=eff_tri
            )
            b_box = int(box_n.shape[0]) == B_r
            return jax.vmap(
                lambda b: _gather_pure_jax_one(
                    pos[b],
                    q[b] if b_q else q,
                    mask[b] if b_m else mask,
                    box_n[b] if b_box else box_n[0],
                    theta[b] if b_th else theta,
                    dims,
                    eff_tri,
                    cdq,
                    f_scale,
                    q_scale,
                )
            )(jnp.arange(B_r))
        return _gather_pure_jax_one(pos, q, mask, box, theta, dims, eff_tri, cdq, f_scale, q_scale)

    _gather_gpu_pallas = functools.partial(
        gather_pallas,
        grid_dims=dims,
        order=order,
        interpret=False,
        compute_dq=cdq,
        force_scale=f_scale,
        pot_scale=q_scale,
        is_triclinic=is_triclinic,
    )

    def _gather_gpu(pos, q, mask, box, theta):
        if not _triton_gpu_available():
            return _gather_ref(pos, q, mask, box, theta)
        return _gather_gpu_pallas(pos, q, mask, box, theta)

    return xtrax.op(
        _gather_ref,
        kernels={
            "gpu": _gather_gpu,
            "cpu": functools.partial(
                gather_pallas,
                grid_dims=dims,
                order=order,
                interpret=True,
                compute_dq=cdq,
                force_scale=f_scale,
                pot_scale=q_scale,
                is_triclinic=is_triclinic,
            ),
        },
        name="pme_gather",
        context=context,
    )


