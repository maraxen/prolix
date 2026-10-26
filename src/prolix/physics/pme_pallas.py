"""Vectorized Pallas SPME B-spline spread/gather (ADR-006, R4.1).

Opt-in kernels. Default production path remains the XLA stencil in ``pme.py``.
Do not autodiff through ``atomic_add``; pair with the existing custom VJP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax._src.pallas.primitives import load as _pl_load, store as _pl_store
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

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


def _spread_charges_pallas_single(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
) -> jax.Array:
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    n_atoms = int(positions.shape[0])
    n_blocks = max(1, (n_atoms + BLOCK_N - 1) // BLOCK_N)
    n_pad = n_blocks * BLOCK_N
    pad = n_pad - n_atoms

    dtype = positions.dtype
    pos_in = positions
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    if pad > 0:
        pos_in = jnp.pad(pos_in, ((0, pad), (0, 0)), constant_values=0.0)
        q_in = jnp.pad(q_in, (0, pad), constant_values=0.0)
        mask_in = jnp.pad(mask_in, (0, pad), constant_values=0)

    pos_x = jnp.asarray(pos_in[:, 0])
    pos_y = jnp.asarray(pos_in[:, 1])
    pos_z = jnp.asarray(pos_in[:, 2])
    box = jnp.asarray(box_size, dtype=dtype)
    mesh0_flat = jnp.zeros((Kx * Ky * Kz,), dtype=dtype)

    def _spread_kernel(px_ref, py_ref, pz_ref, charges_ref, mask_ref, box_ref, mesh_in_ref, mesh_out_ref):
        del mesh_in_ref
        offs = pl.program_id(0) * BLOCK_N + jnp.arange(BLOCK_N, dtype=jnp.int32)
        px = _pl_load(px_ref, (offs,))
        py = _pl_load(py_ref, (offs,))
        pz = _pl_load(pz_ref, (offs,))
        qi = _pl_load(charges_ref, (offs,))
        is_real = (_pl_load(mask_ref, (offs,)) > 0) & (offs < n_atoms)

        bx = _pl_load(box_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (jnp.ones((1,), dtype=jnp.int32),))[0]
        bz = _pl_load(box_ref, (jnp.full((1,), 2, dtype=jnp.int32),))[0]
        kx = jnp.asarray(Kx, dtype=dtype)
        ky = jnp.asarray(Ky, dtype=dtype)
        kz = jnp.asarray(Kz, dtype=dtype)

        frac_x = (px / bx) * kx
        frac_y = (py / by) * ky
        frac_z = (pz / bz) * kz
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

        s_idx = jnp.arange(64, dtype=jnp.int32)
        ox = (s_idx // 16)[None, :]
        oy = ((s_idx // 4) % 4)[None, :]
        oz = (s_idx % 4)[None, :]

        ix = jnp.mod(gxi[:, None] + ox - 1, Kx)
        iy = jnp.mod(gyi[:, None] + oy - 1, Ky)
        iz = jnp.mod(gzi[:, None] + oz - 1, Kz)
        flat_idx = (ix * Ky + iy) * Kz + iz

        wx_s = _select_stencil(wx, ox)
        wy_s = _select_stencil(wy, oy)
        wz_s = _select_stencil(wz, oz)

        mask_2d = jnp.broadcast_to(is_real[:, None], (BLOCK_N, 64))
        w_2d = jnp.where(mask_2d, (wx_s * wy_s * wz_s) * qi[:, None], jnp.asarray(0.0, dtype=dtype)).astype(dtype)

        flat_idx_1d = flat_idx.reshape(BLOCK_N * 64)
        w_1d = w_2d.reshape(BLOCK_N * 64)
        mask_1d = mask_2d.reshape(BLOCK_N * 64)
        if interpret:
            def _serial_atomic(k, _):
                plt.atomic_add(mesh_out_ref, (flat_idx_1d[k],), w_1d[k])
                return None

            jax.lax.fori_loop(0, BLOCK_N * 64, _serial_atomic, None)
        else:
            plt.atomic_add(mesh_out_ref, (flat_idx_1d,), w_1d, mask=mask_1d)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    mesh_flat = pl.pallas_call(
        _spread_kernel,
        out_shape=jax.ShapeDtypeStruct((Kx * Ky * Kz,), dtype),
        in_specs=[any_spec] * 7,
        out_specs=any_spec,
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        input_output_aliases={6: 0},
        name="pme_charge_spread",
    )(pos_x, pos_y, pos_z, q_in, mask_in, box, mesh0_flat)
    return mesh_flat.reshape((Kx, Ky, Kz))


def spread_charges_pallas(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Spread charges onto the PME mesh with a vectorized ``(BLOCK_N=32, 64)`` stencil."""
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, mask, box):
        return _spread_charges_pallas_single(pos, q, mask, box, grid_tuple, interpret)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, mask, box):
        pos_b, q_b, mask_b, box_b = in_batched

        def _body(idx):
            p_i = pos[idx] if pos_b else pos
            q_i = q[idx] if q_b else q
            m_i = mask[idx] if mask_b else mask
            b_i = box[idx] if box_b else box
            return _vmappable(p_i, q_i, m_i, b_i)

        out = jax.lax.map(_body, jnp.arange(axis_size))
        return out, True

    return _vmappable(positions, charges, atom_mask, box_size)


def _gather_pallas_single(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    n_atoms = int(positions.shape[0])
    n_blocks = max(1, (n_atoms + BLOCK_N - 1) // BLOCK_N)
    n_pad = n_blocks * BLOCK_N
    pad = n_pad - n_atoms

    dtype = positions.dtype
    pos_in = positions
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    if pad > 0:
        pos_in = jnp.pad(pos_in, ((0, pad), (0, 0)), constant_values=0.0)
        q_in = jnp.pad(q_in, (0, pad), constant_values=0.0)
        mask_in = jnp.pad(mask_in, (0, pad), constant_values=0)

    pos_x = jnp.asarray(pos_in[:, 0])
    pos_y = jnp.asarray(pos_in[:, 1])
    pos_z = jnp.asarray(pos_in[:, 2])
    box = jnp.asarray(box_size, dtype=dtype)
    theta_flat = jnp.asarray(theta_norm, dtype=dtype).reshape(-1)

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
        dq_out_ref,
    ):
        offs = pl.program_id(0) * BLOCK_N + jnp.arange(BLOCK_N, dtype=jnp.int32)
        px = _pl_load(px_ref, (offs,))
        py = _pl_load(py_ref, (offs,))
        pz = _pl_load(pz_ref, (offs,))
        qi = _pl_load(charges_ref, (offs,))
        is_real = (_pl_load(mask_ref, (offs,)) > 0) & (offs < n_atoms)

        bx = _pl_load(box_ref, (jnp.zeros((1,), dtype=jnp.int32),))[0]
        by = _pl_load(box_ref, (jnp.ones((1,), dtype=jnp.int32),))[0]
        bz = _pl_load(box_ref, (jnp.full((1,), 2, dtype=jnp.int32),))[0]
        kx = jnp.asarray(Kx, dtype=dtype)
        ky = jnp.asarray(Ky, dtype=dtype)
        kz = jnp.asarray(Kz, dtype=dtype)

        frac_x = (px / bx) * kx
        frac_y = (py / by) * ky
        frac_z = (pz / bz) * kz
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

        s_idx = jnp.arange(64, dtype=jnp.int32)
        ox = (s_idx // 16)[None, :]
        oy = ((s_idx // 4) % 4)[None, :]
        oz = (s_idx % 4)[None, :]

        ix = jnp.mod(gxi[:, None] + ox - 1, Kx)
        iy = jnp.mod(gyi[:, None] + oy - 1, Ky)
        iz = jnp.mod(gzi[:, None] + oz - 1, Kz)
        flat_idx = (ix * Ky + iy) * Kz + iz

        wx_s = _select_stencil(wx, ox)
        wy_s = _select_stencil(wy, oy)
        wz_s = _select_stencil(wz, oz)
        dwx_s = _select_stencil(dwx, ox)
        dwy_s = _select_stencil(dwy, oy)
        dwz_s = _select_stencil(dwz, oz)

        zero = jnp.asarray(0.0, dtype=dtype)
        mask_2d = jnp.broadcast_to(is_real[:, None], (BLOCK_N, 64))
        flat_idx_1d = flat_idx.reshape(BLOCK_N * 64)
        mask_1d = mask_2d.reshape(BLOCK_N * 64)
        t_val = _pl_load(theta_flat_ref, (flat_idx_1d,), mask=mask_1d, other=zero).reshape(
            BLOCK_N, 64
        )

        sx = kx / bx
        sy = ky / by
        sz = kz / bz

        w_unit = wx_s * wy_s * wz_s
        dq = jnp.where(is_real, jnp.sum(w_unit * t_val, axis=-1), zero).astype(dtype)
        dpx = jnp.where(
            is_real,
            qi * sx * jnp.sum(dwx_s * wy_s * wz_s * t_val, axis=-1),
            zero,
        ).astype(dtype)
        dpy = jnp.where(
            is_real,
            qi * sy * jnp.sum(wx_s * dwy_s * wz_s * t_val, axis=-1),
            zero,
        ).astype(dtype)
        dpz = jnp.where(
            is_real,
            qi * sz * jnp.sum(wx_s * wy_s * dwz_s * t_val, axis=-1),
            zero,
        ).astype(dtype)

        _pl_store(fx_out_ref, (offs,), dpx)
        _pl_store(fy_out_ref, (offs,), dpy)
        _pl_store(fz_out_ref, (offs,), dpz)
        _pl_store(dq_out_ref, (offs,), dq)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    fx_pad, fy_pad, fz_pad, dq_pad = pl.pallas_call(
        _gather_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((n_pad,), dtype),
            jax.ShapeDtypeStruct((n_pad,), dtype),
            jax.ShapeDtypeStruct((n_pad,), dtype),
            jax.ShapeDtypeStruct((n_pad,), dtype),
        ],
        in_specs=[any_spec] * 7,
        out_specs=[any_spec] * 4,
        grid=(n_blocks,),
        interpret=interpret,
        compiler_params=plt.CompilerParams() if not interpret else None,
        name="pme_bwd_gather",
    )(pos_x, pos_y, pos_z, q_in, mask_in, box, theta_flat)
    dpos = jnp.stack([fx_pad[:n_atoms], fy_pad[:n_atoms], fz_pad[:n_atoms]], axis=-1)
    return dpos, dq_pad[:n_atoms]


def gather_pallas(
    positions: jax.Array,
    charges: jax.Array,
    atom_mask: jax.Array,
    box_size: jax.Array,
    theta_norm: jax.Array,
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Gather potential and dE/dr contributions from ``theta_norm``.

    Returns ``(forces, potentials)`` in the same convention as ``_spme_bwd``
    *before* multiplying by ``COULOMB_CONSTANT`` / ``g``.
    """
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    @jax.custom_batching.custom_vmap
    def _vmappable(pos, q, mask, box, dE_dmesh):
        return _gather_pallas_single(pos, q, mask, box, dE_dmesh, grid_tuple, interpret)

    @_vmappable.def_vmap
    def _vmap_rule(axis_size, in_batched, pos, q, mask, box, dE_dmesh):
        pos_b, q_b, mask_b, box_b, mesh_b = in_batched

        def _body(idx):
            p_i = pos[idx] if pos_b else pos
            q_i = q[idx] if q_b else q
            m_i = mask[idx] if mask_b else mask
            b_i = box[idx] if box_b else box
            t_i = dE_dmesh[idx] if mesh_b else dE_dmesh
            return _vmappable(p_i, q_i, m_i, b_i, t_i)

        dpos_out, dq_out = jax.lax.map(_body, jnp.arange(axis_size))
        return (dpos_out, dq_out), (True, True)

    return _vmappable(positions, charges, atom_mask, box_size, theta_norm)
