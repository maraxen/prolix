"""Vectorized Pallas SPME B-spline spread/gather (ADR-006, R4.1).

Opt-in kernels. Default production path remains the XLA stencil in ``pme.py``.
Do not autodiff through ``atomic_add``; pair with the existing custom VJP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax._src.pallas.primitives import load as _pl_load
from jax._src.pallas.primitives import store as _pl_store
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

    orig_dtype = positions.dtype
    dtype = jnp.float32
    pos_in = positions.astype(dtype)
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    if pad > 0:
        pos_in = jnp.pad(pos_in, ((0, pad), (0, 0)), constant_values=0.0)
        q_in = jnp.pad(q_in, (0, pad), constant_values=0.0)
        mask_in = jnp.pad(mask_in, (0, pad), constant_values=0)

    pos_x = jnp.asarray(pos_in[:, 0], dtype=dtype)
    pos_y = jnp.asarray(pos_in[:, 1], dtype=dtype)
    pos_z = jnp.asarray(pos_in[:, 2], dtype=dtype)
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
    return mesh_flat.reshape((Kx, Ky, Kz)).astype(orig_dtype)


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
        pos_b, q_b, mask_b, box_b = in_batched

        def _body(idx):
            p_i = pos[idx] if pos_b else pos
            q_i = q[idx] if q_b else q
            m_i = mask[idx] if mask_b else mask
            b_i = box[idx] if box_b else box
            return _vmappable(p_i, q_i, m_i, b_i)

        out = jax.lax.map(_body, jnp.arange(axis_size))
        return out, True

    if positions.ndim == 3:
        B = positions.shape[0]
        b_q = charges.ndim == 2
        b_m = atom_mask.ndim == 2
        b_box = jnp.asarray(box_size).ndim >= 2
        return jax.vmap(
            lambda b: _vmappable(
                positions[b],
                charges[b] if b_q else charges,
                atom_mask[b] if b_m else atom_mask,
                box_size[b] if b_box else box_size,
            )
        )(jnp.arange(B))

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

    orig_dtype = positions.dtype
    dtype = jnp.float32
    pos_in = positions.astype(dtype)
    q_in = charges.astype(dtype)
    mask_in = jnp.asarray(atom_mask, dtype=jnp.int32)
    if pad > 0:
        pos_in = jnp.pad(pos_in, ((0, pad), (0, 0)), constant_values=0.0)
        q_in = jnp.pad(q_in, (0, pad), constant_values=0.0)
        mask_in = jnp.pad(mask_in, (0, pad), constant_values=0)

    pos_x = jnp.asarray(pos_in[:, 0], dtype=dtype)
    pos_y = jnp.asarray(pos_in[:, 1], dtype=dtype)
    pos_z = jnp.asarray(pos_in[:, 2], dtype=dtype)
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
    return dpos.astype(orig_dtype), dq_pad[:n_atoms].astype(orig_dtype)


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
) -> tuple[jax.Array, jax.Array]:
    """Gather potential and dE/dr contributions from ``theta_norm``.

    Returns ``(forces, potentials)`` in the same convention as ``_spme_bwd``
    *before* multiplying by ``COULOMB_CONSTANT`` / ``g``.
    """
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    grid_tuple = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    inv_p = None
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

    if positions.ndim == 3:
        B = positions.shape[0]
        b_q = charges.ndim == 2
        b_m = atom_mask.ndim == 2
        b_box = jnp.asarray(box_size).ndim >= 2
        b_mesh = theta_norm.ndim == 4
        dpos_s, dq_s = jax.vmap(
            lambda b: _vmappable(
                positions[b],
                charges[b] if b_q else charges,
                atom_mask[b] if b_m else atom_mask,
                box_size[b] if b_box else box_size,
                theta_norm[b] if b_mesh else theta_norm,
            )
        )(jnp.arange(B))
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
    context: object | None = None,
):
    """Build the SPME backward gather op for one mesh shape."""
    import functools

    import xtrax

    dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

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
        pot = jnp.where(mask_1 > 0, jnp.sum(wx_s * wy_s * wz_s * t_val, axis=1), 0.0).astype(dtype)
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
            "gpu": functools.partial(gather_pallas, grid_dims=dims, order=order, interpret=False),
            "cpu": functools.partial(gather_pallas, grid_dims=dims, order=order, interpret=True),
        },
        name="pme_gather",
        context=context,
    )

