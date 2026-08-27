"""Shared SPME tiny-system fixtures and 4³ spread oracle for stencil/Pallas tests."""

from __future__ import annotations

import jax.numpy as jnp

from prolix.physics.pme import _bspline4


def tiny_system():
    box = jnp.array([12.0, 12.0, 12.0])
    pos = jnp.array(
        [
            [3.1, 4.2, 5.3],
            [8.7, 1.2, 9.4],
            [0.2, 11.8, 6.0],
        ]
    )
    charges = jnp.array([1.0, -0.5, -0.5])
    mask = jnp.array([True, True, False])
    grid_dims = (12, 12, 12)
    return pos, charges, mask, box, grid_dims


def spread_loop_oracle(positions, charges, atom_mask, box_size, grid_dims, order=4):
    """Reference: Python 4³ scatter-add."""
    Kx, Ky, Kz = grid_dims
    work_dtype = positions.dtype
    K_arr = jnp.array([Kx, Ky, Kz], dtype=work_dtype)
    frac = (positions / box_size) * K_arr
    grid_idx = jnp.floor(frac).astype(jnp.int32)
    u = frac - grid_idx.astype(work_dtype)
    wx = _bspline4(u[:, 0])
    wy = _bspline4(u[:, 1])
    wz = _bspline4(u[:, 2])
    Q = jnp.zeros(grid_dims, dtype=work_dtype)
    q = charges * atom_mask.astype(charges.dtype)
    for dx in range(order):
        for dy in range(order):
            for dz in range(order):
                ix = (grid_idx[:, 0] + dx - 1) % Kx
                iy = (grid_idx[:, 1] + dy - 1) % Ky
                iz = (grid_idx[:, 2] + dz - 1) % Kz
                w = (wx[dx] * wy[dy] * wz[dz] * q).astype(Q.dtype)
                Q = Q.at[ix, iy, iz].add(w)
    return Q
