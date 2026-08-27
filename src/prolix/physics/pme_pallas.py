"""Atom-parallel Pallas SPME B-spline spread/gather (ADR-006).

Opt-in kernels. Default production path remains the XLA stencil in ``pme.py``.
Do not autodiff through ``atomic_add``; pair with the existing custom VJP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def _bspline4_regs(u):
    """Order-4 cardinal B-spline weights as four scalars."""
    one = u * 0 + 1.0
    six = one * 6.0
    u2 = u * u
    u3 = u2 * u
    w0 = (one - 3.0 * u + 3.0 * u2 - u3) / six
    w1 = (4.0 * one - 6.0 * u2 + 3.0 * u3) / six
    w2 = (one + 3.0 * u + 3.0 * u2 - 3.0 * u3) / six
    w3 = u3 / six
    return (w0, w1, w2, w3)


def _bspline4_deriv_regs(u):
    one = u * 0 + 1.0
    six = one * 6.0
    u2 = u * u
    dw0 = (-3.0 * one + 6.0 * u - 3.0 * u2) / six
    dw1 = (-12.0 * u + 9.0 * u2) / six
    dw2 = (3.0 * one + 6.0 * u - 9.0 * u2) / six
    dw3 = (3.0 * u2) / six
    return (dw0, dw1, dw2, dw3)


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
    """Spread charges onto the PME mesh with a per-atom register stencil."""
    if order != 4:
        raise NotImplementedError("Pallas SPME stencil is order-4 only")
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    n_atoms = int(positions.shape[0])
    dtype = positions.dtype
    q_in = charges.astype(dtype)
    box = box_size.astype(dtype)
    mesh0 = jnp.zeros((Kx, Ky, Kz), dtype=dtype)

    def kernel(pos_ref, q_ref, mask_ref, box_ref, mesh_in_ref, mesh_out_ref):
        del mesh_in_ref
        i = pl.program_id(0)
        is_real = mask_ref[i]
        px = pos_ref[i, 0]
        py = pos_ref[i, 1]
        pz = pos_ref[i, 2]
        bx = box_ref[0]
        by = box_ref[1]
        bz = box_ref[2]
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
        wx = _bspline4_regs(ux)
        wy = _bspline4_regs(uy)
        wz = _bspline4_regs(uz)
        qi = q_ref[i]
        gxi = gx.astype(jnp.int32)
        gyi = gy.astype(jnp.int32)
        gzi = gz.astype(jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        for dx in range(4):
            for dy in range(4):
                for dz in range(4):
                    ix = (gxi + dx - 1) % Kx
                    iy = (gyi + dy - 1) % Ky
                    iz = (gzi + dz - 1) % Kz
                    w = wx[dx] * wy[dy] * wz[dz] * qi
                    w = jnp.where(is_real, w, zero)
                    plt.atomic_add(mesh_out_ref, (ix, iy, iz), w)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((Kx, Ky, Kz), dtype),
        in_specs=[any_spec, any_spec, any_spec, any_spec, any_spec],
        out_specs=any_spec,
        grid=(n_atoms,),
        interpret=interpret,
        input_output_aliases={4: 0},
    )(positions, q_in, atom_mask, box, mesh0)


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
    Kx, Ky, Kz = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    n_atoms = int(positions.shape[0])
    dtype = positions.dtype
    q_in = charges.astype(dtype)
    box = box_size.astype(dtype)

    def kernel(pos_ref, q_ref, mask_ref, box_ref, theta_ref, force_in, pot_in, force_ref, pot_ref):
        del force_in, pot_in
        i = pl.program_id(0)
        is_real = mask_ref[i]
        px = pos_ref[i, 0]
        py = pos_ref[i, 1]
        pz = pos_ref[i, 2]
        bx = box_ref[0]
        by = box_ref[1]
        bz = box_ref[2]
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
        wx = _bspline4_regs(ux)
        wy = _bspline4_regs(uy)
        wz = _bspline4_regs(uz)
        dwx = _bspline4_deriv_regs(ux)
        dwy = _bspline4_deriv_regs(uy)
        dwz = _bspline4_deriv_regs(uz)
        qi = q_ref[i]
        gxi = gx.astype(jnp.int32)
        gyi = gy.astype(jnp.int32)
        gzi = gz.astype(jnp.int32)
        zero = jnp.asarray(0.0, dtype=dtype)
        sx = kx / bx
        sy = ky / by
        sz = kz / bz
        fx = zero
        fy = zero
        fz = zero
        pot = zero
        for dx in range(4):
            for dy in range(4):
                for dz in range(4):
                    ix = (gxi + dx - 1) % Kx
                    iy = (gyi + dy - 1) % Ky
                    iz = (gzi + dz - 1) % Kz
                    t_val = theta_ref[ix, iy, iz]
                    w = wx[dx] * wy[dy] * wz[dz]
                    pot = pot + w * t_val
                    fx = fx + qi * sx * dwx[dx] * wy[dy] * wz[dz] * t_val
                    fy = fy + qi * sy * wx[dx] * dwy[dy] * wz[dz] * t_val
                    fz = fz + qi * sz * wx[dx] * wy[dy] * dwz[dz] * t_val
        force_ref[i, 0] = jnp.where(is_real, fx, zero)
        force_ref[i, 1] = jnp.where(is_real, fy, zero)
        force_ref[i, 2] = jnp.where(is_real, fz, zero)
        pot_ref[i] = jnp.where(is_real, pot, zero)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    forces0 = jnp.zeros((n_atoms, 3), dtype=dtype)
    pots0 = jnp.zeros((n_atoms,), dtype=dtype)
    # Two outputs: forces, potentials. Alias last two inputs.
    out = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((n_atoms, 3), dtype),
            jax.ShapeDtypeStruct((n_atoms,), dtype),
        ],
        in_specs=[any_spec] * 7,
        out_specs=[any_spec, any_spec],
        grid=(n_atoms,),
        interpret=interpret,
        input_output_aliases={5: 0, 6: 1},
    )(positions, q_in, atom_mask, box, theta_norm, forces0, pots0)
    return out[0], out[1]
