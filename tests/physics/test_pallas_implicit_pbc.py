"""CPU parity and export tests for PBC box preservation in implicit GB/SASA kernels and Born radii flags."""

from __future__ import annotations

import jax
import jax.export
import jax.numpy as jnp
import numpy as np

from prolix.export import export_context
from prolix.pallas_kernels import _pallas_born_radii_single
from prolix.physics.generalized_born import compute_born_radii_neighbor_list
from prolix.physics.pallas_implicit import (
    gb_ace_nl_reference,
    pallas_gb_ace_nl,
    pallas_sasa_nl,
    sasa_nl_reference,
)


def _straddling_pbc_system():
    """4 atoms straddling an orthorhombic (15, 16, 17) periodic boundary."""
    box = jnp.array([15.0, 16.0, 17.0], dtype=jnp.float32)
    box6 = jnp.array([15.0, 16.0, 17.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    # Atom 0 near x=0.5, Atom 1 near x=13.7 -> MIC dx = 1.8 A (raw = 13.2 A)
    pos = jnp.array(
        [
            [0.5, 2.0, 3.0],
            [13.7, 2.2, 3.1],
            [1.9, 15.1, 3.4],
            [14.2, 0.8, 16.1],
        ],
        dtype=jnp.float32,
    )
    q = jnp.array([0.45, -0.55, 0.30, -0.20], dtype=jnp.float32)
    rad = jnp.array([1.7, 1.55, 1.6, 1.5], dtype=jnp.float32)
    sr = jnp.array([1.3, 1.2, 1.25, 1.15], dtype=jnp.float32)
    mask = jnp.ones((4,), dtype=jnp.bool_)
    nbr = jnp.array(
        [
            [1, 2, 3, -1],
            [0, 2, 3, -1],
            [0, 1, 3, -1],
            [0, 1, 2, -1],
        ],
        dtype=jnp.int32,
    )
    return pos, q, rad, sr, mask, nbr, box, box6


def _fd_grad(fn, arg, eps: float = 1e-3):
    flat_0 = np.asarray(arg, dtype=np.float64).ravel()
    grad = np.zeros_like(flat_0)
    for idx in range(flat_0.size):
        delta = np.zeros_like(flat_0)
        delta[idx] = eps
        e_p = float(fn(jnp.asarray((flat_0 + delta).reshape(arg.shape), dtype=arg.dtype)))
        e_m = float(fn(jnp.asarray((flat_0 - delta).reshape(arg.shape), dtype=arg.dtype)))
        grad[idx] = (e_p - e_m) / (2.0 * eps)
    return grad.reshape(arg.shape)


def test_gb_ace_nl_pbc_energy_grads_and_export():
    pos, q, rad, sr, mask, nbr, box, box6 = _straddling_pbc_system()

    e_pallas = pallas_gb_ace_nl(pos, q, rad, nbr, mask, scaled_radii=sr, box=box, interpret=True)
    e_ref = gb_ace_nl_reference(pos, q, rad, sr, mask, nbr, box6, has_box=True)
    e_nobox = gb_ace_nl_reference(pos, q, rad, sr, mask, nbr, box6, has_box=False)
    np.testing.assert_allclose(float(e_pallas), float(e_ref), rtol=1e-4, atol=1e-4)
    assert abs(float(e_ref) - float(e_nobox)) > 1.0

    # Gradients w.r.t. positions, radii, scaled_radii must match reference & finite differences
    g_pallas = jax.grad(
        lambda p, r, s: pallas_gb_ace_nl(p, q, r, nbr, mask, scaled_radii=s, box=box, interpret=True),
        argnums=(0, 1, 2),
    )(pos, rad, sr)
    g_ref = jax.grad(
        lambda p, r, s: gb_ace_nl_reference(p, q, r, s, mask, nbr, box6, has_box=True),
        argnums=(0, 1, 2),
    )(pos, rad, sr)
    np.testing.assert_allclose(g_pallas[0], g_ref[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(g_pallas[1], g_ref[1], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(g_pallas[2], g_ref[2], rtol=2e-3, atol=2e-3)

    fd_r = _fd_grad(
        lambda r: pallas_gb_ace_nl(pos, q, r, nbr, mask, scaled_radii=sr, box=box, interpret=True),
        rad,
    )
    np.testing.assert_allclose(np.asarray(g_pallas[1]), fd_r, rtol=1e-2, atol=1e-2)

    # Export context lowers cleanly to reference under PBC
    with export_context():
        exported = jax.export.export(
            jax.jit(lambda p: pallas_gb_ace_nl(p, q, rad, nbr, mask, scaled_radii=sr, box=box))
        )(pos)
        e_exp = pallas_gb_ace_nl(pos, q, rad, nbr, mask, scaled_radii=sr, box=box)
    hlo_text = str(exported.mlir_module())
    assert "triton" not in hlo_text.lower()
    np.testing.assert_allclose(float(e_exp), float(e_ref), rtol=1e-4, atol=1e-4)


def test_sasa_nl_pbc_energy_grads_and_export():
    pos, _, rad, _, mask, nbr, box, box6 = _straddling_pbc_system()

    e_pallas = pallas_sasa_nl(pos, rad, nbr, mask, box=box, interpret=True)
    e_ref = sasa_nl_reference(pos, rad, mask, nbr, box6, has_box=True)
    e_nobox = sasa_nl_reference(pos, rad, mask, nbr, box6, has_box=False)
    np.testing.assert_allclose(float(e_pallas), float(e_ref), rtol=1e-4, atol=1e-4)
    assert abs(float(e_ref) - float(e_nobox)) > 1e-2

    gp, gr = jax.grad(lambda p, r: pallas_sasa_nl(p, r, nbr, mask, box=box, interpret=True), argnums=(0, 1))(
        pos, rad
    )
    gp_ref, gr_ref = jax.grad(
        lambda p, r: sasa_nl_reference(p, r, mask, nbr, box6, has_box=True), argnums=(0, 1)
    )(pos, rad)
    np.testing.assert_allclose(gp, gp_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(gr, gr_ref, rtol=2e-3, atol=2e-3)

    with export_context():
        exported = jax.export.export(
            jax.jit(lambda p: pallas_sasa_nl(p, rad, nbr, mask, box=box))
        )(pos)
        e_exp = pallas_sasa_nl(pos, rad, nbr, mask, box=box)
    assert "triton" not in str(exported.mlir_module()).lower()
    np.testing.assert_allclose(float(e_exp), float(e_ref), rtol=1e-4, atol=1e-4)


def test_pallas_born_radii_single_masks_flags_and_self_pairs():
    pos, _, rad, sr, _, nbr, _, _ = _straddling_pbc_system()
    # Set 1-4 / full exclusion flag bits and an explicit self-pair on row 0
    flagged_nbr = nbr.at[0, 0].set(nbr[0, 0] | (1 << 29)).at[0, 3].set(0)
    br_pallas = _pallas_born_radii_single(pos, rad, sr, flagged_nbr, 0.09, interpret=True)
    br_ref = compute_born_radii_neighbor_list(pos, rad, flagged_nbr, 0.09, scaled_radii=sr)
    np.testing.assert_allclose(br_pallas, br_ref, rtol=1e-5, atol=1e-5)
