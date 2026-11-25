"""Tests for 2-pass 32x32 symmetric tile-pair OBC2 GBSA + MK-73 custom_vjp (MK-67, MK-73)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.export import export_context
from prolix.physics.generalized_born import (
    GB_COULOMB_CONSTANT,
    SURFACE_TENSION,
    compute_ace_nonpolar_energy,
    compute_debye_kappa,
    compute_gb_energy,
)
from prolix.physics.pallas_bonded import _triton_gpu_available
from prolix.physics.pallas_gbsa_tile import (
    build_gbsa_tile_pair_list,
    gbsa_tile_reference,
    pallas_gbsa_tile,
    pallas_gbsa_tile_energy_and_forces,
)


def _make_synthetic_gbsa_system(n_atoms: int = 70, n_real: int = 65, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Place atoms on a well-separated 3D grid + jitter so min distance > 2.2 A
    side = math.ceil(n_atoms ** (1.0 / 3.0))
    coords = []
    for idx in range(n_atoms):
        ix = idx % side
        iy = (idx // side) % side
        iz = idx // (side * side)
        coords.append([3.1 * ix, 3.1 * iy, 3.1 * iz])
    pos = np.asarray(coords, dtype=np.float64) + rng.uniform(-0.35, 0.35, size=(n_atoms, 3))
    charges = rng.uniform(-0.65, 0.65, size=(n_atoms,))
    charges -= np.mean(charges[:n_real])
    radii = rng.uniform(1.35, 1.95, size=(n_atoms,))
    scaled_radii = radii * rng.uniform(0.78, 0.88, size=(n_atoms,))
    sigmas = rng.uniform(2.8, 3.6, size=(n_atoms,))
    epsilons = rng.uniform(0.04, 0.18, size=(n_atoms,))
    mask = np.zeros((n_atoms,), dtype=np.int32)
    mask[:n_real] = 1

    # Include lane-31 boundary exclusions (0, 31), (31, 63) to exercise uint32 bit 31
    idx_12_13 = np.asarray(
        [[i, i + 1] for i in range(0, min(n_real - 1, 50), 2)]
        + ([[0, 31], [31, min(63, n_real - 1)]] if n_real > 32 else []),
        dtype=np.int32,
    )
    idx_14 = np.asarray(
        [[i, i + 3] for i in range(0, min(n_real - 3, 48), 3)],
        dtype=np.int32,
    )
    return pos, charges, radii, scaled_radii, sigmas, epsilons, mask, idx_12_13, idx_14


def test_bug_gb1_exact_akma_constants_and_isolated_sphere():
    """Verify BUG-GB1 exact AKMA OpenMM surfaceAreaEnergy (0.0054 kcal/mol/A^2) and Coulomb."""
    assert math.isclose(SURFACE_TENSION * 418.4, 2.25936, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(GB_COULOMB_CONSTANT, 332.06371329919216, rel_tol=0.0, abs_tol=1e-12)

    pos = jnp.zeros((1, 3), dtype=jnp.float32)
    q = jnp.zeros((1,), dtype=jnp.float32)
    r = jnp.asarray([1.7], dtype=jnp.float32)
    sr = jnp.asarray([1.36], dtype=jnp.float32)
    # Isolated uncharged atom with dielectric_offset=0.0 -> Born radius R_0 == r_0 == 1.7 A
    e_tile = pallas_gbsa_tile(
        pos, q, r, sr, dielectric_offset=0.0, probe_radius=1.4,
        include_coulomb=False, include_lj=False, interpret=True,
    )
    e_ref = compute_ace_nonpolar_energy(r, r, probe_radius=1.4)
    expected = 4.0 * math.pi * 0.0054 * ((1.7 + 1.4) ** 2)
    np.testing.assert_allclose(float(e_tile), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(jnp.sum(e_ref)), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not _triton_gpu_available(), reason="No Triton GPU"))],
)
def test_gbsa_tile_parity_vs_generalized_born_and_reference(interpret: bool):
    """Verify Born radii, energy, and forces across non-multiple-of-32 + masked atoms + exclusions."""
    pos, q, r, sr, sig, eps, mask, i12, i14 = _make_synthetic_gbsa_system(n_atoms=70, n_real=65)
    pos32 = jnp.asarray(pos, dtype=jnp.float32)
    q32 = jnp.asarray(q, dtype=jnp.float32)
    r32 = jnp.asarray(r, dtype=jnp.float32)
    sr32 = jnp.asarray(sr, dtype=jnp.float32)
    sig32 = jnp.asarray(sig, dtype=jnp.float32)
    eps32 = jnp.asarray(eps, dtype=jnp.float32)
    m32 = jnp.asarray(mask, dtype=jnp.int32)
    tl = build_gbsa_tile_pair_list(
        pos32, atom_mask=m32, idx_12_13=jnp.asarray(i12), idx_14=jnp.asarray(i14)
    )

    # 1. Pure GB + ACE parity vs generalized_born.py
    e_gb_tile, f_gb_tile, br_tile = pallas_gbsa_tile(
        pos32, q32, r32, sr32, sig32, eps32, tl,
        atom_mask=m32, include_coulomb=False, include_lj=False,
        return_forces=True, interpret=interpret,
    )

    def _gb_ace_ref_fn(p):
        e_gb, br = compute_gb_energy(p, q32, r32, mask=m32 > 0, scaled_radii=sr32)
        e_ace = jnp.sum(compute_ace_nonpolar_energy(r32, br, mask=m32 > 0))
        return e_gb + e_ace, br

    (e_gb_ref, br_ref), g_gb_ref = jax.value_and_grad(_gb_ace_ref_fn, has_aux=True)(pos32)
    e_gb_slice, br_slice = compute_gb_energy(pos32[:65], q32[:65], r32[:65], scaled_radii=sr32[:65])
    e_slice_tot = e_gb_slice + jnp.sum(compute_ace_nonpolar_energy(r32[:65], br_slice))
    np.testing.assert_allclose(float(e_gb_ref), float(e_slice_tot), rtol=2e-6, atol=1e-4)
    np.testing.assert_allclose(br_tile[:65], br_ref[:65], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(float(e_gb_tile), float(e_gb_ref), rtol=5e-5, atol=5e-4)
    np.testing.assert_allclose(f_gb_tile[:65], -g_gb_ref[:65], rtol=1e-4, atol=5e-4)
    np.testing.assert_allclose(f_gb_tile[65:], 0.0, atol=1e-7)

    # 2. Full fused GB + ACE + Coulomb + LJ + 1-2/1-3/1-4 exclusions parity
    e_full, f_full = pallas_gbsa_tile_energy_and_forces(
        pos32, q32, r32, sr32, sig32, eps32, tl, atom_mask=m32, interpret=interpret
    )
    e_ref_full, f_ref_full, _ = gbsa_tile_reference(
        pos32, q32, r32, sr32, sig32, eps32, tl, atom_mask=m32, return_forces=True
    )
    np.testing.assert_allclose(float(e_full), float(e_ref_full), rtol=5e-5, atol=1e-3)
    np.testing.assert_allclose(f_full, f_ref_full, rtol=1e-4, atol=1e-3)


def test_bug_gb2_debye_huckel_kappa_and_obc1_params():
    """Verify BUG-GB2 salt screening (kappa > 0) and custom obc_alpha/beta/gamma."""
    pos, q, r, sr, _sig, _eps, mask, _, _ = _make_synthetic_gbsa_system(n_atoms=40, n_real=37, seed=7)
    p32, q32, r32, sr32 = (jnp.asarray(x, dtype=jnp.float32) for x in (pos, q, r, sr))
    m32 = jnp.asarray(mask, dtype=jnp.int32)
    kappa = compute_debye_kappa(0.15, solvent_dielectric=78.5, temperature_k=300.0)
    assert kappa > 0.1

    e_tile, f_tile, _ = pallas_gbsa_tile(
        p32, q32, r32, sr32, atom_mask=m32,
        kappa=kappa, obc_alpha=0.8, obc_beta=0.0, obc_gamma=2.909125,
        include_coulomb=False, include_lj=False, return_forces=True, interpret=True,
    )

    def _ref_screened(p):
        egb, br = compute_gb_energy(
            p, q32, r32, mask=m32 > 0, scaled_radii=sr32,
            kappa=kappa, obc_alpha=0.8, obc_beta=0.0, obc_gamma=2.909125,
        )
        return egb + jnp.sum(compute_ace_nonpolar_energy(r32, br, mask=m32 > 0))

    e_ref, g_ref = jax.value_and_grad(_ref_screened)(p32)
    np.testing.assert_allclose(float(e_tile), float(e_ref), rtol=5e-5, atol=5e-4)
    np.testing.assert_allclose(f_tile[:37], -g_ref[:37], rtol=1e-4, atol=5e-4)


def test_morton_cutoff_and_ragged_multi_replica_batch_and_vmap():
    """Verify Morton spatial tile-pair pruning, 1D ragged multi-replica batching, and jax.vmap."""
    pos0, q0, r0, sr0, sig0, eps0, m0, i12, i14 = _make_synthetic_gbsa_system(80, 72, seed=11)
    # Morton cutoff mode
    tl_morton = build_gbsa_tile_pair_list(
        jnp.asarray(pos0, dtype=jnp.float32),
        atom_mask=jnp.asarray(m0),
        idx_12_13=jnp.asarray(i12),
        idx_14=jnp.asarray(i14),
        cutoff=12.0,
        use_morton=True,
    )
    assert tl_morton.atom_perm is not None
    e_m, f_m, _ = pallas_gbsa_tile(
        jnp.asarray(pos0, dtype=jnp.float32),
        jnp.asarray(q0, dtype=jnp.float32),
        jnp.asarray(r0, dtype=jnp.float32),
        jnp.asarray(sr0, dtype=jnp.float32),
        jnp.asarray(sig0, dtype=jnp.float32),
        jnp.asarray(eps0, dtype=jnp.float32),
        tl_morton,
        atom_mask=jnp.asarray(m0),
        cutoff=12.0,
        switch_dist=10.0,
        return_forces=True,
        interpret=True,
    )
    e_m_ref, f_m_ref, _ = gbsa_tile_reference(
        jnp.asarray(pos0, dtype=jnp.float32),
        jnp.asarray(q0, dtype=jnp.float32),
        jnp.asarray(r0, dtype=jnp.float32),
        jnp.asarray(sr0, dtype=jnp.float32),
        jnp.asarray(sig0, dtype=jnp.float32),
        jnp.asarray(eps0, dtype=jnp.float32),
        tl_morton,
        atom_mask=jnp.asarray(m0),
        cutoff=12.0,
        switch_dist=10.0,
        return_forces=True,
    )
    np.testing.assert_allclose(float(e_m), float(e_m_ref), rtol=2e-5, atol=5e-4)
    np.testing.assert_allclose(f_m, f_m_ref, rtol=5e-5, atol=5e-4)

    # 1D ragged batch [18, 42, 72] in bucket N=80 -> eliminates ghost CTAs
    n_reals = [18, 42, 72]
    pos_b = np.stack([pos0 + 0.1 * b for b in range(3)], axis=0).astype(np.float32)
    msk_b = np.zeros((3, 80), dtype=np.int32)
    for b, nr in enumerate(n_reals):
        msk_b[b, :nr] = 1
    tl_rag = build_gbsa_tile_pair_list(
        jnp.asarray(pos_b), atom_mask=jnp.asarray(msk_b), n_real_per_replica=n_reals
    )
    assert tl_rag.pair_queue is not None
    assert tl_rag.pair_queue.total_blocks == (1 + 3 + 6)  # 1, 2, 3 active 32-atom blocks -> 1, 3, 6 pairs!

    e_rag, f_rag, br_rag = pallas_gbsa_tile(
        jnp.asarray(pos_b),
        jnp.asarray(q0, dtype=jnp.float32),
        jnp.asarray(r0, dtype=jnp.float32),
        jnp.asarray(sr0, dtype=jnp.float32),
        jnp.asarray(sig0, dtype=jnp.float32),
        jnp.asarray(eps0, dtype=jnp.float32),
        tl_rag,
        atom_mask=jnp.asarray(msk_b),
        return_forces=True,
        interpret=True,
    )
    for b, nr in enumerate(n_reals):
        tl_single = build_gbsa_tile_pair_list(
            jnp.asarray(pos_b[b]), atom_mask=jnp.asarray(msk_b[b]), n_active_atoms=nr
        )
        eb, fb, brb = pallas_gbsa_tile(
            jnp.asarray(pos_b[b]),
            jnp.asarray(q0, dtype=jnp.float32),
            jnp.asarray(r0, dtype=jnp.float32),
            jnp.asarray(sr0, dtype=jnp.float32),
            jnp.asarray(sig0, dtype=jnp.float32),
            jnp.asarray(eps0, dtype=jnp.float32),
            tl_single,
            atom_mask=jnp.asarray(msk_b[b]),
            return_forces=True,
            interpret=True,
        )
        np.testing.assert_allclose(float(e_rag[b]), float(eb), rtol=2e-5, atol=5e-4)
        np.testing.assert_allclose(f_rag[b, :nr], fb[:nr], rtol=5e-5, atol=5e-4)
        np.testing.assert_allclose(br_rag[b, :nr], brb[:nr], rtol=2e-5, atol=2e-5)

    # Also verify jax.vmap over shared tile_list
    tl_shared = build_gbsa_tile_pair_list(jnp.asarray(pos0, dtype=jnp.float32))
    e_vmap = jax.vmap(
        lambda p, m: pallas_gbsa_tile(
            p,
            jnp.asarray(q0, dtype=jnp.float32),
            jnp.asarray(r0, dtype=jnp.float32),
            jnp.asarray(sr0, dtype=jnp.float32),
            jnp.asarray(sig0, dtype=jnp.float32),
            jnp.asarray(eps0, dtype=jnp.float32),
            tl_shared,
            atom_mask=m,
            interpret=True,
        )
    )(jnp.asarray(pos_b), jnp.asarray(msk_b))
    np.testing.assert_allclose(e_vmap, e_rag, rtol=2e-5, atol=5e-4)


@pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not _triton_gpu_available(), reason="No Triton GPU"))],
)
def test_mk73_custom_vjp_all_six_params_vs_float64_fd(interpret: bool):
    """Verify MK-73 tile-pair custom_vjp wrt all 6 arrays vs float64 central finite differences."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        pos, q, r, sr, sig, eps, mask, i12, i14 = _make_synthetic_gbsa_system(38, 35, seed=99)
        m_j = jnp.asarray(mask, dtype=jnp.int32)
        tl = build_gbsa_tile_pair_list(
            jnp.asarray(pos, dtype=jnp.float32),
            atom_mask=m_j,
            idx_12_13=jnp.asarray(i12),
            idx_14=jnp.asarray(i14),
        )
        kappa = compute_debye_kappa(0.1)

        in_dtype = jnp.float64 if interpret else jnp.float32
        args = tuple(jnp.asarray(x, dtype=in_dtype) for x in (pos, q, r, sr, sig, eps))

        def _loss(p_, q_, r_, sr_, sig_, eps_):
            return pallas_gbsa_tile(
                p_, q_, r_, sr_, sig_, eps_, tl,
                atom_mask=m_j, kappa=kappa, softcore_lambda=0.85, interpret=interpret,
            )

        grads = jax.grad(_loss, argnums=(0, 1, 2, 3, 4, 5))(*args)

        # Float64 central finite-difference oracle via gbsa_tile_reference
        p64, q64, r64, sr64, sig64, eps64 = (jnp.asarray(x, dtype=jnp.float64) for x in (pos, q, r, sr, sig, eps))

        def _ref64(p_, q_, r_, sr_, sig_, eps_):
            return float(
                gbsa_tile_reference(
                    p_, q_, r_, sr_, sig_, eps_, tl,
                    atom_mask=m_j, kappa=kappa, softcore_lambda=0.85,
                )
            )

        h = 1e-6
        all_64 = [np.array(x, dtype=np.float64) for x in (p64, q64, r64, sr64, sig64, eps64)]
        for k, g_arr in enumerate(grads):
            g_np = np.asarray(g_arr, dtype=np.float64)
            # Ghost atoms (index >= 35) must receive exact zero gradient
            np.testing.assert_allclose(g_np[35:], 0.0, atol=1e-12)
            fd_np = np.zeros_like(g_np)
            it = np.ndindex(all_64[k].shape)
            for idx in it:
                orig = all_64[k][idx]
                all_64[k][idx] = orig + h
                ep = _ref64(*(jnp.asarray(a) for a in all_64))
                all_64[k][idx] = orig - h
                em = _ref64(*(jnp.asarray(a) for a in all_64))
                all_64[k][idx] = orig
                fd_np[idx] = (ep - em) / (2.0 * h)
            rtol = 1e-7 if interpret else 1e-5
            atol = 1e-7 if interpret else 5e-5
            np.testing.assert_allclose(g_np, fd_np, rtol=rtol, atol=atol)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


def test_stablehlo_export_context_contains_zero_custom_calls():
    """Verify export_context() lowers forward and 6-param grad with zero Triton/Pallas targets."""
    pos, q, r, sr, sig, eps, mask, i12, i14 = _make_synthetic_gbsa_system(36, 34, seed=5)
    args = tuple(jnp.asarray(x, dtype=jnp.float32) for x in (pos, q, r, sr, sig, eps))
    m32 = jnp.asarray(mask, dtype=jnp.int32)
    tl = build_gbsa_tile_pair_list(args[0], atom_mask=m32, idx_12_13=jnp.asarray(i12), idx_14=jnp.asarray(i14))

    forbidden = ("triton", "pallas", "mosaic", "tpu_custom_call", "xla.gpu", "cusolver", "cublas", "cudnn")
    with export_context():
        fn = jax.jit(
            lambda p_, q_, r_, sr_, sg_, ep_: jax.value_and_grad(
                lambda a, b, c, d, e, f: pallas_gbsa_tile(a, b, c, d, e, f, tl, atom_mask=m32),
                argnums=(0, 1, 2, 3, 4, 5),
            )(p_, q_, r_, sr_, sg_, ep_)
        )
        hlo_text = fn.lower(*args).as_text().lower()
        for bad in forbidden:
            assert bad not in hlo_text, f"Forbidden target {bad!r} found in StableHLO export"


def test_build_gbsa_tile_pair_list_under_jit_and_vmap():
    """Verify build_gbsa_tile_pair_list and tile_list=None work cleanly inside jit/vmap."""
    pos, q, r, sr, sig, eps, mask, i12, i14 = _make_synthetic_gbsa_system(40, 36, seed=21)
    q32, r32, sr32, sig32, eps32 = (jnp.asarray(x, dtype=jnp.float32) for x in (q, r, sr, sig, eps))
    m32 = jnp.asarray(mask, dtype=jnp.int32)
    i12_j, i14_j = jnp.asarray(i12, dtype=jnp.int32), jnp.asarray(i14, dtype=jnp.int32)
    pos_b = jnp.asarray(np.stack([pos, pos + 0.05], axis=0), dtype=jnp.float32)
    m_b = jnp.stack([m32, m32], axis=0)

    # 1. Host concrete arrays passed to build_gbsa_tile_pair_list inside jit + vmap (EnsemblePlan path)
    @jax.jit
    def _vmap_host_tl(pb):
        def _one(p):
            tl_h = build_gbsa_tile_pair_list(
                np.asarray(pos, dtype=np.float32),
                atom_mask=np.asarray(mask, dtype=np.int32),
                idx_12_13=i12,
                idx_14=i14,
            )
            return pallas_gbsa_tile(
                p, q32, r32, sr32, sig32, eps32, tl_h, atom_mask=m32, interpret=True
            )

        return jax.vmap(_one)(pb)

    e_host_tl = _vmap_host_tl(pos_b)

    # 2. Fully traced positions, atom_mask, and exclusions with tile_list=None inside jit + vmap
    @jax.jit
    def _vmap_traced_tl(pb, mb, ex12, ex14):
        return jax.vmap(
            lambda p, m: pallas_gbsa_tile(
                p, q32, r32, sr32, sig32, eps32,
                atom_mask=m, idx_12_13=ex12, idx_14=ex14, interpret=True,
            )
        )(pb, mb)

    e_traced_tl = _vmap_traced_tl(pos_b, m_b, i12_j, i14_j)
    np.testing.assert_allclose(e_host_tl, e_traced_tl, rtol=1e-6, atol=1e-5)
