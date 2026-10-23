"""Tests for Task 2 (#6093): custom_vjp gradients parity, GB cotangents, binary-search exclusions, and NL Coulomb."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.batched_energy import (
    single_padded_energy,
    single_padded_energy_nl,
    single_padded_energy_nl_cvjp,
)
from prolix.pallas_kernels import (
    gb_coulomb_energy_dense,
    gb_energy_split_cutoff,
)
from prolix.physics.neighbor_list import (
    ExclusionSpec,
    get_neighbor_exclusion_scales,
    map_exclusions_to_dense_padded,
)
from prolix.physics.optimization import (
    chunked_coulomb_energy,
    chunked_coulomb_energy_nl,
    chunked_lj_energy,
    chunked_lj_energy_nl,
    _lj_switch_weight_and_deriv,
)
from prolix.utils import topology


def _bonds_to_dense_exclusions(bonds, n_atoms: int, max_exclusions: int = 8):
    excl = topology.find_bonded_exclusions(bonds, n_atoms)
    spec = ExclusionSpec(
        idx_12_13=jnp.concatenate([excl.idx_12, excl.idx_13], axis=0),
        idx_14=excl.idx_14,
        scale_14_elec=0.83333333,
        scale_14_vdw=0.5,
        n_atoms=n_atoms,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=jnp.float32),
        exception_epsilons=jnp.zeros((0,), dtype=jnp.float32),
        exception_chargeprods=jnp.zeros((0,), dtype=jnp.float32),
    )
    return map_exclusions_to_dense_padded(spec, max_exclusions=max_exclusions)


def _make_nonbonded_inputs(n_atoms: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = jnp.asarray(rng.uniform(0.0, 6.0, size=(n_atoms, 3)).astype(np.float32))
    sigma = jnp.asarray(rng.uniform(2.5, 3.5, size=(n_atoms,)).astype(np.float32))
    eps = jnp.asarray(rng.uniform(0.05, 0.25, size=(n_atoms,)).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.8, 0.8, size=(n_atoms,)).astype(np.float32))

    bonds = np.array([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=np.int32)
    excl_indices, excl_vdw, excl_elec = _bonds_to_dense_exclusions(bonds, n_atoms, max_exclusions=8)

    # Create neighbor list (include all other atoms plus padded slot = n_atoms)
    K = n_atoms
    nbr_idx = np.full((n_atoms, K), n_atoms, dtype=np.int32)
    for i in range(n_atoms):
        col = 0
        for j in range(n_atoms):
            if i != j:
                nbr_idx[i, col] = j
                col += 1
    return pos, sigma, eps, q, excl_indices, excl_vdw, excl_elec, jnp.asarray(nbr_idx)


def _pure_jax_lj_dense(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, cutoff=9.0):
    """Exact pure-JAX differentiable reference for chunked_lj_energy."""
    N = r.shape[0]
    dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(r, r)
    ds = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-30)
    idx_i = jnp.arange(N)[:, None]
    idx_j = jnp.arange(N)[None, :]
    matches = (idx_j[:, None, :] == excl_indices[:, :, None]) & (excl_indices[:, :, None] >= 0)
    pair_scale = jnp.sum(jnp.where(matches, excl_scales[:, :, None], 0.0), axis=1)
    pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
    s_ij = 0.5 * (sigmas[None, :] + sigmas[:, None])
    e_ij = jnp.sqrt(jnp.maximum(epsilons[None, :] * epsilons[:, None], 0.0) + 1e-30)
    m = (idx_j != idx_i) & (ds < cutoff) & (ds > 1e-12)
    inv_ds = jnp.where(m, 1.0 / ds, 0.0)
    inv_r6 = (s_ij * inv_ds) ** 6
    energy = jnp.where(m, 4.0 * e_ij * (inv_r6 ** 2 - inv_r6), 0.0)
    return 0.5 * jnp.sum(energy * pair_scale)


def _pure_jax_lj_nl(r, sigmas, epsilons, excl_indices, excl_scales, neighbor_idx, displacement_fn, cutoff=9.0, switch_width=0.0):
    """Exact pure-JAX differentiable reference for chunked_lj_energy_nl."""
    N, K = neighbor_idx.shape
    valid_nb = (neighbor_idx >= 0) & (neighbor_idx < N)
    safe_nb = jnp.where(valid_nb, neighbor_idx, 0)
    pos_j = r[safe_nb]
    dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(r, pos_j)
    ds = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-30)
    matches = (neighbor_idx[:, None, :] == excl_indices[:, :, None]) & (excl_indices[:, :, None] >= 0)
    pair_scale = jnp.sum(jnp.where(matches, excl_scales[:, :, None], 0.0), axis=1)
    pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
    s_ij = 0.5 * (sigmas[:, None] + sigmas[safe_nb])
    e_ij = jnp.sqrt(jnp.maximum(epsilons[:, None] * epsilons[safe_nb], 0.0) + 1e-30)
    idx_i = jnp.arange(N)[:, None]
    m = valid_nb & (neighbor_idx != idx_i) & (ds < cutoff) & (ds > 1e-12)
    inv_ds = jnp.where(m, 1.0 / ds, 0.0)
    inv_r6 = (s_ij * inv_ds) ** 6
    s_w, _ = _lj_switch_weight_and_deriv(ds, cutoff, switch_width)
    energy = jnp.where(m, 4.0 * e_ij * (inv_r6 ** 2 - inv_r6) * s_w, 0.0)
    return 0.5 * jnp.sum(energy * pair_scale)


def _pure_jax_coulomb_dense(r, charges, excl_indices, excl_scales, displacement_fn, pme_alpha=0.34, coulomb_constant=332.0637, cutoff=9.0):
    """Exact pure-JAX differentiable reference for chunked_coulomb_energy."""
    N = r.shape[0]
    dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(r, r)
    ds = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-30)
    idx_i = jnp.arange(N)[:, None]
    idx_j = jnp.arange(N)[None, :]
    matches = (idx_j[:, None, :] == excl_indices[:, :, None]) & (excl_indices[:, :, None] >= 0)
    pair_scale = jnp.sum(jnp.where(matches, excl_scales[:, :, None], 0.0), axis=1)
    pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
    q_ij = charges[:, None] * charges[None, :]
    m = (idx_j != idx_i) & (ds < cutoff) & (ds > 1e-12)
    inv_ds = jnp.where(m, 1.0 / ds, 0.0)
    e_pair = coulomb_constant * q_ij * inv_ds * (pair_scale - jax.scipy.special.erf(pme_alpha * ds))
    return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))


def _pure_jax_coulomb_nl(r, charges, excl_indices, excl_scales, neighbor_idx, displacement_fn, pme_alpha=0.34, coulomb_constant=332.0637, cutoff=9.0):
    """Exact pure-JAX differentiable reference for chunked_coulomb_energy_nl."""
    N, K = neighbor_idx.shape
    valid_nb = (neighbor_idx >= 0) & (neighbor_idx < N)
    safe_nb = jnp.where(valid_nb, neighbor_idx, 0)
    pos_j = r[safe_nb]
    dr = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, 0))(r, pos_j)
    ds = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-30)
    matches = (neighbor_idx[:, None, :] == excl_indices[:, :, None]) & (excl_indices[:, :, None] >= 0)
    pair_scale = jnp.sum(jnp.where(matches, excl_scales[:, :, None], 0.0), axis=1)
    pair_scale = jnp.where(jnp.any(matches, axis=1), pair_scale, 1.0)
    q_ij = charges[:, None] * charges[safe_nb]
    idx_i = jnp.arange(N)[:, None]
    m = valid_nb & (neighbor_idx != idx_i) & (ds < cutoff) & (ds > 1e-12)
    inv_ds = jnp.where(m, 1.0 / ds, 0.0)
    e_pair = coulomb_constant * q_ij * inv_ds * (pair_scale - jax.scipy.special.erf(pme_alpha * ds))
    return 0.5 * jnp.sum(jnp.where(m, e_pair, 0.0))


def test_chunked_lj_nl_gradients_match_autodiff():
    """Verify chunked_lj_energy_nl custom_vjp matches autodiff wrt positions, sigma, eps, and excl_scales (with switching)."""
    pos, sigma, eps, _, excl_indices, excl_vdw, _, nbr_idx = _make_nonbonded_inputs(n_atoms=8)
    displacement_fn, _ = space.free()
    cutoff = 6.5
    sw_width = 2.0

    val_cvjp, grads_cvjp = jax.value_and_grad(
        lambda r, s, ep, es: chunked_lj_energy_nl(
            r, s, ep, excl_indices, es, nbr_idx, displacement_fn, cutoff=cutoff, tile_size=4, switch_width=sw_width
        ),
        argnums=(0, 1, 2, 3),
    )(pos, sigma, eps, excl_vdw)

    val_ref, grads_ref = jax.value_and_grad(
        lambda r, s, ep, es: _pure_jax_lj_nl(
            r, s, ep, excl_indices, es, nbr_idx, displacement_fn, cutoff=cutoff, switch_width=sw_width
        ),
        argnums=(0, 1, 2, 3),
    )(pos, sigma, eps, excl_vdw)

    np.testing.assert_allclose(val_cvjp, val_ref, rtol=1e-5, atol=1e-5)
    assert float(jnp.linalg.norm(grads_cvjp[0])) > 1e-3
    for idx, name in enumerate(["pos", "sigma", "eps", "excl_vdw"]):
        np.testing.assert_allclose(
            grads_cvjp[idx], grads_ref[idx], rtol=1e-4, atol=1e-5, err_msg=f"Mismatch in LJ NL grad wrt {name}"
        )


def test_chunked_coulomb_nl_gradients_match_autodiff():
    """Verify chunked_coulomb_energy_nl custom_vjp matches autodiff wrt positions, q, and excl_scales."""
    pos, _, _, q, excl_indices, _, excl_elec, nbr_idx = _make_nonbonded_inputs(n_atoms=8)
    displacement_fn, _ = space.free()
    cutoff = 6.5
    pme_alpha = 0.34

    val_cvjp, grads_cvjp = jax.value_and_grad(
        lambda r, ch, es: chunked_coulomb_energy_nl(
            r, ch, excl_indices, es, nbr_idx, displacement_fn, pme_alpha=pme_alpha, cutoff=cutoff, tile_size=4
        ),
        argnums=(0, 1, 2),
    )(pos, q, excl_elec)

    val_ref, grads_ref = jax.value_and_grad(
        lambda r, ch, es: _pure_jax_coulomb_nl(
            r, ch, excl_indices, es, nbr_idx, displacement_fn, pme_alpha=pme_alpha, cutoff=cutoff
        ),
        argnums=(0, 1, 2),
    )(pos, q, excl_elec)

    np.testing.assert_allclose(val_cvjp, val_ref, rtol=1e-5, atol=1e-5)
    assert float(jnp.linalg.norm(grads_cvjp[0])) > 1e-2
    for idx, name in enumerate(["pos", "q", "excl_elec"]):
        np.testing.assert_allclose(
            grads_cvjp[idx], grads_ref[idx], rtol=1e-4, atol=1e-4, err_msg=f"Mismatch in Coulomb NL grad wrt {name}"
        )


def test_chunked_dense_lj_and_coulomb_gradients_match_autodiff():
    """Verify dense chunked_lj_energy and chunked_coulomb_energy custom_vjp match autodiff across all diff args."""
    pos, sigma, eps, q, excl_indices, excl_vdw, excl_elec, _ = _make_nonbonded_inputs(n_atoms=8)
    displacement_fn, _ = space.free()
    cutoff = 6.5
    pme_alpha = 0.34

    val_lj, grads_lj = jax.value_and_grad(
        lambda r, s, ep, es: chunked_lj_energy(
            r, s, ep, excl_indices, es, displacement_fn, cutoff=cutoff, tile_size=4
        ),
        argnums=(0, 1, 2, 3),
    )(pos, sigma, eps, excl_vdw)

    val_lj_ref, grads_lj_ref = jax.value_and_grad(
        lambda r, s, ep, es: _pure_jax_lj_dense(
            r, s, ep, excl_indices, es, displacement_fn, cutoff=cutoff
        ),
        argnums=(0, 1, 2, 3),
    )(pos, sigma, eps, excl_vdw)

    np.testing.assert_allclose(val_lj, val_lj_ref, rtol=1e-5, atol=1e-5)
    for idx, name in enumerate(["pos", "sigma", "eps", "excl_vdw"]):
        np.testing.assert_allclose(
            grads_lj[idx], grads_lj_ref[idx], rtol=1e-4, atol=1e-5, err_msg=f"Mismatch in dense LJ grad wrt {name}"
        )

    val_coul, grads_coul = jax.value_and_grad(
        lambda r, ch, es: chunked_coulomb_energy(
            r, ch, excl_indices, es, displacement_fn, pme_alpha=pme_alpha, cutoff=cutoff, tile_size=4
        ),
        argnums=(0, 1, 2),
    )(pos, q, excl_elec)

    val_coul_ref, grads_coul_ref = jax.value_and_grad(
        lambda r, ch, es: _pure_jax_coulomb_dense(
            r, ch, excl_indices, es, displacement_fn, pme_alpha=pme_alpha, cutoff=cutoff
        ),
        argnums=(0, 1, 2),
    )(pos, q, excl_elec)

    np.testing.assert_allclose(val_coul, val_coul_ref, rtol=1e-5, atol=1e-5)
    for idx, name in enumerate(["pos", "q", "excl_elec"]):
        np.testing.assert_allclose(
            grads_coul[idx], grads_coul_ref[idx], rtol=1e-4, atol=1e-4, err_msg=f"Mismatch in dense Coulomb grad wrt {name}"
        )


def test_gb_coulomb_dense_and_split_cutoff_gradients_match_exact_autodiff():
    """Verify gb_coulomb_energy_dense (no factor-of-2 bug; returns charges & born_radii grads) and gb_energy_split_cutoff."""
    from prolix.physics.generalized_born import compute_born_radii, compute_born_radii_neighbor_list

    rng = np.random.default_rng(7)
    n = 6
    pos = jnp.asarray(rng.uniform(0.0, 5.0, size=(n, 3)).astype(np.float32))
    charges = jnp.asarray(np.array([-0.5, 0.3, 0.2, -0.4, 0.4, 0.0], dtype=np.float32))
    radii = jnp.asarray(np.array([1.5, 1.2, 1.7, 1.5, 1.2, 1.5], dtype=np.float32))
    scaled_radii = radii * 0.85
    mask = jnp.array([True, True, True, True, True, False])

    born_radii = compute_born_radii(pos, radii, mask=mask, scaled_radii=scaled_radii)

    def _ref_gb_coulomb(p, q, br):
        dr = p[:, None, :] - p[None, :, :]
        dist_sq = jnp.sum(dr ** 2, axis=-1)
        bb = br[:, None] * br[None, :]
        exp_term = jnp.exp(-dist_sq / (4.0 * jnp.maximum(bb, 1e-12)))
        f_gb = jnp.sqrt(dist_sq + bb * exp_term + 1e-12)
        tau = (1.0 / 1.0) - (1.0 / 78.5)
        e_pair = -0.5 * 332.0637 * tau * (q[:, None] * q[None, :]) / f_gb
        m2d = (mask[:, None] & mask[None, :]).astype(jnp.float32)
        return jnp.sum(e_pair * m2d)

    val_got, grads_got = jax.value_and_grad(gb_coulomb_energy_dense, argnums=(0, 1, 2))(
        pos, charges, born_radii, mask
    )
    val_ref, grads_ref = jax.value_and_grad(_ref_gb_coulomb, argnums=(0, 1, 2))(
        pos, charges, born_radii
    )

    np.testing.assert_allclose(val_got, val_ref, rtol=1e-5, atol=1e-5)
    for idx, name in enumerate(["positions", "charges", "born_radii"]):
        np.testing.assert_allclose(
            grads_got[idx], grads_ref[idx], rtol=1e-4, atol=1e-4, err_msg=f"GB Coulomb grad mismatch wrt {name}"
        )

    # Neighbor list containing all real pairs for split-cutoff test
    nbr_np = np.full((n, n), n, dtype=np.int32)
    for i in range(n - 1):
        c = 0
        for j in range(n - 1):
            if i != j:
                nbr_np[i, c] = j
                c += 1
    nbr_idx = jnp.asarray(nbr_np)

    def _ref_full_gb(p, q):
        br = compute_born_radii_neighbor_list(
            p, radii, nbr_idx, dielectric_offset=0.09, scaled_radii=scaled_radii
        )
        return _ref_gb_coulomb(p, q, br)

    e_split, g_split = jax.value_and_grad(
        lambda p, q: gb_energy_split_cutoff(
            p, q, radii, mask, nbr_idx, scaled_radii=scaled_radii, dielectric_offset=0.09, use_pallas=False
        ),
        argnums=(0, 1),
    )(pos, charges)
    e_full_ref, g_full_ref = jax.value_and_grad(_ref_full_gb, argnums=(0, 1))(pos, charges)

    np.testing.assert_allclose(e_split, e_full_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_split[0], g_full_ref[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(g_split[1], g_full_ref[1], rtol=1e-4, atol=1e-4)


def test_binary_search_exclusion_scales_matches_reference():
    """Verify 5-step binary search in get_neighbor_exclusion_scales matches linear scan for all exclusions & sentinels."""
    n_atoms = 16
    bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [5, 10], [7, 15]], dtype=np.int32)
    excl_indices, excl_vdw, excl_elec = _bonds_to_dense_exclusions(bonds, n_atoms, max_exclusions=32)

    rng = np.random.default_rng(99)
    K = 12
    neighbor_idx = jnp.asarray(rng.integers(0, n_atoms + 1, size=(n_atoms, K), dtype=np.int32))

    c_vdw, c_elec = get_neighbor_exclusion_scales(excl_indices, excl_vdw, excl_elec, neighbor_idx)

    def _linear_ref(e_idx, e_v, e_e, nbr):
        N, K_dim = nbr.shape
        out_v = np.ones((N, K_dim), dtype=np.float32)
        out_e = np.ones((N, K_dim), dtype=np.float32)
        for i in range(N):
            for k in range(K_dim):
                nb = int(nbr[i, k])
                if 0 <= nb < N:
                    for m in range(e_idx.shape[1]):
                        if int(e_idx[i, m]) == nb:
                            out_v[i, k] = float(e_v[i, m])
                            out_e[i, k] = float(e_e[i, m])
                            break
        return out_v, out_e

    ref_v, ref_e = _linear_ref(
        np.asarray(excl_indices), np.asarray(excl_vdw), np.asarray(excl_elec), np.asarray(neighbor_idx)
    )
    np.testing.assert_allclose(np.asarray(c_vdw), ref_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(c_elec), ref_e, rtol=1e-6, atol=1e-6)


def test_single_padded_energy_nl_coulomb_parity():
    """Verify O(N*K) neighbor-list Coulomb in batched_energy matches dense masked Coulomb when all pairs in NL."""
    import dataclasses
    from tests.physics.test_xtrax_pallas_boundary import _make_minimal_periodic_padded_system

    sys = _make_minimal_periodic_padded_system(n_real=6, n_padded=8)
    sys_coul = dataclasses.replace(sys, box_size=None, radii=None, pme_alpha=0.0)
    displacement_fn, _ = space.free()

    N = sys_coul.n_padded_atoms
    nbr = np.full((N, N), N, dtype=np.int32)
    for i in range(N):
        c = 0
        for j in range(N):
            if i != j:
                nbr[i, c] = j
                c += 1
    nbr_idx = jnp.asarray(nbr)

    e_dense, g_dense = jax.value_and_grad(
        lambda p: single_padded_energy(sys_coul.replace(positions=p), displacement_fn, implicit_solvent=False)
    )(sys_coul.positions)

    e_nl, g_nl = jax.value_and_grad(
        lambda p: single_padded_energy_nl(
            sys_coul.replace(positions=p), nbr_idx, displacement_fn, implicit_solvent=False
        )
    )(sys_coul.positions)

    e_nl_cvjp, g_nl_cvjp = jax.value_and_grad(
        lambda p: single_padded_energy_nl_cvjp(
            sys_coul.replace(positions=p), nbr_idx, displacement_fn, implicit_solvent=False
        )
    )(sys_coul.positions)

    np.testing.assert_allclose(e_nl, e_dense, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(e_nl_cvjp, e_dense, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_nl, g_dense, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(g_nl_cvjp, g_dense, rtol=1e-4, atol=1e-4)
