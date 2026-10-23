"""Tests for Task 5 (#6096): Pallas nonbonded kernels (pallas_nonbonded_nl, pallas_flash_explicit_nonbonded, pallas_born_radii)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.pallas_kernels import (
    _triton_gpu_available,
    pallas_born_radii,
    pallas_flash_explicit_nonbonded,
    pallas_nonbonded_nl,
)
from prolix.physics.flash_explicit import chunked_explicit_nonbonded_energy
from prolix.physics.generalized_born import compute_born_radii_neighbor_list
from prolix.physics.neighbor_list import ExclusionSpec, map_exclusions_to_dense_padded
from prolix.physics.optimization import (
    chunked_coulomb_energy_nl,
    chunked_lj_energy_nl,
)
from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas
from prolix.utils import topology


def test_pallas_nonbonded_nl_forward_grad_and_vmap():
    """Verify pallas_nonbonded_nl matches chunked_lj_energy_nl + chunked_coulomb_energy_nl on N=37, K=35."""
    rng = np.random.default_rng(20261023)
    N = 37  # non-multiple of 32
    n_real = 34
    K = 35  # non-multiple of 32
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    displacement_fn, _ = space.periodic(box)

    pos = jnp.asarray(rng.uniform(1.0, 24.0, size=(N, 3)).astype(np.float32))
    sigma = jnp.asarray(rng.uniform(2.5, 3.4, size=(N,)).astype(np.float32))
    eps = jnp.asarray(rng.uniform(0.05, 0.2, size=(N,)).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.6, 0.6, size=(N,)).astype(np.float32))
    mask = jnp.arange(N) < n_real

    bonds = np.array([[0, 1], [1, 2], [2, 3], [5, 6], [10, 11]], dtype=np.int32)
    excl = topology.find_bonded_exclusions(bonds, N)
    excl_spec = ExclusionSpec(
        idx_12_13=jnp.concatenate([excl.idx_12, excl.idx_13], axis=0),
        idx_14=excl.idx_14,
        scale_14_elec=0.83333333,
        scale_14_vdw=0.5,
        n_atoms=N,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=jnp.float32),
        exception_epsilons=jnp.zeros((0,), dtype=jnp.float32),
        exception_chargeprods=jnp.zeros((0,), dtype=jnp.float32),
    )
    excl_indices, excl_vdw, excl_elec = map_exclusions_to_dense_padded(excl_spec, max_exclusions=32)

    nbr_np = np.full((N, K), N, dtype=np.int32)
    for i in range(n_real):
        col = 0
        for j in range(n_real):
            if i != j and col < K:
                nbr_np[i, col] = j
                col += 1
    nbr_idx = jnp.asarray(nbr_np)

    cutoff = 9.0
    sw_width = 1.5
    pme_alpha = 0.34

    # Zero out ghost atoms' epsilons and charges for exact comparison with chunked_*_nl
    eps_masked = jnp.where(mask, eps, 0.0)
    q_masked = jnp.where(mask, q, 0.0)

    def ref_fn(p, ch, s, ep):
        e_lj = chunked_lj_energy_nl(
            p, s, ep, excl_indices, excl_vdw, nbr_idx, displacement_fn,
            cutoff=cutoff, tile_size=16, switch_width=sw_width,
        )
        e_coul = chunked_coulomb_energy_nl(
            p, ch, excl_indices, excl_elec, nbr_idx, displacement_fn,
            pme_alpha=pme_alpha, cutoff=cutoff, tile_size=16,
        )
        return e_lj + e_coul

    interp = not _triton_gpu_available()

    def pallas_fn(p, ch, s, ep):
        return pallas_nonbonded_nl(
            p, ch, s, ep, mask, nbr_idx, excl_indices, excl_vdw, excl_elec,
            box=box, pme_alpha=pme_alpha, cutoff=cutoff, lj_switch_width=sw_width,
            soft_core_lambda=1.0, include_pme_excl_erf=True, interpret=interp,
        )

    val_p, grads_p = jax.value_and_grad(pallas_fn, argnums=(0, 1, 2, 3))(
        pos, q_masked, sigma, eps_masked
    )
    val_r, grads_r = jax.value_and_grad(ref_fn, argnums=(0, 1, 2, 3))(
        pos, q_masked, sigma, eps_masked
    )

    np.testing.assert_allclose(val_p, val_r, rtol=1e-4, atol=1e-4)
    for idx, name in enumerate(["pos", "q", "sigma", "eps"]):
        np.testing.assert_allclose(
            grads_p[idx][:n_real], grads_r[idx][:n_real], rtol=1e-4, atol=1e-4,
            err_msg=f"pallas_nonbonded_nl grad mismatch wrt {name}",
        )

    # Test jax.vmap over B=2 conformations
    pos_b = jnp.stack([pos, pos + 0.1], axis=0)
    vals_vmap = jax.vmap(lambda p: pallas_fn(p, q_masked, sigma, eps_masked))(pos_b)
    vals_ref_vmap = jax.vmap(lambda p: ref_fn(p, q_masked, sigma, eps_masked))(pos_b)
    np.testing.assert_allclose(vals_vmap, vals_ref_vmap, rtol=1e-4, atol=1e-4)


def test_pallas_flash_explicit_nonbonded_forward_grad_and_vmap():
    """Verify pallas_flash_explicit_nonbonded matches chunked_explicit_nonbonded_energy on PME and EFA modes."""
    from tests.physics.test_xtrax_pallas_boundary import _make_minimal_periodic_padded_system

    sys = _make_minimal_periodic_padded_system(n_real=6, n_padded=8)

    for mode, include_elec in [("pme", True), ("efa", True), ("lj_only", False)]:
        alpha = float(sys.pme_alpha) if mode == "pme" else 0.0
        q_in = sys.charges if include_elec else jnp.zeros_like(sys.charges)

        def ref_e(p):
            s = sys.replace(positions=p)
            return chunked_explicit_nonbonded_energy(
                p, q_in, s.sigmas, s.epsilons, s.atom_mask, s,
                T=8, pme_alpha=alpha, use_pallas=False
            )

        def pallas_e(p):
            s = sys.replace(positions=p)
            return chunked_explicit_nonbonded_energy(
                p, q_in, s.sigmas, s.epsilons, s.atom_mask, s,
                T=8, pme_alpha=alpha, use_pallas=True
            )

        val_r, grad_r = jax.value_and_grad(ref_e)(sys.positions)
        val_p, grad_p = jax.value_and_grad(pallas_e)(sys.positions)

        np.testing.assert_allclose(val_p, val_r, rtol=1e-4, atol=1e-4, err_msg=f"Energy mismatch for mode={mode}")
        np.testing.assert_allclose(grad_p, grad_r, rtol=1e-4, atol=1e-4, err_msg=f"Grad mismatch for mode={mode}")

    # Test jax.vmap over B=2 conformations on pallas_flash_explicit_nonbonded directly
    interp = not _triton_gpu_available()
    box = jnp.asarray(sys.box_size, dtype=jnp.float32)
    pos_b = jnp.stack([sys.positions, sys.positions + 0.05], axis=0)
    e_vmap = jax.vmap(
        lambda p: pallas_flash_explicit_nonbonded(
            p, sys.charges, sys.sigmas, sys.epsilons, sys.atom_mask,
            sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec,
            box=box, pme_alpha=float(sys.pme_alpha), interpret=interp,
        )
    )(pos_b)
    assert e_vmap.shape == (2,)
    assert jnp.all(jnp.isfinite(e_vmap))


def test_pallas_born_radii_forward_grad_and_vmap():
    """Verify vectorized (32, 32) pallas_born_radii matches compute_born_radii_neighbor_list forward, jax.grad, and jax.vmap."""
    rng = np.random.default_rng(888)
    N = 35  # non-multiple of 32
    K = 34  # non-multiple of 32
    pos = jnp.asarray(rng.uniform(0.0, 12.0, size=(N, 3)).astype(np.float32))
    radii = jnp.asarray(rng.uniform(1.2, 1.8, size=(N,)).astype(np.float32))
    scaled_radii = radii * 0.85

    nbr_np = np.full((N, K), N, dtype=np.int32)
    for i in range(N):
        c = 0
        for j in range(N):
            if i != j and c < K:
                nbr_np[i, c] = j
                c += 1
    nbr_idx = jnp.asarray(nbr_np)
    interp = not _triton_gpu_available()

    br_pallas = pallas_born_radii(
        pos, radii, nbr_idx, dielectric_offset=0.09, scaled_radii=scaled_radii, interpret=interp
    )
    br_ref = compute_born_radii_neighbor_list(
        pos, radii, nbr_idx, dielectric_offset=0.09, scaled_radii=scaled_radii
    )
    np.testing.assert_allclose(br_pallas, br_ref, rtol=1e-4, atol=1e-4)

    # Verify custom_vjp gradient wrt (positions, radii, scaled_radii)
    weights = jnp.asarray(rng.standard_normal(N).astype(np.float32))
    grads_p = jax.grad(
        lambda p, r, sr: jnp.sum(
            pallas_born_radii(p, r, nbr_idx, dielectric_offset=0.09, scaled_radii=sr, interpret=interp) * weights
        ),
        argnums=(0, 1, 2),
    )(pos, radii, scaled_radii)
    grads_r = jax.grad(
        lambda p, r, sr: jnp.sum(
            compute_born_radii_neighbor_list(p, r, nbr_idx, dielectric_offset=0.09, scaled_radii=sr) * weights
        ),
        argnums=(0, 1, 2),
    )(pos, radii, scaled_radii)
    for idx, name in enumerate(["pos", "radii", "scaled_radii"]):
        np.testing.assert_allclose(grads_p[idx], grads_r[idx], rtol=1e-4, atol=1e-4, err_msg=f"Grad mismatch wrt {name}")

    # Verify custom_vmap over B=2 conformations
    pos_b = jnp.stack([pos, pos + 0.1], axis=0)
    br_vmap = jax.vmap(
        lambda p: pallas_born_radii(p, radii, nbr_idx, dielectric_offset=0.09, scaled_radii=scaled_radii, interpret=interp)
    )(pos_b)
    br_ref_vmap = jax.vmap(
        lambda p: compute_born_radii_neighbor_list(p, radii, nbr_idx, dielectric_offset=0.09, scaled_radii=scaled_radii)
    )(pos_b)
    np.testing.assert_allclose(br_vmap, br_ref_vmap, rtol=1e-4, atol=1e-4)


def test_all_pallas_kernels_lower_to_cuda_triton(monkeypatch):
    """Verify all 5 Pallas kernels lower cleanly to CUDA Triton MLIR without Mosaic GPU or primitive errors."""
    pytest.importorskip("jaxlib.triton")
    monkeypatch.setattr(jax, "local_devices", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("mock sm90")))

    pos = jnp.ones((32, 3), dtype=jnp.float32)
    q = jnp.ones((32,), dtype=jnp.float32)
    sig = jnp.ones((32,), dtype=jnp.float32)
    eps = jnp.ones((32,), dtype=jnp.float32)
    mask = jnp.ones((32,), dtype=jnp.bool_)
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    theta = jnp.ones((16, 16, 16), dtype=jnp.float32)
    nbr = jnp.zeros((32, 32), dtype=jnp.int32)
    excl_idx = jnp.full((32, 8), -1, dtype=jnp.int32)
    excl_s = jnp.ones((32, 8), dtype=jnp.float32)

    jax.jit(lambda p, c, m, b: spread_charges_pallas(p, c, m, b, (16, 16, 16), interpret=False)).trace(
        pos, q, mask, box
    ).lower(lowering_platforms=("cuda",))
    jax.jit(lambda p, c, m, b, t: gather_pallas(p, c, m, b, t, (16, 16, 16), interpret=False)).trace(
        pos, q, mask, box, theta
    ).lower(lowering_platforms=("cuda",))
    jax.jit(lambda p, r, n: pallas_born_radii(p, r, n, interpret=False)).trace(
        pos, sig, nbr
    ).lower(lowering_platforms=("cuda",))
    jax.jit(
        lambda p, c, s, e, m, n, ei, ev, ee, b: pallas_nonbonded_nl(
            p, c, s, e, m, n, ei, ev, ee, box=b, pme_alpha=0.34, interpret=False
        )
    ).trace(pos, q, sig, eps, mask, nbr, excl_idx, excl_s, excl_s, box).lower(lowering_platforms=("cuda",))
    jax.jit(
        lambda p, c, s, e, m, ei, ev, ee, b: pallas_flash_explicit_nonbonded(
            p, c, s, e, m, ei, ev, ee, box=b, pme_alpha=0.34, interpret=False
        )
    ).trace(pos, q, sig, eps, mask, excl_idx, excl_s, excl_s, box).lower(lowering_platforms=("cuda",))

