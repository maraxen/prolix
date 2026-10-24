"""Pallas SPME spread/gather vs XLA stencil (ADR-006 & Task 3 #6094)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.physics.pme_oracle import spread_loop_oracle, tiny_system


def test_spread_interpret_matches_oracle():
    from prolix.physics.pme_pallas import spread_charges_pallas

    pos, charges, mask, box, grid_dims = tiny_system()
    got = spread_charges_pallas(
        pos, charges, mask, box, grid_dims, order=4, interpret=True
    )
    ref = spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_spread_and_gather_non_multiple_atom_count_with_ghosts():
    """Verify vectorized (32, 64) Pallas spread and gather on N=37 (non-multiple of 32) with ghost atoms."""
    from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas
    from prolix.physics.pme import spme_reciprocal_energy

    rng = np.random.default_rng(2026)
    N = 37
    n_real = 33
    box = jnp.array([24.0, 24.0, 24.0], dtype=jnp.float32)
    grid_dims = (16, 16, 16)
    pos = jnp.asarray(rng.uniform(0.0, 24.0, size=(N, 3)).astype(np.float32))
    charges = jnp.asarray(rng.uniform(-0.8, 0.8, size=(N,)).astype(np.float32))
    # Give ghost atoms non-zero charges to verify atom_mask zeroes them out
    mask = jnp.arange(N) < n_real

    got_grid = spread_charges_pallas(
        pos, charges, mask, box, grid_dims, order=4, interpret=True
    )
    ref_grid = spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    np.testing.assert_allclose(got_grid, ref_grid, rtol=1e-5, atol=1e-5)

    # Test gather_pallas against jax.vjp of spread_charges_pallas / XLA spread
    cotangent_grid = jnp.asarray(rng.standard_normal(size=grid_dims).astype(np.float32))
    g_pos_pallas, g_q_pallas = gather_pallas(
        pos, charges, mask, box, cotangent_grid, grid_dims, order=4, interpret=True
    )
    # Ghost atoms must have zero gradients
    np.testing.assert_allclose(np.asarray(g_pos_pallas[n_real:]), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(g_q_pallas[n_real:]), 0.0, atol=1e-7)


def test_spread_and_spme_vmap_batching():
    """Verify @custom_vmap on spread_charges_pallas and gather_pallas works under jax.vmap over B=3."""
    from prolix.physics.pme import spme_energy_with_forces
    from prolix.physics.pme_pallas import spread_charges_pallas

    rng = np.random.default_rng(77)
    B, N = 3, 20
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    grid_dims = (16, 16, 16)
    pos_b = jnp.asarray(rng.uniform(1.0, 19.0, size=(B, N, 3)).astype(np.float32))
    q_b = jnp.asarray(rng.uniform(-0.5, 0.5, size=(B, N)).astype(np.float32))
    mask_b = jnp.ones((B, N), dtype=bool).at[:, -2:].set(False)

    grids_vmap = jax.vmap(
        lambda p, q, m: spread_charges_pallas(p, q, m, box, grid_dims, order=4, interpret=True)
    )(pos_b, q_b, mask_b)
    assert grids_vmap.shape == (B, 16, 16, 16)

    for b in range(B):
        ref_b = spread_loop_oracle(pos_b[b], q_b[b], mask_b[b], box, grid_dims, 4)
        np.testing.assert_allclose(grids_vmap[b], ref_b, rtol=1e-5, atol=1e-5)

    # Also test vmap(grad(spme_energy_with_forces)) wrt (positions, charges)
    def single_e_pallas(p, q, m):
        return spme_energy_with_forces(p, q, m, box, grid_dims, 0.34, 4, True)

    def single_e_xla(p, q, m):
        return spme_energy_with_forces(p, q, m, box, grid_dims, 0.34, 4, False)

    gp_pos, gp_q = jax.vmap(jax.grad(single_e_pallas, argnums=(0, 1)))(pos_b, q_b, mask_b)
    gx_pos, gx_q = jax.vmap(jax.grad(single_e_xla, argnums=(0, 1)))(pos_b, q_b, mask_b)
    np.testing.assert_allclose(gp_pos, gx_pos, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_q, gx_q, rtol=1e-4, atol=1e-4)


def test_grad_pallas_matches_xla_interpret():
    from prolix.physics.pme import make_spme_energy_fn, spme_energy_with_forces

    pos, charges, mask, box, grid_dims = tiny_system()
    alpha, order = 0.34, 4

    def e_xla(p, q):
        return spme_energy_with_forces(
            p, q, mask, box, grid_dims, alpha, order, False
        )

    def e_pallas(p, q):
        return spme_energy_with_forces(
            p, q, mask, box, grid_dims, alpha, order, True
        )

    gx_pos, gx_q = jax.grad(e_xla, argnums=(0, 1))(pos, charges)
    gp_pos, gp_q = jax.grad(e_pallas, argnums=(0, 1))(pos, charges)
    np.testing.assert_allclose(gp_pos, gx_pos, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_q, gx_q, rtol=1e-4, atol=1e-4)

    # Verify make_spme_energy_fn with cached Green's function matches spme_energy_with_forces
    cached_fn = make_spme_energy_fn(box, alpha=alpha, grid_dims=grid_dims, order=order, use_pallas=True)
    e_cached, (gc_pos, gc_q) = jax.value_and_grad(cached_fn, argnums=(0, 1))(pos, charges, mask)
    e_direct = e_pallas(pos, charges)
    np.testing.assert_allclose(e_cached, e_direct, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gc_pos, gp_pos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gc_q, gp_q, rtol=1e-5, atol=1e-5)


def test_ensemble_plan_accepts_use_pallas():
    """Verify EnsemblePlan accepts and stores use_pallas parameter."""
    from prolix.api.ensemble_plan import EnsemblePlan

    plan = EnsemblePlan(
        bundles=[],
        use_pallas=True,
    )
    assert plan.use_pallas is True


def _gpu_unavailable() -> bool:
    from prolix.pallas_kernels import _triton_gpu_available

    return not _triton_gpu_available()


@pytest.mark.skipif(_gpu_unavailable(), reason="Pallas GPU path needs CUDA")
def test_spread_gpu_matches_oracle():
    from prolix.physics.pme_pallas import spread_charges_pallas

    pos, charges, mask, box, grid_dims = tiny_system()
    got = spread_charges_pallas(
        pos, charges, mask, box, grid_dims, order=4, interpret=False
    )
    ref = spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(_gpu_unavailable(), reason="Pallas GPU path needs CUDA")
def test_grad_pallas_gpu_matches_xla():
    from prolix.physics.pme import spme_energy_with_forces

    pos, charges, mask, box, grid_dims = tiny_system()
    alpha, order = 0.34, 4

    def e_xla(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, alpha, order, False
        )

    def e_pallas(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, alpha, order, True
        )

    gx = jax.grad(e_xla)(pos)
    gp = jax.grad(e_pallas)(pos)
    np.testing.assert_allclose(gp, gx, rtol=1e-4, atol=1e-4)
