"""Tests for 1D prefix-sum ragged CTA work queue & cost-modeled multi-tier PME (MK-68)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.export import export_context
from prolix.physics.pallas_bonded import _triton_gpu_available
from prolix.physics.pallas_gbsa_tile import build_gbsa_tile_pair_list
from prolix.physics.pallas_ragged_queue import (
    build_ragged_tile_pair_queue,
    build_ragged_work_queue,
    gather_pallas_ragged,
    pallas_nonbonded_tile_pairs_ragged,
    plan_pme_tiers,
    spme_energy_and_forces_ragged_tiered,
    spme_energy_ragged_tiered,
    spme_forces_ragged_tiered,
    spread_charges_pallas_ragged,
)
from prolix.physics.pme import (
    _gather_reference,
    _spread_charges_reference,
    spme_reciprocal_energy,
    spme_self_energy,
)


def test_ragged_work_queue_bijection_zero_ghost_ctas_and_pytree_jit():
    """Verify 1D prefix-sum queue eliminates ghost CTAs on [1963..23558] and passes jit cleanly."""
    n_reals = [1963, 4199, 6658, 15002, 23558]
    q = build_ragged_work_queue(n_reals, tile_size=128, padded_size=25000)

    expected_blocks = [math.ceil(n / 128) for n in n_reals]
    assert q.total_blocks == sum(expected_blocks)
    uniform_blocks = len(n_reals) * math.ceil(25000 / 128)
    assert q.total_blocks < uniform_blocks // 2  # >50% ghost CTAs eliminated

    rep_np = np.asarray(q.cta_to_replica)
    blk_np = np.asarray(q.cta_to_local_block)
    reconstructed = list(zip(rep_np.tolist(), blk_np.tolist(), strict=True))
    expected_pairs = [(b, k) for b, nb in enumerate(expected_blocks) for k in range(nb)]
    assert reconstructed == expected_pairs

    # Verify Pytree dynamic LUT pass-through under jax.jit without recompiling when LUT values match shape
    traces = []

    @jax.jit
    def _consume_queue(queue):
        traces.append(1)
        return jnp.sum(queue.cta_to_replica) + jnp.sum(queue.cta_to_local_block)

    _ = _consume_queue(q)
    _ = _consume_queue(q)
    assert len(traces) == 1


def test_plan_pme_tiers_cost_model_auto_grouped_uniform():
    """Verify launch-vs-FFT DP heuristic collapses tiny savings and splits large cross-size tiers."""
    # Case 1: Tiny grid differences (32^3 vs 36^3, B=2) -> saved volume ~14k pts (< 1 us) << 15 us launch!
    boxes_small_diff = np.asarray([[30.0, 30.0, 30.0], [34.0, 34.0, 34.0]], dtype=np.float32)
    plan_auto_small = plan_pme_tiers(
        boxes_small_diff,
        grid_dims_per_replica=[(32, 32, 32), (36, 36, 36)],
        n_real_per_replica=[2000, 2600],
        padded_atoms=3000,
        pme_tier_mode="auto",
        tier_launch_overhead_us=15.0,
        fft_points_per_us=18000.0,
    )
    plan_grouped_small = plan_pme_tiers(
        boxes_small_diff,
        grid_dims_per_replica=[(32, 32, 32), (36, 36, 36)],
        n_real_per_replica=[2000, 2600],
        padded_atoms=3000,
        pme_tier_mode="grouped",
    )
    assert len(plan_auto_small.tiers) == 1
    assert plan_auto_small.tiers[0].grid_dims == (36, 36, 36)
    assert len(plan_grouped_small.tiers) == 2

    # Case 2: Cross-size batch (4x 1UAO @ 32^3 alongside 1x DHFR @ 80^3):
    # Uniform = 5 * 512,000 = 2,560,000 pts (~142 us); 2-tier = 4*32768 + 512000 = 643,072 pts (~35.7 us + 15 us)
    boxes_cross = np.asarray(
        [[30.0, 30.0, 30.0]] * 4 + [[75.0, 75.0, 75.0]], dtype=np.float32
    )
    grids_cross = [(32, 32, 32)] * 4 + [(80, 80, 80)]
    n_reals_cross = [1963, 1963, 2100, 2200, 23558]
    plan_auto_cross = plan_pme_tiers(
        boxes_cross,
        grid_dims_per_replica=grids_cross,
        n_real_per_replica=n_reals_cross,
        padded_atoms=24000,
        pme_tier_mode="auto",
    )
    plan_uniform_cross = plan_pme_tiers(
        boxes_cross,
        grid_dims_per_replica=grids_cross,
        n_real_per_replica=n_reals_cross,
        padded_atoms=24000,
        pme_tier_mode="uniform",
    )
    assert len(plan_auto_cross.tiers) == 2
    assert plan_auto_cross.estimated_tiered_us < plan_auto_cross.estimated_uniform_us
    assert len(plan_uniform_cross.tiers) == 1
    assert plan_uniform_cross.tiers[0].grid_dims == (80, 80, 80)


@pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not _triton_gpu_available(), reason="No Triton GPU"))],
)
def test_spread_and_gather_pallas_ragged_parity(interpret: bool):
    """Verify 1D ragged charge spreading and force/potential gathering vs pure-JAX reference."""
    rng = np.random.default_rng(123)
    B, N = 3, 140
    n_reals = [19, 65, 131]
    pos = jnp.asarray(rng.uniform(1.0, 29.0, size=(B, N, 3)), dtype=jnp.float32)
    charges = jnp.asarray(rng.uniform(-0.8, 0.8, size=(B, N)), dtype=jnp.float32)
    mask = np.zeros((B, N), dtype=np.int32)
    for b, nr in enumerate(n_reals):
        mask[b, :nr] = 1
    mask_j = jnp.asarray(mask)
    box = jnp.asarray([[30.0, 30.0, 30.0], [32.0, 32.0, 32.0], [35.0, 35.0, 35.0]], dtype=jnp.float32)
    grid_dims = (24, 24, 24)

    q_mesh_rag = spread_charges_pallas_ragged(
        pos, charges, mask_j, box, grid_dims,
        n_real_per_replica=n_reals, order=4, interpret=interpret,
    )
    for b in range(B):
        q_ref_b = _spread_charges_reference(pos[b], charges[b], mask_j[b], box[b], grid_dims=grid_dims, order=4)
        np.testing.assert_allclose(q_mesh_rag[b], q_ref_b, rtol=2e-5, atol=2e-5)

    theta = jnp.asarray(rng.normal(size=(B, *grid_dims)), dtype=jnp.float32)
    f_rag, dq_rag = gather_pallas_ragged(
        pos, charges, mask_j, box, theta, grid_dims,
        n_real_per_replica=n_reals, order=4,
        force_scale=-332.06371329919216, pot_scale=332.06371329919216,
        compute_dq=True, interpret=interpret,
    )
    for b, nr in enumerate(n_reals):
        f_ref_b, dq_ref_b = _gather_reference(
            pos[b], charges[b], mask_j[b], box[b], theta[b], grid_dims=grid_dims, order=4,
            force_scale=-332.06371329919216, pot_scale=332.06371329919216,
        )
        np.testing.assert_allclose(f_rag[b, :nr], f_ref_b[:nr], rtol=2e-5, atol=5e-4)
        np.testing.assert_allclose(dq_rag[b, :nr], dq_ref_b[:nr], rtol=2e-5, atol=5e-4)
        np.testing.assert_allclose(f_rag[b, nr:], 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "interpret",
    [True, pytest.param(False, marks=pytest.mark.skipif(not _triton_gpu_available(), reason="No Triton GPU"))],
)
def test_spme_ragged_tiered_parity_and_custom_vjp(interpret: bool):
    """Verify multi-tier SPME energy, forces, and custom_vjp wrt (positions, charges, box)."""
    rng = np.random.default_rng(77)
    B, N = 3, 48
    n_reals = [12, 26, 44]
    boxes = jnp.asarray([[22.0, 22.0, 22.0], [28.0, 28.0, 28.0], [36.0, 36.0, 36.0]], dtype=jnp.float32)
    grids = [(16, 16, 16), (20, 20, 20), (24, 24, 24)]
    alpha = 0.34

    pos_np = rng.uniform(1.0, 20.0, size=(B, N, 3)).astype(np.float32)
    q_np = rng.uniform(-0.5, 0.5, size=(B, N)).astype(np.float32)
    mask_np = np.zeros((B, N), dtype=np.int32)
    for b, nr in enumerate(n_reals):
        q_np[b, :nr] -= np.mean(q_np[b, :nr])
        q_np[b, nr:] = 0.0
        mask_np[b, :nr] = 1

    pos = jnp.asarray(pos_np)
    q = jnp.asarray(q_np)
    mask = jnp.asarray(mask_np)

    plan_grp = plan_pme_tiers(
        boxes, alpha=alpha, grid_dims_per_replica=grids,
        n_real_per_replica=n_reals, padded_atoms=N, pme_tier_mode="grouped",
    )
    assert len(plan_grp.tiers) == 3

    e_grp, f_grp = spme_energy_and_forces_ragged_tiered(
        pos, q, mask, boxes, tier_plan=plan_grp, interpret=interpret
    )
    f_direct = spme_forces_ragged_tiered(pos, q, mask, boxes, tier_plan=plan_grp, interpret=interpret)
    np.testing.assert_allclose(f_grp, f_direct, rtol=2e-5, atol=1e-4)

    for b, nr in enumerate(n_reals):
        def _ref_single(pb, b=b):
            er = spme_reciprocal_energy(pb, q[b], mask[b], boxes[b], grids[b], alpha, 4, False)
            es = spme_self_energy(q[b], mask[b], alpha)
            return er + es

        e_ref_b, g_ref_b = jax.value_and_grad(_ref_single)(pos[b])
        np.testing.assert_allclose(float(e_grp[b]), float(e_ref_b), rtol=5e-5, atol=1e-3)
        np.testing.assert_allclose(f_grp[b, :nr], -g_ref_b[:nr], rtol=1e-4, atol=2e-3)

    # Verify custom_vjp wrt positions, charges, and box_size
    g_pos, g_q, g_box = jax.grad(
        lambda p_, q_, bx_: jnp.sum(
            spme_energy_ragged_tiered(p_, q_, mask, bx_, tier_plan=plan_grp, interpret=interpret)
        ),
        argnums=(0, 1, 2),
    )(pos, q, boxes)

    g_pos_ref, g_q_ref, g_box_ref = jax.grad(
        lambda p_, q_, bx_: sum(
            spme_reciprocal_energy(p_[b], q_[b], mask[b], bx_[b], grids[b], alpha, 4, False)
            + spme_self_energy(q_[b], mask[b], alpha)
            for b in range(B)
        ),
        argnums=(0, 1, 2),
    )(pos, q, boxes)
    np.testing.assert_allclose(g_pos, g_pos_ref, rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(g_q, g_q_ref, rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(g_box, g_box_ref, rtol=1e-4, atol=2e-3)


def test_ragged_tile_pairs_nonbonded_vjp_and_export_context():
    """Verify 1D ragged nonbonded tile-pair kernel + custom_vjp + StableHLO export fallback."""
    rng = np.random.default_rng(31)
    B, N = 2, 42
    n_reals = [20, 38]
    # Well-separated grid coordinates to avoid repulsive LJ blowup
    grid_pts = np.asarray(
        [[3.2 * (i % 4), 3.2 * ((i // 4) % 4), 3.2 * (i // 16)] for i in range(N)],
        dtype=np.float32,
    )
    pos = jnp.asarray(np.stack([grid_pts + rng.uniform(-0.2, 0.2, grid_pts.shape) for _ in range(B)]).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.4, 0.4, size=(B, N)), dtype=jnp.float32)
    sig = jnp.asarray(rng.uniform(2.9, 3.4, size=(B, N)), dtype=jnp.float32)
    eps = jnp.asarray(rng.uniform(0.05, 0.15, size=(B, N)), dtype=jnp.float32)
    mask_np = np.zeros((B, N), dtype=np.int32)
    for b, nr in enumerate(n_reals):
        mask_np[b, :nr] = 1
    mask = jnp.asarray(mask_np)
    box = jnp.asarray([[30.0, 30.0, 30.0], [32.0, 32.0, 32.0]], dtype=jnp.float32)

    tl = build_gbsa_tile_pair_list(pos, atom_mask=mask, n_real_per_replica=n_reals, idx_12_13=jnp.asarray([[0, 1]]), idx_14=jnp.asarray([[0, 3]]))
    pq = build_ragged_tile_pair_queue(tl.num_valid_pairs, padded_pairs=tl.tile_pairs.shape[1], tp_group=1)

    e_nb, f_nb = pallas_nonbonded_tile_pairs_ragged(
        pos, q, sig, eps, mask, tl.tile_pairs, tl.excl_full_mask, tl.excl_14_mask,
        pair_queue=pq, box=box, cutoff=9.0, pme_alpha=0.34,
        include_pme_excl_erf=True, return_forces=True, interpret=True,
    )
    grads_pallas = jax.grad(
        lambda p_, q_, s_, ep_: jnp.sum(
            pallas_nonbonded_tile_pairs_ragged(
                p_, q_, s_, ep_, mask, tl.tile_pairs, tl.excl_full_mask, tl.excl_14_mask,
                pair_queue=pq, box=box, cutoff=9.0, pme_alpha=0.34,
                include_pme_excl_erf=True, interpret=True,
            )
        ),
        argnums=(0, 1, 2, 3),
    )(pos, q, sig, eps)

    with export_context():
        e_exp, f_exp = pallas_nonbonded_tile_pairs_ragged(
            pos, q, sig, eps, mask, tl.tile_pairs, tl.excl_full_mask, tl.excl_14_mask,
            pair_queue=pq, box=box, cutoff=9.0, pme_alpha=0.34,
            include_pme_excl_erf=True, return_forces=True,
        )
        grads_exp = jax.grad(
            lambda p_, q_, s_, ep_: jnp.sum(
                pallas_nonbonded_tile_pairs_ragged(
                    p_, q_, s_, ep_, mask, tl.tile_pairs, tl.excl_full_mask, tl.excl_14_mask,
                    pair_queue=pq, box=box, cutoff=9.0, pme_alpha=0.34,
                    include_pme_excl_erf=True,
                )
            ),
            argnums=(0, 1, 2, 3),
        )(pos, q, sig, eps)
        hlo = jax.jit(
            lambda p_: spme_energy_ragged_tiered(p_, q, mask, box, alpha=0.34, n_real_per_replica=n_reals)
            + pallas_nonbonded_tile_pairs_ragged(
                p_, q, sig, eps, mask, tl.tile_pairs, tl.excl_full_mask, tl.excl_14_mask,
                pair_queue=pq, box=box, cutoff=9.0, pme_alpha=0.34,
            )
        ).lower(pos).as_text().lower()
        for bad in ("triton", "pallas", "mosaic", "tpu_custom_call", "xla.gpu"):
            assert bad not in hlo

    np.testing.assert_allclose(e_nb, e_exp, rtol=2e-5, atol=5e-4)
    np.testing.assert_allclose(f_nb, f_exp, rtol=5e-5, atol=5e-4)
    for g_pal, g_ref in zip(grads_pallas, grads_exp, strict=True):
        np.testing.assert_allclose(g_pal, g_ref, rtol=5e-5, atol=5e-4)
