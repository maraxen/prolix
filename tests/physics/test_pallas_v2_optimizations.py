"""Verification suite for OpenMM-Referenced Pallas v2 Optimizations (O1–O5) & Bonded Megakernel.

Tests:
- O1.1: Pre-encoded exclusion bits in neighbor list (`encode_nl_exclusions`, `excl_mode="pre_encoded"`,
  atom-index >= 2^29 guardrail).
- O1.2: Sorted exclusions + `[excl_min_i, excl_max_i]` bounding-interval fast skip in `"table"` mode.
- O2.1: Active-AD gating (`symbolic_zeros=True`) verifying zero parameter residuals when differentiating
  only w.r.t. `positions`, exact parameter gradients under `jax.grad(f)`, `jax.jit(jax.grad(f))`, and
  `jax.grad(jax.jit(f))`, plus `return_forces=True` direct forward force bypass.
- O2.2: `pallas_bonded_all_fused` single-launch megakernel and `make_bonded_all_fused_op` (`xtrax.op`
  `assert_kernel_ran` and `assert_kernel_parity`) across all 6 bonded groups + partial-group dead-branch
  pruning + `jax.vmap`.
- O3.1: Bekker 2-cross-product plane-normal torsion angle & force projection (`b0_is_ji=True` and `False`).
- O3.2: Cross-product harmonic angle force projection (`legacy_denom_eps` vs `openmm_cross`).
- O3.3: Horner 4-tuple CMAP bicubic spline evaluation.
- O4.1: Interior-block vs boundary-straddling periodic coordinate wrapping in `pallas_nonbonded_nl`.
- O5: 3D Morton (Z-order) spatial atom sorting (`compute_morton_codes_3d`,
  `permute_neighbor_list_morton`, `permute_and_sort_exclusions`, `atom_perm`/`atom_inv_perm` in
  `pallas_nonbonded_nl`, `spread_charges_pallas`, and `gather_pallas`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.intent import Context
from xtrax.testing.kernel_parity import assert_kernel_parity, assert_kernel_ran

from prolix.pallas_kernels import (
    bonded_all_fused_reference,
    compute_morton_codes_3d,
    encode_nl_exclusions,
    make_bonded_all_fused_op,
    pallas_angles_harmonic,
    pallas_bonded_all_fused,
    pallas_bonds_2body,
    pallas_cmap_torsions,
    pallas_exceptions_14,
    pallas_nonbonded_nl,
    pallas_torsions_4body,
    permute_and_sort_exclusions,
    permute_neighbor_list_morton,
)
from prolix.physics.cmap import precompute_cmap_coefficients
from prolix.physics.pallas_bonded import _pallas_bonded_all_fused_diff, _pallas_bonds_2body_diff
from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas


# ==============================================================================
# O1.1 & O1.2: Neighbor-List Exclusion Bit-Encoding & Bounding-Interval Skip
# ==============================================================================


def test_o1_pre_encoded_nl_exclusions_and_sorted_table_parity():
    """Verify O1.1 (`excl_mode="pre_encoded"`) and O1.2 (sorted table skip) match reference."""
    rng = np.random.default_rng(2026)
    N = 32
    K = 12
    E = 6
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)

    pos = jnp.asarray(rng.uniform(1.0, 24.0, size=(N, 3)).astype(np.float32))
    charges = jnp.asarray(rng.uniform(-0.8, 0.8, size=(N,)).astype(np.float32))
    sigmas = jnp.asarray(rng.uniform(2.5, 3.8, size=(N,)).astype(np.float32))
    epsilons = jnp.asarray(rng.uniform(0.05, 0.25, size=(N,)).astype(np.float32))
    atom_mask = jnp.ones((N,), dtype=jnp.bool_)

    # Build neighbor list and exclusions with both full exclusions (scale=0) and 1-4 scaled (0.5, 0.8333)
    neighbor_idx = np.full((N, K), N, dtype=np.int32)
    excl_idx = np.full((N, E), -1, dtype=np.int32)
    excl_vdw = np.ones((N, E), dtype=np.float32)
    excl_elec = np.ones((N, E), dtype=np.float32)

    for i in range(N):
        nbrs = [(i + d) % N for d in range(1, K + 1)]
        neighbor_idx[i, : len(nbrs)] = nbrs
        # Exclusions: j1 = (i+1)%N (full exclusion), j2 = (i+2)%N (full), j3 = (i+3)%N (1-4 scaled)
        excl_idx[i, 0] = (i + 3) % N
        excl_vdw[i, 0] = 0.5
        excl_elec[i, 0] = 0.8333333
        excl_idx[i, 1] = (i + 1) % N
        excl_vdw[i, 1] = 0.0
        excl_elec[i, 1] = 0.0
        excl_idx[i, 2] = (i + 2) % N
        excl_vdw[i, 2] = 0.0
        excl_elec[i, 2] = 0.0

    neighbor_idx_j = jnp.asarray(neighbor_idx)
    excl_idx_unsorted = jnp.asarray(excl_idx)
    excl_vdw_unsorted = jnp.asarray(excl_vdw)
    excl_elec_unsorted = jnp.asarray(excl_elec)

    # Sort exclusions per row (O1.2)
    perm_ident = jnp.arange(N, dtype=jnp.int32)
    excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted = permute_and_sort_exclusions(
        excl_idx_unsorted, excl_vdw_unsorted, excl_elec_unsorted, perm_ident, perm_ident
    )

    # Pre-encode exclusions into neighbor_idx high bits (O1.1)
    neighbor_idx_encoded = encode_nl_exclusions(
        neighbor_idx_j, excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted, n_atoms=N
    )

    def eval_mode(nb_mat, ex_i, ex_v, ex_e, mode):
        return pallas_nonbonded_nl(
            pos,
            charges,
            sigmas,
            epsilons,
            atom_mask,
            nb_mat,
            ex_i,
            ex_v,
            ex_e,
            box=box,
            pme_alpha=0.34,
            cutoff=10.0,
            lj_switch_width=1.0,
            include_pme_excl_erf=True,
            excl_mode=mode,
            scale_14_vdw=0.5,
            scale_14_elec=0.8333333,
            interpret=True,
        )

    e_unsorted, g_unsorted = jax.value_and_grad(
        lambda p: pallas_nonbonded_nl(
            p, charges, sigmas, epsilons, atom_mask, neighbor_idx_j,
            excl_idx_unsorted, excl_vdw_unsorted, excl_elec_unsorted,
            box=box, pme_alpha=0.34, cutoff=10.0, lj_switch_width=1.0,
            include_pme_excl_erf=True, excl_mode="table", interpret=True,
        )
    )(pos)

    e_sorted, g_sorted = jax.value_and_grad(
        lambda p: pallas_nonbonded_nl(
            p, charges, sigmas, epsilons, atom_mask, neighbor_idx_j,
            excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted,
            box=box, pme_alpha=0.34, cutoff=10.0, lj_switch_width=1.0,
            include_pme_excl_erf=True, excl_mode="table", interpret=True,
        )
    )(pos)

    e_encoded, g_encoded = jax.value_and_grad(
        lambda p: pallas_nonbonded_nl(
            p, charges, sigmas, epsilons, atom_mask, neighbor_idx_encoded,
            excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted,
            box=box, pme_alpha=0.34, cutoff=10.0, lj_switch_width=1.0,
            include_pme_excl_erf=True, excl_mode="pre_encoded",
            scale_14_vdw=0.5, scale_14_elec=0.8333333, interpret=True,
        )
    )(pos)

    np.testing.assert_allclose(e_sorted, e_unsorted, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_sorted, g_unsorted, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(e_encoded, e_unsorted, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_encoded, g_unsorted, rtol=1e-5, atol=1e-5)

    # Guardrail: n_atoms >= 2^29 must raise ValueError
    with pytest.raises(ValueError, match="2\\^29"):
        encode_nl_exclusions(
            neighbor_idx_j, excl_idx_sorted, excl_vdw_sorted, excl_elec_sorted, n_atoms=1 << 29
        )


# ==============================================================================
# O2.1: Active-AD Gating (`symbolic_zeros=True`) & Direct Forward Force Bypass
# ==============================================================================


def test_o2_1_active_ad_gating_and_return_forces():
    """Verify O2.1 active-AD gating skips saving parameter buffers when unperturbed."""
    rng = np.random.default_rng(42)
    N = 16
    pos = jnp.asarray(rng.uniform(1.0, 10.0, size=(N, 3)).astype(np.float32))
    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=jnp.int32)
    bond_params = jnp.array([[1.5, 300.0], [1.4, 320.0], [1.55, 280.0], [1.3, 400.0]], dtype=jnp.float32)
    bond_mask = jnp.array([True, True, True, False], dtype=jnp.bool_)
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)

    # 1. Direct inspection of custom_vjp forward residuals when compute_param_grads=False
    from prolix.physics.pallas_bonded import _pallas_bonds_2body_fwd

    _, res_pos_only = _pallas_bonds_2body_fwd(
        pos, bonds, bond_params, bond_mask.astype(jnp.float32), box, True, 1e-10, True, False
    )
    assert res_pos_only[-1] is None, "Expected param_res to be None when parameters are unperturbed"

    # 2. Verify jax.grad w.r.t. positions only and w.r.t. (positions, bond_params) under jit/grad
    fn_pos = lambda p: pallas_bonds_2body(p, bonds, bond_params, bond_mask, box=box, interpret=True)
    fn_both = lambda p, bp: pallas_bonds_2body(p, bonds, bp, bond_mask, box=box, interpret=True)

    g_pos_eager = jax.grad(fn_pos)(pos)
    g_pos_jit_grad = jax.jit(jax.grad(fn_pos))(pos)
    g_pos_grad_jit = jax.grad(jax.jit(fn_pos))(pos)

    np.testing.assert_allclose(g_pos_jit_grad, g_pos_eager, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g_pos_grad_jit, g_pos_eager, rtol=1e-6, atol=1e-6)

    (g_p1, g_bp1) = jax.grad(fn_both, argnums=(0, 1))(pos, bond_params)
    (g_p2, g_bp2) = jax.jit(jax.grad(fn_both, argnums=(0, 1)))(pos, bond_params)
    (g_p3, g_bp3) = jax.grad(jax.jit(fn_both), argnums=(0, 1))(pos, bond_params)

    np.testing.assert_allclose(g_p1, g_pos_eager, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g_bp2, g_bp1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g_bp3, g_bp1, rtol=1e-6, atol=1e-6)

    # 3. Direct forward force return (`return_forces=True`) matches `-jax.grad(energy)(pos)`
    e_fwd, f_fwd = pallas_bonds_2body(
        pos, bonds, bond_params, bond_mask, box=box, return_forces=True, interpret=True
    )
    np.testing.assert_allclose(e_fwd, fn_pos(pos), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f_fwd, -g_pos_eager, rtol=1e-6, atol=1e-6)


# ==============================================================================
# O2.2: Fused Bonded Megakernel (`pallas_bonded_all_fused` & `make_bonded_all_fused_op`)
# ==============================================================================


def test_o2_2_bonded_all_fused_megakernel_parity_and_xtrax_op():
    """Verify O2.2 single-launch bonded megakernel matches sum of 6 individual kernels + xtrax.op."""
    rng = np.random.default_rng(1337)
    N = 24
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)
    pos = jnp.asarray(rng.uniform(2.0, 28.0, size=(N, 3)).astype(np.float32))

    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [4, 5], [6, 7]], dtype=jnp.int32)
    bond_params = jnp.asarray(
        np.column_stack([rng.uniform(1.2, 1.6, size=(5,)), rng.uniform(250.0, 450.0, size=(5,))]).astype(np.float32)
    )
    bond_mask = jnp.array([True, True, True, True, False], dtype=jnp.bool_)

    ub_bonds = jnp.array([[0, 2], [1, 3], [4, 6]], dtype=jnp.int32)
    ub_params = jnp.asarray(
        np.column_stack([rng.uniform(2.2, 2.6, size=(3,)), rng.uniform(50.0, 150.0, size=(3,))]).astype(np.float32)
    )
    ub_mask = jnp.array([True, True, False], dtype=jnp.bool_)

    angles = jnp.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
    angle_params = jnp.asarray(
        np.column_stack([rng.uniform(1.8, 2.2, size=(4,)), rng.uniform(60.0, 120.0, size=(4,))]).astype(np.float32)
    )
    angle_mask = jnp.array([True, True, True, False], dtype=jnp.bool_)

    dihedrals = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [4, 5, 6, 7]], dtype=jnp.int32)
    dihedral_params = jnp.array(
        [[2.0, 0.0, 1.5], [3.0, 3.14159265, 0.8], [1.0, 0.5, 1.2]], dtype=jnp.float32
    )
    dihedral_mask = jnp.array([True, True, False], dtype=jnp.bool_)

    impropers = jnp.array([[5, 4, 6, 7], [8, 9, 10, 11]], dtype=jnp.int32)
    improper_params = jnp.array([[45.0, 0.0], [60.0, 0.1]], dtype=jnp.float32)
    improper_mask = jnp.array([True, True], dtype=jnp.bool_)

    exc_pairs = jnp.array([[0, 3], [1, 4], [2, 5]], dtype=jnp.int32)
    exc_sigmas = jnp.array([3.1, 3.3, 2.9], dtype=jnp.float32)
    exc_epsilons = jnp.array([0.12, 0.15, 0.08], dtype=jnp.float32)
    exc_qprods = jnp.array([-0.25, 0.18, -0.10], dtype=jnp.float32)
    exc_mask = jnp.array([True, True, False], dtype=jnp.bool_)

    cmap_torsions = jnp.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=jnp.int32)
    cmap_indices = jnp.array([0, 0], dtype=jnp.int32)
    cmap_mask = jnp.array([True, False], dtype=jnp.bool_)
    raw_grids = jnp.asarray(rng.uniform(-1.0, 1.0, size=(1, 24, 24)).astype(np.float32))
    cmap_coeffs = precompute_cmap_coefficients(raw_grids)

    def sum_individual(p, bp, ubp, ap, dp, ip, esig, eeps, eqp, ccoeff):
        e1 = pallas_bonds_2body(p, bonds, bp, bond_mask, box=box, dist_eps=1e-10, interpret=True)
        e2 = pallas_bonds_2body(p, ub_bonds, ubp, ub_mask, box=box, dist_eps=1e-10, interpret=True)
        e3 = pallas_angles_harmonic(p, angles, ap, angle_mask, box=box, dist_eps=1e-10, interpret=True)
        e4 = pallas_torsions_4body(
            p, dihedrals, dp, dihedral_mask, box=box, b0_is_ji=True, is_harmonic=False, interpret=True
        )
        e5 = pallas_torsions_4body(
            p, impropers, ip, improper_mask, box=box, b0_is_ji=True, is_harmonic=True, interpret=True
        )
        e6 = pallas_exceptions_14(
            p, exc_pairs, esig, eeps, eqp, exc_mask, box=box, soft_core_lambda=0.8, interpret=True
        )
        e7 = pallas_cmap_torsions(
            p, cmap_torsions, cmap_indices, cmap_mask, ccoeff, box=box, first_is_psi=True, interpret=True
        )
        return e1 + e2 + e3 + e4 + e5 + e6 + e7

    def fused_call(p, bp, ubp, ap, dp, ip, esig, eeps, eqp, ccoeff):
        return pallas_bonded_all_fused(
            p,
            bonds=bonds,
            bond_params=bp,
            bond_mask=bond_mask,
            urey_bradley_bonds=ub_bonds,
            urey_bradley_params=ubp,
            urey_bradley_mask=ub_mask,
            angles=angles,
            angle_params=ap,
            angle_mask=angle_mask,
            dihedrals=dihedrals,
            dihedral_params=dp,
            dihedral_mask=dihedral_mask,
            impropers=impropers,
            improper_params=ip,
            improper_mask=improper_mask,
            improper_is_harmonic=True,
            exception_pairs=exc_pairs,
            exception_sigmas=esig,
            exception_epsilons=eeps,
            exception_chargeprods=eqp,
            exception_mask=exc_mask,
            cmap_torsions=cmap_torsions,
            cmap_indices=cmap_indices,
            cmap_mask=cmap_mask,
            cmap_coeffs=ccoeff,
            box=box,
            dist_eps=1e-10,
            b0_is_ji=True,
            cmap_first_is_psi=True,
            soft_core_lambda=0.8,
            interpret=True,
        )

    args = (
        pos,
        bond_params,
        ub_params,
        angle_params,
        dihedral_params,
        improper_params,
        exc_sigmas,
        exc_epsilons,
        exc_qprods,
        cmap_coeffs,
    )

    e_ind, grads_ind = jax.value_and_grad(sum_individual, argnums=tuple(range(10)))(*args)
    e_fus, grads_fus = jax.value_and_grad(fused_call, argnums=tuple(range(10)))(*args)

    np.testing.assert_allclose(e_fus, e_ind, rtol=1e-5, atol=1e-5)
    for idx, (g_f, g_i) in enumerate(zip(grads_fus, grads_ind)):
        np.testing.assert_allclose(g_f, g_i, rtol=1e-5, atol=1e-5, err_msg=f"Grad mismatch at arg {idx}")

    # Direct return_forces=True check on megakernel
    e_f_dir, f_f_dir = pallas_bonded_all_fused(
        pos,
        bonds=bonds,
        bond_params=bond_params,
        bond_mask=bond_mask,
        urey_bradley_bonds=ub_bonds,
        urey_bradley_params=ub_params,
        urey_bradley_mask=ub_mask,
        angles=angles,
        angle_params=angle_params,
        angle_mask=angle_mask,
        dihedrals=dihedrals,
        dihedral_params=dihedral_params,
        dihedral_mask=dihedral_mask,
        impropers=impropers,
        improper_params=improper_params,
        improper_mask=improper_mask,
        improper_is_harmonic=True,
        exception_pairs=exc_pairs,
        exception_sigmas=exc_sigmas,
        exception_epsilons=exc_epsilons,
        exception_chargeprods=exc_qprods,
        exception_mask=exc_mask,
        cmap_torsions=cmap_torsions,
        cmap_indices=cmap_indices,
        cmap_mask=cmap_mask,
        cmap_coeffs=cmap_coeffs,
        box=box,
        dist_eps=1e-10,
        b0_is_ji=True,
        cmap_first_is_psi=True,
        soft_core_lambda=0.8,
        return_forces=True,
        interpret=True,
    )
    np.testing.assert_allclose(e_f_dir, e_ind, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(f_f_dir, -grads_ind[0], rtol=1e-5, atol=1e-5)

    # Dead-branch pruning check: only bonds + angles provided (all other groups None/absent)
    e_subset = pallas_bonded_all_fused(
        pos,
        bonds=bonds,
        bond_params=bond_params,
        bond_mask=bond_mask,
        angles=angles,
        angle_params=angle_params,
        angle_mask=angle_mask,
        box=box,
        dist_eps=1e-10,
        interpret=True,
    )
    e_subset_ref = (
        pallas_bonds_2body(pos, bonds, bond_params, bond_mask, box=box, dist_eps=1e-10, interpret=True)
        + pallas_angles_harmonic(pos, angles, angle_params, angle_mask, box=box, dist_eps=1e-10, interpret=True)
    )
    np.testing.assert_allclose(e_subset, e_subset_ref, rtol=1e-5, atol=1e-5)

    # Verify xtrax.op contract (`assert_kernel_ran` and `assert_kernel_parity`)
    cpu_ctx = Context(kernels=True, backend="cpu")
    ref_ctx = Context(kernels=False)
    exp_ctx = Context(kernels=True, target="export", export_format="stablehlo")

    op_k = make_bonded_all_fused_op(
        has_box=True,
        dist_eps=1e-10,
        b0_is_ji=True,
        cmap_first_is_psi=True,
        context=cpu_ctx,
    )
    op_r = make_bonded_all_fused_op(
        has_box=True,
        dist_eps=1e-10,
        b0_is_ji=True,
        cmap_first_is_psi=True,
        context=ref_ctx,
    )
    op_e = make_bonded_all_fused_op(
        has_box=True,
        dist_eps=1e-10,
        b0_is_ji=True,
        cmap_first_is_psi=True,
        context=exp_ctx,
    )
    assert op_k.select().primal == "kernel:cpu"
    assert op_r.select().primal == "reference"
    assert op_e.select().primal == "reference"

    b_idx_all = jnp.concatenate([bonds, ub_bonds], axis=0)
    b_prm_all = jnp.concatenate([bond_params, ub_params], axis=0)
    b_msk_all = jnp.concatenate([bond_mask.astype(jnp.float32), ub_mask.astype(jnp.float32)], axis=0)
    box6 = jnp.array([30.0, 30.0, 30.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    lam_1 = jnp.array([0.8], dtype=jnp.float32)

    op_args = (
        pos,
        b_idx_all,
        b_prm_all,
        b_msk_all,
        angles,
        angle_params,
        angle_mask.astype(jnp.float32),
        dihedrals,
        dihedral_params,
        dihedral_mask.astype(jnp.float32),
        impropers,
        improper_params,
        improper_mask.astype(jnp.float32),
        exc_pairs,
        exc_sigmas,
        exc_epsilons,
        exc_qprods,
        exc_mask.astype(jnp.float32),
        cmap_torsions,
        cmap_indices,
        cmap_coeffs,
        cmap_mask.astype(jnp.float32),
        box6,
        lam_1,
    )
    assert_kernel_ran(op_k, *op_args)
    assert_kernel_parity(op_k, op_r, *op_args, rel=1e-4)
    np.testing.assert_allclose(
        jax.grad(lambda r: op_k(r, *op_args[1:]))(pos),
        jax.grad(lambda r: op_r(r, *op_args[1:]))(pos),
        rtol=1e-4,
        atol=1e-4,
    )


# ==============================================================================
# O3.1, O3.2, O3.3: Bekker Torsion, Cross-Product Angle, Horner CMAP
# ==============================================================================


def test_o3_bekker_torsion_cross_angle_and_horner_cmap():
    """Verify O3.1 Bekker torsion (both b0_is_ji modes), O3.2 cross angle modes, and O3.3 Horner CMAP."""
    rng = np.random.default_rng(99)
    N = 16
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    pos = jnp.asarray(rng.uniform(2.0, 22.0, size=(N, 3)).astype(np.float32))

    # O3.1: Both b0_is_ji=False and b0_is_ji=True match reference under export_context
    from prolix.export import export_context

    torsions = jnp.array([[0, 1, 2, 3], [2, 3, 4, 5], [5, 6, 7, 8]], dtype=jnp.int32)
    t_params = jnp.array([[2.0, 0.4, 1.8], [3.0, 1.2, 0.9], [1.0, -0.5, 2.1]], dtype=jnp.float32)
    t_mask = jnp.array([True, True, True], dtype=jnp.bool_)

    for b0_ji in (False, True):
        for is_harm in (False, True):
            p_arr = t_params[:, :2] if is_harm else t_params
            e_pal, g_pal = jax.value_and_grad(
                lambda x: pallas_torsions_4body(
                    x, torsions, p_arr, t_mask, box=box, b0_is_ji=b0_ji, is_harmonic=is_harm, interpret=True
                )
            )(pos)
            with export_context():
                e_ref, g_ref = jax.value_and_grad(
                    lambda x: pallas_torsions_4body(
                        x, torsions, p_arr, t_mask, box=box, b0_is_ji=b0_ji, is_harmonic=is_harm, interpret=True
                    )
                )(pos)
            np.testing.assert_allclose(e_pal, e_ref, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(g_pal, g_ref, rtol=1e-5, atol=1e-5)

    # O3.2: Cross-product angle force projection (`legacy_denom_eps` and `openmm_cross`)
    angles = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=jnp.int32)
    a_params = jnp.array([[1.9, 100.0], [2.1, 80.0], [1.7, 120.0]], dtype=jnp.float32)
    a_mask = jnp.array([True, True, True], dtype=jnp.bool_)

    e_leg, g_leg = jax.value_and_grad(
        lambda x: pallas_angles_harmonic(
            x, angles, a_params, a_mask, box=box, angle_regularization_mode="legacy_denom_eps", interpret=True
        )
    )(pos)
    e_omm, g_omm = jax.value_and_grad(
        lambda x: pallas_angles_harmonic(
            x, angles, a_params, a_mask, box=box, angle_regularization_mode="openmm_cross", interpret=True
        )
    )(pos)
    with export_context():
        e_aref, g_aref = jax.value_and_grad(
            lambda x: pallas_angles_harmonic(x, angles, a_params, a_mask, box=box, interpret=True)
        )(pos)
    np.testing.assert_allclose(e_leg, e_aref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_leg, g_aref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(e_omm, e_aref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_omm, g_aref, rtol=1e-4, atol=1e-4)

    # O3.3: Horner 4-tuple CMAP bicubic evaluation
    cmap_tors = jnp.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=jnp.int32)
    cmap_idx = jnp.array([0, 1], dtype=jnp.int32)
    cmap_m = jnp.array([True, True], dtype=jnp.bool_)
    grids = jnp.asarray(rng.uniform(-2.0, 2.0, size=(2, 24, 24)).astype(np.float32))
    coeffs = precompute_cmap_coefficients(grids)

    e_cpal, g_cpal = jax.value_and_grad(
        lambda x: pallas_cmap_torsions(x, cmap_tors, cmap_idx, cmap_m, coeffs, box=box, interpret=True)
    )(pos)
    with export_context():
        e_cref, g_cref = jax.value_and_grad(
            lambda x: pallas_cmap_torsions(x, cmap_tors, cmap_idx, cmap_m, coeffs, box=box, interpret=True)
        )(pos)
    np.testing.assert_allclose(e_cpal, e_cref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_cpal, g_cref, rtol=1e-5, atol=1e-5)


# ==============================================================================
# O4.1 & O5: Block-Level Periodic Wrapping & 3D Morton (Z-Order) Atom Sorting
# ==============================================================================


def test_o4_and_o5_morton_permutation_invariance_nonbonded_and_pme():
    """Verify O4.1 periodic block wrapping and O5 Morton permutation invariance in nonbonded & PME."""
    rng = np.random.default_rng(777)
    N = 48
    K = 10
    E = 4
    box = jnp.array([18.0, 18.0, 18.0], dtype=jnp.float32)

    # Include atoms near both 0.0 and box_L boundaries to exercise both interior and boundary-straddling blocks
    pos = jnp.asarray(rng.uniform(0.2, 17.8, size=(N, 3)).astype(np.float32))
    charges = jnp.asarray(rng.uniform(-0.6, 0.6, size=(N,)).astype(np.float32))
    sigmas = jnp.asarray(rng.uniform(2.8, 3.6, size=(N,)).astype(np.float32))
    epsilons = jnp.asarray(rng.uniform(0.05, 0.2, size=(N,)).astype(np.float32))
    atom_mask = jnp.ones((N,), dtype=jnp.bool_)
    atom_mask = atom_mask.at[-2:].set(False)  # Include 2 padded ghost atoms at the tail

    # Build canonical neighbor list and exclusions
    nb_np = np.full((N, K), N, dtype=np.int32)
    ex_idx_np = np.full((N, E), -1, dtype=np.int32)
    ex_vdw_np = np.ones((N, E), dtype=np.float32)
    ex_elec_np = np.ones((N, E), dtype=np.float32)
    for i in range(N - 2):
        nbrs = [(i + d) % (N - 2) for d in range(1, K + 1)]
        nb_np[i, : len(nbrs)] = nbrs
        ex_idx_np[i, 0] = (i + 1) % (N - 2)
        ex_vdw_np[i, 0] = 0.0
        ex_elec_np[i, 0] = 0.0
        ex_idx_np[i, 1] = (i + 2) % (N - 2)
        ex_vdw_np[i, 1] = 0.5
        ex_elec_np[i, 1] = 0.8333333

    neighbor_idx = jnp.asarray(nb_np)
    excl_idx = jnp.asarray(ex_idx_np)
    excl_vdw = jnp.asarray(ex_vdw_np)
    excl_elec = jnp.asarray(ex_elec_np)

    # Compute 3D Morton permutation
    perm, inv_perm = compute_morton_codes_3d(pos, box, atom_mask=atom_mask)
    assert perm.shape == (N,) and inv_perm.shape == (N,)
    # Verify perm and inv_perm are exact inverses
    np.testing.assert_array_equal(np.asarray(perm[inv_perm]), np.arange(N, dtype=np.int32))
    # Verify padded ghost atoms stay at the tail
    assert int(perm[-1]) >= N - 2 and int(perm[-2]) >= N - 2

    nb_morton = permute_neighbor_list_morton(neighbor_idx, perm, inv_perm, n_atoms=N)
    ex_idx_morton, ex_vdw_morton, ex_elec_morton = permute_and_sort_exclusions(
        excl_idx, excl_vdw, excl_elec, perm, inv_perm
    )

    # Evaluate canonical vs Morton-sorted nonbonded energy and canonical forces
    e_canon, g_canon = jax.value_and_grad(
        lambda p: pallas_nonbonded_nl(
            p, charges, sigmas, epsilons, atom_mask, neighbor_idx,
            excl_idx, excl_vdw, excl_elec,
            box=box, pme_alpha=0.32, cutoff=8.0, lj_switch_width=1.0,
            include_pme_excl_erf=True, interpret=True,
        )
    )(pos)

    e_morton_pre, g_morton_pre = jax.value_and_grad(
        lambda p: pallas_nonbonded_nl(
            p, charges, sigmas, epsilons, atom_mask, nb_morton,
            ex_idx_morton, ex_vdw_morton, ex_elec_morton,
            box=box, pme_alpha=0.32, cutoff=8.0, lj_switch_width=1.0,
            include_pme_excl_erf=True,
            atom_perm=perm, atom_inv_perm=inv_perm, nl_already_permuted=True,
            interpret=True,
        )
    )(pos)

    e_morton_auto, g_morton_auto = jax.value_and_grad(
        lambda p: pallas_nonbonded_nl(
            p, charges, sigmas, epsilons, atom_mask, neighbor_idx,
            excl_idx, excl_vdw, excl_elec,
            box=box, pme_alpha=0.32, cutoff=8.0, lj_switch_width=1.0,
            include_pme_excl_erf=True,
            atom_perm=perm, atom_inv_perm=inv_perm, nl_already_permuted=False,
            interpret=True,
        )
    )(pos)

    np.testing.assert_allclose(e_morton_pre, e_canon, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_morton_pre, g_canon, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(e_morton_auto, e_canon, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g_morton_auto, g_canon, rtol=1e-5, atol=1e-5)

    # Verify PME spread_charges_pallas and gather_pallas Morton permutation invariance
    grid_dims = (16, 16, 16)

    grid_canon = spread_charges_pallas(
        pos, charges, atom_mask, box, grid_dims, order=4, interpret=True
    )
    grid_morton = spread_charges_pallas(
        pos, charges, atom_mask, box, grid_dims, order=4, interpret=True, atom_perm=perm
    )
    np.testing.assert_allclose(grid_morton, grid_canon, rtol=1e-5, atol=1e-5)

    theta_norm = jnp.asarray(rng.uniform(-1.0, 1.0, size=grid_dims).astype(np.float32))
    dpos_canon, dq_canon = gather_pallas(
        pos, charges, atom_mask, box, theta_norm, grid_dims, order=4, interpret=True
    )
    dpos_morton, dq_morton = gather_pallas(
        pos,
        charges,
        atom_mask,
        box,
        theta_norm,
        grid_dims,
        order=4,
        interpret=True,
        atom_perm=perm,
        atom_inv_perm=inv_perm,
    )
    np.testing.assert_allclose(dpos_morton, dpos_canon, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dq_morton, dq_canon, rtol=1e-5, atol=1e-5)
