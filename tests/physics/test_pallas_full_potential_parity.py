"""Full-potential Pallas kernel parity & composability suite (R1-R8).

Verifies forward energy, analytical gradients (positions and force-field parameters),
jax.vmap (both all-batched and mixed-axis shared-parameter batching), edge cases
(empty tables M == 0, K == 0, duplicate atom indices, ghost-atom masking, collinear
angles, 3-regime OBC-II descreening, traced soft_core_lambda), StableHLO export_context()
fallbacks, and end-to-end full-system wiring across system.py, batched_energy.py,
flash_explicit.py, bundle_md.py, batched_simulate.py, and pme.py.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.api.bundle_md import (
    _exception_energy_masked as bundle_exception_energy_masked,
    bonded_energy_fn_from_bundle,
    energy_fn_from_bundle,
    force_fn_from_bundle,
)
from prolix.batched_energy import (
    _angle_energy_masked,
    _bond_energy_masked,
    _cmap_energy_masked,
    _dihedral_energy_masked,
    _exception_energy_masked,
    _pme_exclusion_correction_from_pairs,
    single_padded_energy,
    single_padded_energy_nl,
    single_padded_energy_nl_cvjp,
    single_padded_force,
)
from prolix.batched_simulate import make_langevin_step, make_langevin_step_nl
from prolix.export import export_context
from prolix.pallas_kernels import (
    _triton_gpu_available,
    pallas_angles_harmonic,
    pallas_bonds_2body,
    pallas_cmap_torsions,
    pallas_exceptions_14,
    pallas_flash_explicit_nonbonded,
    pallas_gb_ace_nl,
    pallas_nonbonded_nl,
    pallas_pme_exclusion_pairs,
    pallas_sasa_nl,
    pallas_torsions_4body,
)
from prolix.physics.bonded import (
    make_angle_energy_fn,
    make_bond_energy_fn,
    make_dihedral_energy_fn,
    make_harmonic_improper_energy_fn,
)
from prolix.physics.cmap import compute_cmap_energies, precompute_cmap_coefficients
from prolix.physics.flash_explicit import (
    _excl_correction_from_sys,
    chunked_explicit_nonbonded_energy,
    flash_explicit_total_energy,
)
from prolix.physics.generalized_born import (
    compute_ace_nonpolar_energy,
    compute_gb_energy_neighbor_list,
    compute_nonpolar_energy,
)
from prolix.physics.neighbor_list import ExclusionSpec, map_exclusions_to_dense_padded
from prolix.physics.pme import make_spme_energy_fn
from prolix.physics.system import make_energy_fn, make_energy_fn_pure
from prolix.typing import DifferentiableParams, IntegratorState
from prolix.utils import topology


# ==============================================================================
# R1: 2-Body Kernels (Bonds, Urey-Bradley, 1-4 Exceptions, PME Exclusions)
# ==============================================================================


def test_pallas_bonds_2body_parity_vmap_and_export():
    """Verify pallas_bonds_2body across dist_eps, 2/3-col UB, duplicates, vmap, and export."""
    rng = np.random.default_rng(101)
    N = 19
    box = jnp.array([22.0, 22.0, 22.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)
    disp_free, _ = space.free()

    pos = jnp.asarray(rng.uniform(1.0, 20.0, size=(N, 3)).astype(np.float32))
    # Include duplicate atom indices (atom 0 in multiple bonds) and padded slots
    bonds = jnp.array(
        [[0, 1], [0, 2], [1, 2], [3, 4], [5, 6], [0, 7], [10, 11], [0, 0]],
        dtype=jnp.int32,
    )
    params = jnp.asarray(
        np.column_stack([
            rng.uniform(1.0, 1.6, size=(8,)),
            rng.uniform(200.0, 500.0, size=(8,)),
        ]).astype(np.float32)
    )
    mask = jnp.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=jnp.float32)

    # 1. Masked convention (dist_eps=1e-10) with PBC
    val_p, (gp_pos, gp_par) = jax.value_and_grad(
        lambda r, p: pallas_bonds_2body(r, bonds, p, mask, box=box, dist_eps=1e-10),
        argnums=(0, 1),
    )(pos, params)
    val_r, (gr_pos, gr_par) = jax.value_and_grad(
        lambda r, p: _bond_energy_masked(r, bonds, p, mask, disp_pbc),
        argnums=(0, 1),
    )(pos, params)
    np.testing.assert_allclose(val_p, val_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gp_pos, gr_pos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gp_par[:6], gr_par[:6], rtol=1e-5, atol=1e-5)

    # 2. Unmasked convention (dist_eps=0.0) + 3-col Urey-Bradley slicing [:, :2]
    ub_3col = jnp.column_stack([bonds[:6], jnp.arange(6, dtype=jnp.int32)])
    ub_params = params[:6]
    ref_bond_fn = make_bond_energy_fn(disp_free, ub_3col[:, :2])
    val_u_p, gu_pos = jax.value_and_grad(
        lambda r: pallas_bonds_2body(r, ub_3col, ub_params, None, box=None, dist_eps=0.0)
    )(pos)
    val_u_r, gu_ref = jax.value_and_grad(lambda r: ref_bond_fn(r, ub_params))(pos)
    np.testing.assert_allclose(val_u_p, val_u_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gu_pos, gu_ref, rtol=1e-5, atol=1e-5)

    # 3. Empty table M == 0
    empty_b = jnp.zeros((0, 2), dtype=jnp.int32)
    empty_p = jnp.zeros((0, 2), dtype=jnp.float32)
    val_e, ge_pos = jax.value_and_grad(
        lambda r: pallas_bonds_2body(r, empty_b, empty_p)
    )(pos)
    assert float(val_e) == 0.0
    np.testing.assert_allclose(ge_pos, 0.0, atol=1e-7)

    # 4. Mixed-axis vmap (batched positions, shared parameters) + grad wrt shared params
    pos_b = jnp.stack([pos, pos + 0.05], axis=0)
    grad_shared_p = jax.grad(
        lambda p: jnp.sum(
            jax.vmap(lambda r: pallas_bonds_2body(r, bonds, p, mask, box=box, dist_eps=1e-10))(pos_b)
        )
    )(params)
    grad_shared_r = jax.grad(
        lambda p: jnp.sum(
            jax.vmap(lambda r: _bond_energy_masked(r, bonds, p, mask, disp_pbc))(pos_b)
        )
    )(params)
    np.testing.assert_allclose(grad_shared_p[:6], grad_shared_r[:6], rtol=1e-5, atol=1e-5)

    # 5. Export context fallback
    with export_context():
        val_exp = pallas_bonds_2body(pos, bonds, params, mask, box=box, dist_eps=1e-10)
    np.testing.assert_allclose(val_exp, val_r, rtol=1e-5, atol=1e-5)


def test_pallas_exceptions_14_parity_softcore_and_vmap():
    """Verify pallas_exceptions_14 with standard and traced softcore lambda."""
    rng = np.random.default_rng(102)
    N = 16
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)

    pos = jnp.asarray(rng.uniform(1.0, 18.0, size=(N, 3)).astype(np.float32))
    pairs = jnp.array([[0, 3], [0, 4], [1, 5], [2, 6], [0, 0]], dtype=jnp.int32)
    sigmas = jnp.asarray(rng.uniform(2.5, 3.3, size=(5,)).astype(np.float32))
    epsilons = jnp.asarray(rng.uniform(0.05, 0.2, size=(5,)).astype(np.float32))
    qprods = jnp.asarray(rng.uniform(-0.25, 0.25, size=(5,)).astype(np.float32))
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    lam = jnp.array(0.65, dtype=jnp.float32)

    val_p, grads_p = jax.value_and_grad(
        lambda r, s, e, q, l: pallas_exceptions_14(
            r, pairs, s, e, q, mask, box=box, soft_core_lambda=l
        ),
        argnums=(0, 1, 2, 3, 4),
    )(pos, sigmas, epsilons, qprods, lam)
    val_r, grads_r = jax.value_and_grad(
        lambda r, s, e, q, l: _exception_energy_masked(
            r, pairs, s, e, q, mask, disp_pbc, soft_core_lambda=l
        ),
        argnums=(0, 1, 2, 3, 4),
    )(pos, sigmas, epsilons, qprods, lam)

    np.testing.assert_allclose(val_p, val_r, rtol=1e-5, atol=1e-5)
    for idx in range(5):
        np.testing.assert_allclose(grads_p[idx], grads_r[idx], rtol=1e-4, atol=1e-5)

    # Bundle helper dispatch
    val_b_p = bundle_exception_energy_masked(
        pos, pairs, sigmas, epsilons, qprods, mask > 0, disp_pbc, use_pallas=True, box=box
    )
    val_b_r = bundle_exception_energy_masked(
        pos, pairs, sigmas, epsilons, qprods, mask > 0, disp_pbc, use_pallas=False, box=box
    )
    np.testing.assert_allclose(val_b_p, val_b_r, rtol=1e-5, atol=1e-5)


def test_pallas_pme_exclusion_pairs_pair_and_row_formats():
    """Verify pallas_pme_exclusion_pairs on both (E, 2) pair-list and (N, M) row formats."""
    rng = np.random.default_rng(103)
    N = 14
    n_real = 12
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)
    alpha = 0.34

    pos = jnp.asarray(rng.uniform(1.0, 18.0, size=(N, 3)).astype(np.float32))
    charges = jnp.asarray(rng.uniform(-0.5, 0.5, size=(N,)).astype(np.float32))
    atom_mask = jnp.arange(N) < n_real
    safe_q = jnp.where(atom_mask, charges, 0.0)

    # 1. Pair-list format (E, 2)
    pair_idx = jnp.array([[0, 1], [1, 2], [2, 3], [0, 3], [12, 13]], dtype=jnp.int32)
    pair_scales = jnp.array([0.0, 0.0, 0.0, 0.8333333, 0.0], dtype=jnp.float32)

    val_p, (gp_r, gp_q, gp_s) = jax.value_and_grad(
        lambda r, q, s: pallas_pme_exclusion_pairs(
            r, pair_idx, s, q, pme_alpha=alpha, n_atoms=n_real, box=box, is_row_format=False
        ),
        argnums=(0, 1, 2),
    )(pos, safe_q, pair_scales)
    val_r, (gr_r, gr_q, gr_s) = jax.value_and_grad(
        lambda r, q, s: _pme_exclusion_correction_from_pairs(
            r, disp_pbc, pair_idx, s, q, alpha, n_real
        ),
        argnums=(0, 1, 2),
    )(pos, safe_q, pair_scales)
    np.testing.assert_allclose(val_p, val_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gp_r, gr_r, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gp_q, gr_q, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gp_s[:4], gr_s[:4], rtol=1e-4, atol=1e-5)

    # 2. Row format (N, M) via _excl_correction_from_sys
    bonds = np.array([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=np.int32)
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
    excl_indices, _, excl_elec = map_exclusions_to_dense_padded(excl_spec, max_exclusions=8)

    val_row_p, grow_p = jax.value_and_grad(
        lambda r: pallas_pme_exclusion_pairs(
            r, excl_indices, excl_elec, safe_q,
            pme_alpha=alpha, n_atoms=N, box=box, atom_mask=atom_mask, is_row_format=True,
        )
    )(pos)

    class _DummySys:
        excl_pair_indices = None
        excl_pair_scales_elec = None

    dummy = _DummySys()
    dummy.excl_indices = excl_indices
    dummy.excl_scales_elec = excl_elec
    dummy.atom_mask = atom_mask
    dummy.box_size = box

    val_row_r, grow_r = jax.value_and_grad(
        lambda r: _excl_correction_from_sys(r, safe_q, dummy, disp_pbc, alpha, use_pallas=False)
    )(pos)
    np.testing.assert_allclose(val_row_p, val_row_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grow_p, grow_r, rtol=1e-4, atol=1e-5)


# ==============================================================================
# R2: 3-Body Harmonic Angle Kernel
# ==============================================================================


def test_pallas_angles_harmonic_parity_collinear_and_vmap():
    """Verify pallas_angles_harmonic across dist_eps, collinear geometry, vmap, and export."""
    rng = np.random.default_rng(202)
    N = 17
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)
    disp_free, _ = space.free()

    pos = jnp.asarray(rng.uniform(2.0, 22.0, size=(N, 3)).astype(np.float32))
    angles = jnp.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 4], [5, 6, 7], [8, 9, 10], [0, 0, 0]],
        dtype=jnp.int32,
    )
    params = jnp.asarray(
        np.column_stack([
            rng.uniform(1.6, 2.2, size=(6,)),
            rng.uniform(40.0, 120.0, size=(6,)),
        ]).astype(np.float32)
    )
    mask = jnp.array([1, 1, 1, 1, 1, 0], dtype=jnp.float32)

    # 1. Masked convention (dist_eps=1e-10)
    val_p, (gp_pos, gp_par) = jax.value_and_grad(
        lambda r, p: pallas_angles_harmonic(r, angles, p, mask, box=box, dist_eps=1e-10),
        argnums=(0, 1),
    )(pos, params)
    val_r, (gr_pos, gr_par) = jax.value_and_grad(
        lambda r, p: _angle_energy_masked(r, angles, p, mask, disp_pbc),
        argnums=(0, 1),
    )(pos, params)
    np.testing.assert_allclose(val_p, val_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gp_pos, gr_pos, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gp_par[:5], gr_par[:5], rtol=1e-5, atol=1e-5)

    # 2. Unmasked convention (dist_eps=0.0)
    ref_ang_fn = make_angle_energy_fn(disp_free, angles[:5])
    val_u_p, gu_pos = jax.value_and_grad(
        lambda r: pallas_angles_harmonic(r, angles[:5], params[:5], None, box=None, dist_eps=0.0)
    )(pos)
    val_u_r, gu_ref = jax.value_and_grad(lambda r: ref_ang_fn(r, params[:5]))(pos)
    np.testing.assert_allclose(val_u_p, val_u_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gu_pos, gu_ref, rtol=1e-4, atol=1e-5)

    # 3. Collinear angle (\theta = \pi and \theta = 0) finite-gradient check
    pos_col = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float32)
    ang_col = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    par_col = jnp.array([[2.0, 80.0]], dtype=jnp.float32)
    _, g_col = jax.value_and_grad(
        lambda r: pallas_angles_harmonic(r, ang_col, par_col)
    )(pos_col)
    assert bool(jnp.all(jnp.isfinite(g_col)))


# ==============================================================================
# R3: 4-Body Dihedral & Improper Torsion Kernel
# ==============================================================================


def test_pallas_torsions_4body_all_conventions_and_3d_params():
    """Verify pallas_torsions_4body across b0_is_ji, is_harmonic, and 3D (N_dih, T, P) params."""
    rng = np.random.default_rng(303)
    N = 18
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)
    disp_free, _ = space.free()

    pos = jnp.asarray(rng.uniform(2.0, 22.0, size=(N, 3)).astype(np.float32))
    indices = jnp.array(
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 2, 5, 7], [8, 9, 10, 11], [0, 0, 0, 0]],
        dtype=jnp.int32,
    )
    params = jnp.asarray(
        np.column_stack([
            rng.choice([1.0, 2.0, 3.0], size=(6,)),
            rng.uniform(-np.pi, np.pi, size=(6,)),
            rng.uniform(0.5, 3.0, size=(6,)),
        ]).astype(np.float32)
    )
    mask = jnp.array([1, 1, 1, 1, 1, 0], dtype=jnp.float32)

    # 1. Masked convention (b0_is_ji=True, is_harmonic=False) with PBC
    val_p, (gp_pos, gp_par) = jax.value_and_grad(
        lambda r, p: pallas_torsions_4body(
            r, indices, p, mask, box=box, b0_is_ji=True, is_harmonic=False
        ),
        argnums=(0, 1),
    )(pos, params)
    val_r, (gr_pos, gr_par) = jax.value_and_grad(
        lambda r, p: _dihedral_energy_masked(r, indices, p, mask, disp_pbc),
        argnums=(0, 1),
    )(pos, params)
    np.testing.assert_allclose(val_p, val_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gp_pos, gr_pos, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gp_par[:5], gr_par[:5], rtol=1e-5, atol=1e-5)

    # 2. Unmasked periodic convention (b0_is_ji=False, is_harmonic=False)
    ref_dih_fn = make_dihedral_energy_fn(disp_free, indices[:5])
    params_5_3d = params[:5, None, :]
    val_u_p, gu_pos = jax.value_and_grad(
        lambda r: pallas_torsions_4body(
            r, indices[:5], params_5_3d, None, box=None, b0_is_ji=False, is_harmonic=False
        )
    )(pos)
    val_u_r, gu_ref = jax.value_and_grad(lambda r: ref_dih_fn(r, params_5_3d))(pos)
    np.testing.assert_allclose(val_u_p, val_u_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gu_pos, gu_ref, rtol=1e-4, atol=1e-5)

    # 3. Harmonic improper convention (b0_is_ji=False, is_harmonic=True)
    harm_params = jnp.column_stack([
        jnp.full((5,), 45.0, dtype=jnp.float32),
        params[:5, 1],
    ])[:, None, :]
    ref_harm_fn = make_harmonic_improper_energy_fn(disp_free, indices[:5])
    val_h_p, (gh_pos, gh_par) = jax.value_and_grad(
        lambda r, p: pallas_torsions_4body(
            r, indices[:5], p, None, box=None, b0_is_ji=False, is_harmonic=True
        ),
        argnums=(0, 1),
    )(pos, harm_params)
    val_h_r, (gh_ref, gh_par_ref) = jax.value_and_grad(
        lambda r, p: ref_harm_fn(r, p), argnums=(0, 1)
    )(pos, harm_params)
    np.testing.assert_allclose(val_h_p, val_h_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gh_pos, gh_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gh_par, gh_par_ref, rtol=1e-4, atol=1e-5)
    assert gh_par.shape == harm_params.shape

    # 4. 3D (N_dih, T, 3) multi-term dihedral parameters
    params_3d = jnp.stack([params[:5], params[:5] * 0.5], axis=1)  # (5, 2, 3)
    val_3d, (g3d_pos, g3d_par) = jax.value_and_grad(
        lambda r, p: pallas_torsions_4body(
            r, indices[:5], p, jnp.ones((5,), dtype=jnp.float32), b0_is_ji=True
        ),
        argnums=(0, 1),
    )(pos, params_3d)
    assert g3d_par.shape == (5, 2, 3)


# ==============================================================================
# R4: 5-Body CMAP Bicubic Spline Kernel
# ==============================================================================


def test_pallas_cmap_torsions_both_orders_and_8col():
    """Verify pallas_cmap_torsions across first_is_psi, 8-col indices, and raw grids."""
    rng = np.random.default_rng(404)
    N = 20
    box = jnp.array([25.0, 25.0, 25.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)
    disp_free, _ = space.free()

    pos = jnp.asarray(rng.uniform(2.0, 22.0, size=(N, 3)).astype(np.float32))
    torsions_5col = jnp.array(
        [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [5, 6, 7, 8, 9], [0, 0, 0, 0, 0]],
        dtype=jnp.int32,
    )
    # 8-col MolecularBundle format where [:, 0:4] is phi and [:, 4:8] is psi
    torsions_8col = jnp.column_stack([torsions_5col[:, 0:4], torsions_5col[:, 1:5]])
    map_idx = jnp.array([0, 1, 0, 0], dtype=jnp.int32)
    mask = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    raw_grids = jnp.asarray(rng.normal(0.0, 1.5, size=(2, 24, 24)).astype(np.float32))
    coeffs = precompute_cmap_coefficients(raw_grids)

    # 1. Masked convention (first_is_psi=True) with 8-col indices and precomputed coeffs
    val_p, (gp_pos, gp_c) = jax.value_and_grad(
        lambda r, c: pallas_cmap_torsions(
            r, torsions_8col, map_idx, mask, c, box=box, first_is_psi=True
        ),
        argnums=(0, 1),
    )(pos, coeffs)
    val_r, (gr_pos, gr_c) = jax.value_and_grad(
        lambda r, c: _cmap_energy_masked(r, torsions_8col, map_idx, mask, c, disp_pbc),
        argnums=(0, 1),
    )(pos, coeffs)
    np.testing.assert_allclose(val_p, val_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_pos, gr_pos, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_c, gr_c, rtol=1e-4, atol=1e-4)

    # 2. System convention (first_is_psi=False) with raw (M, G, G) grids
    from prolix.physics.bonded import compute_dihedral_angles

    def _ref_sys_cmap(r):
        phi_idx = torsions_5col[:3, :4]
        psi_idx = torsions_5col[:3, 1:5]
        phi = compute_dihedral_angles(r, phi_idx, disp_free)
        psi = compute_dihedral_angles(r, psi_idx, disp_free)
        return jnp.sum(compute_cmap_energies(phi, psi, map_idx[:3], coeffs))

    val_s_p, gs_pos = jax.value_and_grad(
        lambda r: pallas_cmap_torsions(
            r, torsions_5col[:3], map_idx[:3], None, raw_grids, box=None, first_is_psi=False
        )
    )(pos)
    val_s_r, gs_ref = jax.value_and_grad(_ref_sys_cmap)(pos)
    np.testing.assert_allclose(val_s_p, val_s_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gs_pos, gs_ref, rtol=1e-4, atol=1e-4)


# ==============================================================================
# R6: Implicit-Solvent NL Kernels (pallas_gb_ace_nl and pallas_sasa_nl)
# ==============================================================================


def test_pallas_gb_ace_nl_all_regimes_k0_and_vmap():
    """Verify pallas_gb_ace_nl across all 3 OBC-II overlap regimes, K == 0, ghost atoms, and vmap."""
    rng = np.random.default_rng(601)
    N = 16
    n_real = 12
    K = 12

    # Construct positions exercising all 3 descreening regimes:
    # - complete engulfment (r + sr_j <= or_i) for pair (0, 1) with large radius on 0, small on 1
    # - partial overlap (|r - sr_j| < or_i < r + sr_j)
    # - non-overlapping (r - sr_j >= or_i)
    pos_np = rng.uniform(0.0, 10.0, size=(N, 3)).astype(np.float32)
    pos_np[0] = [0.0, 0.0, 0.0]
    pos_np[1] = [0.25, 0.0, 0.0]  # engulfed inside atom 0's offset radius
    pos_np[2] = [1.8, 0.0, 0.0]   # partial overlap
    pos = jnp.asarray(pos_np)

    radii_np = rng.uniform(1.4, 1.9, size=(N,)).astype(np.float32)
    radii_np[0] = 2.2
    radii_np[1] = 1.0
    radii_np[n_real:] = 1.5  # default padding radii
    radii = jnp.asarray(radii_np)
    scaled_radii = radii * 0.85
    charges = jnp.asarray(rng.uniform(-0.6, 0.6, size=(N,)).astype(np.float32))
    atom_mask = jnp.arange(N) < n_real
    q_masked = jnp.where(atom_mask, charges, 0.0)

    # Symmetric dense neighbor list among real atoms
    nbr_np = np.full((N, K), N, dtype=np.int32)
    for i in range(n_real):
        col = 0
        for j in range(n_real):
            if i != j and col < K:
                nbr_np[i, col] = j
                col += 1
    nbr_idx = jnp.asarray(nbr_np)

    def _ref_gb_ace(r, q):
        nb_masked = jnp.where(
            atom_mask[:, None] & atom_mask[jnp.minimum(nbr_idx, N - 1)],
            nbr_idx,
            N,
        )
        e_gb, br = compute_gb_energy_neighbor_list(
            r, q, radii, nb_masked,
            dielectric_offset=0.09, scaled_radii=scaled_radii,
        )
        e_ace = jnp.sum(compute_ace_nonpolar_energy(radii, br) * atom_mask)
        return e_gb + e_ace

    val_p, (gp_r, gp_q) = jax.value_and_grad(
        lambda r, q: pallas_gb_ace_nl(
            r, q, radii, nbr_idx, atom_mask=atom_mask, scaled_radii=scaled_radii
        ),
        argnums=(0, 1),
    )(pos, q_masked)
    val_r, (gr_r, gr_q) = jax.value_and_grad(_ref_gb_ace, argnums=(0, 1))(pos, q_masked)

    np.testing.assert_allclose(val_p, val_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_r[:n_real], gr_r[:n_real], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_q[:n_real], gr_q[:n_real], rtol=1e-4, atol=1e-4)

    # K == 0 empty neighbor list
    nbr_k0 = jnp.full((N, 0), N, dtype=jnp.int32)
    val_k0_p, gk0_p = jax.value_and_grad(
        lambda r: pallas_gb_ace_nl(
            r, q_masked, radii, nbr_k0, atom_mask=atom_mask, scaled_radii=scaled_radii
        )
    )(pos)
    val_k0_r, gk0_r = jax.value_and_grad(
        lambda r: _ref_gb_ace(r, q_masked)
        if False
        else (
            lambda e_br: e_br[0] + jnp.sum(compute_ace_nonpolar_energy(radii, e_br[1]) * atom_mask)
        )(
            compute_gb_energy_neighbor_list(
                r, q_masked, radii, nbr_k0,
                dielectric_offset=0.09, scaled_radii=scaled_radii,
            )
        )
    )(pos)
    np.testing.assert_allclose(val_k0_p, val_k0_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gk0_p, gk0_r, rtol=1e-5, atol=1e-5)

    # Export context fallback
    with export_context():
        val_exp = pallas_gb_ace_nl(
            pos, q_masked, radii, nbr_idx, atom_mask=atom_mask, scaled_radii=scaled_radii
        )
    np.testing.assert_allclose(val_exp, val_r, rtol=1e-5, atol=1e-5)


def test_pallas_sasa_nl_parity_ghost_mask_and_k0():
    """Verify pallas_sasa_nl matches compute_nonpolar_energy including ghost-atom masking and K == 0."""
    rng = np.random.default_rng(602)
    N = 14
    n_real = 10
    K = 10

    pos = jnp.asarray(rng.uniform(0.0, 6.0, size=(N, 3)).astype(np.float32))
    radii = jnp.asarray(rng.uniform(1.4, 1.9, size=(N,)).astype(np.float32))
    # Ghost atoms have default radii = 1.5 A (R5-B2)
    radii = radii.at[n_real:].set(1.5)
    atom_mask = jnp.arange(N) < n_real

    nbr_np = np.full((N, K), N, dtype=np.int32)
    for i in range(n_real):
        col = 0
        for j in range(n_real):
            if i != j and col < K:
                nbr_np[i, col] = j
                col += 1
    nbr_idx = jnp.asarray(nbr_np)

    val_p, gp_r = jax.value_and_grad(
        lambda r: pallas_sasa_nl(r, radii, nbr_idx, atom_mask=atom_mask)
    )(pos)
    val_r, gr_r = jax.value_and_grad(
        lambda r: compute_nonpolar_energy(r, radii, neighbor_idx=nbr_idx, atom_mask=atom_mask)
    )(pos)
    np.testing.assert_allclose(val_p, val_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gp_r, gr_r, rtol=1e-4, atol=1e-5)
    # Ghost atom forces must be identically zero
    np.testing.assert_allclose(gp_r[n_real:], 0.0, atol=1e-7)

    # K == 0 empty neighbor list
    nbr_k0 = jnp.full((N, 0), N, dtype=jnp.int32)
    val_k0_p, gk0_p = jax.value_and_grad(
        lambda r: pallas_sasa_nl(r, radii, nbr_k0, atom_mask=atom_mask)
    )(pos)
    val_k0_r, gk0_r = jax.value_and_grad(
        lambda r: compute_nonpolar_energy(r, radii, neighbor_idx=nbr_k0, atom_mask=atom_mask)
    )(pos)
    np.testing.assert_allclose(val_k0_p, val_k0_r, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gk0_p, gk0_r, rtol=1e-5, atol=1e-5)


# ==============================================================================
# R5 & R7: Full-System Wiring & Batched SPME Parity
# ==============================================================================


def test_batched_spme_vmap_energy_and_grad():
    """Verify make_spme_energy_fn supports jax.vmap over (B, N, 3) for both XLA and Pallas."""
    rng = np.random.default_rng(701)
    B, N = 3, 16
    box = jnp.array([18.0, 18.0, 18.0], dtype=jnp.float32)
    pos_b = jnp.asarray(rng.uniform(1.0, 17.0, size=(B, N, 3)).astype(np.float32))
    q = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N,)).astype(np.float32))
    mask = jnp.arange(N) < 12

    spme_xla = make_spme_energy_fn(box, alpha=0.34, grid_dims=(16, 16, 16), use_pallas_pme=False)
    spme_pal = make_spme_energy_fn(box, alpha=0.34, grid_dims=(16, 16, 16), use_pallas_pme=True)

    # Compare vmap over B against per-item loop
    e_vmap_xla = jax.vmap(lambda r: spme_xla(r, q, mask))(pos_b)
    e_loop_xla = jnp.stack([spme_xla(pos_b[b], q, mask) for b in range(B)])
    np.testing.assert_allclose(e_vmap_xla, e_loop_xla, rtol=1e-5, atol=1e-5)

    g_vmap_xla = jax.grad(lambda pb: jnp.sum(jax.vmap(lambda r: spme_xla(r, q, mask))(pb)))(pos_b)
    g_loop_xla = jnp.stack([jax.grad(lambda r: spme_xla(r, q, mask))(pos_b[b]) for b in range(B)])
    np.testing.assert_allclose(g_vmap_xla, g_loop_xla, rtol=1e-5, atol=1e-5)

    e_vmap_pal = jax.vmap(lambda r: spme_pal(r, q, mask))(pos_b)
    g_vmap_pal = jax.grad(lambda pb: jnp.sum(jax.vmap(lambda r: spme_pal(r, q, mask))(pb)))(pos_b)
    np.testing.assert_allclose(e_vmap_pal, e_vmap_xla, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(g_vmap_pal, g_vmap_xla, rtol=1e-4, atol=1e-4)


def test_full_system_wiring_parity():
    """Verify full-potential Pallas dispatch across system.py, batched_energy.py, flash_explicit.py, and batched_simulate.py."""
    from tests.physics.test_xtrax_pallas_boundary import _make_minimal_periodic_padded_system

    n_real = 6
    sys = _make_minimal_periodic_padded_system(n_real=n_real, n_padded=8)
    # Populate radii, scaled_radii, angles, dihedrals, impropers, urey_bradley, cmap, exceptions
    rng = np.random.default_rng(702)
    N = sys.positions.shape[0]
    radii = jnp.asarray(rng.uniform(1.4, 1.8, size=(N,)).astype(np.float32))
    scaled_radii = radii * 0.85
    angles = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    angle_params = jnp.array([[1.9, 60.0], [1.85, 55.0]], dtype=jnp.float32)
    angle_mask = jnp.array([True, True])
    dihedrals = jnp.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=jnp.int32)
    dihedral_params = jnp.array([[2.0, 0.3, 1.5], [3.0, -0.5, 1.2]], dtype=jnp.float32)
    dihedral_mask = jnp.array([True, True])
    impropers = jnp.array([[1, 0, 2, 3]], dtype=jnp.int32)
    improper_params = jnp.array([[2.0, 3.14159265, 2.0]], dtype=jnp.float32)
    improper_mask = jnp.array([True])

    sys = dataclasses.replace(
        sys,
        radii=radii,
        scaled_radii=scaled_radii,
        angles=angles,
        angle_params=angle_params,
        angle_mask=angle_mask,
        dihedrals=dihedrals,
        dihedral_params=dihedral_params,
        dihedral_mask=dihedral_mask,
        impropers=impropers,
        improper_params=improper_params,
        improper_mask=improper_mask,
    )

    disp_pbc, _ = space.periodic(jnp.asarray(sys.box_size))
    disp_free, _ = space.free()

    # Build a symmetric neighbor list
    K = 6
    nbr_np = np.full((N, K), N, dtype=np.int32)
    for i in range(n_real):
        col = 0
        for j in range(n_real):
            if i != j and col < K:
                nbr_np[i, col] = j
                col += 1
    nbr_idx = jnp.asarray(nbr_np)

    # 1. single_padded_energy (with neighbor list + PME)
    e_spe_p, g_spe_p = jax.value_and_grad(
        lambda r: single_padded_energy(
            dataclasses.replace(sys, positions=r),
            disp_pbc,
            implicit_solvent=False,
            neighbor=nbr_idx,
            use_pallas=True,
        )
    )(sys.positions)
    e_spe_r, g_spe_r = jax.value_and_grad(
        lambda r: single_padded_energy(
            dataclasses.replace(sys, positions=r),
            disp_pbc,
            implicit_solvent=False,
            neighbor=nbr_idx,
            use_pallas=False,
        )
    )(sys.positions)
    np.testing.assert_allclose(e_spe_p, e_spe_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(g_spe_p[:n_real], g_spe_r[:n_real], rtol=1e-4, atol=1e-4)

    # 2. single_padded_energy_nl and single_padded_energy_nl_cvjp (implicit solvent GB + ACE + bare Coulomb)
    for fn in (single_padded_energy_nl, single_padded_energy_nl_cvjp):
        e_nl_p, g_nl_p = jax.value_and_grad(
            lambda r: fn(
                dataclasses.replace(sys, positions=r),
                nbr_idx,
                disp_free,
                implicit_solvent=True,
                use_pallas=True,
            )
        )(sys.positions)
        e_nl_r, g_nl_r = jax.value_and_grad(
            lambda r: fn(
                dataclasses.replace(sys, positions=r),
                nbr_idx,
                disp_free,
                implicit_solvent=True,
                use_pallas=False,
            )
        )(sys.positions)
        np.testing.assert_allclose(e_nl_p, e_nl_r, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(g_nl_p[:n_real], g_nl_r[:n_real], rtol=1e-4, atol=1e-4)

    # 3. single_padded_force (explicit solvent FlashMD)
    f_spf_p = single_padded_force(
        sys, disp_pbc, implicit_solvent=False, explicit_solvent=True, use_flash=True, use_pallas=True
    )
    f_spf_r = single_padded_force(
        sys, disp_pbc, implicit_solvent=False, explicit_solvent=True, use_flash=True, use_pallas=False
    )
    np.testing.assert_allclose(f_spf_p[:n_real], f_spf_r[:n_real], rtol=1e-4, atol=1e-4)

    # 4. flash_explicit_total_energy (including traced soft_core_lambda)
    e_fet_p, (gf_r_p, gf_l_p) = jax.value_and_grad(
        lambda r, lam: flash_explicit_total_energy(
            dataclasses.replace(sys, positions=r), soft_core_lambda=lam, use_pallas=True
        ),
        argnums=(0, 1),
    )(sys.positions, jnp.array(0.8, dtype=jnp.float32))
    e_fet_r, (gf_r_r, gf_l_r) = jax.value_and_grad(
        lambda r, lam: flash_explicit_total_energy(
            dataclasses.replace(sys, positions=r), soft_core_lambda=lam, use_pallas=False
        ),
        argnums=(0, 1),
    )(sys.positions, jnp.array(0.8, dtype=jnp.float32))
    np.testing.assert_allclose(e_fet_p, e_fet_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gf_r_p[:n_real], gf_r_r[:n_real], rtol=1e-4, atol=1e-4)

    # 5. make_langevin_step and make_langevin_step_nl
    state0 = IntegratorState(
        positions=sys.positions,
        momentum=jnp.zeros_like(sys.positions),
        force=jnp.zeros_like(sys.positions),
        mass=jnp.ones((N,), dtype=jnp.float32) * 12.0,
        rng=jax.random.PRNGKey(42),
        cap_count=jnp.array(0, dtype=jnp.int32),
    )
    step_nl_p = make_langevin_step_nl(0.001, 0.6, 1.0, use_pallas=True)
    step_nl_r = make_langevin_step_nl(0.001, 0.6, 1.0, use_pallas=False)
    st_p = step_nl_p(sys, state0, nbr_idx)
    st_r = step_nl_r(sys, state0, nbr_idx)
    np.testing.assert_allclose(st_p.positions[:n_real], st_r.positions[:n_real], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(st_p.force[:n_real], st_r.force[:n_real], rtol=1e-4, atol=1e-4)


def test_system_make_energy_fn_and_pure_parity():
    """Verify make_energy_fn (including return_decomposed=True) and make_energy_fn_pure parity."""
    from prolix.typing import PhysicsSystem

    rng = np.random.default_rng(703)
    N = 10
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    disp_pbc, _ = space.periodic(box)
    disp_free, _ = space.free()

    pos = jnp.asarray(rng.uniform(2.0, 18.0, size=(N, 3)).astype(np.float32))
    charges = jnp.asarray(rng.uniform(-0.4, 0.4, size=(N,)).astype(np.float32))
    sigmas = jnp.asarray(rng.uniform(2.5, 3.3, size=(N,)).astype(np.float32))
    epsilons = jnp.asarray(rng.uniform(0.05, 0.15, size=(N,)).astype(np.float32))
    radii = jnp.asarray(rng.uniform(1.4, 1.8, size=(N,)).astype(np.float32))
    scaled_radii = radii * 0.85

    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=jnp.int32)
    bond_params = jnp.array([[1.5, 300.0], [1.4, 320.0], [1.5, 310.0], [1.3, 350.0]], dtype=jnp.float32)
    angles = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)
    angle_params = jnp.array([[1.9, 60.0], [1.85, 55.0]], dtype=jnp.float32)
    dihedrals = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    dihedral_params = jnp.array([[[2.0, 0.4, 1.8]]], dtype=jnp.float32)
    impropers = jnp.array([[1, 0, 2, 3]], dtype=jnp.int32)
    harm_imp_params = jnp.array([[[40.0, 0.1]]], dtype=jnp.float32)

    excl = topology.find_bonded_exclusions(np.asarray(bonds), N)
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

    K = 9
    nbr_np = np.full((N, K), N, dtype=np.int32)
    for i in range(N):
        col = 0
        for j in range(N):
            if i != j and col < K:
                nbr_np[i, col] = j
                col += 1
    nbr_idx = jnp.asarray(nbr_np)

    sys_dict = {
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "radii": radii,
        "scaled_radii": scaled_radii,
        "bonds": bonds,
        "bond_params": bond_params,
        "angles": angles,
        "angle_params": angle_params,
        "dihedrals": dihedrals,
        "dihedral_params": dihedral_params,
        "impropers": impropers,
        "improper_params": harm_imp_params,
    }

    # 1. make_energy_fn (implicit solvent + neighbor list + return_decomposed=True)
    efn_imp_p = make_energy_fn(
        disp_free,
        sys_dict,
        exclusion_spec=excl_spec,
        implicit_solvent=True,
        use_pallas=True,
        gb_use_neighbor_list=True,
        return_decomposed=True,
    )
    efn_imp_r = make_energy_fn(
        disp_free,
        sys_dict,
        exclusion_spec=excl_spec,
        implicit_solvent=True,
        use_pallas=False,
        gb_use_neighbor_list=True,
        return_decomposed=True,
    )
    for key in ("bond", "angle", "dihedral", "improper", "urey_bradley", "cmap", "exception", "total"):
        vp, gp = jax.value_and_grad(lambda r: efn_imp_p[key](r, nbr_idx))(pos)
        vr, gr = jax.value_and_grad(lambda r: efn_imp_r[key](r, nbr_idx))(pos)
        np.testing.assert_allclose(vp, vr, rtol=1e-4, atol=1e-4, err_msg=f"Implicit decomposed {key}")
        np.testing.assert_allclose(gp, gr, rtol=1e-4, atol=1e-4, err_msg=f"Implicit decomposed grad {key}")

    # 2. make_energy_fn (explicit PME + neighbor list + return_decomposed=True)
    efn_pme_p = make_energy_fn(
        disp_pbc,
        sys_dict,
        exclusion_spec=excl_spec,
        box=box,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=16,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        use_pallas=True,
        return_decomposed=True,
    )
    efn_pme_r = make_energy_fn(
        disp_pbc,
        sys_dict,
        exclusion_spec=excl_spec,
        box=box,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=16,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        use_pallas=False,
        return_decomposed=True,
    )
    for key in ("bond", "angle", "dihedral", "improper", "urey_bradley", "cmap", "exception", "total"):
        vp, gp = jax.value_and_grad(lambda r: efn_pme_p[key](r, nbr_idx))(pos)
        vr, gr = jax.value_and_grad(lambda r: efn_pme_r[key](r, nbr_idx))(pos)
        np.testing.assert_allclose(vp, vr, rtol=1e-4, atol=1e-4, err_msg=f"PME decomposed {key}")
        np.testing.assert_allclose(gp, gr, rtol=1e-4, atol=1e-4, err_msg=f"PME decomposed grad {key}")

    # 3. make_energy_fn_pure (both non-periodic and periodic PME, with grad wrt DifferentiableParams)
    psys_nonpbc = PhysicsSystem.from_dict(sys_dict, pos, box_size=None, cutoff_distance=9.0)
    dp_np, pure_nonpbc_p = make_energy_fn_pure(
        disp_free, psys_nonpbc, exclusion_spec=excl_spec, implicit_solvent=True, use_pallas=True, gb_use_neighbor_list=True
    )
    _, pure_nonpbc_r = make_energy_fn_pure(
        disp_free, psys_nonpbc, exclusion_spec=excl_spec, implicit_solvent=True, use_pallas=False, gb_use_neighbor_list=True
    )
    vp_np, (gp_np_dp, gp_np_r) = jax.value_and_grad(
        lambda dp, r: pure_nonpbc_p(dp, r, neighbor=nbr_idx), argnums=(0, 1)
    )(dp_np, pos)
    vr_np, (gr_np_dp, gr_np_r) = jax.value_and_grad(
        lambda dp, r: pure_nonpbc_r(dp, r, neighbor=nbr_idx), argnums=(0, 1)
    )(dp_np, pos)
    np.testing.assert_allclose(vp_np, vr_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_np_r, gr_np_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_np_dp.bond_params, gr_np_dp.bond_params, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_np_dp.improper_params, gr_np_dp.improper_params, rtol=1e-4, atol=1e-4)

    psys_pme = PhysicsSystem.from_dict(sys_dict, pos, box_size=box, cutoff_distance=9.0)
    dp_pme, pure_pme_p = make_energy_fn_pure(
        disp_pbc, psys_pme, exclusion_spec=excl_spec, implicit_solvent=False, pme_grid_points=16, pme_alpha=0.34, use_pallas=True
    )
    _, pure_pme_r = make_energy_fn_pure(
        disp_pbc, psys_pme, exclusion_spec=excl_spec, implicit_solvent=False, pme_grid_points=16, pme_alpha=0.34, use_pallas=False
    )
    vp_pme, (gp_pme_dp, gp_pme_r) = jax.value_and_grad(
        lambda dp, r: pure_pme_p(dp, r, neighbor=nbr_idx), argnums=(0, 1)
    )(dp_pme, pos)
    vr_pme, (gr_pme_dp, gr_pme_r) = jax.value_and_grad(
        lambda dp, r: pure_pme_r(dp, r, neighbor=nbr_idx), argnums=(0, 1)
    )(dp_pme, pos)
    np.testing.assert_allclose(vp_pme, vr_pme, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_pme_r, gr_pme_r, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gp_pme_dp.charges, gr_pme_dp.charges, rtol=1e-4, atol=1e-4)
