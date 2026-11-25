"""Track 3B (v3.7 Phase 2): Separate bonded/tile-pair CTAs & (hi_i32, lo_i32) 2^20 deterministic force planes."""

from __future__ import annotations

import math
import re
from typing import Any

import jax
import jax.experimental
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.api import ensemble_plan as ep_mod
from prolix.api.ensemble_plan import EnsemblePlan
from prolix.constants import COULOMB_CONSTANT
from prolix.export import export_context
from prolix.knob_surface import KnobSurface, PallasControls
from prolix.physics import neighbor_list as nl_mod
from prolix.physics.cmap import precompute_cmap_coefficients
from prolix.physics.pallas_bonded import (
    _PTX_ATOM_ADD_S32,
    _reconstruct_hi_lo_i32_to_f32,
    _split_f32_to_hi_lo_i32,
)
from prolix.physics.pallas_tile_pairs import (
    _pallas_bonded_and_tile_pairs_soa,
    pallas_bonded_and_tile_pairs_fused,
    pallas_build_morton_tile_pairs_gpu,
    precompute_dedup_excl_edges,
    prepare_fused_bonded_buffers,
)
from prolix.physics.system import make_bundle_from_system
from prolix.typing import PhysicsSystem


def _dense_nonbonded_f64_ref(
    pos: np.ndarray,
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
    atom_mask: np.ndarray,
    box: np.ndarray,
    *,
    cutoff: float,
    pme_alpha: float = 0.34,
    lj_switch_width: float = 1.0,
    excl_full_pairs: set[tuple[int, int]] | None = None,
    excl_14_pairs: set[tuple[int, int]] | None = None,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 1.0 / 1.2,
) -> tuple[float, np.ndarray]:
    """Exact float64 O(N^2) minimum-image nonbonded reference."""
    pos64 = np.asarray(pos, dtype=np.float64)
    q64 = np.asarray(charges, dtype=np.float64)
    s64 = np.asarray(sigmas, dtype=np.float64)
    e64 = np.asarray(epsilons, dtype=np.float64)
    m64 = np.asarray(atom_mask, dtype=bool)
    bx64 = np.asarray(box, dtype=np.float64)
    ef = excl_full_pairs or set()
    e14 = excl_14_pairs or set()

    ii, jj = np.triu_indices(pos64.shape[0], k=1)
    keep = m64[ii] & m64[jj]
    ii = ii[keep]
    jj = jj[keep]

    dv = pos64[ii] - pos64[jj]
    dv = dv - bx64[None, :] * np.round(dv / bx64[None, :])
    r2 = np.sum(dv * dv, axis=-1)
    r = np.sqrt(np.maximum(r2, 1e-30))
    in_cut = r < float(cutoff)

    is_full = np.array([(int(a), int(b)) in ef for a, b in zip(ii, jj, strict=True)], dtype=bool)
    is_14 = (~is_full) & np.array([(int(a), int(b)) in e14 for a, b in zip(ii, jj, strict=True)], dtype=bool)
    s_vdw = np.where(is_full, 0.0, np.where(is_14, float(scale_14_vdw), 1.0))
    s_el = np.where(is_full, 0.0, np.where(is_14, float(scale_14_elec), 1.0))

    sig_ij = 0.5 * (s64[ii] + s64[jj])
    eps_ij = np.sqrt(np.maximum(e64[ii] * e64[jj], 0.0))
    sr = np.where(in_cut & (s_vdw > 0.0), sig_ij / np.maximum(r, 1e-12), 0.0)
    sr6 = sr**6
    sr12 = sr6 * sr6
    e_lj_raw = 4.0 * eps_ij * (sr12 - sr6)
    f_lj_raw = 24.0 * eps_ij * (2.0 * sr12 - sr6) / np.maximum(r, 1e-12)
    if lj_switch_width > 0.0:
        r_sw = float(cutoff) - float(lj_switch_width)
        x_sw = np.clip((r - r_sw) / float(lj_switch_width), 0.0, 1.0)
        x2 = x_sw * x_sw
        x3 = x2 * x_sw
        sw = 1.0 - x3 * (10.0 - 15.0 * x_sw + 6.0 * x2)
        in_window = (r > r_sw) & (r < float(cutoff))
        dsw = np.where(
            in_window,
            (-30.0 * x2 + 60.0 * x3 - 30.0 * x2 * x2) / float(lj_switch_width),
            0.0,
        )
        e_lj = np.where(in_cut, s_vdw * e_lj_raw * sw, 0.0)
        f_lj = np.where(in_cut, s_vdw * (f_lj_raw * sw - e_lj_raw * dsw), 0.0)
    else:
        e_lj = np.where(in_cut, s_vdw * e_lj_raw, 0.0)
        f_lj = np.where(in_cut, s_vdw * f_lj_raw, 0.0)

    qq = float(COULOMB_CONSTANT) * q64[ii] * q64[jj]
    inv_r = 1.0 / np.maximum(r, 1e-12)
    inv_r2 = inv_r * inv_r
    if pme_alpha > 0.0:
        ar = float(pme_alpha) * r
        erfc_v = np.vectorize(math.erfc, otypes=[np.float64])(ar)
        erf_v = np.vectorize(math.erf, otypes=[np.float64])(ar)
        gauss_term = (2.0 * float(pme_alpha) / math.sqrt(math.pi)) * np.exp(-(ar * ar)) * inv_r
        e_el_std = qq * erfc_v * inv_r
        f_el_std = qq * (erfc_v * inv_r2 + gauss_term)
        c_ex = 1.0 - s_el
        e_el_ex = -c_ex * qq * erf_v * inv_r
        f_el_ex = c_ex * qq * (gauss_term - erf_v * inv_r2)
        use_ex = is_full | is_14
        e_el = np.where(
            in_cut,
            np.where(is_full, 0.0, np.where(is_14, s_el * e_el_std, e_el_std))
            + np.where(use_ex, e_el_ex, 0.0),
            0.0,
        )
        f_el = np.where(
            in_cut,
            np.where(is_full, 0.0, np.where(is_14, s_el * f_el_std, f_el_std))
            + np.where(use_ex, f_el_ex, 0.0),
            0.0,
        )
    else:
        e_el = np.where(in_cut, s_el * qq * inv_r, 0.0)
        f_el = np.where(in_cut, s_el * qq * inv_r2, 0.0)

    e_tot = float(np.sum(e_lj + e_el))
    f_vec = ((f_lj + f_el) * inv_r)[:, None] * dv
    forces = np.zeros_like(pos64)
    np.add.at(forces, ii, f_vec)
    np.add.at(forces, jj, -f_vec)
    return e_tot, forces


def _make_all_six_bonded_buffers(B: int = 1, seed: int = 99):
    """Build FusedBondedBuffers exercising all 6 groups: bonds+UB, angles, dihedrals, impropers, exc14, CMAP."""
    rng = np.random.default_rng(seed)
    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32)
    bond_params = jnp.array([[1.52, 280.0], [1.49, 310.0], [1.53, 260.0], [1.50, 295.0], [1.51, 305.0]], dtype=jnp.float32)
    bond_mask = jnp.ones((5,), dtype=jnp.float32)

    ub_bonds = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)
    ub_params = jnp.array([[2.45, 45.0], [2.48, 50.0]], dtype=jnp.float32)
    ub_mask = jnp.ones((2,), dtype=jnp.float32)

    angles = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=jnp.int32)
    angle_params = jnp.array([[1.91, 55.0], [1.94, 62.0], [1.89, 48.0], [1.92, 58.0]], dtype=jnp.float32)
    angle_mask = jnp.ones((4,), dtype=jnp.float32)

    dihedrals = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.int32)
    dihedral_params = jnp.array([[2.0, 0.2, 1.4], [3.0, math.pi, 0.9], [1.0, 0.6, 1.1]], dtype=jnp.float32)
    dihedral_mask = jnp.ones((3,), dtype=jnp.float32)

    impropers = jnp.array([[0, 1, 2, 4], [1, 2, 3, 5]], dtype=jnp.int32)
    improper_params = jnp.array([[42.0, 0.05], [38.0, -0.04]], dtype=jnp.float32)
    improper_mask = jnp.ones((2,), dtype=jnp.float32)

    exc_pairs = jnp.array([[0, 3], [1, 4], [2, 5]], dtype=jnp.int32)
    exc_sig = jnp.array([3.2, 3.1, 3.3], dtype=jnp.float32)
    exc_eps = jnp.array([0.07, 0.06, 0.08], dtype=jnp.float32)
    exc_qp = jnp.array([-0.04, 0.03, -0.02], dtype=jnp.float32)
    exc_mask = jnp.ones((3,), dtype=jnp.float32)

    grid_24 = jnp.asarray(rng.uniform(-1.2, 1.2, size=(1, 24, 24)), dtype=jnp.float32)
    cmap_coeffs = precompute_cmap_coefficients(grid_24)
    cmap_torsions = jnp.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=jnp.int32)
    cmap_indices = jnp.array([0, 0], dtype=jnp.int32)
    cmap_mask = jnp.ones((2,), dtype=jnp.float32)

    return prepare_fused_bonded_buffers(
        B,
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
        exception_sigmas=exc_sig,
        exception_epsilons=exc_eps,
        exception_chargeprods=exc_qp,
        exception_mask=exc_mask,
        cmap_torsions=cmap_torsions,
        cmap_indices=cmap_indices,
        cmap_coeffs=cmap_coeffs,
        cmap_mask=cmap_mask,
    )


def _make_bundle(n_waters: int = 4, box_len: float = 24.0, seed: int = 11, dtype: Any = jnp.float32):
    rng = np.random.default_rng(seed)
    n_sol = 6
    n_atoms = n_sol + 3 * n_waters
    box = jnp.array([box_len, box_len, box_len], dtype=dtype)
    solute_pos = np.array(
        [
            [3.0, 3.5, 3.5],
            [4.5, 3.5, 3.5],
            [5.2, 4.8, 3.5],
            [6.7, 4.8, 3.7],
            [7.4, 6.1, 4.0],
            [8.9, 6.1, 4.0],
        ],
        dtype=np.float64,
    ) + rng.normal(scale=0.02, size=(6, 3))
    theta_half = math.asin(1.5139 / (2.0 * 0.9572))
    water_coords = []
    water_idx_list = []
    for k in range(n_waters):
        ox = 11.5 + 3.8 * (k % 3)
        oy = 11.5 + 3.8 * ((k // 3) % 3)
        oz = 11.5 + 3.8 * (k // 9)
        o_pt = np.array([ox, oy, oz], dtype=np.float64)
        h1_pt = o_pt + np.array([0.9572 * math.sin(theta_half), 0.9572 * math.cos(theta_half), 0.0])
        h2_pt = o_pt + np.array([-0.9572 * math.sin(theta_half), 0.9572 * math.cos(theta_half), 0.0])
        water_coords.extend([o_pt, h1_pt, h2_pt])
        base = n_sol + 3 * k
        water_idx_list.append([base, base + 1, base + 2])
    pos_np = np.vstack([solute_pos] + [w[None, :] for w in water_coords])
    q_np = np.concatenate([[0.15, -0.15, 0.20, -0.20, 0.10, -0.10], np.tile([-0.834, 0.417, 0.417], n_waters)])
    sig_np = np.concatenate([[3.4, 3.2, 3.3, 3.1, 3.4, 3.0], np.tile([3.1507, 1.0, 1.0], n_waters)])
    eps_np = np.concatenate([[0.10, 0.08, 0.12, 0.09, 0.11, 0.07], np.tile([0.1521, 0.0, 0.0], n_waters)])
    m_np = np.concatenate([[12.01, 14.01, 12.01, 16.0, 12.01, 1.008], np.tile([15.999, 1.008, 1.008], n_waters)])
    ex_12_13 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 2], [1, 3], [2, 4], [3, 5]]
    for o_i, h1_i, h2_i in water_idx_list:
        ex_12_13.extend([[o_i, h1_i], [o_i, h2_i], [h1_i, h2_i]])
    excl_spec = nl_mod.ExclusionSpec(
        n_atoms=n_atoms,
        idx_12_13=jnp.asarray(ex_12_13, dtype=jnp.int32),
        idx_14=jnp.array([[0, 3], [1, 4], [2, 5]], dtype=jnp.int32),
        scale_14_elec=0.8333333333333334,
        scale_14_vdw=0.5,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros((0,), dtype=dtype),
        exception_epsilons=jnp.zeros((0,), dtype=dtype),
        exception_chargeprods=jnp.zeros((0,), dtype=dtype),
    )
    sys = PhysicsSystem.from_dict(
        {
            "charges": jnp.asarray(q_np, dtype=dtype),
            "sigmas": jnp.asarray(sig_np, dtype=dtype),
            "epsilons": jnp.asarray(eps_np, dtype=dtype),
            "masses": jnp.asarray(m_np, dtype=dtype),
            "bonds": jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32),
            "bond_params": jnp.array([[1.50, 300.0]] * 5, dtype=dtype),
            "angles": jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=jnp.int32),
            "angle_params": jnp.array([[1.91, 55.0]] * 4, dtype=dtype),
            "dihedrals": jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.int32),
            "dihedral_params": jnp.array([[2.0, 0.0, 1.2], [3.0, math.pi, 0.8], [1.0, 0.5, 1.5]], dtype=dtype),
            "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
            "improper_params": jnp.zeros((0, 3), dtype=dtype),
        },
        positions=jnp.asarray(pos_np, dtype=dtype),
        box_size=box,
        cutoff_distance=8.0,
    ).replace(
        water_indices=jnp.asarray(water_idx_list, dtype=jnp.int32),
        water_mask=jnp.ones((n_waters,), dtype=bool),
        pme_alpha=0.34,
    )
    return make_bundle_from_system(sys, boundary_condition="periodic", exclusion_spec=excl_spec)


@pytest.mark.fast
def test_bitwise_invariance_under_tile_pairs_row_permutation_and_tp_group():
    """deterministic_forces=True yields bitwise-identical forces across tile_pairs permutations & tp_group."""
    rng = np.random.default_rng(20261118)
    n_act = 128  # 4 Morton tiles -> 10 tile pairs
    nb = 4
    max_pairs = 16
    box_len = 22.0
    box_b = jnp.array([[box_len, box_len, box_len]], dtype=jnp.float32)

    # 5x5x6 lattice spacing ~ 3.5 A so all 4 tiles strongly interact across PBC
    gx, gy, gz = np.meshgrid(np.arange(6), np.arange(5), np.arange(5), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)[:n_act].astype(np.float64)
    pos_np = np.mod((pts + 0.3) * 3.5 + rng.uniform(-0.25, 0.25, size=(n_act, 3)), box_len).astype(np.float32)
    q_np = rng.uniform(-0.6, 0.6, size=(n_act,)).astype(np.float32)
    sig_np = rng.uniform(2.9, 3.5, size=(n_act,)).astype(np.float32)
    eps_np = rng.uniform(0.05, 0.18, size=(n_act,)).astype(np.float32)
    mask_np = np.ones((n_act,), dtype=bool)
    mask_np[-3:] = False

    dedup = precompute_dedup_excl_edges(None, None, None, n_act)
    tp = pallas_build_morton_tile_pairs_gpu(
        jnp.asarray(pos_np[None, :, :]),
        jnp.asarray(q_np[None, :]),
        jnp.asarray(sig_np[None, :]),
        jnp.asarray(eps_np[None, :]),
        jnp.asarray(mask_np[None, :]),
        box_b,
        dedup,
        n_act=n_act,
        max_pairs=max_pairs,
        cutoff=9.0,
        skin=2.0,
        host_min_box=box_len,
        interpret=True,
    )
    n_val = int(np.asarray(tp.n_valid_pairs[0]))
    assert n_val >= 8
    pos_m = jnp.asarray(pos_np)[tp.perm[0]]
    bonded = _make_all_six_bonded_buffers(B=1, seed=42)

    def _run_perm(perm_order: np.ndarray, tp_grp: int, sep_bonded: bool):
        ti_np = np.asarray(tp.pair_ti[0]).copy()
        tj_np = np.asarray(tp.pair_tj[0]).copy()
        ef_np = np.asarray(tp.excl_full_bits[0]).copy()
        e14_np = np.asarray(tp.excl_14_bits[0]).copy()
        ti_np[:n_val] = ti_np[perm_order]
        tj_np[:n_val] = tj_np[perm_order]
        ef_np[:n_val] = ef_np[perm_order]
        e14_np[:n_val] = e14_np[perm_order]
        e_out, fx_m, fy_m, fz_m = _pallas_bonded_and_tile_pairs_soa(
            1,
            pos_m[:, 0],
            pos_m[:, 1],
            pos_m[:, 2],
            tp.q_m[0],
            tp.sig_m[0],
            tp.eps_m[0],
            tp.mask_m[0],
            tp.inv_perm[0],
            jnp.asarray(ti_np),
            jnp.asarray(tj_np),
            jnp.asarray(ef_np).reshape(max_pairs * 32),
            jnp.asarray(e14_np).reshape(max_pairs * 32),
            tp.n_valid_pairs.reshape(1),
            box_b.reshape(3),
            jnp.ones((1,), dtype=jnp.float32),
            bonded,
            n_act=n_act,
            n_act_blocks=nb,
            max_pairs=max_pairs,
            pme_alpha=0.34,
            cutoff=9.0,
            lj_switch_width=1.0,
            include_pme_excl_erf=True,
            tp_group=tp_grp,
            compute_energy=True,
            deterministic_forces=True,
            separate_bonded_kernel=sep_bonded,
            interpret=True,
        )
        return (
            float(np.asarray(e_out[0])),
            np.asarray(fx_m).view(np.uint32),
            np.asarray(fy_m).view(np.uint32),
            np.asarray(fz_m).view(np.uint32),
        )

    base_order = np.arange(n_val, dtype=np.int32)
    e_base, fx_u32_0, fy_u32_0, fz_u32_0 = _run_perm(base_order, tp_grp=4, sep_bonded=True)

    # Reverse order + 4 random shuffles across tp_group in {1, 2, 4, 8} and separate_bonded_kernel in {True, False}
    orders = [
        base_order[::-1],
        rng.permutation(n_val),
        rng.permutation(n_val),
        rng.permutation(n_val),
    ]
    for p_ord in orders:
        for tpg in (1, 2, 4, 8):
            for sep_b in (True, False):
                e_p, fx_u, fy_u, fz_u = _run_perm(p_ord, tp_grp=tpg, sep_bonded=sep_b)
                np.testing.assert_allclose(e_p, e_base, rtol=1e-6, atol=1e-4)
                assert np.array_equal(fx_u, fx_u32_0), f"fx bit mismatch at tp_group={tpg}, sep_b={sep_b}"
                assert np.array_equal(fy_u, fy_u32_0), f"fy bit mismatch at tp_group={tpg}, sep_b={sep_b}"
                assert np.array_equal(fz_u, fz_u32_0), f"fz bit mismatch at tp_group={tpg}, sep_b={sep_b}"


@pytest.mark.fast
def test_deterministic_forces_float64_parity_and_1e4_clash_no_int32_overflow_x64_false():
    """Verify < 1e-5 parity vs float64 and zero int32 overflow on > 10^4 kcal/(mol*A) clashes with x64=False."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        # Direct split/reconstruct round-trip with x64=False including > 10^4 clashes & tie cases
        raw_vals = jnp.array(
            [0.0, -0.0, 0.5 / 1048576.0, 1.5 / 1048576.0, -24567.375, 48912.8125, -1.25e5, 2.5e5],
            dtype=jnp.float32,
        )
        msk = jnp.array([True, True, True, True, True, True, True, False])
        hi, lo = _split_f32_to_hi_lo_i32(raw_vals, msk)
        assert hi.dtype == jnp.int32
        assert lo.dtype == jnp.int32
        assert int(np.asarray(hi[-1])) == 0
        assert int(np.asarray(lo[-1])) == 0
        rec = _reconstruct_hi_lo_i32_to_f32(hi, lo)
        expected = np.where(np.asarray(msk), np.asarray(raw_vals), 0.0)
        # Halfway 0.5 / 2^20 rounds to even 0; 1.5 / 2^20 rounds to even 2 / 2^20
        assert int(np.asarray(lo[2])) == 0
        assert int(np.asarray(lo[3])) == 2
        np.testing.assert_allclose(np.asarray(rec[4:]), expected[4:], rtol=1e-6, atol=1e-6)

        # Full 64-atom periodic tile-pair system with a deliberate ~3.5e4 kcal/(mol*A) steric clash
        rng = np.random.default_rng(1048576)
        n_act = 64
        box_len = 24.0
        box_np = np.array([box_len, box_len, box_len], dtype=np.float64)
        gx, gy, gz = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
        pos_np = ((np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1) + 0.5) * 5.0).astype(np.float32)
        # Place atom 33 at r = 1.62 A from atom 0 (with sigma = 3.5 A -> ~3.2e4 kcal/(mol*A) clash force!)
        pos_np[33] = pos_np[0] + np.array([1.62, 0.0, 0.0], dtype=np.float32)
        pos_np64 = pos_np.astype(np.float64)

        q_np = rng.uniform(-0.4, 0.4, size=(n_act,)).astype(np.float32)
        sig_np = rng.uniform(3.0, 3.4, size=(n_act,)).astype(np.float32)
        sig_np[0] = 3.5
        sig_np[33] = 3.5
        eps_np = rng.uniform(0.08, 0.16, size=(n_act,)).astype(np.float32)
        eps_np[0] = 0.20
        eps_np[33] = 0.20
        mask_np = np.ones((n_act,), dtype=bool)

        dedup = precompute_dedup_excl_edges(None, None, None, n_act)
        box_b = jnp.asarray(box_np[None, :], dtype=jnp.float32)
        tp = pallas_build_morton_tile_pairs_gpu(
            jnp.asarray(pos_np[None, :, :]),
            jnp.asarray(q_np[None, :]),
            jnp.asarray(sig_np[None, :]),
            jnp.asarray(eps_np[None, :]),
            jnp.asarray(mask_np[None, :]),
            box_b,
            dedup,
            n_act=n_act,
            max_pairs=8,
            cutoff=9.0,
            skin=2.0,
            host_min_box=box_len,
            interpret=True,
        )
        pos_m = jnp.asarray(pos_np)[tp.perm[0]]
        bbufs = prepare_fused_bonded_buffers(1)
        e_det, fx_det, fy_det, fz_det = _pallas_bonded_and_tile_pairs_soa(
            1,
            pos_m[:, 0],
            pos_m[:, 1],
            pos_m[:, 2],
            tp.q_m[0],
            tp.sig_m[0],
            tp.eps_m[0],
            tp.mask_m[0],
            tp.inv_perm[0],
            tp.pair_ti[0],
            tp.pair_tj[0],
            tp.excl_full_bits.reshape(8 * 32),
            tp.excl_14_bits.reshape(8 * 32),
            tp.n_valid_pairs.reshape(1),
            box_b.reshape(3),
            jnp.ones((1,), dtype=jnp.float32),
            bbufs,
            n_act=n_act,
            n_act_blocks=2,
            max_pairs=8,
            pme_alpha=0.34,
            cutoff=9.0,
            lj_switch_width=1.0,
            include_pme_excl_erf=True,
            compute_energy=True,
            deterministic_forces=True,
            separate_bonded_kernel=True,
            interpret=True,
        )
        f_canon = np.stack([np.asarray(fx_det), np.asarray(fy_det), np.asarray(fz_det)], axis=-1)[
            np.asarray(tp.inv_perm[0])
        ]
        e_ref, f_ref = _dense_nonbonded_f64_ref(
            pos_np64, q_np, sig_np, eps_np, mask_np, box_np, cutoff=9.0, pme_alpha=0.34, lj_switch_width=1.0
        )
        max_clash_f = float(np.max(np.abs(f_ref)))
        assert max_clash_f > 1.0e4, f"Expected > 10^4 kcal/(mol*A) clash force, got {max_clash_f}"
        np.testing.assert_allclose(float(np.asarray(e_det[0])), e_ref, rtol=1e-5, atol=1e-3)
        rel_err = float(np.max(np.abs(f_canon - f_ref)) / max_clash_f)
        assert rel_err < 1e-5, f"Relative force error {rel_err:.3e} >= 1e-5"
        np.testing.assert_allclose(f_canon, f_ref, rtol=1e-5, atol=2e-3)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


@pytest.mark.fast
def test_separate_bonded_kernel_true_vs_false_all_six_groups_and_pure_nonbonded():
    """Verify exact separate_bonded_kernel=True vs False parity across all 6 bonded groups + pure nonbonded."""
    rng = np.random.default_rng(77)
    B = 2
    n_act = 64
    nb = 2
    max_pairs = 8
    box_len = 24.0
    gx, gy, gz = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
    base_pts = ((np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1) + 0.5) * 4.5).astype(np.float32)
    # Place chain atoms 0..5 at realistic ~1.5 A bond geometry so CMAP/torsions/angles are well-conditioned
    chain = np.array(
        [
            [4.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [6.2, 6.3, 5.1],
            [7.7, 6.4, 5.4],
            [8.4, 7.7, 5.8],
            [9.9, 7.8, 6.1],
        ],
        dtype=np.float32,
    )
    pos0 = base_pts.copy()
    pos0[:6] = chain
    pos1 = np.mod(pos0 + rng.uniform(-0.08, 0.08, size=pos0.shape).astype(np.float32), box_len)
    pos_b = jnp.asarray(np.stack([pos0, pos1], axis=0))
    q_b = jnp.asarray(rng.uniform(-0.35, 0.35, size=(B, n_act)).astype(np.float32))
    sig_b = jnp.asarray(rng.uniform(2.9, 3.3, size=(B, n_act)).astype(np.float32))
    eps_b = jnp.asarray(rng.uniform(0.05, 0.14, size=(B, n_act)).astype(np.float32))
    mask_b = jnp.ones((B, n_act), dtype=jnp.bool_)
    box_b = jnp.broadcast_to(jnp.array([[box_len, box_len, box_len]], dtype=jnp.float32), (B, 3))

    dedup = precompute_dedup_excl_edges(None, None, None, n_act)
    tp = pallas_build_morton_tile_pairs_gpu(
        pos_b, q_b, sig_b, eps_b, mask_b, box_b, dedup,
        n_act=n_act, max_pairs=max_pairs, cutoff=8.5, skin=2.0, host_min_box=box_len, interpret=True,
    )
    pos_m = jnp.take_along_axis(pos_b, tp.perm[:, :, None], axis=1)
    bonded_all6 = _make_all_six_bonded_buffers(B=B, seed=99)
    assert bonded_all6.B_real == 7  # 5 bonds + 2 Urey-Bradley
    assert bonded_all6.A_real == 4
    assert bonded_all6.D_real == 3
    assert bonded_all6.H_real == 2
    assert bonded_all6.E_real == 3
    assert bonded_all6.C_real == 2

    for bonded_case in (bonded_all6, prepare_fused_bonded_buffers(B)):
        for det_mode in (False, True):
            e_sep, fx_sep, fy_sep, fz_sep = _pallas_bonded_and_tile_pairs_soa(
                B,
                pos_m[:, :, 0].reshape(B * n_act),
                pos_m[:, :, 1].reshape(B * n_act),
                pos_m[:, :, 2].reshape(B * n_act),
                tp.q_m.reshape(B * n_act),
                tp.sig_m.reshape(B * n_act),
                tp.eps_m.reshape(B * n_act),
                tp.mask_m.reshape(B * n_act),
                tp.inv_perm.reshape(B * n_act),
                tp.pair_ti.reshape(B * max_pairs),
                tp.pair_tj.reshape(B * max_pairs),
                tp.excl_full_bits.reshape(B * max_pairs * 32),
                tp.excl_14_bits.reshape(B * max_pairs * 32),
                tp.n_valid_pairs.reshape(B),
                box_b.reshape(B * 3),
                jnp.ones((1,), dtype=jnp.float32),
                bonded_case,
                n_act=n_act,
                n_act_blocks=nb,
                max_pairs=max_pairs,
                pme_alpha=0.34,
                cutoff=8.5,
                lj_switch_width=1.0,
                include_pme_excl_erf=True,
                compute_energy=True,
                deterministic_forces=det_mode,
                separate_bonded_kernel=True,
                interpret=True,
            )
            e_fus, fx_fus, fy_fus, fz_fus = _pallas_bonded_and_tile_pairs_soa(
                B,
                pos_m[:, :, 0].reshape(B * n_act),
                pos_m[:, :, 1].reshape(B * n_act),
                pos_m[:, :, 2].reshape(B * n_act),
                tp.q_m.reshape(B * n_act),
                tp.sig_m.reshape(B * n_act),
                tp.eps_m.reshape(B * n_act),
                tp.mask_m.reshape(B * n_act),
                tp.inv_perm.reshape(B * n_act),
                tp.pair_ti.reshape(B * max_pairs),
                tp.pair_tj.reshape(B * max_pairs),
                tp.excl_full_bits.reshape(B * max_pairs * 32),
                tp.excl_14_bits.reshape(B * max_pairs * 32),
                tp.n_valid_pairs.reshape(B),
                box_b.reshape(B * 3),
                jnp.ones((1,), dtype=jnp.float32),
                bonded_case,
                n_act=n_act,
                n_act_blocks=nb,
                max_pairs=max_pairs,
                pme_alpha=0.34,
                cutoff=8.5,
                lj_switch_width=1.0,
                include_pme_excl_erf=True,
                compute_energy=True,
                deterministic_forces=det_mode,
                separate_bonded_kernel=False,
                interpret=True,
            )
            np.testing.assert_allclose(np.asarray(e_sep), np.asarray(e_fus), rtol=1e-7, atol=1e-5)
            if det_mode:
                assert np.array_equal(np.asarray(fx_sep).view(np.uint32), np.asarray(fx_fus).view(np.uint32))
                assert np.array_equal(np.asarray(fy_sep).view(np.uint32), np.asarray(fy_fus).view(np.uint32))
                assert np.array_equal(np.asarray(fz_sep).view(np.uint32), np.asarray(fz_fus).view(np.uint32))
            else:
                np.testing.assert_allclose(np.asarray(fx_sep), np.asarray(fx_fus), rtol=1e-6, atol=1e-5)
                np.testing.assert_allclose(np.asarray(fy_sep), np.asarray(fy_fus), rtol=1e-6, atol=1e-5)
                np.testing.assert_allclose(np.asarray(fz_sep), np.asarray(fz_fus), rtol=1e-6, atol=1e-5)


@pytest.mark.fast
def test_cuda_triton_lowering_inline_ptx_atom_add_s32_and_separate_kernels():
    """Verify CUDA Triton lowering succeeds for disjoint & fused deterministic kernels with PTX atom.global.add.s32."""
    assert "atom.global.add.s32" in _PTX_ATOM_ADD_S32
    bonded = _make_all_six_bonded_buffers(B=1, seed=5)
    n_act = 32
    max_pairs = 4
    px = jnp.zeros((32,), dtype=jnp.float32)
    q = jnp.zeros((32,), dtype=jnp.float32)
    sig = jnp.ones((32,), dtype=jnp.float32)
    eps = jnp.zeros((32,), dtype=jnp.float32)
    msk = jnp.ones((32,), dtype=jnp.int32)
    inv_p = jnp.arange(32, dtype=jnp.int32)
    tpi = jnp.zeros((max_pairs,), dtype=jnp.int32)
    tpj = jnp.zeros((max_pairs,), dtype=jnp.int32)
    ef = jnp.zeros((max_pairs * 32,), dtype=jnp.int32)
    e14 = jnp.zeros((max_pairs * 32,), dtype=jnp.int32)
    nv = jnp.ones((1,), dtype=jnp.int32)
    bx = jnp.array([24.0, 24.0, 24.0], dtype=jnp.float32)
    lam = jnp.ones((1,), dtype=jnp.float32)

    for sep_b in (True, False):
        def _fn(px_in, py_in, pz_in, _sep: bool = sep_b):
            return _pallas_bonded_and_tile_pairs_soa(
                1, px_in, py_in, pz_in, q, sig, eps, msk, inv_p,
                tpi, tpj, ef, e14, nv, bx, lam, bonded,
                n_act=n_act, n_act_blocks=1, max_pairs=max_pairs,
                pme_alpha=0.34, cutoff=8.0, lj_switch_width=1.0,
                include_pme_excl_erf=True, compute_energy=True,
                deterministic_forces=True, separate_bonded_kernel=_sep, interpret=False,
            )

        hlo_cuda = jax.jit(_fn).trace(px, px, px).lower(lowering_platforms=("cuda",)).as_text()
        assert "__gpu$xla.gpu.triton" in hlo_cuda
        if sep_b:
            assert "morton_tile_pairs_direct" in hlo_cuda
            assert "morton_bonded_groups" in hlo_cuda
        else:
            assert "pallas_bonded_and_tile_pairs_fused" in hlo_cuda


@pytest.mark.fast
def test_custom_vjp_grad_and_stablehlo_export_zero_custom_calls():
    """Verify jax.grad wrt coordinates & parameters and zero custom calls under export_context()."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        bun = _make_bundle(n_waters=3, box_len=24.0, seed=31, dtype=jnp.float32)
        n_act = int(np.asarray(bun.n_atoms))
        n_pad = 32
        box_vec = jnp.diag(bun.box).astype(jnp.float32) if bun.box.ndim == 2 else bun.box.astype(jnp.float32)
        dedup = precompute_dedup_excl_edges(
            bun.excl_dense_indices, bun.excl_dense_scales_vdw, bun.excl_dense_scales_elec, n_act
        )
        tp = pallas_build_morton_tile_pairs_gpu(
            bun.positions[:n_act],
            bun.charges[:n_act],
            bun.sigmas[:n_act],
            bun.epsilons[:n_act],
            bun.atom_mask[:n_act],
            box_vec,
            dedup,
            n_act=n_act,
            max_pairs=8,
            cutoff=8.0,
            skin=2.0,
            host_min_box=24.0,
            interpret=True,
        )
        pos_pad = jnp.pad(bun.positions[:n_act].astype(jnp.float32), ((0, n_pad - n_act), (0, 0)))
        pos_m = pos_pad[tp.perm[0]]
        bonded = prepare_fused_bonded_buffers(
            1,
            bonds=bun.bond_idx,
            bond_params=bun.bond_params,
            bond_mask=bun.bond_mask,
            angles=bun.angle_idx,
            angle_params=bun.angle_params,
            angle_mask=bun.angle_mask,
            dihedrals=bun.dihedral_idx,
            dihedral_params=bun.dihedral_params,
            dihedral_mask=bun.dihedral_mask,
        )

        def _energy_fn(p_in, q_in, s_in, e_in):
            res = pallas_bonded_and_tile_pairs_fused(
                p_in,
                q_in,
                s_in,
                e_in,
                tp.mask_m[0],
                tp.inv_perm[0],
                tp.pair_ti[0],
                tp.pair_tj[0],
                tp.excl_full_bits[0],
                tp.excl_14_bits[0],
                tp.n_valid_pairs,
                box_vec,
                bonded,
                n_act=n_act,
                pme_alpha=0.34,
                cutoff=8.0,
                lj_switch_width=1.0,
                include_pme_excl_erf=True,
                scale_14_vdw=float(dedup.scale_14_vdw),
                scale_14_elec=float(dedup.scale_14_elec),
                deterministic_forces=True,
                separate_bonded_kernel=True,
                return_forces=False,
                interpret=True,
            )
            assert not isinstance(res, tuple)
            return res

        val, (gp, gq, gs, ge) = jax.value_and_grad(_energy_fn, argnums=(0, 1, 2, 3))(
            pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0]
        )
        assert bool(jnp.isfinite(val))
        for g_arr in (gp, gq, gs, ge):
            assert bool(jnp.all(jnp.isfinite(g_arr)))

        # Central finite-difference check on active Morton atom coordinate and charge
        m0 = int(np.asarray(tp.inv_perm[0])[0])
        h_pos = 1e-3
        p_plus = pos_m.at[m0, 0].add(h_pos)
        p_minus = pos_m.at[m0, 0].add(-h_pos)
        fd_p = float(_energy_fn(p_plus, tp.q_m[0], tp.sig_m[0], tp.eps_m[0]) - _energy_fn(p_minus, tp.q_m[0], tp.sig_m[0], tp.eps_m[0])) / (2.0 * h_pos)
        np.testing.assert_allclose(float(gp[m0, 0]), fd_p, rtol=1e-3, atol=5e-3)

        h_q = 1e-3
        q_plus = tp.q_m[0].at[m0].add(h_q)
        q_minus = tp.q_m[0].at[m0].add(-h_q)
        fd_q = float(_energy_fn(pos_m, q_plus, tp.sig_m[0], tp.eps_m[0]) - _energy_fn(pos_m, q_minus, tp.sig_m[0], tp.eps_m[0])) / (2.0 * h_q)
        np.testing.assert_allclose(float(gq[m0]), fd_q, rtol=5e-4, atol=2e-3)

        # Portable StableHLO export guardrail check (both forward and grad)
        with export_context():
            def _exp_val_and_grad(p_in, q_in, s_in, e_in):
                v_e, grads_e = jax.value_and_grad(_energy_fn, argnums=(0, 1, 2, 3))(p_in, q_in, s_in, e_in)
                return v_e, grads_e[0], grads_e[1], grads_e[2], grads_e[3]

            lowered_txt = (
                jax.jit(_exp_val_and_grad)
                .lower(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0])
                .as_text()
                .lower()
            )
            for forbidden in ("triton", "pallas", "mosaic", "tpu_custom_call", "xla.gpu", "lapack", "cublas", "cusolver", "cudnn"):
                assert forbidden not in lowered_txt

            exported = jax.export.export(jax.jit(_exp_val_and_grad))(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0])
            mlir_mod = exported.mlir_module()
            targets = set(re.findall(r"stablehlo\.custom_call\s+@([\w.$\-]+)", mlir_mod)) | set(
                re.findall(r'call_target_name\s*=\s*"([^"]+)"', mlir_mod)
            )
            assert not targets
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


@pytest.mark.fast
def test_ensemble_plan_deterministic_forces_wiring(monkeypatch):
    """EnsemblePlan threads deterministic_forces into _pallas_bonded_and_tile_pairs_soa across init & run."""
    bun = _make_bundle(n_waters=2, box_len=24.0, seed=17)
    seen_det_flags: list[bool] = []
    orig_soa = ep_mod._pallas_bonded_and_tile_pairs_soa

    def _spy_soa(*args, **kwargs):
        seen_det_flags.append(bool(kwargs.get("deterministic_forces", False)))
        return orig_soa(*args, **kwargs)

    monkeypatch.setattr(ep_mod, "_pallas_bonded_and_tile_pairs_soa", _spy_soa)

    ks = KnobSurface(pallas=PallasControls(deterministic_forces=True, nonbonded_engine="morton_tile_pairs"))
    plan = EnsemblePlan.from_bundle(bun, use_pallas=True, knob_surface=ks)
    assert plan.deterministic_forces is True

    traj_a = plan.run(
        n_steps=3, dt=1.0, kT=0.596, gamma=5.0, seed=101,
        run_mode="inference", use_neighbor_list=True, nl_update_every=2,
    )
    traj_b = plan.run(
        n_steps=3, dt=1.0, kT=0.596, gamma=5.0, seed=101,
        run_mode="inference", use_neighbor_list=True, nl_update_every=2,
    )
    assert not isinstance(traj_a, list)
    assert not isinstance(traj_b, list)
    assert seen_det_flags
    assert all(seen_det_flags)
    assert np.array_equal(
        np.asarray(traj_a.positions).view(np.uint32),
        np.asarray(traj_b.positions).view(np.uint32),
    )
