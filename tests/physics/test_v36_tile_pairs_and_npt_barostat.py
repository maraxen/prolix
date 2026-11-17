"""Track 3 (v3.6 P0-D): Morton tile-pair cell-grid/octant-sub-AABB pruning, RF conductor limit, and NPT barostat tests."""

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
from jax.scipy.special import erf, erfc

from prolix.api import ensemble_plan as ep_mod
from prolix.api.ensemble_plan import (
    EnsemblePlan,
    _compute_mc_barostat_dh,
    _compute_morton_npt_vol_constants,
)
from prolix.api.observables import KCAL_MOL_PER_A3_TO_BAR
from prolix.constants import COULOMB_CONSTANT
from prolix.export import export_context
from prolix.knob_surface import KnobSurface, NonbondedControls
from prolix.pallas_kernels import pallas_nonbonded_tile_pairs
from prolix.physics import explicit_corrections
from prolix.physics import neighbor_list as nl_mod
from prolix.physics.nonbonded_config import NeighborListConfig, NonbondedConfig, NonbondedEngine
from prolix.physics.optimization import _lj_switch_weight_and_deriv
from prolix.physics.pallas_tile_pairs import (
    DEFAULT_NPT_BOX_SAFETY,
    _build_tile_pairs_jax_fallback,
    _compute_tile_aabbs,
    _discover_pairs_gpu,
    _morton3d_encode_jax,
    _pallas_bonded_and_tile_pairs_soa,
    estimate_max_tile_pairs,
    pallas_build_morton_tile_pairs_gpu,
    pallas_tile_pairs_param_grads,
    precompute_dedup_excl_edges,
    prepare_fused_bonded_buffers,
    reaction_field_k_c,
)
from prolix.physics.pme import spme_background_energy
from prolix.physics.system import make_bundle_from_system
from prolix.typing import PhysicsSystem


def _dense_nonbonded_f64_reference(
    pos: np.ndarray,
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
    atom_mask: np.ndarray,
    box: np.ndarray,
    *,
    cutoff: float,
    pme_alpha: float = 0.0,
    lj_switch_width: float = 0.0,
    rf_dielectric: float | None = None,
    excl_full_pairs: set[tuple[int, int]] | None = None,
    excl_14_pairs: set[tuple[int, int]] | None = None,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.8333333333333334,
) -> tuple[float, np.ndarray, int]:
    """Exact float64 O(N^2) minimum-image reference returning (energy, forces, n_pairs_within_cutoff)."""
    pos64 = np.asarray(pos, dtype=np.float64)
    q64 = np.asarray(charges, dtype=np.float64)
    s64 = np.asarray(sigmas, dtype=np.float64)
    e64 = np.asarray(epsilons, dtype=np.float64)
    m64 = np.asarray(atom_mask, dtype=bool)
    bx64 = np.asarray(box, dtype=np.float64)
    n = pos64.shape[0]
    ef = excl_full_pairs or set()
    e14 = excl_14_pairs or set()

    # Vectorized pair indices i < j
    ii, jj = np.triu_indices(n, k=1)
    valid_ij = m64[ii] & m64[jj]
    ii = ii[valid_ij]
    jj = jj[valid_ij]

    dv = pos64[ii] - pos64[jj]
    dv = dv - bx64[None, :] * np.round(dv / bx64[None, :])
    r2 = np.sum(dv * dv, axis=-1)
    r = np.sqrt(np.maximum(r2, 1e-30))
    in_cut = r < float(cutoff)
    n_in_cut = int(np.sum(in_cut))

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
        sw_j, dsw_j = _lj_switch_weight_and_deriv(
            jnp.asarray(r, dtype=jnp.float64), float(cutoff), float(lj_switch_width)
        )
        sw = np.asarray(sw_j, dtype=np.float64)
        dsw = np.asarray(dsw_j, dtype=np.float64)
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
        erfc_v = np.asarray(erfc(jnp.asarray(ar, dtype=jnp.float64)), dtype=np.float64)
        erf_v = np.asarray(erf(jnp.asarray(ar, dtype=jnp.float64)), dtype=np.float64)
        two_a_sqrt_pi = 2.0 * float(pme_alpha) / math.sqrt(math.pi)
        gauss_term = two_a_sqrt_pi * np.exp(-(ar * ar)) * inv_r
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
    elif rf_dielectric is not None:
        k_rf, c_rf = reaction_field_k_c(cutoff, rf_dielectric)
        e_el = np.where(in_cut, s_el * qq * (inv_r + k_rf * r2 - c_rf), 0.0)
        f_el = np.where(in_cut, s_el * qq * (inv_r2 - 2.0 * k_rf * r), 0.0)
    else:
        e_el = np.where(in_cut, s_el * qq * inv_r, 0.0)
        f_el = np.where(in_cut, s_el * qq * inv_r2, 0.0)

    e_tot = float(np.sum(e_lj + e_el))
    f_over_r = (f_lj + f_el) * inv_r
    f_vec = f_over_r[:, None] * dv
    forces = np.zeros_like(pos64)
    np.add.at(forces, ii, f_vec)
    np.add.at(forces, jj, -f_vec)
    return e_tot, forces, n_in_cut


def _make_periodic_test_bundle(
    *,
    n_waters: int = 2,
    box_len: float = 24.0,
    seed: int = 0,
    net_charge_shift: float = 0.0,
    pme_alpha: float = 0.34,
    cutoff: float = 8.0,
    dtype: Any = jnp.float32,
):
    """Create a periodic solute + rigid-water MolecularBundle of configurable size and net charge."""
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

    # Place waters on a well-separated 3D lattice avoiding solute and overlaps
    grid_n = max(2, math.ceil((n_waters + 8) ** (1.0 / 3.0)) + 1)
    spacing = box_len / float(grid_n)
    water_coords: list[np.ndarray] = []
    water_idx_list: list[list[int]] = []
    for ix in range(grid_n):
        for iy in range(grid_n):
            for iz in range(grid_n):
                if len(water_idx_list) >= n_waters:
                    break
                ox = (ix + 0.5) * spacing
                oy = (iy + 0.5) * spacing
                oz = (iz + 0.5) * spacing
                d_sol = np.min(np.linalg.norm(solute_pos - np.array([ox, oy, oz]), axis=-1))
                if d_sol < 3.2:
                    continue
                base = n_sol + 3 * len(water_idx_list)
                o_pt = np.array([ox, oy, oz], dtype=np.float64)
                # Exact TIP3P geometry: r_OH = 0.9572 A, r_HH = 1.5139 A
                theta_half = math.asin(1.5139 / (2.0 * 0.9572))
                h1_pt = o_pt + np.array([0.9572 * math.sin(theta_half), 0.9572 * math.cos(theta_half), 0.0])
                h2_pt = o_pt + np.array([-0.9572 * math.sin(theta_half), 0.9572 * math.cos(theta_half), 0.0])
                water_coords.extend([o_pt, h1_pt, h2_pt])
                water_idx_list.append([base, base + 1, base + 2])
            if len(water_idx_list) >= n_waters:
                break
        if len(water_idx_list) >= n_waters:
            break
    assert len(water_idx_list) == n_waters, f"Placed {len(water_idx_list)} < {n_waters} waters"

    pos_np = np.vstack([solute_pos] + [w[None, :] for w in water_coords])
    q_sol = np.array([0.15 + net_charge_shift, -0.15, 0.20, -0.20, 0.10, -0.10], dtype=np.float64)
    q_wat = np.tile(np.array([-0.834, 0.417, 0.417], dtype=np.float64), n_waters)
    sig_sol = np.array([3.4, 3.2, 3.3, 3.1, 3.4, 3.0], dtype=np.float64)
    sig_wat = np.tile(np.array([3.1507, 1.0, 1.0], dtype=np.float64), n_waters)
    eps_sol = np.array([0.10, 0.08, 0.12, 0.09, 0.11, 0.07], dtype=np.float64)
    eps_wat = np.tile(np.array([0.1521, 0.0, 0.0], dtype=np.float64), n_waters)
    m_sol = np.array([12.01, 14.01, 12.01, 16.00, 12.01, 1.008], dtype=np.float64)
    m_wat = np.tile(np.array([15.999, 1.008, 1.008], dtype=np.float64), n_waters)

    bonds = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32)
    bond_params = jnp.array([[1.50, 300.0]] * 5, dtype=dtype)
    angles = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=jnp.int32)
    angle_params = jnp.array([[1.91, 55.0]] * 4, dtype=dtype)
    dihedrals = jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.int32)
    dihedral_params = jnp.array([[2.0, 0.0, 1.2], [3.0, math.pi, 0.8], [1.0, 0.5, 1.5]], dtype=dtype)

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
            "charges": jnp.asarray(np.concatenate([q_sol, q_wat]), dtype=dtype),
            "sigmas": jnp.asarray(np.concatenate([sig_sol, sig_wat]), dtype=dtype),
            "epsilons": jnp.asarray(np.concatenate([eps_sol, eps_wat]), dtype=dtype),
            "masses": jnp.asarray(np.concatenate([m_sol, m_wat]), dtype=dtype),
            "bonds": bonds,
            "bond_params": bond_params,
            "angles": angles,
            "angle_params": angle_params,
            "dihedrals": dihedrals,
            "dihedral_params": dihedral_params,
            "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
            "improper_params": jnp.zeros((0, 3), dtype=dtype),
        },
        positions=jnp.asarray(pos_np, dtype=dtype),
        box_size=box,
        cutoff_distance=cutoff,
    ).replace(
        water_indices=jnp.asarray(water_idx_list, dtype=jnp.int32),
        water_mask=jnp.ones((n_waters,), dtype=bool),
        pme_alpha=pme_alpha,
    )
    return make_bundle_from_system(sys, boundary_condition="periodic", exclusion_spec=excl_spec)


@pytest.mark.fast
def test_octant_jump_sub_aabb_pruning_and_zero_missed_pairs():
    """Sub-AABB octant-jump splitting prunes dead Morton straddle pairs with zero missed cutoff pairs."""
    rng = np.random.default_rng(20261117)
    box_len = 56.0
    cutoff = 9.0
    skin = 2.0
    r_eff = cutoff + skin
    box_np = np.array([box_len, box_len, box_len], dtype=np.float64)

    # Place 64 compact clusters of 8 atoms each (N = 512 = 16 tiles), with sizes NOT aligned to 32
    # around octant midplanes and periodic boundaries so 32-atom Z-order tiles straddle octant jumps.
    anchors = np.array(
        [
            [6.0, 6.0, 6.0],
            [27.2, 6.5, 27.2],
            [6.5, 27.3, 49.5],
            [28.8, 6.5, 6.5],
            [6.5, 28.9, 6.5],
            [28.9, 28.9, 6.5],
            [6.5, 6.5, 28.9],
            [28.9, 6.5, 28.9],
            [49.5, 49.5, 49.5],
            [0.4, 49.5, 0.4],
            [55.6, 0.4, 55.6],
            [14.0, 42.0, 14.0],
            [42.0, 14.0, 42.0],
            [20.0, 20.0, 46.0],
            [46.0, 20.0, 20.0],
            [35.0, 35.0, 35.0],
        ],
        dtype=np.float64,
    )
    counts = [12, 20, 12, 20, 10, 22, 14, 18, 12, 20, 12, 20, 48, 80, 96, 96]
    assert sum(counts) == 512
    # Select distinct sites from a global 28^3 periodic lattice (spacing 2.0 A) nearest each anchor
    # so every atom pair globally across PBC has MIC separation >= 1.72 A while preserving clusters.
    g_ax = (np.arange(28, dtype=np.float64) + 0.5) * 2.0
    gx, gy, gz = np.meshgrid(g_ax, g_ax, g_ax, indexing="ij")
    all_sites = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    used = np.zeros(all_sites.shape[0], dtype=bool)
    pts_list = []
    for anc, cnt in zip(anchors, counts, strict=True):
        dv_s = all_sites - anc[None, :]
        dv_s = dv_s - box_np[None, :] * np.round(dv_s / box_np[None, :])
        d2_s = np.where(used, 1e18, np.sum(dv_s * dv_s, axis=-1))
        chosen = np.argsort(d2_s)[:cnt]
        used[chosen] = True
        jitter = rng.uniform(-0.08, 0.08, size=(cnt, 3))
        pts_list.append(np.mod(all_sites[chosen] + jitter, box_len))
    pos_np = np.vstack(pts_list).astype(np.float32).astype(np.float64)
    n_act = pos_np.shape[0]
    nb = n_act // 32

    q_np = rng.uniform(-0.45, 0.45, size=(n_act,)).astype(np.float32)
    sig_np = rng.uniform(2.8, 3.5, size=(n_act,)).astype(np.float32)
    eps_np = rng.uniform(0.05, 0.16, size=(n_act,)).astype(np.float32)
    mask_np = np.ones((n_act,), dtype=bool)
    # Mark last 5 atoms inactive to test tail masking across sub-AABBs as well
    mask_np[-5:] = False

    ef_pairs = {(0, 1), (2, 3), (10, 25), (100, 101)}
    e14_pairs = {(0, 3), (11, 26), (200, 205)}
    excl_idx = np.full((n_act, 32), -1, dtype=np.int32)
    excl_sv = np.ones((n_act, 32), dtype=np.float32)
    excl_se = np.ones((n_act, 32), dtype=np.float32)
    slot_cnt = np.zeros((n_act,), dtype=np.int32)
    for u, v in ef_pairs:
        excl_idx[u, slot_cnt[u]] = v
        excl_sv[u, slot_cnt[u]] = 0.0
        excl_se[u, slot_cnt[u]] = 0.0
        slot_cnt[u] += 1
    for u, v in e14_pairs:
        excl_idx[u, slot_cnt[u]] = v
        excl_sv[u, slot_cnt[u]] = 0.5
        excl_se[u, slot_cnt[u]] = 0.8333333
        slot_cnt[u] += 1
    dedup = precompute_dedup_excl_edges(
        jnp.asarray(excl_idx), jnp.asarray(excl_sv), jnp.asarray(excl_se), n_act
    )

    pos_b = jnp.asarray(pos_np[None, :, :], dtype=jnp.float32)
    mask_b = jnp.asarray(mask_np[None, :])
    box_b = jnp.asarray(box_np[None, :], dtype=jnp.float32)
    codes = _morton3d_encode_jax(pos_b, box_b, mask_b)
    perm = jnp.argsort(codes, axis=-1).astype(jnp.int32)
    inv_perm = jnp.argsort(perm, axis=-1).astype(jnp.int32)
    pos_m = jnp.take_along_axis(pos_b, perm[:, :, None], axis=1)
    mask_m = jnp.take_along_axis(mask_b.astype(jnp.int32), perm, axis=1)
    center, half_ext, tile_has, sub_c, sub_h, sub_has = _compute_tile_aabbs(pos_m, mask_m, box_b, nb)

    # Compare naive 32-atom AABB pair count vs octant-jump sub-AABB pruned pair count
    dc = np.abs(np.asarray(center[0, :, None, :] - center[0, None, :, :]))
    dc = np.minimum(dc, box_np[None, None, :] - dc)
    gap = np.maximum(0.0, dc - np.asarray(half_ext[0, :, None, :] + half_ext[0, None, :, :]))
    iu, ju = np.triu_indices(nb, k=1)
    naive_hits = int(
        np.sum(
            (np.sum(gap[iu, ju] ** 2, axis=-1) <= (r_eff + 1e-3) ** 2)
            & np.asarray(tile_has[0, iu] > 0)
            & np.asarray(tile_has[0, ju] > 0)
        )
        + np.sum(np.asarray(tile_has[0] > 0))
    )

    max_pairs = nb * (nb + 1) // 2
    r_eff_sq = (r_eff + 1e-3) ** 2
    pti_fb, ptj_fb, ef_fb, e14_fb, n_val_fb = _build_tile_pairs_jax_fallback(
        center, half_ext, tile_has, sub_c, sub_h, sub_has, box_b, inv_perm, dedup, r_eff_sq, max_pairs
    )
    pti_gpu, ptj_gpu, _, _, n_val_gpu = _discover_pairs_gpu(
        1, nb, max_pairs, center, half_ext, tile_has, sub_c, sub_h, sub_has, box_b, r_eff, box_len,
        interpret=True,
    )
    pruned_hits = int(np.asarray(n_val_fb[0]))
    assert int(np.asarray(n_val_gpu[0])) == pruned_hits
    assert pruned_hits < naive_hits, f"Expected sub-AABB pruning ({pruned_hits}) < naive ({naive_hits})"

    # Verify EVERY real atom pair with MIC distance < r_eff is covered by an emitted tile pair
    emitted_set = {
        (int(pti_gpu[0, k]), int(ptj_gpu[0, k])) for k in range(pruned_hits)
    }
    inv_np = np.asarray(inv_perm[0])
    pos_act_f64 = pos_np
    for ai in range(n_act):
        if not mask_np[ai]:
            continue
        for aj in range(ai + 1, n_act):
            if not mask_np[aj]:
                continue
            dv = pos_act_f64[ai] - pos_act_f64[aj]
            dv = dv - box_np * np.round(dv / box_np)
            d = float(np.linalg.norm(dv))
            if d < cutoff:
                ti_a, tj_a = int(inv_np[ai]) // 32, int(inv_np[aj]) // 32
                pair_key = (min(ti_a, tj_a), max(ti_a, tj_a))
                assert pair_key in emitted_set, f"Missed real pair ({ai}, {aj}) at d={d:.4f} A in tile {pair_key}"

    # Verify energy and forces match dense float64 reference to float32 machine precision
    tp_state = pallas_build_morton_tile_pairs_gpu(
        pos_b,
        jnp.asarray(q_np[None, :]),
        jnp.asarray(sig_np[None, :]),
        jnp.asarray(eps_np[None, :]),
        mask_b,
        box_b,
        dedup,
        n_act=n_act,
        max_pairs=max_pairs,
        cutoff=cutoff,
        skin=skin,
        host_min_box=box_len,
        interpret=True,
    )
    bbufs = prepare_fused_bonded_buffers(1)
    pm0 = jnp.take_along_axis(pos_b, tp_state.perm[:, :, None], axis=1)
    e_tp, fx_m, fy_m, fz_m = _pallas_bonded_and_tile_pairs_soa(
        1,
        pm0[0, :, 0],
        pm0[0, :, 1],
        pm0[0, :, 2],
        tp_state.q_m[0],
        tp_state.sig_m[0],
        tp_state.eps_m[0],
        tp_state.mask_m[0],
        tp_state.inv_perm[0],
        tp_state.pair_ti[0],
        tp_state.pair_tj[0],
        tp_state.excl_full_bits.reshape(max_pairs * 32),
        tp_state.excl_14_bits.reshape(max_pairs * 32),
        tp_state.n_valid_pairs.reshape(1),
        box_b.reshape(3),
        jnp.ones((1,), dtype=jnp.float32),
        bbufs,
        n_act=n_act,
        n_act_blocks=nb,
        max_pairs=max_pairs,
        pme_alpha=0.34,
        cutoff=cutoff,
        lj_switch_width=1.0,
        include_pme_excl_erf=True,
        scale_14_vdw=dedup.scale_14_vdw,
        scale_14_elec=dedup.scale_14_elec,
        compute_energy=True,
        interpret=True,
    )
    f_m = np.stack([np.asarray(fx_m), np.asarray(fy_m), np.asarray(fz_m)], axis=-1)
    f_canon = f_m[np.asarray(tp_state.inv_perm[0])]

    # Exact 0.00 ppm check against unpruned all-pairs tile list (skin = box_len -> all nb*(nb+1)//2 tiles)
    tp_all = pallas_build_morton_tile_pairs_gpu(
        pos_b,
        jnp.asarray(q_np[None, :]),
        jnp.asarray(sig_np[None, :]),
        jnp.asarray(eps_np[None, :]),
        mask_b,
        box_b,
        dedup,
        n_act=n_act,
        max_pairs=max_pairs,
        cutoff=cutoff,
        skin=box_len,
        host_min_box=box_len,
        interpret=True,
    )
    assert int(np.asarray(tp_all.n_valid_pairs[0])) == max_pairs
    e_all, fx_all, fy_all, fz_all = _pallas_bonded_and_tile_pairs_soa(
        1,
        pm0[0, :, 0],
        pm0[0, :, 1],
        pm0[0, :, 2],
        tp_all.q_m[0],
        tp_all.sig_m[0],
        tp_all.eps_m[0],
        tp_all.mask_m[0],
        tp_all.inv_perm[0],
        tp_all.pair_ti[0],
        tp_all.pair_tj[0],
        tp_all.excl_full_bits.reshape(max_pairs * 32),
        tp_all.excl_14_bits.reshape(max_pairs * 32),
        tp_all.n_valid_pairs.reshape(1),
        box_b.reshape(3),
        jnp.ones((1,), dtype=jnp.float32),
        bbufs,
        n_act=n_act,
        n_act_blocks=nb,
        max_pairs=max_pairs,
        pme_alpha=0.34,
        cutoff=cutoff,
        lj_switch_width=1.0,
        include_pme_excl_erf=True,
        scale_14_vdw=dedup.scale_14_vdw,
        scale_14_elec=dedup.scale_14_elec,
        compute_energy=True,
        interpret=True,
    )
    f_all_canon = np.stack([np.asarray(fx_all), np.asarray(fy_all), np.asarray(fz_all)], axis=-1)[
        np.asarray(tp_all.inv_perm[0])
    ]
    np.testing.assert_allclose(float(np.asarray(e_tp[0])), float(np.asarray(e_all[0])), rtol=1e-7, atol=1e-4)
    np.testing.assert_allclose(f_canon, f_all_canon, rtol=1e-6, atol=5e-4)

    e_ref, f_ref, n_real_pairs = _dense_nonbonded_f64_reference(
        pos_np,
        q_np,
        sig_np,
        eps_np,
        mask_np,
        box_np,
        cutoff=cutoff,
        pme_alpha=0.34,
        lj_switch_width=1.0,
        excl_full_pairs=ef_pairs,
        excl_14_pairs=e14_pairs,
        scale_14_vdw=float(dedup.scale_14_vdw),
        scale_14_elec=float(dedup.scale_14_elec),
    )
    assert n_real_pairs > 500
    np.testing.assert_allclose(float(np.asarray(e_tp[0])), e_ref, rtol=1e-5, atol=1e-4)
    assert float(np.max(np.abs(f_canon - f_ref)) / np.max(np.abs(f_ref))) < 1e-5
    np.testing.assert_allclose(f_canon, f_ref, rtol=5e-5, atol=5e-3)


@pytest.mark.fast
def test_cell_grid_nb_ge_64_discovery_parity_interpret():
    """3D cell-grid tile discovery activates for nb >= 64 (interpret=True) with zero missed pairs."""
    rng = np.random.default_rng(64066)
    n_act = 66 * 32  # nb = 66 >= 64
    nb = 66
    box_len = 46.0
    cutoff = 7.5
    skin = 2.0
    r_eff = cutoff + skin
    # Verify cell-grid precondition: nb >= 64 and host_min_box (46) >= 3 * (9.5 + 1.0) = 31.5
    assert nb >= 64
    assert box_len >= 3.0 * (r_eff + 1.0)

    # Lattice with jitter + periodic face wrap points so pairs cross all 3 periodic faces
    side = 13  # 13^3 = 2197 >= 2112
    ix, iy, iz = np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij")
    grid_pts = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)[:n_act].astype(np.float64)
    pos_np = np.mod((grid_pts + 0.15) * (box_len / side) + rng.uniform(-0.35, 0.35, size=(n_act, 3)), box_len)
    q_np = rng.uniform(-0.35, 0.35, size=(n_act,)).astype(np.float32)
    sig_np = rng.uniform(2.9, 3.4, size=(n_act,)).astype(np.float32)
    eps_np = rng.uniform(0.04, 0.12, size=(n_act,)).astype(np.float32)
    mask_np = np.ones((n_act,), dtype=bool)
    mask_np[-3:] = False

    box_np = np.array([box_len, box_len, box_len], dtype=np.float64)
    pos_b = jnp.asarray(pos_np[None, :, :], dtype=jnp.float32)
    mask_b = jnp.asarray(mask_np[None, :])
    box_b = jnp.asarray(box_np[None, :], dtype=jnp.float32)
    dedup = precompute_dedup_excl_edges(None, None, None, n_act)
    max_pairs = estimate_max_tile_pairs(nb, cutoff=cutoff, skin=skin)

    codes = _morton3d_encode_jax(pos_b, box_b, mask_b)
    perm = jnp.argsort(codes, axis=-1).astype(jnp.int32)
    inv_perm = jnp.argsort(perm, axis=-1).astype(jnp.int32)
    pos_m = jnp.take_along_axis(pos_b, perm[:, :, None], axis=1)
    mask_m = jnp.take_along_axis(mask_b.astype(jnp.int32), perm, axis=1)
    center, half_ext, tile_has, sub_c, sub_h, sub_has = _compute_tile_aabbs(pos_m, mask_m, box_b, nb)

    r_eff_sq = (r_eff + 1e-3) ** 2
    pti_fb, ptj_fb, _, _, n_val_fb = _build_tile_pairs_jax_fallback(
        center, half_ext, tile_has, sub_c, sub_h, sub_has, box_b, inv_perm, dedup, r_eff_sq, max_pairs
    )
    pti_cg, ptj_cg, _, _, n_val_cg = _discover_pairs_gpu(
        1, nb, max_pairs, center, half_ext, tile_has, sub_c, sub_h, sub_has, box_b, r_eff, box_len,
        interpret=True,
    )
    nf = int(np.asarray(n_val_fb[0]))
    nc = int(np.asarray(n_val_cg[0]))
    assert nc == nf
    assert nc < max_pairs
    set_fb = {(int(pti_fb[0, k]), int(ptj_fb[0, k])) for k in range(nf)}
    set_cg = {(int(pti_cg[0, k]), int(ptj_cg[0, k])) for k in range(nc)}
    assert set_cg == set_fb

    # End-to-end pallas_build_morton_tile_pairs_gpu + force/energy parity against dense float64
    tp = pallas_build_morton_tile_pairs_gpu(
        pos_b,
        jnp.asarray(q_np[None, :]),
        jnp.asarray(sig_np[None, :]),
        jnp.asarray(eps_np[None, :]),
        mask_b,
        box_b,
        dedup,
        n_act=n_act,
        max_pairs=max_pairs,
        cutoff=cutoff,
        skin=skin,
        host_min_box=box_len,
        interpret=True,
    )
    assert int(np.asarray(tp.n_valid_pairs[0])) == nc
    bbufs = prepare_fused_bonded_buffers(1)
    e_tp, fx_m, fy_m, fz_m = _pallas_bonded_and_tile_pairs_soa(
        1,
        pos_m[0, :, 0],
        pos_m[0, :, 1],
        pos_m[0, :, 2],
        tp.q_m[0],
        tp.sig_m[0],
        tp.eps_m[0],
        tp.mask_m[0],
        tp.inv_perm[0],
        tp.pair_ti[0],
        tp.pair_tj[0],
        tp.excl_full_bits.reshape(max_pairs * 32),
        tp.excl_14_bits.reshape(max_pairs * 32),
        tp.n_valid_pairs.reshape(1),
        box_b.reshape(3),
        jnp.ones((1,), dtype=jnp.float32),
        bbufs,
        n_act=n_act,
        n_act_blocks=nb,
        max_pairs=max_pairs,
        pme_alpha=0.34,
        cutoff=cutoff,
        lj_switch_width=0.0,
        include_pme_excl_erf=True,
        compute_energy=True,
        interpret=True,
    )
    f_canon = np.stack([np.asarray(fx_m), np.asarray(fy_m), np.asarray(fz_m)], axis=-1)[
        np.asarray(tp.inv_perm[0])
    ]
    e_ref, f_ref, n_in_cut = _dense_nonbonded_f64_reference(
        pos_np, q_np, sig_np, eps_np, mask_np, box_np, cutoff=cutoff, pme_alpha=0.34
    )
    assert n_in_cut > 5000
    np.testing.assert_allclose(float(np.asarray(e_tp[0])), e_ref, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(f_canon, f_ref, rtol=3e-5, atol=3e-4)


@pytest.mark.fast
def test_reaction_field_conductor_limit_inf_nan_free_and_vjp():
    """rf_dielectric=inf (conductor limit) produces exact NaN-free energies, forces, and param grads."""
    rc = 9.0
    k_inf, c_inf = reaction_field_k_c(rc, math.inf)
    k_big, c_big = reaction_field_k_c(rc, 1e12)
    assert math.isfinite(k_inf)
    assert math.isfinite(c_inf)
    assert math.isclose(k_inf, 1.0 / (2.0 * rc**3), rel_tol=1e-15)
    assert math.isclose(c_inf, 3.0 / (2.0 * rc), rel_tol=1e-15)
    assert math.isclose(k_inf, k_big, rel_tol=1e-11)
    assert math.isclose(c_inf, c_big, rel_tol=1e-11)

    bun = _make_periodic_test_bundle(n_waters=10, box_len=24.0, seed=7, pme_alpha=0.0, cutoff=rc)
    n_act = int(np.asarray(bun.n_atoms))
    nb = max(1, (n_act + 31) // 32)
    n_pad = nb * 32
    box_vec = jnp.diag(bun.box).astype(jnp.float32) if bun.box.ndim == 2 else bun.box.astype(jnp.float32)
    dedup = precompute_dedup_excl_edges(
        bun.excl_dense_indices, bun.excl_dense_scales_vdw, bun.excl_dense_scales_elec, n_act
    )
    tp = pallas_build_morton_tile_pairs_gpu(
        bun.positions[:n_act].astype(jnp.float32),
        bun.charges[:n_act].astype(jnp.float32),
        bun.sigmas[:n_act].astype(jnp.float32),
        bun.epsilons[:n_act].astype(jnp.float32),
        bun.atom_mask[:n_act].astype(jnp.bool_),
        box_vec,
        dedup,
        n_act=n_act,
        max_pairs=32,
        cutoff=rc,
        skin=2.0,
        host_min_box=24.0,
        interpret=True,
    )
    pad_tail = n_pad - n_act
    pos_act = bun.positions[:n_act].astype(jnp.float32)
    if pad_tail > 0:
        pos_act = jnp.pad(pos_act, ((0, pad_tail), (0, 0)))
    pos_m = pos_act[tp.perm[0]]

    bbufs = prepare_fused_bonded_buffers(1)
    e_rf, fx_m, fy_m, fz_m = _pallas_bonded_and_tile_pairs_soa(
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
        tp.excl_full_bits.reshape(32 * 32),
        tp.excl_14_bits.reshape(32 * 32),
        tp.n_valid_pairs.reshape(1),
        box_vec,
        jnp.ones((1,), dtype=jnp.float32),
        bbufs,
        n_act=n_act,
        n_act_blocks=nb,
        max_pairs=32,
        pme_alpha=0.0,
        cutoff=rc,
        rf_dielectric=math.inf,
        scale_14_vdw=dedup.scale_14_vdw,
        scale_14_elec=dedup.scale_14_elec,
        compute_energy=True,
        interpret=True,
    )
    assert bool(jnp.all(jnp.isfinite(e_rf)))
    f_m = np.stack([np.asarray(fx_m), np.asarray(fy_m), np.asarray(fz_m)], axis=-1)
    assert np.all(np.isfinite(f_m))
    f_canon = f_m[np.asarray(tp.inv_perm[0])][:n_act]

    # Build exclusion sets for dense reference check
    ef_set = {
        (int(u), int(v))
        for u, v in zip(
            np.asarray(dedup.full_u)[: dedup.n_full],
            np.asarray(dedup.full_v)[: dedup.n_full],
            strict=True,
        )
    }
    e14_set = {
        (int(u), int(v))
        for u, v in zip(
            np.asarray(dedup.excl14_u)[: dedup.n_14],
            np.asarray(dedup.excl14_v)[: dedup.n_14],
            strict=True,
        )
    }
    e_ref, f_ref, _ = _dense_nonbonded_f64_reference(
        np.asarray(bun.positions[:n_act]),
        np.asarray(bun.charges[:n_act]),
        np.asarray(bun.sigmas[:n_act]),
        np.asarray(bun.epsilons[:n_act]),
        np.asarray(bun.atom_mask[:n_act]),
        np.asarray(box_vec),
        cutoff=rc,
        pme_alpha=0.0,
        rf_dielectric=math.inf,
        excl_full_pairs=ef_set,
        excl_14_pairs=e14_set,
        scale_14_vdw=float(dedup.scale_14_vdw),
        scale_14_elec=float(dedup.scale_14_elec),
    )
    np.testing.assert_allclose(float(np.asarray(e_rf[0])), e_ref, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(f_canon, f_ref, rtol=2e-5, atol=2e-4)

    # Verify parameter gradients at rf_dielectric=math.inf are NaN-free and match finite differences
    nv = int(np.asarray(tp.n_valid_pairs[0]))
    tp_pairs = jnp.stack([tp.pair_ti[0, :nv], tp.pair_tj[0, :nv]], axis=-1)
    dq, dsig, deps, dlam = pallas_tile_pairs_param_grads(
        1,
        (False,) * 10,
        pos_m,
        tp.q_m[0],
        tp.sig_m[0],
        tp.eps_m[0],
        tp.mask_m[0],
        tp_pairs,
        tp.excl_full_bits[0, :nv],
        tp.excl_14_bits[0, :nv],
        box_vec,
        jnp.ones((1,), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        has_box=True,
        pme_alpha=0.0,
        cutoff=rc,
        coulomb_constant=float(COULOMB_CONSTANT),
        lj_switch_width=0.0,
        softcore_alpha=0.5,
        softcore_exponent=1,
        include_pme_excl_erf=False,
        scale_14_vdw=float(dedup.scale_14_vdw),
        scale_14_elec=float(dedup.scale_14_elec),
        rf_dielectric=math.inf,
        interpret=True,
    )
    for g_arr in (dq, dsig, deps, dlam):
        assert bool(jnp.all(jnp.isfinite(g_arr)))

    # Finite-difference check on atom 0 charge in Morton space
    eps_fd = 1e-3
    q_m_np = np.asarray(tp.q_m[0], dtype=np.float64)
    q_plus = q_m_np.copy()
    q_minus = q_m_np.copy()
    idx_m0 = int(np.asarray(tp.inv_perm[0])[0])
    q_plus[idx_m0] += eps_fd
    q_minus[idx_m0] -= eps_fd
    q_c_plus = np.asarray(bun.charges[:n_act], dtype=np.float64).copy()
    q_c_minus = np.asarray(bun.charges[:n_act], dtype=np.float64).copy()
    q_c_plus[0] += eps_fd
    q_c_minus[0] -= eps_fd
    ep_fd, _, _ = _dense_nonbonded_f64_reference(
        np.asarray(bun.positions[:n_act]),
        q_c_plus,
        np.asarray(bun.sigmas[:n_act]),
        np.asarray(bun.epsilons[:n_act]),
        np.asarray(bun.atom_mask[:n_act]),
        np.asarray(box_vec),
        cutoff=rc,
        rf_dielectric=math.inf,
        excl_full_pairs=ef_set,
        excl_14_pairs=e14_set,
        scale_14_vdw=float(dedup.scale_14_vdw),
        scale_14_elec=float(dedup.scale_14_elec),
    )
    em_fd, _, _ = _dense_nonbonded_f64_reference(
        np.asarray(bun.positions[:n_act]),
        q_c_minus,
        np.asarray(bun.sigmas[:n_act]),
        np.asarray(bun.epsilons[:n_act]),
        np.asarray(bun.atom_mask[:n_act]),
        np.asarray(box_vec),
        cutoff=rc,
        rf_dielectric=math.inf,
        excl_full_pairs=ef_set,
        excl_14_pairs=e14_set,
        scale_14_vdw=float(dedup.scale_14_vdw),
        scale_14_elec=float(dedup.scale_14_elec),
    )
    dq_fd = (ep_fd - em_fd) / (2.0 * eps_fd)
    np.testing.assert_allclose(float(np.asarray(dq[idx_m0])), dq_fd, rtol=1e-4, atol=1e-3)

    # Also verify KnobSurface routes reaction_field_dielectric=inf cleanly into EnsemblePlan
    ks_rf = KnobSurface(
        nonbonded=NonbondedControls(
            nonbonded_method="reaction_field",
            reaction_field_dielectric=math.inf,
            cutoff_angstrom=rc,
        )
    )
    plan_rf = EnsemblePlan.from_bundle(
        bun, use_pallas=True, nonbonded_engine="morton_tile_pairs", knob_surface=ks_rf
    )
    entry_rf = plan_rf._build_morton_soa_runner(
        [bun], integration_prefix=None, gamma=5.0, nl_update_every=2, nb_cfg=None
    )
    assert entry_rf["rf_dielectric"] == math.inf
    traj_rf = plan_rf.run(
        n_steps=2, dt=1.0, kT=0.596, gamma=5.0, seed=3, run_mode="inference", use_neighbor_list=True
    )
    assert not isinstance(traj_rf, list)
    assert bool(jnp.all(jnp.isfinite(traj_rf.positions)))


@pytest.mark.fast
def test_ensemble_plan_morton_skin_npt_headroom_and_zero_extra_eval_barostat(monkeypatch):
    """Verify 2.0 A skin default, 1.15x NPT box headroom, and zero redundant e_old evals in _mc_baro."""
    assert math.isclose(DEFAULT_NPT_BOX_SAFETY, 1.15, rel_tol=1e-12)
    cap_nvt = estimate_max_tile_pairs(400, cutoff=9.0, skin=2.0, box_safety=1.0)
    cap_npt = estimate_max_tile_pairs(400, cutoff=9.0, skin=2.0, box_safety=DEFAULT_NPT_BOX_SAFETY)
    assert cap_npt > cap_nvt
    assert cap_npt >= int(cap_nvt * 1.45)

    bun = _make_periodic_test_bundle(n_waters=2, box_len=24.0, seed=19, cutoff=8.0)
    # Default NeighborListConfig has skin=1.0; Morton runner must widen eff_skin to >= 2.0
    nb_cfg = NonbondedConfig(
        engine=NonbondedEngine.NEIGHBOR_LIST,
        neighbor_list=NeighborListConfig(cutoff=8.0, skin=1.0),
    )
    plan_nvt = EnsemblePlan.from_bundle(bun, use_pallas=True, nonbonded_engine="morton_tile_pairs")
    entry_nvt = plan_nvt._build_morton_soa_runner(
        [bun], integration_prefix=None, gamma=5.0, nl_update_every=4, nb_cfg=nb_cfg
    )
    assert math.isclose(entry_nvt["eff_skin"], 2.0, rel_tol=1e-12)
    assert math.isclose(entry_nvt["box_safety"], 1.0, rel_tol=1e-12)

    plan_npt = EnsemblePlan.from_bundle(
        bun,
        use_pallas=True,
        nonbonded_engine="morton_tile_pairs",
        pressure_atm=1.0,
        barostat_interval=2,
    )
    entry_npt = plan_npt._build_morton_soa_runner(
        [bun], integration_prefix=None, gamma=5.0, nl_update_every=4, nb_cfg=nb_cfg
    )
    assert math.isclose(entry_npt["eff_skin"], 2.0, rel_tol=1e-12)
    assert math.isclose(entry_npt["box_safety"], DEFAULT_NPT_BOX_SAFETY, rel_tol=1e-12)

    # Count actual runtime invocations of _pallas_bonded_and_tile_pairs_soa via io_callback
    orig_soa = ep_mod._pallas_bonded_and_tile_pairs_soa
    call_log: list[bool] = []

    def _counting_soa(*args, **kwargs):
        ce = bool(kwargs.get("compute_energy", False))
        e_s, fx_s, fy_s, fz_s = orig_soa(*args, **kwargs)
        jax.experimental.io_callback(
            lambda flag: call_log.append(bool(np.asarray(flag))),
            None,
            jnp.bool_(ce),
            ordered=True,
        )
        return e_s, fx_s, fy_s, fz_s

    monkeypatch.setattr(ep_mod, "_pallas_bonded_and_tile_pairs_soa", _counting_soa)

    # Case A: barostat_interval=3, nl_update_every=5, n_steps=3 -> 1 init call + 3 MD steps + 1 trial = 5 calls total!
    # (Previously with redundant e_old inside _mc_baro this would have been 6 calls with 3 compute_energy=True!)
    call_log.clear()
    plan_k3 = EnsemblePlan.from_bundle(
        bun,
        use_pallas=True,
        nonbonded_engine="morton_tile_pairs",
        pressure_atm=1.0,
        barostat_interval=3,
    )
    traj_k3 = plan_k3.run(
        n_steps=3,
        dt=1.0,
        kT=0.596,
        gamma=5.0,
        seed=42,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=5,
    )
    assert not isinstance(traj_k3, list)
    assert bool(jnp.all(jnp.isfinite(traj_k3.positions)))
    assert "box" in traj_k3.observable_values
    # 1 initial force eval (compute_energy=False) + 3 MD steps (F, F, T) + 1 MC trial (T) = 5 calls, 2 with energy
    assert len(call_log) == 5, f"Expected 5 force/energy calls (1 init + 3 MD + 1 trial), got {len(call_log)}: {call_log}"
    assert call_log == [False, False, False, True, True]

    # Case B: barostat_interval=1, nl_update_every=1, n_steps=2 (k_cur == 1 path!) + mixed precision
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        call_log.clear()
        plan_k1 = EnsemblePlan.from_bundle(
            bun,
            use_pallas=True,
            nonbonded_engine="morton_tile_pairs",
            precision="mixed",
            pressure_atm=1.0,
            barostat_interval=1,
        )
        traj_k1 = plan_k1.run(
            n_steps=2,
            dt=1.0,
            kT=0.596,
            gamma=5.0,
            seed=77,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=1,
        )
        assert not isinstance(traj_k1, list)
        assert traj_k1.positions.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(traj_k1.positions)))
        # 1 init (F) + 2 * (1 MD step (T) + 1 MC trial (T)) = 5 calls
        assert call_log == [False, True, True, True, True]
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


@pytest.mark.fast
def test_mc_barostat_dh_exact_switched_tail_and_charged_plasma_correction():
    """_compute_mc_barostat_dh includes exact switched LJ tail 1/V delta and charged-box plasma 1/V delta."""
    prev_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        # Non-neutral periodic system: net charge Q_net = +1.5 e, cutoff = 8.0 A, lj_switch_width = 1.0 A
        bun = _make_periodic_test_bundle(
            n_waters=4,
            box_len=24.0,
            seed=101,
            net_charge_shift=1.5,
            pme_alpha=0.34,
            cutoff=8.0,
            dtype=jnp.float64,
        )
        n_act = int(np.asarray(bun.n_atoms))
        box_old = jnp.array([24.0, 24.0, 24.0], dtype=jnp.float64)
        dlnv = jnp.asarray(-0.003, dtype=jnp.float64)
        mu = jnp.exp(dlnv / 3.0)
        box_new = box_old * mu
        v_old = jnp.prod(box_old)
        v_new = jnp.prod(box_new)

        c_tail_b, c_net_q_b = _compute_morton_npt_vol_constants(
            box_b=np.asarray(box_old[None, :]),
            sig_b=bun.sigmas[None, :n_act],
            eps_b=bun.epsilons[None, :n_act],
            q_b=bun.charges[None, :n_act],
            am_b=bun.atom_mask[None, :n_act],
            cutoff=8.0,
            lj_switch_width=1.0,
            combining_rule="lorentz_berthelot",
            use_dispersion_correction=True,
            do_pme=True,
            pme_alpha=0.34,
        )
        assert c_tail_b[0] < 0.0
        assert c_net_q_b[0] < 0.0

        # Exact reference tail and neutralizing plasma energies at V_old and V_new
        e_tail_old = float(
            explicit_corrections.lj_dispersion_tail_energy(
                box_old,
                bun.sigmas[:n_act],
                bun.epsilons[:n_act],
                8.0,
                bun.atom_mask[:n_act],
                lj_switch_width=1.0,
            )
        )
        e_tail_new = float(
            explicit_corrections.lj_dispersion_tail_energy(
                box_new,
                bun.sigmas[:n_act],
                bun.epsilons[:n_act],
                8.0,
                bun.atom_mask[:n_act],
                lj_switch_width=1.0,
            )
        )
        e_bg_old = float(
            spme_background_energy(bun.charges[:n_act], bun.atom_mask[:n_act], 0.34, box_old)
        )
        e_bg_new = float(
            spme_background_energy(bun.charges[:n_act], bun.atom_mask[:n_act], 0.34, box_new)
        )

        de_tail_fast = float(c_tail_b[0] * (1.0 / float(v_new) - 1.0 / float(v_old)))
        de_bg_fast = float(c_net_q_b[0] * (1.0 / float(v_new) - 1.0 / float(v_old)))
        assert abs(de_tail_fast) > 1e-6
        assert abs(de_bg_fast) > 1e-4
        np.testing.assert_allclose(de_tail_fast, e_tail_new - e_tail_old, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(de_bg_fast, e_bg_new - e_bg_old, rtol=1e-12, atol=1e-14)

        # Verify _compute_mc_barostat_dh matches exact Delta H (and differs from uncorrected dh!)
        e_old_bare = jnp.asarray([-42.5], dtype=jnp.float32)
        e_new_bare = jnp.asarray([-41.8], dtype=jnp.float32)
        p0_akma = 1.0 * 1.01325 / KCAL_MOL_PER_A3_TO_BAR
        n_mols = 5
        kT = jnp.asarray([0.596], dtype=jnp.float32)

        dh_v36 = float(
            np.asarray(
                _compute_mc_barostat_dh(
                    e_old_bare,
                    e_new_bare,
                    v_old[None],
                    v_new[None],
                    c_tail_b=jnp.asarray(c_tail_b, dtype=jnp.float64),
                    c_net_q_b=jnp.asarray(c_net_q_b, dtype=jnp.float64),
                    p0_akma=p0_akma,
                    n_mols=n_mols,
                    kT=kT,
                    dlnv=dlnv[None],
                )
            )[0]
        )
        dh_ref = (
            float(e_new_bare[0] - e_old_bare[0])
            + (e_tail_new - e_tail_old)
            + (e_bg_new - e_bg_old)
            + p0_akma * float(v_new - v_old)
            - n_mols * float(kT[0]) * float(dlnv)
        )
        dh_uncorrected = (
            float(e_new_bare[0] - e_old_bare[0])
            + p0_akma * float(v_new - v_old)
            - n_mols * float(kT[0]) * float(dlnv)
        )
        assert abs(dh_v36 - dh_uncorrected) > 1e-4
        np.testing.assert_allclose(dh_v36, dh_ref, rtol=1e-6, atol=1e-6)

        # Also verify dispersion_correction=False zeros c_tail_b while keeping c_net_q_b
        c_tail_off, c_net_q_on = _compute_morton_npt_vol_constants(
            box_b=np.asarray(box_old[None, :]),
            sig_b=bun.sigmas[None, :n_act],
            eps_b=bun.epsilons[None, :n_act],
            q_b=bun.charges[None, :n_act],
            am_b=bun.atom_mask[None, :n_act],
            cutoff=8.0,
            lj_switch_width=1.0,
            use_dispersion_correction=False,
            do_pme=True,
            pme_alpha=0.34,
        )
        assert c_tail_off[0] == 0.0
        np.testing.assert_allclose(c_net_q_on[0], c_net_q_b[0], rtol=1e-14)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


@pytest.mark.fast
def test_tile_pairs_export_context_zero_custom_calls():
    """Verify pallas_nonbonded_tile_pairs and pallas_build_morton_tile_pairs_gpu lower cleanly under export_context."""
    bun = _make_periodic_test_bundle(n_waters=2, box_len=24.0, seed=5, cutoff=8.0)
    n_act = int(np.asarray(bun.n_atoms))
    n_pad = 32
    box_vec = jnp.diag(bun.box).astype(jnp.float32) if bun.box.ndim == 2 else bun.box.astype(jnp.float32)
    dedup = precompute_dedup_excl_edges(
        bun.excl_dense_indices, bun.excl_dense_scales_vdw, bun.excl_dense_scales_elec, n_act
    )
    with export_context():
        tp = pallas_build_morton_tile_pairs_gpu(
            bun.positions[:n_act].astype(jnp.float32),
            bun.charges[:n_act].astype(jnp.float32),
            bun.sigmas[:n_act].astype(jnp.float32),
            bun.epsilons[:n_act].astype(jnp.float32),
            bun.atom_mask[:n_act].astype(jnp.bool_),
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
        tp_nl = (jnp.stack([tp.pair_ti[0], tp.pair_tj[0]], axis=-1), tp.excl_full_bits[0], tp.excl_14_bits[0])

        def _exported_fn(pos_in, q_in, sig_in, eps_in):
            return pallas_nonbonded_tile_pairs(
                pos_in,
                q_in,
                sig_in,
                eps_in,
                tp_nl,
                atom_mask=tp.mask_m[0],
                box=box_vec,
                cutoff=8.0,
                pme_alpha=0.34,
                return_forces=True,
            )

        lowered = jax.jit(_exported_fn).lower(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0]).as_text().lower()
        for forbidden in ("triton", "pallas", "mosaic", "tpu_custom_call", "xla.gpu", "lapack", "cublas", "cusolver", "cudnn"):
            assert forbidden not in lowered

        exp = jax.export.export(jax.jit(_exported_fn))(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0])
        mlir_txt = exp.mlir_module()
        custom_targets = set(re.findall(r"stablehlo\.custom_call\s+@([\w.$\-]+)", mlir_txt)) | set(
            re.findall(r'call_target_name\s*=\s*"([^"]+)"', mlir_txt)
        )
        assert not custom_targets

