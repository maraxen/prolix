"""Track 3B (v3.8 Fast-Path Complete): 1D ragged deterministic (hi_i32, lo_i32) forces, geometric LJ & RF."""

from __future__ import annotations

import math
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.constants import COULOMB_CONSTANT
from prolix.export import export_context
from prolix.physics.pallas_bonded import _PTX_ATOM_ADD_S32
from prolix.physics.pallas_gbsa_tile import build_gbsa_tile_pair_list
from prolix.physics.pallas_ragged_queue import (
    _nonbonded_tile_pairs_ragged_reference,
    _pallas_nonbonded_tile_pairs_ragged_core,
    _reaction_field_k_c,
    build_ragged_tile_pair_queue,
    pallas_nonbonded_tile_pairs_ragged,
)

pytestmark = pytest.mark.fast

_FORBIDDEN_EXPORT_SUBSTRINGS = (
    "triton",
    "mosaic",
    "pallas",
    "tpu_custom_call",
    "xla.gpu",
    "lapack",
    "cusolver",
    "cublas",
    "cudnn",
)
_CUSTOM_CALL_RE = re.compile(r"stablehlo\.custom_call\s+@([\w.$\-]+)")
_TARGET_NAME_RE = re.compile(r'call_target_name\s*=\s*"([^"]+)"')


def _assert_portable_stablehlo(hlo_text: str, label: str) -> None:
    low = hlo_text.lower()
    leaked = [s for s in _FORBIDDEN_EXPORT_SUBSTRINGS if s in low]
    assert not leaked, f"{label}: forbidden backend markers in StableHLO: {leaked}"
    targets = set(_CUSTOM_CALL_RE.findall(hlo_text)) | set(_TARGET_NAME_RE.findall(hlo_text))
    assert not targets, f"{label}: unexpected custom_call targets in StableHLO: {sorted(targets)}"


def _dense_ragged_f64_reference(  # noqa: PLR0913
    pos_b: np.ndarray,
    q_b: np.ndarray,
    sig_b: np.ndarray,
    eps_b: np.ndarray,
    mask_b: np.ndarray,
    box_b: np.ndarray,
    idx_12_13: list[tuple[int, int]],
    idx_14: list[tuple[int, int]],
    *,
    cutoff: float,
    pme_alpha: float = 0.0,
    rf_dielectric: float | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    lj_switch_width: float = 0.0,
    scale_14_vdw: float = 0.5,
    scale_14_elec: float = 0.8333333333333334,
    include_pme_excl_erf: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent pure-NumPy IEEE float64 reference over ragged B replicas."""
    B, N, _ = pos_b.shape
    ef_set = {(min(u, v), max(u, v)) for u, v in idx_12_13 if u != v}
    e14_set = {(min(u, v), max(u, v)) for u, v in idx_14 if u != v} - ef_set
    has_rf = (pme_alpha <= 0.0) and (rf_dielectric is not None)
    if has_rf and rf_dielectric is not None:
        k_rf, c_rf = _reaction_field_k_c(cutoff, rf_dielectric)
    else:
        k_rf, c_rf = 0.0, 0.0

    energies = np.zeros((B,), dtype=np.float64)
    forces = np.zeros((B, N, 3), dtype=np.float64)

    for b in range(B):
        pos = np.asarray(pos_b[b], dtype=np.float64)
        q = np.asarray(q_b[b], dtype=np.float64)
        sig = np.asarray(sig_b[b], dtype=np.float64)
        eps = np.asarray(eps_b[b], dtype=np.float64)
        msk = np.asarray(mask_b[b], dtype=bool)
        bx = np.asarray(box_b[b], dtype=np.float64)
        pos_w = pos - bx[None, :] * np.floor(pos / bx[None, :])

        ii, jj = np.triu_indices(N, k=1)
        valid_atoms = msk[ii] & msk[jj]
        ii = ii[valid_atoms]
        jj = jj[valid_atoms]
        if ii.size == 0:
            continue

        is_full = np.array([(int(u), int(v)) in ef_set for u, v in zip(ii, jj, strict=True)], dtype=bool)
        is_14 = (~is_full) & np.array(
            [(int(u), int(v)) in e14_set for u, v in zip(ii, jj, strict=True)], dtype=bool
        )
        if not include_pme_excl_erf:
            keep = ~is_full
            ii = ii[keep]
            jj = jj[keep]
            is_full = is_full[keep]
            is_14 = is_14[keep]

        dv = pos_w[ii] - pos_w[jj]
        dv = dv - bx[None, :] * np.floor(dv / bx[None, :] + 0.5)
        raw_r2 = np.sum(dv * dv, axis=-1)
        in_cut = raw_r2 < (cutoff * cutoff)
        r2 = raw_r2 + 1e-12
        r = np.sqrt(r2)
        inv_r = np.where(in_cut, 1.0 / r, 0.0)
        inv_r2 = inv_r * inv_r

        w_v = np.where(is_full, 0.0, np.where(is_14, scale_14_vdw, 1.0))
        w_e = np.where(is_full, 0.0, np.where(is_14, scale_14_elec, 1.0))

        if lj_combining_rule == "geometric":
            sp = sig[ii] * sig[jj]
            sig_ij = np.where(sp > 0.0, np.sqrt(np.maximum(sp, 0.0)), 0.0)
        else:
            sig_ij = 0.5 * (sig[ii] + sig[jj])

        ep = eps[ii] * eps[jj]
        eps_ij = np.where(ep > 0.0, np.sqrt(np.maximum(ep, 0.0)), 0.0)

        sr = np.where(in_cut & (sig_ij > 0.0), sig_ij * inv_r, 0.0)
        sr6 = sr**6
        sr12 = sr6 * sr6
        v_lj = 4.0 * eps_ij * (sr12 - sr6)
        dv_lj_dr = -24.0 * eps_ij * (2.0 * sr12 - sr6) * inv_r
        if lj_switch_width > 0.0:
            r_sw = cutoff - lj_switch_width
            x = np.clip((r - r_sw) / lj_switch_width, 0.0, 1.0)
            x2 = x * x
            x3 = x2 * x
            sw = 1.0 - x3 * (10.0 - 15.0 * x + 6.0 * x2)
            dsw = np.where(
                (r > r_sw) & (r < cutoff),
                (-30.0 * x2 + 60.0 * x3 - 30.0 * x2 * x2) / lj_switch_width,
                0.0,
            )
            e_lj = w_v * v_lj * sw
            de_lj_dr = w_v * (dv_lj_dr * sw + v_lj * dsw)
        else:
            e_lj = w_v * v_lj
            de_lj_dr = w_v * dv_lj_dr

        qq = float(COULOMB_CONSTANT) * q[ii] * q[jj]
        if pme_alpha > 0.0:
            ar = pme_alpha * r
            erfc_v = np.vectorize(math.erfc, otypes=[np.float64])(ar)
            erf_v = np.vectorize(math.erf, otypes=[np.float64])(ar)
            two_a_pi = 2.0 * pme_alpha / math.sqrt(math.pi)
            exp_t = np.exp(-(ar * ar))
            if include_pme_excl_erf:
                eff_w = w_e - erf_v
                e_c = qq * eff_w * inv_r
                de_c_dr = qq * (-eff_w * inv_r2 - two_a_pi * exp_t * inv_r)
            else:
                e_c = qq * w_e * erfc_v * inv_r
                de_c_dr = qq * w_e * (-erfc_v * inv_r2 - two_a_pi * exp_t * inv_r)
        elif has_rf:
            vc = r >= 1e-4
            e_c = np.where(vc, qq * w_e * (inv_r + k_rf * r2 - c_rf), 0.0)
            de_c_dr = np.where(vc, qq * w_e * (2.0 * k_rf * r - inv_r2), 0.0)
        else:
            vc = r >= 1e-4
            e_c = np.where(vc, qq * w_e * inv_r, 0.0)
            de_c_dr = np.where(vc, -qq * w_e * inv_r2, 0.0)

        energies[b] = float(np.sum(np.where(in_cut, e_lj + e_c, 0.0)))
        f_sc = np.where(in_cut, -(de_lj_dr + de_c_dr) * inv_r, 0.0)
        f_vec = f_sc[:, None] * dv
        np.add.at(forces[b], ii, f_vec)
        np.add.at(forces[b], jj, -f_vec)

    return energies, forces


def _make_ragged_b3_batch(seed: int = 20261118) -> dict[str, Any]:
    """Create a cross-size B=3 ragged batch (n_reals=[29, 61, 95], N=100 -> 1, 2, 3 active tiles)."""
    rng = np.random.default_rng(seed)
    B, N = 3, 100
    n_reals = [29, 61, 95]
    # 5x5x4 lattice spacing 3.6 A inside boxes [22, 24, 26] A -> rich cross-tile interactions, no blowups
    gx, gy, gz = np.meshgrid(np.arange(5), np.arange(5), np.arange(4), indexing="ij")
    base_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)[:N].astype(np.float64) * 3.6 + 1.1

    pos_np = np.stack(
        [base_pts + rng.uniform(-0.25, 0.25, size=(N, 3)) for _ in range(B)], axis=0
    ).astype(np.float32)
    q_np = rng.uniform(-0.55, 0.55, size=(B, N)).astype(np.float32)
    sig_np = rng.uniform(2.6, 3.6, size=(B, N)).astype(np.float32)
    eps_np = rng.uniform(0.04, 0.22, size=(B, N)).astype(np.float32)
    # Include zero-sigma / zero-epsilon sites (e.g. TIP3P/OPLS polar hydrogens)
    sig_np[0, 3] = 0.0
    eps_np[0, 3] = 0.0
    sig_np[1, 7] = 0.0
    eps_np[2, 11] = 0.0

    mask_np = np.zeros((B, N), dtype=np.int32)
    for b, nr in enumerate(n_reals):
        mask_np[b, :nr] = 1
    box_np = np.array(
        [[22.0, 22.0, 22.0], [24.0, 24.0, 24.0], [26.0, 26.0, 26.0]], dtype=np.float32
    )
    idx_12_13 = [(0, 1), (1, 2), (2, 3), (4, 35), (10, 42)]
    idx_14 = [(0, 3), (1, 4), (5, 36), (12, 66)]

    tl = build_gbsa_tile_pair_list(
        jnp.asarray(pos_np),
        atom_mask=jnp.asarray(mask_np),
        n_real_per_replica=n_reals,
        idx_12_13=jnp.asarray(idx_12_13, dtype=jnp.int32),
        idx_14=jnp.asarray(idx_14, dtype=jnp.int32),
    )
    return {
        "B": B,
        "N": N,
        "n_reals": n_reals,
        "pos_np": pos_np,
        "q_np": q_np,
        "sig_np": sig_np,
        "eps_np": eps_np,
        "mask_np": mask_np,
        "box_np": box_np,
        "idx_12_13": idx_12_13,
        "idx_14": idx_14,
        "tl": tl,
    }


def test_ragged_b3_deterministic_forces_permutation_invariance_and_clash_x64_false():
    """Ragged B=3 permutation invariance, tp_group invariance, repeatability, and 1e4 clash safety."""
    data = _make_ragged_b3_batch(seed=101)
    tl = data["tl"]
    nv_np = np.asarray(tl.num_valid_pairs, dtype=np.int32)
    assert list(nv_np) == [1, 3, 6]  # true ragged CTA counts across B=3

    pos = jnp.asarray(data["pos_np"])
    q = jnp.asarray(data["q_np"])
    sig = jnp.asarray(data["sig_np"])
    eps = jnp.asarray(data["eps_np"])
    mask = jnp.asarray(data["mask_np"])
    box = jnp.asarray(data["box_np"])
    M_pairs = int(tl.tile_pairs.shape[1])

    def _eval_permuted(orders: list[np.ndarray], tpg: int):
        tp_p = np.asarray(tl.tile_pairs).copy()
        ef_p = np.asarray(tl.excl_full_mask).copy()
        e14_p = np.asarray(tl.excl_14_mask).copy()
        for b, ord_b in enumerate(orders):
            nv = int(nv_np[b])
            tp_p[b, :nv] = tp_p[b, ord_b]
            ef_p[b, :nv] = ef_p[b, ord_b]
            e14_p[b, :nv] = e14_p[b, ord_b]
        pq = build_ragged_tile_pair_queue(nv_np, padded_pairs=M_pairs, tp_group=tpg)
        e_out, f_out = pallas_nonbonded_tile_pairs_ragged(
            pos,
            q,
            sig,
            eps,
            mask,
            jnp.asarray(tp_p),
            jnp.asarray(ef_p),
            jnp.asarray(e14_p),
            pair_queue=pq,
            box=box,
            cutoff=9.5,
            rf_dielectric=78.3,
            lj_combining_rule="geometric",
            lj_switch_width=1.0,
            deterministic_forces=True,
            return_forces=True,
            interpret=True,
        )
        return np.asarray(e_out), np.asarray(f_out)

    id_orders = [np.arange(int(nv_np[b])) for b in range(3)]
    rev_orders = [np.arange(int(nv_np[b]))[::-1] for b in range(3)]
    rng = np.random.default_rng(777)
    rand_orders_1 = [rng.permutation(int(nv_np[b])) for b in range(3)]
    rand_orders_2 = [rng.permutation(int(nv_np[b])) for b in range(3)]

    e_ref_det, f_ref_det = _eval_permuted(id_orders, tpg=1)
    # Repeatability check on identical inputs
    _, f_repeat = _eval_permuted(id_orders, tpg=1)
    assert np.array_equal(f_ref_det.view(np.uint32), f_repeat.view(np.uint32))

    # Bitwise uint32 invariance across row permutations AND tp_group in (1, 2, 4)
    for tpg in (1, 2, 4):
        for ords in (id_orders, rev_orders, rand_orders_1, rand_orders_2):
            e_cur, f_cur = _eval_permuted(ords, tpg=tpg)
            np.testing.assert_allclose(e_cur, e_ref_det, rtol=1e-6, atol=1e-5)
            assert np.array_equal(f_cur.view(np.uint32), f_ref_det.view(np.uint32)), (
                f"Bitwise mismatch for tp_group={tpg}"
            )

    # Masked padding atoms must have strictly zero force
    for b, nr in enumerate(data["n_reals"]):
        assert np.all(f_ref_det[b, nr:] == 0.0)

    # Verify > 2e4 kcal/(mol*A) steric clash under jax_enable_x64=False has zero int32 overflow
    prev_x64 = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        pos_clash_np = data["pos_np"].copy()
        # Place unexcluded pair (atom 6, atom 7) in replica 1 at r = 1.65 A -> ~3e4 kcal/(mol*A) LJ force
        pos_clash_np[1, 7] = pos_clash_np[1, 6] + np.array([1.65, 0.0, 0.0], dtype=np.float32)
        sig_clash_np = data["sig_np"].copy()
        sig_clash_np[1, 6] = 3.4
        sig_clash_np[1, 7] = 3.4
        eps_clash_np = data["eps_np"].copy()
        eps_clash_np[1, 6] = 0.20
        eps_clash_np[1, 7] = 0.20

        e_det_c, f_det_c = pallas_nonbonded_tile_pairs_ragged(
            jnp.asarray(pos_clash_np, dtype=jnp.float32),
            jnp.asarray(data["q_np"], dtype=jnp.float32),
            jnp.asarray(sig_clash_np, dtype=jnp.float32),
            jnp.asarray(eps_clash_np, dtype=jnp.float32),
            jnp.asarray(data["mask_np"], dtype=jnp.int32),
            tl.tile_pairs,
            tl.excl_full_mask,
            tl.excl_14_mask,
            num_valid_pairs=nv_np,
            box=jnp.asarray(data["box_np"], dtype=jnp.float32),
            cutoff=9.5,
            rf_dielectric=78.3,
            lj_combining_rule="geometric",
            lj_switch_width=1.0,
            deterministic_forces=True,
            return_forces=True,
            interpret=True,
        )
        e_f64, f_f64 = _dense_ragged_f64_reference(
            pos_clash_np,
            data["q_np"],
            sig_clash_np,
            eps_clash_np,
            data["mask_np"],
            data["box_np"],
            data["idx_12_13"],
            data["idx_14"],
            cutoff=9.5,
            rf_dielectric=78.3,
            lj_combining_rule="geometric",
            lj_switch_width=1.0,
        )
        f_det_np = np.asarray(f_det_c, dtype=np.float64)
        assert float(np.max(np.abs(f_det_np))) > 2.0e4
        np.testing.assert_allclose(np.asarray(e_det_c), e_f64, rtol=1e-5, atol=2e-3)
        np.testing.assert_allclose(f_det_np, f_f64, rtol=1e-5, atol=2e-3)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)


@pytest.mark.parametrize("rf_dielectric", [None, 4.0, 78.3, float("inf")])
@pytest.mark.parametrize("lj_rule", ["lorentz_berthelot", "geometric"])
@pytest.mark.parametrize("det_forces", [False, True])
def test_geometric_and_reaction_field_finite_and_conductor_parity(
    rf_dielectric: float | None,
    lj_rule: str,
    det_forces: bool,
):
    """Verify Lorentz-Berthelot & geometric LJ + finite/conductor RF parity vs IEEE float64."""
    data = _make_ragged_b3_batch(seed=202)
    tl = data["tl"]
    nv_np = np.asarray(tl.num_valid_pairs, dtype=np.int32)

    e_pal, f_pal = pallas_nonbonded_tile_pairs_ragged(
        jnp.asarray(data["pos_np"]),
        jnp.asarray(data["q_np"]),
        jnp.asarray(data["sig_np"]),
        jnp.asarray(data["eps_np"]),
        jnp.asarray(data["mask_np"]),
        tl.tile_pairs,
        tl.excl_full_mask,
        tl.excl_14_mask,
        num_valid_pairs=nv_np,
        box=jnp.asarray(data["box_np"]),
        cutoff=9.0,
        pme_alpha=0.0,
        rf_dielectric=rf_dielectric,
        lj_combining_rule=lj_rule,
        lj_switch_width=1.0,
        deterministic_forces=det_forces,
        return_forces=True,
        interpret=True,
    )
    assert np.all(np.isfinite(np.asarray(e_pal)))
    assert np.all(np.isfinite(np.asarray(f_pal)))

    e_ref, f_ref = _dense_ragged_f64_reference(
        data["pos_np"],
        data["q_np"],
        data["sig_np"],
        data["eps_np"],
        data["mask_np"],
        data["box_np"],
        data["idx_12_13"],
        data["idx_14"],
        cutoff=9.0,
        pme_alpha=0.0,
        rf_dielectric=rf_dielectric,
        lj_combining_rule=lj_rule,
        lj_switch_width=1.0,
    )
    np.testing.assert_allclose(np.asarray(e_pal), e_ref, rtol=1e-5, atol=5e-4)
    np.testing.assert_allclose(np.asarray(f_pal), f_ref, rtol=1e-5, atol=5e-4)


def test_geometric_differs_from_lb_and_validation_guards():
    """Confirm geometric != LB on heterogeneous sigmas, conductor k_rf/c_rf exactness, and error guards."""
    rc = 9.0
    k_inf, c_inf = _reaction_field_k_c(rc, float("inf"))
    assert math.isclose(k_inf, 0.5 / (rc**3), rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(c_inf, 1.5 / rc, rel_tol=0.0, abs_tol=1e-15)

    data = _make_ragged_b3_batch(seed=303)
    # Strongly heterogeneous sigmas so arithmetic mean (1.0 + 5.0)/2 = 3.0 != sqrt(5.0) = 2.236
    sig_het = np.where((np.arange(data["N"]) % 2) == 0, 1.5, 4.5)[None, :].repeat(3, axis=0).astype(np.float32)
    tl = data["tl"]

    e_lb = pallas_nonbonded_tile_pairs_ragged(
        jnp.asarray(data["pos_np"]),
        jnp.asarray(data["q_np"]),
        jnp.asarray(sig_het),
        jnp.asarray(data["eps_np"]),
        jnp.asarray(data["mask_np"]),
        tl.tile_pairs,
        tl.excl_full_mask,
        tl.excl_14_mask,
        num_valid_pairs=tl.num_valid_pairs,
        box=jnp.asarray(data["box_np"]),
        cutoff=9.0,
        lj_combining_rule="lorentz_berthelot",
        interpret=True,
    )
    e_geo = pallas_nonbonded_tile_pairs_ragged(
        jnp.asarray(data["pos_np"]),
        jnp.asarray(data["q_np"]),
        jnp.asarray(sig_het),
        jnp.asarray(data["eps_np"]),
        jnp.asarray(data["mask_np"]),
        tl.tile_pairs,
        tl.excl_full_mask,
        tl.excl_14_mask,
        num_valid_pairs=tl.num_valid_pairs,
        box=jnp.asarray(data["box_np"]),
        cutoff=9.0,
        lj_combining_rule="geometric",
        interpret=True,
    )
    assert float(np.max(np.abs(np.asarray(e_lb) - np.asarray(e_geo)))) > 0.5

    with pytest.raises(ValueError, match="Unsupported lj_combining_rule"):
        pallas_nonbonded_tile_pairs_ragged(
            jnp.asarray(data["pos_np"]),
            jnp.asarray(data["q_np"]),
            jnp.asarray(sig_het),
            jnp.asarray(data["eps_np"]),
            jnp.asarray(data["mask_np"]),
            tl.tile_pairs,
            tl.excl_full_mask,
            tl.excl_14_mask,
            lj_combining_rule="waldman_hagler",
            interpret=True,
        )

    with pytest.raises(ValueError, match="rf_dielectric must be positive"):
        pallas_nonbonded_tile_pairs_ragged(
            jnp.asarray(data["pos_np"]),
            jnp.asarray(data["q_np"]),
            jnp.asarray(sig_het),
            jnp.asarray(data["eps_np"]),
            jnp.asarray(data["mask_np"]),
            tl.tile_pairs,
            tl.excl_full_mask,
            tl.excl_14_mask,
            rf_dielectric=0.0,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("rf_dielectric", "det_forces"),
    [(78.3, False), (float("inf"), True)],
)
def test_custom_vjp_grads_coords_and_params_geometric_and_rf(
    rf_dielectric: float,
    det_forces: bool,
):
    """Verify jax.grad wrt (positions, charges, sigmas, epsilons) including zero-sigma/eps safety."""
    rng = np.random.default_rng(404)
    B, N = 3, 40
    n_reals = [12, 25, 37]
    gx, gy, gz = np.meshgrid(np.arange(4), np.arange(4), np.arange(3), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)[:N].astype(np.float64) * 3.5 + 1.2
    pos_np = np.stack([pts + rng.uniform(-0.2, 0.2, size=(N, 3)) for _ in range(B)], axis=0).astype(np.float32)
    q_np = rng.uniform(-0.5, 0.5, size=(B, N)).astype(np.float32)
    sig_np = rng.uniform(2.7, 3.4, size=(B, N)).astype(np.float32)
    eps_np = rng.uniform(0.05, 0.18, size=(B, N)).astype(np.float32)
    # Zero sigma & zero epsilon test sites for NaN-free geometric sqrt derivatives
    sig_np[0, 2] = 0.0
    sig_np[1, 5] = 0.0
    eps_np[1, 6] = 0.0
    eps_np[2, 4] = 0.0

    mask_np = np.zeros((B, N), dtype=np.int32)
    for b, nr in enumerate(n_reals):
        mask_np[b, :nr] = 1
    box_np = np.array([[22.0, 22.0, 22.0], [23.0, 23.0, 23.0], [24.0, 24.0, 24.0]], dtype=np.float32)
    tl = build_gbsa_tile_pair_list(
        jnp.asarray(pos_np),
        atom_mask=jnp.asarray(mask_np),
        n_real_per_replica=n_reals,
        idx_12_13=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
        idx_14=jnp.array([[0, 3], [1, 4]], dtype=jnp.int32),
    )
    pq = build_ragged_tile_pair_queue(tl.num_valid_pairs, padded_pairs=tl.tile_pairs.shape[1], tp_group=1)
    weights = jnp.array([1.0, 0.7, 1.3], dtype=jnp.float32)

    pos = jnp.asarray(pos_np)
    q = jnp.asarray(q_np)
    sig = jnp.asarray(sig_np)
    eps = jnp.asarray(eps_np)
    mask = jnp.asarray(mask_np)
    box = jnp.asarray(box_np)

    def _loss_pal(p_, q_, s_, e_):
        ev = pallas_nonbonded_tile_pairs_ragged(
            p_,
            q_,
            s_,
            e_,
            mask,
            tl.tile_pairs,
            tl.excl_full_mask,
            tl.excl_14_mask,
            pair_queue=pq,
            box=box,
            cutoff=9.0,
            rf_dielectric=rf_dielectric,
            lj_combining_rule="geometric",
            lj_switch_width=1.0,
            deterministic_forces=det_forces,
            interpret=True,
        )
        return jnp.sum(ev * weights)

    def _loss_ref(p_, q_, s_, e_):
        box6 = jnp.concatenate([box, jnp.zeros_like(box)], axis=-1)
        ev = _nonbonded_tile_pairs_ragged_reference(
            p_,
            q_,
            s_,
            e_,
            mask,
            tl.tile_pairs,
            tl.excl_full_mask,
            tl.excl_14_mask,
            box6,
            jnp.ones((1,), dtype=p_.dtype),
            jnp.array([0.5], dtype=p_.dtype),
            jnp.array([0.8333333333333334], dtype=p_.dtype),
            has_box=True,
            cutoff=9.0,
            pme_alpha=0.0,
            coulomb_constant=COULOMB_CONSTANT,
            lj_switch_width=1.0,
            softcore_alpha=0.5,
            softcore_exponent=2,
            include_pme_excl_erf=False,
            lj_combining_rule="geometric",
            rf_dielectric=rf_dielectric,
            deterministic_forces=det_forces,
            n_valid_per_replica=pq.n_real_per_replica,
        )
        return jnp.sum(ev * weights)

    val_pal, grads_pal = jax.value_and_grad(_loss_pal, argnums=(0, 1, 2, 3))(pos, q, sig, eps)
    val_ref, grads_ref = jax.value_and_grad(_loss_ref, argnums=(0, 1, 2, 3))(pos, q, sig, eps)

    np.testing.assert_allclose(val_pal, val_ref, rtol=1e-5, atol=5e-4)
    for g_p, g_r in zip(grads_pal, grads_ref, strict=True):
        assert np.all(np.isfinite(np.asarray(g_p)))
        np.testing.assert_allclose(np.asarray(g_p), np.asarray(g_r), rtol=2e-5, atol=5e-4)

    # Spot-check IEEE float64 central finite differences across coords & all 3 parameter arrays
    weights_f64 = np.array([1.0, 0.7, 1.3], dtype=np.float64)
    idx_12_13_fd = [(0, 1), (2, 3)]
    idx_14_fd = [(0, 3), (1, 4)]

    def _loss_f64(p_arr: np.ndarray, q_arr: np.ndarray, s_arr: np.ndarray, e_arr: np.ndarray) -> float:
        ev64, _ = _dense_ragged_f64_reference(
            p_arr,
            q_arr,
            s_arr,
            e_arr,
            mask_np,
            box_np,
            idx_12_13_fd,
            idx_14_fd,
            cutoff=9.0,
            pme_alpha=0.0,
            rf_dielectric=rf_dielectric,
            lj_combining_rule="geometric",
            lj_switch_width=1.0,
        )
        return float(np.sum(ev64 * weights_f64))

    eps_fd = 1e-6
    b_chk, i_chk, d_chk = 2, 9, 1
    p64 = pos_np.astype(np.float64)
    q64 = q_np.astype(np.float64)
    s64 = sig_np.astype(np.float64)
    e64 = eps_np.astype(np.float64)

    p_plus, p_minus = p64.copy(), p64.copy()
    p_plus[b_chk, i_chk, d_chk] += eps_fd
    p_minus[b_chk, i_chk, d_chk] -= eps_fd
    fd_pos = (_loss_f64(p_plus, q64, s64, e64) - _loss_f64(p_minus, q64, s64, e64)) / (2.0 * eps_fd)
    np.testing.assert_allclose(float(grads_pal[0][b_chk, i_chk, d_chk]), fd_pos, rtol=5e-5, atol=5e-5)

    for param_idx, base_arr in enumerate((q64, s64, e64), start=1):
        a_plus, a_minus = base_arr.copy(), base_arr.copy()
        a_plus[b_chk, i_chk] += eps_fd
        a_minus[b_chk, i_chk] -= eps_fd
        args_p = [q64, s64, e64]
        args_m = [q64, s64, e64]
        args_p[param_idx - 1] = a_plus
        args_m[param_idx - 1] = a_minus
        fd_par = (_loss_f64(p64, *args_p) - _loss_f64(p64, *args_m)) / (2.0 * eps_fd)
        np.testing.assert_allclose(float(grads_pal[param_idx][b_chk, i_chk]), fd_par, rtol=5e-5, atol=5e-5)


def test_export_context_stablehlo_portable_and_cuda_triton_ptx_lowering():
    """Verify portable StableHLO export (0 custom_calls) AND CUDA Triton inline PTX atom.global.add.s32."""
    data = _make_ragged_b3_batch(seed=505)
    tl = data["tl"]
    nv_np = np.asarray(tl.num_valid_pairs, dtype=np.int32)
    pq = build_ragged_tile_pair_queue(nv_np, padded_pairs=tl.tile_pairs.shape[1], tp_group=2)

    pos = jnp.asarray(data["pos_np"])
    q = jnp.asarray(data["q_np"])
    sig = jnp.asarray(data["sig_np"])
    eps = jnp.asarray(data["eps_np"])
    mask = jnp.asarray(data["mask_np"])
    box = jnp.asarray(data["box_np"])

    e_pal, f_pal = pallas_nonbonded_tile_pairs_ragged(
        pos,
        q,
        sig,
        eps,
        mask,
        tl.tile_pairs,
        tl.excl_full_mask,
        tl.excl_14_mask,
        pair_queue=pq,
        box=box,
        cutoff=9.0,
        rf_dielectric=float("inf"),
        lj_combining_rule="geometric",
        lj_switch_width=1.0,
        deterministic_forces=True,
        return_forces=True,
        interpret=True,
    )

    with export_context():
        def _exp_energy_and_det_forces(p_, q_, s_, e_):
            return pallas_nonbonded_tile_pairs_ragged(
                p_,
                q_,
                s_,
                e_,
                mask,
                tl.tile_pairs,
                tl.excl_full_mask,
                tl.excl_14_mask,
                pair_queue=pq,
                box=box,
                cutoff=9.0,
                rf_dielectric=float("inf"),
                lj_combining_rule="geometric",
                lj_switch_width=1.0,
                deterministic_forces=True,
                return_forces=True,
            )

        def _exp_grad_all(p_, q_, s_, e_):
            return jax.grad(
                lambda pp, qq, ss, ee: jnp.sum(
                    pallas_nonbonded_tile_pairs_ragged(
                        pp,
                        qq,
                        ss,
                        ee,
                        mask,
                        tl.tile_pairs,
                        tl.excl_full_mask,
                        tl.excl_14_mask,
                        pair_queue=pq,
                        box=box,
                        cutoff=9.0,
                        rf_dielectric=float("inf"),
                        lj_combining_rule="geometric",
                        lj_switch_width=1.0,
                        deterministic_forces=True,
                    )
                ),
                argnums=(0, 1, 2, 3),
            )(p_, q_, s_, e_)

        hlo_det = jax.jit(_exp_energy_and_det_forces).lower(pos, q, sig, eps).as_text()
        hlo_grad = jax.jit(_exp_grad_all).lower(pos, q, sig, eps).as_text()
        _assert_portable_stablehlo(hlo_det, "ragged_det_forces_export")
        _assert_portable_stablehlo(hlo_grad, "ragged_grad_export")

        e_exp, f_exp = jax.jit(_exp_energy_and_det_forces)(pos, q, sig, eps)
        np.testing.assert_allclose(np.asarray(e_exp), np.asarray(e_pal), rtol=1e-5, atol=5e-4)
        # Export deterministic forces use the same (hi_i32, lo_i32) 2^20 quantizer;
        # allow <= 4 ticks of 2^-20 for cross-lowering FMA/div-vs-recip rounding differences,
        # and verify bitwise repeatability of the exported function itself.
        np.testing.assert_allclose(np.asarray(f_exp), np.asarray(f_pal), rtol=1e-6, atol=4.0 / (1 << 20))
        _, f_exp_repeat = jax.jit(_exp_energy_and_det_forces)(pos, q, sig, eps)
        assert np.array_equal(np.asarray(f_exp).view(np.uint32), np.asarray(f_exp_repeat).view(np.uint32))

    # Real CUDA Triton lowering check (interpret=False) verifying inline PTX atom.global.add.s32
    assert "atom.global.add.s32" in _PTX_ATOM_ADD_S32
    box6 = jnp.concatenate([box, jnp.zeros_like(box)], axis=-1)
    cuda_hlo = (
        jax.jit(
            lambda p_, q_, s_, e_: _pallas_nonbonded_tile_pairs_ragged_core(
                p_,
                q_,
                s_,
                e_,
                mask,
                tl.tile_pairs,
                tl.excl_full_mask,
                tl.excl_14_mask,
                box6,
                jnp.ones((1,), dtype=jnp.float32),
                jnp.array([0.5], dtype=jnp.float32),
                jnp.array([0.8333333333333334], dtype=jnp.float32),
                pq,
                has_box=True,
                cutoff=9.0,
                pme_alpha=0.0,
                coulomb_constant=COULOMB_CONSTANT,
                lj_switch_width=1.0,
                softcore_alpha=0.5,
                softcore_exponent=2,
                include_pme_excl_erf=False,
                compute_grad_q=True,
                interpret=False,
                deterministic_forces=True,
                lj_combining_rule="geometric",
                rf_dielectric=float("inf"),
            )
        )
        .trace(pos, q, sig, eps)
        .lower(lowering_platforms=("cuda",))
        .as_text()
    )
    assert "__gpu$xla.gpu.triton" in cuda_hlo
    assert "pallas_nonbonded_tile_pairs_ragged_1d_det" in cuda_hlo
