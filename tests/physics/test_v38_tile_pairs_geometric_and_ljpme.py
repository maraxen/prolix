"""v3.8 Track 2 validation: Morton 32x32 tile-pair geometric combining, P=6 LJ-PME
direct/exclusion/param-grad kernels, StableHLO portability, and EnsemblePlan wiring.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, cast

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import openmm
import pytest
from openmm import unit as _omm_unit

from prolix.api.ensemble_plan import EnsemblePlan, _spread_and_gather_ljpme_order5_soa
from prolix.constants import COULOMB_CONSTANT
from prolix.export import export_context
from prolix.knob_surface import KnobSurface, NonbondedControls, PallasControls
from prolix.physics.lj_pme import (
    dispersion_charges,
    ewald_p6_long_range_value,
    lj_pme_influence_function,
    lj_pme_reciprocal_energy,
)
from prolix.physics.pallas_tile_pairs import (
    _pallas_bonded_and_tile_pairs_soa,
    estimate_max_tile_pairs,
    pallas_bonded_and_tile_pairs_fused,
    pallas_build_morton_tile_pairs_gpu,
    pallas_tile_pairs_param_grads,
    precompute_dedup_excl_edges,
    prepare_fused_bonded_buffers,
)

unit = cast("Any", _omm_unit)
pytestmark = pytest.mark.fast

FORBIDDEN_STABLEHLO_SUBSTRINGS = (
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


def _build_morton_fixture(
    *,
    n_real: int = 12,
    n_pad_bundle: int = 16,
    seed: int = 2028,
) -> dict[str, Any]:
    """Construct a well-separated periodic test system with zero-sigma/eps edge cases, 1-2 exclusions, and 1-4 pairs."""
    rng = np.random.default_rng(seed)
    box_np = np.array([26.0, 27.0, 28.0], dtype=np.float64)
    # Place atoms on a jittered grid with a short 1-2 excluded bond (0, 1) at ~1.02 A (y << 2.0)
    grid_pts = []
    for ix in range(3):
        for iy in range(2):
            for iz in range(2):
                grid_pts.append([4.0 + 6.0 * ix, 4.5 + 6.5 * iy, 5.0 + 6.5 * iz])
    pos_real = np.array(grid_pts[:n_real], dtype=np.float64) + rng.uniform(-0.25, 0.25, size=(n_real, 3))
    # Short 1-2 covalent bond between 0 and 1 (tests small-y Horner Taylor branch in LJPME!)
    pos_real[1] = pos_real[0] + np.array([0.96, 0.25, -0.18], dtype=np.float64)
    # 1-4 pair between 2 and 3 at ~2.9 A
    pos_real[3] = pos_real[2] + np.array([2.65, -0.85, 0.55], dtype=np.float64)

    pos_np = np.zeros((n_pad_bundle, 3), dtype=np.float64)
    pos_np[:n_real] = pos_real
    mask_np = np.zeros((n_pad_bundle,), dtype=bool)
    mask_np[:n_real] = True

    q_np = np.zeros((n_pad_bundle,), dtype=np.float64)
    q_np[:n_real] = np.array(
        [-0.82, 0.41, 0.41, -0.35, 0.28, -0.19, 0.24, -0.15, 0.31, -0.22, 0.14, -0.06][:n_real]
    )
    # Include zero-epsilon H (atom 1) and zero-sigma particle (atom 5) + ghost padding atoms
    sig_np = np.ones((n_pad_bundle,), dtype=np.float64) * 3.2
    sig_np[:n_real] = np.array(
        [3.15, 0.40, 3.40, 3.05, 3.55, 0.0, 2.95, 3.30, 3.10, 2.85, 3.25, 3.00][:n_real]
    )
    eps_np = np.zeros((n_pad_bundle,), dtype=np.float64)
    eps_np[:n_real] = np.array(
        [0.16, 0.0, 0.12, 0.19, 0.22, 0.15, 0.14, 0.18, 0.11, 0.17, 0.13, 0.10][:n_real]
    )
    # Put non-zero params on padded ghost slots to verify strict mask zeroing
    q_np[n_real:] = 0.9
    sig_np[n_real:] = 3.5
    eps_np[n_real:] = 0.4

    # Dense exclusion table: (0, 1) full exclusion, (4, 6) full exclusion, (2, 3) 1-4 scaled pair
    excl_idx = np.full((n_pad_bundle, 4), -1, dtype=np.int32)
    excl_sv = np.ones((n_pad_bundle, 4), dtype=np.float32)
    excl_se = np.ones((n_pad_bundle, 4), dtype=np.float32)

    excl_idx[0, 0], excl_sv[0, 0], excl_se[0, 0] = 1, 0.0, 0.0
    excl_idx[1, 0], excl_sv[1, 0], excl_se[1, 0] = 0, 0.0, 0.0
    excl_idx[4, 0], excl_sv[4, 0], excl_se[4, 0] = 6, 0.0, 0.0
    excl_idx[6, 0], excl_sv[6, 0], excl_se[6, 0] = 4, 0.0, 0.0
    excl_idx[2, 0], excl_sv[2, 0], excl_se[2, 0] = 3, 0.5, 1.0 / 1.2
    excl_idx[3, 0], excl_sv[3, 0], excl_se[3, 0] = 2, 0.5, 1.0 / 1.2

    dedup_excl = precompute_dedup_excl_edges(excl_idx, excl_sv, excl_se, n_real)
    bonds = jnp.array([[0, 1], [4, 6]], dtype=jnp.int32)
    bond_params = jnp.array([[1.00, 320.0], [5.80, 40.0]], dtype=jnp.float32)
    bond_mask = jnp.array([1.0, 1.0], dtype=jnp.float32)

    return {
        "n_real": n_real,
        "n_pad_bundle": n_pad_bundle,
        "pos": pos_np,
        "q": q_np,
        "sig": sig_np,
        "eps": eps_np,
        "mask": mask_np,
        "box": box_np,
        "excl_idx": excl_idx,
        "excl_sv": excl_sv,
        "excl_se": excl_se,
        "dedup_excl": dedup_excl,
        "bonds": bonds,
        "bond_params": bond_params,
        "bond_mask": bond_mask,
    }


def _reference_direct_energy_f64(  # noqa: PLR0913
    pos: jnp.ndarray,
    q: jnp.ndarray,
    sig: jnp.ndarray,
    eps: jnp.ndarray,
    mask: jnp.ndarray,
    box: jnp.ndarray,
    *,
    n_real: int,
    cutoff: float,
    pme_alpha: float,
    include_pme_excl_erf: bool,
    lj_combining_rule: str,
    lj_pme_alpha: float,
    include_ljpme_excl: bool,
    lj_switch_width: float = 0.0,
    soft_core_lambda: float | jnp.ndarray = 1.0,
    softcore_alpha: float = 0.5,
    softcore_exponent: int = 2,
    bonds: jnp.ndarray | None = None,
    bond_params: jnp.ndarray | None = None,
    bond_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Exact float64 reference matching Morton tile-pair direct + bonded physics."""
    pos_r = pos[:n_real]
    q_r = q[:n_real]
    sig_r = sig[:n_real]
    eps_r = eps[:n_real]
    msk_r = mask[:n_real].astype(bool)
    lam = jnp.asarray(soft_core_lambda, dtype=pos.dtype)
    soft_lj = softcore_alpha * ((1.0 - lam) ** softcore_exponent)

    dr = pos_r[:, None, :] - pos_r[None, :, :]
    dr = dr - box[None, None, :] * jnp.floor(dr / box[None, None, :] + 0.5)
    r2 = jnp.sum(dr * dr, axis=-1)
    iu = jnp.triu(jnp.ones((n_real, n_real), dtype=bool), k=1) & msk_r[:, None] & msk_r[None, :]

    is_full = jnp.zeros((n_real, n_real), dtype=bool).at[0, 1].set(True).at[4, 6].set(True)
    is_14 = jnp.zeros((n_real, n_real), dtype=bool).at[2, 3].set(True)
    wv = jnp.where(is_full, 0.0, jnp.where(is_14, 0.5, 1.0))
    we = jnp.where(is_full, 0.0, jnp.where(is_14, 1.0 / 1.2, 1.0))

    keep_excl = include_pme_excl_erf or (lj_pme_alpha > 0.0 and include_ljpme_excl)
    act = iu & (r2 < cutoff * cutoff) & (True if keep_excl else (~is_full))
    r = jnp.sqrt(jnp.where(act, r2, 1.0) + 1e-12)

    if lj_combining_rule == "geometric":
        sp = jnp.maximum(sig_r[:, None] * sig_r[None, :], 0.0)
        sig_ij = jnp.where(sp > 0.0, jnp.sqrt(jnp.where(sp > 0.0, sp, 1.0)), 0.0)
    else:
        sig_ij = 0.5 * (sig_r[:, None] + sig_r[None, :])
    ep = jnp.maximum(eps_r[:, None] * eps_r[None, :], 0.0)
    eps_ij = jnp.where(ep > 0.0, jnp.sqrt(jnp.where(ep > 0.0, ep, 1.0)), 0.0)
    has_lj = act & (sig_ij > 0.0) & (ep > 0.0)
    sig_s = jnp.where(has_lj, sig_ij, 1.0)

    u6 = (r / sig_s) ** 6
    xi6 = 1.0 / (soft_lj + u6)
    v_lj = jnp.where(has_lj, 4.0 * eps_ij * lam * (xi6 * xi6 - xi6), 0.0)

    if lj_pme_alpha > 0.0:
        b = dispersion_charges(sig_r, eps_r, msk_r)
        b_prod = lam * b[:, None] * b[None, :]
        l6_r = ewald_p6_long_range_value(lj_pme_alpha, jnp.where(act, r2, 1.0))
        l6_rc = ewald_p6_long_range_value(lj_pme_alpha, jnp.asarray(cutoff * cutoff, dtype=pos.dtype))
        u6_c = (cutoff / sig_s) ** 6
        xi6_c = 1.0 / (soft_lj + u6_c)
        v_lj_rc = jnp.where(has_lj, 4.0 * eps_ij * lam * (xi6_c * xi6_c - xi6_c), 0.0)
        w_lr = jnp.ones_like(wv) if include_ljpme_excl else wv
        e_lj = wv * (v_lj - v_lj_rc - b_prod * l6_rc) + w_lr * b_prod * l6_r
    else:
        if lj_switch_width > 0.0:
            r_sw = cutoff - lj_switch_width
            z = jnp.clip((r - r_sw) / lj_switch_width, 0.0, 1.0)
            sw = jnp.where(r <= r_sw, 1.0, jnp.where(r < cutoff, 1.0 - 10.0 * z**3 + 15.0 * z**4 - 6.0 * z**5, 0.0))
        else:
            sw = 1.0
        e_lj = wv * v_lj * sw

    qq = COULOMB_CONSTANT * q_r[:, None] * q_r[None, :]
    if pme_alpha > 0.0:
        if include_pme_excl_erf:
            e_c = qq * (we - jax.lax.erf(pme_alpha * r)) / r
        else:
            e_c = qq * we * jax.lax.erfc(pme_alpha * r) / r
    else:
        e_c = qq * we / r

    e_tot = jnp.sum(jnp.where(act, e_lj + e_c, 0.0))
    if bonds is not None and bond_params is not None and bond_mask is not None:
        bi, bj = bonds[:, 0], bonds[:, 1]
        bd = pos_r[bi] - pos_r[bj]
        bd = bd - box[None, :] * jnp.floor(bd / box[None, :] + 0.5)
        bdist = jnp.sqrt(jnp.sum(bd * bd, axis=-1) + 1e-10)
        df = bdist - bond_params[:, 0].astype(pos.dtype)
        e_tot = e_tot + jnp.sum(0.5 * bond_mask.astype(pos.dtype) * bond_params[:, 1].astype(pos.dtype) * df * df)
    return e_tot


# =============================================================================
# 1. Cross-Track Signature Contract & Combining-Rule Validation
# =============================================================================


def test_cross_track_signature_contract_and_validation() -> None:
    for fn in (
        _pallas_bonded_and_tile_pairs_soa,
        pallas_bonded_and_tile_pairs_fused,
        pallas_tile_pairs_param_grads,
    ):
        params = inspect.signature(fn).parameters
        assert params["lj_combining_rule"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["lj_combining_rule"].default == "lorentz_berthelot"
        assert params["lj_pme_alpha"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["lj_pme_alpha"].default == 0.0
        assert params["include_ljpme_excl"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["include_ljpme_excl"].default is False

    fx = _build_morton_fixture()
    n_act = fx["n_real"]
    max_pairs = estimate_max_tile_pairs(1, cutoff=9.5, skin=2.0)
    tp = pallas_build_morton_tile_pairs_gpu(
        jnp.asarray(fx["pos"], dtype=jnp.float32),
        jnp.asarray(fx["q"], dtype=jnp.float32),
        jnp.asarray(fx["sig"], dtype=jnp.float32),
        jnp.asarray(fx["eps"], dtype=jnp.float32),
        jnp.asarray(fx["mask"]),
        jnp.asarray(fx["box"], dtype=jnp.float32),
        fx["dedup_excl"],
        n_act=n_act,
        max_pairs=max_pairs,
        cutoff=9.5,
        skin=2.0,
        host_min_box=26.0,
        interpret=True,
    )
    pos_pad = jnp.pad(jnp.asarray(fx["pos"][:n_act], dtype=jnp.float32), ((0, 32 - n_act), (0, 0)))
    pos_m = pos_pad[tp.perm[0]]
    with pytest.raises(ValueError, match="Unsupported lj_combining_rule"):
        pallas_bonded_and_tile_pairs_fused(
            pos_m,
            tp.q_m[0],
            tp.sig_m[0],
            tp.eps_m[0],
            tp.mask_m[0],
            tp.inv_perm[0],
            tp.pair_ti[0],
            tp.pair_tj[0],
            tp.excl_full_bits[0],
            tp.excl_14_bits[0],
            tp.n_valid_pairs,
            jnp.asarray(fx["box"], dtype=jnp.float32),
            lj_combining_rule="waldman_hagler",
            interpret=True,
        )


# =============================================================================
# 2. Geometric Combining & Direct LJ-PME Energy/Force/Permutation Parity
# =============================================================================


@pytest.mark.parametrize("lj_combining_rule", ["lorentz_berthelot", "geometric"])
@pytest.mark.parametrize(
    ("lj_pme_alpha", "include_ljpme_excl"),
    [(0.0, False), (0.32, False), (0.32, True)],
)
@pytest.mark.parametrize("separate_bonded_kernel", [True, False])
@pytest.mark.parametrize("deterministic_forces", [False, True])
def test_tile_pairs_geometric_and_ljpme_energy_force_parity(
    lj_combining_rule: str,
    lj_pme_alpha: float,
    include_ljpme_excl: bool,
    separate_bonded_kernel: bool,
    deterministic_forces: bool,
) -> None:
    with jax.enable_x64(True):
        fx = _build_morton_fixture()
        n_act = fx["n_real"]
        cutoff = 9.5
        pme_alpha = 0.34
        # Pass non-zero lj_switch_width=1.5 to verify LJPME ignores switching while non-LJPME applies it
        lj_sw = 1.5
        max_pairs = estimate_max_tile_pairs(1, cutoff=cutoff, skin=2.0)
        tp = pallas_build_morton_tile_pairs_gpu(
            jnp.asarray(fx["pos"], dtype=jnp.float32),
            jnp.asarray(fx["q"], dtype=jnp.float32),
            jnp.asarray(fx["sig"], dtype=jnp.float32),
            jnp.asarray(fx["eps"], dtype=jnp.float32),
            jnp.asarray(fx["mask"]),
            jnp.asarray(fx["box"], dtype=jnp.float32),
            fx["dedup_excl"],
            n_act=n_act,
            max_pairs=max_pairs,
            cutoff=cutoff,
            skin=2.0,
            host_min_box=26.0,
            interpret=True,
        )
        bonded = prepare_fused_bonded_buffers(
            1,
            bonds=fx["bonds"],
            bond_params=fx["bond_params"],
            bond_mask=fx["bond_mask"],
        )
        pos_pad = jnp.pad(jnp.asarray(fx["pos"][:n_act], dtype=jnp.float32), ((0, 32 - n_act), (0, 0)))
        pos_m = pos_pad[tp.perm[0]]

        e_pal, f_pal_m = cast(
            "tuple[jax.Array, jax.Array]",
            pallas_bonded_and_tile_pairs_fused(
                pos_m,
                tp.q_m[0],
                tp.sig_m[0],
                tp.eps_m[0],
                tp.mask_m[0],
                tp.inv_perm[0],
                tp.pair_ti[0],
                tp.pair_tj[0],
                tp.excl_full_bits[0],
                tp.excl_14_bits[0],
                tp.n_valid_pairs,
                jnp.asarray(fx["box"], dtype=jnp.float32),
                bonded,
                n_act=n_act,
                pme_alpha=pme_alpha,
                cutoff=cutoff,
                lj_switch_width=lj_sw,
                include_pme_excl_erf=True,
                lj_combining_rule=lj_combining_rule,
                lj_pme_alpha=lj_pme_alpha,
                include_ljpme_excl=include_ljpme_excl,
                scale_14_vdw=fx["dedup_excl"].scale_14_vdw,
                scale_14_elec=fx["dedup_excl"].scale_14_elec,
                deterministic_forces=deterministic_forces,
                separate_bonded_kernel=separate_bonded_kernel,
                return_forces=True,
                interpret=True,
            ),
        )
        f_pal = f_pal_m[tp.inv_perm[0][:n_act]]

        pos64 = jnp.asarray(fx["pos"], dtype=jnp.float64)
        q64 = jnp.asarray(fx["q"], dtype=jnp.float64)
        sig64 = jnp.asarray(fx["sig"], dtype=jnp.float64)
        eps64 = jnp.asarray(fx["eps"], dtype=jnp.float64)
        msk64 = jnp.asarray(fx["mask"])
        box64 = jnp.asarray(fx["box"], dtype=jnp.float64)

        e_ref, g_pos_ref = jax.value_and_grad(
            lambda p: _reference_direct_energy_f64(
                p,
                q64,
                sig64,
                eps64,
                msk64,
                box64,
                n_real=n_act,
                cutoff=cutoff,
                pme_alpha=pme_alpha,
                include_pme_excl_erf=True,
                lj_combining_rule=lj_combining_rule,
                lj_pme_alpha=lj_pme_alpha,
                include_ljpme_excl=include_ljpme_excl,
                lj_switch_width=lj_sw,
                bonds=fx["bonds"],
                bond_params=fx["bond_params"],
                bond_mask=fx["bond_mask"],
            )
        )(pos64)
        f_ref = -g_pos_ref[:n_act]

        np.testing.assert_allclose(float(e_pal), float(e_ref), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(np.asarray(f_pal), np.asarray(f_ref), rtol=5e-5, atol=5e-5)


def test_deterministic_forces_bitwise_tile_pair_permutation_invariance_ljpme_geometric() -> None:
    """Verify 2^20 fixed-point deterministic_forces=True is bitwise invariant to tile_pair ordering under geometric + LJPME."""
    fx = _build_morton_fixture(n_real=12, n_pad_bundle=16, seed=91)
    # Replicate into 40 atoms (2 Morton 32-atom tiles -> multiple tile pairs)
    base12 = np.asarray(fx["pos"][:12], dtype=np.float64)
    pos40 = np.vstack(
        [
            base12,
            base12 + np.array([0.4, 3.1, -0.3]),
            base12 + np.array([-0.5, -2.8, 3.2]),
            base12[:4] + np.array([2.2, 1.9, -2.5]),
        ]
    )
    pos40 = np.mod(pos40, fx["box"][None, :])
    q40 = np.tile(fx["q"][:10], 4).astype(np.float32)
    sig40 = np.tile(fx["sig"][:10], 4).astype(np.float32)
    eps40 = np.tile(fx["eps"][:10], 4).astype(np.float32)
    msk40 = np.ones((40,), dtype=bool)
    dedup = precompute_dedup_excl_edges(None, None, None, 40)
    max_pairs = estimate_max_tile_pairs(2, cutoff=10.5, skin=2.0)
    tp = pallas_build_morton_tile_pairs_gpu(
        jnp.asarray(pos40, dtype=jnp.float32),
        jnp.asarray(q40),
        jnp.asarray(sig40),
        jnp.asarray(eps40),
        jnp.asarray(msk40),
        jnp.asarray(fx["box"], dtype=jnp.float32),
        dedup,
        n_act=40,
        max_pairs=max_pairs,
        cutoff=10.5,
        skin=2.0,
        host_min_box=26.0,
        interpret=True,
    )
    n_val = int(tp.n_valid_pairs[0])
    assert n_val >= 3
    pos_pad = jnp.pad(jnp.asarray(pos40, dtype=jnp.float32), ((0, 64 - 40), (0, 0)))
    pos_m = pos_pad[tp.perm[0]]

    # Reverse valid tile-pair rows
    rev = np.arange(max_pairs, dtype=np.int32)
    rev[:n_val] = rev[:n_val][::-1]
    rev_j = jnp.asarray(rev)

    _, f1 = cast(
        "tuple[jax.Array, jax.Array]",
        pallas_bonded_and_tile_pairs_fused(
            pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0], tp.mask_m[0], tp.inv_perm[0],
            tp.pair_ti[0], tp.pair_tj[0], tp.excl_full_bits[0], tp.excl_14_bits[0],
            tp.n_valid_pairs, jnp.asarray(fx["box"], dtype=jnp.float32),
            n_act=40, pme_alpha=0.33, cutoff=10.5, include_pme_excl_erf=True,
            lj_combining_rule="geometric", lj_pme_alpha=0.31, include_ljpme_excl=True,
            tp_group=1, deterministic_forces=True, return_forces=True, interpret=True,
        ),
    )
    _, f2 = cast(
        "tuple[jax.Array, jax.Array]",
        pallas_bonded_and_tile_pairs_fused(
            pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0], tp.mask_m[0], tp.inv_perm[0],
            tp.pair_ti[0][rev_j], tp.pair_tj[0][rev_j], tp.excl_full_bits[0][rev_j], tp.excl_14_bits[0][rev_j],
            tp.n_valid_pairs, jnp.asarray(fx["box"], dtype=jnp.float32),
            n_act=40, pme_alpha=0.33, cutoff=10.5, include_pme_excl_erf=True,
            lj_combining_rule="geometric", lj_pme_alpha=0.31, include_ljpme_excl=True,
            tp_group=1, deterministic_forces=True, return_forces=True, interpret=True,
        ),
    )
    np.testing.assert_array_equal(np.asarray(f1), np.asarray(f2))


# =============================================================================
# 3. Analytical @jax.custom_vjp Position & Parameter Gradients (q, sig, eps, lambda)
# =============================================================================


@pytest.mark.parametrize("lj_combining_rule", ["lorentz_berthelot", "geometric"])
@pytest.mark.parametrize(
    ("lj_pme_alpha", "include_ljpme_excl"),
    [(0.0, False), (0.31, False), (0.31, True)],
)
def test_custom_vjp_grads_positions_and_params_match_float64_reference(
    lj_combining_rule: str,
    lj_pme_alpha: float,
    include_ljpme_excl: bool,
) -> None:
    """Verify custom_vjp grad wrt positions, charges, sigmas, epsilons, and soft_core_lambda (including zero sig/eps!)."""
    with jax.enable_x64(True):
        fx = _build_morton_fixture()
        n_act = fx["n_real"]
        cutoff = 9.5
        pme_alpha = 0.34
        lam_val = 0.75
        max_pairs = estimate_max_tile_pairs(1, cutoff=cutoff, skin=2.0)
        tp = pallas_build_morton_tile_pairs_gpu(
            jnp.asarray(fx["pos"], dtype=jnp.float32),
            jnp.asarray(fx["q"], dtype=jnp.float32),
            jnp.asarray(fx["sig"], dtype=jnp.float32),
            jnp.asarray(fx["eps"], dtype=jnp.float32),
            jnp.asarray(fx["mask"]),
            jnp.asarray(fx["box"], dtype=jnp.float32),
            fx["dedup_excl"],
            n_act=n_act,
            max_pairs=max_pairs,
            cutoff=cutoff,
            skin=2.0,
            host_min_box=26.0,
            interpret=True,
        )
        pos_pad = jnp.pad(jnp.asarray(fx["pos"][:n_act], dtype=jnp.float32), ((0, 32 - n_act), (0, 0)))
        pos_m = pos_pad[tp.perm[0]]
        q_m = tp.q_m[0]
        sig_m = tp.sig_m[0]
        eps_m = tp.eps_m[0]
        lam_j = jnp.asarray(lam_val, dtype=jnp.float32)

        def _e_pallas(pm, qm, sm, em, lm):
            return pallas_bonded_and_tile_pairs_fused(
                pm,
                qm,
                sm,
                em,
                tp.mask_m[0],
                tp.inv_perm[0],
                tp.pair_ti[0],
                tp.pair_tj[0],
                tp.excl_full_bits[0],
                tp.excl_14_bits[0],
                tp.n_valid_pairs,
                jnp.asarray(fx["box"], dtype=jnp.float32),
                n_act=n_act,
                pme_alpha=pme_alpha,
                cutoff=cutoff,
                lj_switch_width=1.2,
                soft_core_lambda=lm,
                softcore_alpha=0.5,
                softcore_exponent=2,
                include_pme_excl_erf=True,
                lj_combining_rule=lj_combining_rule,
                lj_pme_alpha=lj_pme_alpha,
                include_ljpme_excl=include_ljpme_excl,
                scale_14_vdw=fx["dedup_excl"].scale_14_vdw,
                scale_14_elec=fx["dedup_excl"].scale_14_elec,
                interpret=True,
            )

        gp_m, gq_m, gs_m, ge_m, glam = jax.grad(_e_pallas, argnums=(0, 1, 2, 3, 4))(
            pos_m, q_m, sig_m, eps_m, lam_j
        )
        assert bool(jnp.all(jnp.isfinite(gp_m)))
        assert bool(jnp.all(jnp.isfinite(gq_m)))
        assert bool(jnp.all(jnp.isfinite(gs_m)))
        assert bool(jnp.all(jnp.isfinite(ge_m)))
        assert bool(jnp.isfinite(glam))

        inv_act = tp.inv_perm[0][:n_act]
        gp = gp_m[inv_act]
        gq = gq_m[inv_act]
        gs = gs_m[inv_act]
        ge = ge_m[inv_act]

        # Atom 1 has eps == 0 -> exact 0 grad wrt eps and sigma; Atom 5 under geometric has sig == 0 -> exact 0 grad wrt sig and eps!
        assert float(ge[1]) == 0.0
        assert float(gs[1]) == 0.0
        if lj_combining_rule == "geometric":
            assert float(gs[5]) == 0.0
            assert float(ge[5]) == 0.0

        # Compare against exact float64 reference autodiff
        gp_ref, gq_ref, gs_ref, ge_ref, glam_ref = jax.grad(
            lambda p, q, s, e, lm: _reference_direct_energy_f64(
                p,
                q,
                s,
                e,
                jnp.asarray(fx["mask"]),
                jnp.asarray(fx["box"], dtype=jnp.float64),
                n_real=n_act,
                cutoff=cutoff,
                pme_alpha=pme_alpha,
                include_pme_excl_erf=True,
                lj_combining_rule=lj_combining_rule,
                lj_pme_alpha=lj_pme_alpha,
                include_ljpme_excl=include_ljpme_excl,
                lj_switch_width=1.2,
                soft_core_lambda=lm,
                softcore_alpha=0.5,
                softcore_exponent=2,
            ),
            argnums=(0, 1, 2, 3, 4),
        )(
            jnp.asarray(fx["pos"], dtype=jnp.float64),
            jnp.asarray(fx["q"], dtype=jnp.float64),
            jnp.asarray(fx["sig"], dtype=jnp.float64),
            jnp.asarray(fx["eps"], dtype=jnp.float64),
            jnp.asarray(lam_val, dtype=jnp.float64),
        )

        np.testing.assert_allclose(np.asarray(gp), np.asarray(gp_ref[:n_act]), rtol=5e-5, atol=5e-5)
        np.testing.assert_allclose(np.asarray(gq), np.asarray(gq_ref[:n_act]), rtol=5e-5, atol=5e-5)
        np.testing.assert_allclose(np.asarray(gs), np.asarray(gs_ref[:n_act]), rtol=5e-5, atol=5e-5)
        np.testing.assert_allclose(np.asarray(ge), np.asarray(ge_ref[:n_act]), rtol=5e-5, atol=5e-5)
        np.testing.assert_allclose(float(glam), float(glam_ref), rtol=5e-5, atol=5e-5)


# =============================================================================
# 4. Portable StableHLO Export Guardrail (0 custom_calls)
# =============================================================================


def test_geometric_and_ljpme_tile_pairs_lower_portably_in_export_context() -> None:
    fx = _build_morton_fixture()
    n_act = fx["n_real"]
    max_pairs = estimate_max_tile_pairs(1, cutoff=9.5, skin=2.0)
    tp = pallas_build_morton_tile_pairs_gpu(
        jnp.asarray(fx["pos"], dtype=jnp.float32),
        jnp.asarray(fx["q"], dtype=jnp.float32),
        jnp.asarray(fx["sig"], dtype=jnp.float32),
        jnp.asarray(fx["eps"], dtype=jnp.float32),
        jnp.asarray(fx["mask"]),
        jnp.asarray(fx["box"], dtype=jnp.float32),
        fx["dedup_excl"],
        n_act=n_act,
        max_pairs=max_pairs,
        cutoff=9.5,
        skin=2.0,
        host_min_box=26.0,
        interpret=True,
    )
    pos_pad = jnp.pad(jnp.asarray(fx["pos"][:n_act], dtype=jnp.float32), ((0, 32 - n_act), (0, 0)))
    pos_m = pos_pad[tp.perm[0]]
    box32 = jnp.asarray(fx["box"], dtype=jnp.float32)

    def _fn(pm: jax.Array, qm: jax.Array, sm: jax.Array, em: jax.Array):
        return jax.value_and_grad(
            lambda p_i, q_i, s_i, e_i: pallas_bonded_and_tile_pairs_fused(
                p_i,
                q_i,
                s_i,
                e_i,
                tp.mask_m[0],
                tp.inv_perm[0],
                tp.pair_ti[0],
                tp.pair_tj[0],
                tp.excl_full_bits[0],
                tp.excl_14_bits[0],
                tp.n_valid_pairs,
                box32,
                n_act=n_act,
                pme_alpha=0.34,
                cutoff=9.5,
                include_pme_excl_erf=True,
                lj_combining_rule="geometric",
                lj_pme_alpha=0.32,
                include_ljpme_excl=True,
                interpret=False,  # Even when interpret=False, export_context forces portable XLA!
            ),
            argnums=(0, 1, 2, 3),
        )(pm, qm, sm, em)

    with export_context():
        jitted = jax.jit(_fn)
        mlir_text = jitted.lower(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0]).as_text()
        exported = jax.export.export(jitted)(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0])

    non_loc_mlir = "\n".join(
        line for line in exported.mlir_module().splitlines() if not line.lstrip().startswith("#loc")
    ).lower()
    for tok in FORBIDDEN_STABLEHLO_SUBSTRINGS:
        assert tok not in mlir_text.lower(), f"Forbidden marker {tok!r} in lowered HLO"
        assert tok not in non_loc_mlir, f"Forbidden marker {tok!r} in exported StableHLO"
    targets = set(_CUSTOM_CALL_RE.findall(mlir_text)) | set(_TARGET_NAME_RE.findall(mlir_text))
    assert not targets

    val_e, grads = exported.call(pos_m, tp.q_m[0], tp.sig_m[0], tp.eps_m[0])
    assert bool(jnp.isfinite(val_e))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


# =============================================================================
# 5. Order-5 Dispersion Gather, OpenMM Reference LJPME Parity, & EnsemblePlan Rollout
# =============================================================================


def _build_openmm_ljpme_or_geometric_system(
    *,
    use_ljpme: bool = True,
    with_charges: bool = True,
) -> tuple[openmm.System, np.ndarray]:
    sys = openmm.System()
    sys.setDefaultPeriodicBoxVectors([2.6, 0.0, 0.0], [0.0, 2.8, 0.0], [0.0, 0.0, 3.0])
    masses = [15.999, 1.008, 1.008, 12.011, 14.007, 15.999]
    for m in masses:
        sys.addParticle(m)

    harm_b = openmm.HarmonicBondForce()
    harm_b.addBond(3, 4, 0.145, 250000.0)
    harm_b.addBond(4, 5, 0.142, 240000.0)
    sys.addForce(harm_b)

    # Rigid water on (0, 1, 2)
    sys.addConstraint(0, 1, 0.09572)
    sys.addConstraint(0, 2, 0.09572)
    sys.addConstraint(1, 2, 0.15139)

    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME if use_ljpme else openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.95 * unit.nanometer)
    nb.setEwaldErrorTolerance(5e-4)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.75 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    nb.setPMEParameters(3.2, 48, 48, 48)
    if use_ljpme:
        nb.setLJPMEParameters(3.0, 16, 18, 20)

    params = [
        (-0.834, 0.31507, 0.63597),
        (0.417, 0.04000, 0.0),
        (0.417, 0.04000, 0.0),
        (-0.200, 0.33997, 0.45773),
        (0.350, 0.32500, 0.71128),
        (-0.150, 0.29500, 0.88000),
    ]
    for q, sig, eps in params:
        nb.addParticle(q if with_charges else 0.0, sig, eps)

    # Water exclusions + 1-2 bonds (3,4), (4,5) + 1-3 scaled exception on (3,5)
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    nb.addException(0, 2, 0.0, 1.0, 0.0)
    nb.addException(1, 2, 0.0, 1.0, 0.0)
    nb.addException(3, 4, 0.0, 1.0, 0.0)
    nb.addException(4, 5, 0.0, 1.0, 0.0)
    nb.addException(3, 5, 0.025 if with_charges else 0.0, 0.315, 0.32)
    sys.addForce(nb)

    h_y = float(np.sqrt(0.09572**2 - (0.5 * 0.15139) ** 2))
    pos_nm = np.array(
        [
            [0.50, 0.60, 0.70],
            [0.50 + 0.5 * 0.15139, 0.60 + h_y, 0.70],
            [0.50 - 0.5 * 0.15139, 0.60 + h_y, 0.70],
            [1.35, 1.20, 1.10],
            [1.495, 1.20, 1.10],
            [1.58, 1.31, 1.12],
        ],
        dtype=np.float64,
    )
    return sys, pos_nm


def test_order5_ljpme_spread_gather_matches_lj_pme_autodiff() -> None:
    """Verify _spread_and_gather_ljpme_order5_soa matches lj_pme_reciprocal_energy and -jax.grad."""
    fx = _build_morton_fixture()
    n_act = fx["n_real"]
    lj_dims = (16, 18, 20)
    alpha_lj = 0.30
    pos_pad = jnp.pad(jnp.asarray(fx["pos"][:n_act], dtype=jnp.float32), ((0, 32 - n_act), (0, 0)))
    sig_pad = jnp.pad(jnp.asarray(fx["sig"][:n_act], dtype=jnp.float32), (0, 32 - n_act), constant_values=1.0)
    eps_pad = jnp.pad(jnp.asarray(fx["eps"][:n_act], dtype=jnp.float32), (0, 32 - n_act))
    msk_pad = jnp.arange(32, dtype=jnp.int32) < n_act
    box_b = jnp.asarray(fx["box"], dtype=jnp.float32)[None, :]
    b_disp = dispersion_charges(sig_pad, eps_pad, msk_pad)[None, :]
    g6_b = lj_pme_influence_function(lj_dims, box_b[0], alpha_lj, order=5).astype(jnp.float32)[None, ...]

    e_soa, fx_soa, fy_soa, fz_soa = _spread_and_gather_ljpme_order5_soa(
        1,
        32,
        pos_pad[:, 0],
        pos_pad[:, 1],
        pos_pad[:, 2],
        b_disp.reshape(-1),
        msk_pad.astype(jnp.int32),
        box_b,
        g6_b,
        lj_dims,
    )
    f_soa = jnp.stack([fx_soa, fy_soa, fz_soa], axis=-1)[:n_act]

    e_ref, g_ref = jax.value_and_grad(
        lambda p: lj_pme_reciprocal_energy(
            p,
            b_disp[0, :n_act],
            jnp.ones((n_act,), dtype=bool),
            box_b[0],
            lj_dims,
            alpha_lj=alpha_lj,
            order=5,
        )
    )(pos_pad[:n_act])

    np.testing.assert_allclose(float(e_soa[0]), float(e_ref), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(f_soa), np.asarray(-g_ref), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("with_charges", [False, True])
def test_ensemble_plan_morton_ljpme_matches_openmm_reference_and_rolls_out_nvt_npt(
    with_charges: bool,
) -> None:
    """Verify EnsemblePlan Morton SoA runner matches OpenMM Reference LJPME energy/forces and runs NVT/NPT."""
    sys_omm, pos_nm = _build_openmm_ljpme_or_geometric_system(use_ljpme=True, with_charges=with_charges)
    ref_ctx = openmm.Context(
        sys_omm, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
    )
    ref_ctx.setPositions(pos_nm * unit.nanometer)
    ref_st = ref_ctx.getState(getEnergy=True, getForces=True)
    e_ref_kcal = ref_st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    f_ref_akma = (
        ref_st.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer) / 41.84
    )

    plan = EnsemblePlan.from_openmm_systems(
        [sys_omm, sys_omm],
        positions_list=[pos_nm * unit.nanometer, pos_nm * unit.nanometer],
        nonbonded_engine="morton_tile_pairs",
        precision="mixed",
    )
    assert plan._can_use_morton_soa_runner(plan.bundles, use_neighbor_list=True, use_flash_forces=False)
    entry = plan._build_morton_soa_runner(
        plan.bundles,
        integration_prefix=None,
        gamma=1.0,
        nl_update_every=10,
        nb_cfg=None,
    )
    assert entry["is_lj_pme"] is True
    assert entry["lj_grid_dims"] == (16, 18, 20)
    np.testing.assert_allclose(entry["lj_pme_alpha"], 0.30, rtol=1e-12)

    tp0 = entry["tp0"]
    n_act = entry["n_act"]
    n_act_pad = entry["n_act_pad"]
    pos_act = jnp.pad(
        jnp.stack([b.positions[:n_act].astype(jnp.float32) for b in plan.bundles], axis=0),
        ((0, 0), (0, n_act_pad - n_act), (0, 0)),
    )
    pos_m = jnp.take_along_axis(pos_act, tp0.perm[:, :, None], axis=1)
    e_morton, fx_m, fy_m, fz_m = entry["eval_morton_forces"](
        pos_m[:, :, 0].reshape(-1),
        pos_m[:, :, 1].reshape(-1),
        pos_m[:, :, 2].reshape(-1),
        tp0,
        entry["box_init_b"],
        entry["G_init"],
        entry["G6_init"],
        compute_energy=True,
    )
    f_m_b = jnp.stack(
        [fx_m.reshape(2, n_act_pad), fy_m.reshape(2, n_act_pad), fz_m.reshape(2, n_act_pad)],
        axis=-1,
    )
    f_unperm = jnp.take_along_axis(f_m_b, tp0.inv_perm[:, :n_act, None], axis=1)

    rtol = 1e-4 if not with_charges else 5e-4
    atol = 1e-3 if not with_charges else 1e-2
    np.testing.assert_allclose(float(e_morton[0]), e_ref_kcal, rtol=rtol, atol=atol)
    np.testing.assert_allclose(float(e_morton[1]), e_ref_kcal, rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(f_unperm[0]), f_ref_akma, rtol=rtol * 5.0, atol=atol * 5.0)

    # Multi-step NVT & NPT EnsemblePlan.run(..., run_mode="inference") rollout parity + SETTLE geometry
    trajs_nvt = cast(
        "list[Any]",
        plan.run(n_steps=6, dt=1.0, kT=0.596, seed=42, run_mode="inference"),
    )
    assert isinstance(trajs_nvt, list)
    assert len(trajs_nvt) == 2
    for tr in trajs_nvt:
        p_fin = np.asarray(tr.positions[0])
        assert p_fin.shape == (6, 3)
        assert np.all(np.isfinite(p_fin))
        # Check rigid water O-H bond preserved by SETTLE
        d_oh1 = p_fin[1] - p_fin[0]
        d_oh1 -= np.array([26.0, 28.0, 30.0]) * np.round(d_oh1 / np.array([26.0, 28.0, 30.0]))
        np.testing.assert_allclose(float(np.linalg.norm(d_oh1)), 0.9572, atol=1e-5)

    # Also test KnobSurface(use_lj_pme=True, lj_combining_rule="geometric") + NPT barostat on PME system
    sys_pme, _ = _build_openmm_ljpme_or_geometric_system(use_ljpme=False, with_charges=True)
    ks_geom_ljpme = KnobSurface(
        nonbonded=NonbondedControls(
            nonbonded_method="pme",
            cutoff_angstrom=9.5,
            pme_alpha=0.32,
            lj_combining_rule="geometric",
            use_lj_pme=True,
        ),
        pallas=PallasControls(nonbonded_engine="morton_tile_pairs", precision="mixed"),
    )
    plan_geom = EnsemblePlan.from_openmm_systems(
        [sys_pme],
        positions_list=[pos_nm * unit.nanometer],
        pressure_atm=1.0,
        barostat_interval=2,
        knob_surface=ks_geom_ljpme,
    )
    assert plan_geom._can_use_morton_soa_runner(
        plan_geom.bundles, use_neighbor_list=True, use_flash_forces=False
    )
    traj_npt = plan_geom.run(n_steps=4, dt=1.0, kT=0.596, seed=7, run_mode="inference")
    traj0 = cast("Any", traj_npt[0] if isinstance(traj_npt, list) else traj_npt)
    assert np.all(np.isfinite(np.asarray(traj0.positions)))
    assert "box" in traj0.observable_values
    assert np.all(np.asarray(traj0.observable_values["box"]) > 20.0)
