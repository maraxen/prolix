"""Tests for clean-room C^2 Prolate polynomial Ewald splittings (P=1 and P=6, v3.9)."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from prolix import export
from prolix.physics.lj_pme import lj_pme_total_dispersion_energy
from prolix.physics.pbc import triclinic_box_from_lengths_angles
from prolix.physics.pme import (
    COULOMB_CONSTANT,
    spme_background_energy,
    spme_energy_with_forces,
)
from prolix.physics.prolate_splitting import (
    PROLATE_C1,
    PROLATE_C6,
    PROLATE_P1_BETAS,
    PROLATE_P1_COEFFS,
    PROLATE_P6_BETAS,
    PROLATE_P6_COEFFS,
    derive_prolate_polynomial_betas,
    prolate_dispersion_pme,
    prolate_p1_dc_coeff,
    prolate_p1_fourier_kernel_and_d_dk2,
    prolate_p1_long_range_value_and_d_dr2,
    prolate_p1_self_energy_coeff,
    prolate_p1_short_range_value_and_d_dr2,
    prolate_p6_fourier_kernel_and_d_dk2,
    prolate_p6_long_range_value_and_d_dr2,
    prolate_p6_self_energy_coeff,
    prolate_p6_short_range_value_and_d_dr2,
    prolate_spme_electrostatics,
)

if TYPE_CHECKING:
    from collections.abc import Generator

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


@pytest.fixture(autouse=True)
def _enable_x64() -> Generator[None]:
    prev = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _assert_portable_stablehlo(mlir_text: str, label: str) -> None:
    non_loc = "\n".join(
        line for line in mlir_text.splitlines() if not line.lstrip().startswith("#loc")
    ).lower()
    leaked = [s for s in FORBIDDEN_STABLEHLO_SUBSTRINGS if s in non_loc]
    assert not leaked, f"{label}: forbidden StableHLO substrings: {leaked}"
    targets = set(_CUSTOM_CALL_RE.findall(mlir_text)) | set(
        _TARGET_NAME_RE.findall(mlir_text)
    )
    assert not targets, f"{label}: unexpected custom_call targets: {sorted(targets)}"


# =============================================================================
# 1. First-Principles Spherical-Bessel / Parseval Gram Optimality & C^2 Vanishing
# =============================================================================


def test_first_principles_gram_derivation_matches_constants() -> None:
    """Verify spherical-Bessel Gram matrix solver reproduces hardcoded P=1 and P=6 polynomials."""
    b1, c1 = derive_prolate_polynomial_betas(1, PROLATE_C1, degree=6)
    b6, c6 = derive_prolate_polynomial_betas(6, PROLATE_C6, degree=7)

    np.testing.assert_allclose(b1, np.asarray(PROLATE_P1_BETAS), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(c1, np.asarray(PROLATE_P1_COEFFS), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(b6, np.asarray(PROLATE_P6_BETAS), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(c6, np.asarray(PROLATE_P6_COEFFS), rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("p", [1, 6])
def test_c2_cutoff_vanishing_and_exact_partition_identity(p: int) -> None:
    """Verify S_p(r_c^2) = S_p'(r_c^2) = S_p''(r_c^2) = 0 and S_p + L_p == r^{-p}."""
    r_c = 8.5
    rc2 = r_c * r_c
    short_fn = (
        prolate_p1_short_range_value_and_d_dr2
        if p == 1
        else prolate_p6_short_range_value_and_d_dr2
    )
    long_fn = (
        prolate_p1_long_range_value_and_d_dr2
        if p == 1
        else prolate_p6_long_range_value_and_d_dr2
    )

    # Exact values at r^2 = r_c^2 and outside cutoff
    r2_outside = jnp.array([rc2, rc2 * 1.0001, rc2 * 2.5], dtype=jnp.float64)
    s_out, ds_out = short_fn(r2_outside, r_c)
    np.testing.assert_allclose(np.asarray(s_out), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(ds_out), 0.0, atol=1e-15)

    # Approach r^2 -> (r_c^2)^- from inside: check S_p, dS_p/d(r^2), and d^2S_p/d(r^2)^2 -> 0
    eps_r2 = 1e-6
    r2_in = jnp.asarray(rc2 - eps_r2, dtype=jnp.float64)
    s_in, ds_in = short_fn(r2_in, r_c)
    d2s_fn = jax.grad(lambda x: short_fn(x, r_c)[1])
    d2s_in = d2s_fn(r2_in)
    # Since S_p(r_c^2 - eps) = O(eps^3), dS_p = O(eps^2), d^2S_p = O(eps)
    assert abs(float(s_in)) < 1e-15
    assert abs(float(ds_in)) < 1e-10
    assert abs(float(d2s_in)) < 1e-5

    # Exact partition S_p(r^2) + L_p(r^2) = r^{-p} across (0.4, 2.0 * r_c)
    r_vals = jnp.linspace(0.45, 16.0, 64, dtype=jnp.float64)
    r2_vals = r_vals * r_vals
    s_v, ds_v = short_fn(r2_vals, r_c)
    l_v, dl_v = long_fn(r2_vals, r_c)
    exact_val = r_vals ** (-p)
    exact_der = (-0.5 * float(p)) * (r_vals ** (-(p + 2)))
    np.testing.assert_allclose(np.asarray(s_v + l_v), np.asarray(exact_val), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.asarray(ds_v + dl_v), np.asarray(exact_der), rtol=1e-13, atol=1e-14)

    # Finite smooth limit at r^2 = 0 for L_p
    l0, dl0 = long_fn(jnp.asarray(0.0, dtype=jnp.float64), r_c)
    assert np.isfinite(float(l0))
    assert np.isfinite(float(dl0))


# =============================================================================
# 2. Central Finite-Difference Checks for d/d(r^2), d/d(k^2), and d/dr_c
# =============================================================================


@pytest.mark.parametrize("p", [1, 6])
def test_real_and_fourier_kernel_derivatives_match_finite_differences(p: int) -> None:
    """Verify analytical d/d(r^2), d/d(k^2), and custom_vjp d/dr_c against float64 central FD."""
    r_c = 9.0
    short_fn = (
        prolate_p1_short_range_value_and_d_dr2
        if p == 1
        else prolate_p6_short_range_value_and_d_dr2
    )
    long_fn = (
        prolate_p1_long_range_value_and_d_dr2
        if p == 1
        else prolate_p6_long_range_value_and_d_dr2
    )
    fourier_fn = (
        prolate_p1_fourier_kernel_and_d_dk2
        if p == 1
        else prolate_p6_fourier_kernel_and_d_dk2
    )

    # Real-space r^2 FD
    r2_pts = jnp.array([0.8, 3.5, 18.0, 62.0, 95.0], dtype=jnp.float64)
    h_r2 = 1e-6
    for fn in (short_fn, long_fn):
        _, d_ana = fn(r2_pts, r_c)
        d_ad = jax.grad(lambda x, _f=fn: jnp.sum(_f(x, r_c)[0]))(r2_pts)
        vp, _ = fn(r2_pts + h_r2, r_c)
        vm, _ = fn(r2_pts - h_r2, r_c)
        d_fd = (vp - vm) / (2.0 * h_r2)
        np.testing.assert_allclose(np.asarray(d_ana), np.asarray(d_fd), rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(np.asarray(d_ad), np.asarray(d_fd), rtol=1e-7, atol=1e-10)

    # d/dr_c VJP check on short-range kernel
    rc_j = jnp.asarray(r_c, dtype=jnp.float64)
    g_rc = float(jax.grad(lambda c: jnp.sum(short_fn(r2_pts, c)[0]))(rc_j))
    h_rc = 1e-6
    fd_rc = float(
        (jnp.sum(short_fn(r2_pts, r_c + h_rc)[0]) - jnp.sum(short_fn(r2_pts, r_c - h_rc)[0]))
        / (2.0 * h_rc)
    )
    assert g_rc == pytest.approx(fd_rc, rel=1e-7, abs=1e-10)

    # Fourier k^2 FD across all regimes: small, mid, and high kappa
    k_vals = np.array([0.08, 0.25, 0.65, 1.35, 2.80], dtype=np.float64)
    k2_pts = jnp.asarray(k_vals * k_vals, dtype=jnp.float64)
    _, dk2_ana = fourier_fn(k2_pts, r_c)
    dk2_ad = jax.grad(lambda x: jnp.sum(fourier_fn(x, r_c)[0]))(k2_pts)
    h_k2 = 1e-6
    kp, _ = fourier_fn(k2_pts + h_k2, r_c)
    km, _ = fourier_fn(k2_pts - h_k2, r_c)
    dk2_fd = (kp - km) / (2.0 * h_k2)
    np.testing.assert_allclose(np.asarray(dk2_ana), np.asarray(dk2_fd), rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(dk2_ad), np.asarray(dk2_fd), rtol=2e-6, atol=1e-9)

    # DC k^2 == 0 behavior
    v0, d0 = fourier_fn(jnp.asarray(0.0, dtype=jnp.float64), r_c)
    g0 = float(jax.grad(lambda x: fourier_fn(x, r_c)[0])(jnp.asarray(0.0, dtype=jnp.float64)))
    if p == 1:
        assert float(v0) == 0.0
    else:
        assert float(v0) > 0.0
    assert float(d0) == 0.0
    assert g0 == 0.0


# =============================================================================
# 3. Direct Numerical 3D Radial Fourier Transform Identity
# =============================================================================


def test_3d_fourier_transform_identity_p1_and_p6() -> None:
    """Verify hat{L}_p(k^2; r_c) equals 4*pi * int_0^inf L_p(r^2; r_c) sin(k r)/(k r) r^2 dr."""
    r_c = 8.0

    # P=1: hat{L}_1(k^2) = 4*pi/k^2 - 4*pi * int_0^{r_c} S_1(r^2; r_c) sin(k r)/(k r) r^2 dr
    for k in (0.15, 0.55, 1.35, 2.60):
        k2 = jnp.asarray(k * k, dtype=jnp.float64)
        got_p1 = float(prolate_p1_fourier_kernel_and_d_dk2(k2, r_c)[0])

        def _s1_integrand(r: float, _k: float = k) -> float:
            s1, _ = prolate_p1_short_range_value_and_d_dr2(jnp.asarray(r * r), r_c)
            return 4.0 * math.pi * float(s1) * (math.sin(_k * r) / (_k * r)) * (r * r)

        s1_hat, _ = integrate.quad(_s1_integrand, 0.0, r_c, epsabs=1e-13, epsrel=1e-13, limit=200)
        ref_p1 = 4.0 * math.pi / (k * k) - s1_hat
        assert got_p1 == pytest.approx(ref_p1, rel=1e-9, abs=1e-11)

    # P=6: direct integral 4*pi * int_0^inf L_6(r^2; r_c) j_0(k r) r^2 dr including k = 0
    for k in (0.0, 0.2, 0.8, 1.5955):
        k2 = jnp.asarray(k * k, dtype=jnp.float64)
        got_p6 = float(prolate_p6_fourier_kernel_and_d_dk2(k2, r_c)[0])

        def _l6_integrand(r: float, _k: float = k) -> float:
            l6, _ = prolate_p6_long_range_value_and_d_dr2(jnp.asarray(r * r), r_c)
            j0 = 1.0 if _k == 0.0 else math.sin(_k * r) / (_k * r)
            return 4.0 * math.pi * float(l6) * j0 * (r * r)

        i_in, _ = integrate.quad(_l6_integrand, 0.0, r_c, epsabs=1e-13, epsrel=1e-13, limit=200)
        i_out, _ = integrate.quad(_l6_integrand, r_c, np.inf, epsabs=1e-12, epsrel=1e-12, limit=400)
        assert got_p6 == pytest.approx(i_in + i_out, rel=1e-8, abs=1e-10)


# =============================================================================
# 4. Converged Ewald Parity (Ortho + Triclinic + Non-Neutral + Exclusions + B==3)
# =============================================================================


def _ewald_reference_electrostatics(
    pos: jnp.ndarray,
    charges: jnp.ndarray,
    box: jnp.ndarray,
    *,
    r_c: float,
    alpha: float = 0.45,
    grid_dims: tuple[int, int, int] = (48, 48, 48),
    excl_indices: jnp.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    n = int(pos.shape[0])
    mask = jnp.ones(n, dtype=bool)
    excl_set: set[tuple[int, int]] = set()
    if excl_indices is not None:
        for row in np.asarray(excl_indices):
            i, j = int(row[0]), int(row[1])
            excl_set.add((min(i, j), max(i, j)))

    def _tot_e(r_in: jnp.ndarray) -> jnp.ndarray:
        e_rec_self = spme_energy_with_forces(
            r_in, charges, mask, box, grid_dims, alpha, 4, False
        )
        e_bg = spme_background_energy(charges, mask, alpha, box)
        # Direct + exclusions
        dr = r_in[:, None, :] - r_in[None, :, :]
        if box.ndim == 1:
            dr = dr - box * jnp.floor(dr / box + 0.5)
        else:
            dr = dr - jnp.floor(dr[..., 2:3] / box[2, 2] + 0.5) * box[2]
            dr = dr - jnp.floor(dr[..., 1:2] / box[1, 1] + 0.5) * box[1]
            dr = dr - jnp.floor(dr[..., 0:1] / box[0, 0] + 0.5) * box[0]
        r2 = jnp.sum(dr * dr, axis=-1)
        idx = jnp.arange(n)
        upper = idx[:, None] < idx[None, :]
        ex_mat = np.zeros((n, n), dtype=bool)
        for i, j in excl_set:
            ex_mat[i, j] = True
            ex_mat[j, i] = True
        ex_j = jnp.asarray(ex_mat)
        valid_dir = upper & (~ex_j) & (r2 < r_c * r_c)
        r_safe = jnp.sqrt(jnp.where(upper, r2, 1.0))
        u_dir = COULOMB_CONSTANT * (charges[:, None] * charges[None, :]) * jax.scipy.special.erfc(alpha * r_safe) / r_safe
        e_dir = jnp.sum(jnp.where(valid_dir, u_dir, 0.0))

        e_ex = jnp.asarray(0.0, dtype=r_in.dtype)
        if excl_indices is not None and excl_indices.shape[0] > 0:
            p1, p2 = excl_indices[:, 0], excl_indices[:, 1]
            dr_e = r_in[p1] - r_in[p2]
            re = jnp.sqrt(jnp.sum(dr_e * dr_e, axis=-1))
            e_ex = -jnp.sum(COULOMB_CONSTANT * charges[p1] * charges[p2] * jax.scipy.special.erf(alpha * re) / re)
        return e_rec_self + e_bg + e_dir + e_ex

    val, grad = jax.value_and_grad(_tot_e)(pos)
    return float(val), np.asarray(-grad, dtype=np.float64)


@pytest.mark.parametrize("shape_kind", ["orthorhombic", "rhombic_dodecahedron"])
def test_prolate_spme_electrostatics_parity_and_gradients(shape_kind: str) -> None:
    """Verify P=1 Prolate SPME matches converged Ewald (< 1e-4 energy, < 1e-3 force) & FD grads."""
    r_c = 8.5
    if shape_kind == "orthorhombic":
        box = jnp.array([22.0, 22.0, 22.0], dtype=jnp.float64)
        is_tric = False
    else:
        box = triclinic_box_from_lengths_angles(
            26.0, 26.0, 26.0, 60.0, 60.0, 60.0, reduce=True, dtype=jnp.float64
        )
        is_tric = True

    # Place ionic cluster at condensed-phase separations (2.8 - 5.2 Å) inside central cell
    box_mat = np.diag(np.asarray(box)) if box.ndim == 1 else np.asarray(box)
    center = 0.5 * np.sum(box_mat, axis=0)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.45, 0.15, 0.0],   # bonded excluded pair (0, 1)
            [3.10, 0.25, -0.15],
            [0.20, 3.05, 0.30],
            [3.15, 3.10, 0.25],
            [1.60, 1.55, 2.95],
        ],
        dtype=np.float64,
    )
    pos = jnp.asarray(center[None, :] + offsets, dtype=jnp.float64)
    # Non-neutral net charge (+0.4) + bonded exclusion (0, 1) to test self + DC + exclusions
    charges = jnp.array([0.85, -0.45, -0.75, -0.65, 0.80, 0.60], dtype=jnp.float64)
    excl_idx = jnp.array([[0, 1]], dtype=jnp.int32)

    e_ref, f_ref = _ewald_reference_electrostatics(
        pos, charges, box, r_c=r_c, alpha=0.52, grid_dims=(64, 64, 64), excl_indices=excl_idx
    )

    val_and_grads = jax.value_and_grad(
        lambda r, q, b: prolate_spme_electrostatics(
            r,
            q,
            b,
            r_c=r_c,
            grid_dims=(64, 64, 64),
            excl_indices=excl_idx,
            is_triclinic=is_tric,
            deconv_mode="euler",
        ),
        argnums=(0, 1, 2),
    )
    e_got_j, (g_pos, g_q, g_box) = val_and_grads(pos, charges, box)
    e_got = float(e_got_j)
    f_got = np.asarray(-g_pos, dtype=np.float64)

    rel_e = abs(e_got - e_ref) / abs(e_ref)
    rel_f = float(np.max(np.linalg.norm(f_got - f_ref, axis=-1)) / np.max(np.linalg.norm(f_ref, axis=-1)))
    assert rel_e < 1e-4
    assert rel_f < 1e-3

    # Verify custom_vjp gradients wrt charges and box against central FD
    hq = 1e-5
    q0_p = charges.at[0].add(hq)
    q0_m = charges.at[0].add(-hq)
    fd_q0 = float(
        (
            prolate_spme_electrostatics(pos, q0_p, box, r_c=r_c, grid_dims=(32, 32, 32), excl_indices=excl_idx, is_triclinic=is_tric)
            - prolate_spme_electrostatics(pos, q0_m, box, r_c=r_c, grid_dims=(32, 32, 32), excl_indices=excl_idx, is_triclinic=is_tric)
        )
        / (2.0 * hq)
    )
    _, g_q_32 = jax.value_and_grad(
        lambda q: prolate_spme_electrostatics(pos, q, box, r_c=r_c, grid_dims=(32, 32, 32), excl_indices=excl_idx, is_triclinic=is_tric)
    )(charges)
    assert float(g_q_32[0]) == pytest.approx(fd_q0, rel=1e-6, abs=1e-7)
    assert np.all(np.isfinite(np.asarray(g_box)))

    # Verify B == 3 ortho batch disambiguation (box shape (3, 3) with is_triclinic=False)
    if not is_tric:
        pos_b3 = jnp.stack([pos, pos, pos], axis=0)
        box_b3 = jnp.stack([box, box, box], axis=0)
        e_b3 = prolate_spme_electrostatics(
            pos_b3, charges, box_b3, r_c=r_c, grid_dims=(32, 32, 32), excl_indices=excl_idx, is_triclinic=False
        )
        assert e_b3.shape == (3,)
        np.testing.assert_allclose(np.asarray(e_b3), float(e_b3[0]), rtol=1e-12)


@pytest.mark.parametrize("rule", ["geometric", "lorentz_berthelot"])
@pytest.mark.parametrize("triclinic", [False, True])
def test_prolate_dispersion_pme_parity_and_gradients(rule: str, triclinic: bool) -> None:
    """Verify P=6 Prolate dispersion PME matches Ewald LJ-PME (< 1e-4 energy, < 1e-3 force) & FD grads."""
    r_c = 9.0
    if not triclinic:
        box = jnp.array([24.0, 24.0, 24.0], dtype=jnp.float64)
    else:
        oct_deg = math.degrees(math.acos(-1.0 / 3.0))
        box = triclinic_box_from_lengths_angles(
            28.0, 28.0, 28.0, oct_deg, oct_deg, oct_deg, reduce=True, dtype=jnp.float64
        )

    # Place cluster at condensed-phase LJ separations (3.6 - 5.2 Å) inside central cell
    center = 0.5 * np.sum(
        np.diag(np.asarray(box)) if box.ndim == 1 else np.asarray(box), axis=0
    )
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.45, 0.15, 0.0],  # bonded excluded pair (0, 1)
            [3.65, 0.30, -0.20],
            [0.25, 3.75, 0.40],
            [3.55, 3.60, 0.35],
            [1.80, 1.85, 3.45],
        ],
        dtype=np.float64,
    )
    pos = jnp.asarray(center[None, :] + offsets, dtype=jnp.float64)
    sigmas = jnp.array([3.35, 3.40, 3.25, 3.30, 3.45, 3.20], dtype=jnp.float64)
    epsilons = jnp.array([0.22, 0.25, 0.20, 0.24, 0.26, 0.21], dtype=jnp.float64)
    excl_idx = jnp.array([[0, 1]], dtype=jnp.int32)

    n = int(pos.shape[0])
    all_idx = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32)[None, :], (n, n))
    self_col = jnp.arange(n, dtype=jnp.int32)[:, None]
    nb_idx = jnp.where((all_idx == self_col) | ((self_col == 0) & (all_idx == 1)) | ((self_col == 1) & (all_idx == 0)), -1, all_idx)

    e_ref, g_pos_ref = jax.value_and_grad(
        lambda r: lj_pme_total_dispersion_energy(
            r,
            sigmas,
            epsilons,
            box,
            nb_idx,
            cutoff=r_c,
            alpha_lj=0.55,
            grid_dims=(64, 64, 64),
            order=5,
            excl_indices=excl_idx,
            lj_combining_rule=rule,
            shift_at_cutoff=False,
        )
    )(pos)

    (e_got, (g_pos, g_sig, g_eps, g_box)) = jax.value_and_grad(
        lambda r, s, ep, b: prolate_dispersion_pme(
            r,
            s,
            ep,
            b,
            r_c=r_c,
            grid_dims=(64, 64, 64),
            order=5,
            neighbor_idx=nb_idx,
            excl_indices=excl_idx,
            lj_combining_rule=rule,
            c2_repulsion_switch=True,
            is_triclinic=triclinic,
        ),
        argnums=(0, 1, 2, 3),
    )(pos, sigmas, epsilons, box)

    rel_e = abs(float(e_got) - float(e_ref)) / abs(float(e_ref))
    f_ref_np = np.asarray(-g_pos_ref)
    f_got_np = np.asarray(-g_pos)
    rel_f = float(np.max(np.linalg.norm(f_got_np - f_ref_np, axis=-1)) / np.max(np.linalg.norm(f_ref_np, axis=-1)))
    assert rel_e < 1e-4
    assert rel_f < 1e-3

    # Verify custom_vjp gradients wrt sigma, epsilon, and box against central FD
    hs = 1e-5
    fd_sig0 = float(
        (
            prolate_dispersion_pme(pos, sigmas.at[0].add(hs), epsilons, box, r_c=r_c, grid_dims=(24, 24, 24), neighbor_idx=nb_idx, excl_indices=excl_idx, lj_combining_rule=rule, is_triclinic=triclinic)
            - prolate_dispersion_pme(pos, sigmas.at[0].add(-hs), epsilons, box, r_c=r_c, grid_dims=(24, 24, 24), neighbor_idx=nb_idx, excl_indices=excl_idx, lj_combining_rule=rule, is_triclinic=triclinic)
        )
        / (2.0 * hs)
    )
    _, (gs_24, ge_24) = jax.value_and_grad(
        lambda s, ep: prolate_dispersion_pme(pos, s, ep, box, r_c=r_c, grid_dims=(24, 24, 24), neighbor_idx=nb_idx, excl_indices=excl_idx, lj_combining_rule=rule, is_triclinic=triclinic),
        argnums=(0, 1),
    )(sigmas, epsilons)
    assert float(gs_24[0]) == pytest.approx(fd_sig0, rel=1e-6, abs=1e-7)
    assert np.all(np.isfinite(np.asarray(ge_24)))
    assert np.all(np.isfinite(np.asarray(g_box)))

    # Verify self-energy & DC coefficients have expected signs
    assert float(prolate_p1_self_energy_coeff(r_c)) < 0.0
    assert float(prolate_p6_self_energy_coeff(r_c)) > 0.0
    assert float(prolate_p1_dc_coeff(r_c)) < 0.0


# =============================================================================
# 5. NVE Total-Energy Conservation Drift vs Unshifted Cutoff
# =============================================================================


def test_nve_energy_conservation_beats_unshifted_cutoff() -> None:
    """C^2 Prolate short-range splitting eliminates cutoff-crossing energy jumps in NVE."""
    # Two opposite charges oscillating across r = r_c = 5.0 Å in a 16 Å periodic box:
    # an unshifted Ewald erfc(alpha * r)/r cutoff with alpha = 0.28 Å^-1 has a step
    # discontinuity at r = r_c, whereas C^2 Prolate has S_1(r_c^2) = S_1' = S_1'' = 0.
    r_c = 5.0
    alpha_coarse = 0.28
    mass = 12.0
    dt = 0.02  # AKMA time units (~0.98 fs)
    n_steps = 160

    def _u_prolate(x_sep: jnp.ndarray) -> jnp.ndarray:
        s1, _ = prolate_p1_short_range_value_and_d_dr2(x_sep * x_sep, r_c)
        return -COULOMB_CONSTANT * 0.6 * s1 + 0.5 * 4.0 * ((x_sep - r_c) ** 2)

    def _u_unshifted(x_sep: jnp.ndarray) -> jnp.ndarray:
        inside = x_sep < r_c
        r_s = jnp.where(inside, x_sep, r_c)
        e_dir = jnp.where(
            inside,
            -COULOMB_CONSTANT * 0.6 * jax.scipy.special.erfc(alpha_coarse * r_s) / r_s,
            0.0,
        )
        return e_dir + 0.5 * 4.0 * ((x_sep - r_c) ** 2)

    def _run_verlet_1d(u_fn) -> float:
        f_fn = jax.jit(lambda x: -jax.grad(u_fn)(x))
        e_fn = jax.jit(u_fn)
        x = jnp.asarray(4.55, dtype=jnp.float64)
        v = jnp.asarray(0.85, dtype=jnp.float64)
        e0 = float(e_fn(x) + 0.5 * mass * v * v)
        max_dev = 0.0
        f = f_fn(x)
        for _ in range(n_steps):
            v_half = v + 0.5 * dt * (f / mass)
            x = x + dt * v_half
            f = f_fn(x)
            v = v_half + 0.5 * dt * (f / mass)
            et = float(e_fn(x) + 0.5 * mass * v * v)
            max_dev = max(max_dev, abs(et - e0))
        return max_dev

    dev_prolate = _run_verlet_1d(_u_prolate)
    dev_unshifted = _run_verlet_1d(_u_unshifted)
    assert dev_prolate < 1e-3
    assert dev_unshifted > 50.0 * dev_prolate


# =============================================================================
# 6. Portable StableHLO Export Guardrail
# =============================================================================


def test_prolate_spme_and_dispersion_lower_to_portable_stablehlo() -> None:
    """Both Prolate SPME solvers and their gradients export to portable StableHLO with zero custom calls."""
    n = 8
    rng = np.random.default_rng(99)
    pos = jnp.asarray(rng.uniform(1.5, 14.5, (n, 3)).astype(np.float32))
    charges = jnp.asarray(np.where(np.arange(n) % 2 == 0, 0.4, -0.4).astype(np.float32))
    sigmas = jnp.full((n,), 3.1, dtype=jnp.float32)
    epsilons = jnp.full((n,), 0.14, dtype=jnp.float32)
    box = jnp.array([16.0, 16.0, 16.0], dtype=jnp.float32)

    def _p1_e_and_g(r, q, b):
        return jax.value_and_grad(
            lambda x: prolate_spme_electrostatics(
                x, q, b, r_c=6.5, grid_dims=(16, 16, 16), use_pallas_pme=True
            )
        )(r)

    def _p6_e_and_g(r, s, ep, b):
        return jax.value_and_grad(
            lambda x: prolate_dispersion_pme(
                x, s, ep, b, r_c=6.5, grid_dims=(16, 16, 16)
            )
        )(r)

    with export.export_context():
        exp_p1 = jax.export.export(jax.jit(_p1_e_and_g))(pos, charges, box)
        exp_p6 = jax.export.export(jax.jit(_p6_e_and_g))(pos, sigmas, epsilons, box)

    _assert_portable_stablehlo(exp_p1.mlir_module(), "prolate_spme_electrostatics")
    _assert_portable_stablehlo(exp_p6.mlir_module(), "prolate_dispersion_pme")
