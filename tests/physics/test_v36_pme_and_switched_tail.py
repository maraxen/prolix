"""v3.6 Track 1 validation: exact SPME isotropic/orthorhombic influence scaling,
16-offset RMS discrete deconvolution, and switched/geometric LJ dispersion tails.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest

from prolix import export
from prolix.physics.explicit_corrections import (
    lj_dispersion_tail_energy,
    lj_dispersion_tail_pressure,
    lj_switched_tail_integrals,
)
from prolix.physics.pallas_ragged_queue import (
    plan_pme_tiers,
    spme_energy_ragged_tiered,
    spme_forces_ragged_tiered,
)
from prolix.physics.pme import (
    COULOMB_CONSTANT,
    DeconvMode,
    influence_function,
    make_spme_energy_fn,
    scale_influence_function_isotropic,
    spme_energy_with_forces,
    spme_reciprocal_energy,
)

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


def _assert_portable_stablehlo(fn, *args, label: str) -> jax.export.Exported:
    with export.export_context():
        jitted = jax.jit(fn)
        mlir_text = jitted.lower(*args).as_text()
        exported = jax.export.export(jitted)(*args)
    low = mlir_text.lower()
    leaked = [s for s in FORBIDDEN_STABLEHLO_SUBSTRINGS if s in low]
    assert not leaked, f"{label}: forbidden StableHLO substrings: {leaked}"
    targets = set(_CUSTOM_CALL_RE.findall(mlir_text)) | set(_TARGET_NAME_RE.findall(mlir_text))
    assert not targets, f"{label}: unexpected StableHLO custom_call targets: {sorted(targets)}"
    return exported


def _gauss_legendre_switched_tail_integrals(r_switch: float, r_cutoff: float) -> tuple[float, float]:
    """64-point Gauss-Legendre reference (split at geometric midpoint) for switched LJ integrals."""
    i6_ext = 1.0 / (3.0 * (r_cutoff**3))
    i12_ext = 1.0 / (9.0 * (r_cutoff**9))
    if r_switch == r_cutoff:
        return i6_ext, i12_ext
    nodes, weights = np.polynomial.legendre.leggauss(64)
    r_geom = float(np.sqrt(r_switch * r_cutoff))
    i6_sw = 0.0
    i12_sw = 0.0
    for a, b in ((r_switch, r_geom), (r_geom, r_cutoff)):
        half = 0.5 * (b - a)
        mid = 0.5 * (b + a)
        r = mid + half * nodes
        x = (r - r_switch) / (r_cutoff - r_switch)
        s_quintic = 1.0 - 10.0 * (x**3) + 15.0 * (x**4) - 6.0 * (x**5)
        omitted = 1.0 - s_quintic
        i6_sw += float(np.sum(weights * half * omitted * (r**-4)))
        i12_sw += float(np.sum(weights * half * omitted * (r**-10)))
    return i6_sw + i6_ext, i12_sw + i12_ext


def _exact_ewald_reciprocal_forces_f64(
    positions: np.ndarray,
    charges: np.ndarray,
    box: np.ndarray,
    alpha: float,
    n_max: int = 10,
) -> np.ndarray:
    """Direct Fourier lattice sum reference for Ewald reciprocal forces (no B-spline mesh)."""
    vol = float(np.prod(box))
    grid = np.arange(-n_max, n_max + 1, dtype=np.float64)
    nx, ny, nz = np.meshgrid(grid, grid, grid, indexing="ij")
    nx = nx.reshape(-1)
    ny = ny.reshape(-1)
    nz = nz.reshape(-1)
    keep = (nx != 0) | (ny != 0) | (nz != 0)
    m_vecs = np.stack(
        [nx[keep] / box[0], ny[keep] / box[1], nz[keep] / box[2]], axis=-1
    )  # (M, 3)
    m_sq = np.sum(m_vecs**2, axis=-1)  # (M,)
    coeff = np.exp(-(np.pi**2) * m_sq / (alpha**2)) / (np.pi * vol * m_sq)  # (M,)
    # E_recip = (COULOMB_CONSTANT / 2) * sum_{m != 0} coeff(m) * |sum_j q_j exp(2*pi*i m.r_j)|^2
    phases = 2.0 * np.pi * (positions @ m_vecs.T)  # (N, M)
    cos_p = np.cos(phases)
    sin_p = np.sin(phases)
    re_s = np.sum(charges[:, None] * cos_p, axis=0)  # (M,)
    im_s = np.sum(charges[:, None] * sin_p, axis=0)  # (M,)
    # F_i = -dE/dr_i = 2*pi * COULOMB_CONSTANT * q_i * sum_{m != 0} coeff(m) * m * (sin(2*pi m.r_i)*Re - cos(2*pi m.r_i)*Im)
    weight = coeff[None, :] * (sin_p * re_s[None, :] - cos_p * im_s[None, :])  # (N, M)
    return 2.0 * np.pi * COULOMB_CONSTANT * charges[:, None] * (weight @ m_vecs)


# =============================================================================
# 1. Exact Isotropic & Orthorhombic SPME Influence Scaling (P0-A)
# =============================================================================


@pytest.mark.parametrize("deconv_mode", ["euler", "ik_rms16"])
@pytest.mark.parametrize("scale_factor", [0.85, 0.94, 1.0, 1.07, 1.15])
def test_scale_influence_function_isotropic_matches_influence_function(
    deconv_mode: DeconvMode,
    scale_factor: float,
) -> None:
    with jax.enable_x64(True):
        grid_dims = (24, 30, 36)
        alpha = 0.37
        box_ref = jnp.array([25.0, 31.0, 37.0], dtype=jnp.float64)
        box_cur = box_ref * scale_factor

        g_ref = influence_function(
            grid_dims, box_ref, alpha, order=4, deconv_mode=deconv_mode
        )
        g_exact = influence_function(
            grid_dims, box_cur, alpha, order=4, deconv_mode=deconv_mode
        )
        g_scaled = scale_influence_function_isotropic(
            g_ref, box_ref, box_cur, alpha=alpha, grid_dims=grid_dims, order=4
        )
        assert float(g_scaled[0, 0, 0]) == 0.0
        rel_err = float(jnp.max(jnp.abs(g_scaled - g_exact)) / jnp.max(jnp.abs(g_exact)))
        assert rel_err < 1e-13

        # Float32 parity (< 1e-6) + legacy 3-arg call when alpha == 0.34
        g_ref_34 = influence_function(grid_dims, box_ref.astype(jnp.float32), 0.34, order=4)
        g_exact_34 = influence_function(grid_dims, box_cur.astype(jnp.float32), 0.34, order=4)
        g_legacy_34 = scale_influence_function_isotropic(
            g_ref_34, box_ref.astype(jnp.float32), box_cur.astype(jnp.float32)
        )
        rel_err_f32 = float(
            jnp.max(jnp.abs(g_legacy_34 - g_exact_34)) / jnp.max(jnp.abs(g_exact_34))
        )
        assert rel_err_f32 < 1e-6


@pytest.mark.parametrize("batch_size", [3, 4])
def test_scale_influence_function_batched_and_underflow_safe(batch_size: int) -> None:
    """Verify batched (B, 3) scaling (including B=3 shape edge case) and float32 Nyquist safety."""
    grid_dims = (24, 24, 24)
    alpha = 0.32
    rng = np.random.default_rng(17)
    box_ref = jnp.asarray(26.0 + rng.uniform(-1.5, 1.5, size=(batch_size, 3)), dtype=jnp.float32)
    lambdas = jnp.asarray(np.linspace(0.88, 1.14, batch_size)[:, None], dtype=jnp.float32)
    box_cur = box_ref * lambdas

    g_ref_b = jax.vmap(lambda b: influence_function(grid_dims, b, alpha, 4))(box_ref)
    g_exact_b = jax.vmap(lambda b: influence_function(grid_dims, b, alpha, 4))(box_cur)
    g_scaled_b = jax.jit(
        lambda br, bc: scale_influence_function_isotropic(
            g_ref_b, br, bc, alpha=alpha, grid_dims=grid_dims
        )
    )(box_ref, box_cur)

    rel_err = float(jnp.max(jnp.abs(g_scaled_b - g_exact_b)) / jnp.max(jnp.abs(g_exact_b)))
    assert rel_err < 1e-6

    # Fine-grid float32 Nyquist corner underflow guard: must never produce 0.0 * inf = NaN
    fine_dims = (64, 64, 64)
    box_small = jnp.array([22.0, 22.0, 22.0], dtype=jnp.float32)
    g_fine = influence_function(fine_dims, box_small, 0.22, 4)
    g_expanded = scale_influence_function_isotropic(
        g_fine, box_small, box_small * 1.25, alpha=0.22, grid_dims=fine_dims
    )
    assert bool(jnp.all(jnp.isfinite(g_expanded)))


def test_npt_molecular_com_reciprocal_pressure_eliminates_353bar_bias() -> None:
    """Regression test for the +352.82 bar MC barostat reciprocal-pressure bias on TIP3P water."""
    with jax.enable_x64(True):
        rng = np.random.default_rng(20261117)
        n_waters = 64
        l0 = 18.6
        box_ref = jnp.array([l0, l0, l0], dtype=jnp.float64)
        grid_dims = (24, 24, 24)
        alpha = 0.34

        # Rigid TIP3P water geometry: O at COM, H1/H2 at rigid internal offsets
        coms = rng.uniform(0.0, l0, size=(n_waters, 3))
        theta_hoh = np.deg2rad(104.52)
        r_oh = 0.9572
        h1_off = np.array([r_oh, 0.0, 0.0])
        h2_off = np.array([r_oh * np.cos(theta_hoh), r_oh * np.sin(theta_hoh), 0.0])

        def _build_positions(scale: float) -> jnp.ndarray:
            scaled_coms = coms * scale
            pos = np.empty((n_waters * 3, 3), dtype=np.float64)
            pos[0::3] = scaled_coms
            pos[1::3] = scaled_coms + h1_off
            pos[2::3] = scaled_coms + h2_off
            return jnp.asarray(pos)

        charges = jnp.asarray(np.tile([-0.834, 0.417, 0.417], n_waters), dtype=jnp.float64)
        mask = jnp.ones((n_waters * 3,), dtype=jnp.float64)

        cached_spme_fn = make_spme_energy_fn(
            box_ref, alpha=alpha, order=4, use_pallas_pme=False, grid_dims=grid_dims
        )

        delta = 1e-4
        v_plus = float((l0 * (1.0 + delta)) ** 3)
        v_minus = float((l0 * (1.0 - delta)) ** 3)
        pos_plus = _build_positions(1.0 + delta)
        pos_minus = _build_positions(1.0 - delta)
        box_plus = box_ref * (1.0 + delta)
        box_minus = box_ref * (1.0 - delta)

        e_cached_plus = float(cached_spme_fn(pos_plus, charges, mask, box=box_plus))
        e_cached_minus = float(cached_spme_fn(pos_minus, charges, mask, box=box_minus))
        e_exact_plus = float(
            spme_energy_with_forces(pos_plus, charges, mask, box_plus, grid_dims, alpha, 4, False)
        )
        e_exact_minus = float(
            spme_energy_with_forces(pos_minus, charges, mask, box_minus, grid_dims, alpha, 4, False)
        )

        akma_to_bar = 69476.95457
        p_cached_bar = -((e_cached_plus - e_cached_minus) / (v_plus - v_minus)) * akma_to_bar
        p_exact_bar = -((e_exact_plus - e_exact_minus) / (v_plus - v_minus)) * akma_to_bar
        assert abs(p_cached_bar - p_exact_bar) < 1e-6


def test_spme_box_vjp_and_ragged_tiered_cached_scaling() -> None:
    """Verify exact dE/dbox VJP and ragged multi-tier cached Green's scaling at box_cur != box_ref."""
    with jax.enable_x64(True):
        rng = np.random.default_rng(42)
        n_atoms = 24
        box_ref = jnp.array([20.0, 22.0, 24.0], dtype=jnp.float64)
        box_cur = jnp.array([21.2, 20.9, 25.4], dtype=jnp.float64)
        pos = jnp.asarray(rng.uniform(1.0, 19.0, size=(n_atoms, 3)), dtype=jnp.float64)
        q_raw = rng.normal(0.0, 0.5, size=(n_atoms,))
        q_raw -= np.mean(q_raw)
        charges = jnp.asarray(q_raw, dtype=jnp.float64)
        mask = jnp.ones((n_atoms,), dtype=jnp.float64)
        grid_dims = (24, 24, 24)
        alpha = 0.35

        modes: tuple[DeconvMode, ...] = ("euler", "ik_rms16")
        for mode in modes:
            cached_fn = make_spme_energy_fn(
                box_ref,
                alpha=alpha,
                order=4,
                use_pallas_pme=False,
                grid_dims=grid_dims,
                deconv_mode=mode,
            )
            grad_box = np.asarray(
                jax.grad(lambda b, fn=cached_fn: fn(pos, charges, mask, box=b))(box_cur)
            )
            eps = 1e-5
            fd_box = np.zeros(3, dtype=np.float64)
            for d in range(3):
                step = jnp.zeros(3, dtype=jnp.float64).at[d].set(eps)
                ep = float(cached_fn(pos, charges, mask, box=box_cur + step))
                em = float(cached_fn(pos, charges, mask, box=box_cur - step))
                fd_box[d] = (ep - em) / (2.0 * eps)
            np.testing.assert_allclose(grad_box, fd_box, rtol=1e-7, atol=1e-9)

    # Ragged multi-tier cached vs uncached parity under box scaling (float32 / interpret=True)
    rng_b = np.random.default_rng(99)
    pos_b = jnp.asarray(rng_b.uniform(1.0, 18.0, size=(2, 32, 3)), dtype=jnp.float32)
    q_b_np = rng_b.normal(0.0, 0.4, size=(2, 32)).astype(np.float32)
    q_b_np[:, 20:] = 0.0
    q_b_np -= np.sum(q_b_np, axis=1, keepdims=True) / np.array([[20.0], [32.0]], dtype=np.float32)
    mask_b = jnp.zeros((2, 32), dtype=jnp.float32).at[0, :20].set(1.0).at[1, :32].set(1.0)
    q_b = jnp.asarray(q_b_np) * mask_b
    box_ref_b = jnp.array([[20.0, 20.0, 20.0], [28.0, 28.0, 28.0]], dtype=jnp.float32)
    box_cur_b = box_ref_b * jnp.array([[0.93], [1.09]], dtype=jnp.float32)

    plan_cached = plan_pme_tiers(
        box_ref_b,
        n_real_per_replica=[20, 32],
        padded_atoms=32,
        pme_tier_mode="grouped",
        alpha=0.34,
        cache_influence=True,
    )
    plan_uncached = plan_pme_tiers(
        box_ref_b,
        n_real_per_replica=[20, 32],
        padded_atoms=32,
        pme_tier_mode="grouped",
        alpha=0.34,
        cache_influence=False,
    )
    e_cached = spme_energy_ragged_tiered(
        pos_b, q_b, mask_b, box_cur_b, tier_plan=plan_cached, use_pallas_pme=False, interpret=True
    )
    e_uncached = spme_energy_ragged_tiered(
        pos_b,
        q_b,
        mask_b,
        box_cur_b,
        tier_plan=plan_uncached,
        use_pallas_pme=False,
        interpret=True,
    )
    f_cached = spme_forces_ragged_tiered(
        pos_b, q_b, mask_b, box_cur_b, tier_plan=plan_cached, use_pallas_pme=False, interpret=True
    )
    f_uncached = spme_forces_ragged_tiered(
        pos_b,
        q_b,
        mask_b,
        box_cur_b,
        tier_plan=plan_uncached,
        use_pallas_pme=False,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(e_cached), np.asarray(e_uncached), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_cached), np.asarray(f_uncached), rtol=1e-6, atol=1e-5)


# =============================================================================
# 2. 16-Offset RMS Discrete Fourier Deconvolution (P1-B)
# =============================================================================


def test_ik_rms16_deconvolution_properties_and_alias_reduction() -> None:
    """Verify Nyquist analytical value 17/315, reduced sub-grid force aliasing, and exact VJP."""
    with jax.enable_x64(True):
        grid_dims = (16, 16, 16)
        box = jnp.array([16.0, 16.0, 16.0], dtype=jnp.float64)
        alpha = 0.32

        g_euler = influence_function(grid_dims, box, alpha, order=4, deconv_mode="euler")
        g_rms16 = influence_function(grid_dims, box, alpha, order=4, deconv_mode="ik_rms16")
        assert float(g_rms16[0, 0, 0]) == 0.0

        # At 1D Nyquist (kx = 8, ky = 0, kz = 0): |b_euler|^2 = 1/9,
        # 16-offset discrete mean |b_rms16|^2 = mean_{s=0..15} ((-1 + 6 u_s^2 - 4 u_s^3)/3)^2
        # (within ~2.06e-6 of continuous integral 17/315 due to l = +-16 alias harmonic).
        u_16 = np.arange(16, dtype=np.float64) / 16.0
        b_nyq_sq_exact_16 = float(np.mean(((6.0 * (u_16**2) - 4.0 * (u_16**3) - 1.0) / 3.0) ** 2))
        ratio_nyq = float(g_euler[8, 0, 0] / g_rms16[8, 0, 0])
        assert abs(ratio_nyq - b_nyq_sq_exact_16 / (1.0 / 9.0)) < 1e-13
        assert abs(ratio_nyq - 17.0 / 35.0) < 5e-6

        # Sub-grid shift ensemble: ik_rms16 reduces reciprocal force RMSE vs exact Ewald lattice sum
        rng = np.random.default_rng(7)
        base_pos = rng.uniform(0.0, 16.0, size=(12, 3))
        q = np.array([0.8, -0.8, 0.5, -0.5, 0.6, -0.6, 0.4, -0.4, 0.7, -0.7, 0.3, -0.3])
        mask = jnp.ones((12,), dtype=jnp.float64)

        f_exact = _exact_ewald_reciprocal_forces_f64(
            base_pos, q, np.array([16.0, 16.0, 16.0]), alpha, n_max=10
        )
        sq_err_euler = []
        sq_err_rms16 = []
        for s in range(16):
            shift = np.array([(s + 0.3) / 16.0, ((3 * s + 1) % 16) / 16.0, ((5 * s + 2) % 16) / 16.0])
            pos_s = jnp.asarray((base_pos + shift) % 16.0, dtype=jnp.float64)
            q_j = jnp.asarray(q, dtype=jnp.float64)
            f_eu = -np.asarray(
                jax.grad(
                    lambda p, qv=q_j: spme_energy_with_forces(
                        p, qv, mask, box, grid_dims, alpha, 4, False, "euler"
                    )
                )(pos_s)
            )
            f_ik = -np.asarray(
                jax.grad(
                    lambda p, qv=q_j: spme_energy_with_forces(
                        p, qv, mask, box, grid_dims, alpha, 4, False, "ik_rms16"
                    )
                )(pos_s)
            )
            sq_err_euler.append(np.mean((f_eu - f_exact) ** 2))
            sq_err_rms16.append(np.mean((f_ik - f_exact) ** 2))

        rmse_euler = float(np.sqrt(np.mean(sq_err_euler)))
        rmse_rms16 = float(np.sqrt(np.mean(sq_err_rms16)))
        assert rmse_rms16 < rmse_euler

        # Verify _spme_bwd matches central FD wrt positions and charges under ik_rms16
        pos_test = jnp.asarray(base_pos, dtype=jnp.float64)
        q_test = jnp.asarray(q, dtype=jnp.float64)
        grad_pos, grad_q = jax.grad(
            lambda p, ch: spme_energy_with_forces(
                p, ch, mask, box, grid_dims, alpha, 4, False, "ik_rms16"
            ),
            argnums=(0, 1),
        )(pos_test, q_test)
        eps = 1e-6
        ep = float(
            spme_reciprocal_energy(
                pos_test.at[0, 1].add(eps), q_test, mask, box, grid_dims, alpha, 4, False, "ik_rms16"
            )
        )
        em = float(
            spme_reciprocal_energy(
                pos_test.at[0, 1].add(-eps), q_test, mask, box, grid_dims, alpha, 4, False, "ik_rms16"
            )
        )
        assert abs(float(grad_pos[0, 1]) - (ep - em) / (2.0 * eps)) < 1e-7
        assert bool(jnp.all(jnp.isfinite(grad_q)))


# =============================================================================
# 3. Switched LJ Dispersion Tail & Geometric Combining Rules (P1-B)
# =============================================================================


@pytest.mark.parametrize(
    ("r_switch", "r_cutoff"),
    [
        (9.0, 9.0),      # Unswitched limit
        (8.995, 9.0),    # Extreme narrow switch (cancellation test: z^2 ~ 7.7e-8)
        (8.0, 9.0),      # Standard 1.0 A switch (series branch: z^2 ~ 3.46e-3)
        (1.15, 9.0),     # Just below z^2 = 0.61 series/log crossover (z^2 ~ 0.598)
        (1.05, 9.0),     # Just above z^2 = 0.61 series/log crossover (z^2 ~ 0.626)
        (0.6, 9.0),      # Wide switch (logarithmic branch: z^2 ~ 0.766)
    ],
)
def test_lj_switched_tail_integrals_match_64pt_gauss_legendre(
    r_switch: float,
    r_cutoff: float,
) -> None:
    with jax.enable_x64(True):
        i6_ref, i12_ref = _gauss_legendre_switched_tail_integrals(r_switch, r_cutoff)
        i6, i12 = lj_switched_tail_integrals(
            jnp.asarray(r_switch, dtype=jnp.float64),
            jnp.asarray(r_cutoff, dtype=jnp.float64),
        )
        np.testing.assert_allclose(float(i6), i6_ref, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(float(i12), i12_ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("combining_rule", ["lorentz_berthelot", "geometric"])
@pytest.mark.parametrize("lj_switch_width", [0.0, 1.0, 8.2])
def test_lj_dispersion_tail_energy_pressure_and_grad(
    combining_rule: str,
    lj_switch_width: float,
) -> None:
    """Verify O(N) LB & geometric moments vs O(N^2) quadrature, P_tail = -dE/dV, and NaN-free AD."""
    with jax.enable_x64(True):
        box = jnp.array([24.0, 26.0, 28.0], dtype=jnp.float64)
        # Include zero-epsilon hydrogens and masked-out padding atoms to test AD NaN-safety!
        sigma = jnp.array([3.1506, 0.4000, 3.4000, 2.9500, 3.8000], dtype=jnp.float64)
        epsilon = jnp.array([0.1521, 0.0, 0.1094, 0.2100, 0.5000], dtype=jnp.float64)
        atom_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
        cutoff = 9.0
        r_switch = cutoff - lj_switch_width if lj_switch_width > 0.0 else cutoff

        # Dense O(N^2) reference over active atoms
        sig_np = np.asarray(sigma[:4])
        eps_np = np.asarray(epsilon[:4])
        eps_ij = np.sqrt(eps_np[:, None] * eps_np[None, :])
        if combining_rule == "lorentz_berthelot":
            sig_ij = 0.5 * (sig_np[:, None] + sig_np[None, :])
        else:
            sig_ij = np.sqrt(sig_np[:, None] * sig_np[None, :])
        c6_dense = float(np.sum(eps_ij * (sig_ij**6)))
        c12_dense = float(np.sum(eps_ij * (sig_ij**12)))
        i6_ref, i12_ref = _gauss_legendre_switched_tail_integrals(r_switch, cutoff)
        vol = float(np.prod(np.asarray(box)))
        e_ref = (8.0 * np.pi / vol) * (c12_dense * i12_ref - c6_dense * i6_ref)
        p_ref = e_ref / vol

        e_val = lj_dispersion_tail_energy(
            box,
            sigma,
            epsilon,
            cutoff,
            atom_mask,
            lj_switch_width=lj_switch_width,
            combining_rule=combining_rule,
        )
        p_val = lj_dispersion_tail_pressure(
            box,
            sigma,
            epsilon,
            cutoff,
            atom_mask,
            lj_switch_width=lj_switch_width,
            combining_rule=combining_rule,
        )
        np.testing.assert_allclose(float(e_val), e_ref, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(float(p_val), p_ref, rtol=1e-12, atol=0.0)

        # Verify P_tail == -dE_tail / dV via isotropic volume FD
        delta = 1e-5
        e_vp = float(
            lj_dispersion_tail_energy(
                box * (1.0 + delta),
                sigma,
                epsilon,
                cutoff,
                atom_mask,
                lj_switch_width=lj_switch_width,
                combining_rule=combining_rule,
            )
        )
        e_vm = float(
            lj_dispersion_tail_energy(
                box * (1.0 - delta),
                sigma,
                epsilon,
                cutoff,
                atom_mask,
                lj_switch_width=lj_switch_width,
                combining_rule=combining_rule,
            )
        )
        p_fd = -(e_vp - e_vm) / (vol * ((1.0 + delta) ** 3 - (1.0 - delta) ** 3))
        np.testing.assert_allclose(float(p_val), p_fd, rtol=1e-8, atol=0.0)

        # Analytical jax.grad wrt (box, sigma, epsilon) — must be finite even at epsilon[1] == 0.0
        g_box, g_sig, g_eps = jax.grad(
            lambda b, s, ep: lj_dispersion_tail_energy(
                b,
                s,
                ep,
                cutoff,
                atom_mask,
                lj_switch_width=lj_switch_width,
                combining_rule=combining_rule,
            ),
            argnums=(0, 1, 2),
        )(box, sigma, epsilon)
        assert bool(jnp.all(jnp.isfinite(g_box)))
        assert bool(jnp.all(jnp.isfinite(g_sig)))
        assert bool(jnp.all(jnp.isfinite(g_eps)))
        assert float(g_sig[4]) == 0.0
        assert float(g_eps[4]) == 0.0
        assert float(g_eps[1]) == 0.0

        # Central FD check on positive-epsilon active atoms & box
        h = 1e-6
        for idx in (0, 2, 3):
            es_p = float(
                lj_dispersion_tail_energy(
                    box,
                    sigma.at[idx].add(h),
                    epsilon,
                    cutoff,
                    atom_mask,
                    lj_switch_width=lj_switch_width,
                    combining_rule=combining_rule,
                )
            )
            es_m = float(
                lj_dispersion_tail_energy(
                    box,
                    sigma.at[idx].add(-h),
                    epsilon,
                    cutoff,
                    atom_mask,
                    lj_switch_width=lj_switch_width,
                    combining_rule=combining_rule,
                )
            )
            np.testing.assert_allclose(float(g_sig[idx]), (es_p - es_m) / (2.0 * h), rtol=1e-7)

            ee_p = float(
                lj_dispersion_tail_energy(
                    box,
                    sigma,
                    epsilon.at[idx].add(h),
                    cutoff,
                    atom_mask,
                    lj_switch_width=lj_switch_width,
                    combining_rule=combining_rule,
                )
            )
            ee_m = float(
                lj_dispersion_tail_energy(
                    box,
                    sigma,
                    epsilon.at[idx].add(-h),
                    cutoff,
                    atom_mask,
                    lj_switch_width=lj_switch_width,
                    combining_rule=combining_rule,
                )
            )
            np.testing.assert_allclose(float(g_eps[idx]), (ee_p - ee_m) / (2.0 * h), rtol=1e-7)


def test_lj_switched_tail_validation_errors() -> None:
    with pytest.raises(ValueError, match="positive finite r_cutoff"):
        lj_switched_tail_integrals(8.0, 0.0)
    with pytest.raises(ValueError, match="0 < r_switch <= r_cutoff"):
        lj_switched_tail_integrals(0.0, 9.0)
    with pytest.raises(ValueError, match="0 < r_switch <= r_cutoff"):
        lj_switched_tail_integrals(9.5, 9.0)
    with pytest.raises(ValueError, match="Unsupported LJ combining_rule"):
        lj_dispersion_tail_energy(
            jnp.array([20.0, 20.0, 20.0]),
            jnp.array([3.0]),
            jnp.array([0.1]),
            9.0,
            jnp.array([1.0]),
            combining_rule="invalid_rule",
        )


# =============================================================================
# 4. Portable StableHLO Export & Root NOTICE Guardrails
# =============================================================================


def test_v36_track1_stablehlo_export_and_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROLIX_USE_PALLAS", "1")
    monkeypatch.setenv("PROLIX_USE_PALLAS_PME", "1")

    box_ref = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    box_cur = jnp.array([20.8, 20.8, 20.8], dtype=jnp.float32)
    pos = jnp.array(
        [[2.0, 3.0, 4.0], [5.5, 6.5, 7.5], [10.0, 11.0, 12.0], [14.0, 15.0, 16.0]],
        dtype=jnp.float32,
    )
    charges = jnp.array([0.5, -0.5, 0.3, -0.3], dtype=jnp.float32)
    sigma = jnp.array([3.1, 2.9, 3.4, 0.4], dtype=jnp.float32)
    epsilon = jnp.array([0.15, 0.12, 0.10, 0.0], dtype=jnp.float32)
    mask = jnp.ones((4,), dtype=jnp.float32)
    grid_dims = (16, 16, 16)

    cached_ik_fn = make_spme_energy_fn(
        box_ref,
        alpha=0.34,
        order=4,
        use_pallas_pme=True,
        grid_dims=grid_dims,
        deconv_mode="ik_rms16",
    )

    def _combined_energy_and_grads(p, q, sig, eps, b_cur):
        def _tot(p_i, q_i, s_i, e_i, b_i):
            e_pme = cached_ik_fn(p_i, q_i, mask, box=b_i)
            e_tail = lj_dispersion_tail_energy(
                b_i,
                s_i,
                e_i,
                9.0,
                mask,
                lj_switch_width=1.0,
                combining_rule="geometric",
            )
            p_tail = lj_dispersion_tail_pressure(
                b_i,
                s_i,
                e_i,
                9.0,
                mask,
                lj_switch_width=1.0,
                combining_rule="lorentz_berthelot",
            )
            return e_pme + e_tail + 0.0 * p_tail

        val, grads = jax.value_and_grad(_tot, argnums=(0, 1, 2, 3, 4))(p, q, sig, eps, b_cur)
        return val, grads

    exported = _assert_portable_stablehlo(
        _combined_energy_and_grads,
        pos,
        charges,
        sigma,
        epsilon,
        box_cur,
        label="v36_track1_pme_and_switched_tail",
    )
    with export.export_context():
        ref_val, ref_grads = _combined_energy_and_grads(pos, charges, sigma, epsilon, box_cur)
    rt_val, rt_grads = exported.call(pos, charges, sigma, epsilon, box_cur)
    np.testing.assert_allclose(np.asarray(rt_val), np.asarray(ref_val), rtol=1e-5, atol=1e-5)
    for g_rt, g_ref in zip(rt_grads, ref_grads, strict=True):
        np.testing.assert_allclose(np.asarray(g_rt), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

    # Verify root NOTICE carries Apache-2.0, MIT, and BSD-3-Clause attributions
    repo_root = Path(__file__).resolve().parents[2]
    notice_path = repo_root / "NOTICE"
    assert notice_path.is_file(), f"Missing root NOTICE at {notice_path}"
    notice_text = notice_path.read_text(encoding="utf-8")
    assert "Apache-2.0" in notice_text
    assert "lj_tail.hpp" in notice_text
    assert "NonbondedForceImpl.cpp" in notice_text
    assert "MIT" in notice_text
    assert "BSD-3-Clause" in notice_text
    assert "fltflt.hpp" in notice_text
