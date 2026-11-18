"""Clean-room C^2-continuous Prolate polynomial Ewald splittings for P=1 and P=6.

Derives from first principles (Liang et al., ``arXiv:2505.09727``; Slepian-Pollak-Landau
radial prolate spheroidal wave operator) compact-support real-space / band-limited
reciprocal-space splittings of ``r^{-p}`` for electrostatics (``p = 1``,
``c_1 = 10.75``) and dispersion (``p = 6``, ``c_6 = 12.764031957789484``):

  - ``r^{-p} = S_p(r^2; r_c) + L_p(r^2; r_c)`` with strictly compact short-range
    ``S_p(r^2; r_c) = 0`` for ``r >= r_c`` and smooth polynomial interior
    ``L_p(r^2; r_c) = r_c^{-p} P_p(r^2 / r_c^2)`` for ``r < r_c``.
  - Exact ``C^2`` vanishing at the cutoff sphere ``r = r_c``:
    ``S_p(r_c^2) = dS_p/d(r^2)|_{r_c^2} = d^2S_p/d(r^2)^2|_{r_c^2} = 0``,
    parameterized in the shifted ``y = 1 - r^2 / r_c^2`` basis so boundary
    continuity holds without floating-point cancellation in ``float32`` or ``float64``.
  - Optimal free polynomial coefficients ``beta_3 .. beta_M`` minimizing the
    out-of-band 3D Fourier tail energy ``int_{c_p}^inf |hat{L}_p(kappa)|^2 kappa^2 dkappa``
    via Parseval's theorem and spherical-Bessel Sonine Gram matrices on ``[0, c_p]``.
  - Cancellation-free analytical Fourier kernels ``hat{L}_1(k^2; r_c)`` and
    ``hat{L}_6(k^2; r_c)`` (retaining finite ``k^2 = 0`` dispersion DC value and
    exact zero ``d/d(k^2)`` at ``k^2 = 0``) plus ``@jax.custom_vjp`` SPME solvers
    supporting orthorhombic and reduced-form lower-triangular triclinic boxes and
    portable StableHLO lowering under ``prolix.export.export_context()``.
"""

from __future__ import annotations

import decimal
import functools
import math
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from xtrax.intent import Context

from prolix.export import in_export_context
from prolix.physics.lj_pme import (
    _bspline5,
    _bspline_moduli_sq_1d,
    _combine_lj_pair,
    dispersion_charges,
    resolve_ljpme_parameters,
    spread_dispersion_charges,
)
from prolix.physics.pbc import box_volume, invert_box_lower_triangular
from prolix.physics.pme import (
    COULOMB_CONSTANT,
    DeconvMode,
    _bspline4,
    _bspline4_deriv,
    _bspline_modulation,
    _gather_reference,
    _is_ortho_box,
    _is_triclinic_box_array,
    compute_openmm_pme_params,
    resolve_use_pallas_pme,
    spread_charges,
)
from prolix.physics.pme_pallas import make_gather_op

__all__ = [
    "PROLATE_C1",
    "PROLATE_C6",
    "PROLATE_P1_BETAS",
    "PROLATE_P1_COEFFS",
    "PROLATE_P6_BETAS",
    "PROLATE_P6_COEFFS",
    "derive_prolate_polynomial_betas",
    "prolate_p1_short_range_value_and_d_dr2",
    "prolate_p1_long_range_value_and_d_dr2",
    "prolate_p6_short_range_value_and_d_dr2",
    "prolate_p6_long_range_value_and_d_dr2",
    "prolate_p1_fourier_kernel_and_d_dk2",
    "prolate_p6_fourier_kernel_and_d_dk2",
    "prolate_p1_self_energy_coeff",
    "prolate_p6_self_energy_coeff",
    "prolate_p1_dc_coeff",
    "prolate_spme_electrostatics",
    "prolate_dispersion_pme",
]

# Dimensionless wavenumber bandwidths c_p = k_max * r_c (arXiv:2505.09727)
PROLATE_C1: float = 10.75
PROLATE_C6: float = 12.764031957789484

# 90-digit arbitrary-precision optimal C^2 polynomial coefficients in y = 1 - r^2 / r_c^2:
# P_p(x) = sum_{j=0}^M beta_j * (1 - x)^j = sum_{m=0}^M a_m * x^m
PROLATE_P1_BETAS: tuple[float, ...] = (
    1.0,
    0.5,
    0.375,
    0.24172490174584002,
    0.37912310262347643,
    -0.14000915083912296,
    0.24981000628276703,
)

PROLATE_P1_COEFFS: tuple[float, ...] = (
    2.6056488598129603,
    -4.2904813992324131,
    5.7219719068286548,
    -5.3543259295038563,
    3.4262274426693669,
    -1.3588508868574791,
    0.24981000628276703,
)

PROLATE_P6_BETAS: tuple[float, ...] = (
    1.0,
    3.0,
    6.0,
    9.2993668215376033,
    5.5431359606492538,
    28.727514457279913,
    -20.03937491286133,
    21.611122007684113,
)

PROLATE_P6_COEFFS: tuple[float, ...] = (
    55.141764334289554,
    -239.74982117023018,
    507.67499926975387,
    -674.34882724865111,
    604.97935482307275,
    -362.32482714147829,
    131.23847914092747,
    -21.611122007684113,
)


# =============================================================================
# 1. First-Principles Spherical-Bessel / Parseval Gram Solver
# =============================================================================


def _sonine_phi_hat_np(j: int, kappa: np.ndarray) -> np.ndarray:
    """Exact 3D Fourier transform of ``(1 - u^2)^j * 1_{u < 1}``.

    By Sonine's spherical-Bessel integral,
      ``hat{phi}_j(kappa) = 4*pi * 2^j * j! * j_{j+1}(kappa) / kappa^{j+1}``
      ``= 4*pi * sum_{s>=0} (-kappa^2)^s / (2s + 1)! * int_0^1 (1 - u^2)^j u^{2s+2} du``.
    """
    k2 = np.asarray(kappa, dtype=np.float64) ** 2
    s_tot = np.zeros_like(k2)
    fact = 1.0
    neg_k2_pow = np.ones_like(k2)
    for s in range(55):
        if s > 0:
            fact *= float(2 * s) * float(2 * s + 1)
            neg_k2_pow = neg_k2_pow * (-k2)
        iu = 1.0 / float(2 * s + 3)
        for m in range(1, j + 1):
            iu *= float(2 * m) / float(2 * s + 2 * m + 3)
        term = neg_k2_pow * (iu / fact)
        s_tot = s_tot + term
        if float(np.max(np.abs(term))) < 1e-18:
            break
    return 4.0 * math.pi * s_tot


def _half_beta_rational(n: int) -> float:
    """Exact ``int_0^1 (1 - u^2)^n u^2 du = (1/2) B(3/2, n + 1)``."""
    val = 1.0 / 3.0
    for k in range(1, n + 1):
        val *= float(2 * k) / float(2 * k + 3)
    return val


def derive_prolate_polynomial_betas(
    p: int,
    c: float,
    degree: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive optimal ``C^2`` Prolate polynomial coefficients from first principles.

    Minimizes ``J_p = int_c^inf |hat{L}_p(kappa)|^2 kappa^2 dkappa`` over polynomials
    ``P_p(x) = 1 + (p/2)(1-x) + (p(p+2)/8)(1-x)^2 + sum_{j=3}^M beta_j (1-x)^j``
    using Parseval's identity on ``R^3`` and 70-digit ``Decimal`` arithmetic so the
    spherical-Bessel Gram system is solved without floating-point conditioning loss.

    Args:
        p: Inverse power exponent (``1`` for Coulomb, ``6`` for dispersion).
        c: Dimensionless wavenumber cutoff ``c_p = k_max * r_c``.
        degree: Polynomial degree ``M >= 3`` in ``x = r^2 / r_c^2`` (defaults to ``6``
            for ``p = 1`` and ``7`` for ``p = 6``).

    Returns:
        ``(betas, coeffs)`` where ``betas[j]`` multiplies ``(1 - x)^j`` and ``coeffs[m]``
        multiplies ``x^m`` for ``0 <= j, m <= M``.
    """
    if p not in (1, 6):
        raise ValueError(f"derive_prolate_polynomial_betas supports p in {{1, 6}}, got {p}")
    m_deg = (6 if p == 1 else 7) if degree is None else int(degree)
    if m_deg < 3:
        raise ValueError(f"C^2 Prolate polynomial requires degree >= 3, got {m_deg}")

    dec = decimal.Decimal
    with decimal.localcontext() as dctx:
        dctx.prec = 75
        pi_d = dec(
            "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
        )
        c_d = dec(repr(float(c)))
        b_fixed = [dec(1), dec(p) / dec(2), dec(p * (p + 2)) / dec(8)]

        def _d_cos(x: decimal.Decimal) -> decimal.Decimal:
            mx2 = -x * x
            term = dec(1)
            s_val = dec(1)
            for k_i in range(1, 55):
                term = term * mx2 / dec((2 * k_i - 1) * (2 * k_i))
                s_val += term
                if abs(term) < dec("1e-70"):
                    break
            return s_val

        def _d_half_beta(n_b: int) -> decimal.Decimal:
            v_b = dec(1) / dec(3)
            for k_i in range(1, n_b + 1):
                v_b = v_b * dec(2 * k_i) / dec(2 * k_i + 3)
            return v_b

        def _d_phi_hat(j_idx: int, k_v: decimal.Decimal) -> decimal.Decimal:
            mk2 = -k_v * k_v
            s_tot = dec(0)
            fact = dec(1)
            k2_pow = dec(1)
            for s_i in range(60):
                if s_i > 0:
                    fact *= dec(2 * s_i) * dec(2 * s_i + 1)
                    k2_pow *= mk2
                iu = dec(1) / dec(2 * s_i + 3)
                for m_i in range(1, j_idx + 1):
                    iu = iu * dec(2 * m_i) / dec(2 * s_i + 2 * m_i + 3)
                term = k2_pow * iu / fact
                s_tot += term
                if abs(term) < dec("1e-70"):
                    break
            return dec(4) * pi_d * s_tot

        # Refine 40-point Gauss-Legendre nodes/weights to 75 digits via Newton
        x_np, _ = np.polynomial.legendre.leggauss(40)
        kq: list[decimal.Decimal] = []
        wq: list[decimal.Decimal] = []
        n_gl = 40
        for x0 in x_np:
            xd = dec(str(float(x0)))
            for _ in range(6):
                p0, p1 = dec(1), xd
                for m_i in range(2, n_gl + 1):
                    p0, p1 = (
                        p1,
                        (dec(2 * m_i - 1) * xd * p1 - dec(m_i - 1) * p0) / dec(m_i),
                    )
                dp = dec(n_gl) * (xd * p1 - p0) / (xd * xd - dec(1))
                xd = xd - p1 / dp
            wd = dec(2) / ((dec(1) - xd * xd) * dp * dp)
            kq.append(dec("0.5") * c_d * (xd + dec(1)))
            wq.append(dec("0.5") * c_d * wd)

        l0_k2: list[decimal.Decimal] = []
        for k_v in kq:
            poly0 = sum(b_fixed[j] * _d_phi_hat(j, k_v) for j in range(3)) * k_v * k_v
            if p == 1:
                l0_k2.append(dec(4) * pi_d * _d_cos(k_v) + poly0)
            else:
                ser6 = (
                    dec(1) / dec(3)
                    - (k_v * k_v) / dec(6)
                    + pi_d * (k_v * k_v * k_v) / dec(48)
                )
                fact = dec(6)
                k2_pow = k_v * k_v
                for s_i in range(2, 60):
                    fact *= dec(2 * s_i) * dec(2 * s_i + 1)
                    k2_pow *= k_v * k_v
                    sgn_d = dec(-1) if ((s_i + 1) % 2 == 1) else dec(1)
                    term = sgn_d * k2_pow / (dec(2 * s_i - 3) * fact)
                    ser6 += term
                    if abs(term) < dec("1e-70"):
                        break
                l0_k2.append(dec(4) * pi_d * ser6 * k_v * k_v + poly0)

        n_free = m_deg - 2
        phis = [[_d_phi_hat(a + 3, k_v) for k_v in kq] for a in range(n_free)]
        mat_a = [[dec(0)] * n_free for _ in range(n_free)]
        vec_b = [dec(0)] * n_free
        eight_pi3 = dec(8) * (pi_d * pi_d * pi_d)

        for a in range(n_free):
            ja = a + 3
            full_r = eight_pi3 * sum(
                b_fixed[m_i] * _d_half_beta(m_i + ja) for m_i in range(3)
            )
            low_r = sum(wq[q] * l0_k2[q] * phis[a][q] for q in range(len(kq)))
            vec_b[a] = -(full_r - low_r)
            for b in range(a, n_free):
                jb = b + 3
                full_g = eight_pi3 * _d_half_beta(ja + jb)
                low_g = sum(
                    wq[q] * phis[a][q] * phis[b][q] * kq[q] * kq[q]
                    for q in range(len(kq))
                )
                mat_a[a][b] = mat_a[b][a] = full_g - low_g

        aug = [[*mat_a[r], vec_b[r]] for r in range(n_free)]
        for col in range(n_free):
            piv = max(range(col, n_free), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            for r in range(col + 1, n_free):
                fac = aug[r][col] / aug[col][col]
                for kc in range(col, n_free + 1):
                    aug[r][kc] -= fac * aug[col][kc]
        sol = [dec(0)] * n_free
        for r in range(n_free - 1, -1, -1):
            sol[r] = (
                aug[r][n_free]
                - sum(aug[r][kc] * sol[kc] for kc in range(r + 1, n_free))
            ) / aug[r][r]

        betas_d = b_fixed + sol
        coeffs_d = [dec(0)] * (m_deg + 1)
        for j_i in range(m_deg + 1):
            for m_i in range(j_i + 1):
                sgn_d = dec(-1) if (m_i % 2 == 1) else dec(1)
                coeffs_d[m_i] += betas_d[j_i] * sgn_d * dec(math.comb(j_i, m_i))

        betas = np.array([float(x) for x in betas_d], dtype=np.float64)
        coeffs = np.array([float(x) for x in coeffs_d], dtype=np.float64)
        return betas, coeffs


# =============================================================================
# 2. Precomputed Quadrature & Power-Series Tables for Fourier Kernels
# =============================================================================


def _precompute_p1_fourier_tables() -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Precompute exact series, quadrature, and finite-IBP coefficients for P=1."""
    # 1. Small-kappa series (kappa < 3.5):
    # hat{l}_1(kappa) = 4*pi/kappa^2 - 4*pi * sum_{s>=0} d_s * (kappa^2)^s
    # where d_s = (-1)^s / (2s+1)! * (1/(2s+2) - sum_m a_m / (2m + 2s + 3))
    n_ser = 22
    d_val: list[float] = []
    for s in range(n_ser):
        rem_int = 1.0 / float(2 * s + 2) - sum(
            PROLATE_P1_COEFFS[m] / float(2 * m + 2 * s + 3)
            for m in range(len(PROLATE_P1_COEFFS))
        )
        coeff = -4.0 * math.pi * (((-1.0) ** s) / float(math.factorial(2 * s + 1))) * rem_int
        d_val.append(coeff)
    # Horner descending in z = kappa^2
    ser_val_desc = tuple(d_val[::-1])
    ser_der_desc = tuple(float(s) * d_val[s] for s in range(n_ser - 1, 0, -1))

    # 2. Mid-kappa (3.5 <= kappa <= 18.0): 32-pt Gauss-Legendre on [0, 1] for Q_1(u) = (u P_1(u^2))'''
    q1_asc = [
        float((2 * m + 1) * (2 * m) * (2 * m - 1)) * PROLATE_P1_COEFFS[m]
        for m in range(1, len(PROLATE_P1_COEFFS))
    ]
    u_gl, w_gl = np.polynomial.legendre.leggauss(32)
    u_01 = 0.5 * (u_gl + 1.0)
    q1_vals = sum(q1_asc[r] * (u_01 ** (2 * r)) for r in range(len(q1_asc)))
    w_q1 = 0.5 * w_gl * q1_vals
    w_u_q1 = w_q1 * u_01

    # 3. Large-kappa (kappa > 18.0): Exact finite IBP of int_0^1 Q_1(u) cos(kappa u) du
    q1_poly_u = np.poly1d(
        [
            q1_asc[r // 2] if r % 2 == 0 else 0.0
            for r in range(2 * len(q1_asc) - 2, -1, -1)
        ]
    )
    even_c = [
        ((-1.0) ** r) * float(np.polyder(q1_poly_u, 2 * r)(1.0))
        for r in range(len(q1_asc))
    ]
    odd_c = [
        ((-1.0) ** r) * float(np.polyder(q1_poly_u, 2 * r + 1)(1.0))
        for r in range(len(q1_asc) - 1)
    ]
    # Also exact finite IBP of int_0^1 u * Q_1(u) sin(kappa u) du for derivative!
    # Simpler: differentiate hat{l}_1(kappa) = -4*pi * (sin(k) A(k^-2)/k^5 + cos(k) B(k^-2)/k^6) analytically!
    even_desc = tuple(even_c[::-1])
    odd_desc = tuple(odd_c[::-1])
    even_d_desc = tuple(
        float(r) * even_c[r] for r in range(len(even_c) - 1, 0, -1)
    )
    odd_d_desc = tuple(
        float(r) * odd_c[r] for r in range(len(odd_c) - 1, 0, -1)
    )
    return (
        ser_val_desc,
        ser_der_desc,
        u_01,
        w_q1,
        w_u_q1,
        even_desc,
        odd_desc,
        even_d_desc,
        odd_d_desc,
    )


(
    _P1_SER_VAL_DESC,
    _P1_SER_DER_DESC,
    _P1_GL_U,
    _P1_GL_WQ,
    _P1_GL_WUQ,
    _P1_IBP_EVEN_DESC,
    _P1_IBP_ODD_DESC,
    _P1_IBP_EVEN_D_DESC,
    _P1_IBP_ODD_D_DESC,
) = _precompute_p1_fourier_tables()


def _precompute_p6_fourier_tables() -> tuple[
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Precompute exact series and split-contour quadrature tables for P=6."""
    n_ser = 28
    c_even: list[float] = []
    for s in range(n_ser):
        p_int = sum(
            PROLATE_P6_COEFFS[m] / float(2 * m + 2 * s + 3)
            for m in range(len(PROLATE_P6_COEFFS))
        ) * (((-1.0) ** s) / float(math.factorial(2 * s + 1)))
        if s == 0:
            t_int = 1.0 / 3.0
        elif s == 1:
            t_int = -1.0 / 6.0
        else:
            t_int = ((-1.0) ** (s + 1)) / (
                float(2 * s - 3) * float(math.factorial(2 * s + 1))
            )
        c_even.append(4.0 * math.pi * (p_int + t_int))
    c_k3 = 4.0 * math.pi * (math.pi / 48.0)
    dc_val = c_even[0]
    ser_val_desc = tuple(c_even[::-1])
    ser_der_desc = tuple(float(s) * c_even[s] for s in range(n_ser - 1, 0, -1))

    # For kappa >= 4.0: IBP3 gives hat{l}_6(kappa) = -(4*pi / kappa^4) * (I_poly + I_12 + I_2inf)
    # where I_poly = int_0^1 Q_6(u) cos(k u) du, I_12 = -210 int_1^2 u^-8 cos(k u) du,
    # and I_2inf = -210 int_2^inf u^-8 cos(k u) du (Wick-rotated u = 2 + i s / k).
    q6_asc = [
        float((2 * m + 1) * (2 * m) * (2 * m - 1)) * PROLATE_P6_COEFFS[m]
        for m in range(1, len(PROLATE_P6_COEFFS))
    ]
    u_gl, w_gl = np.polynomial.legendre.leggauss(32)
    u_01 = 0.5 * (u_gl + 1.0)
    q6_vals = sum(q6_asc[r] * (u_01 ** (2 * r)) for r in range(len(q6_asc)))
    w_q6 = 0.5 * w_gl * q6_vals
    w_u_q6 = w_q6 * u_01

    u_12 = 1.5 + 0.5 * u_gl
    w_12 = 0.5 * w_gl * (-210.0 * (u_12**-8))
    w_u_12 = w_12 * u_12

    s_lag, w_lag = np.polynomial.laguerre.laggauss(28)
    return (
        dc_val,
        c_k3,
        ser_val_desc,
        ser_der_desc,
        u_01,
        w_q6,
        w_u_q6,
        u_12,
        w_12,
        w_u_12,
        s_lag,
        w_lag,
    )


(
    _P6_DC_UNIT,
    _P6_C_K3,
    _P6_SER_VAL_DESC,
    _P6_SER_DER_DESC,
    _P6_GL_U01,
    _P6_GL_WQ6,
    _P6_GL_WUQ6,
    _P6_GL_U12,
    _P6_GL_W12,
    _P6_GL_WU12,
    _P6_LAG_S,
    _P6_LAG_W,
) = _precompute_p6_fourier_tables()


# =============================================================================
# 3. Real-Space Short-Range & Long-Range Kernels + d/d(r^2)
# =============================================================================


def _horner_eval(x: jnp.ndarray, coeffs_desc: tuple[float, ...]) -> jnp.ndarray:
    acc = jnp.zeros_like(x)
    for c in coeffs_desc:
        acc = jnp.asarray(c, dtype=x.dtype) + x * acc
    return acc


def _eval_poly_and_dx_in_y(
    x: jnp.ndarray,
    betas_asc: tuple[float, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate ``P(x)`` and ``dP/dx`` via Horner in ``y = 1 - x`` (`dP/dx = -dP/dy`)."""
    y = jnp.asarray(1.0, dtype=x.dtype) - x
    p_val = _horner_eval(y, tuple(betas_asc[::-1]))
    dp_dy_desc = tuple(
        float(j) * betas_asc[j] for j in range(len(betas_asc) - 1, 0, -1)
    )
    dp_dx = -_horner_eval(y, dp_dy_desc)
    return p_val, dp_dx


@jax.custom_vjp
def _p1_short_val(r2: jnp.ndarray, r_c: jnp.ndarray) -> jnp.ndarray:
    v, _ = _p1_short_impl(r2, r_c)
    return v


def _p1_short_impl(
    r2: jnp.ndarray, r_c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = r2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    rc2 = rc * rc
    inside = (r2 > 0.0) & (r2 < rc2)
    r2_safe = jnp.where(inside, r2, rc2)
    inv_r = jax.lax.rsqrt(r2_safe)
    inv_r3 = inv_r * inv_r * inv_r

    x = jnp.where(inside, r2_safe / rc2, jnp.ones_like(r2))
    p_val, dp_dx = _eval_poly_and_dx_in_y(x, PROLATE_P1_BETAS)
    inv_rc = jnp.asarray(1.0, dtype=dtype) / rc
    inv_rc3 = inv_rc * inv_rc * inv_rc

    val = jnp.where(inside, inv_r - inv_rc * p_val, jnp.zeros_like(r2))
    d_dr2 = jnp.where(
        inside,
        jnp.asarray(-0.5, dtype=dtype) * inv_r3 - inv_rc3 * dp_dx,
        jnp.zeros_like(r2),
    )
    return val, d_dr2


def _p1_short_fwd(r2: jnp.ndarray, r_c: jnp.ndarray):
    val, d_dr2 = _p1_short_impl(r2, r_c)
    return val, (r2, r_c, d_dr2)


def _p1_short_bwd(res, g):
    r2, r_c, d_dr2 = res
    dtype = r2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    rc2 = rc * rc
    inside = (r2 > 0.0) & (r2 < rc2)
    x = jnp.where(inside, r2 / rc2, jnp.ones_like(r2))
    p_val, dp_dx = _eval_poly_and_dx_in_y(x, PROLATE_P1_BETAS)
    # d/dr_c [-r_c^-1 P(r^2 / r_c^2)] = r_c^-2 P(x) + 2 r^2 r_c^-4 dP/dx = r_c^-2 (P(x) + 2 x dP/dx)
    d_drc_elem = jnp.where(
        inside, (p_val + 2.0 * x * dp_dx) / rc2, jnp.zeros_like(r2)
    )
    d_rc = jnp.sum(g * d_drc_elem)
    if jnp.ndim(r_c) == 0:
        d_rc = jnp.reshape(d_rc, ()).astype(jnp.asarray(r_c).dtype)
    return (g * d_dr2).astype(dtype), d_rc


_p1_short_val.defvjp(_p1_short_fwd, _p1_short_bwd)


def prolate_p1_short_range_value_and_d_dr2(
    r2: jnp.ndarray,
    r_c: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate compact-support ``P=1`` short-range ``S_1(r^2; r_c)`` and ``dS_1/d(r^2)``."""
    r2_arr = jnp.asarray(r2)
    dtype = (
        r2_arr.dtype
        if jnp.issubdtype(r2_arr.dtype, jnp.floating)
        else jnp.result_type(r2_arr, 1.0)
    )
    r2_f = r2_arr.astype(dtype)
    rc_f = jnp.asarray(r_c, dtype=dtype)
    val = _p1_short_val(r2_f, rc_f)
    _, d_dr2 = _p1_short_impl(r2_f, rc_f)
    return val, d_dr2


def prolate_p1_long_range_value_and_d_dr2(
    r2: jnp.ndarray,
    r_c: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate smooth ``P=1`` long-range ``L_1(r^2; r_c)`` and ``dL_1/d(r^2)`` (finite at ``r^2 = 0``)."""
    r2_arr = jnp.asarray(r2)
    dtype = (
        r2_arr.dtype
        if jnp.issubdtype(r2_arr.dtype, jnp.floating)
        else jnp.result_type(r2_arr, 1.0)
    )
    r2_f = r2_arr.astype(dtype)
    rc = jnp.asarray(r_c, dtype=dtype)
    rc2 = rc * rc
    inside = r2_f < rc2

    x_in = jnp.where(inside, r2_f / rc2, jnp.ones_like(r2_f))
    p_val, dp_dx = _eval_poly_and_dx_in_y(x_in, PROLATE_P1_BETAS)
    inv_rc = jnp.asarray(1.0, dtype=dtype) / rc
    inv_rc3 = inv_rc * inv_rc * inv_rc

    r2_out = jnp.where(inside, rc2, r2_f)
    inv_r = jax.lax.rsqrt(r2_out)
    inv_r3 = inv_r * inv_r * inv_r

    val = jnp.where(inside, inv_rc * p_val, inv_r)
    d_dr2 = jnp.where(inside, inv_rc3 * dp_dx, jnp.asarray(-0.5, dtype=dtype) * inv_r3)
    return val, d_dr2


@jax.custom_vjp
def _p6_short_val(r2: jnp.ndarray, r_c: jnp.ndarray) -> jnp.ndarray:
    v, _ = _p6_short_impl(r2, r_c)
    return v


def _p6_short_impl(
    r2: jnp.ndarray, r_c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = r2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    rc2 = rc * rc
    inside = (r2 > 0.0) & (r2 < rc2)
    r2_safe = jnp.where(inside, r2, rc2)
    inv_r2 = jnp.asarray(1.0, dtype=dtype) / r2_safe
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r8 = inv_r6 * inv_r2

    x = jnp.where(inside, r2_safe / rc2, jnp.ones_like(r2))
    p_val, dp_dx = _eval_poly_and_dx_in_y(x, PROLATE_P6_BETAS)
    inv_rc2 = jnp.asarray(1.0, dtype=dtype) / rc2
    inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
    inv_rc8 = inv_rc6 * inv_rc2

    val = jnp.where(inside, inv_r6 - inv_rc6 * p_val, jnp.zeros_like(r2))
    d_dr2 = jnp.where(
        inside,
        jnp.asarray(-3.0, dtype=dtype) * inv_r8 - inv_rc8 * dp_dx,
        jnp.zeros_like(r2),
    )
    return val, d_dr2


def _p6_short_fwd(r2: jnp.ndarray, r_c: jnp.ndarray):
    val, d_dr2 = _p6_short_impl(r2, r_c)
    return val, (r2, r_c, d_dr2)


def _p6_short_bwd(res, g):
    r2, r_c, d_dr2 = res
    dtype = r2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    rc2 = rc * rc
    inside = (r2 > 0.0) & (r2 < rc2)
    x = jnp.where(inside, r2 / rc2, jnp.ones_like(r2))
    p_val, dp_dx = _eval_poly_and_dx_in_y(x, PROLATE_P6_BETAS)
    inv_rc7 = jnp.asarray(1.0, dtype=dtype) / (rc**7)
    d_drc_elem = jnp.where(
        inside,
        inv_rc7 * (6.0 * p_val + 2.0 * x * dp_dx),
        jnp.zeros_like(r2),
    )
    d_rc = jnp.sum(g * d_drc_elem)
    if jnp.ndim(r_c) == 0:
        d_rc = jnp.reshape(d_rc, ()).astype(jnp.asarray(r_c).dtype)
    return (g * d_dr2).astype(dtype), d_rc


_p6_short_val.defvjp(_p6_short_fwd, _p6_short_bwd)


def prolate_p6_short_range_value_and_d_dr2(
    r2: jnp.ndarray,
    r_c: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate compact-support ``P=6`` short-range ``S_6(r^2; r_c)`` and ``dS_6/d(r^2)``."""
    r2_arr = jnp.asarray(r2)
    dtype = (
        r2_arr.dtype
        if jnp.issubdtype(r2_arr.dtype, jnp.floating)
        else jnp.result_type(r2_arr, 1.0)
    )
    r2_f = r2_arr.astype(dtype)
    rc_f = jnp.asarray(r_c, dtype=dtype)
    val = _p6_short_val(r2_f, rc_f)
    _, d_dr2 = _p6_short_impl(r2_f, rc_f)
    return val, d_dr2


def prolate_p6_long_range_value_and_d_dr2(
    r2: jnp.ndarray,
    r_c: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate smooth ``P=6`` long-range ``L_6(r^2; r_c)`` and ``dL_6/d(r^2)`` (finite at ``r^2 = 0``)."""
    r2_arr = jnp.asarray(r2)
    dtype = (
        r2_arr.dtype
        if jnp.issubdtype(r2_arr.dtype, jnp.floating)
        else jnp.result_type(r2_arr, 1.0)
    )
    r2_f = r2_arr.astype(dtype)
    rc = jnp.asarray(r_c, dtype=dtype)
    rc2 = rc * rc
    inside = r2_f < rc2

    x_in = jnp.where(inside, r2_f / rc2, jnp.ones_like(r2_f))
    p_val, dp_dx = _eval_poly_and_dx_in_y(x_in, PROLATE_P6_BETAS)
    inv_rc2 = jnp.asarray(1.0, dtype=dtype) / rc2
    inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
    inv_rc8 = inv_rc6 * inv_rc2

    r2_out = jnp.where(inside, rc2, r2_f)
    inv_r2 = jnp.asarray(1.0, dtype=dtype) / r2_out
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r8 = inv_r6 * inv_r2

    val = jnp.where(inside, inv_rc6 * p_val, inv_r6)
    d_dr2 = jnp.where(inside, inv_rc8 * dp_dx, jnp.asarray(-3.0, dtype=dtype) * inv_r8)
    return val, d_dr2


# =============================================================================
# 4. Self-Energy & Neutralizing-Plasma DC Coefficients
# =============================================================================


def prolate_p1_self_energy_coeff(r_c: float | jnp.ndarray = 1.0) -> jnp.ndarray:
    """Self-energy prefactor ``-P_1(0) / (2 * r_c)`` such that ``E_self = COULOMB * coeff * sum(q_i^2)``."""
    rc = jnp.asarray(r_c)
    dtype = rc.dtype if jnp.issubdtype(rc.dtype, jnp.floating) else jnp.result_type(rc, 1.0)
    rc_f = rc.astype(dtype)
    return jnp.asarray(-0.5 * PROLATE_P1_COEFFS[0], dtype=dtype) / rc_f


def prolate_p6_self_energy_coeff(r_c: float | jnp.ndarray = 1.0) -> jnp.ndarray:
    """Positive dispersion self-energy prefactor ``+P_6(0) / (2 * r_c^6)`` for ``E_self = coeff * sum(b_i^2)``."""
    rc = jnp.asarray(r_c)
    dtype = rc.dtype if jnp.issubdtype(rc.dtype, jnp.floating) else jnp.result_type(rc, 1.0)
    rc_f = rc.astype(dtype)
    return jnp.asarray(0.5 * PROLATE_P6_COEFFS[0], dtype=dtype) / (rc_f**6)


def prolate_p1_dc_coeff(r_c: float | jnp.ndarray = 1.0) -> jnp.ndarray:
    """Neutralizing-plasma DC prefactor ``-(1/2) int_{R^3} S_1(r; r_c) d^3r`` for ``E_bg = COULOMB * (coeff / V) * Q_tot^2``."""
    rc = jnp.asarray(r_c)
    dtype = rc.dtype if jnp.issubdtype(rc.dtype, jnp.floating) else jnp.result_type(rc, 1.0)
    rc_f = rc.astype(dtype)
    unit_int = 0.5 - sum(
        PROLATE_P1_COEFFS[m] / float(2 * m + 3)
        for m in range(len(PROLATE_P1_COEFFS))
    )
    return jnp.asarray(-2.0 * math.pi * unit_int, dtype=dtype) * (rc_f * rc_f)


# =============================================================================
# 5. Cancellation-Free Analytical 3D Fourier Kernels & d/d(k^2)
# =============================================================================


def _p1_fourier_unit_val_and_d_dkappa2(
    kappa2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dimensionless ``hat{l}_1(kappa)`` and ``d(hat{l}_1)/d(kappa^2)`` for ``kappa^2 >= 0``."""
    dtype = kappa2.dtype
    is_dc = kappa2 <= 0.0
    k2_pos = jnp.where(is_dc, jnp.ones_like(kappa2), kappa2)

    # Regime 1: 0 < kappa < 3.5 (kappa^2 < 12.25) — exact power series in kappa^2
    is_small = (~is_dc) & (k2_pos < 12.25)
    k2_sm = jnp.where(is_small, k2_pos, jnp.ones_like(k2_pos))
    four_pi = jnp.asarray(4.0 * math.pi, dtype=dtype)
    val_sm = four_pi / k2_sm + _horner_eval(k2_sm, _P1_SER_VAL_DESC)
    der_sm = -four_pi / (k2_sm * k2_sm) + _horner_eval(k2_sm, _P1_SER_DER_DESC)

    # Regime 2: 3.5 <= kappa <= 18.0 (12.25 <= kappa^2 <= 324.0) — 32-pt Gauss-Legendre of IBP3
    is_mid = (~is_dc) & (k2_pos >= 12.25) & (k2_pos <= 324.0)
    k2_md = jnp.where(is_mid, k2_pos, jnp.asarray(25.0, dtype=dtype))
    k_md = jnp.sqrt(k2_md)
    u_gl = jnp.asarray(_P1_GL_U, dtype=dtype)
    wq_gl = jnp.asarray(_P1_GL_WQ, dtype=dtype)
    wuq_gl = jnp.asarray(_P1_GL_WUQ, dtype=dtype)
    arg_md = k_md[..., None] * u_gl
    i_cos = jnp.sum(wq_gl * jnp.cos(arg_md), axis=-1)
    i_usin = jnp.sum(wuq_gl * jnp.sin(arg_md), axis=-1)
    k4_md = k2_md * k2_md
    val_md = (-four_pi / k4_md) * i_cos
    # d/d(k^2) [-4*pi * k^-4 * I_cos(k)] = (8*pi / k^6) * I_cos(k) + (2*pi / k^5) * I_usin(k)
    der_md = (2.0 * four_pi / (k4_md * k2_md)) * i_cos + (
        0.5 * four_pi / (k4_md * k_md)
    ) * i_usin

    # Regime 3: kappa > 18.0 (kappa^2 > 324.0) — exact finite IBP Horner in 1 / kappa^2
    k2_lg = jnp.where(k2_pos > 324.0, k2_pos, jnp.asarray(400.0, dtype=dtype))
    k_lg = jnp.sqrt(k2_lg)
    inv_k2 = jnp.asarray(1.0, dtype=dtype) / k2_lg
    inv_k = jnp.asarray(1.0, dtype=dtype) / k_lg
    inv_k4 = inv_k2 * inv_k2

    a_pol = _horner_eval(inv_k2, _P1_IBP_EVEN_DESC)
    b_pol = _horner_eval(inv_k2, _P1_IBP_ODD_DESC)
    da_du = _horner_eval(inv_k2, _P1_IBP_EVEN_D_DESC)
    db_du = _horner_eval(inv_k2, _P1_IBP_ODD_D_DESC)

    s_lg = jnp.sin(k_lg)
    c_lg = jnp.cos(k_lg)
    # hat{l}_1 = -4*pi * (s * inv_k^5 * A(u) + c * inv_k^6 * B(u)), u = inv_k2
    term_s = s_lg * inv_k * inv_k4 * a_pol
    term_c = c_lg * inv_k2 * inv_k4 * b_pol
    val_lg = -four_pi * (term_s + term_c)

    # Differentiate w.r.t. kappa^2: dk/d(k^2) = 1/(2k), du/d(k^2) = -inv_k4
    d_term_s = (
        c_lg * 0.5 * inv_k2 * inv_k4 * a_pol
        - 2.5 * s_lg * inv_k * inv_k2 * inv_k4 * a_pol
        - s_lg * inv_k * inv_k4 * inv_k4 * da_du
    )
    d_term_c = (
        -s_lg * 0.5 * inv_k * inv_k2 * inv_k4 * b_pol
        - 3.0 * c_lg * inv_k4 * inv_k4 * b_pol
        - c_lg * inv_k2 * inv_k4 * inv_k4 * db_du
    )
    der_lg = -four_pi * (d_term_s + d_term_c)

    val_nz = jnp.where(is_small, val_sm, jnp.where(is_mid, val_md, val_lg))
    der_nz = jnp.where(is_small, der_sm, jnp.where(is_mid, der_md, der_lg))
    return (
        jnp.where(is_dc, jnp.zeros_like(kappa2), val_nz),
        jnp.where(is_dc, jnp.zeros_like(kappa2), der_nz),
    )


@jax.custom_vjp
def _p1_fourier_val_vjp(k2: jnp.ndarray, r_c: jnp.ndarray) -> jnp.ndarray:
    rc2 = r_c * r_c
    v_u, _ = _p1_fourier_unit_val_and_d_dkappa2(k2 * rc2)
    return rc2 * v_u


def _p1_fourier_fwd(k2: jnp.ndarray, r_c: jnp.ndarray):
    rc2 = r_c * r_c
    v_u, d_u = _p1_fourier_unit_val_and_d_dkappa2(k2 * rc2)
    val = rc2 * v_u
    d_dk2 = (rc2 * rc2) * d_u
    return val, (k2, r_c, v_u, d_u, d_dk2)


def _p1_fourier_bwd(res, g):
    k2, r_c, v_u, d_u, d_dk2 = res
    dtype = k2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    # d/dr_c [r_c^2 hat{l}_1(k^2 r_c^2)] = 2 r_c v_u + 2 k^2 r_c^3 d_u
    d_drc_elem = jnp.where(
        k2 > 0.0, 2.0 * rc * (v_u + k2 * rc * rc * d_u), jnp.zeros_like(k2)
    )
    d_rc = jnp.sum(g * d_drc_elem)
    if jnp.ndim(r_c) == 0:
        d_rc = jnp.reshape(d_rc, ()).astype(jnp.asarray(r_c).dtype)
    return (g * d_dk2).astype(dtype), d_rc


_p1_fourier_val_vjp.defvjp(_p1_fourier_fwd, _p1_fourier_bwd)


def prolate_p1_fourier_kernel_and_d_dk2(
    k2: jnp.ndarray,
    r_c: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate 3D Fourier kernel ``hat{L}_1(k^2; r_c)`` and ``d(hat{L}_1)/d(k^2)``.

    Here ``k^2 = |k|^2 = (2 * pi * |m|)^2`` is the squared angular wavevector norm.
    Returns ``(0.0, 0.0)`` at the ``k^2 = 0`` DC mode with ``NaN``-free gradients.
    """
    k2_arr = jnp.asarray(k2)
    dtype = (
        k2_arr.dtype
        if jnp.issubdtype(k2_arr.dtype, jnp.floating)
        else jnp.result_type(k2_arr, 1.0)
    )
    k2_f = k2_arr.astype(dtype)
    rc_f = jnp.asarray(r_c, dtype=dtype)
    val = _p1_fourier_val_vjp(k2_f, rc_f)
    _, d_u = _p1_fourier_unit_val_and_d_dkappa2(k2_f * (rc_f * rc_f))
    d_dk2 = (rc_f**4) * d_u
    return val, d_dk2


def _p6_fourier_unit_val_and_d_dkappa2(
    kappa2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dimensionless ``hat{l}_6(kappa)`` and ``d(hat{l}_6)/d(kappa^2)`` (zero grad at ``kappa^2 == 0``)."""
    dtype = kappa2.dtype
    is_dc = kappa2 <= 0.0
    k2_pos = jnp.where(is_dc, jnp.ones_like(kappa2), kappa2)

    # Regime 1: 0 < kappa < 4.5 (kappa^2 < 20.25) — exact power series in kappa^2 + c_k3 * kappa^3
    is_small = (~is_dc) & (k2_pos < 20.25)
    k2_sm = jnp.where(is_small, k2_pos, jnp.ones_like(k2_pos))
    k_sm = jnp.sqrt(k2_sm)
    c_k3 = jnp.asarray(_P6_C_K3, dtype=dtype)
    val_sm = c_k3 * (k2_sm * k_sm) + _horner_eval(k2_sm, _P6_SER_VAL_DESC)
    der_sm = (1.5 * c_k3) * k_sm + _horner_eval(k2_sm, _P6_SER_DER_DESC)

    # Regime 2: kappa >= 4.5 (kappa^2 >= 20.25) — IBP3 on [0,1] + [1,2] + Wick Gauss-Laguerre on [2, inf)
    k2_lg = jnp.where(k2_pos >= 20.25, k2_pos, jnp.asarray(25.0, dtype=dtype))
    k_lg = jnp.sqrt(k2_lg)
    k4_lg = k2_lg * k2_lg
    four_pi = jnp.asarray(4.0 * math.pi, dtype=dtype)

    u01 = jnp.asarray(_P6_GL_U01, dtype=dtype)
    wq6 = jnp.asarray(_P6_GL_WQ6, dtype=dtype)
    wuq6 = jnp.asarray(_P6_GL_WUQ6, dtype=dtype)
    arg01 = k_lg[..., None] * u01
    i01 = jnp.sum(wq6 * jnp.cos(arg01), axis=-1)
    di01_dk = -jnp.sum(wuq6 * jnp.sin(arg01), axis=-1)

    u12 = jnp.asarray(_P6_GL_U12, dtype=dtype)
    w12 = jnp.asarray(_P6_GL_W12, dtype=dtype)
    wu12 = jnp.asarray(_P6_GL_WU12, dtype=dtype)
    arg12 = k_lg[..., None] * u12
    i12 = jnp.sum(w12 * jnp.cos(arg12), axis=-1)
    di12_dk = -jnp.sum(wu12 * jnp.sin(arg12), axis=-1)

    # Wick contour u = 2 + i * s / k on [2, inf):
    # int_2^inf u^-8 cos(k u) du = (1/k) int_0^inf e^-s Re[ i * e^{2 i k} * (2 + i s/k)^-8 ] ds
    # Let t = s / (2 k), so (2 + i s/k)^-8 = 2^-8 * (1 + i t)^-8 = 2^-8 * (1 + t^2)^-4 * e^{-8 i arctan(t)}
    # Re[ i * e^{i (2 k - 8 arctan(t))} ] = -sin(2 k - 8 arctan(t)).
    s_lag = jnp.asarray(_P6_LAG_S, dtype=dtype)
    w_lag = jnp.asarray(_P6_LAG_W, dtype=dtype)
    inv_2k = jnp.asarray(0.5, dtype=dtype) / k_lg
    t_lag = inv_2k[..., None] * s_lag
    one_p_t2 = 1.0 + t_lag * t_lag
    inv_r8 = 1.0 / (one_p_t2**4)
    phase = (2.0 * k_lg)[..., None] - 8.0 * jnp.arctan(t_lag)
    sin_ph = jnp.sin(phase)
    cos_ph = jnp.cos(phase)

    pref_2inf = jnp.asarray(-210.0 / 256.0, dtype=dtype)
    # f(k, s) = -sin(phase) * inv_r8 / k
    # dt/dk = -t / k; d(phase)/dk = 2 + 8 * (t / k) / (1 + t^2)
    # d(inv_r8)/dk = -4 * (1 + t^2)^-5 * 2 * t * (-t / k) = 8 * t^2 / (k * (1 + t^2)) * inv_r8
    inv_k_b = (jnp.asarray(1.0, dtype=dtype) / k_lg)[..., None]
    f_lag = -sin_ph * inv_r8 * inv_k_b
    dphase_dk = 2.0 + (8.0 * t_lag * inv_k_b) / one_p_t2
    df_dk = (
        -cos_ph * dphase_dk * inv_r8 * inv_k_b
        - sin_ph * inv_r8 * (inv_k_b * inv_k_b) * ((8.0 * t_lag * t_lag) / one_p_t2 - 1.0)
    )
    i2inf = pref_2inf * jnp.sum(w_lag * f_lag, axis=-1)
    di2inf_dk = pref_2inf * jnp.sum(w_lag * df_dk, axis=-1)

    i_tot = i01 + i12 + i2inf
    di_dk = di01_dk + di12_dk + di2inf_dk
    val_lg = (-four_pi / k4_lg) * i_tot
    # d/d(k^2) = (1 / (2 k)) * d/dk [-4*pi * k^-4 * i_tot]
    der_lg = (2.0 * four_pi / (k4_lg * k2_lg)) * i_tot - (
        0.5 * four_pi / (k4_lg * k_lg)
    ) * di_dk

    dc_val = jnp.asarray(_P6_DC_UNIT, dtype=dtype)
    val_out = jnp.where(is_dc, dc_val, jnp.where(is_small, val_sm, val_lg))
    der_out = jnp.where(is_dc, jnp.zeros_like(kappa2), jnp.where(is_small, der_sm, der_lg))
    return val_out, der_out


@jax.custom_vjp
def _p6_fourier_val_vjp(k2: jnp.ndarray, r_c: jnp.ndarray) -> jnp.ndarray:
    rc2 = r_c * r_c
    v_u, _ = _p6_fourier_unit_val_and_d_dkappa2(k2 * rc2)
    return v_u / (rc2 * r_c)


def _p6_fourier_fwd(k2: jnp.ndarray, r_c: jnp.ndarray):
    rc2 = r_c * r_c
    v_u, d_u = _p6_fourier_unit_val_and_d_dkappa2(k2 * rc2)
    inv_rc = jnp.asarray(1.0, dtype=k2.dtype) / r_c
    inv_rc3 = inv_rc * inv_rc * inv_rc
    val = inv_rc3 * v_u
    d_dk2 = inv_rc * d_u
    return val, (k2, r_c, v_u, d_u, d_dk2)


def _p6_fourier_bwd(res, g):
    k2, r_c, v_u, d_u, d_dk2 = res
    dtype = k2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    inv_rc2 = jnp.asarray(1.0, dtype=dtype) / (rc * rc)
    inv_rc4 = inv_rc2 * inv_rc2
    # d/dr_c [r_c^-3 hat{l}_6(k^2 r_c^2)] = -3 r_c^-4 v_u + 2 k^2 r_c^-2 d_u
    d_drc_elem = -3.0 * inv_rc4 * v_u + 2.0 * k2 * inv_rc2 * d_u
    d_rc = jnp.sum(g * d_drc_elem)
    if jnp.ndim(r_c) == 0:
        d_rc = jnp.reshape(d_rc, ()).astype(jnp.asarray(r_c).dtype)
    return (g * d_dk2).astype(dtype), d_rc


_p6_fourier_val_vjp.defvjp(_p6_fourier_fwd, _p6_fourier_bwd)


def prolate_p6_fourier_kernel_and_d_dk2(
    k2: jnp.ndarray,
    r_c: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate 3D Fourier kernel ``hat{L}_6(k^2; r_c)`` and ``d(hat{L}_6)/d(k^2)``.

    Retains the finite positive ``k^2 = 0`` continuum dispersion limit
    ``hat{L}_6(0; r_c) = 4 * pi * r_c^{-3} * (int_0^1 P_6(u^2) u^2 du + 1/3)``
    while returning exact ``0.0`` for ``d/d(k^2)`` at ``k^2 == 0``.
    """
    k2_arr = jnp.asarray(k2)
    dtype = (
        k2_arr.dtype
        if jnp.issubdtype(k2_arr.dtype, jnp.floating)
        else jnp.result_type(k2_arr, 1.0)
    )
    k2_f = k2_arr.astype(dtype)
    rc_f = jnp.asarray(r_c, dtype=dtype)
    val = _p6_fourier_val_vjp(k2_f, rc_f)
    _, d_u = _p6_fourier_unit_val_and_d_dkappa2(k2_f * (rc_f * rc_f))
    d_dk2 = d_u / rc_f
    return val, d_dk2


# =============================================================================
# 6. End-to-End Differentiable Prolate SPME Electrostatics & Dispersion PME
# =============================================================================


def _minimum_image_dr(
    dr: jnp.ndarray,
    box_size: jnp.ndarray,
    is_triclinic: bool = False,
) -> jnp.ndarray:
    """Minimum-image displacement for orthorhombic ``(3,)`` or lower-triangular ``(3, 3)`` box."""
    box_arr = jnp.asarray(box_size, dtype=dr.dtype)
    if box_arr.ndim == 1 and not is_triclinic:
        return dr - box_arr * jnp.floor(dr / box_arr + 0.5)
    c_vec = box_arr[2]
    b_vec = box_arr[1]
    a_vec = box_arr[0]
    dr_out = dr - jnp.floor(dr[..., 2:3] / c_vec[2] + 0.5) * c_vec
    dr_out = dr_out - jnp.floor(dr_out[..., 1:2] / b_vec[1] + 0.5) * b_vec
    return dr_out - jnp.floor(dr_out[..., 0:1] / a_vec[0] + 0.5) * a_vec


def _reciprocal_k2_and_vol(
    grid_dims: tuple[int, int, int],
    box_arr: jnp.ndarray,
    is_triclinic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute angular wavevector norm squared ``k^2 = 4 * pi^2 * |m|^2`` and cell volume ``V``."""
    kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    nx = jnp.fft.fftfreq(kx, dtype=box_arr.dtype) * kx
    ny = jnp.fft.fftfreq(ky, dtype=box_arr.dtype) * ky
    nz = jnp.fft.rfftfreq(kz, dtype=box_arr.dtype) * kz
    four_pi2 = jnp.asarray(4.0 * (math.pi**2), dtype=box_arr.dtype)
    if box_arr.ndim == 2 or is_triclinic:
        n_grid = jnp.stack(jnp.meshgrid(nx, ny, nz, indexing="ij"), axis=0)
        b_inv = invert_box_lower_triangular(box_arr)
        m_vecs = jnp.einsum(
            "ij,jxyz->ixyz", b_inv, n_grid, precision=jax.lax.Precision.HIGHEST
        )
        m_sq = jnp.sum(m_vecs * m_vecs, axis=0)
        return four_pi2 * m_sq, box_volume(box_arr)
    mx = nx / box_arr[0]
    my = ny / box_arr[1]
    mz = nz / box_arr[2]
    m_sq = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2
    return four_pi2 * m_sq, box_arr[0] * box_arr[1] * box_arr[2]


def _prolate_p1_influence_function(
    grid_dims: tuple[int, int, int],
    box_size: jnp.ndarray,
    r_c: float,
    order: int = 4,
    deconv_mode: DeconvMode = "ik_rms16",
    is_triclinic: bool = False,
) -> jnp.ndarray:
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim == 3 or (
        box_arr.ndim == 2 and not is_triclinic and not _is_triclinic_box_array(box_arr)
    ):
        return jax.vmap(
            lambda b: _prolate_p1_influence_function(
                grid_dims,
                b,
                r_c,
                order=order,
                deconv_mode=deconv_mode,
                is_triclinic=(box_arr.ndim == 3 or is_triclinic),
            )
        )(box_arr)
    tric = bool(is_triclinic or box_arr.ndim == 2)
    kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    k2, vol = _reciprocal_k2_and_vol(grid_dims, box_arr, tric)
    hat_l1, _ = prolate_p1_fourier_kernel_and_d_dk2(k2, r_c)

    bx = _bspline_modulation(
        jnp.fft.fftfreq(kx, dtype=box_arr.dtype), kx, order, deconv_mode=deconv_mode
    )
    by = _bspline_modulation(
        jnp.fft.fftfreq(ky, dtype=box_arr.dtype), ky, order, deconv_mode=deconv_mode
    )
    bz = _bspline_modulation(
        jnp.fft.rfftfreq(kz, dtype=box_arr.dtype), kz, order, deconv_mode=deconv_mode
    )
    b_sq = jnp.maximum(
        (bx[:, None, None] * by[None, :, None] * bz[None, None, :]) ** 2,
        jnp.asarray(1e-10, dtype=box_arr.dtype),
    )
    g_mesh = hat_l1 / (vol * b_sq)
    return g_mesh.at[0, 0, 0].set(0.0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _prolate_p1_recip_and_self_energy(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    r_c: float,
    order: int,
    use_pallas_pme: bool,
    deconv_mode: DeconvMode,
    is_triclinic: bool,
) -> jnp.ndarray:
    orig_dtype = positions.dtype
    tric = bool(is_triclinic or _is_triclinic_box_array(box_size))
    is_ortho = (not tric) and _is_ortho_box(positions, box_size)
    use_p = bool(use_pallas_pme and not in_export_context() and is_ortho)

    pos_w = positions.astype(jnp.float32) if use_p else positions
    q_w = charges.astype(jnp.float32) if use_p else jnp.asarray(charges, dtype=orig_dtype)
    box_w = jnp.asarray(box_size, dtype=jnp.float32 if use_p else orig_dtype)

    q_grid = spread_charges(pos_w, q_w, atom_mask, box_w, grid_dims, order, use_p)
    if use_p:
        q_grid = jnp.asarray(q_grid, dtype=jnp.float32)
    q_hat = jnp.fft.rfftn(q_grid, s=grid_dims, axes=(-3, -2, -1))
    g1 = _prolate_p1_influence_function(
        grid_dims,
        box_w,
        r_c,
        order=order,
        deconv_mode=deconv_mode,
        is_triclinic=tric,
    ).astype(q_grid.dtype)
    theta = jnp.fft.irfftn(q_hat * g1, s=grid_dims, axes=(-3, -2, -1)).astype(q_grid.dtype)
    n_grid = jnp.asarray(
        float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=q_grid.dtype
    )
    e_rec = (
        jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=q_grid.dtype)
        * jnp.sum(q_grid * theta, axis=(-3, -2, -1))
        * n_grid
    ).astype(orig_dtype)

    q_m = jnp.asarray(charges, dtype=orig_dtype) * atom_mask.astype(orig_dtype)
    c_self = prolate_p1_self_energy_coeff(jnp.asarray(r_c, dtype=orig_dtype))
    e_self = jnp.asarray(COULOMB_CONSTANT, dtype=orig_dtype) * c_self * jnp.sum(q_m * q_m, axis=-1)
    return (e_rec + e_self).astype(orig_dtype)


def _unwrap_p(x: Any) -> Any:
    return x.value if hasattr(x, "perturbed") else x


def _is_pert(x: Any, default: bool = True) -> bool:
    return bool(x.perturbed) if hasattr(x, "perturbed") else bool(default)


def _prolate_p1_recip_fwd(
    positions,
    charges,
    atom_mask,
    box_size,
    grid_dims,
    r_c,
    order,
    use_pallas_pme,
    deconv_mode,
    is_triclinic,
):
    pos_v = _unwrap_p(positions)
    q_v = _unwrap_p(charges)
    mask_v = _unwrap_p(atom_mask)
    box_v = _unwrap_p(box_size)
    box_pert = _is_pert(box_size)
    q_pert = _is_pert(charges)

    orig_dtype = pos_v.dtype
    tric = bool(is_triclinic or _is_triclinic_box_array(box_v))
    is_ortho = (not tric) and _is_ortho_box(pos_v, box_v)
    use_p = bool(use_pallas_pme and not in_export_context() and is_ortho)

    pos_w = pos_v.astype(jnp.float32) if use_p else pos_v
    q_w = q_v.astype(jnp.float32) if use_p else jnp.asarray(q_v, dtype=orig_dtype)
    box_w = jnp.asarray(box_v, dtype=jnp.float32 if use_p else orig_dtype)

    q_grid = spread_charges(pos_w, q_w, mask_v, box_w, grid_dims, order, use_p)
    if use_p:
        q_grid = jnp.asarray(q_grid, dtype=jnp.float32)
    q_hat = jnp.fft.rfftn(q_grid, s=grid_dims, axes=(-3, -2, -1))
    g1 = _prolate_p1_influence_function(
        grid_dims,
        box_w,
        r_c,
        order=order,
        deconv_mode=deconv_mode,
        is_triclinic=tric,
    ).astype(q_grid.dtype)
    theta = jnp.fft.irfftn(q_hat * g1, s=grid_dims, axes=(-3, -2, -1)).astype(q_grid.dtype)
    n_grid = jnp.asarray(
        float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=q_grid.dtype
    )
    theta_norm = (theta * n_grid).astype(q_grid.dtype)
    e_rec = (
        jnp.asarray(0.5 * COULOMB_CONSTANT, dtype=q_grid.dtype)
        * jnp.sum(q_grid * theta_norm, axis=(-3, -2, -1))
    ).astype(orig_dtype)

    q_m = jnp.asarray(q_v, dtype=orig_dtype) * mask_v.astype(orig_dtype)
    c_self = prolate_p1_self_energy_coeff(jnp.asarray(r_c, dtype=orig_dtype))
    e_self = jnp.asarray(COULOMB_CONSTANT, dtype=orig_dtype) * c_self * jnp.sum(q_m * q_m, axis=-1)
    return (e_rec + e_self).astype(orig_dtype), (
        pos_v,
        q_v,
        mask_v,
        box_v,
        theta_norm,
        box_pert,
        q_pert,
    )


def _prolate_p1_recip_bwd(
    grid_dims,
    r_c,
    order,
    use_pallas_pme,
    deconv_mode,
    is_triclinic,
    residuals,
    g,
):
    pos_v, q_v, mask_v, box_v, theta_norm, box_pert, q_pert = residuals
    dtype = pos_v.dtype
    box_arr = jnp.asarray(box_v, dtype=dtype)
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return (
            jnp.zeros_like(pos_v),
            jnp.zeros_like(q_v),
            jnp.zeros_like(mask_v, dtype=dtype),
            jnp.zeros_like(box_arr),
        )

    tric = bool(is_triclinic or _is_triclinic_box_array(box_arr))
    use_p = bool(
        use_pallas_pme
        and not in_export_context()
        and (not tric)
        and _is_ortho_box(pos_v, box_arr)
    )
    if use_p:
        backend = "gpu" if jax.default_backend() == "gpu" else "cpu"
        gather_op = make_gather_op(
            grid_dims,
            order,
            compute_dq=q_pert,
            context=Context(kernels=True, backend=backend),
        )
        forces, potentials = gather_op(
            pos_v.astype(jnp.float32),
            q_v.astype(jnp.float32),
            mask_v,
            box_arr.astype(jnp.float32),
            theta_norm.astype(jnp.float32),
        )
        forces = forces.astype(dtype)
        if q_pert:
            potentials = potentials.astype(dtype)
    else:
        forces, potentials = _gather_reference(
            pos_v,
            q_v,
            mask_v,
            box_arr,
            theta_norm.astype(dtype),
            grid_dims=grid_dims,
            order=order,
        )

    g_pos = jnp.expand_dims(g, axis=(-2, -1)) if g.ndim > 0 else g
    de_dpos = (COULOMB_CONSTANT * forces * g_pos).astype(dtype)

    if q_pert:
        g_q = jnp.expand_dims(g, axis=-1) if g.ndim > 0 else g
        q_self = q_v[None, :] if (potentials.ndim == 2 and q_v.ndim == 1) else q_v
        m_self = mask_v[None, :] if (potentials.ndim == 2 and mask_v.ndim == 1) else mask_v
        c_self = prolate_p1_self_energy_coeff(jnp.asarray(r_c, dtype=dtype))
        de_dq = COULOMB_CONSTANT * (potentials + 2.0 * c_self * q_self.astype(dtype))
        de_dq = (de_dq * m_self.astype(dtype) * g_q).astype(q_v.dtype)
        while de_dq.ndim > q_v.ndim:
            de_dq = jnp.sum(de_dq, axis=0)
    else:
        de_dq = jnp.zeros_like(q_v)

    if box_pert:
        def _pure_rec(b_in: jnp.ndarray) -> jnp.ndarray:
            qg = spread_charges(pos_v, q_v.astype(dtype), mask_v, b_in, grid_dims, order, False)
            qh = jnp.fft.rfftn(qg, s=grid_dims, axes=(-3, -2, -1))
            gf = _prolate_p1_influence_function(
                grid_dims,
                b_in,
                r_c,
                order=order,
                deconv_mode=deconv_mode,
                is_triclinic=tric,
            ).astype(dtype)
            th = jnp.fft.irfftn(qh * gf, s=grid_dims, axes=(-3, -2, -1)).astype(dtype)
            ng = jnp.asarray(
                float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=dtype
            )
            return (0.5 * COULOMB_CONSTANT * jnp.sum(qg * th, axis=(-3, -2, -1)) * ng).astype(dtype)

        _, vjp_b = jax.vjp(_pure_rec, box_arr)
        (de_dbox,) = vjp_b(g.astype(dtype))
        de_dbox = de_dbox.astype(box_arr.dtype)
    else:
        de_dbox = jnp.zeros_like(box_arr)

    return de_dpos, de_dq, jnp.zeros_like(mask_v, dtype=dtype), de_dbox


_prolate_p1_recip_and_self_energy.defvjp(
    _prolate_p1_recip_fwd, _prolate_p1_recip_bwd, symbolic_zeros=True
)


def prolate_spme_electrostatics(  # noqa: PLR0913
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    box_size: jnp.ndarray,
    *,
    r_c: float = 9.0,
    cutoff: float | None = None,
    grid_dims: tuple[int, int, int] | None = None,
    ewald_error_tolerance: float = 5e-4,
    order: int = 4,
    neighbor_idx: jnp.ndarray | None = None,
    atom_mask: jnp.ndarray | None = None,
    excl_indices: jnp.ndarray | None = None,
    excl_mask: jnp.ndarray | None = None,
    excl_scale_elec: jnp.ndarray | None = None,
    exceptions_use_pbc: bool = False,
    include_background: bool = True,
    is_triclinic: bool = False,
    deconv_mode: DeconvMode = "ik_rms16",
    use_pallas_pme: bool | None = None,
) -> jnp.ndarray:
    """End-to-end ``C^2`` Prolate SPME electrostatics (direct + reciprocal + self + DC + exclusions).

    Supports orthorhombic ``(3,)`` and lower-triangular triclinic ``(3, 3)`` boxes
    (with ``B == 3`` disambiguated via ``is_triclinic``), ``O(N * K)`` neighbor lists
    (or dense small-``N`` evaluation when ``neighbor_idx is None``), analytical
    ``@jax.custom_vjp`` forces/gradients, and pure StableHLO export.
    """
    if cutoff is not None:
        r_c = float(cutoff)
    dtype = positions.dtype
    n = int(positions.shape[-2])
    q_arr = jnp.asarray(charges, dtype=dtype)
    box_arr = jnp.asarray(box_size, dtype=dtype)
    tric = bool(is_triclinic or _is_triclinic_box_array(box_size))

    if atom_mask is None:
        mask_arr = jnp.ones(positions.shape[:-1], dtype=bool)
    else:
        mask_arr = jnp.asarray(atom_mask, dtype=bool)

    if grid_dims is None:
        _, eff_dims = compute_openmm_pme_params(
            box_size, cutoff=r_c, ewald_error_tolerance=ewald_error_tolerance
        )
    else:
        eff_dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    pallas_ok = resolve_use_pallas_pme(
        use_pallas_pme, box=box_arr, is_triclinic=tric
    )

    # 1. Reciprocal + self energy (supports single or batched)
    e_rec_self = _prolate_p1_recip_and_self_energy(
        positions,
        q_arr,
        mask_arr,
        box_arr,
        eff_dims,
        float(r_c),
        int(order),
        pallas_ok,
        deconv_mode,
        tric,
    )

    # 2. Neutralizing-plasma DC term
    if include_background:
        q_tot = jnp.sum(q_arr * mask_arr.astype(dtype), axis=-1)
        if positions.ndim == 3 and box_arr.ndim == 2 and not tric:
            vol = jnp.prod(box_arr, axis=-1)
        elif box_arr.ndim == 3:
            vol = jax.vmap(box_volume)(box_arr)
        else:
            vol = box_volume(box_arr)
        c_dc = prolate_p1_dc_coeff(jnp.asarray(r_c, dtype=dtype))
        e_bg = jnp.asarray(COULOMB_CONSTANT, dtype=dtype) * (c_dc / vol) * (q_tot * q_tot)
    else:
        e_bg = jnp.zeros_like(e_rec_self)

    def _direct_and_excl_one(
        pos_1: jnp.ndarray,
        q_1: jnp.ndarray,
        m_1: jnp.ndarray,
        bx_1: jnp.ndarray,
        nb_1: jnp.ndarray | None,
        ex_s_1: jnp.ndarray | None,
    ) -> jnp.ndarray:
        rc_sq = float(r_c) * float(r_c)
        if nb_1 is not None:
            i_idx = jnp.arange(n, dtype=jnp.int32)[:, None]
            nb_clean = jnp.where(nb_1 < 0, n, nb_1 & ((1 << 29) - 1))
            valid_p = (nb_clean >= 0) & (nb_clean < n) & (i_idx < nb_clean)
            j_clip = jnp.clip(nb_clean, 0, max(n - 1, 0))
            valid_p = valid_p & m_1[:, None] & m_1[j_clip]
            j_s = jnp.where(valid_p, nb_clean, 0)
            dr = _minimum_image_dr(pos_1[:, None, :] - pos_1[j_s], bx_1, tric)
            r2 = jnp.sum(dr * dr, axis=-1)
            in_cut = valid_p & (r2 < rc_sq)
            r2_safe = jnp.where(in_cut, r2, jnp.asarray(rc_sq, dtype=dtype))
            s1, _ = prolate_p1_short_range_value_and_d_dr2(r2_safe, r_c)
            u_p = COULOMB_CONSTANT * (q_1[:, None] * q_1[j_s]) * s1
            if ex_s_1 is not None:
                u_p = u_p * jnp.asarray(ex_s_1, dtype=dtype)
            e_dir_1 = jnp.sum(jnp.where(in_cut, u_p, 0.0))
        else:
            dr = _minimum_image_dr(pos_1[:, None, :] - pos_1[None, :, :], bx_1, tric)
            r2 = jnp.sum(dr * dr, axis=-1)
            idx_a = jnp.arange(n)
            valid_p = (idx_a[:, None] < idx_a[None, :]) & m_1[:, None] & m_1[None, :]
            if excl_indices is not None and excl_indices.shape[0] > 0:
                # Zero out direct pairs that appear in excl_indices (1-2/1-3 full exclusions)
                ex_v = (
                    (excl_indices[:, 0] >= 0)
                    & (excl_indices[:, 0] < n)
                    & (excl_indices[:, 1] >= 0)
                    & (excl_indices[:, 1] < n)
                )
                if excl_mask is not None:
                    ex_v = ex_v & jnp.asarray(excl_mask, dtype=bool)
                ea = jnp.where(ex_v, excl_indices[:, 0], 0)
                eb = jnp.where(ex_v, excl_indices[:, 1], 0)
                ex_mat = jnp.zeros((n, n), dtype=bool)
                ex_mat = ex_mat.at[ea, eb].set(ex_v).at[eb, ea].set(ex_v)
                valid_p = valid_p & (~ex_mat)
            in_cut = valid_p & (r2 < rc_sq)
            r2_safe = jnp.where(in_cut, r2, jnp.asarray(rc_sq, dtype=dtype))
            s1, _ = prolate_p1_short_range_value_and_d_dr2(r2_safe, r_c)
            u_p = COULOMB_CONSTANT * (q_1[:, None] * q_1[None, :]) * s1
            e_dir_1 = jnp.sum(jnp.where(in_cut, u_p, 0.0))

        e_ex_1 = jnp.asarray(0.0, dtype=dtype)
        if excl_indices is not None and excl_indices.shape[0] > 0:
            p1 = excl_indices[:, 0]
            p2 = excl_indices[:, 1]
            v_ex = (p1 >= 0) & (p1 < n) & (p2 >= 0) & (p2 < n)
            if excl_mask is not None:
                v_ex = v_ex & jnp.asarray(excl_mask, dtype=bool)
            p1_s = jnp.where(v_ex, p1, 0)
            p2_s = jnp.where(v_ex, p2, 0)
            dr_ex = pos_1[p1_s] - pos_1[p2_s]
            if exceptions_use_pbc:
                dr_ex = _minimum_image_dr(dr_ex, bx_1, tric)
            r2_ex = jnp.where(v_ex, jnp.sum(dr_ex * dr_ex, axis=-1), jnp.ones((p1.shape[0],), dtype=dtype))
            l1_ex, _ = prolate_p1_long_range_value_and_d_dr2(r2_ex, r_c)
            u_sub = -COULOMB_CONSTANT * q_1[p1_s] * q_1[p2_s] * l1_ex
            e_ex_1 = jnp.sum(jnp.where(v_ex, u_sub, 0.0))
        return e_dir_1 + e_ex_1

    if positions.ndim == 3:
        b_cnt = positions.shape[0]
        q_b = q_arr.ndim == 2
        m_b = mask_arr.ndim == 2
        bx_b = box_arr.ndim == 3 or (box_arr.ndim == 2 and not tric)
        nb_b = neighbor_idx is not None and neighbor_idx.ndim == 3
        exs_b = excl_scale_elec is not None and excl_scale_elec.ndim == 3
        e_de = jax.vmap(
            lambda b: _direct_and_excl_one(
                positions[b],
                q_arr[b] if q_b else q_arr,
                mask_arr[b] if m_b else mask_arr,
                box_arr[b] if bx_b else box_arr,
                neighbor_idx[b] if (neighbor_idx is not None and nb_b) else neighbor_idx,
                excl_scale_elec[b] if (excl_scale_elec is not None and exs_b) else excl_scale_elec,
            )
        )(jnp.arange(b_cnt))
    else:
        e_de = _direct_and_excl_one(
            positions, q_arr, mask_arr, box_arr, neighbor_idx, excl_scale_elec
        )

    return (e_rec_self + e_bg + e_de).astype(dtype)


def _bspline5_deriv(u: jnp.ndarray) -> jnp.ndarray:
    """Analytical derivative ``d M_5(u + k) / du`` for 5th-order cardinal B-splines."""
    m4 = _bspline4(u)
    return jnp.stack(
        [
            -m4[0],
            m4[0] - m4[1],
            m4[1] - m4[2],
            m4[2] - m4[3],
            m4[3],
        ]
    )


def _gather_dispersion_reference(
    positions: jnp.ndarray,
    disp_charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    theta_norm: jnp.ndarray,
    *,
    grid_dims: tuple[int, int, int],
    order: int = 5,
    is_triclinic: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gather ``(dE/dpos, dE/db)`` for order-4 or order-5 dispersion SPME."""
    kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])

    def _one(pos_1, b_1, mask_1, box_1, th_1):
        dtype = pos_1.dtype
        k_arr = jnp.array([kx, ky, kz], dtype=dtype)
        box_arr = jnp.asarray(box_1, dtype=dtype)
        tric = bool(is_triclinic or box_arr.ndim == 2)
        if tric:
            b_inv = invert_box_lower_triangular(box_arr)
            s = jnp.matmul(pos_1, b_inv, precision=jax.lax.Precision.HIGHEST)
            s_wrap = s - jnp.floor(s)
            frac = s_wrap * k_arr
        else:
            frac = (pos_1 / box_arr) * k_arr
        frac = frac - jnp.floor(frac / k_arr) * k_arr
        grid_idx = jnp.floor(frac).astype(jnp.int32)
        u = frac - grid_idx.astype(dtype)
        if order == 5:
            wx, wy, wz = _bspline5(u[:, 0]), _bspline5(u[:, 1]), _bspline5(u[:, 2])
            dwx, dwy, dwz = (
                _bspline5_deriv(u[:, 0]),
                _bspline5_deriv(u[:, 1]),
                _bspline5_deriv(u[:, 2]),
            )
            shift = 0
        else:
            wx, wy, wz = _bspline4(u[:, 0]), _bspline4(u[:, 1]), _bspline4(u[:, 2])
            dwx, dwy, dwz = (
                _bspline4_deriv(u[:, 0]),
                _bspline4_deriv(u[:, 1]),
                _bspline4_deriv(u[:, 2]),
            )
            shift = -1

        ox, oy, oz = jnp.meshgrid(
            jnp.arange(order, dtype=jnp.int32),
            jnp.arange(order, dtype=jnp.int32),
            jnp.arange(order, dtype=jnp.int32),
            indexing="ij",
        )
        ox, oy, oz = ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)
        ix = (grid_idx[:, None, 0] + ox[None, :] + shift) % kx
        iy = (grid_idx[:, None, 1] + oy[None, :] + shift) % ky
        iz = (grid_idx[:, None, 2] + oz[None, :] + shift) % kz

        t_val = th_1[ix, iy, iz]
        w_unit = jnp.transpose(wx[ox] * wy[oy] * wz[oz], (1, 0))
        pot_1 = jnp.where(mask_1, jnp.sum(w_unit * t_val, axis=1), 0.0)

        b_masked = b_1.astype(dtype) * mask_1.astype(dtype)
        du_x = b_masked[:, None] * jnp.transpose(dwx[ox] * wy[oy] * wz[oz], (1, 0)) * t_val
        du_y = b_masked[:, None] * jnp.transpose(wx[ox] * dwy[oy] * wz[oz], (1, 0)) * t_val
        du_z = b_masked[:, None] * jnp.transpose(wx[ox] * wy[oy] * dwz[oz], (1, 0)) * t_val
        de_ds = jnp.stack(
            [
                jnp.sum(du_x, axis=1) * k_arr[0],
                jnp.sum(du_y, axis=1) * k_arr[1],
                jnp.sum(du_z, axis=1) * k_arr[2],
            ],
            axis=-1,
        )
        if tric:
            b_inv = invert_box_lower_triangular(box_arr)
            grad_pos = jnp.matmul(de_ds, b_inv.T, precision=jax.lax.Precision.HIGHEST)
        else:
            grad_pos = de_ds / box_arr[None, :]
        return grad_pos.astype(dtype), pot_1.astype(dtype)

    if positions.ndim == 3:
        b_cnt = positions.shape[0]
        b_b = disp_charges.ndim == 2
        m_b = atom_mask.ndim == 2
        bx_arr = jnp.asarray(box_size)
        bx_b = bx_arr.ndim == 3 or (bx_arr.ndim == 2 and not is_triclinic)
        return jax.vmap(
            lambda b: _one(
                positions[b],
                disp_charges[b] if b_b else disp_charges,
                atom_mask[b] if m_b else atom_mask,
                box_size[b] if bx_b else box_size,
                theta_norm[b],
            )
        )(jnp.arange(b_cnt))
    return _one(positions, disp_charges, atom_mask, box_size, theta_norm)


def _prolate_p6_influence_function(
    grid_dims: tuple[int, int, int],
    box_size: jnp.ndarray,
    r_c: float,
    order: int = 5,
    deconv_mode: DeconvMode = "euler",
    is_triclinic: bool = False,
) -> jnp.ndarray:
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim == 3 or (
        box_arr.ndim == 2 and not is_triclinic and not _is_triclinic_box_array(box_arr)
    ):
        return jax.vmap(
            lambda b: _prolate_p6_influence_function(
                grid_dims,
                b,
                r_c,
                order=order,
                deconv_mode=deconv_mode,
                is_triclinic=(box_arr.ndim == 3 or is_triclinic),
            )
        )(box_arr)
    tric = bool(is_triclinic or box_arr.ndim == 2)
    kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    k2, vol = _reciprocal_k2_and_vol(grid_dims, box_arr, tric)
    hat_l6, _ = prolate_p6_fourier_kernel_and_d_dk2(k2, r_c)

    bx2 = _bspline_moduli_sq_1d(kx, order, box_arr.dtype, real_axis=False, deconv_mode=deconv_mode)
    by2 = _bspline_moduli_sq_1d(ky, order, box_arr.dtype, real_axis=False, deconv_mode=deconv_mode)
    bz2 = _bspline_moduli_sq_1d(kz, order, box_arr.dtype, real_axis=True, deconv_mode=deconv_mode)
    b_sq = bx2[:, None, None] * by2[None, :, None] * bz2[None, None, :]
    return -hat_l6 / (vol * b_sq)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _prolate_p6_recip_and_self_energy(
    positions: jnp.ndarray,
    disp_charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    r_c: float,
    order: int,
    deconv_mode: DeconvMode,
    is_triclinic: bool,
) -> jnp.ndarray:
    orig_dtype = positions.dtype
    tric = bool(is_triclinic or _is_triclinic_box_array(box_size))
    b_w = jnp.asarray(disp_charges, dtype=orig_dtype)
    box_w = jnp.asarray(box_size, dtype=orig_dtype)

    q_grid = spread_dispersion_charges(positions, b_w, atom_mask, box_w, grid_dims, order=order)
    q_hat = jnp.fft.rfftn(q_grid, s=grid_dims, axes=(-3, -2, -1))
    g6 = _prolate_p6_influence_function(
        grid_dims,
        box_w,
        r_c,
        order=order,
        deconv_mode=deconv_mode,
        is_triclinic=tric,
    ).astype(q_grid.dtype)
    theta = jnp.fft.irfftn(q_hat * g6, s=grid_dims, axes=(-3, -2, -1)).astype(q_grid.dtype)
    n_grid = jnp.asarray(
        float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=q_grid.dtype
    )
    e_rec = (0.5 * jnp.sum(q_grid * theta, axis=(-3, -2, -1)) * n_grid).astype(orig_dtype)

    b_m = b_w * atom_mask.astype(orig_dtype)
    c_self = prolate_p6_self_energy_coeff(jnp.asarray(r_c, dtype=orig_dtype))
    e_self = c_self * jnp.sum(b_m * b_m, axis=-1)
    return (e_rec + e_self).astype(orig_dtype)


def _prolate_p6_recip_fwd(
    positions,
    disp_charges,
    atom_mask,
    box_size,
    grid_dims,
    r_c,
    order,
    deconv_mode,
    is_triclinic,
):
    pos_v = _unwrap_p(positions)
    b_v = _unwrap_p(disp_charges)
    mask_v = _unwrap_p(atom_mask)
    box_v = _unwrap_p(box_size)
    box_pert = _is_pert(box_size)
    b_pert = _is_pert(disp_charges)

    orig_dtype = pos_v.dtype
    tric = bool(is_triclinic or _is_triclinic_box_array(box_v))
    b_w = jnp.asarray(b_v, dtype=orig_dtype)
    box_w = jnp.asarray(box_v, dtype=orig_dtype)

    q_grid = spread_dispersion_charges(pos_v, b_w, mask_v, box_w, grid_dims, order=order)
    q_hat = jnp.fft.rfftn(q_grid, s=grid_dims, axes=(-3, -2, -1))
    g6 = _prolate_p6_influence_function(
        grid_dims,
        box_w,
        r_c,
        order=order,
        deconv_mode=deconv_mode,
        is_triclinic=tric,
    ).astype(q_grid.dtype)
    theta = jnp.fft.irfftn(q_hat * g6, s=grid_dims, axes=(-3, -2, -1)).astype(q_grid.dtype)
    n_grid = jnp.asarray(
        float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=q_grid.dtype
    )
    theta_norm = (theta * n_grid).astype(q_grid.dtype)
    e_rec = (0.5 * jnp.sum(q_grid * theta_norm, axis=(-3, -2, -1))).astype(orig_dtype)

    b_m = b_w * mask_v.astype(orig_dtype)
    c_self = prolate_p6_self_energy_coeff(jnp.asarray(r_c, dtype=orig_dtype))
    e_self = c_self * jnp.sum(b_m * b_m, axis=-1)
    return (e_rec + e_self).astype(orig_dtype), (
        pos_v,
        b_v,
        mask_v,
        box_v,
        theta_norm,
        box_pert,
        b_pert,
    )


def _prolate_p6_recip_bwd(
    grid_dims,
    r_c,
    order,
    deconv_mode,
    is_triclinic,
    residuals,
    g,
):
    pos_v, b_v, mask_v, box_v, theta_norm, box_pert, b_pert = residuals
    dtype = pos_v.dtype
    box_arr = jnp.asarray(box_v, dtype=dtype)
    if hasattr(jax.custom_derivatives, "SymbolicZero") and isinstance(
        g, jax.custom_derivatives.SymbolicZero
    ):
        return (
            jnp.zeros_like(pos_v),
            jnp.zeros_like(b_v),
            jnp.zeros_like(mask_v, dtype=dtype),
            jnp.zeros_like(box_arr),
        )

    tric = bool(is_triclinic or _is_triclinic_box_array(box_arr))
    grad_pos, pot_b = _gather_dispersion_reference(
        pos_v,
        b_v,
        mask_v,
        box_arr,
        theta_norm.astype(dtype),
        grid_dims=grid_dims,
        order=order,
        is_triclinic=tric,
    )
    g_pos = jnp.expand_dims(g, axis=(-2, -1)) if g.ndim > 0 else g
    de_dpos = (grad_pos * g_pos).astype(dtype)

    if b_pert:
        g_b = jnp.expand_dims(g, axis=-1) if g.ndim > 0 else g
        b_self = b_v[None, :] if (pot_b.ndim == 2 and b_v.ndim == 1) else b_v
        m_self = mask_v[None, :] if (pot_b.ndim == 2 and mask_v.ndim == 1) else mask_v
        c_self = prolate_p6_self_energy_coeff(jnp.asarray(r_c, dtype=dtype))
        de_db = (pot_b + 2.0 * c_self * b_self.astype(dtype)) * m_self.astype(dtype) * g_b
        de_db = de_db.astype(b_v.dtype)
        while de_db.ndim > b_v.ndim:
            de_db = jnp.sum(de_db, axis=0)
    else:
        de_db = jnp.zeros_like(b_v)

    if box_pert:
        def _pure_rec6(b_in: jnp.ndarray) -> jnp.ndarray:
            qg = spread_dispersion_charges(
                pos_v, b_v.astype(dtype), mask_v, b_in, grid_dims, order=order
            )
            qh = jnp.fft.rfftn(qg, s=grid_dims, axes=(-3, -2, -1))
            gf = _prolate_p6_influence_function(
                grid_dims,
                b_in,
                r_c,
                order=order,
                deconv_mode=deconv_mode,
                is_triclinic=tric,
            ).astype(dtype)
            th = jnp.fft.irfftn(qh * gf, s=grid_dims, axes=(-3, -2, -1)).astype(dtype)
            ng = jnp.asarray(
                float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=dtype
            )
            return (0.5 * jnp.sum(qg * th, axis=(-3, -2, -1)) * ng).astype(dtype)

        _, vjp_b = jax.vjp(_pure_rec6, box_arr)
        (de_dbox,) = vjp_b(g.astype(dtype))
        de_dbox = de_dbox.astype(box_arr.dtype)
    else:
        de_dbox = jnp.zeros_like(box_arr)

    return de_dpos, de_db, jnp.zeros_like(mask_v, dtype=dtype), de_dbox


_prolate_p6_recip_and_self_energy.defvjp(
    _prolate_p6_recip_fwd, _prolate_p6_recip_bwd, symbolic_zeros=True
)


def _c2_quintic_shell_switch(r2: jnp.ndarray, r_c: float) -> jnp.ndarray:
    """``C^2`` quintic taper equal to ``1`` for ``r <= 0.9 * r_c`` and ``0`` at ``r >= r_c``."""
    dtype = r2.dtype
    rc = jnp.asarray(r_c, dtype=dtype)
    r_sw = jnp.asarray(0.9 * float(r_c), dtype=dtype)
    in_shell = (r2 > r_sw * r_sw) & (r2 < rc * rc)
    r_safe = jnp.sqrt(jnp.where(in_shell, r2, r_sw * r_sw))
    z = (r_safe - r_sw) / (rc - r_sw)
    z2 = z * z
    z3 = z2 * z
    s_poly = 1.0 - z3 * (10.0 - 15.0 * z + 6.0 * z2)
    return jnp.where(r2 <= r_sw * r_sw, jnp.ones_like(r2), jnp.where(r2 >= rc * rc, jnp.zeros_like(r2), s_poly))


def prolate_dispersion_pme(  # noqa: PLR0913
    positions: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    box_size: jnp.ndarray,
    *,
    r_c: float = 9.0,
    cutoff: float | None = None,
    grid_dims: tuple[int, int, int] | None = None,
    ewald_error_tolerance: float = 5e-4,
    order: int = 5,
    neighbor_idx: jnp.ndarray | None = None,
    atom_mask: jnp.ndarray | None = None,
    excl_indices: jnp.ndarray | None = None,
    excl_mask: jnp.ndarray | None = None,
    excl_scale_vdw: jnp.ndarray | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    exceptions_use_pbc: bool = False,
    include_repulsion: bool = True,
    c2_repulsion_switch: bool = True,
    is_triclinic: bool = False,
    deconv_mode: DeconvMode = "euler",
) -> jnp.ndarray:
    """End-to-end ``C^2`` Prolate ``P=6`` dispersion + ``r^{-12}`` LJ-PME energy.

    Supports orthorhombic ``(3,)`` and lower-triangular triclinic ``(3, 3)`` boxes,
    both ``geometric`` and ``lorentz_berthelot`` short-range LJ combining rules,
    analytical ``@jax.custom_vjp`` gradients w.r.t. ``(positions, sigmas, epsilons, box_size)``,
    and pure portable StableHLO export.
    """
    if cutoff is not None:
        r_c = float(cutoff)
    dtype = positions.dtype
    n = int(positions.shape[-2])
    sig_arr = jnp.asarray(sigmas, dtype=dtype)
    eps_arr = jnp.asarray(epsilons, dtype=dtype)
    box_arr = jnp.asarray(box_size, dtype=dtype)
    tric = bool(is_triclinic or _is_triclinic_box_array(box_size))

    if atom_mask is None:
        mask_arr = jnp.ones(positions.shape[:-1], dtype=bool)
    else:
        mask_arr = jnp.asarray(atom_mask, dtype=bool)

    if grid_dims is None:
        _, eff_dims = resolve_ljpme_parameters(
            box_size, cutoff=r_c, ewald_error_tolerance=ewald_error_tolerance
        )
    else:
        eff_dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    b_disp = dispersion_charges(sig_arr, eps_arr, atom_mask=mask_arr)

    e_rec_self = _prolate_p6_recip_and_self_energy(
        positions,
        b_disp,
        mask_arr,
        box_arr,
        eff_dims,
        float(r_c),
        int(order),
        deconv_mode,
        tric,
    )

    def _pair_lj_prolate(
        r2_safe: jnp.ndarray,
        sig_ij: jnp.ndarray,
        eps_ij: jnp.ndarray,
        b_prod: jnp.ndarray,
    ) -> jnp.ndarray:
        s2 = sig_ij * sig_ij
        sr2 = s2 / r2_safe
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        c6_phys = 4.0 * eps_ij * (s2 * s2 * s2)
        delta_c6 = c6_phys - b_prod
        s6, _ = prolate_p6_short_range_value_and_d_dr2(r2_safe, r_c)
        inv_r6 = 1.0 / (r2_safe * r2_safe * r2_safe)
        u_short_6 = -b_prod * s6 - delta_c6 * inv_r6
        u_rep = (
            jnp.where(eps_ij > 0.0, 4.0 * eps_ij * sr12, 0.0)
            if include_repulsion
            else jnp.zeros_like(r2_safe)
        )
        if c2_repulsion_switch:
            taper = _c2_quintic_shell_switch(r2_safe, r_c)
            return taper * u_rep - b_prod * s6 - taper * (delta_c6 * inv_r6)
        return u_rep + u_short_6

    def _direct_and_excl6_one(
        pos_1: jnp.ndarray,
        sig_1: jnp.ndarray,
        eps_1: jnp.ndarray,
        b_1: jnp.ndarray,
        m_1: jnp.ndarray,
        bx_1: jnp.ndarray,
        nb_1: jnp.ndarray | None,
        ex_s_1: jnp.ndarray | None,
    ) -> jnp.ndarray:
        rc_sq = float(r_c) * float(r_c)
        if nb_1 is not None:
            i_idx = jnp.arange(n, dtype=jnp.int32)[:, None]
            nb_clean = jnp.where(nb_1 < 0, n, nb_1 & ((1 << 29) - 1))
            valid_p = (nb_clean >= 0) & (nb_clean < n) & (i_idx < nb_clean)
            j_clip = jnp.clip(nb_clean, 0, max(n - 1, 0))
            valid_p = valid_p & m_1[:, None] & m_1[j_clip]
            j_s = jnp.where(valid_p, nb_clean, 0)
            dr = _minimum_image_dr(pos_1[:, None, :] - pos_1[j_s], bx_1, tric)
            r2 = jnp.sum(dr * dr, axis=-1)
            in_cut = valid_p & (r2 < rc_sq)
            r2_safe = jnp.where(in_cut, r2, jnp.asarray(rc_sq, dtype=dtype))
            sig_ij, eps_ij = _combine_lj_pair(
                sig_1[:, None], sig_1[j_s], eps_1[:, None], eps_1[j_s], lj_combining_rule
            )
            u_p = _pair_lj_prolate(r2_safe, sig_ij, eps_ij, b_1[:, None] * b_1[j_s])
            if ex_s_1 is not None:
                u_p = u_p * jnp.asarray(ex_s_1, dtype=dtype)
            e_dir_1 = jnp.sum(jnp.where(in_cut, u_p, 0.0))
        else:
            dr = _minimum_image_dr(pos_1[:, None, :] - pos_1[None, :, :], bx_1, tric)
            r2 = jnp.sum(dr * dr, axis=-1)
            idx_a = jnp.arange(n)
            valid_p = (idx_a[:, None] < idx_a[None, :]) & m_1[:, None] & m_1[None, :]
            if excl_indices is not None and excl_indices.shape[0] > 0:
                ex_v = (
                    (excl_indices[:, 0] >= 0)
                    & (excl_indices[:, 0] < n)
                    & (excl_indices[:, 1] >= 0)
                    & (excl_indices[:, 1] < n)
                )
                if excl_mask is not None:
                    ex_v = ex_v & jnp.asarray(excl_mask, dtype=bool)
                ea = jnp.where(ex_v, excl_indices[:, 0], 0)
                eb = jnp.where(ex_v, excl_indices[:, 1], 0)
                ex_mat = jnp.zeros((n, n), dtype=bool)
                ex_mat = ex_mat.at[ea, eb].set(ex_v).at[eb, ea].set(ex_v)
                valid_p = valid_p & (~ex_mat)
            in_cut = valid_p & (r2 < rc_sq)
            r2_safe = jnp.where(in_cut, r2, jnp.asarray(rc_sq, dtype=dtype))
            sig_ij, eps_ij = _combine_lj_pair(
                sig_1[:, None], sig_1[None, :], eps_1[:, None], eps_1[None, :], lj_combining_rule
            )
            u_p = _pair_lj_prolate(r2_safe, sig_ij, eps_ij, b_1[:, None] * b_1[None, :])
            e_dir_1 = jnp.sum(jnp.where(in_cut, u_p, 0.0))

        e_ex_1 = jnp.asarray(0.0, dtype=dtype)
        if excl_indices is not None and excl_indices.shape[0] > 0:
            p1 = excl_indices[:, 0]
            p2 = excl_indices[:, 1]
            v_ex = (p1 >= 0) & (p1 < n) & (p2 >= 0) & (p2 < n)
            if excl_mask is not None:
                v_ex = v_ex & jnp.asarray(excl_mask, dtype=bool)
            p1_s = jnp.where(v_ex, p1, 0)
            p2_s = jnp.where(v_ex, p2, 0)
            dr_ex = pos_1[p1_s] - pos_1[p2_s]
            if exceptions_use_pbc:
                dr_ex = _minimum_image_dr(dr_ex, bx_1, tric)
            r2_ex = jnp.where(
                v_ex,
                jnp.sum(dr_ex * dr_ex, axis=-1),
                jnp.ones((p1.shape[0],), dtype=dtype),
            )
            l6_ex, _ = prolate_p6_long_range_value_and_d_dr2(r2_ex, r_c)
            u_sub = b_1[p1_s] * b_1[p2_s] * l6_ex
            e_ex_1 = jnp.sum(jnp.where(v_ex, u_sub, 0.0))
        return e_dir_1 + e_ex_1

    if positions.ndim == 3:
        b_cnt = positions.shape[0]
        s_b = sig_arr.ndim == 2
        e_b = eps_arr.ndim == 2
        bd_b = b_disp.ndim == 2
        m_b = mask_arr.ndim == 2
        bx_b = box_arr.ndim == 3 or (box_arr.ndim == 2 and not tric)
        nb_b = neighbor_idx is not None and neighbor_idx.ndim == 3
        exs_b = excl_scale_vdw is not None and excl_scale_vdw.ndim == 3
        e_de = jax.vmap(
            lambda b: _direct_and_excl6_one(
                positions[b],
                sig_arr[b] if s_b else sig_arr,
                eps_arr[b] if e_b else eps_arr,
                b_disp[b] if bd_b else b_disp,
                mask_arr[b] if m_b else mask_arr,
                box_arr[b] if bx_b else box_arr,
                neighbor_idx[b] if (neighbor_idx is not None and nb_b) else neighbor_idx,
                excl_scale_vdw[b] if (excl_scale_vdw is not None and exs_b) else excl_scale_vdw,
            )
        )(jnp.arange(b_cnt))
    else:
        e_de = _direct_and_excl6_one(
            positions,
            sig_arr,
            eps_arr,
            b_disp,
            mask_arr,
            box_arr,
            neighbor_idx,
            excl_scale_vdw,
        )

    return (e_rec_self + e_de).astype(dtype)
