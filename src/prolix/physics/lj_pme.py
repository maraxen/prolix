"""P=6 Dispersion Particle-Mesh Ewald (LJ-PME) in pure differentiable JAX.

Implements exact periodic ``1/r^6`` dispersion Ewald / SPME matching OpenMM
``NonbondedForce.LJPME`` (Reference & CUDA platforms):
  - Geometric dispersion charges ``b_i = 2 * sqrt(eps_i) * sigma_i^3``.
  - Direct-space pair interaction ``4 * eps_ij * ((sig_ij/r)^12 - (sig_ij/r)^6)
    + b_i * b_j * (1 - g_6(alpha_lj * r)) / r^6`` shifted continuously to zero
    at ``r = cutoff`` (preserving exact Lorentz-Berthelot or geometric short-range
    mixing inside ``r < cutoff`` and reducing to ``-b_i * b_j * g_6(alpha_lj * r) / r^6``
    under geometric combining).
  - Reciprocal SPME mesh with ``K_6(k^2) = (pi^{3/2} / 2) * R_6(k^2)``, retaining
    the finite DC ``k^2 = 0`` mode ``R_6(0) = (2/3) * alpha_lj^3`` while keeping
    ``dK_6/dk^2|_{k^2=0} = 0`` NaN-free for box/virial autodiff.
  - Positive dispersion self-energy ``+(alpha_lj^6 / 12) * sum_i b_i^2``.
  - Degree-24 cancellation-free Taylor series for excluded-pair reciprocal
    subtraction ``+b_i * b_j * (1 - g_6(alpha_lj * r_ex)) / r_ex^6`` when
    ``y = (alpha_lj * r_ex)^2 < 2.0`` with double-``jnp.where`` AD safety.

Portions of the bare ``p=6`` Ewald kernel and small-``y`` Taylor expansion follow
``samples/ltoir-usage/ewald.py`` from NVIDIA ``cuPotential`` under the Apache
License 2.0 (Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES; see root
``NOTICE``). Zero lazy intra-package imports; lowers to pure portable StableHLO
with zero custom calls under ``prolix.export.export_context()``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from prolix.physics.pbc import box_volume, invert_box_lower_triangular
from prolix.physics.pme import (
    DeconvMode,
    _bspline4,
    _bspline_modulation,
    _is_ortho_box,
    _is_triclinic_box_array,
    compute_openmm_ljpme_params,
)

# Precomputed coefficients c_n = (-1)^n / (2 * n! * (n + 3)) for n = 24 .. 0
# (Horner descending order) for (1 - g_6(sqrt(y))) / r^6 = alpha^6 * sum_{n>=0} c_n y^n.
_P6_LONG_RANGE_HORNER_DESC: tuple[float, ...] = tuple(
    ((-1.0) ** n) / (2.0 * float(math.factorial(n)) * float(n + 3))
    for n in range(24, -1, -1)
)


def _masked_sqrt_eps(epsilons: jnp.ndarray) -> jnp.ndarray:
    """AD-safe ``sqrt(max(eps, 0))`` avoiding ``1 / (2 * sqrt(0))`` NaNs in ``jax.grad``."""
    eps_pos = epsilons > 0
    return jnp.where(
        eps_pos,
        jnp.sqrt(jnp.where(eps_pos, epsilons, jnp.ones_like(epsilons))),
        jnp.zeros_like(epsilons),
    )


def _combine_lj_pair(
    sig_i: jnp.ndarray,
    sig_j: jnp.ndarray,
    eps_i: jnp.ndarray,
    eps_j: jnp.ndarray,
    lj_combining_rule: str = "lorentz_berthelot",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Combine pair ``(sigma_ij, epsilon_ij)`` with NaN-safe square-root gradients."""
    rule = lj_combining_rule.strip().lower().replace("-", "_")
    if rule == "geometric":
        sig_prod = sig_i * sig_j
        sig_pos = sig_prod > 0
        sig_ij = jnp.where(
            sig_pos,
            jnp.sqrt(jnp.where(sig_pos, sig_prod, jnp.ones_like(sig_prod))),
            jnp.zeros_like(sig_prod),
        )
    elif rule == "lorentz_berthelot":
        sig_ij = 0.5 * (sig_i + sig_j)
    else:
        raise ValueError(
            f"Unsupported lj_combining_rule={lj_combining_rule!r}; "
            "expected 'lorentz_berthelot' or 'geometric'."
        )
    eps_ij = _masked_sqrt_eps(eps_i) * _masked_sqrt_eps(eps_j)
    return sig_ij, eps_ij


def _minimum_image_displacement(
    dr: jnp.ndarray,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """Minimum-image displacement for orthorhombic ``(3,)`` or lower-triangular ``(3, 3)`` box."""
    box_arr = jnp.asarray(box_size, dtype=dr.dtype)
    if box_arr.ndim == 1:
        return dr - box_arr * jnp.floor(dr / box_arr + 0.5)
    # Sequential z -> y -> x triclinic minimum image matching OpenMM Reference
    c_vec = box_arr[2]
    b_vec = box_arr[1]
    a_vec = box_arr[0]
    dr_out = dr - jnp.floor(dr[..., 2:3] / c_vec[2] + 0.5) * c_vec
    dr_out = dr_out - jnp.floor(dr_out[..., 1:2] / b_vec[1] + 0.5) * b_vec
    return dr_out - jnp.floor(dr_out[..., 0:1] / a_vec[0] + 0.5) * a_vec


def dispersion_charges(
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute geometric dispersion charges ``b_i = 2 * sqrt(eps_i) * sigma_i^3``."""
    sig = jnp.asarray(sigmas)
    eps = jnp.asarray(epsilons, dtype=sig.dtype)
    b = 2.0 * _masked_sqrt_eps(eps) * (sig**3)
    if atom_mask is not None:
        b = b * jnp.asarray(atom_mask, dtype=sig.dtype)
    return b


def ewald_p6_fourier_kernel_value(
    alpha_lj: float | jnp.ndarray,
    k2: jnp.ndarray,
) -> jnp.ndarray:
    """Bare Ewald ``1/r^6`` reciprocal kernel ``K_6(k^2) = (pi^{3/2} / 2) * R_6(k^2)``.

    Retains the finite DC limit ``K_6(0) = (pi^{3/2} / 3) * alpha_lj^3`` at ``k^2 = 0``
    while sanitizing ``sqrt(k^2)`` via double-``jnp.where`` so ``jax.grad`` w.r.t.
    ``k^2`` and ``box`` yields exact ``0.0`` at ``k^2 = 0`` without ``NaN``.
    """
    k2_arr = jnp.asarray(k2)
    dtype = k2_arr.dtype if jnp.issubdtype(k2_arr.dtype, jnp.floating) else jnp.result_type(k2_arr, 1.0)
    k2_f = k2_arr.astype(dtype)
    alpha_arr = jnp.asarray(alpha_lj, dtype=dtype)

    is_dc = k2_f <= 0.0
    k2_safe = jnp.where(is_dc, jnp.ones_like(k2_f), k2_f)
    k = jnp.sqrt(k2_safe)
    pi_c = jnp.asarray(jnp.pi, dtype=dtype)
    pi32 = pi_c**1.5

    t = pi_c * k / alpha_arr
    exp_t2 = jnp.exp(-t * t)
    r4 = 2.0 * alpha_arr * exp_t2 - 2.0 * pi32 * k * erfc(t)
    r6 = (4.0 * (alpha_arr**3) * exp_t2 - 4.0 * (pi_c**2) * k2_safe * r4) / 6.0
    k6_nz = 0.5 * pi32 * r6
    k6_dc = (pi32 / 3.0) * (alpha_arr**3)
    return jnp.where(is_dc, k6_dc, k6_nz)


def ewald_p6_fourier_kernel_grad(
    alpha_lj: float | jnp.ndarray,
    k2: jnp.ndarray,
) -> jnp.ndarray:
    """Derivative ``dK_6 / d(k^2)`` with exact ``0.0`` at the ``k^2 = 0`` DC mode."""
    k2_arr = jnp.asarray(k2)
    dtype = k2_arr.dtype if jnp.issubdtype(k2_arr.dtype, jnp.floating) else jnp.result_type(k2_arr, 1.0)
    k2_f = k2_arr.astype(dtype)
    grad_val = jax.grad(lambda x: jnp.sum(ewald_p6_fourier_kernel_value(alpha_lj, x)))(k2_f)
    return jnp.where(k2_f <= 0.0, jnp.zeros_like(grad_val), grad_val)


def ewald_p6_short_range_value(
    alpha_lj: float | jnp.ndarray,
    r2: jnp.ndarray,
) -> jnp.ndarray:
    """Bare short-range damped dispersion factor ``g_6(alpha_lj * r) / r^6``."""
    r2_arr = jnp.asarray(r2)
    dtype = r2_arr.dtype if jnp.issubdtype(r2_arr.dtype, jnp.floating) else jnp.result_type(r2_arr, 1.0)
    r2_f = r2_arr.astype(dtype)
    a2 = jnp.asarray(alpha_lj, dtype=dtype) ** 2
    y = a2 * r2_f
    g6 = jnp.exp(-y) * (1.0 + y + 0.5 * y * y)
    return g6 / (r2_f**3)


def ewald_p6_long_range_value(
    alpha_lj: float | jnp.ndarray,
    r2: jnp.ndarray,
) -> jnp.ndarray:
    """Bare long-range / reciprocal-exclusion dispersion factor ``(1 - g_6(alpha_lj * r)) / r^6``.

    Evaluates the cancellation-free degree-24 Taylor series for ``y = (alpha_lj * r)^2 < 2.0``
    and the direct expression for ``y >= 2.0`` with double-``jnp.where`` branch sanitization
    so ``r -> 0`` values and ``jax.grad`` derivatives remain finite in ``float32`` and ``float64``.
    """
    r2_arr = jnp.asarray(r2)
    dtype = r2_arr.dtype if jnp.issubdtype(r2_arr.dtype, jnp.floating) else jnp.result_type(r2_arr, 1.0)
    r2_f = r2_arr.astype(dtype)
    a2 = jnp.asarray(alpha_lj, dtype=dtype) ** 2
    a6 = a2**3
    y = a2 * r2_f

    small = y < 2.0
    small_y = jnp.where(small, y, jnp.zeros_like(y))
    taylor = jnp.zeros_like(y)
    for coeff in _P6_LONG_RANGE_HORNER_DESC:
        taylor = jnp.asarray(coeff, dtype=dtype) + small_y * taylor
    taylor = taylor * a6

    safe_r2 = jnp.where(small, jnp.ones_like(r2_f), r2_f)
    y_large = jnp.where(small, jnp.asarray(2.0, dtype=dtype), y)
    g6 = jnp.exp(-y_large) * (1.0 + y_large + 0.5 * y_large * y_large)
    direct = (1.0 - g6) / (safe_r2**3)
    return jnp.where(small, taylor, direct)


def resolve_ljpme_parameters(
    box_size: jnp.ndarray | np.ndarray,
    cutoff: float = 9.0,
    ewald_error_tolerance: float = 5e-4,
    *,
    explicit_alpha: float = 0.0,
    explicit_grid_dims: tuple[int, int, int] = (0, 0, 0),
    min_dim: int = 6,
) -> tuple[float, tuple[int, int, int]]:
    """Resolve ``(alpha_lj, (nx_lj, ny_lj, nz_lj))`` matching OpenMM ``LJPME`` rules.

    When ``getLJPMEParameters()`` returns ``(0, 0, 0, 0)``, falls back to
    ``alpha_lj = sqrt(-ln(2 * delta)) / r_c`` and
    ``n_lj,d = max(6, ceil(alpha_lj * L_d / (3 * delta**(1/5))))``.
    """
    default_alpha, default_dims = compute_openmm_ljpme_params(
        box_size,
        cutoff=cutoff,
        ewald_error_tolerance=ewald_error_tolerance,
        min_dim=min_dim,
    )
    if float(explicit_alpha) > 0.0:
        alpha_lj = float(explicit_alpha)
    else:
        alpha_lj = default_alpha

    nx, ny, nz = int(explicit_grid_dims[0]), int(explicit_grid_dims[1]), int(explicit_grid_dims[2])
    if nx > 0 and ny > 0 and nz > 0:
        return alpha_lj, (nx, ny, nz)

    if float(explicit_alpha) > 0.0:
        box_np = np.asarray(box_size, dtype=np.float64)
        if box_np.ndim == 3:
            box_np = np.diag(box_np[0])
        elif box_np.ndim == 2:
            box_np = np.diag(box_np) if _is_triclinic_box_array(box_np) else box_np[0]
        tol = max(float(ewald_error_tolerance), 1e-12)
        denom = 3.0 * (tol**0.2)
        dims = (
            max(int(min_dim), math.ceil(alpha_lj * float(box_np[0]) / denom)),
            max(int(min_dim), math.ceil(alpha_lj * float(box_np[1]) / denom)),
            max(int(min_dim), math.ceil(alpha_lj * float(box_np[2]) / denom)),
        )
        return alpha_lj, dims
    return alpha_lj, default_dims


def _bspline5(u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate 5th-order cardinal B-spline basis at fractional offsets ``u`` in ``[0, 1)``."""
    m4 = _bspline4(u)
    m4_0, m4_1, m4_2, m4_3 = m4[0], m4[1], m4[2], m4[3]
    w0 = (1.0 - u) * m4_0 * 0.25
    w1 = ((u + 3.0) * m4_0 + (2.0 - u) * m4_1) * 0.25
    w2 = ((u + 2.0) * m4_1 + (3.0 - u) * m4_2) * 0.25
    w3 = ((u + 1.0) * m4_2 + (4.0 - u) * m4_3) * 0.25
    w4 = u * m4_3 * 0.25
    return jnp.stack([w0, w1, w2, w3, w4])


def _bspline_moduli_sq_1d(
    k_dim: int,
    order: int,
    dtype: jnp.dtype,
    *,
    real_axis: bool = False,
    deconv_mode: DeconvMode = "euler",
) -> jnp.ndarray:
    """Compute 1D squared B-spline Euler / discrete-RMS moduli ``|b(n)|^2``."""
    if order == 4:
        freqs = (
            jnp.fft.rfftfreq(k_dim, dtype=dtype)
            if real_axis
            else jnp.fft.fftfreq(k_dim, dtype=dtype)
        )
        b_amp = _bspline_modulation(freqs, k_dim, 4, deconv_mode=deconv_mode)
        return b_amp * b_amp
    if order == 5:
        freqs_full = jnp.fft.fftfreq(k_dim, dtype=dtype)
        m5_0 = jnp.array(
            [1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0, 0.0], dtype=dtype
        )
        k_idx = jnp.arange(5, dtype=dtype)
        phases = jnp.exp(2j * jnp.pi * freqs_full[:, None] * k_idx[None, :])
        mod_sq = jnp.abs(jnp.sum(m5_0[None, :] * phases, axis=-1)) ** 2
        # OpenMM / Essmann odd-order even-grid Nyquist continuity rule:
        # at n = K/2, the raw odd-order Euler sum vanishes; replace with 0.5*(mod[K/2-1]+mod[K/2+1]) == mod[K/2-1].
        if k_dim % 2 == 0 and k_dim >= 2:
            half = k_dim // 2
            mod_sq = mod_sq.at[half].set(mod_sq[half - 1])
        mod_sq = jnp.maximum(mod_sq, jnp.asarray(1e-12, dtype=dtype))
        return mod_sq[: k_dim // 2 + 1] if real_axis else mod_sq
    raise NotImplementedError(f"LJ-PME B-spline modulation supports order 4 or 5, got {order}")


def spread_dispersion_charges(
    positions: jnp.ndarray,
    disp_charges: jnp.ndarray,
    atom_mask: jnp.ndarray | None,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    order: int = 5,
) -> jnp.ndarray:
    """Spread dispersion charges ``b_i`` onto ``(nx_lj, ny_lj, nz_lj)`` mesh."""
    if order not in (4, 5):
        raise NotImplementedError(f"spread_dispersion_charges supports order 4 or 5, got {order}")
    kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])

    def _spread_one(
        pos_1: jnp.ndarray,
        b_1: jnp.ndarray,
        mask_1: jnp.ndarray,
        box_1: jnp.ndarray,
    ) -> jnp.ndarray:
        work_dtype = pos_1.dtype
        k_arr = jnp.array([kx, ky, kz], dtype=work_dtype)
        box_arr = jnp.asarray(box_1, dtype=work_dtype)
        if box_arr.ndim == 2:
            b_inv = invert_box_lower_triangular(box_arr)
            s = jnp.matmul(pos_1, b_inv, precision=jax.lax.Precision.HIGHEST)
            s_wrap = s - jnp.floor(s)
            frac = s_wrap * k_arr
        else:
            frac = (pos_1 / box_arr) * k_arr
        frac = frac - jnp.floor(frac / k_arr) * k_arr
        grid_idx = jnp.floor(frac).astype(jnp.int32)
        u = frac - grid_idx.astype(work_dtype)
        if order == 5:
            wx = _bspline5(u[:, 0])
            wy = _bspline5(u[:, 1])
            wz = _bspline5(u[:, 2])
            shift = 0
        else:
            wx = _bspline4(u[:, 0])
            wy = _bspline4(u[:, 1])
            wz = _bspline4(u[:, 2])
            shift = -1
        b_masked = b_1.astype(work_dtype) * mask_1.astype(work_dtype)
        ox, oy, oz = jnp.meshgrid(
            jnp.arange(order, dtype=jnp.int32),
            jnp.arange(order, dtype=jnp.int32),
            jnp.arange(order, dtype=jnp.int32),
            indexing="ij",
        )
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)
        oz = oz.reshape(-1)
        ix = (grid_idx[:, None, 0] + ox[None, :] + shift) % kx
        iy = (grid_idx[:, None, 1] + oy[None, :] + shift) % ky
        iz = (grid_idx[:, None, 2] + oz[None, :] + shift) % kz
        weights = jnp.transpose(wx[ox] * wy[oy] * wz[oz] * b_masked[None, :], (1, 0))
        q_grid = jnp.zeros((kx, ky, kz), dtype=work_dtype)
        return q_grid.at[ix, iy, iz].add(weights)

    if atom_mask is None:
        mask_arr = jnp.ones(positions.shape[:-1], dtype=bool)
    else:
        mask_arr = jnp.asarray(atom_mask)

    if positions.ndim == 3:
        batch_n = positions.shape[0]
        b_batched = disp_charges.ndim == 2
        m_batched = mask_arr.ndim == 2
        box_top = jnp.asarray(box_size)
        box_batched = (
            box_top.ndim >= 2
            and box_top.shape[0] == batch_n
            and (_is_ortho_box(positions, box_top) or box_top.ndim == 3)
        )
        return jax.vmap(
            lambda idx: _spread_one(
                positions[idx],
                disp_charges[idx] if b_batched else disp_charges,
                mask_arr[idx] if m_batched else mask_arr,
                box_top[idx] if box_batched else box_top,
            )
        )(jnp.arange(batch_n))
    return _spread_one(positions, disp_charges, mask_arr, jnp.asarray(box_size))


def lj_pme_influence_function(
    grid_dims: tuple[int, int, int],
    box_size: jnp.ndarray,
    alpha_lj: float,
    order: int = 5,
    deconv_mode: DeconvMode = "euler",
) -> jnp.ndarray:
    """Dispersion reciprocal Green's tensor ``G_6(n) = -K_6(|m(n)|^2) / (V * b^2(n))``.

    Keeps the finite ``n = (0, 0, 0)`` DC entry ``G_6(0) = -(pi^{3/2} / 3) * alpha_lj^3 / V``
    (matching continuum isotropic dispersion at ``k = 0``) and remains ``NaN``-free under
    ``jax.grad`` w.r.t. ``box_size``.
    """
    box_arr = jnp.asarray(box_size)
    if box_arr.ndim == 2 and not _is_triclinic_box_array(box_arr):
        return jax.vmap(
            lambda b: lj_pme_influence_function(
                grid_dims, b, alpha_lj, order=order, deconv_mode=deconv_mode
            )
        )(box_arr)

    kx, ky, kz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    vol = box_volume(box_arr)

    if box_arr.ndim == 2:
        nx = jnp.fft.fftfreq(kx, dtype=box_arr.dtype) * kx
        ny = jnp.fft.fftfreq(ky, dtype=box_arr.dtype) * ky
        nz = jnp.fft.rfftfreq(kz, dtype=box_arr.dtype) * kz
        n_grid = jnp.stack(jnp.meshgrid(nx, ny, nz, indexing="ij"), axis=0)
        b_inv = invert_box_lower_triangular(box_arr)
        m_vecs = jnp.einsum("ij,jxyz->ixyz", b_inv, n_grid, precision=jax.lax.Precision.HIGHEST)
        k2 = jnp.sum(m_vecs**2, axis=0)
    else:
        mx = jnp.fft.fftfreq(kx, dtype=box_arr.dtype) * kx / box_arr[0]
        my = jnp.fft.fftfreq(ky, dtype=box_arr.dtype) * ky / box_arr[1]
        mz = jnp.fft.rfftfreq(kz, dtype=box_arr.dtype) * kz / box_arr[2]
        k2 = mx[:, None, None] ** 2 + my[None, :, None] ** 2 + mz[None, None, :] ** 2

    k6 = ewald_p6_fourier_kernel_value(alpha_lj, k2)
    bx2 = _bspline_moduli_sq_1d(kx, order, box_arr.dtype, real_axis=False, deconv_mode=deconv_mode)
    by2 = _bspline_moduli_sq_1d(ky, order, box_arr.dtype, real_axis=False, deconv_mode=deconv_mode)
    bz2 = _bspline_moduli_sq_1d(kz, order, box_arr.dtype, real_axis=True, deconv_mode=deconv_mode)
    b_sq = bx2[:, None, None] * by2[None, :, None] * bz2[None, None, :]
    return -k6 / (vol * b_sq)


def lj_pme_reciprocal_energy(
    positions: jnp.ndarray,
    disp_charges: jnp.ndarray,
    atom_mask: jnp.ndarray | None,
    box_size: jnp.ndarray,
    grid_dims: tuple[int, int, int],
    alpha_lj: float = 0.34,
    order: int = 5,
    deconv_mode: DeconvMode = "euler",
) -> jnp.ndarray:
    """Evaluate dispersion SPME reciprocal energy ``-(1 / 2V) * sum_n (K_6 / b^2) |Q_hat|^2``."""
    orig_dtype = positions.dtype
    b_w = jnp.asarray(disp_charges, dtype=orig_dtype)
    box_w = jnp.asarray(box_size, dtype=orig_dtype)
    q_grid = spread_dispersion_charges(
        positions, b_w, atom_mask, box_w, grid_dims, order=order
    )
    q_hat = jnp.fft.rfftn(q_grid, s=grid_dims, axes=(-3, -2, -1))
    is_ortho = _is_ortho_box(positions, box_w)
    if is_ortho and box_w.ndim == 2:
        g6 = jax.vmap(
            lambda b: lj_pme_influence_function(
                grid_dims, b, alpha_lj, order=order, deconv_mode=deconv_mode
            )
        )(box_w).astype(q_grid.dtype)
    else:
        g6 = lj_pme_influence_function(
            grid_dims, box_w, alpha_lj, order=order, deconv_mode=deconv_mode
        ).astype(q_grid.dtype)
    theta = jnp.fft.irfftn(q_hat * g6, s=grid_dims, axes=(-3, -2, -1)).astype(q_grid.dtype)
    n_grid = jnp.asarray(float(grid_dims[0] * grid_dims[1] * grid_dims[2]), dtype=q_grid.dtype)
    return (0.5 * jnp.sum(q_grid * theta, axis=(-3, -2, -1)) * n_grid).astype(orig_dtype)


def lj_pme_self_energy(
    disp_charges: jnp.ndarray,
    atom_mask: jnp.ndarray | None = None,
    alpha_lj: float = 0.34,
) -> jnp.ndarray:
    """Positive dispersion Ewald self-energy ``+(alpha_lj^6 / 12) * sum_i b_i^2``."""
    b = jnp.asarray(disp_charges)
    if atom_mask is not None:
        b = b * jnp.asarray(atom_mask, dtype=b.dtype)
    coeff = jnp.asarray((float(alpha_lj) ** 6) / 12.0, dtype=b.dtype)
    return (coeff * jnp.sum(b * b, axis=-1)).astype(b.dtype)


def lj_pme_exclusion_energy(
    positions: jnp.ndarray,
    disp_charges: jnp.ndarray,
    excl_indices: jnp.ndarray,
    excl_mask: jnp.ndarray | None = None,
    *,
    alpha_lj: float = 0.34,
    box_size: jnp.ndarray | None = None,
    use_pbc: bool = False,
) -> jnp.ndarray:
    """Reciprocal dispersion subtraction ``+sum_{(i,j)} b_i b_j (1 - g_6(alpha_lj r)) / r^6``."""
    if excl_indices.shape[0] == 0:
        return jnp.asarray(0.0, dtype=positions.dtype)
    n = int(positions.shape[0])
    p1 = excl_indices[:, 0]
    p2 = excl_indices[:, 1]
    valid = (p1 >= 0) & (p1 < n) & (p2 >= 0) & (p2 < n)
    if excl_mask is not None:
        valid = valid & jnp.asarray(excl_mask, dtype=bool)
    p1_s = jnp.where(valid, p1, 0)
    p2_s = jnp.where(valid, p2, 0)

    dr = positions[p1_s] - positions[p2_s]
    if use_pbc and box_size is not None:
        dr = _minimum_image_displacement(dr, box_size)
    r2 = jnp.sum(dr * dr, axis=-1)
    r2_safe = jnp.where(valid, r2, jnp.ones_like(r2))
    l6 = ewald_p6_long_range_value(alpha_lj, r2_safe)
    b_arr = jnp.asarray(disp_charges, dtype=positions.dtype)
    u_sub = b_arr[p1_s] * b_arr[p2_s] * l6
    return jnp.sum(jnp.where(valid, u_sub, 0.0))


def lj_pme_direct_pair_energy(
    r_safe: jnp.ndarray,
    sig_ij: jnp.ndarray,
    eps_ij: jnp.ndarray,
    b_prod_ij: jnp.ndarray,
    alpha_lj: float | jnp.ndarray,
    cutoff: float,
    *,
    shift_at_cutoff: bool = True,
) -> jnp.ndarray:
    """Pairwise direct-space LJ-PME energy for ``r < cutoff``.

    Computes ``4 * eps_ij * ((sig_ij/r)^12 - (sig_ij/r)^6) + b_i b_j L_6(alpha_lj, r^2)``
    (minus its value at ``r = cutoff`` when ``shift_at_cutoff=True``, matching OpenMM
    ``NonbondedForce.LJPME`` exactly under both Lorentz-Berthelot and geometric mixing).
    """
    sr = sig_ij / r_safe
    sr6 = sr**6
    sr12 = sr6 * sr6
    u_lj_full = jnp.where(eps_ij > 0.0, 4.0 * eps_ij * (sr12 - sr6), 0.0)
    u_rec_sub = b_prod_ij * ewald_p6_long_range_value(alpha_lj, r_safe * r_safe)
    u_raw = u_lj_full + u_rec_sub
    if shift_at_cutoff and cutoff < float("inf"):
        rc = jnp.asarray(cutoff, dtype=r_safe.dtype)
        src = sig_ij / rc
        src6 = src**6
        src12 = src6 * src6
        u_lj_rc = jnp.where(eps_ij > 0.0, 4.0 * eps_ij * (src12 - src6), 0.0)
        u_rec_rc = b_prod_ij * ewald_p6_long_range_value(alpha_lj, rc * rc)
        return u_raw - (u_lj_rc + u_rec_rc)
    return u_raw


def lj_pme_direct_energy_neighbor_list(
    positions: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    *,
    alpha_lj: float,
    cutoff: float,
    box_size: jnp.ndarray | None = None,
    atom_mask: jnp.ndarray | None = None,
    excl_scale_vdw: jnp.ndarray | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    shift_at_cutoff: bool = True,
) -> jnp.ndarray:
    """Sparse ``O(N * K)`` neighbor-list evaluation of direct-space LJ-PME pair energy."""
    n = int(positions.shape[0])
    dtype = positions.dtype
    sig = jnp.asarray(sigmas, dtype=dtype)
    eps = jnp.asarray(epsilons, dtype=dtype)
    b_disp = dispersion_charges(sig, eps, atom_mask=atom_mask)

    i_idx = jnp.arange(n, dtype=jnp.int32)[:, None]
    nb_clean = jnp.where(neighbor_idx < 0, n, neighbor_idx & ((1 << 29) - 1))
    mask_pair = (nb_clean >= 0) & (nb_clean < n) & (i_idx < nb_clean)
    if atom_mask is not None:
        am = jnp.asarray(atom_mask, dtype=bool)
        j_clip = jnp.clip(nb_clean, 0, max(n - 1, 0))
        mask_pair = mask_pair & am[:, None] & am[j_clip]
    j_safe = jnp.where(mask_pair, nb_clean, 0)

    dr = positions[:, None, :] - positions[j_safe]
    if box_size is not None:
        dr = _minimum_image_displacement(dr, box_size)
    r2 = jnp.sum(dr * dr, axis=-1)
    within_cut = mask_pair & (r2 < (cutoff * cutoff)) if cutoff < float("inf") else mask_pair
    r_safe = jnp.sqrt(jnp.where(within_cut, r2, jnp.ones_like(r2)) + jnp.asarray(1e-12, dtype=dtype))

    sig_ij, eps_ij = _combine_lj_pair(
        sig[:, None], sig[j_safe], eps[:, None], eps[j_safe], lj_combining_rule
    )
    b_prod = b_disp[:, None] * b_disp[j_safe]
    u_pair = lj_pme_direct_pair_energy(
        r_safe,
        sig_ij,
        eps_ij,
        b_prod,
        alpha_lj,
        cutoff,
        shift_at_cutoff=shift_at_cutoff,
    )
    if excl_scale_vdw is not None:
        u_pair = u_pair * jnp.asarray(excl_scale_vdw, dtype=dtype)
    return jnp.sum(jnp.where(within_cut, u_pair, 0.0))


def lj_pme_total_dispersion_energy(  # noqa: PLR0913
    positions: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    box_size: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    *,
    cutoff: float = 9.0,
    ewald_error_tolerance: float = 5e-4,
    alpha_lj: float | None = None,
    grid_dims: tuple[int, int, int] | None = None,
    order: int = 5,
    atom_mask: jnp.ndarray | None = None,
    excl_indices: jnp.ndarray | None = None,
    excl_mask: jnp.ndarray | None = None,
    excl_scale_vdw: jnp.ndarray | None = None,
    lj_combining_rule: str = "lorentz_berthelot",
    exceptions_use_pbc: bool = False,
    shift_at_cutoff: bool = True,
) -> jnp.ndarray:
    """End-to-end ``O(N * K)`` LJ-PME energy (direct + reciprocal + self + exclusions)."""
    if alpha_lj is None or grid_dims is None:
        res_alpha, res_dims = resolve_ljpme_parameters(
            box_size,
            cutoff=cutoff,
            ewald_error_tolerance=ewald_error_tolerance,
            explicit_alpha=0.0 if alpha_lj is None else float(alpha_lj),
            explicit_grid_dims=(0, 0, 0) if grid_dims is None else grid_dims,
        )
        eff_alpha = float(alpha_lj) if alpha_lj is not None else res_alpha
        eff_dims = grid_dims if grid_dims is not None else res_dims
    else:
        eff_alpha = float(alpha_lj)
        eff_dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))

    dtype = positions.dtype
    sig = jnp.asarray(sigmas, dtype=dtype)
    eps = jnp.asarray(epsilons, dtype=dtype)
    b_disp = dispersion_charges(sig, eps, atom_mask=atom_mask)

    e_dir = lj_pme_direct_energy_neighbor_list(
        positions,
        sig,
        eps,
        neighbor_idx,
        alpha_lj=eff_alpha,
        cutoff=cutoff,
        box_size=box_size,
        atom_mask=atom_mask,
        excl_scale_vdw=excl_scale_vdw,
        lj_combining_rule=lj_combining_rule,
        shift_at_cutoff=shift_at_cutoff,
    )
    e_rec = lj_pme_reciprocal_energy(
        positions,
        b_disp,
        atom_mask,
        box_size,
        eff_dims,
        alpha_lj=eff_alpha,
        order=order,
    )
    e_self = lj_pme_self_energy(b_disp, atom_mask=atom_mask, alpha_lj=eff_alpha)
    e_excl = (
        lj_pme_exclusion_energy(
            positions,
            b_disp,
            excl_indices,
            excl_mask,
            alpha_lj=eff_alpha,
            box_size=box_size,
            use_pbc=exceptions_use_pbc,
        )
        if excl_indices is not None
        else jnp.asarray(0.0, dtype=dtype)
    )
    return e_dir + e_rec + e_self + e_excl
