"""Shared periodic explicit-solvent corrections (PME exclusions, LJ dispersion tail).

Used by both `physics.system.make_energy_fn` and FlashMD (`flash_explicit`) so
Stack A and Stack B stay physically aligned.

Portions of ``lj_switched_tail_integrals`` are adapted from NVIDIA ``cuPotential``
``plugins/openmm/cpp/openmmapi/src/lj_tail.hpp`` under the Apache License 2.0
(Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES; see root ``NOTICE``).
"""

from __future__ import annotations

import math

import jax
import jax.core
import jax.numpy as jnp
from jax_md import space
from jax_md.util import Array

from prolix.physics import pbc

# kcal*A/(mol*e^2) — exact OpenMM ONE_4PI_EPS0 * 10 / 4.184
COULOMB_CONSTANT = 332.06371329919216

_BINOM_6: tuple[float, ...] = tuple(float(math.comb(6, k)) for k in range(7))
_BINOM_12: tuple[float, ...] = tuple(float(math.comb(12, k)) for k in range(13))

# Exact positive series coefficients a_n = (n + 1) / ((2n + 3)(2n + 5)) for the
# switched r^-6 radial integral over z^2 <= 0.61 (0.61^85 < 6e-19 < eps_f64).
_SWITCH_I6_SERIES_COEFFS: tuple[float, ...] = tuple(
    (n + 1.0) / ((2.0 * n + 3.0) * (2.0 * n + 5.0)) for n in range(85)
)


def _masked_sqrt_epsilon(epsilon: Array, atom_mask: Array) -> Array:
    """AD-safe ``sqrt(epsilon) * atom_mask`` avoiding ``1 / (2 * sqrt(0))`` NaNs."""
    eps_pos = epsilon > 0
    return (
        jnp.where(
            eps_pos,
            jnp.sqrt(jnp.where(eps_pos, epsilon, jnp.ones_like(epsilon))),
            jnp.zeros_like(epsilon),
        )
        * atom_mask
    )


def _lorentz_berthelot_pair_moments(
    sigma: Array,
    epsilon: Array,
    atom_mask: Array,
) -> tuple[Array, Array]:
    """Exact O(N) pair sums of eps_ij * sigma_ij^6 and eps_ij * sigma_ij^12.

    Uses Lorentz-Berthelot combining rules ``eps_ij = sqrt(eps_i * eps_j)`` and
    ``sigma_ij = (sigma_i + sigma_j) / 2`` expanded via the 13-term binomial
    identity ``sum_{i,j} sqrt(eps_i eps_j) ((sig_i + sig_j)/2)^m =
    2^-m * sum_{k=0}^m comb(m, k) * S_k * S_{m-k}`` where
    ``S_k = sum_i sqrt(eps_i) * sig_i^k * mask_i``.
    """
    sqrt_eps = _masked_sqrt_epsilon(epsilon, atom_mask)
    moments = [jnp.sum(sqrt_eps * (sigma**k)) for k in range(13)]
    c6_pair_sum = jnp.zeros((), dtype=epsilon.dtype)
    for k, binom_k in enumerate(_BINOM_6):
        c6_pair_sum = c6_pair_sum + binom_k * moments[k] * moments[6 - k]
    c6_pair_sum = c6_pair_sum * (1.0 / 64.0)

    c12_pair_sum = jnp.zeros((), dtype=epsilon.dtype)
    for k, binom_k in enumerate(_BINOM_12):
        c12_pair_sum = c12_pair_sum + binom_k * moments[k] * moments[12 - k]
    c12_pair_sum = c12_pair_sum * (1.0 / 4096.0)
    return c6_pair_sum, c12_pair_sum


def _geometric_pair_moments(
    sigma: Array,
    epsilon: Array,
    atom_mask: Array,
) -> tuple[Array, Array]:
    """Exact O(N) pair sums under geometric combining rules ``sigma_ij = sqrt(sigma_i * sigma_j)``."""
    sqrt_eps = _masked_sqrt_epsilon(epsilon, atom_mask)
    m3 = jnp.sum(sqrt_eps * (sigma**3))
    m6 = jnp.sum(sqrt_eps * (sigma**6))
    return m3 * m3, m6 * m6


def _resolve_pair_moments(
    sigma: Array,
    epsilon: Array,
    atom_mask: Array,
    combining_rule: str,
) -> tuple[Array, Array]:
    rule = combining_rule.strip().lower().replace("-", "_")
    if rule == "lorentz_berthelot":
        return _lorentz_berthelot_pair_moments(sigma, epsilon, atom_mask)
    if rule == "geometric":
        return _geometric_pair_moments(sigma, epsilon, atom_mask)
    raise ValueError(
        f"Unsupported LJ combining_rule={combining_rule!r}; expected 'lorentz_berthelot' or 'geometric'"
    )


def lj_switched_tail_integrals(
    r_switch: float | Array,
    r_cutoff: float | Array,
) -> tuple[Array, Array]:
    """Radial integrals ``(sixth, twelfth)`` of omitted ``r^-6`` and ``r^-12`` LJ terms.

    Includes both the exterior tail ``[r_cutoff, inf)`` and the quintic potential-switch
    deficit on ``[r_switch, r_cutoff]``:
      ``I_p = int_{r_switch}^{r_cutoff} (1 - S(r)) r^{2 - p} dr + int_{r_cutoff}^{inf} r^{2 - p} dr``.

    Adapted from ``lj_tail.hpp`` (Apache-2.0, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES)
    with branch-sanitized ``jnp.where`` selection so ``jax.grad`` and ``prolix.export`` StableHLO
    lowering remain completely free of ``0/0`` NaNs and custom calls.
    """
    if not isinstance(r_cutoff, jax.core.Tracer) and not isinstance(r_switch, jax.core.Tracer):
        rc_host = float(jnp.asarray(r_cutoff))
        rs_host = float(jnp.asarray(r_switch))
        if not math.isfinite(rc_host) or rc_host <= 0.0:
            raise ValueError(f"LJ tail requires a positive finite r_cutoff, got {rc_host}")
        if not math.isfinite(rs_host) or rs_host <= 0.0 or rs_host > rc_host:
            raise ValueError(
                f"Switched LJ tail requires 0 < r_switch <= r_cutoff, got r_switch={rs_host}, r_cutoff={rc_host}"
            )

    rc_raw = jnp.asarray(r_cutoff)
    dtype = rc_raw.dtype if jnp.issubdtype(rc_raw.dtype, jnp.floating) else jnp.result_type(rc_raw, 1.0)
    rc = rc_raw.astype(dtype)
    rs = jnp.asarray(r_switch, dtype=dtype)

    sixth_unsw = 1.0 / (3.0 * (rc**3))
    twelfth_unsw = 1.0 / (9.0 * (rc**9))

    is_switched = rs < rc
    rs_sw = jnp.where(is_switched, jnp.maximum(rs, rc * 1e-12), rc * 0.5)
    t = rs_sw / rc
    twelfth_sw = (((5.0 * t + 9.0) * t + 9.0) * t + 5.0) / (252.0 * (rs_sw**6) * (rc**3))

    z = (rc - rs_sw) / (rc + rs_sw)
    z2 = z * z
    use_series = z2 <= 0.61

    # Positive rational series for narrow/moderate switch widths (z^2 <= 0.61).
    z2_ser = jnp.where(use_series, z2, jnp.asarray(0.25, dtype=dtype))
    coeffs = jnp.asarray(_SWITCH_I6_SERIES_COEFFS, dtype=dtype)
    powers = z2_ser[..., None] ** jnp.arange(len(_SWITCH_I6_SERIES_COEFFS), dtype=dtype)
    series_sum = jnp.sum(powers * coeffs, axis=-1)
    sixth_ser = 40.0 * series_sum / ((rc + rs_sw) ** 3)

    # Closed logarithmic form for wide switch intervals (z^2 > 0.61, t < ~0.123).
    t_log = jnp.where(use_series, jnp.asarray(0.1, dtype=dtype), t)
    sixth_log = (
        10.0
        * ((t_log * t_log + 4.0 * t_log + 1.0) * (-jnp.log(t_log)) - 3.0 * (1.0 - t_log * t_log))
        / ((rc**3) * ((1.0 - t_log) ** 5))
    )

    sixth_sw = jnp.where(use_series, sixth_ser, sixth_log)
    sixth = jnp.where(is_switched, sixth_sw, sixth_unsw)
    twelfth = jnp.where(is_switched, twelfth_sw, twelfth_unsw)
    return sixth, twelfth


def pme_exclusion_correction_energy(
  r: Array,
  displacement_fn: space.DisplacementFn,
  idx_12: Array,
  idx_13: Array,
  idx_14: Array,
  charges: Array,
  pme_alpha: float,
  coul_14_scale: float,
) -> Array:
  """Remove reciprocal-space double counting for excluded / scaled bonded pairs.

  Same convention as `compute_pme_exceptions` in system.py:
  E_corr = - sum_pairs (1 - scale) * q_i q_j * erf(alpha r) / r * COULOMB_CONSTANT

  Returns a scalar energy (kcal/mol) to **add** to the direct+reciprocal electrostatic total.
  """
  charges = jnp.asarray(charges)

  def calc_correction_term(indices: Array, scale_factor: float) -> Array:
    if indices.shape[0] == 0:
      return jnp.array(0.0, dtype=r.dtype)

    r_i = r[indices[:, 0]]
    r_j = r[indices[:, 1]]
    dr = jax.vmap(displacement_fn)(r_i, r_j)
    dist = space.distance(dr)

    q_i = charges[indices[:, 0]]
    q_j = charges[indices[:, 1]]

    dist_safe = dist + 1e-6
    erf_term = jax.scipy.special.erf(pme_alpha * dist)
    e_pair_recip = COULOMB_CONSTANT * (q_i * q_j / dist_safe) * erf_term
    factor = 1.0 - scale_factor
    return jnp.sum(e_pair_recip * factor)

  e_corr = jnp.array(0.0, dtype=r.dtype)
  e_corr = e_corr + calc_correction_term(idx_12, 0.0)
  e_corr = e_corr + calc_correction_term(idx_13, 0.0)
  e_corr = e_corr + calc_correction_term(idx_14, coul_14_scale)
  return -e_corr


def lj_dispersion_tail_energy(
    box: Array,
    sigma: Array,
    epsilon: Array,
    cutoff: float | Array,
    atom_mask: Array,
    *,
    lj_switch_width: float | Array = 0.0,
    combining_rule: str = "lorentz_berthelot",
) -> Array:
    """Isotropic long-range Lennard-Jones dispersion correction (exterior + quintic switch deficit).

    Supports both Lorentz-Berthelot (13-term binomial) and geometric (factored ``(M_3^2, M_6^2)``)
    combining rules in exact ``O(N)`` time:
      ``E_tail = (8 * pi / V) * (C12_pair_sum * I_12(r_s, r_c) - C6_pair_sum * I_6(r_s, r_c))``.
    """
    box = jnp.asarray(box)
    sigma = jnp.asarray(sigma)
    epsilon = jnp.asarray(epsilon)
    # debt 832: dtype follows epsilon (the ambient float precision) rather than hardcoding float32.
    atom_mask = jnp.asarray(atom_mask).astype(epsilon.dtype)
    volume = pbc.box_volume(box)

    c6_pair_sum, c12_pair_sum = _resolve_pair_moments(
        sigma, epsilon, atom_mask, combining_rule
    )

    if not isinstance(cutoff, jax.core.Tracer) and not isinstance(lj_switch_width, jax.core.Tracer):
        rc_f = float(cutoff)
        sw_f = float(lj_switch_width)
        r_switch_arg: float | Array = rc_f - sw_f if sw_f > 0.0 else rc_f
        i6, i12 = lj_switched_tail_integrals(
            jnp.asarray(r_switch_arg, dtype=epsilon.dtype)
            if sw_f < rc_f and rc_f > 0.0
            else r_switch_arg,
            jnp.asarray(rc_f, dtype=epsilon.dtype) if rc_f > 0.0 else rc_f,
        )
    else:
        rc_arr = jnp.asarray(cutoff, dtype=epsilon.dtype)
        sw_arr = jnp.asarray(lj_switch_width, dtype=epsilon.dtype)
        r_switch_arr = jnp.where(sw_arr > 0.0, rc_arr - sw_arr, rc_arr)
        i6, i12 = lj_switched_tail_integrals(r_switch_arr, rc_arr)

    inv_vol = 1.0 / volume
    return (8.0 * jnp.pi * inv_vol) * (c12_pair_sum * i12 - c6_pair_sum * i6)


def lj_dispersion_tail_pressure(
    box: Array,
    sigma: Array,
    epsilon: Array,
    cutoff: float | Array,
    atom_mask: Array,
    *,
    lj_switch_width: float | Array = 0.0,
    combining_rule: str = "lorentz_berthelot",
) -> Array:
    """Isotropic long-range Lennard-Jones dispersion pressure correction.

    P_tail = -d(E_tail)/dV = E_tail / V.
    """
    volume = pbc.box_volume(box)
    e_tail = lj_dispersion_tail_energy(
        box,
        sigma,
        epsilon,
        cutoff,
        atom_mask,
        lj_switch_width=lj_switch_width,
        combining_rule=combining_rule,
    )
    return e_tail / volume


def lj_dispersion_tail_impulsive_pressure(
    box: Array,
    sigma: Array,
    epsilon: Array,
    cutoff: float,
    atom_mask: Array,
) -> Array:
    """Isotropic impulsive pressure correction for unshifted potentials."""
    box = jnp.asarray(box)
    sigma = jnp.asarray(sigma)
    epsilon = jnp.asarray(epsilon)
    # debt 832: dtype follows epsilon (the ambient float precision) rather than hardcoding float32.
    atom_mask = jnp.asarray(atom_mask).astype(epsilon.dtype)
    volume = pbc.box_volume(box)

    c6_pair_sum, c12_pair_sum = _lorentz_berthelot_pair_moments(
        sigma, epsilon, atom_mask
    )

    inv_vol2 = 1.0 / (volume**2)
    sum_u = 4.0 * (c12_pair_sum * (cutoff**-12) - c6_pair_sum * (cutoff**-6))

    return (2.0 * jnp.pi / 3.0) * inv_vol2 * (cutoff**3) * sum_u
