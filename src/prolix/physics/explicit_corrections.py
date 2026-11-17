"""Shared periodic explicit-solvent corrections (PME exclusions, LJ dispersion tail).

Used by both `physics.system.make_energy_fn` and FlashMD (`flash_explicit`) so
Stack A and Stack B stay physically aligned.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax_md import space
from jax_md.util import Array

from prolix.physics import pbc

# kcal*A/(mol*e^2) — exact OpenMM ONE_4PI_EPS0 * 10 / 4.184
COULOMB_CONSTANT = 332.06371329919216

_BINOM_6: tuple[float, ...] = tuple(float(math.comb(6, k)) for k in range(7))
_BINOM_12: tuple[float, ...] = tuple(float(math.comb(12, k)) for k in range(13))


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
    eps_pos = epsilon > 0
    sqrt_eps = (
        jnp.where(
            eps_pos,
            jnp.sqrt(jnp.where(eps_pos, epsilon, jnp.ones_like(epsilon))),
            jnp.zeros_like(epsilon),
        )
        * atom_mask
    )
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
    cutoff: float,
    atom_mask: Array,
) -> Array:
    """Isotropic long-range Lennard-Jones dispersion correction beyond the cutoff.

    Exact O(N) binomial evaluation under Lorentz-Berthelot combining rules
    ``eps_ij = sqrt(eps_i * eps_j)`` and ``sigma_ij = (sigma_i + sigma_j) / 2``:
    ``E_tail = (8 * pi / V) * [ C12_pair_sum / (9 * r_c^9) - C6_pair_sum / (3 * r_c^3) ]``.
    """
    box = jnp.asarray(box)
    sigma = jnp.asarray(sigma)
    epsilon = jnp.asarray(epsilon)
    # debt 832: dtype follows epsilon (the ambient float precision) rather than hardcoding float32.
    atom_mask = jnp.asarray(atom_mask).astype(epsilon.dtype)
    volume = pbc.box_volume(box)

    c6_pair_sum, c12_pair_sum = _lorentz_berthelot_pair_moments(
        sigma, epsilon, atom_mask
    )

    inv_vol = 1.0 / volume
    e_c12 = (8.0 * jnp.pi / 9.0) * c12_pair_sum * inv_vol * (cutoff**-9)
    e_c6 = (8.0 * jnp.pi / 3.0) * c6_pair_sum * inv_vol * (cutoff**-3)
    return e_c12 - e_c6


def lj_dispersion_tail_pressure(
    box: Array,
    sigma: Array,
    epsilon: Array,
    cutoff: float,
    atom_mask: Array,
) -> Array:
    """Isotropic long-range Lennard-Jones dispersion pressure correction.

    P_tail = -d(E_tail)/dV = E_tail / V.
    """
    volume = pbc.box_volume(box)
    e_tail = lj_dispersion_tail_energy(box, sigma, epsilon, cutoff, atom_mask)
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
