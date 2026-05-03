"""Shared periodic explicit-solvent corrections (PME exclusions, LJ dispersion tail).

Used by both `physics.system.make_energy_fn` and FlashMD (`flash_explicit`) so
Stack A and Stack B stay physically aligned.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_md import space
from jax_md.util import Array
from proxide.physics.constants import COULOMB_CONSTANT


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

    Formula based on heterogenous sum-of-sqrt approximation:
    E_tail = (4 * pi / V) * [ C12_sum / (9 * r_c^9) - C6_sum / (3 * r_c^3) ]
    where C12_sum = (sum_i sqrt(eps_i) * sig_i^6)^2
          C6_sum  = (sum_i sqrt(eps_i) * sig_i^3)^2
    """
    box = jnp.asarray(box)
    sigma = jnp.asarray(sigma)
    epsilon = jnp.asarray(epsilon)
    atom_mask = jnp.asarray(atom_mask).astype(jnp.float32)
    volume = box[0] * box[1] * box[2] if box.ndim == 1 else jnp.linalg.det(box)

    sum_c6 = jnp.sum(jnp.sqrt(epsilon) * (sigma**3) * atom_mask)
    sum_c12 = jnp.sum(jnp.sqrt(epsilon) * (sigma**6) * atom_mask)

    inv_vol = 1.0 / volume
    e_c12 = (8.0 * jnp.pi / 9.0) * (sum_c12**2) * inv_vol * (cutoff**-9)
    e_c6 = (8.0 * jnp.pi / 3.0) * (sum_c6**2) * inv_vol * (cutoff**-3)
    return e_c12 - e_c6


def lj_dispersion_tail_pressure(
    box: Array,
    sigma: Array,
    epsilon: Array,
    cutoff: float,
    atom_mask: Array,
) -> Array:
    """Isotropic long-range Lennard-Jones dispersion pressure correction.

    P_tail = E_tail / V.
    """
    volume = box[0] * box[1] * box[2] if box.ndim == 1 else jnp.linalg.det(box)
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
    atom_mask = jnp.asarray(atom_mask).astype(jnp.float32)
    volume = box[0] * box[1] * box[2] if box.ndim == 1 else jnp.linalg.det(box)

    sum_c6 = jnp.sum(jnp.sqrt(epsilon) * (sigma**3) * atom_mask)
    sum_c12 = jnp.sum(jnp.sqrt(epsilon) * (sigma**6) * atom_mask)

    # u(rc) = 4 * eps * ((sig/rc)^12 - (sig/rc)^6)
    # Impulsive term: -(2/3) * pi * rho^2 * rc^3 * u(rc)
    # rho = N / V
    # For mixture: -(2/3) * pi * (1/V^2) * rc^3 * sum_{i,j} u_{ij}(rc)
    # sum_{i,j} u_{ij}(rc) = 4 * [ (sum_c12^2) * rc^-12 - (sum_c6^2) * rc^-6 ]
    
    inv_vol2 = 1.0 / (volume**2)
    sum_u = 4.0 * ( (sum_c12**2) * (cutoff**-12) - (sum_c6**2) * (cutoff**-6) )
    
    return - (2.0 * jnp.pi / 3.0) * inv_vol2 * (cutoff**3) * sum_u
