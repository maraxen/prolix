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
  n_atoms_for_tail: int | Array,
) -> Array:
  """Isotropic long-range Lennard-Jones dispersion correction beyond the cutoff.

  Matches `compute_lj_tail_correction` in system.py. Uses mean sigma/epsilon
  and N^2 scaling; `n_atoms_for_tail` should be the real atom count for
  padded systems.
  """
  box = jnp.asarray(box)
  sigma = jnp.asarray(sigma)
  epsilon = jnp.asarray(epsilon)
  volume = box[0] * box[1] * box[2] if box.ndim == 1 else jnp.linalg.det(box)

  avg_sig = jnp.mean(sigma)
  avg_eps = jnp.mean(epsilon)
  n_sq = jnp.asarray(n_atoms_for_tail, dtype=jnp.float32) ** 2

  rc3 = cutoff**3
  rc9 = rc3**3
  sig3 = avg_sig**3
  sig6 = sig3**2
  sig9 = sig3**3

  term = (1.0 / 9.0) * (sig9 / rc9) - (1.0 / 3.0) * (sig3 / rc3)
  return (8.0 * jnp.pi * n_sq / volume) * avg_eps * sig6 * term
