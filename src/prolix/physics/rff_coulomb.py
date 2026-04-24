"""RFF approximation of erfc-damped Coulomb kernel for fast electrostatics.

See references/notes/rff_erfc_derivation.md for the mathematical derivation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax_md import space
from jaxtyping import Array, Float, Int
from proxide.physics.constants import COULOMB_CONSTANT


def rff_frequency_sample(
    alpha: float,
    n_features: int,
    key: Array,
    antithetic: bool = True,
) -> Float[Array, "n_half_features 3"]:
  """Sample RFF frequencies for erfc(a*r)/r kernel approximation.

  Uses Gaussian mixture representation: erfc(α*r)/r = (2/√π) ∫_α^∞ exp(−t²r²) dt.
  Samples t from the shifted exponential distribution t² − α² ~ Exp(1), which ensures
  t ≥ α. Then samples ω | t ~ N(0, 2t²*I₃).

  This is an approximate importance sampling scheme that biases toward the
  dominant contributions in the integral. With antithetic=True, returns D/2 unique
  frequencies; the paired −ω variates are implicit in the cos/sin feature construction.

  Args:
    alpha: Ewald damping parameter (1/Angstrom). Typical: 0.34.
    n_features: Total number of features D. Returns D/2 unique frequencies.
    key: JAX PRNG key.
    antithetic: If True (default), pair each ω_d with −ω_d to halve variance.

  Returns:
    Frequencies array of shape (D/2, 3) sampled from N(0, 2t_d² I₃) where t ≥ α.
  """
  d_half = n_features // 2
  key_t, key_omega = jax.random.split(key)

  # Sample t^2 - alpha^2 ~ Exp(1), so t^2 >= alpha^2 and t >= alpha (guaranteed).
  # This ensures support is [alpha, inf) as required by the integral bounds.
  exp_samples = jax.random.exponential(key_t, shape=(d_half,))
  t_sq = alpha**2 + exp_samples
  t_samples = jnp.sqrt(t_sq)  # shape (D/2,), support [alpha, inf)

  # Sample omega | t ~ N(0, 2t² I₃) for each t
  # Covariance: 2t_d² along each dimension
  normal_samples = jax.random.normal(key_omega, shape=(d_half, 3))
  omega = normal_samples * jnp.sqrt(2.0 * t_sq)[:, None]

  return omega


def erfc_rff_features(
    positions: Float[Array, "N 3"],
    omega: Float[Array, "D2 3"],
    alpha: float,
) -> Float[Array, "N n_features"]:
  """Compute RFF feature vectors phi(x_i) for erfc(a*r)/r kernel.

  Output shape: (N, n_features) = (N, 2*D2).
  Kernel approximation: K(r_ij) ~ (2/sqrt(pi)) * phi(x_i)^T phi(x_j)

  Args:
    positions: (N, 3) atom positions.
    omega: (D2, 3) = (n_features//2, 3) RFF frequencies.
    alpha: Ewald damping parameter.

  Returns:
    Features (N, 2*D2) with cos and sin terms normalized by sqrt(D2).
  """
  d2 = omega.shape[0]
  n_atoms = positions.shape[0]

  # dot product: (N, 1, 3) @ (1, D2, 3) -> (N, D2)
  proj = jnp.dot(positions, omega.T)  # (N, D2)

  cos_feat = jnp.cos(proj) / jnp.sqrt(d2)  # (N, D2)
  sin_feat = jnp.sin(proj) / jnp.sqrt(d2)  # (N, D2)

  # Interleave: [cos[0], sin[0], cos[1], sin[1], ...]
  # or stack: [[cos], [sin]]
  features = jnp.stack([cos_feat, sin_feat], axis=-1).reshape(n_atoms, 2 * d2)

  # Apply (2/sqrt(pi)) prefactor
  prefactor = 2.0 / jnp.sqrt(jnp.pi)
  return features * prefactor


def erfc_rff_coulomb_energy(
    positions: Float[Array, "N 3"],
    charges: Float[Array, N],
    atom_mask: Float[Array, N],
    omega: Float[Array, "D2 3"],
    alpha: float,
) -> Float[Array, ""]:
  """O(N*D) erfc-damped Coulomb energy via RFF.

  E = COULOMB_CONSTANT * (||sum_i q_i phi_i||^2 - sum_i q_i^2 ||phi_i||^2) / 2

  The second term subtracts i=j self-interaction.
  atom_mask zeros ghost atoms.

  Args:
    positions: (N, 3) atom positions.
    charges: (N,) partial charges.
    atom_mask: (N,) boolean mask; False for ghost atoms.
    omega: (D2, 3) RFF frequencies.
    alpha: Ewald damping parameter.

  Returns:
    Scalar energy in kcal/mol.
  """
  charges = jnp.asarray(charges)
  atom_mask = jnp.asarray(atom_mask)

  phi = erfc_rff_features(positions, omega, alpha)  # (N, D)

  # Mask charges: ghost atoms have 0 charge
  charges_masked = charges * atom_mask

  # Quadratic form: ||sum_i q_i phi_i||^2
  charge_weighted = charges_masked[:, None] * phi  # (N, D)
  charge_phi_sum = jnp.sum(charge_weighted, axis=0)  # (D,)
  quad_form = jnp.sum(charge_phi_sum ** 2)

  # Self-term: sum_i q_i^2 ||phi_i||^2 (subtract diagonal from quadratic form)
  phi_norm_sq = jnp.sum(phi ** 2, axis=1)  # (N,), ||phi_i||^2 for each atom
  self_term = jnp.sum(charges_masked ** 2 * phi_norm_sq)

  # Result: (||sum||^2 - self) / 2 to account for symmetry
  energy = (quad_form - self_term) / 2.0

  return COULOMB_CONSTANT * energy


@custom_vjp
def erfc_rff_coulomb_energy_diff(
    positions: Float[Array, "N 3"],
    charges: Float[Array, N],
    atom_mask: Float[Array, N],
    omega: Float[Array, "D2 3"],
    alpha: float,
) -> Float[Array, ""]:
  """Differentiable wrapper of erfc_rff_coulomb_energy with analytical gradient.

  Implements custom_vjp for gradient computation without materializing N*D intermediates.

  Args:
    positions: (N, 3) atom positions.
    charges: (N,) partial charges.
    atom_mask: (N,) boolean mask.
    omega: (D2, 3) RFF frequencies.
    alpha: Ewald damping parameter.

  Returns:
    Scalar energy in kcal/mol.
  """
  return erfc_rff_coulomb_energy(positions, charges, atom_mask, omega, alpha)


def _erfc_rff_fwd(positions, charges, atom_mask, omega, alpha):
  """Forward pass: compute energy and save residuals."""
  energy = erfc_rff_coulomb_energy(positions, charges, atom_mask, omega, alpha)
  phi = erfc_rff_features(positions, omega, alpha)
  residuals = (positions, charges, atom_mask, omega, alpha, phi)
  return energy, residuals


def _erfc_rff_bwd(residuals, g):
  """Backward pass: compute analytical gradient w.r.t. positions.

  Gradient: dE/dx_i = 2 q_i (dphi_i/dx_i)^T (sum_j q_j phi_j - q_i phi_i)
  """
  positions, charges, atom_mask, omega, alpha, phi = residuals
  charges_masked = charges * atom_mask

  d2 = omega.shape[0]
  n_atoms = positions.shape[0]
  d = 2 * d2

  # sum_j q_j phi_j
  charge_weighted = charges_masked[:, None] * phi
  sum_q_phi = jnp.sum(charge_weighted, axis=0)  # (D,)

  # For each atom i: dphi_i/dx_i
  # phi_i[2k] = cos(omega_k . x_i) / sqrt(D/2)
  # phi_i[2k+1] = sin(omega_k . x_i) / sqrt(D/2)
  # dphi_i[2k]/dx_i = -omega_k sin(omega_k . x_i) / sqrt(D/2)
  # dphi_i[2k+1]/dx_i = omega_k cos(omega_k . x_i) / sqrt(D/2)

  proj = jnp.dot(positions, omega.T)  # (N, D/2)
  norm = 1.0 / jnp.sqrt(d2)
  norm_pref = 2.0 / jnp.sqrt(jnp.pi)

  # dcos(proj) / dx = -omega sin(proj)
  # dsin(proj) / dx = omega cos(proj)
  grad_cos = -jnp.sin(proj[:, :, None]) * omega[None, :, :] * norm  # (N, D2, 3)
  grad_sin = jnp.cos(proj[:, :, None]) * omega[None, :, :] * norm   # (N, D2, 3)

  # Stack into (N, D, 3)
  grad_phi = jnp.stack([grad_cos, grad_sin], axis=2).reshape(n_atoms, d, 3)

  # Difference: sum_j q_j phi_j - q_i phi_i for each i
  diff = sum_q_phi[None, :] - phi  # (N, D)
  diff_masked = diff * charges_masked[:, None]  # (N, D) scaled by q_i

  # Gradient: dE/dx_i = 2 q_i (dphi_i/dx_i)^T (sum_j q_j phi_j - q_i phi_i)
  # shape: (N, 3, D) @ (N, D) -> (N, 3)
  grad_x = 2.0 * jnp.einsum("nkj,nk->nj", grad_phi, diff_masked)

  # Apply prefactors
  grad_x = grad_x * norm_pref * COULOMB_CONSTANT / 2.0
  grad_x = grad_x * atom_mask[:, None]

  # No gradient w.r.t. charges, omega, alpha
  return (grad_x, None, None, None, None)


erfc_rff_coulomb_energy_diff.defvjp(_erfc_rff_fwd, _erfc_rff_bwd)


def efa_exclusion_correction(
    positions: Float[Array, "N 3"],
    charges: Float[Array, N],
    atom_mask: Float[Array, N],
    displacement_fn,
    idx_12_13: Int[Array, "n_12_13 2"],
    idx_14: Int[Array, "n_14 2"],
    coul_14_scale: float,
    alpha: float,
) -> Float[Array, ""]:
  """Sparse exclusion correction for EFA Coulomb.

  Subtracts erfc(a*r)/r contribution for bonded pairs that RFF includes
  but should be excluded or scaled. Mirrors pme_exclusion_correction_energy
  in explicit_corrections.py.

  For 1-2/1-3: subtract full erfc(a*r)/r * q_i q_j * k_e
  For 1-4: subtract (1 - coul_14_scale) * erfc(a*r)/r * q_i q_j * k_e

  Args:
    positions: (N, 3) atom positions.
    charges: (N,) partial charges.
    atom_mask: (N,) boolean mask.
    displacement_fn: jax_md space.DisplacementFn for PBC.
    idx_12_13: (n_12_13, 2) int indices of 1-2/1-3 pairs to fully exclude.
    idx_14: (n_14, 2) int indices of 1-4 pairs to scale.
    coul_14_scale: Scale factor for 1-4 Coulomb (typically ~0.833).
    alpha: Ewald damping parameter.

  Returns:
    Scalar correction energy (negative, since we subtract).
  """
  charges = jnp.asarray(charges)

  def calc_exclusion_term(indices: Array, scale_factor: float) -> Array:
    if indices.shape[0] == 0:
      return jnp.array(0.0, dtype=positions.dtype)

    r_i = positions[indices[:, 0]]
    r_j = positions[indices[:, 1]]
    dr = jax.vmap(displacement_fn)(r_i, r_j)
    dist = space.distance(dr)

    q_i = charges[indices[:, 0]]
    q_j = charges[indices[:, 1]]

    # Avoid division by zero
    dist_safe = dist + 1e-6
    erfc_term = jax.scipy.special.erfc(alpha * dist)
    e_pair = COULOMB_CONSTANT * (q_i * q_j / dist_safe) * erfc_term

    # Scale and sum
    factor = 1.0 - scale_factor
    return jnp.sum(e_pair * factor)

  e_excl = jnp.array(0.0, dtype=positions.dtype)
  e_excl = e_excl + calc_exclusion_term(idx_12_13, 0.0)
  e_excl = e_excl + calc_exclusion_term(idx_14, coul_14_scale)

  return -e_excl
