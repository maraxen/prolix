"""Tests for RFF erfc-damped Coulomb approximation."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from prolix.physics.rff_coulomb import (
    rff_frequency_sample, erfc_rff_features, erfc_rff_coulomb_energy,
    efa_exclusion_correction,
)
from prolix.physics.pbc import create_periodic_space
from proxide.physics.constants import COULOMB_CONSTANT
from jax.scipy.special import erfc


def _dense_erfc_coulomb(positions, charges, atom_mask, alpha):
  """Reference O(N²) erfc Coulomb energy."""
  N = positions.shape[0]
  E = 0.0
  for i in range(N):
    for j in range(i+1, N):
      if atom_mask[i] and atom_mask[j]:
        r = jnp.linalg.norm(positions[i] - positions[j])
        E += charges[i] * charges[j] * erfc(alpha * r) / r
  return COULOMB_CONSTANT * E


@pytest.mark.smoke
def test_rff_feature_shape():
  key = jax.random.PRNGKey(42)
  n_features = 64
  alpha = 0.34
  n_atoms = 16
  positions = jax.random.normal(key, (n_atoms, 3)) * 3.0
  omega = rff_frequency_sample(alpha, n_features, key)
  phi = erfc_rff_features(positions, omega, alpha)
  assert phi.shape == (n_atoms, n_features)


@pytest.mark.smoke
def test_rff_determinism():
  key = jax.random.PRNGKey(0)
  omega1 = rff_frequency_sample(0.34, 64, key)
  omega2 = rff_frequency_sample(0.34, 64, key)
  np.testing.assert_array_equal(omega1, omega2)


@pytest.mark.smoke
def test_rff_self_term_finite():
  """Overlapping atoms should not produce NaN/Inf."""
  key = jax.random.PRNGKey(7)
  positions = jnp.zeros((4, 3))
  charges = jnp.array([0.417, 0.417, -0.834, 0.417])
  atom_mask = jnp.ones(4)
  omega = rff_frequency_sample(0.34, 256, key)
  E = erfc_rff_coulomb_energy(positions, charges, atom_mask, omega, 0.34)
  assert jnp.isfinite(E)


@pytest.mark.slow
def test_rff_kernel_bias():
  """RFF energy mean should be unbiased vs dense erfc Coulomb."""
  N = 32
  D = 512
  alpha = 0.34
  M = 64

  base_key = jax.random.PRNGKey(1234)
  pos_key, keys = jax.random.split(base_key)
  positions = jax.random.normal(pos_key, (N, 3)) * 5.0
  charges = jax.random.normal(jax.random.PRNGKey(42), (N,)) * 0.5
  atom_mask = jnp.ones(N)

  E_dense = _dense_erfc_coulomb(positions, charges, atom_mask, alpha)

  seed_keys = jax.random.split(keys, M)
  E_rffs = []
  for k in seed_keys:
    omega = rff_frequency_sample(alpha, D, k)
    E_rffs.append(float(erfc_rff_coulomb_energy(positions, charges, atom_mask, omega, alpha)))

  E_rff_mean = np.mean(E_rffs)
  E_rff_std = np.std(E_rffs)
  E_rff_stderr = E_rff_std / np.sqrt(M)

  # Bias check (z-score)
  z_score = abs(E_rff_mean - float(E_dense)) / (E_rff_stderr + 1e-8)
  assert z_score < 3.0, f"RFF energy is biased: z={z_score:.2f}, mean={E_rff_mean:.3f}, dense={float(E_dense):.3f}"

  # Variance bound: 2/sqrt(D) theoretical
  rel_stderr = E_rff_stderr / (abs(float(E_dense)) + 1e-8)
  assert rel_stderr < 2.0 / np.sqrt(D), f"Variance too high: rel_stderr={rel_stderr:.4f}, bound={2/np.sqrt(D):.4f}"


@pytest.mark.slow
def test_rff_gradient_check():
  """Analytical gradient (custom_vjp) should match numerical gradient."""
  key = jax.random.PRNGKey(99)
  N = 8
  D = 128
  alpha = 0.34
  positions = jax.random.normal(key, (N, 3)) * 3.0
  charges = jnp.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417, -0.834, 0.417])
  atom_mask = jnp.ones(N)
  omega = rff_frequency_sample(alpha, D, key)

  fn = lambda pos: erfc_rff_coulomb_energy(pos, charges, atom_mask, omega, alpha)

  # Numerical gradient via finite differences
  eps = 1e-4
  grad_numerical = jnp.zeros_like(positions)
  for i in range(N):
    for d in range(3):
      pos_p = positions.at[i, d].add(eps)
      pos_m = positions.at[i, d].add(-eps)
      grad_numerical = grad_numerical.at[i, d].set(
          (fn(pos_p) - fn(pos_m)) / (2 * eps)
      )

  grad_analytical = jax.grad(fn)(positions)

  rel_err = jnp.linalg.norm(grad_analytical - grad_numerical) / (jnp.linalg.norm(grad_numerical) + 1e-8)
  assert rel_err < 1e-3, f"Gradient mismatch: relative L2 error = {rel_err:.4f}"


@pytest.mark.smoke
def test_exclusion_correction_basic():
  """Exclusion correction should zero out a single excluded pair's contribution."""
  displacement_fn, _ = create_periodic_space(jnp.array([100.0, 100.0, 100.0]))
  # 2-atom system: charge +1 and -1 at distance 3 Angstroms
  positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
  charges = jnp.array([1.0, -1.0])
  atom_mask = jnp.ones(2)
  alpha = 0.34
  idx_12_13 = jnp.array([[0, 1]])  # this pair is 1-2 bonded (fully excluded)
  idx_14 = jnp.zeros((0, 2), dtype=jnp.int32)  # no 1-4 pairs
  corr = efa_exclusion_correction(positions, charges, atom_mask, displacement_fn,
                                   idx_12_13, idx_14, 0.83333333, alpha)
  # efa_exclusion_correction returns -e_excl. For 1-2 pairs with scale=0.0:
  # e_pair = q1*q2 * erfc(alpha*r)/r * k_e = -1 * erfc(0.34*3)/3 * 332.0637
  # corr = -e_pair = 1 * erfc(0.34*3)/3 * 332.0637
  r = 3.0
  e_pair = -1.0 * float(erfc(alpha * r) / r) * 332.0637  # q1*q2 * term
  expected = -e_pair  # negated by exclusion function
  rel_err = abs(float(corr) - expected) / (abs(expected) + 1e-8)
  assert rel_err < 1e-3, f"Exclusion correction mismatch: got {float(corr):.6f}, expected {expected:.6f}, rel_err={rel_err:.4f}"
