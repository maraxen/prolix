"""Direct validation of _ou_noise_one_water_rigid noise sampling.

Tests that the rigid-body OU noise function produces the correct equipartition
distribution without relying on full MD simulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import settle
from prolix.simulate import BOLTZMANN_KCAL

# XA-CI: heavy parity/compile — deselect from GitHub-faithful suite.
pytestmark = pytest.mark.slow



def _setup_tip3p_water():
  """Create a single TIP3P water at equilibrium geometry."""
  r_OH = 0.9572
  theta = np.radians(104.52 / 2)
  r_stack = jnp.array(
    [
      [0.0, 0.0, 0.0],
      [r_OH * np.sin(theta), r_OH * np.cos(theta), 0.0],
      [-r_OH * np.sin(theta), r_OH * np.cos(theta), 0.0],
    ],
    dtype=jnp.float64,
  )
  m_stack = jnp.array([15.999, 1.008, 1.008], dtype=jnp.float64)
  return r_stack, m_stack


def _compute_inertia_tensor(r_rel: jnp.ndarray, m_stack: jnp.ndarray) -> jnp.ndarray:
  """Compute inertia tensor from COM-relative positions and masses."""
  eye3 = jnp.eye(3, dtype=r_rel.dtype)
  r_sq = jnp.einsum('ia,ia->i', r_rel, r_rel)
  I = jnp.einsum('i,i->', m_stack, r_sq) * eye3 \
      - jnp.einsum('i,ia,ib->ab', m_stack, r_rel, r_rel)
  return I


def _compute_T_rot(p_noise: jnp.ndarray, r_stack: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  """Compute rotational temperature from noise momentum."""
  msum = jnp.sum(m_stack)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum
  r_rel = r_stack - com

  L = jnp.sum(jnp.cross(r_rel, p_noise), axis=0)
  I = _compute_inertia_tensor(r_rel, m_stack)

  I_inv_L = jnp.linalg.solve(I, L)
  L_T_I_inv_L = jnp.dot(L, I_inv_L)

  T_rot = L_T_I_inv_L / (3.0 * BOLTZMANN_KCAL)
  return float(T_rot)


def _compute_T_trans(p_noise: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  """Compute translational temperature from noise momentum."""
  msum = jnp.sum(m_stack)
  p_com = jnp.sum(p_noise, axis=0)
  p_com_sq = jnp.dot(p_com, p_com)
  T_trans = p_com_sq / (3.0 * msum * BOLTZMANN_KCAL)
  return float(T_trans)


def test_ou_noise_one_water_rigid_samples_correct_T_rot():
  """Verify that _ou_noise_one_water_rigid produces correct T_rot from sampling."""
  jax.config.update("jax_enable_x64", True)
  r_stack, m_stack = _setup_tip3p_water()

  T_target = 300.0
  kT = T_target * BOLTZMANN_KCAL

  # Sample many times
  n_samples = 10000
  T_rots = []
  key = jax.random.PRNGKey(42)
  for i in range(n_samples):
    key, subkey = jax.random.split(key)
    noise_p, _ = settle._ou_noise_one_water_rigid(subkey, r_stack, m_stack, kT)
    T_rot = _compute_T_rot(noise_p, r_stack, m_stack)
    T_rots.append(T_rot)

  T_rots = np.array(T_rots)
  mean_T_rot = float(np.mean(T_rots))

  # Should be very close to target (within ±5%)
  assert abs(mean_T_rot - T_target) < 0.05 * T_target, \
    f"T_rot mean {mean_T_rot:.1f} K deviates >5% from target {T_target} K"


def test_ou_noise_one_water_rigid_equipartition():
  """Verify that rotational and translational degrees of freedom are equipartitioned."""
  jax.config.update("jax_enable_x64", True)
  r_stack, m_stack = _setup_tip3p_water()

  T_target = 300.0
  kT = T_target * BOLTZMANN_KCAL

  # Sample many times
  n_samples = 5000
  T_rots = []
  T_transs = []
  key = jax.random.PRNGKey(99)
  for i in range(n_samples):
    key, subkey = jax.random.split(key)
    noise_p, _ = settle._ou_noise_one_water_rigid(subkey, r_stack, m_stack, kT)
    T_rot = _compute_T_rot(noise_p, r_stack, m_stack)
    T_trans = _compute_T_trans(noise_p, m_stack)
    T_rots.append(T_rot)
    T_transs.append(T_trans)

  T_rots = np.array(T_rots)
  T_transs = np.array(T_transs)

  # Ratio should be 1.0 for equipartition
  ratio = np.mean(T_rots) / np.mean(T_transs)
  assert abs(ratio - 1.0) < 0.10, \
    f"T_rot / T_trans = {ratio:.4f}, expected 1.0 (equipartition)"
