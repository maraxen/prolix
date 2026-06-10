#!/usr/bin/env python3
"""Direct test of OU noise sampling: measure T_rot and T_trans empirically.

This script samples directly from _ou_noise_one_water_rigid and measures
the resulting rotational and translational temperatures to identify
whether the noise function produces the correct equipartition distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from prolix.physics import settle
from prolix.simulate import BOLTZMANN_KCAL


def compute_inertia_tensor(r_rel: jnp.ndarray, m_stack: jnp.ndarray) -> jnp.ndarray:
  """Compute inertia tensor from COM-relative positions and masses.

  Args:
    r_rel: (3, 3) COM-relative positions of O, H1, H2
    m_stack: (3,) masses of O, H1, H2

  Returns:
    I: (3, 3) inertia tensor
  """
  eye3 = jnp.eye(3, dtype=r_rel.dtype)
  r_sq = jnp.einsum('ia,ia->i', r_rel, r_rel)
  I = jnp.einsum('i,i->', m_stack, r_sq) * eye3 \
      - jnp.einsum('i,ia,ib->ab', m_stack, r_rel, r_rel)
  return I


def compute_T_rot(p_noise: jnp.ndarray, r_stack: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  """Compute rotational temperature from noise momentum.

  T_rot = (L^T I^{-1} L) / (3 * k_B) where L = sum_i (r_rel_i × p_i) is angular momentum

  Args:
    p_noise: (3, 3) mass-weighted noise momenta
    r_stack: (3, 3) absolute positions of O, H1, H2
    m_stack: (3,) masses

  Returns:
    T_rot: rotational temperature in Kelvin
  """
  msum = jnp.sum(m_stack)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum
  r_rel = r_stack - com

  # Angular momentum: L = sum_i (r_rel_i × p_i)
  L = jnp.sum(jnp.cross(r_rel, p_noise), axis=0)

  # Inertia tensor for equipartition
  I = compute_inertia_tensor(r_rel, m_stack)

  # Equipartition: KE_rot = (1/2) L^T I^{-1} L = (3/2) k_B T_rot
  # So T_rot = L^T I^{-1} L / (3 k_B)
  I_inv_L = jnp.linalg.solve(I, L)
  L_T_I_inv_L = jnp.dot(L, I_inv_L)

  T_rot = L_T_I_inv_L / (3.0 * BOLTZMANN_KCAL)
  return float(T_rot)


def compute_T_trans(p_noise: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  """Compute translational temperature from noise momentum.

  T_trans = p_com^2 / (3 * M * k_B)

  Args:
    p_noise: (3, 3) mass-weighted noise momenta
    m_stack: (3,) masses

  Returns:
    T_trans: translational temperature in Kelvin
  """
  msum = jnp.sum(m_stack)
  # COM momentum: p_com = sum_i p_i
  p_com = jnp.sum(p_noise, axis=0)
  p_com_sq = jnp.dot(p_com, p_com)

  T_trans = p_com_sq / (3.0 * msum * BOLTZMANN_KCAL)
  return float(T_trans)


def main():
  # TIP3P water at equilibrium geometry
  r_OH = 0.9572
  theta = np.radians(104.52 / 2)
  r_stack = jnp.array(
    [
      [0.0, 0.0, 0.0],  # O at origin
      [r_OH * np.sin(theta), r_OH * np.cos(theta), 0.0],  # H1
      [-r_OH * np.sin(theta), r_OH * np.cos(theta), 0.0],  # H2
    ],
    dtype=jnp.float64,
  )
  m_stack = jnp.array([15.999, 1.008, 1.008], dtype=jnp.float64)

  T_target = 300.0
  kT = T_target * BOLTZMANN_KCAL

  print(f"TIP3P water geometry:")
  print(f"  r_stack shape: {r_stack.shape}")
  print(f"  m_stack: {m_stack}")
  print(f"  T_target: {T_target} K")
  print(f"  kT: {kT} kcal/mol")
  print()

  # Sample many times
  n_samples = 50000
  T_rots = []
  T_transs = []

  key = jax.random.PRNGKey(42)
  for i in range(n_samples):
    key, subkey = jax.random.split(key)
    noise_p, _ = settle._ou_noise_one_water_rigid(subkey, r_stack, m_stack, kT)

    T_rot = compute_T_rot(noise_p, r_stack, m_stack)
    T_trans = compute_T_trans(noise_p, m_stack)

    T_rots.append(T_rot)
    T_transs.append(T_trans)

  T_rots = np.array(T_rots)
  T_transs = np.array(T_transs)

  print(f"Results from {n_samples} noise samples:")
  print()
  print(f"Rotational Temperature (T_rot):")
  print(f"  mean:     {np.mean(T_rots):8.2f} K")
  print(f"  std:      {np.std(T_rots):8.2f} K")
  print(f"  min:      {np.min(T_rots):8.2f} K")
  print(f"  max:      {np.max(T_rots):8.2f} K")
  print(f"  expected: {T_target:8.2f} K")
  print(f"  ratio (mean / expected): {np.mean(T_rots) / T_target:8.4f}")
  print()
  print(f"Translational Temperature (T_trans):")
  print(f"  mean:     {np.mean(T_transs):8.2f} K")
  print(f"  std:      {np.std(T_transs):8.2f} K")
  print(f"  min:      {np.min(T_transs):8.2f} K")
  print(f"  max:      {np.max(T_transs):8.2f} K")
  print(f"  expected: {T_target:8.2f} K")
  print(f"  ratio (mean / expected): {np.mean(T_transs) / T_target:8.4f}")
  print()

  # Compute ratio
  ratio_rot_trans = np.mean(T_rots) / np.mean(T_transs)
  print(f"Ratio T_rot / T_trans: {ratio_rot_trans:.4f}")
  print(f"  (Expected: 1.0 for equipartition)")
  print()

  # Interpret
  if np.mean(T_rots) < T_target * 0.95:
    print(f"⚠️  T_rot is UNDER-sampled by {100*(1 - np.mean(T_rots)/T_target):.1f}%")
    print(f"   This matches the observed deficit in BAOAB cycles.")
    print(f"   Scaling factor needed: sqrt({T_target / np.mean(T_rots):.4f}) = {np.sqrt(T_target / np.mean(T_rots)):.6f}")
  elif np.mean(T_rots) > T_target * 1.05:
    print(f"⚠️  T_rot is OVER-sampled by {100*(np.mean(T_rots)/T_target - 1):.1f}%")
  else:
    print(f"✓ T_rot is correctly sampled (within ±5%)")


if __name__ == "__main__":
  main()
