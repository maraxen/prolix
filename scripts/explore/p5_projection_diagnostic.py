#!/usr/bin/env python3
"""Test whether projection _project_one_water_momentum_rigid is lossy or noisy."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from prolix.physics import settle
from prolix.simulate import BOLTZMANN_KCAL


def compute_inertia_tensor(r_rel: jnp.ndarray, m_stack: jnp.ndarray) -> jnp.ndarray:
  eye3 = jnp.eye(3, dtype=r_rel.dtype)
  r_sq = jnp.einsum('ia,ia->i', r_rel, r_rel)
  I = jnp.einsum('i,i->', m_stack, r_sq) * eye3 \
      - jnp.einsum('i,ia,ib->ab', m_stack, r_rel, r_rel)
  return I


def compute_T_rot(p: jnp.ndarray, r_stack: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  msum = jnp.sum(m_stack)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum
  r_rel = r_stack - com

  L = jnp.sum(jnp.cross(r_rel, p), axis=0)
  I = compute_inertia_tensor(r_rel, m_stack)

  I_inv_L = jnp.linalg.solve(I, L)
  L_T_I_inv_L = jnp.dot(L, I_inv_L)

  T_rot = L_T_I_inv_L / (3.0 * BOLTZMANN_KCAL)
  return float(T_rot)


def main():
  # Setup: TIP3P water
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

  T_target = 300.0
  kT = T_target * BOLTZMANN_KCAL

  print("Projection noise diagnostic")
  print()

  # Test 1: project noise that is already rigid
  print("Test 1: Projecting pure rigid-body noise (should be lossy=false)")
  n_samples = 1000
  T_before = []
  T_after = []

  key = jax.random.PRNGKey(42)
  for i in range(n_samples):
    key, subkey = jax.random.split(key)
    p_rigid, _ = settle._ou_noise_one_water_rigid(subkey, r_stack, m_stack, kT)
    T_before.append(compute_T_rot(p_rigid, r_stack, m_stack))

    p_proj = settle._project_one_water_momentum_rigid(p_rigid, r_stack, m_stack)
    T_after.append(compute_T_rot(p_proj, r_stack, m_stack))

  T_before = np.array(T_before)
  T_after = np.array(T_after)

  print(f"  Before projection: mean T_rot = {np.mean(T_before):.2f} K, std = {np.std(T_before):.2f} K")
  print(f"  After projection:  mean T_rot = {np.mean(T_after):.2f} K, std = {np.std(T_after):.2f} K")
  print(f"  Mean loss: {np.mean(T_before) - np.mean(T_after):.4f} K")
  print()

  # Test 2: project random (non-rigid) noise
  print("Test 2: Projecting pure random noise (should be lossy=true)")
  T_before = []
  T_after = []

  key = jax.random.PRNGKey(99)
  for i in range(n_samples):
    key, subkey = jax.random.split(key)
    # Random momentum (NOT from rigid distribution)
    z = jax.random.normal(subkey, (3, 3), dtype=jnp.float64)
    p_random = m_stack[:, None] * z * jnp.sqrt(kT)
    T_before.append(compute_T_rot(p_random, r_stack, m_stack))

    p_proj = settle._project_one_water_momentum_rigid(p_random, r_stack, m_stack)
    T_after.append(compute_T_rot(p_proj, r_stack, m_stack))

  T_before = np.array(T_before)
  T_after = np.array(T_after)

  print(f"  Before projection: mean T_rot = {np.mean(T_before):.2f} K, std = {np.std(T_before):.2f} K")
  print(f"  After projection:  mean T_rot = {np.mean(T_after):.2f} K, std = {np.std(T_after):.2f} K")
  print(f"  Mean loss: {np.mean(T_before) - np.mean(T_after):.2f} K")
  print(f"  Loss fraction: {(np.mean(T_before) - np.mean(T_after)) / np.mean(T_before) * 100:.1f}%")
  print()


if __name__ == "__main__":
  main()
