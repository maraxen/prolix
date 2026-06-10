#!/usr/bin/env python3
"""Diagnostic of the full O-step (projection + decay + noise) to identify drain mechanism.

Tests whether the issue is in the noise function itself (already ruled out) or in
the projection/decay cycle in _langevin_step_o_constrained.
"""

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


def compute_T_rot(p_noise: jnp.ndarray, r_stack: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  msum = jnp.sum(m_stack)
  com = jnp.sum(m_stack[:, None] * r_stack, axis=0) / msum
  r_rel = r_stack - com

  L = jnp.sum(jnp.cross(r_rel, p_noise), axis=0)
  I = compute_inertia_tensor(r_rel, m_stack)

  I_inv_L = jnp.linalg.solve(I, L)
  L_T_I_inv_L = jnp.dot(L, I_inv_L)

  T_rot = L_T_I_inv_L / (3.0 * BOLTZMANN_KCAL)
  return float(T_rot)


def compute_T_trans(p: jnp.ndarray, m_stack: jnp.ndarray) -> float:
  msum = jnp.sum(m_stack)
  p_com = jnp.sum(p, axis=0)
  p_com_sq = jnp.dot(p_com, p_com)
  T_trans = p_com_sq / (3.0 * msum * BOLTZMANN_KCAL)
  return float(T_trans)


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
  gamma = 1.0  # ps^-1 in AKMA units
  dt = 0.5  # fs in AKMA units (very small)

  # O-step coefficients
  c1 = np.exp(-gamma * dt)
  c2 = np.sqrt(1 - c1**2)

  print(f"O-step diagnostics:")
  print(f"  T_target = {T_target} K")
  print(f"  gamma = {gamma} ps^-1")
  print(f"  dt = {dt} fs (AKMA)")
  print(f"  c1 = exp(-gamma*dt) = {c1:.6f}")
  print(f"  c2 = sqrt(1 - c1^2) = {c2:.6f}")
  print()

  # Test 1: Start with equilibrium rigid momentum
  n_samples = 10000
  T_rot_in = []
  T_rot_out = []
  T_rot_proj = []
  T_rot_noise = []

  key = jax.random.PRNGKey(42)
  for i in range(n_samples):
    key, k_init, k_o = jax.random.split(key, 3)

    # Sample initial momentum from equilibrium (should be at T_target)
    p_in, _ = settle._ou_noise_one_water_rigid(k_init, r_stack, m_stack, kT)
    T_in = compute_T_rot(p_in, r_stack, m_stack)
    T_rot_in.append(T_in)

    # Project to rigid subspace
    p_proj = settle._project_one_water_momentum_rigid(p_in, r_stack, m_stack)
    T_proj = compute_T_rot(p_proj, r_stack, m_stack)
    T_rot_proj.append(T_proj)

    # Apply O-step: p_c1 = c1 * p_proj
    p_c1 = c1 * p_proj

    # Sample OU noise (this is correct, as verified)
    noise, _ = settle._ou_noise_one_water_rigid(k_o, r_stack, m_stack, kT)
    T_noise = compute_T_rot(noise, r_stack, m_stack)
    T_rot_noise.append(T_noise)

    # Combine: p_out = p_c1 + c2 * noise
    p_out = p_c1 + c2 * noise

    T_out = compute_T_rot(p_out, r_stack, m_stack)
    T_rot_out.append(T_out)

  T_rot_in = np.array(T_rot_in)
  T_rot_proj = np.array(T_rot_proj)
  T_rot_noise = np.array(T_rot_noise)
  T_rot_out = np.array(T_rot_out)

  print("Test 1: Starting from equilibrium momentum")
  print(f"Input momentum T_rot:        {np.mean(T_rot_in):.2f} K (expected ~300K)")
  print(f"After projection T_rot:      {np.mean(T_rot_proj):.2f} K")
  print(f"Projection loss:             {np.mean(T_rot_in) - np.mean(T_rot_proj):.2f} K")
  print()
  print(f"Noise alone T_rot:           {np.mean(T_rot_noise):.2f} K (expected ~300K)")
  print()
  print(f"After O-step T_rot:          {np.mean(T_rot_out):.2f} K")
  print(f"Expected (c1^2*T_proj + c2^2*T_noise):")
  expected_T = c1**2 * np.mean(T_rot_proj) + c2**2 * np.mean(T_rot_noise)
  print(f"  = {c1**2:.6f}*{np.mean(T_rot_proj):.2f} + {c2**2:.6f}*{np.mean(T_rot_noise):.2f}")
  print(f"  = {expected_T:.2f} K")
  print()
  print(f"Actual vs expected difference: {np.mean(T_rot_out) - expected_T:.2f} K")
  print()

  # Test 2: With many repeated O-steps (to accumulate effect)
  print("Test 2: Repeated O-steps starting from equilibrium")
  n_steps = 100
  p_accum = None
  T_accum = []

  key = jax.random.PRNGKey(99)
  for step in range(n_steps):
    key, k_init = jax.random.split(key)

    # Initialize if needed
    if step == 0:
      p_accum, _ = settle._ou_noise_one_water_rigid(k_init, r_stack, m_stack, kT)
      T_accum.append(compute_T_rot(p_accum, r_stack, m_stack))
    else:
      # Apply one O-step
      key, k_proj, k_noise = jax.random.split(key, 3)
      p_proj = settle._project_one_water_momentum_rigid(p_accum, r_stack, m_stack)
      noise, _ = settle._ou_noise_one_water_rigid(k_noise, r_stack, m_stack, kT)
      p_accum = c1 * p_proj + c2 * noise
      T_accum.append(compute_T_rot(p_accum, r_stack, m_stack))

  T_accum = np.array(T_accum)
  print(f"Step 0:   T_rot = {T_accum[0]:.2f} K")
  print(f"Step 50:  T_rot = {T_accum[50]:.2f} K")
  print(f"Step 100: T_rot = {T_accum[99]:.2f} K")
  print(f"Expected (steady-state): {T_target:.2f} K")
  print()


if __name__ == "__main__":
  main()
