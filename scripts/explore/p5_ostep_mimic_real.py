#!/usr/bin/env python3
"""Test O-step exactly as called in settle_langevin with water projection enabled.

Mimics the real code flow from _langevin_step_o_constrained to see if there's
an interaction with how the function uses water indices and shapes.
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
  # Setup: 2 waters (mimics actual test)
  n_waters = 2
  r_OH = 0.9572
  theta = np.radians(104.52 / 2)

  # Create positions for 2 waters, each at a different location
  r_w1 = jnp.array(
    [
      [0.0, 0.0, 0.0],
      [r_OH * np.sin(theta), r_OH * np.cos(theta), 0.0],
      [-r_OH * np.sin(theta), r_OH * np.cos(theta), 0.0],
    ],
    dtype=jnp.float64,
  )
  r_w2 = r_w1 + jnp.array([5.0, 0.0, 0.0], dtype=jnp.float64)

  # Stack: (2, 3, 3) – 2 waters, 3 atoms each, 3 coords
  r_stack_all = jnp.stack([r_w1, r_w2], axis=0)
  m_stack_all = jnp.array([[15.999, 1.008, 1.008], [15.999, 1.008, 1.008]], dtype=jnp.float64)

  # Full position/momentum arrays (6 atoms total)
  positions = jnp.concatenate([r_w1, r_w2], axis=0)  # (6, 3)
  m_flat = jnp.concatenate([m_stack_all[0], m_stack_all[1]], axis=0).reshape(-1, 1)  # (6, 1)

  T_target = 300.0
  kT = T_target * BOLTZMANN_KCAL
  gamma = 10.0  # Note: from job 15631127 diagnostic
  dt = 0.5
  c1 = np.exp(-gamma * dt)
  c2 = np.sqrt(1 - c1**2)

  # Water indices: atoms 0,1,2 form water 0; atoms 3,4,5 form water 1
  water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)

  print(f"Testing O-step with {n_waters} waters")
  print(f"c1 = {c1:.6f}, c2 = {c2:.6f}")
  print()

  # Run multiple steps
  n_steps = 100
  T_rots = []

  key = jax.random.PRNGKey(42)
  momentum = jnp.zeros((6, 3), dtype=jnp.float64)  # Start with zero momentum

  for step in range(n_steps):
    key, subkey = jax.random.split(key)

    # Initialize momentum if at step 0
    if step == 0:
      for w_idx in range(n_waters):
        key, k_init = jax.random.split(key)
        r_w = r_stack_all[w_idx]
        m_w = m_stack_all[w_idx]
        p_init, _ = settle._ou_noise_one_water_rigid(k_init, r_w, m_w, kT)
        momentum = momentum.at[water_indices[w_idx]].set(p_init)

    # Apply O-step using the real function
    momentum, key = settle._langevin_step_o_constrained(
      momentum, positions, m_flat, gamma, dt, kT, key, water_indices
    )

    # Measure T_rot for each water
    T_rots_step = []
    for w_idx in range(n_waters):
      p_w = momentum[water_indices[w_idx]]
      r_w = r_stack_all[w_idx]
      m_w = m_stack_all[w_idx]
      T_w = compute_T_rot(p_w, r_w, m_w)
      T_rots_step.append(T_w)

    T_rots.append(np.mean(T_rots_step))
    if step % 20 == 0 or step < 5:
      print(f"Step {step:3d}: T_rot = {T_rots[-1]:.2f} K")

  print()
  print(f"Equilibrium stats:")
  burn = 10
  T_eq = np.array(T_rots[burn:])
  print(f"  Mean (after burn):  {np.mean(T_eq):.2f} K")
  print(f"  Std:                {np.std(T_eq):.2f} K")
  print(f"  Expected:           {T_target:.2f} K")
  print(f"  Difference:         {np.mean(T_eq) - T_target:+.2f} K")


if __name__ == "__main__":
  main()
