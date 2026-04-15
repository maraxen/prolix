"""Tests for combined SETTLE (water) and RATTLE (solute) constraints."""

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax_md import space

from prolix.physics import settle

def test_settle_rattle_combined_integrator():
  """Verifies that both SETTLE and RATTLE constraints are obeyed simultaneously."""
  
  # 5 atoms: 0,1,2 = Water (O, H1, H2), 3,4 = Solute (A, B)
  R_init = jnp.array([
    [0.0, 0.0, 0.0],      # 0: O
    [0.9572, 0.0, 0.0],   # 1: H1
    [-0.2399, 0.9266, 0.0], # 2: H2 (approx TIP3P)
    [5.0, 0.0, 0.0],      # 3: A
    [6.5, 0.0, 0.0],      # 4: B (Length = 1.5)
  ])
  
  water_indices = jnp.array([[0, 1, 2]])
  solute_pairs = jnp.array([[3, 4]])
  solute_lengths = jnp.array([1.5])
  
  masses = jnp.array([16.0, 1.0, 1.0, 12.0, 12.0])
  
  # Simple harmonic potential that pulls everything towards origin
  # to force constraints to work against a force
  def energy_fn(R):
    return 0.1 * jnp.sum(R**2)
    
  dt = 0.002 # 2 fs
  kT = 0.5   # Some thermal motion
  gamma = 1.0
  
  init_fn, apply_fn = settle.settle_langevin(
    energy_fn,
    shift_fn=space.free()[1],
    dt=dt,
    kT=kT,
    gamma=gamma,
    mass=masses,
    water_indices=water_indices,
    constraints=(solute_pairs, solute_lengths)
  )
  
  key = random.PRNGKey(42)
  state = init_fn(key, R_init)
  
  # Run 500 steps
  apply_fn_jit = jax.jit(apply_fn)
  for _ in range(500):
    state = apply_fn_jit(state)
    
  R_final = state.position
  
  # 1. Check water geometry
  r_OH1 = jnp.linalg.norm(R_final[1] - R_final[0])
  r_OH2 = jnp.linalg.norm(R_final[2] - R_final[0])
  r_HH = jnp.linalg.norm(R_final[2] - R_final[1])
  
  assert jnp.abs(r_OH1 - settle.TIP3P_ROH) < 1e-4
  assert jnp.abs(r_OH2 - settle.TIP3P_ROH) < 1e-4
  assert jnp.abs(r_HH - settle.TIP3P_RHH) < 1e-4
  
  # 2. Check solute bond length
  r_AB = jnp.linalg.norm(R_final[4] - R_final[3])
  assert jnp.abs(r_AB - 1.5) < 1e-4
  
  # 3. Check velocities are orthogonal (projected)
  # (Strictly speaking, BAOAB constraints happen at end of step)
  # We check the momentum in the state.
  V = state.momentum / masses[:, None]
  
  # Water velocities orthogonal to bonds
  v_OH1 = V[1] - V[0]
  proj_OH1 = jnp.dot(v_OH1, (R_final[1] - R_final[0]) / r_OH1)
  assert jnp.abs(proj_OH1) < 0.05
  
  # Solute velocities orthogonal to bond
  v_AB = V[4] - V[3]
  proj_AB = jnp.dot(v_AB, (R_final[4] - R_final[3]) / r_AB)
  assert jnp.abs(proj_AB) < 0.05

if __name__ == "__main__":
  pytest.main([__file__, "-v"])
