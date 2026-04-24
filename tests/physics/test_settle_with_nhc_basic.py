"""Basic functionality test for settle_with_nhc wrapper."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics import settle
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from jax_md import space


def test_settle_with_nhc_basic_instantiation():
  """Verify settle_with_nhc wrapper can be instantiated."""
  jax.config.update("jax_enable_x64", True)

  # Simple dummy energy function (constant for this test)
  def dummy_energy(positions):
    return jnp.array(0.0)

  # Use free space (no PBC) for simplicity
  _, shift_fn = space.free()

  dt_fs = 1.0
  dt_akma = dt_fs / float(AKMA_TIME_UNIT_FS)
  temperature_k = 300.0
  kT = temperature_k * BOLTZMANN_KCAL

  n_waters = 2
  n_atoms = n_waters * 3
  mass = jnp.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=jnp.float64)
  water_indices = settle.get_water_indices(0, n_waters)

  # Create settle_with_nhc integrator
  nhc_init, nhc_apply = settle.settle_with_nhc(
    dummy_energy,
    shift_fn,
    dt_akma,
    kT,
    mass=mass,
    water_indices=water_indices,
  )

  # Should be callable
  assert callable(nhc_init), "nhc_init should be callable"
  assert callable(nhc_apply), "nhc_apply should be callable"
  print("✓ settle_with_nhc instantiation successful")


def test_settle_with_nhc_basic_step():
  """Verify settle_with_nhc wrapper can run one step."""
  jax.config.update("jax_enable_x64", True)

  # Simple dummy energy function
  def dummy_energy(positions):
    return jnp.sum(positions**2)

  _, shift_fn = space.free()

  dt_fs = 1.0
  dt_akma = dt_fs / float(AKMA_TIME_UNIT_FS)
  temperature_k = 300.0
  kT = temperature_k * BOLTZMANN_KCAL

  n_waters = 2
  n_atoms = n_waters * 3
  mass = jnp.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=jnp.float64)
  water_indices = settle.get_water_indices(0, n_waters)

  nhc_init, nhc_apply = settle.settle_with_nhc(
    dummy_energy,
    shift_fn,
    dt_akma,
    kT,
    mass=mass,
    water_indices=water_indices,
  )

  # Create initial positions and velocities
  positions = jnp.array([
      [0.0, 0.0, 0.0],
      [0.957, 0.0, 0.0],
      [-0.239, 0.927, 0.0],
      [5.0, 5.0, 5.0],
      [5.957, 5.0, 5.0],
      [4.761, 5.927, 5.0],
  ], dtype=jnp.float64)

  # Initialize state
  state = nhc_init(jax.random.PRNGKey(602), positions, mass=mass)

  # Verify initial state is finite
  assert jnp.all(jnp.isfinite(state.position)), "Initial position has NaN"
  assert jnp.all(jnp.isfinite(state.momentum)), "Initial momentum has NaN"
  assert jnp.all(jnp.isfinite(state.force)), "Initial force has NaN"

  # Run one step
  state = nhc_apply(state)

  # Verify final state is finite
  assert jnp.all(jnp.isfinite(state.position)), "Step 1: NaN in position"
  assert jnp.all(jnp.isfinite(state.momentum)), "Step 1: NaN in momentum"
  assert jnp.all(jnp.isfinite(state.force)), "Step 1: NaN in force"

  print("✓ settle_with_nhc single step successful")


def test_settle_with_nhc_jit_compatible():
  """Verify settle_with_nhc apply function is JIT-compatible."""
  jax.config.update("jax_enable_x64", True)

  def dummy_energy(positions):
    return jnp.sum(positions**2)

  _, shift_fn = space.free()

  dt_fs = 1.0
  dt_akma = dt_fs / float(AKMA_TIME_UNIT_FS)
  temperature_k = 300.0
  kT = temperature_k * BOLTZMANN_KCAL

  n_waters = 2
  n_atoms = n_waters * 3
  mass = jnp.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008], dtype=jnp.float64)
  water_indices = settle.get_water_indices(0, n_waters)

  nhc_init, nhc_apply = settle.settle_with_nhc(
    dummy_energy,
    shift_fn,
    dt_akma,
    kT,
    mass=mass,
    water_indices=water_indices,
  )

  positions = jnp.array([
      [0.0, 0.0, 0.0],
      [0.957, 0.0, 0.0],
      [-0.239, 0.927, 0.0],
      [5.0, 5.0, 5.0],
      [5.957, 5.0, 5.0],
      [4.761, 5.927, 5.0],
  ], dtype=jnp.float64)

  state = nhc_init(jax.random.PRNGKey(602), positions, mass=mass)

  # JIT-compile the apply function
  nhc_apply_jit = jax.jit(nhc_apply)

  # Apply one step (should not raise)
  state = nhc_apply_jit(state)
  assert jnp.all(jnp.isfinite(state.position)), "NaN in position after JIT step"
  assert jnp.all(jnp.isfinite(state.momentum)), "NaN in momentum after JIT step"
  print("✓ JIT compilation successful")
