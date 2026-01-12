"""Tests for thermal noising functions."""

import chex
import jax
import jax.numpy as jnp
from proxide.physics.constants import BOLTZMANN_KCAL

from prolix.physics.noising import compute_thermal_sigma, thermal_noise_fn


class TestComputeThermalSigma:
  """Tests for compute_thermal_sigma."""

  def test_zero_temperature(self):
    """At T=0, sigma should be 0."""
    sigma = compute_thermal_sigma(0.0)
    chex.assert_trees_all_close(sigma, 0.0, atol=1e-7)

  def test_room_temperature(self):
    """At T=300K, sigma should be sqrt(0.5 * R * 300)."""
    sigma = compute_thermal_sigma(300.0)
    expected = jnp.sqrt(0.5 * BOLTZMANN_KCAL * 300.0)
    chex.assert_trees_all_close(sigma, expected, atol=1e-7)

  def test_negative_temperature_clamps_to_zero(self):
    """Negative temperature should produce sigma=0."""
    sigma = compute_thermal_sigma(-100.0)
    # Use jax.numpy.array to handle potential scalar conversion during comparison if needed,
    # but check expects like types.
    chex.assert_trees_all_close(sigma, 0.0, atol=1e-7)


class TestThermalNoiseFn:
  """Tests for thermal_noise_fn."""

  def test_output_shape_preserved(self):
    """Output shape should match input."""
    key = jax.random.PRNGKey(42)
    coords = jnp.ones((10, 37, 3))
    noised, new_key = thermal_noise_fn(key, coords, 300.0)
    chex.assert_shape(noised, (10, 37, 3))
    chex.assert_shape(new_key, (2,))

  def test_dtype_preserved(self):
    """Output dtype should match input."""
    key = jax.random.PRNGKey(42)
    coords = jnp.ones((10, 37, 3), dtype=jnp.float32)
    noised, _ = thermal_noise_fn(key, coords, 300.0)
    assert noised.dtype == jnp.float32

    # Test float64 if enabled or just different precision if not
    # coords64 = jnp.ones((10, 37, 3), dtype=jnp.float64) # Don't assume x64 enabled

  def test_zero_temperature_no_noise(self):
    """At T=0, output should equal input."""
    key = jax.random.PRNGKey(42)
    coords = jnp.ones((10, 37, 3))
    noised, _ = thermal_noise_fn(key, coords, 0.0)
    chex.assert_trees_all_close(noised, coords, atol=1e-7)

  def test_reproducibility(self):
    """Same key should produce same noise."""
    key = jax.random.PRNGKey(42)
    coords = jnp.ones((10, 37, 3))
    noised1, _ = thermal_noise_fn(key, coords, 300.0)
    noised2, _ = thermal_noise_fn(key, coords, 300.0)
    chex.assert_trees_all_close(noised1, noised2)

  def test_different_keys_different_noise(self):
    """Different keys should produce different noise."""
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)
    coords = jnp.ones((10, 37, 3))
    noised1, _ = thermal_noise_fn(key1, coords, 300.0)
    noised2, _ = thermal_noise_fn(key2, coords, 300.0)
    # Check that they are NOT close
    diff = jnp.abs(noised1 - noised2).sum()
    assert diff > 1e-4

  def test_noise_distribution_statistics(self):
    """Noise should have correct standard deviation."""
    key = jax.random.PRNGKey(42)
    # Use large sample for stats
    coords = jnp.zeros((1000, 37, 3))
    temperature = 300.0
    noised, _ = thermal_noise_fn(key, coords, temperature)

    # Noised coords should have mean ~0 and std ~sigma
    expected_sigma = jnp.sqrt(0.5 * BOLTZMANN_KCAL * temperature)
    measured_std = jnp.std(noised)

    # Allow 5% tolerance for sampling variance
    chex.assert_trees_all_close(measured_std, expected_sigma, rtol=0.05)

  def test_jit_compatibility(self):
    """Function should be jit-compilable."""
    # We don't need static_argnums as shapes/temp are traceable
    jitted_fn = jax.jit(thermal_noise_fn)
    key = jax.random.PRNGKey(42)
    coords = jnp.ones((10, 37, 3))

    # Should not raise compilation error
    noised, _ = jitted_fn(key, coords, 300.0)
    chex.assert_shape(noised, (10, 37, 3))
