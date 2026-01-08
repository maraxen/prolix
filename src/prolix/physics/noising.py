"""Thermal noising functions for coordinate perturbation.

Provides JAX-jit compatible noising functions following Boltzmann distribution.
These can be used as drop-in alternatives to Gaussian noising for physically-
accurate thermal fluctuation modeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from proxide.physics.constants import BOLTZMANN_KCAL

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray


@jax.jit
def compute_thermal_sigma(temperature: Float[Array, ""] | float) -> Float[Array, ""]:
  """Compute noise standard deviation from temperature.

  Calculates the standard deviation for Boltzmann-distributed thermal noise.

  Math:
  $$
  \\sigma = \\sqrt{\\frac{1}{2} k_B T}
  $$

  Where $k_B = 0.0019872$ kcal/(mol·K) is the Boltzmann constant.

  Args:
      temperature: Temperature in Kelvin.

  Returns:
      Standard deviation σ for thermal noise in Angstroms.

  Example:
      >>> sigma = compute_thermal_sigma(300.0)
      >>> float(sigma)  # ~0.546 Å at room temperature
      0.546...

  """
  kT = jnp.maximum(0.5 * BOLTZMANN_KCAL * jnp.asarray(temperature), 0.0)
  return jnp.sqrt(kT)


@jax.jit
def thermal_noise_fn(
  key: PRNGKeyArray,
  coordinates: Float[Array, "n_res n_atoms 3"],
  temperature: Float[Array, ""] | float,
) -> tuple[Float[Array, "n_res n_atoms 3"], PRNGKeyArray]:
  """Apply Boltzmann-distributed thermal noise to coordinates.

  Samples displacement from N(0, σ) where σ = sqrt(0.5 * k_B * T).
  This models thermal fluctuations at equilibrium, providing physically-
  accurate coordinate perturbation for data augmentation or sampling.

  Process:
  1. Compute σ from temperature using Boltzmann relation.
     - Shapes: temperature: () -> sigma: ()
  2. Sample Gaussian noise with shape matching coordinates.
     - Shapes: coordinates: (N, A, 3) -> noise: (N, A, 3)
  3. Scale noise by σ and add to coordinates.
     - Shapes: (N, A, 3) + () * (N, A, 3) -> (N, A, 3)

  Args:
      key: JAX PRNG key for reproducible random sampling.
      coordinates: Atomic coordinates, shape (n_res, n_atoms, 3).
          Supports atom37 (N,37,3), backbone (N,5,3), or CA-only (N,3).
      temperature: Temperature in Kelvin. At T=0, no noise is applied.

  Returns:
      Tuple of (noised coordinates, new PRNG key).
      The new key should be used for subsequent random operations.

  Example:
      >>> key = jax.random.PRNGKey(0)
      >>> coords = jnp.ones((100, 37, 3))
      >>> noised, key = thermal_noise_fn(key, coords, 300.0)
      >>> noised.shape
      (100, 37, 3)

  Note:
      This function is JAX-jit compatible with zero Python overhead.
      The temperature parameter can be a traced value for use in
      gradient-based optimization.

  """
  key, noise_key = jax.random.split(key)
  sigma = compute_thermal_sigma(temperature)

  noise = jax.random.normal(noise_key, coordinates.shape, dtype=coordinates.dtype)
  noised_coords = coordinates + sigma * noise

  return noised_coords.astype(coordinates.dtype), key
