"""Tests for ``rigid_tip3p_box_ke_kcal`` (COM + rotational rigid-body KE per water)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax_md import quantity

from prolix.physics import settle
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal


def test_project_tip3p_momentum_rigid_is_identity_for_synthetic_rigid_motion() -> None:
  """``project_tip3p_waters_momentum_rigid`` leaves momenta unchanged if already rigid."""
  jax.config.update("jax_enable_x64", True)
  n_waters = 2
  n_atoms = n_waters * 3
  key = jax.random.PRNGKey(4)
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64).reshape(n_atoms, 1)
  m_o, m_h = jnp.float64(15.999), jnp.float64(1.008)
  r = jax.random.uniform(key, (n_atoms, 3), dtype=jnp.float64, minval=0.0, maxval=5.0)
  v_com = jnp.array([0.02, -0.03, 0.01], dtype=jnp.float64)
  omega = jnp.array([0.01, 0.02, -0.015], dtype=jnp.float64)
  idx = settle.get_water_indices(0, n_waters)

  def one_water_vel(rr: jnp.ndarray) -> jnp.ndarray:
    m_flat = jnp.array([m_o, m_h, m_h], dtype=jnp.float64)
    com = jnp.sum(m_flat[:, None] * rr, axis=0) / jnp.sum(m_flat)
    rrel = rr - com
    return v_com + jnp.cross(omega, rrel)

  v_list = []
  for w in range(n_waters):
    sl = slice(w * 3, w * 3 + 3)
    v_list.append(one_water_vel(r[sl]))
  v = jnp.vstack(v_list)
  p = mass.reshape(n_atoms, 1) * v
  p2 = settle.project_tip3p_waters_momentum_rigid(p, r, mass, idx)
  assert jnp.allclose(p2, p, rtol=1e-9, atol=1e-9)


def test_rigid_ke_matches_atomic_for_pure_translation_one_water() -> None:
  """Pure translation: all atoms share velocity ``v`` → rigid and atomic KE coincide."""
  n_waters = 1
  mass = jnp.array([[15.999], [1.008], [1.008]], dtype=jnp.float32)
  # Non-degenerate geometry (TIP3P-like triangle in xy-plane)
  r = jnp.array(
    [
      [0.0, 0.0, 0.0],
      [0.757, 0.0, 0.0],
      [0.239, 0.726, 0.0],
    ],
    dtype=jnp.float32,
  )
  v = jnp.array([0.4, -0.2, 0.1], dtype=jnp.float32)
  p = mass * v
  ke_a = float(quantity.kinetic_energy(momentum=p, mass=mass))
  ke_r = float(rigid_tip3p_box_ke_kcal(r, p, mass, n_waters))
  assert abs(ke_a - ke_r) < 1e-4 * max(abs(ke_a), 1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_atomic_ke_ge_rigid_ke_random_momenta(seed: int) -> None:
  """Atomic ``p^2/(2m)`` includes internal motion; rigid-body KE is a lower bound."""
  n_waters = 3
  n_atoms = n_waters * 3
  key = jax.random.PRNGKey(seed)
  k1, k2 = jax.random.split(key)
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float32).reshape(n_atoms, 1)
  p = jax.random.normal(k1, (n_atoms, 3), dtype=jnp.float32) * jnp.sqrt(mass)
  r = jax.random.normal(k2, (n_atoms, 3), dtype=jnp.float32) * 0.5
  ke_a = float(quantity.kinetic_energy(momentum=p, mass=mass))
  ke_r = float(rigid_tip3p_box_ke_kcal(r, p, mass, n_waters))
  assert ke_a + 1e-5 >= ke_r
