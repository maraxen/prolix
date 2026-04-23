"""Rigid-body kinetic energy for TIP3P-style triatomic waters (O + 2 H per molecule)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array


def rigid_tip3p_box_ke_kcal(
  position: Array,
  momentum: Array,
  mass: Array,
  n_waters: int,
) -> Array:
  """Total kinetic energy (kcal/mol) as sum of per-molecule COM + rotational rigid-body KE.

  Uses laboratory-frame positions ``r_i`` and conjugate momenta ``p_i`` with masses ``m_i``
  grouped as ``(O, H, H)`` per water. This matches the physical kinetic energy counted by
  OpenMM's ``State.getKineticEnergy()`` for ``rigidWater=True`` more closely than
  ``sum_i |p_i|^2 / (2 m_i)``, which still includes internal components that constraints remove.
  """
  m_flat = jnp.asarray(mass).reshape(-1)
  pos = jnp.asarray(position).reshape((n_waters, 3, 3))
  mom = jnp.asarray(momentum).reshape((n_waters, 3, 3))
  mw = m_flat.reshape((n_waters, 3))

  def one_water(r: Array, p: Array, m: Array) -> Array:
    m_sum = jnp.sum(m)
    eps = jnp.array(1e-12, dtype=m_sum.dtype)
    com = jnp.sum(m[:, None] * r, axis=0) / jnp.maximum(m_sum, eps)
    p_tot = jnp.sum(p, axis=0)
    ke_t = 0.5 * jnp.dot(p_tot, p_tot) / jnp.maximum(m_sum, eps)
    rc = r - com
    ang = jnp.cross(rc[0], p[0]) + jnp.cross(rc[1], p[1]) + jnp.cross(rc[2], p[2])
    eye = jnp.eye(3, dtype=r.dtype)
    inertia = (
      m[0] * (jnp.sum(rc[0] * rc[0]) * eye - jnp.outer(rc[0], rc[0]))
      + m[1] * (jnp.sum(rc[1] * rc[1]) * eye - jnp.outer(rc[1], rc[1]))
      + m[2] * (jnp.sum(rc[2] * rc[2]) * eye - jnp.outer(rc[2], rc[2]))
    )
    omega = jnp.linalg.solve(inertia, ang)
    ke_r = 0.5 * jnp.dot(ang, omega)
    return ke_t + ke_r

  return jax.vmap(one_water)(pos, mom, mw).sum()
