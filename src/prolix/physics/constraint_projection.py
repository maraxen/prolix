"""Iterative SHAKE/RATTLE projection primitives (pure functions, no prolix imports).

Leaf module: ``settle`` and ``constraints`` both depend on it, so neither needs a lazy
import to dodge the former ``constraints`` <-> ``settle`` cycle (Phase 1.2).
``prolix.physics.constraints`` re-exports both names for backward compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["project_momenta", "project_positions"]


def project_positions(R, pairs, lengths, mass, shift_fn, constraint_mask=None, tol=1e-5, max_iter=20):
  """Iterative SHAKE projection for positions.

  Args:
      R: (N, 3) atom positions.
      pairs: (n_constraints, 2) constraint atom index pairs.
      lengths: (n_constraints,) target constraint lengths.
      mass: (N, 1) per-atom masses for inverse mass weighting.
      shift_fn: JAX-MD shift function for periodic boundary conditions.
      constraint_mask: Optional (n_constraints,) mask for selective constraint application.
      tol: Convergence tolerance (unused in iterative version).
      max_iter: Maximum SHAKE iterations.

  Returns:
      (N, 3) positions with constraints projected.
  """
  if constraint_mask is None:
    constraint_mask = jnp.ones(len(pairs), dtype=jnp.float32)
  else:
    constraint_mask = constraint_mask.astype(jnp.float32)

  def body_fn(i, R_curr):
    r1 = R_curr[pairs[:, 0]]
    r2 = R_curr[pairs[:, 1]]
    d_vec = r1 - r2  # Free space displacement
    d2 = jnp.sum(d_vec**2, axis=-1)
    diff = d2 - lengths**2

    inv_m1 = 1.0 / mass[pairs[:, 0], 0]
    inv_m2 = 1.0 / mass[pairs[:, 1], 0]
    w_sum = inv_m1 + inv_m2

    g = -diff / (2.0 * w_sum * d2 + 1e-8)
    g = g * constraint_mask

    delta = d_vec * g[:, None]
    d1 = delta * inv_m1[:, None]
    d2_corr = -delta * inv_m2[:, None]

    R_curr = R_curr.at[pairs[:, 0]].add(d1)
    R_curr = R_curr.at[pairs[:, 1]].add(d2_corr)
    return R_curr

  return jax.lax.fori_loop(0, max_iter, body_fn, R)


def project_momenta(P, R, pairs, mass, shift_fn, constraint_mask=None, tol=1e-6, max_iter=20):
  """Iterative RATTLE projection for momenta.

  Args:
      P: (N, 3) momenta (p = m * v).
      R: (N, 3) atom positions.
      pairs: (n_constraints, 2) constraint atom index pairs.
      mass: (N, 1) per-atom masses for inverse mass weighting.
      shift_fn: JAX-MD shift function for periodic boundary conditions.
      constraint_mask: Optional (n_constraints,) mask for selective constraint application.
      tol: Convergence tolerance (unused in iterative version).
      max_iter: Maximum RATTLE iterations.

  Returns:
      (N, 3) momenta projected orthogonal to constraint manifold.
  """
  if constraint_mask is None:
    constraint_mask = jnp.ones(len(pairs), dtype=jnp.float32)
  else:
    constraint_mask = constraint_mask.astype(jnp.float32)

  inv_m1 = 1.0 / mass[pairs[:, 0], 0]
  inv_m2 = 1.0 / mass[pairs[:, 1], 0]
  w_sum = inv_m1 + inv_m2

  r1 = R[pairs[:, 0]]
  r2 = R[pairs[:, 1]]
  r12 = r1 - r2

  def body_fn(i, P_curr):
    v1 = P_curr[pairs[:, 0]] * inv_m1[:, None]
    v2 = P_curr[pairs[:, 1]] * inv_m2[:, None]
    v12 = v1 - v2

    dot = jnp.sum(v12 * r12, axis=-1)
    d2 = jnp.sum(r12**2, axis=-1)
    k = -dot / (w_sum * d2 + 1e-8)
    k = k * constraint_mask

    impulse = r12 * k[:, None]
    P_curr = P_curr.at[pairs[:, 0]].add(impulse)
    P_curr = P_curr.at[pairs[:, 1]].add(-impulse)
    return P_curr

  return jax.lax.fori_loop(0, max_iter, body_fn, P)
