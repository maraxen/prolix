"""Iterative SHAKE/RATTLE projection primitives (pure functions, no prolix imports).

Leaf module: ``settle`` and ``constraints`` both depend on it, so neither needs a lazy
import to dodge the former ``constraints`` <-> ``settle`` cycle (Phase 1.2).
``prolix.physics.constraints`` re-exports both names for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

Array = Any

__all__ = ["project_momenta", "project_positions", "unpack_constraints"]


def unpack_constraints(
  constraints: tuple | None,
) -> tuple[Array, Array, Array | None] | None:
  """Unpack a 2-tuple ``(pairs, lengths)`` or 3-tuple ``(pairs, lengths, mask)``."""
  if constraints is None:
    return None
  if len(constraints) == 2:
    pairs, lengths = constraints
    return pairs, lengths, None
  if len(constraints) == 3:
    pairs, lengths, mask = constraints
    return pairs, lengths, mask
  msg = f"Expected constraints tuple of length 2 or 3, got {len(constraints)}"
  raise ValueError(msg)


def project_positions(
  R: Array,
  pairs: Array | tuple,
  lengths: Array | None = None,
  mass: Array | float = 1.0,
  shift_fn: Callable | None = None,
  constraint_mask: Array | None = None,
  tol: float = 1e-5,
  max_iter: int = 20,
  displacement_fn: Callable[[Array, Array], Array] | None = None,
) -> Array:
  """Iterative SHAKE projection for positions.

  Args:
      R: (N, 3) atom positions.
      pairs: (n_constraints, 2) constraint atom index pairs, or a 2-/3-tuple
          ``(pairs, lengths[, constraint_mask])``.
      lengths: (n_constraints,) target constraint lengths.
      mass: (N, 1) or (N,) per-atom masses for inverse mass weighting.
      shift_fn: JAX-MD shift function for periodic boundary conditions.
      constraint_mask: Optional (n_constraints,) mask for selective constraint application.
      tol: Convergence tolerance (unused in iterative version).
      max_iter: Maximum SHAKE iterations.
      displacement_fn: Optional PBC minimum-image displacement ``(r_a, r_b) -> r_a - r_b``.

  Returns:
      (N, 3) positions with constraints projected.
  """
  del shift_fn
  if isinstance(pairs, tuple):
    unpacked = unpack_constraints(pairs)
    assert unpacked is not None
    pairs, unpacked_lengths, unpacked_mask = unpacked
    if lengths is None:
      lengths = unpacked_lengths
    if constraint_mask is None:
      constraint_mask = unpacked_mask

  pairs = jnp.asarray(pairs, dtype=jnp.int32)
  if pairs.shape[0] == 0:
    return R
  lengths = jnp.asarray(lengths, dtype=R.dtype)

  if constraint_mask is None:
    mask_bool = jnp.ones((pairs.shape[0],), dtype=jnp.bool_)
  else:
    mask_bool = jnp.asarray(constraint_mask) != 0

  mass_flat = jnp.asarray(mass, dtype=R.dtype).reshape(-1)
  m1 = mass_flat[pairs[:, 0]]
  m2 = mass_flat[pairs[:, 1]]
  inv_m1 = jnp.where(m1 > 0, 1.0 / jnp.where(m1 > 0, m1, 1.0), 0.0)
  inv_m2 = jnp.where(m2 > 0, 1.0 / jnp.where(m2 > 0, m2, 1.0), 0.0)
  w_sum = inv_m1 + inv_m2

  def body_fn(i, R_curr):
    del i
    r1 = R_curr[pairs[:, 0]]
    r2 = R_curr[pairs[:, 1]]
    if displacement_fn is not None:
      d_vec = jax.vmap(displacement_fn)(r1, r2)
    else:
      d_vec = r1 - r2  # Free space displacement
    d2 = jnp.sum(d_vec**2, axis=-1)
    diff = d2 - lengths**2

    d_curr = jnp.sqrt(jnp.maximum(d2, 1e-12))
    unconverged = jnp.abs(d_curr - lengths) > float(tol)
    denom = 2.0 * w_sum * d2
    valid = (denom > 1e-12) & mask_bool & unconverged
    safe_denom = jnp.where(valid, denom, 1.0)
    g = jnp.where(valid, -diff / safe_denom, 0.0)

    delta = jnp.where(mask_bool[:, None], d_vec * g[:, None], 0.0)
    d1 = jnp.where(mask_bool[:, None], delta * inv_m1[:, None], 0.0)
    d2_corr = jnp.where(mask_bool[:, None], -delta * inv_m2[:, None], 0.0)

    R_curr = R_curr.at[pairs[:, 0]].add(d1)
    R_curr = R_curr.at[pairs[:, 1]].add(d2_corr)
    return R_curr

  return jax.lax.fori_loop(0, max_iter, body_fn, R)


def project_momenta(
  P: Array,
  R: Array,
  pairs: Array | tuple,
  mass: Array | float = 1.0,
  shift_fn: Callable | None = None,
  constraint_mask: Array | None = None,
  tol: float = 1e-6,
  max_iter: int = 20,
  displacement_fn: Callable[[Array, Array], Array] | None = None,
) -> Array:
  """Iterative RATTLE projection for momenta.

  Args:
      P: (N, 3) momenta (p = m * v).
      R: (N, 3) atom positions.
      pairs: (n_constraints, 2) constraint atom index pairs, or a 2-/3-tuple
          ``(pairs, lengths[, constraint_mask])``.
      mass: (N, 1) or (N,) per-atom masses for inverse mass weighting.
      shift_fn: JAX-MD shift function for periodic boundary conditions.
      constraint_mask: Optional (n_constraints,) mask for selective constraint application.
      tol: Convergence tolerance (unused in iterative version).
      max_iter: Maximum RATTLE iterations.
      displacement_fn: Optional PBC minimum-image displacement ``(r_a, r_b) -> r_a - r_b``.

  Returns:
      (N, 3) momenta projected orthogonal to constraint manifold.
  """
  del shift_fn
  if isinstance(pairs, tuple):
    unpacked = unpack_constraints(pairs)
    assert unpacked is not None
    pairs, _, unpacked_mask = unpacked
    if constraint_mask is None:
      constraint_mask = unpacked_mask

  pairs = jnp.asarray(pairs, dtype=jnp.int32)
  if pairs.shape[0] == 0:
    return P

  if constraint_mask is None:
    mask_bool = jnp.ones((pairs.shape[0],), dtype=jnp.bool_)
  else:
    mask_bool = jnp.asarray(constraint_mask) != 0

  mass_flat = jnp.asarray(mass, dtype=P.dtype).reshape(-1)
  m1 = mass_flat[pairs[:, 0]]
  m2 = mass_flat[pairs[:, 1]]
  inv_m1 = jnp.where(m1 > 0, 1.0 / jnp.where(m1 > 0, m1, 1.0), 0.0)
  inv_m2 = jnp.where(m2 > 0, 1.0 / jnp.where(m2 > 0, m2, 1.0), 0.0)
  w_sum = inv_m1 + inv_m2

  r1 = R[pairs[:, 0]]
  r2 = R[pairs[:, 1]]
  if displacement_fn is not None:
    r12 = jax.vmap(displacement_fn)(r1, r2)
  else:
    r12 = r1 - r2

  d2 = jnp.sum(r12**2, axis=-1)
  denom = w_sum * d2
  valid = (denom > 1e-12) & mask_bool
  safe_denom = jnp.where(valid, denom, 1.0)

  def body_fn(i, P_curr):
    del i
    v1 = P_curr[pairs[:, 0]] * inv_m1[:, None]
    v2 = P_curr[pairs[:, 1]] * inv_m2[:, None]
    v12 = v1 - v2

    dot = jnp.sum(v12 * r12, axis=-1)
    r_norm = jnp.sqrt(jnp.maximum(d2, 1e-12))
    unconverged = (jnp.abs(dot) / r_norm) > float(tol)
    valid_step = valid & unconverged
    k = jnp.where(valid_step, -dot / safe_denom, 0.0)

    impulse = jnp.where(mask_bool[:, None], r12 * k[:, None], 0.0)
    P_curr = P_curr.at[pairs[:, 0]].add(impulse)
    P_curr = P_curr.at[pairs[:, 1]].add(-impulse)
    return P_curr

  return jax.lax.fori_loop(0, max_iter, body_fn, P)
