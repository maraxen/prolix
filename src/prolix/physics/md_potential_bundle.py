"""Bundled potential evaluation for JAX-MD-aligned dynamics.

Exposes a single ``jax.value_and_grad`` (or documented grad-only helpers) so
scalar energy and position gradients are obtained **together**, avoiding
duplicate AD work and keeping force semantics consistent with JAX-MD:
``jax_md.quantity.force`` uses **forces** ``F = -\\nabla E``.

For a SETTLE+Langevin step that writes these forces into
:class:`~prolix.physics.simulate.NVTLangevinState` at the same AD site, see
:mod:`prolix.physics.settle_langevin_potential_propagator`.

Design notes: :doc:`/.agent/docs/adr/ADR_001_kups_prolix_potential_eval` (historical filename).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax import ShapeDtypeStruct

@dataclass(frozen=True)
class EnergyForceBundle:
  """Scalar (or per-batch) energy and MD forces ``F = -\\nabla E`` at the same configuration."""

  energy: Array
  forces: Array


def value_energy_and_forces(
  energy_fn: Callable[..., Array],
  positions: Array,
  **kwargs: Any,
) -> EnergyForceBundle:
  """Return ``(E(R), -dE/dR)`` with one ``value_and_grad`` (same convention as JAX-MD)."""
  energy, dE_dR = jax.value_and_grad(energy_fn)(positions, **kwargs)
  return EnergyForceBundle(energy=energy, forces=-dE_dR)


def make_force_fn_like_canonicalize(
  energy_or_force_fn: Callable[..., Array],
  *,
  template_R: Array,
  template_kwargs: dict[str, Any] | None = None,
) -> Callable[..., Array]:
  """Return ``(R, **kw) -> forces`` matching ``jax_md.quantity.canonicalize_force``.

  When ``jax.eval_shape`` shows a scalar energy, uses :func:`value_energy_and_forces`
  (one reverse-mode AD, same as ``quantity.force``). Otherwise delegates to
  ``canonicalize_force`` for pre-shaped force callables or validated force-shaped
  energies.
  """
  from jax_md import quantity

  template_kwargs = template_kwargs or {}
  try:
    out_shaped = jax.eval_shape(energy_or_force_fn, template_R, **template_kwargs)
  except Exception:
    return quantity.canonicalize_force(energy_or_force_fn)

  if isinstance(out_shaped, ShapeDtypeStruct) and out_shaped.shape == ():

    def force_fn(R: Array, **kwargs: Any) -> Array:
      return value_energy_and_forces(energy_or_force_fn, R, **kwargs).forces

    return force_fn

  return quantity.canonicalize_force(energy_or_force_fn)


def value_energy_and_grad_energy(
  energy_fn: Callable[..., Array],
  positions: Array,
  **kwargs: Any,
) -> tuple[Array, Array]:
  """Return ``(E(R), dE/dR)`` with one ``value_and_grad``.

  Some integrators store **plus** gradient on state and apply ``p -= dt/2 * grad``;
  that matches this return pair (see ADR 001).
  """
  return jax.value_and_grad(energy_fn)(positions, **kwargs)


def eval_potential_bundle(
  energy_fn: Callable[..., Array],
  positions: Array,
  *,
  mode: Literal["value_and_grad", "grad_energy_only"] = "value_and_grad",
  **kwargs: Any,
) -> EnergyForceBundle | tuple[Array, Array]:
  """Dispatch potential evaluation by ``mode`` (extensible without breaking callers).

  Args:
    energy_fn: Maps ``positions`` (and ``**kwargs``) to a scalar total energy.
    positions: Atomic coordinates ``(N, 3)`` or batched ``(B, N, 3)``.
    mode: ``value_and_grad`` → :class:`EnergyForceBundle` with physical forces.
      ``grad_energy_only`` → ``(energy, dE/dR)`` for legacy minus-gradient updates.
    **kwargs: Passed through to ``energy_fn``.
  """
  if mode == "value_and_grad":
    return value_energy_and_forces(energy_fn, positions, **kwargs)
  if mode == "grad_energy_only":
    return value_energy_and_grad_energy(energy_fn, positions, **kwargs)
  raise ValueError(f"unknown eval_potential_bundle mode: {mode!r}")


def masked_mean_scalar(
  values: Array,
  mask: Array,
  *,
  axis: int | tuple[int, ...] | None = None,
) -> Array:
  """Mean of ``values`` over ``axis`` where ``mask`` is True (else ignored).

  ``mask`` must broadcast with ``values`` after reduction semantics consistent
  with ``jnp.where``: typically ``(N,)`` vs ``(N, ...)`` values with reduction
  over leading dims.
  """
  m = mask.astype(values.dtype)
  if axis is None:
    num = jnp.sum(values * m)
    den = jnp.sum(m) + jnp.asarray(1e-30, dtype=values.dtype)
    return num / den
  num = jnp.sum(values * m, axis=axis)
  den = jnp.sum(m, axis=axis) + jnp.asarray(1e-30, dtype=values.dtype)
  return num / den


def sum_real_atoms_per_batch(
  per_atom_quantity: Array,
  atom_mask: Array,
) -> Array:
  """Sum a per-atom quantity over real atoms (axis 0 unbatched, axis 1 batched).

  Args:
    per_atom_quantity: ``(N,)``, ``(N, K)``, ``(B, N)``, or ``(B, N, K)``.
    atom_mask: ``(N,)`` or ``(B, N)`` boolean mask.

  Returns:
    Scalar, ``(K,)``, ``(B,)``, or ``(B, K)`` depending on input rank.
  """
  m = atom_mask.astype(per_atom_quantity.dtype)
  if per_atom_quantity.ndim == 1:
    return jnp.sum(per_atom_quantity * m)
  if per_atom_quantity.ndim == 2 and m.ndim == 1:
    return jnp.sum(per_atom_quantity * m[:, None], axis=0)
  if per_atom_quantity.ndim == 2 and m.ndim == 2:
    return jnp.sum(per_atom_quantity * m, axis=1)
  if per_atom_quantity.ndim == 3 and m.ndim == 1:
    return jnp.sum(per_atom_quantity * m[None, :, None], axis=1)
  if per_atom_quantity.ndim == 3 and m.ndim == 2:
    return jnp.sum(per_atom_quantity * m[:, :, None], axis=1)
  raise ValueError(
    f"unsupported shapes: quantity {per_atom_quantity.shape}, mask {atom_mask.shape}"
  )
