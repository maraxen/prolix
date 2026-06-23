"""JIT-friendly NVT trajectory sampling for rigid TIP3P temperature observables."""

from __future__ import annotations

import functools

import jax

from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import BOLTZMANN_KCAL


def rigid_tip3p_dof(n_waters: int) -> float:
  """Degrees of freedom for rigid TIP3P waters (6 N_w - 3)."""
  return float(6 * n_waters - 3)


def scan_settle_rigid_temperatures(
    state,
    apply_fn,
    *,
    n_steps: int,
    burn: int,
    n_waters: int,
):
  r"""Advance ``apply_fn`` for ``n_steps`` inside ``jax.lax.scan``; return T(K) after burn.

  Host supplies concrete ``n_steps``, ``burn``, and ``n_waters`` (static at compile time).
  Wrap the returned callable in ``jax.jit`` once; do not drive steps from a Python ``for``.
  """
  dof_rigid = rigid_tip3p_dof(n_waters)

  def body(carry, _):
    carry = apply_fn(carry)
    ke_r = rigid_tip3p_box_ke_kcal(
        carry.positions, carry.momentum, carry.mass, n_waters
    )
    t_k = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
    return carry, t_k

  _, temps = jax.lax.scan(body, state, None, length=n_steps)
  return temps[burn:]


def make_jitted_temperature_scan(apply_fn, *, n_steps: int, burn: int, n_waters: int):
  """Return ``jax.jit`` scan over a fixed horizon (compile once per shape/length)."""
  fn = functools.partial(
      scan_settle_rigid_temperatures,
      apply_fn=apply_fn,
      n_steps=n_steps,
      burn=burn,
      n_waters=n_waters,
  )
  return jax.jit(fn)
