"""JIT-friendly NVT trajectory sampling for rigid TIP3P temperature observables."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from prolix.physics.rigid_water_ke import rigid_tip3p_ke_components
from prolix.simulate import BOLTZMANN_KCAL


def rigid_tip3p_dof(n_waters: int) -> float:
  """Degrees of freedom for rigid TIP3P waters, excluding the system COM (6 N_w - 3)."""
  return float(6 * n_waters - 3)


def system_com_ke_kcal(momentum, mass):
  """Kinetic energy (kcal/mol) of the whole-system centre of mass, ``|sum p|^2 / 2M``."""
  m_tot = jnp.sum(jnp.asarray(mass))
  p_sys = jnp.sum(jnp.asarray(momentum).reshape(-1, 3), axis=0)
  return 0.5 * jnp.dot(p_sys, p_sys) / m_tot


def rigid_tip3p_temperatures(position, momentum, mass, n_waters: int) -> dict:
  r"""Translational / rotational / total temperatures (K) of a rigid-TIP3P box.

  The system-COM kinetic energy is subtracted before dividing by DOF counts that
  exclude it (``3N-3`` translational, ``6N-3`` total). This is valid whether or not
  the integrator removes COM momentum: if it does, ``ke_com`` is ~0; if it does not
  (``remove_linear_com_momentum=False``, the ``settle_langevin`` default) the COM is a
  live thermalized 3-DOF mode, and leaving it in inflates ``t_trans`` by
  ``3 T / (3N - 3)`` (+300 K at N=2, +20 K at N=16) and ``t_total`` by
  ``3 T / (6N - 3)``. ``t_com`` is reported separately on its 3 DOF.

  Rotational KE is the rigid-body ``L . omega / 2`` per water, so internal
  (constraint-violating) velocity components are excluded.
  """
  ke_trans, ke_rot = rigid_tip3p_ke_components(position, momentum, mass, n_waters)
  ke_com = system_com_ke_kcal(momentum, mass)
  kb = BOLTZMANN_KCAL
  # A single water has no relative translational DOF (3N - 3 = 0).
  t_trans = (
      jnp.full_like(ke_trans, jnp.nan)
      if n_waters == 1
      else 2.0 * (ke_trans - ke_com) / ((3 * n_waters - 3) * kb)
  )
  return {
      "t_total": 2.0 * (ke_trans + ke_rot - ke_com) / (rigid_tip3p_dof(n_waters) * kb),
      "t_trans": t_trans,
      "t_rot": 2.0 * ke_rot / ((3 * n_waters) * kb),
      "t_com": 2.0 * ke_com / (3 * kb),
  }


def scan_settle_rigid_temperatures(
    state,
    apply_fn,
    *,
    n_steps: int,
    burn: int,
    n_waters: int,
):
  r"""Advance ``apply_fn`` for ``n_steps`` inside ``jax.lax.scan``; return T(K) after burn.

  T is the COM-subtracted rigid-body total temperature (see
  :func:`rigid_tip3p_temperatures`).

  Host supplies concrete ``n_steps``, ``burn``, and ``n_waters`` (static at compile time).
  Wrap the returned callable in ``jax.jit`` once; do not drive steps from a Python ``for``.
  """

  def body(carry, _):
    carry = apply_fn(carry)
    t_k = rigid_tip3p_temperatures(
        carry.positions, carry.momentum, carry.mass, n_waters
    )["t_total"]
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
