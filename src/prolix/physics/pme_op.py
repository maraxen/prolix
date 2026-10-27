"""SPME backward gather behind ``xtrax.op`` (spike, task 261027_xtrax-op-prolix-spike).

``make_gather_op`` pairs the plain-JAX gather that ``_spme_bwd`` runs when the
Pallas path is off (``pme.py``, the ``else`` branch of ``_spme_bwd``) with
``pme_pallas.gather_pallas``. The caller's ``xtrax.intent.Context`` picks one:
``Context(kernels=True)`` runs the Pallas kernel on GPU (compiled) or CPU
(interpret mode), and the default context and every export context run the
reference. ``pme.py`` is unchanged; nothing in prolix calls this module yet.

``grid_dims`` and ``order`` are closed over, never passed as op arguments:
every op argument is traced, and the Pallas kernel needs them as Python ints.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import xtrax
from xtrax.intent import Context

from prolix.physics.pme import _bspline4_deriv, _bspline4_stencil
from prolix.physics.pme_pallas import gather_pallas


def gather_reference(
  positions: jax.Array,
  charges: jax.Array,
  atom_mask: jax.Array,
  box_size: jax.Array,
  theta_norm: jax.Array,
  *,
  grid_dims: tuple[int, int, int],
  order: int = 4,
) -> tuple[jax.Array, jax.Array]:
  """Gather ``(forces, potentials)`` from ``theta_norm`` in plain JAX.

  Same math and convention as the non-Pallas branch of ``_spme_bwd``: the
  values before ``COULOMB_CONSTANT`` and the cotangent ``g`` are applied.
  Ghost-atom potentials are 0 here, matching ``gather_pallas``.

  Parameters
  ----------
  positions : jax.Array
      ``(N, 3)`` atom positions.
  charges : jax.Array
      ``(N,)`` charges.
  atom_mask : jax.Array
      ``(N,)`` bool, ``False`` for ghost atoms.
  box_size : jax.Array
      ``(3,)`` box edge lengths.
  theta_norm : jax.Array
      ``grid_dims`` reciprocal-space potential mesh.
  grid_dims : tuple of int
      Mesh shape.
  order : int
      B-spline order; only 4 is supported.

  Returns
  -------
  tuple of jax.Array
      ``forces`` ``(N, 3)`` and ``potentials`` ``(N,)``.
  """
  dtype = positions.dtype
  st = _bspline4_stencil(positions, charges, atom_mask, box_size, grid_dims, order)
  wx, wy, wz = st["wx"], st["wy"], st["wz"]
  ox, oy, oz = st["ox"], st["oy"], st["oz"]
  u = st["u"]
  dwx = _bspline4_deriv(u[:, 0])
  dwy = _bspline4_deriv(u[:, 1])
  dwz = _bspline4_deriv(u[:, 2])
  q_masked = st["q"].astype(dtype)

  t_val = theta_norm[st["ix"], st["iy"], st["iz"]]
  w_unit = jnp.transpose(wx[ox] * wy[oy] * wz[oz], (1, 0))
  # Ghost atoms get potential 0, as in the Pallas kernel. _spme_bwd's own
  # reference branch leaves them non-zero and masks dE_dq afterwards, so the
  # gradient it returns is the same either way.
  potentials = jnp.where(atom_mask, jnp.sum(w_unit * t_val, axis=1), 0.0)
  scale = st["K_arr"] / box_size
  fx = q_masked[:, None] * scale[0] * jnp.transpose(dwx[ox] * wy[oy] * wz[oz], (1, 0)) * t_val
  fy = q_masked[:, None] * scale[1] * jnp.transpose(wx[ox] * dwy[oy] * wz[oz], (1, 0)) * t_val
  fz = q_masked[:, None] * scale[2] * jnp.transpose(wx[ox] * wy[oy] * dwz[oz], (1, 0)) * t_val
  forces = jnp.stack([jnp.sum(fx, axis=1), jnp.sum(fy, axis=1), jnp.sum(fz, axis=1)], axis=-1)
  return forces.astype(dtype), potentials.astype(dtype)


def make_gather_op(
  grid_dims: tuple[int, int, int],
  order: int = 4,
  *,
  context: Context | None = None,
) -> xtrax.Op:
  """Build the gather op for one mesh shape.

  Parameters
  ----------
  grid_dims : tuple of int
      Mesh shape, closed over by the reference and both kernels.
  order : int
      B-spline order; only 4 is supported.
  context : Context or None
      Context the op selects against. ``None`` leaves it unbound, so it runs
      the reference.

  Returns
  -------
  xtrax.Op
      Callable ``(positions, charges, atom_mask, box_size, theta_norm) ->
      (forces, potentials)``.
  """
  dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
  return xtrax.op(
    functools.partial(gather_reference, grid_dims=dims, order=order),
    kernels={
      "gpu": functools.partial(gather_pallas, grid_dims=dims, order=order, interpret=False),
      "cpu": functools.partial(gather_pallas, grid_dims=dims, order=order, interpret=True),
    },
    name="pme_gather",
    context=context,
  )
