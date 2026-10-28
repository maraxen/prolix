"""SPME charge spread and backward gather behind ``xtrax.op`` (task 261027_xtrax-op-prolix-spike).

``make_gather_op`` pairs the plain-JAX gather that ``_spme_bwd`` runs when the
Pallas path is off (``pme.py``, the ``else`` branch of ``_spme_bwd``) with
``pme_pallas.gather_pallas``. ``make_spread_op`` pairs the plain-JAX B-spline
charge spread with ``pme_pallas.spread_charges_pallas``. The caller's
``xtrax.intent.Context`` picks one: ``Context(kernels=True)`` runs the Pallas
kernel on GPU (compiled) or CPU (interpret mode), and the default context and
every export context run the reference.

``grid_dims`` and ``order`` are closed over, never passed as op arguments:
every op argument is traced, and the Pallas kernel needs them as Python ints.
"""

from __future__ import annotations

import functools

import xtrax
from xtrax.intent import Context

from prolix.physics.pme import (
    _gather_reference as gather_reference,
    _spread_charges_reference as spread_reference,
)
from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas


def make_spread_op(
    grid_dims: tuple[int, int, int],
    order: int = 4,
    *,
    context: Context | None = None,
) -> xtrax.Op:
    """Build the SPME charge spread op for one mesh shape.

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
        Callable ``(positions, charges, atom_mask, box_size) -> Q``.
    """
    dims = (int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2]))
    return xtrax.op(
        functools.partial(spread_reference, grid_dims=dims, order=order),
        kernels={
            "gpu": functools.partial(spread_charges_pallas, grid_dims=dims, order=order, interpret=False),
            "cpu": functools.partial(spread_charges_pallas, grid_dims=dims, order=order, interpret=True),
        },
        name="pme_spread",
        context=context,
    )


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


__all__ = [
    "gather_reference",
    "make_gather_op",
    "make_spread_op",
    "spread_reference",
]
