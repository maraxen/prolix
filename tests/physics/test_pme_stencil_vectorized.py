"""Order-4 B-spline spread/gather must be one scatter and one gather, not 4³ loops."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics.pme import spread_charges, spme_energy_with_forces
from tests.physics.pme_oracle import spread_loop_oracle, tiny_system


def _count_primitive(jaxpr, needle: str) -> int:
    n = 0
    for eqn in jaxpr.eqns:
        if needle in str(eqn.primitive.name):
            n += 1
        for val in eqn.params.values():
            inner = getattr(val, "jaxpr", None)
            if inner is not None:
                n += _count_primitive(inner, needle)
            if hasattr(val, "eqns"):
                n += _count_primitive(val, needle)
    return n


def test_spread_charges_single_scatter_add():
    pos, charges, mask, box, grid_dims = tiny_system()

    def _spread(p, q, m, b):
        return spread_charges(p, q, m, b, grid_dims, 4)

    closed = jax.make_jaxpr(_spread)(pos, charges, mask, box)
    n_scatter = _count_primitive(closed.jaxpr, "scatter")
    assert n_scatter <= 2, f"expected one fused scatter-add, got {n_scatter}"


def test_spread_charges_matches_loop_oracle():
    pos, charges, mask, box, grid_dims = tiny_system()
    got = spread_charges(pos, charges, mask, box, grid_dims, 4)
    ref = spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_spme_force_gather_not_unrolled_64():
    pos, charges, mask, box, grid_dims = tiny_system()
    alpha, order = 0.34, 4

    def energy(p):
        return spme_energy_with_forces(p, charges, mask, box, grid_dims, alpha, order)

    closed = jax.make_jaxpr(jax.grad(energy))(pos)
    n_gather = _count_primitive(closed.jaxpr, "gather")
    # FFT / indexing adds a few gathers; the old stencil was 64 by itself.
    assert n_gather < 32, f"expected not 4³-unrolled gathers, got {n_gather}"
