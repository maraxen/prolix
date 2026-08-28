"""Pallas SPME spread/gather vs XLA stencil (ADR-006)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.physics.pme_oracle import spread_loop_oracle, tiny_system


def test_spread_interpret_matches_oracle():
    from prolix.physics.pme_pallas import spread_charges_pallas

    pos, charges, mask, box, grid_dims = tiny_system()
    got = spread_charges_pallas(
        pos, charges, mask, box, grid_dims, order=4, interpret=True
    )
    ref = spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_grad_pallas_matches_xla_interpret():
    from prolix.physics.pme import spme_energy_with_forces

    pos, charges, mask, box, grid_dims = tiny_system()
    alpha, order = 0.34, 4

    def e_xla(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, alpha, order, False
        )

    def e_pallas(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, alpha, order, True
        )

    gx = jax.grad(e_xla)(pos)
    gp = jax.grad(e_pallas)(pos)
    np.testing.assert_allclose(gp, gx, rtol=1e-4, atol=1e-4)


def _gpu_unavailable() -> bool:
    return jax.default_backend() != "gpu"


@pytest.mark.skipif(_gpu_unavailable(), reason="Pallas GPU path needs CUDA")
def test_spread_gpu_matches_oracle():
    from prolix.physics.pme_pallas import spread_charges_pallas

    pos, charges, mask, box, grid_dims = tiny_system()
    got = spread_charges_pallas(
        pos, charges, mask, box, grid_dims, order=4, interpret=False
    )
    ref = spread_loop_oracle(pos, charges, mask, box, grid_dims, 4)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(_gpu_unavailable(), reason="Pallas GPU path needs CUDA")
def test_grad_pallas_gpu_matches_xla():
    from prolix.physics.pme import spme_energy_with_forces

    pos, charges, mask, box, grid_dims = tiny_system()
    alpha, order = 0.34, 4

    def e_xla(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, alpha, order, False
        )

    def e_pallas(p):
        return spme_energy_with_forces(
            p, charges, mask, box, grid_dims, alpha, order, True
        )

    gx = jax.grad(e_xla)(pos)
    gp = jax.grad(e_pallas)(pos)
    np.testing.assert_allclose(gp, gx, rtol=1e-4, atol=1e-4)
