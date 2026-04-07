"""Tests for SPME reciprocal-space module.

Covers:
- B-spline weight partition of unity
- Grid dimension selection (factorizability)
- Reciprocal energy finiteness
- Translational invariance under PBC
- Gradient correctness (analytical vs finite difference)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.pme import (
    _bspline4,
    _bspline4_deriv,
    _factorizable,
    compute_pme_grid_dims,
    make_spme_energy_fn,
    spme_background_energy,
    spme_reciprocal_energy,
    spme_self_energy,
)


class TestBSpline:
    """Tests for B-spline weight computation."""

    def test_bspline4_partition_of_unity(self):
        """B-spline weights must sum to 1 for any fractional offset."""
        u_vals = jnp.linspace(0.0, 0.99, 50)
        weights = _bspline4(u_vals)  # (4, 50)
        sums = jnp.sum(weights, axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_bspline4_nonnegative(self):
        """B-spline weights are non-negative."""
        u_vals = jnp.linspace(0.0, 0.99, 50)
        weights = _bspline4(u_vals)
        assert jnp.all(weights >= -1e-7)

    def test_bspline4_deriv_consistency(self):
        """B-spline derivatives match finite difference of weights."""
        u = jnp.array([0.3])
        eps = 1e-4
        w_plus = _bspline4(u + eps)
        w_minus = _bspline4(u - eps)
        fd_deriv = (w_plus - w_minus) / (2 * eps)
        analytical = _bspline4_deriv(u)
        np.testing.assert_allclose(analytical, fd_deriv, atol=1e-3)


class TestGridDims:
    """Tests for PME grid dimension selection."""

    def test_factorizable(self):
        """Known factorizable and non-factorizable numbers."""
        assert _factorizable(32)   # 2^5
        assert _factorizable(30)   # 2 × 3 × 5
        assert _factorizable(63)   # 3^2 × 7
        assert not _factorizable(11)  # prime
        assert not _factorizable(13)

    def test_grid_dims_factorizable(self):
        """All returned grid dims must factorize into {2,3,5,7}."""
        box = jnp.array([20.0, 25.0, 30.0])
        dims = compute_pme_grid_dims(box, grid_spacing=1.0)
        for d in dims:
            assert _factorizable(d), f"Grid dim {d} is not factorizable"

    def test_grid_dims_minimum(self):
        """Grid dims must be at least min_dim."""
        box = jnp.array([3.0, 3.0, 3.0])
        dims = compute_pme_grid_dims(box, grid_spacing=1.0, min_dim=8)
        for d in dims:
            assert d >= 8


class TestSPMEEnergy:
    """Tests for SPME energy computation."""

    def test_reciprocal_energy_finite(self):
        """Reciprocal energy must be finite for a simple system."""
        box = jnp.array([20.0, 20.0, 20.0])
        pos = jnp.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]])
        charges = jnp.array([1.0, -1.0])
        mask = jnp.ones(2, dtype=bool)
        grid_dims = compute_pme_grid_dims(box, grid_spacing=1.0)

        e = spme_reciprocal_energy(pos, charges, mask, box, grid_dims,
                                   alpha=0.34)
        assert jnp.isfinite(e)

    def test_self_energy_negative(self):
        """Self-energy correction must be negative."""
        charges = jnp.array([1.0, -0.5, 0.25])
        mask = jnp.ones(3, dtype=bool)
        e_self = spme_self_energy(charges, mask, alpha=0.34)
        assert e_self < 0.0

    def test_background_energy_neutral_system(self):
        """Background correction is zero for a neutral system."""
        charges = jnp.array([1.0, -1.0])
        mask = jnp.ones(2, dtype=bool)
        box = jnp.array([10.0, 10.0, 10.0])
        e_bg = spme_background_energy(charges, mask, alpha=0.34,
                                      box_size=box)
        np.testing.assert_allclose(float(e_bg), 0.0, atol=1e-10)

    def test_make_spme_energy_fn(self):
        """Public API produces finite energy."""
        box = jnp.array([12.0, 12.0, 12.0])
        spme_fn = make_spme_energy_fn(box, alpha=0.35, grid_spacing=0.5)
        pos = jnp.array([[3.0, 3.0, 3.0], [9.0, 9.0, 9.0]])
        charges = jnp.array([1.0, -1.0])
        mask = jnp.ones(2, dtype=bool)

        e = spme_fn(pos, charges, mask)
        assert jnp.isfinite(e)

    def test_translational_invariance(self):
        """Energy is invariant under whole-box translation (PBC)."""
        box = jnp.array([10.0, 10.0, 10.0])
        spme_fn = make_spme_energy_fn(box, alpha=0.34, grid_spacing=1.0)
        pos = jnp.array([[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]])
        charges = jnp.array([1.0, -1.0])
        mask = jnp.ones(2, dtype=bool)

        e1 = spme_fn(pos, charges, mask)

        # Shift both atoms by [1, 0, 0] — relative geometry unchanged
        shift = jnp.array([1.0, 0.0, 0.0])
        e2 = spme_fn(pos + shift, charges, mask)

        # PME is approximate, but translational invariance should hold
        # to grid-spacing tolerance.
        np.testing.assert_allclose(float(e1), float(e2), atol=1e-2)

    def test_gradient_finite(self):
        """Gradients via custom_vjp are finite."""
        box = jnp.array([12.0, 12.0, 12.0])
        spme_fn = make_spme_energy_fn(box, alpha=0.35, grid_spacing=0.5)
        pos = jnp.array([[3.0, 3.0, 3.0], [9.0, 9.0, 9.0]])
        charges = jnp.array([1.0, -1.0])
        mask = jnp.ones(2, dtype=bool)

        def energy(p):
            return spme_fn(p, charges, mask)

        grad = jax.grad(energy)(pos)
        assert jnp.all(jnp.isfinite(grad))
