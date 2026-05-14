"""Unit tests for pressure computation helpers."""
from __future__ import annotations

import pytest
import jax.numpy as jnp

from prolix.physics import pressure


class TestComputePressureAkma:
    """Tests for compute_pressure_akma bare pressure formula: P = (2K + W) / (d*V)."""

    def test_basic_ideal_gas(self):
        """Ideal gas: zero virial, P = 2K / (ndim * V)."""
        ke = jnp.array(1.5)
        virial = jnp.array(0.0)
        volume = jnp.array(100.0)
        p = pressure.compute_pressure_akma(ke, virial, volume, ndim=3)
        expected = 2.0 * 1.5 / (3 * 100.0)
        assert jnp.allclose(p, expected), f"Expected {expected}, got {p}"

    def test_nonzero_virial(self):
        """Nonzero virial shifts pressure correctly."""
        ke = jnp.array(2.0)
        virial = jnp.array(-1.0)
        volume = jnp.array(50.0)
        p = pressure.compute_pressure_akma(ke, virial, volume, ndim=3)
        expected = (2.0 * 2.0 + (-1.0)) / (3.0 * 50.0)
        assert jnp.allclose(p, expected), f"Expected {expected}, got {p}"

    def test_2d(self):
        """2D case uses ndim=2."""
        ke = jnp.array(1.0)
        virial = jnp.array(0.0)
        volume = jnp.array(10.0)
        p = pressure.compute_pressure_akma(ke, virial, volume, ndim=2)
        expected = 2.0 * 1.0 / (2.0 * 10.0)
        assert jnp.allclose(p, expected)

    def test_negative_virial_reduces_pressure(self):
        """Negative virial (attractive forces) reduces pressure vs zero virial."""
        ke = jnp.array(3.0)
        volume = jnp.array(75.0)
        p_attractive = pressure.compute_pressure_akma(ke, jnp.array(-2.0), volume)
        p_zero = pressure.compute_pressure_akma(ke, jnp.array(0.0), volume)
        assert p_attractive < p_zero

    def test_finite_output(self):
        """Output is finite for well-posed inputs."""
        ke = jnp.array(1.0)
        virial = jnp.array(0.0)
        volume = jnp.array(1e-6)  # very small but nonzero
        p = pressure.compute_pressure_akma(ke, virial, volume)
        assert jnp.isfinite(p)
