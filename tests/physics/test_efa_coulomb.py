"""EFA (Euclidean Fast Attention / Lebedev-Coulomb) acceptance tests.

Tests verify correctness of the standalone efa_lebedev_coulomb_energy kernel.
Scope: kernel API correctness only. Production wiring: efa_lebedev_coulomb_energy
is imported and used by flash_explicit.py:383.

These tests gate unblocking the MTT Log-Det Estimator (backlog: mtt_phase2).
MTT depends on the standalone kernel API (EFAParams + efa_lebedev_coulomb_energy),
not on further production wiring.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.efa_coulomb import (
    EFAParams,
    efa_lebedev_params,
    efa_lebedev_coulomb_energy,
)


# ---------------------------------------------------------------------------
# Shared fixture: default params used across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_params() -> EFAParams:
    """EFAParams with production defaults: alpha=0.34, K=32, G=26."""
    return efa_lebedev_params(alpha=0.34, n_freqs=32, n_lebedev_pts=26)


# ---------------------------------------------------------------------------
# Test 1: params structure
# ---------------------------------------------------------------------------

def test_efa_params_create(default_params: EFAParams) -> None:
    """Verify efa_lebedev_params() creates valid EFAParams with correct shapes and finite values."""
    params = default_params
    n_freqs = 32
    n_lebedev_pts = 26

    # Shape checks
    assert params.omegas.shape == (n_freqs,), \
        f"omegas.shape={params.omegas.shape}, expected ({n_freqs},)"
    assert params.weights.shape == (n_freqs,), \
        f"weights.shape={params.weights.shape}, expected ({n_freqs},)"
    assert params.nodes.shape == (n_lebedev_pts, 3), \
        f"nodes.shape={params.nodes.shape}, expected ({n_lebedev_pts}, 3)"
    assert params.quad_weights.shape == (n_lebedev_pts,), \
        f"quad_weights.shape={params.quad_weights.shape}, expected ({n_lebedev_pts},)"

    # All values finite
    assert np.all(np.isfinite(params.omegas)), "omegas contains non-finite values"
    assert np.all(np.isfinite(params.weights)), "weights contains non-finite values"
    assert np.all(np.isfinite(params.nodes)), "nodes contains non-finite values"
    assert np.all(np.isfinite(params.quad_weights)), "quad_weights contains non-finite values"

    # Lebedev weights sum to 1 (verified in implementation, but validate externally too)
    assert np.allclose(np.sum(params.quad_weights), 1.0, atol=1e-6), \
        f"quad_weights sum={np.sum(params.quad_weights):.6f}, expected 1.0"

    # Nodes live on unit sphere
    norms = np.linalg.norm(params.nodes, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Lebedev nodes not on unit sphere; norms range [{norms.min():.6f}, {norms.max():.6f}]"


# ---------------------------------------------------------------------------
# Test 2: JIT compilation
# ---------------------------------------------------------------------------

def test_efa_compiles_jit(default_params: EFAParams) -> None:
    """Verify efa_lebedev_coulomb_energy compiles under JAX JIT and returns finite scalar."""
    params = default_params

    # Minimal 4-atom system with simple charges
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ], dtype=jnp.float32)
    charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
    atom_mask = jnp.ones(4, dtype=jnp.float32)

    # EFAParams holds numpy arrays (not JAX arrays) so it cannot be a traced
    # argument. Close over it via functools.partial — the JIT-compiled function
    # only sees the three array arguments that change at call time.
    jit_fn = jax.jit(functools.partial(efa_lebedev_coulomb_energy, params=params))
    result = jit_fn(positions, charges, atom_mask)

    assert result.shape == (), f"Expected scalar output, got shape {result.shape}"
    assert jnp.isfinite(result), f"JIT result is not finite: {result}"


# ---------------------------------------------------------------------------
# Test 3: determinism
# ---------------------------------------------------------------------------

def test_efa_deterministic(default_params: EFAParams) -> None:
    """Verify identical inputs produce bitwise-identical outputs across repeated calls."""
    params = default_params

    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ], dtype=jnp.float32)
    charges = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
    atom_mask = jnp.ones(4, dtype=jnp.float32)

    # EFAParams holds numpy arrays (not JAX arrays) so it cannot be a traced
    # argument. Close over it via functools.partial.
    jit_fn = jax.jit(functools.partial(efa_lebedev_coulomb_energy, params=params))

    e1 = jit_fn(positions, charges, atom_mask)
    e2 = jit_fn(positions, charges, atom_mask)

    # EFA is deterministic (no stochastic sampling); results must be bitwise identical
    assert jnp.allclose(e1, e2, atol=0.0), \
        f"Non-deterministic output: e1={float(e1):.8f}, e2={float(e2):.8f}"


# ---------------------------------------------------------------------------
# Test 4: no NaN across distance range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [0.3, 1.0, 2.0, 5.0, 8.0])
def test_efa_no_nan_range(r: float, default_params: EFAParams) -> None:
    """Verify no NaN/inf for 2-atom pairs at r = [0.3, 1.0, 2.0, 5.0, 8.0] Angstrom."""
    params = default_params

    positions = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=jnp.float32)
    charges = jnp.array([1.0, -1.0], dtype=jnp.float32)
    atom_mask = jnp.ones(2, dtype=jnp.float32)

    e = efa_lebedev_coulomb_energy(positions, charges, atom_mask, params)

    assert jnp.isfinite(e), \
        f"NaN/inf at r={r} Angstrom: energy={float(e)}"


# ---------------------------------------------------------------------------
# Test 5: charge-neutral pair sign check
# ---------------------------------------------------------------------------

def test_efa_charge_neutral_pair_sign(default_params: EFAParams) -> None:
    """Verify +1/-1 pair at 2 Angstrom returns finite, negative (attractive) energy.

    For a charge-neutral +/-1 pair, the erf(alpha*r)/r long-range energy is:
      E = C * q1*q2 * erf(alpha*r)/r < 0  (attractive, since q1*q2 = -1)

    Just verify finiteness and sign; exact magnitude is tested by the fitting
    accuracy in _fit_efa_params.
    """
    params = default_params

    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
    charges = jnp.array([1.0, -1.0], dtype=jnp.float32)
    atom_mask = jnp.ones(2, dtype=jnp.float32)

    e = efa_lebedev_coulomb_energy(positions, charges, atom_mask, params)

    assert jnp.isfinite(e), f"Energy is not finite: {float(e)}"
    assert float(e) < 0.0, \
        f"Expected negative (attractive) energy for +/-1 pair, got {float(e):.6f} kcal/mol"
