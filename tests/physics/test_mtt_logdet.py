"""Tests for the MTT log-det estimator (Hutchinson + Lanczos).

Three test cases:
  1. Smoke: JIT compilation + determinism
  2. Accuracy gate (slow): N=64, MTT estimate within 5% of dense slogdet baseline
  3. Milestone placeholder: N=256 (skipped until Sprint 10)

Dense baseline construction:
  K = features @ features.T  (same features used by the matvec in mtt_logdet.py)
  L_reg = diag(K.sum(axis=1)) - K + eps * I
  slogdet(L_reg)

This mirrors _matvec() exactly, so both estimate and baseline use the same K.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.efa_coulomb import efa_erf_features, efa_lebedev_params
from prolix.physics.mtt_logdet import (
    MTTParams,
    mtt_estimate_log_det,
    mtt_logdet_params,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_positions_charges(n: int, seed: int = 42):
    """Random positions in [0, 10] Å and standard-normal charges."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 10.0, size=(n, 3)).astype(np.float32)
    charges = rng.standard_normal(size=(n,)).astype(np.float32)
    atom_mask = np.ones(n, dtype=np.float32)
    return (
        jnp.asarray(positions),
        jnp.asarray(charges),
        jnp.asarray(atom_mask),
    )


def _dense_logdet_baseline(positions, atom_mask, efa_params, eps: float) -> float:
    """Dense slogdet baseline for validation.

    Constructs K = Φ Φᵀ explicitly (O(N²) — test-only).
    Must mirror _matvec() in mtt_logdet.py exactly:
      K_ij = φᵢ · φⱼ  (efa_erf_features, masked)
      D_i  = Σⱼ K_ij
      L_reg = D - K + εI
    """
    features = efa_erf_features(positions, efa_params)          # (N, D)
    atom_mask_f = jnp.asarray(atom_mask, dtype=features.dtype)
    features = features * atom_mask_f[:, None]                  # (N, D)

    K = features @ features.T                                   # (N, N)
    D_diag = K.sum(axis=1)                                      # (N,)
    N = K.shape[0]
    L_reg = jnp.diag(D_diag) - K + eps * jnp.eye(N)

    # Use numpy for stable slogdet (avoids potential JAX float32 precision issues)
    sign, logdet = np.linalg.slogdet(np.array(L_reg, dtype=np.float64))
    return float(logdet)


# ---------------------------------------------------------------------------
# Test 1: smoke — JIT compilation and determinism
# ---------------------------------------------------------------------------

def test_mtt_logdet_jit_and_determinism():
    """MTT log-det: JIT compiles without error; two calls with same key are identical."""
    N = 8
    positions, charges, atom_mask = _make_positions_charges(N, seed=0)

    efa_params = efa_lebedev_params(alpha=0.34, n_freqs=16, n_lebedev_pts=26)
    params = mtt_logdet_params(efa_params, n_probes=5, n_lanczos=6, eps=1e-4)

    log_det_fn = jax.jit(
        functools.partial(mtt_estimate_log_det, params=params)
    )

    key = jax.random.PRNGKey(7)

    # First call: triggers JIT compilation
    result_a = log_det_fn(positions, charges, atom_mask, key)
    # Second call: same key → same output (deterministic)
    result_b = log_det_fn(positions, charges, atom_mask, key)

    # JIT compiled successfully if we reach here without exception
    assert jnp.isfinite(result_a), f"MTT estimate is not finite: {result_a}"
    np.testing.assert_allclose(
        float(result_a),
        float(result_b),
        rtol=0.0,
        atol=0.0,
        err_msg="MTT estimate is not deterministic for the same PRNGKey",
    )


# ---------------------------------------------------------------------------
# Test 2: accuracy gate — N=64, 5% relative error vs dense slogdet
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mtt_logdet_n64_accuracy():
    """MTT log-det: N=64 estimate within 5% of dense numpy.linalg.slogdet baseline.

    Uses n_probes=5, n_lanczos=20 (fast; production uses 20/50).
    Dense baseline uses the same feature matrix as the matvec, so any
    discrepancy reflects Hutchinson/Lanczos approximation error, not
    a mismatch between the two codepaths.
    """
    N = 64
    positions, charges, atom_mask = _make_positions_charges(N, seed=123)

    efa_params = efa_lebedev_params(alpha=0.34, n_freqs=32, n_lebedev_pts=26)
    params = mtt_logdet_params(efa_params, n_probes=10, n_lanczos=20, eps=1e-4)

    key = jax.random.PRNGKey(42)
    mtt_estimate = float(
        mtt_estimate_log_det(positions, charges, atom_mask, key, params=params)
    )

    dense_baseline = _dense_logdet_baseline(positions, atom_mask, efa_params, eps=1e-4)

    rel_error = abs(mtt_estimate - dense_baseline) / (abs(dense_baseline) + 1e-12)

    assert rel_error < 0.05, (
        f"MTT estimate deviates {rel_error:.1%} from dense baseline "
        f"(mtt={mtt_estimate:.4f}, dense={dense_baseline:.4f}); "
        f"threshold is 5%"
    )


# ---------------------------------------------------------------------------
# Test 3: Sprint 10 milestone placeholder — N=256
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Sprint 10 milestone — N=256 accuracy not yet validated")
def test_mtt_logdet_n256_milestone():
    """MTT log-det: N=256 estimate within 5% of dense slogdet baseline (Sprint 10).

    Placeholder test for the N=256 scaling milestone. Unskip when Sprint 10
    validates O(ND) performance at this scale.
    """
    N = 256
    positions, charges, atom_mask = _make_positions_charges(N, seed=256)

    efa_params = efa_lebedev_params(alpha=0.34, n_freqs=32, n_lebedev_pts=26)
    params = mtt_logdet_params(efa_params, n_probes=20, n_lanczos=50, eps=1e-4)

    key = jax.random.PRNGKey(99)
    mtt_estimate = float(
        mtt_estimate_log_det(positions, charges, atom_mask, key, params=params)
    )

    dense_baseline = _dense_logdet_baseline(positions, atom_mask, efa_params, eps=1e-4)

    rel_error = abs(mtt_estimate - dense_baseline) / (abs(dense_baseline) + 1e-12)

    assert rel_error < 0.05, (
        f"MTT estimate deviates {rel_error:.1%} from dense baseline "
        f"(mtt={mtt_estimate:.4f}, dense={dense_baseline:.4f}); "
        f"threshold is 5%"
    )
