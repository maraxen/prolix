#!/usr/bin/env python3
"""Validation script: Verify RFF approximates erf(αr)/r (not erfc).

This script tests that the corrected rff_frequency_sample and erfc_rff_features
produce an RFF approximation to erf(αr)/r, the long-range component of the
hybrid Coulomb decomposition: 1/r = erfc(αr)/r + erf(αr)/r.

Exit codes:
  0 = All tests pass (RFF matches erf(αr)/r within tolerance)
  1 = RFF approximation error too large
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from prolix.physics.rff_coulomb import erfc_rff_features, rff_frequency_sample


def erf_kernel_exact(alpha: float, r: float) -> float:
    """Exact kernel K(r) = erf(αr)/r."""
    return float(jax.scipy.special.erf(alpha * r) / r)


def estimate_rff_kernel(
    alpha: float,
    r: float,
    d_features: int = 2048,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate K(r) via RFF and return (estimate, stdev across seeds).

    Runs multiple seeds and computes mean and std of estimates.
    """
    estimates = []

    for seed in range(5):  # 5 independent seeds for variance estimate
        key = jax.random.PRNGKey(seed)
        omega = rff_frequency_sample(alpha, d_features, key)

        # Create two points at distance r along the z-axis
        x_i = jnp.zeros((1, 3))
        x_j = jnp.array([[0.0, 0.0, r]])

        phi_i = erfc_rff_features(x_i, omega, alpha)  # (1, D)
        phi_j = erfc_rff_features(x_j, omega, alpha)  # (1, D)

        # Dot product: phi_i^T phi_j
        k_rff = float(jnp.dot(phi_i[0, :], phi_j[0, :]))
        estimates.append(k_rff)

    estimates_arr = jnp.array(estimates)
    mean_est = float(jnp.mean(estimates_arr))
    std_est = float(jnp.std(estimates_arr))

    return mean_est, std_est


def main():
    """Run validation tests."""
    print("=" * 70)
    print("RFF Kernel Validation: erf(αr)/r Approximation")
    print("=" * 70)

    alpha = 0.34  # Standard Ewald damping
    test_distances = [0.5, 1.0, 2.0, 4.0, 8.0]
    d_features = 2048

    print(f"\nAlpha = {alpha} Å⁻¹")
    print(f"Features D = {d_features}")
    print(f"Test distances: {test_distances} Å\n")

    # Table header
    print(f"{'r (Å)':>8} | {'erf(αr)/r':>12} | {'RFF est':>12} | "
          f"{'RFF std':>12} | {'Rel Error %':>12} | {'Status':>8}")
    print("-" * 80)

    max_rel_error = 0.0
    all_pass = True

    for r in test_distances:
        exact = erf_kernel_exact(alpha, r)
        rff_est, rff_std = estimate_rff_kernel(alpha, r, d_features)

        # Relative error (absolute difference / abs(exact))
        if abs(exact) > 1e-10:
            rel_error = abs(rff_est - exact) / abs(exact) * 100.0
        else:
            # At r=0, erf(0)/0 is handled by limit; skip
            rel_error = 0.0

        # Pass threshold: < 10% relative error at standard distances
        # At r=1.0 Å and r=2.0 Å, expect < 5% with D=2048
        # At r=0.5 Å and r=8.0 Å, allow 10% (tails have more variance)
        threshold = 5.0 if r in [1.0, 2.0] else 10.0
        status = "PASS" if rel_error < threshold else "FAIL"
        if rel_error >= threshold:
            all_pass = False
        max_rel_error = max(max_rel_error, rel_error)

        print(f"{r:8.1f} | {exact:12.6f} | {rff_est:12.6f} | "
              f"{rff_std:12.6f} | {rel_error:12.2f} | {status:>8}")

    print("-" * 80)
    print(f"\nMax relative error: {max_rel_error:.2f}%")

    if all_pass:
        print("\n✓ RFF kernel validation PASSED: erf(αr)/r approximation is correct")
        return 0
    else:
        print("\n✗ RFF kernel validation FAILED: errors exceed tolerance")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
