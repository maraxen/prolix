#!/usr/bin/env python3
"""Diagnose momentum rescaling issue in NPT barostat."""

import jax
import jax.numpy as jnp
import numpy as np

# Simulate the Parrinello-Rahman momentum rescaling issue
def test_momentum_scaling():
    """Test whether momentum should be scaled by μ or 1/μ."""

    # Initial state: 3 water molecules
    n_waters = 3
    n_atoms = n_waters * 3

    # Mass array (O, H, H per water)
    mass_flat = jnp.array([15.999, 1.008, 1.008] * n_waters)

    # Positions: arbitrary
    positions = jnp.ones((n_atoms, 3)) * 5.0

    # Momentum: choose specific values to track
    momentum = jnp.ones((n_atoms, 3)) * 2.0  # 2.0 amu*Å/fs in each component

    # Compute initial kinetic energy
    velocity = momentum / mass_flat[:, jnp.newaxis]
    ke_initial = 0.5 * jnp.sum(mass_flat[:, jnp.newaxis] * velocity**2)

    print("=== Initial State ===")
    print(f"Momentum sum: {jnp.sum(momentum):.6e}")
    print(f"KE: {ke_initial:.6e}")

    # Now scale the box (and positions) by μ = 0.99 (box contracts)
    mu = 0.99
    scaled_positions = positions * mu

    print(f"\n=== After scaling positions by μ = {mu} ===")
    print(f"Scaled positions (sample): {scaled_positions[0]}")

    # Question: how should momentum be rescaled?
    # Scenario A: momentum = momentum / mu (current code)
    momentum_A = momentum / mu
    velocity_A = momentum_A / mass_flat[:, jnp.newaxis]
    ke_A = 0.5 * jnp.sum(mass_flat[:, jnp.newaxis] * velocity_A**2)

    # Scenario B: momentum = momentum * mu (correct Parrinello-Rahman)
    momentum_B = momentum * mu
    velocity_B = momentum_B / mass_flat[:, jnp.newaxis]
    ke_B = 0.5 * jnp.sum(mass_flat[:, jnp.newaxis] * velocity_B**2)

    # Scenario C: no momentum scaling
    ke_C = ke_initial

    print(f"\nScenario A (divide by mu): KE = {ke_A:.6e}, ratio to initial = {ke_A/ke_initial:.4f}")
    print(f"Scenario B (multiply by mu): KE = {ke_B:.6e}, ratio to initial = {ke_B/ke_initial:.4f}")
    print(f"Scenario C (no rescale): KE = {ke_C:.6e}, ratio to initial = {ke_C/ke_initial:.4f}")

    print("\n=== Physical Analysis ===")
    print("Box contracts by factor μ = 0.99 (volume decreases)")
    print("If particle velocity stays constant in lab frame:")
    print("  - Position scales: r' = μ*r")
    print("  - Distance scales: |r'| = μ*|r|  (shorter)")
    print("  - Velocity scales: v' = μ*v  (moved shorter distance in same time)")
    print("  - Momentum should scale: p' = m*v' = m*μ*v = μ*p")
    print()
    print("Result: Scenario B (multiply by μ) is physically correct!")
    print(f"Current code uses Scenario A (divide by μ), causing KE↑ by {ke_A/ke_B:.1f}x")

    # Now check what happens with a larger box change
    print("\n=== Sensitivity to box scaling factor ===")
    for mu_test in [0.95, 0.99, 1.01, 1.05]:
        ke_wrong = 0.5 * jnp.sum(mass_flat[:, jnp.newaxis] * (momentum / mu_test / mass_flat[:, jnp.newaxis])**2)
        ke_correct = 0.5 * jnp.sum(mass_flat[:, jnp.newaxis] * (momentum * mu_test / mass_flat[:, jnp.newaxis])**2)
        print(f"μ = {mu_test:.2f}: wrong gives {ke_wrong/ke_initial:7.4f}x, correct gives {ke_correct/ke_initial:7.4f}x")


if __name__ == "__main__":
    test_momentum_scaling()
