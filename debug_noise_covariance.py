#!/usr/bin/env python3
"""Diagnostic: Check noise covariance generation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics.settle import (
    _ou_noise_one_water_rigid,
    _project_one_water_momentum_rigid,
    get_water_indices,
)
from tests.physics.test_explicit_langevin_tip3p_parity import _grid_water_positions


def check_noise_covariance():
    """Check that noise covariance matches theory."""
    jax.config.update("jax_enable_x64", True)

    # Setup for ONE water molecule
    positions_a, box_edge = _grid_water_positions(1, spacing_angstrom=10.0)
    r_w = positions_a[:3]  # O, H1, H2 positions
    m_w = jnp.array([15.999, 1.008, 1.008])  # O, H1, H2 masses
    kT = 0.596161  # 300K in kcal/mol

    print("=== NOISE COVARIANCE DIAGNOSTIC ===\n")
    print(f"Water at: {r_w}\n")
    print(f"Masses: {m_w}")
    print(f"kT: {kT:.6f} kcal/mol\n")

    # Compute projection info
    msum = jnp.sum(m_w)
    com = jnp.sum(m_w[:, None] * r_w, axis=0) / msum
    rrel = r_w - com

    # Construct Jacobian (from _project_one_water_momentum_rigid)
    from prolix.physics.settle import _skew_symmetric3
    rows = []
    for i in range(3):
        row = jnp.concatenate([jnp.eye(3, dtype=r_w.dtype), -_skew_symmetric3(rrel[i])], axis=1)
        rows.append(row)
    jmat = jnp.vstack(rows)
    m_rep = jnp.repeat(m_w, 3)
    g = (jmat.T * m_rep) @ jmat

    print(f"Gramian G = J^T M J:")
    print(g)
    print(f"\nG eigenvalues: {jnp.linalg.eigvals(g)}")
    print(f"G^-1 eigenvalues: {1.0 / jnp.linalg.eigvals(g)}\n")

    # Sample many noise vectors and compute empirical covariance
    n_samples = 10000
    key = jax.random.PRNGKey(42)
    noise_samples = []

    for i in range(n_samples):
        key, split = jax.random.split(key)
        p_noise, _ = _ou_noise_one_water_rigid(split, r_w, m_w, kT)
        noise_samples.append(p_noise.reshape(-1))

    noise_arr = np.array(noise_samples)  # (n_samples, 9)
    empirical_cov = np.cov(noise_arr.T)  # (9, 9)

    # Theoretical covariance: kT * M * J * G^-1 * J^T * M (current formula)
    # But we think it should be: kT * M * P_rigid = kT * M * J * G^-1 * J^T
    p_rigid_proj = jmat @ jnp.linalg.inv(g) @ jmat.T  # (9, 9)
    m_mat = jnp.diag(m_rep)  # (9, 9)

    theoretical_cov_v1 = kT * m_mat @ p_rigid_proj  # What we think is correct
    theoretical_cov_v2 = kT * m_mat @ jmat @ jnp.linalg.inv(g) @ jmat.T @ m_mat  # Current formula

    print("=== COVARIANCE COMPARISON ===\n")
    print(f"Empirical trace:              {np.trace(empirical_cov):.6f}")
    print(f"Theory (M * J * G^-1 * J^T):   {np.trace(theoretical_cov_v1):.6f}")
    print(f"Theory (M * J * G^-1 * J^T*M): {np.trace(theoretical_cov_v2):.6f}\n")

    # Ratio of empirical to theoretical
    ratio_v1 = np.trace(empirical_cov) / np.trace(theoretical_cov_v1)
    ratio_v2 = np.trace(empirical_cov) / np.trace(theoretical_cov_v2)

    print(f"Ratio (emp / theory_v1): {ratio_v1:.4f}")
    print(f"Ratio (emp / theory_v2): {ratio_v2:.4f}\n")

    # Expected kinetic energy (for 6 DOF rigid body)
    # K = 0.5 * sum(p_i^2 / m_i) averaged = 3 * kT
    expected_ke = 3.0 * kT
    empirical_ke = 0.5 * np.mean([np.sum(noise_arr[i]**2 / m_rep) for i in range(n_samples)])

    print(f"Expected KE (3*kT):    {expected_ke:.6f} kcal/mol")
    print(f"Empirical KE:          {empirical_ke:.6f} kcal/mol")
    print(f"Ratio (emp / theory):  {empirical_ke / expected_ke:.4f}\n")

    # For comparison: unconstrained noise (full 3N space)
    # Should be: p ~ N(0, kT * M)
    unconstrained_noise = jnp.sqrt(m_w * kT)[:, None] * jax.random.normal(
        key, (3, 3)
    )
    unconstrained_cov = kT * m_mat
    unconstrained_ke = 0.5 * np.trace(unconstrained_cov @ np.linalg.inv(m_mat))
    print(f"Unconstrained KE (9 * kT):  {9.0 * kT:.6f} kcal/mol")
    print(f"Constrained / Unconstrained: {empirical_ke / (9.0 * kT):.4f}")


if __name__ == "__main__":
    check_noise_covariance()
