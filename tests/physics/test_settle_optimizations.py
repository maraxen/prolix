"""Tests for Task 4 (#6095): 1-pass closed-form 3x3 water RATTLE, race-free Atom-0 scatters, and 3x3 Cholesky."""

from __future__ import annotations

import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix.physics import settle
from prolix.physics.settle import (
    TIP3P_ROH,
    TIP3P_RHH,
    _cholesky_upper_solve_3x3,
    _project_one_water_momentum_rigid,
    _solve_3x3,
    _solve_3x3_batch,
    project_tip3p_waters_momentum_rigid,
    settle_positions,
    settle_velocities,
)


def _make_water_system():
    """Create 3 water molecules (atoms 0..8) plus 2 padded water slots [0, 0, 0] with water_mask=False."""
    theta = np.arccos(1.0 - (TIP3P_RHH ** 2) / (2.0 * TIP3P_ROH ** 2))
    w0 = np.array(
        [
            [2.0, 2.0, 2.0],
            [2.0 + TIP3P_ROH, 2.0, 2.0],
            [2.0 + TIP3P_ROH * np.cos(theta), 2.0 + TIP3P_ROH * np.sin(theta), 2.0],
        ],
        dtype=np.float32,
    )
    w1 = w0 + np.array([5.0, 1.0, 0.0], dtype=np.float32)
    w2 = w0 + np.array([0.0, 6.0, 3.0], dtype=np.float32)
    pos = jnp.asarray(np.concatenate([w0, w1, w2], axis=0))

    rng = np.random.default_rng(101)
    vel = jnp.asarray(rng.standard_normal(size=pos.shape).astype(np.float32) * 2.0)

    water_indices = jnp.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    water_mask = jnp.array([True, True, True, False, False], dtype=bool)
    masses = jnp.array([15.999, 1.008, 1.008] * 3, dtype=jnp.float32)
    return pos, vel, water_indices, water_mask, masses


def test_no_scratch_row_concat_in_settle():
    """Verify all jnp.concatenate([..., [:1]], axis=0) scratch-row patterns are removed from settle.py."""
    src = inspect.getsource(settle)
    assert "[:1]], axis=0)" not in src


def test_settle_velocities_1pass_gram_matrix_satisfies_holonomic_constraints():
    """Verify 1-pass closed-form 3x3 Gram matrix settle_velocities achieves < 1e-5 relative velocity projection."""
    pos, vel, water_indices, water_mask, masses = _make_water_system()

    v_settled = settle_velocities(
        vel, pos, pos, water_indices, dt=0.002, mass_oxygen=15.999, mass_hydrogen=1.008, water_mask=water_mask
    )

    for w in range(3):
        idx = water_indices[w]
        r_O, r_H1, r_H2 = pos[idx[0]], pos[idx[1]], pos[idx[2]]
        v_O, v_H1, v_H2 = v_settled[idx[0]], v_settled[idx[1]], v_settled[idx[2]]

        for r_a, r_b, v_a, v_b in [
            (r_O, r_H1, v_O, v_H1),
            (r_O, r_H2, v_O, v_H2),
            (r_H1, r_H2, v_H1, v_H2),
        ]:
            dr = r_b - r_a
            u_ab = dr / jnp.linalg.norm(dr)
            proj = jnp.abs(jnp.dot(v_b - v_a, u_ab))
            assert float(proj) < 1e-5, f"Water {w} holonomic velocity residual {float(proj)} >= 1e-5"

    for w in range(3):
        idx = water_indices[w]
        m_w = masses[idx][:, None]
        p_before = jnp.sum(m_w * vel[idx], axis=0)
        p_after = jnp.sum(m_w * v_settled[idx], axis=0)
        np.testing.assert_allclose(p_after, p_before, rtol=1e-5, atol=1e-5)


def test_padded_water_mask_false_does_not_corrupt_atom_zero():
    """Verify padded water rows [0, 0, 0] with water_mask=False never overwrite Atom 0 in positions, velocities, or momenta."""
    pos, vel, water_indices, water_mask, masses = _make_water_system()

    wi_real = water_indices[:3]
    wm_real = water_mask[:3]

    rng = np.random.default_rng(55)
    pos_perturbed = pos + jnp.asarray(rng.normal(0.0, 0.02, size=pos.shape).astype(np.float32))

    pos_only_real = settle_positions(
        pos_perturbed, pos, wi_real, water_mask=wm_real
    )
    pos_with_padded = settle_positions(
        pos_perturbed, pos, water_indices, water_mask=water_mask
    )
    np.testing.assert_allclose(pos_with_padded, pos_only_real, rtol=1e-6, atol=1e-6)

    vel_only_real = settle_velocities(vel, pos, pos, wi_real, dt=0.002, water_mask=wm_real)
    vel_with_padded = settle_velocities(vel, pos, pos, water_indices, dt=0.002, water_mask=water_mask)
    np.testing.assert_allclose(vel_with_padded, vel_only_real, rtol=1e-6, atol=1e-6)

    mom = vel * masses[:, None]
    mom_only_real = project_tip3p_waters_momentum_rigid(mom, pos, masses, wi_real, water_mask=wm_real)
    mom_with_padded = project_tip3p_waters_momentum_rigid(mom, pos, masses, water_indices, water_mask=water_mask)
    np.testing.assert_allclose(mom_with_padded, mom_only_real, rtol=1e-6, atol=1e-6)


def test_cholesky_upper_solve_3x3_and_rigid_momentum_projection():
    """Verify _cholesky_upper_solve_3x3 solves L^T x = rhs (where L L^T = A) and _project_one_water_momentum_rigid is exact."""
    rng = np.random.default_rng(303)
    M = rng.standard_normal((3, 3)).astype(np.float32)
    I_spd = jnp.asarray(M @ M.T + 2.0 * np.eye(3, dtype=np.float32))
    z_rhs = jnp.asarray(rng.standard_normal(3).astype(np.float32))

    got_omega = _cholesky_upper_solve_3x3(I_spd, z_rhs)
    L_chol = jnp.linalg.cholesky(I_spd)
    ref_omega = jnp.linalg.solve(L_chol.T, z_rhs)
    np.testing.assert_allclose(got_omega, ref_omega, rtol=1e-5, atol=1e-5)
    # Also verify that if z ~ N(0, I_3), then omega = L^{-T} z has covariance L^{-T} L^{-1} = (L L^T)^{-1} = I_spd^{-1}
    np.testing.assert_allclose(L_chol.T @ got_omega, z_rhs, rtol=1e-5, atol=1e-5)

    # Verify _project_one_water_momentum_rigid produces rigid-body velocities (zero internal bond rate of change)
    pos, vel, _, _, masses = _make_water_system()
    r_3 = pos[:3]
    m_3 = masses[:3]
    p_3 = vel[:3] * m_3[:, None]
    p_proj = _project_one_water_momentum_rigid(p_3, r_3, m_3)
    v_proj = p_proj / m_3[:, None]

    for a, b in [(0, 1), (0, 2), (1, 2)]:
        dr = r_3[b] - r_3[a]
        dv = v_proj[b] - v_proj[a]
        assert float(jnp.abs(jnp.dot(dr, dv))) < 1e-5
