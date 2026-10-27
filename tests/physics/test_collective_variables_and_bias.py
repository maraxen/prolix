"""Tests for native differentiable JAX collective variables and enhanced-sampling bias potentials."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix import export
from prolix.physics.collective_variables import (
    BOLTZMANN_KCAL,
    BiasSpec,
    BiasState,
    cv_distance,
    cv_plumed_kd_distance,
    cv_radius_of_gyration,
    cv_rmsd_kabsch,
    cv_torsion,
    make_bias_state_from_controls,
    opes_metad_bias_energy,
    opes_metad_update_state,
    smd_bias_energy,
    umbrella_bias_energy,
    wt_metad_bias_energy,
    wt_metad_update_state,
)

pytestmark = pytest.mark.fast


def _finite_diff_grad(fn, pos: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central finite-difference gradient of scalar function ``fn(positions)``."""
    pos_64 = np.asarray(pos, dtype=np.float64)
    grad_fd = np.zeros_like(pos_64, dtype=np.float64)
    with jax.enable_x64(True):
        for i in range(pos_64.shape[0]):
            for d in range(pos_64.shape[1]):
                pos_plus = pos_64.copy()
                pos_minus = pos_64.copy()
                pos_plus[i, d] += eps
                pos_minus[i, d] -= eps
                f_plus = float(fn(jnp.asarray(pos_plus, dtype=jnp.float64)))
                f_minus = float(fn(jnp.asarray(pos_minus, dtype=jnp.float64)))
                grad_fd[i, d] = (f_plus - f_minus) / (2.0 * eps)
    return grad_fd


def test_cv_distance_value_grad_and_vmap():
    pos = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 6.0, 3.0],
            [1.0, 2.0, 8.0],
        ],
        dtype=np.float32,
    )
    pos_j = jnp.asarray(pos)

    d01 = cv_distance(pos_j, 0, 1)
    assert float(d01) == pytest.approx(5.0, abs=1e-5)

    grad_ad = np.asarray(jax.grad(lambda p: cv_distance(p, 0, 1))(pos_j))
    grad_fd = _finite_diff_grad(lambda p: cv_distance(p, 0, 1), pos, eps=1e-3)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-4, atol=1e-4)

    # Periodic minimum-image distance
    disp_fn, _ = space.periodic(jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32))
    pos_pbc = jnp.array([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=jnp.float32)
    assert float(cv_distance(pos_pbc, 0, 1, displacement_fn=disp_fn)) == pytest.approx(
        1.0, abs=1e-5
    )

    # Batch vmap over configurations
    batch_pos = jnp.stack([pos_j, pos_j * 2.0], axis=0)
    d_batch = jax.vmap(lambda p: cv_distance(p, 0, 1))(batch_pos)
    np.testing.assert_allclose(np.asarray(d_batch), np.array([5.0, 10.0]), atol=1e-5)


def test_cv_torsion_value_grad_and_pad_safety():
    # 90-degree dihedral configuration
    pos = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    pos_j = jnp.asarray(pos)
    idx = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    phi = cv_torsion(pos_j, idx)
    assert abs(abs(float(phi)) - (math.pi / 2.0)) < 1e-5

    grad_ad = np.asarray(jax.grad(lambda p: cv_torsion(p, idx))(pos_j))
    grad_fd = _finite_diff_grad(lambda p: cv_torsion(p, idx), pos, eps=1e-3)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-3, atol=1e-4)

    # Padded [0, 0, 0, 0] quadruplet produces finite 0.0 angle and finite 0.0 gradient
    idx_pad = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    phi_pad, grad_pad = jax.value_and_grad(lambda p: cv_torsion(p, idx_pad))(pos_j)
    assert float(phi_pad) == pytest.approx(0.0, abs=1e-7)
    assert np.all(np.isfinite(np.asarray(grad_pad)))
    np.testing.assert_allclose(np.asarray(grad_pad), 0.0, atol=1e-7)


def test_cv_radius_of_gyration_value_grad_and_mask():
    rng = np.random.default_rng(42)
    pos_real = rng.normal(size=(6, 3)).astype(np.float32) * 3.0
    pos_padded = np.vstack([pos_real, np.array([[999.0, -500.0, 250.0]], dtype=np.float32)])
    masses_real = np.array([12.0, 14.0, 16.0, 12.0, 1.0, 32.0], dtype=np.float32)
    masses_padded = np.append(masses_real, np.float32(100.0))
    mask_padded = jnp.array([True, True, True, True, True, True, False])

    rg_unpadded = cv_radius_of_gyration(jnp.asarray(pos_real), masses=jnp.asarray(masses_real))
    rg_padded = cv_radius_of_gyration(
        jnp.asarray(pos_padded),
        atom_mask=mask_padded,
        masses=jnp.asarray(masses_padded),
    )
    assert float(rg_unpadded) == pytest.approx(float(rg_padded), abs=1e-5)

    grad_ad = np.asarray(
        jax.grad(
            lambda p: cv_radius_of_gyration(
                p, atom_mask=mask_padded, masses=jnp.asarray(masses_padded)
            )
        )(jnp.asarray(pos_padded))
    )
    grad_fd = _finite_diff_grad(
        lambda p: cv_radius_of_gyration(
            p, atom_mask=mask_padded, masses=jnp.asarray(masses_padded)
        ),
        pos_padded,
        eps=1e-3,
    )
    np.testing.assert_allclose(grad_ad[:6], grad_fd[:6], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(grad_ad[6], 0.0, atol=1e-7)


def test_cv_rmsd_kabsch_superposition_grad_and_stablehlo_portability():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(8, 3)).astype(np.float32) * 4.0

    # Apply exact 3D rotation + translation -> RMSD must be ~0
    theta = 1.15
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    trans = np.array([12.5, -7.3, 4.1], dtype=np.float32)
    pos_rigid = ref @ rot.T + trans

    rmsd_zero = cv_rmsd_kabsch(jnp.asarray(pos_rigid), jnp.asarray(ref))
    assert float(rmsd_zero) < 1e-4

    # Perturb positions so RMSD > 0 and verify analytical gradient vs finite differences
    perturbation = rng.normal(size=(8, 3)).astype(np.float32) * 0.35
    pos_pert = pos_rigid + perturbation
    pos_padded = np.vstack([pos_pert, np.array([[50.0, -50.0, 50.0]], dtype=np.float32)])
    ref_padded = np.vstack([ref, np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])
    mask = jnp.array([True] * 8 + [False])

    rmsd_val = cv_rmsd_kabsch(jnp.asarray(pos_pert), jnp.asarray(ref))
    rmsd_masked = cv_rmsd_kabsch(
        jnp.asarray(pos_padded), jnp.asarray(ref_padded), atom_mask=mask
    )
    assert float(rmsd_val) == pytest.approx(float(rmsd_masked), abs=1e-5)

    grad_ad = np.asarray(
        jax.grad(lambda p: cv_rmsd_kabsch(p, jnp.asarray(ref_padded), atom_mask=mask))(
            jnp.asarray(pos_padded)
        )
    )
    grad_fd = _finite_diff_grad(
        lambda p: cv_rmsd_kabsch(p, jnp.asarray(ref_padded), atom_mask=mask),
        pos_padded,
        eps=1e-3,
    )
    np.testing.assert_allclose(grad_ad[:8], grad_fd[:8], rtol=2e-3, atol=1e-4)
    np.testing.assert_allclose(grad_ad[8], 0.0, atol=1e-7)

    # Verify zero @lapack / cusolver custom calls in lowered StableHLO
    with export.export_context():
        hlo_text = (
            jax.jit(jax.value_and_grad(lambda p: cv_rmsd_kabsch(p, jnp.asarray(ref))))
            .lower(jnp.asarray(pos_pert))
            .as_text()
            .lower()
        )
    assert "lapack" not in hlo_text
    assert "cusolver" not in hlo_text


def test_cv_plumed_kd_distance_ca1_exact_parity_and_padding_mask():
    """Verify CA-1 (Spike 7a): d1=d2=3.0 A with beta_nm=20.0 -> 200/(200/3 + ln 2) ~= 2.96914 A."""
    # Atoms 0-1 at distance 3.0 A, atoms 2-3 at distance 3.0 A, plus padded pair [0, 0]
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 8.0, 0.0],
            [42.0, -19.0, 7.0],  # unused padded atom
        ],
        dtype=np.float32,
    )
    pos_j = jnp.asarray(pos)

    pairs_unpadded = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    d_kd = cv_plumed_kd_distance(pos_j, pairs_unpadded, beta_nm=20.0)

    expected = 200.0 / (200.0 / 3.0 + math.log(2.0))
    assert float(d_kd) == pytest.approx(expected, abs=1e-5)
    assert abs(float(d_kd) - 3.0) < 0.031

    # Add padded pairs (including degenerate [0, 0] and unused atom [0, 4]) with pair_mask=False
    pairs_padded = jnp.array([[0, 1], [2, 3], [0, 0], [0, 4]], dtype=jnp.int32)
    pair_mask = jnp.array([True, True, False, False])

    d_kd_masked, grad_masked = jax.value_and_grad(
        lambda p: cv_plumed_kd_distance(p, pairs_padded, pair_mask=pair_mask, beta_nm=20.0)
    )(pos_j)
    d_kd_unpadded, grad_unpadded = jax.value_and_grad(
        lambda p: cv_plumed_kd_distance(p, pairs_unpadded, beta_nm=20.0)
    )(pos_j)

    assert float(d_kd_masked) == pytest.approx(float(d_kd_unpadded), abs=1e-6)
    np.testing.assert_allclose(np.asarray(grad_masked), np.asarray(grad_unpadded), atol=1e-6)
    # Padded atom 4 has identically zero gradient
    np.testing.assert_allclose(np.asarray(grad_masked[4]), 0.0, atol=1e-7)

    # Check gradient against finite differences on non-equal pair distances
    pos_asym = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.8, 0.4, -0.2],
            [1.0, 5.0, 0.0],
            [1.2, 8.3, -0.5],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    grad_ad = np.asarray(
        jax.grad(
            lambda p: cv_plumed_kd_distance(p, pairs_padded, pair_mask=pair_mask, beta_nm=20.0)
        )(jnp.asarray(pos_asym))
    )
    grad_fd = _finite_diff_grad(
        lambda p: cv_plumed_kd_distance(p, pairs_padded, pair_mask=pair_mask, beta_nm=20.0),
        pos_asym,
        eps=1e-3,
    )
    np.testing.assert_allclose(grad_ad[:4], grad_fd[:4], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(grad_ad[4], 0.0, atol=1e-7)


def test_umbrella_and_smd_bias_energy_and_grad():
    pos = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=np.float32)
    pos_j = jnp.asarray(pos)

    def e_umb(p):
        s = cv_distance(p, 0, 1)
        return umbrella_bias_energy(s, center=3.0, kappa=20.0)

    val_umb, grad_umb = jax.value_and_grad(e_umb)(pos_j)
    # 0.5 * 20.0 * (3.5 - 3.0)^2 = 2.5
    assert float(val_umb) == pytest.approx(2.5, abs=1e-5)
    grad_umb_fd = _finite_diff_grad(e_umb, pos, eps=1e-3)
    np.testing.assert_allclose(np.asarray(grad_umb), grad_umb_fd, rtol=1e-4, atol=1e-4)

    def e_smd(p):
        s = cv_distance(p, 0, 1)
        return smd_bias_energy(
            s, initial_center=2.5, kappa=10.0, pulling_velocity=0.1, time_or_step=5.0
        )

    val_smd, grad_smd = jax.value_and_grad(e_smd)(pos_j)
    # target = 2.5 + 0.1 * 5.0 = 3.0 -> 0.5 * 10.0 * (3.5 - 3.0)^2 = 1.25
    assert float(val_smd) == pytest.approx(1.25, abs=1e-5)
    grad_smd_fd = _finite_diff_grad(e_smd, pos, eps=1e-3)
    np.testing.assert_allclose(np.asarray(grad_smd), grad_smd_fd, rtol=1e-4, atol=1e-4)


def test_wt_metad_and_opes_metad_in_lax_scan_and_hill_decay():
    spec = BiasSpec(
        bias_mode="wt_metad",
        metad_height=0.5,
        metad_sigma=0.25,
        metad_bias_factor=10.0,
        metad_pace=2,
        grid_min=0.0,
        grid_max=10.0,
        grid_bin=201,
        temperature_k=300.0,
    )
    state0 = make_bias_state_from_controls(
        spec,
        grid_size=201,
        grid_min=0.0,
        grid_max=10.0,
        temperature_k=300.0,
    )

    s_deposit = jnp.array(5.0, dtype=jnp.float32)

    # Step 0 (0 % 2 == 0): deposits initial hill of height w0 = 0.5 at s = 5.0
    state1 = wt_metad_update_state(s_deposit, state0)
    v_after_1 = float(wt_metad_bias_energy(s_deposit, state1))
    assert v_after_1 == pytest.approx(0.5, abs=1e-5)
    assert int(state1.step) == 1

    # Step 1 (1 % 2 != 0): no hill deposited
    state2 = wt_metad_update_state(s_deposit, state1)
    v_after_2 = float(wt_metad_bias_energy(s_deposit, state2))
    assert v_after_2 == pytest.approx(0.5, abs=1e-5)
    assert int(state2.step) == 2

    # Step 2 (2 % 2 == 0): hill height decays by exp(-V / (kB * T * (gamma - 1)))
    state3 = wt_metad_update_state(s_deposit, state2)
    v_after_3 = float(wt_metad_bias_energy(s_deposit, state3))
    kb_delta_t = BOLTZMANN_KCAL * 300.0 * (10.0 - 1.0)
    expected_second_hill = 0.5 * math.exp(-0.5 / kb_delta_t)
    assert (v_after_3 - v_after_2) == pytest.approx(expected_second_hill, rel=1e-4)

    # Verify spline gradient wrt positions matches finite differences
    pos = np.array([[0.0, 0.0, 0.0], [5.15, 0.0, 0.0]], dtype=np.float32)
    grad_wt = np.asarray(
        jax.grad(lambda p: wt_metad_bias_energy(cv_distance(p, 0, 1), state3))(
            jnp.asarray(pos)
        )
    )
    grad_wt_fd = _finite_diff_grad(
        lambda p: wt_metad_bias_energy(cv_distance(p, 0, 1), state3),
        pos,
        eps=1e-3,
    )
    np.testing.assert_allclose(grad_wt, grad_wt_fd, rtol=2e-3, atol=1e-4)

    # Test OPES Metadynamics inside jax.lax.scan carrying BiasState
    opes_state0 = state0.replace(mode="opes_metad")
    cv_traj = jnp.array([4.8, 4.9, 5.0, 5.0, 5.1, 5.0], dtype=jnp.float32)

    def scan_body(st: BiasState, s_val: jax.Array):
        e_val = opes_metad_bias_energy(s_val, st)
        st_next = opes_metad_update_state(s_val, st)
        return st_next, e_val

    opes_final, energies = jax.lax.scan(scan_body, opes_state0, cv_traj)
    assert int(opes_final.step) == 6
    assert float(opes_final.opes_zed) > float(opes_state0.opes_zed)
    assert np.all(np.isfinite(np.asarray(energies)))
    # Bias energy at visited s=5.0 increases after KDE depositions
    assert float(opes_metad_bias_energy(5.0, opes_final)) > float(
        opes_metad_bias_energy(5.0, opes_state0)
    )

    # Verify OPES position gradient matches finite differences
    grad_opes = np.asarray(
        jax.grad(lambda p: opes_metad_bias_energy(cv_distance(p, 0, 1), opes_final))(
            jnp.asarray(pos)
        )
    )
    grad_opes_fd = _finite_diff_grad(
        lambda p: opes_metad_bias_energy(cv_distance(p, 0, 1), opes_final),
        pos,
        eps=1e-3,
    )
    np.testing.assert_allclose(grad_opes, grad_opes_fd, rtol=2e-3, atol=1e-4)
