"""Tests for FR-8: Beutler (1-lambda)^2 soft-core minimization and unhalved OpenMM restraints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_md import space
import pytest

from prolix.physics import pbc
from prolix.physics.softcore_minimizer import (
    DEFAULT_NAURMALADE_WINDOWS,
    SoftcoreWindow,
    compute_beutler_softcore_lj_energy,
    compute_positional_restraint_energy,
    compute_softcore_coulomb_energy,
    k_kj_nm2_to_kcal_a2,
    run_staged_softcore_minimization,
    select_restraint_mask,
)


def test_beutler_softcore_lj_squared_vs_linear_exponent():
    """Verify exact s=1.125 (exponent=2) vs s=1.25 (exponent=1) at lambda=0.5, alpha=0.5, r=sigma."""
    sigma = 3.4
    epsilon = 0.25
    lam = 0.5
    alpha = 0.5

    positions = jnp.array([[0.0, 0.0, 0.0], [sigma, 0.0, 0.0]], dtype=jnp.float32)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    sigmas = jnp.array([sigma, sigma], dtype=jnp.float32)
    epsilons = jnp.array([epsilon, epsilon], dtype=jnp.float32)
    mask = jnp.array([1.0], dtype=jnp.float32)
    displacement_fn, _ = space.free()

    # Squared exponent p=2 (naurmalade default): s = 0.5 * (1 - 0.5)^2 + 1^6 = 1.125
    s_sq = 1.125
    expected_sq = 4.0 * epsilon * lam * (1.0 / (s_sq ** 2) - 1.0 / s_sq)
    e_sq = compute_beutler_softcore_lj_energy(
        positions,
        pairs,
        sigmas,
        epsilons,
        mask,
        displacement_fn,
        lambda_lj=lam,
        softcore_alpha=alpha,
        softcore_exponent=2,
    )
    assert float(e_sq) == pytest.approx(expected_sq, rel=1e-6)

    # Linear exponent p=1: s = 0.5 * (1 - 0.5)^1 + 1^6 = 1.25
    s_lin = 1.25
    expected_lin = 4.0 * epsilon * lam * (1.0 / (s_lin ** 2) - 1.0 / s_lin)
    e_lin = compute_beutler_softcore_lj_energy(
        positions,
        pairs,
        sigmas,
        epsilons,
        mask,
        displacement_fn,
        lambda_lj=lam,
        softcore_alpha=alpha,
        softcore_exponent=1,
    )
    assert float(e_lin) == pytest.approx(expected_lin, rel=1e-6)
    assert abs(float(e_sq) - float(e_lin)) > 1e-2


def test_beutler_softcore_lj_finite_at_zero_distance_and_standard_at_lambda_one():
    """Verify finite energy/forces as r -> 0 for lambda_lj < 1 and exact 12-6 LJ at lambda_lj = 1."""
    sigma = 3.0
    epsilon = 0.2
    sigmas = jnp.array([sigma, sigma], dtype=jnp.float32)
    epsilons = jnp.array([epsilon, epsilon], dtype=jnp.float32)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    mask = jnp.array([1.0], dtype=jnp.float32)
    displacement_fn, _ = space.free()

    # Coincident atoms (r = 0) at lambda_lj = 0.5
    pos_zero = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)
    energy_fn = lambda pos: compute_beutler_softcore_lj_energy(
        pos, pairs, sigmas, epsilons, mask, displacement_fn, lambda_lj=0.5, softcore_alpha=0.5, softcore_exponent=2
    )
    e_val, grad_val = jax.value_and_grad(energy_fn)(pos_zero)
    # At r = 0, s = 0.5 * 0.25 = 0.125 -> 1/s^2 - 1/s = 64 - 8 = 56
    expected_zero = 4.0 * epsilon * 0.5 * (1.0 / (0.125 ** 2) - 1.0 / 0.125)
    assert jnp.isfinite(e_val)
    assert float(e_val) == pytest.approx(expected_zero, rel=1e-5)
    assert jnp.all(jnp.isfinite(grad_val))

    # Severe steric clash r = 0.1 A at lambda_lj = 0.5 has finite repulsive force
    pos_clash = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=jnp.float32)
    e_clash, grad_clash = jax.value_and_grad(energy_fn)(pos_clash)
    assert jnp.isfinite(e_clash)
    assert jnp.all(jnp.isfinite(grad_clash))

    # Standard 12-6 LJ recovery at lambda_lj = 1.0
    r_test = 3.5
    pos_std = jnp.array([[0.0, 0.0, 0.0], [r_test, 0.0, 0.0]], dtype=jnp.float32)
    e_lam1 = compute_beutler_softcore_lj_energy(
        pos_std, pairs, sigmas, epsilons, mask, displacement_fn, lambda_lj=1.0
    )
    sr6 = (sigma / r_test) ** 6
    expected_lj = 4.0 * epsilon * (sr6 ** 2 - sr6)
    assert float(e_lam1) == pytest.approx(expected_lj, rel=1e-5)


def test_softcore_coulomb_disabled_by_default_and_active_when_enabled():
    """Verify soft_coulomb=False yields 0 and soft_coulomb=True follows alpha_q=0.01 A^2 formula."""
    pos = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    charges = jnp.array([0.5, -0.5], dtype=jnp.float32)
    mask = jnp.array([1.0], dtype=jnp.float32)
    displacement_fn, _ = space.free()

    e_off = compute_softcore_coulomb_energy(
        pos, pairs, charges, mask, displacement_fn, lambda_elec=0.5, soft_coulomb=False
    )
    assert float(e_off) == 0.0

    e_on = compute_softcore_coulomb_energy(
        pos,
        pairs,
        charges,
        mask,
        displacement_fn,
        lambda_elec=0.5,
        soft_coulomb=True,
        alpha_q_angstrom2=0.01,
        coulomb_constant=332.0637,
    )
    expected = 0.5 * 332.0637 * (0.5 * -0.5) / float(jnp.sqrt(4.0 + 0.01 * (0.5 ** 2)))
    assert float(e_on) == pytest.approx(expected, rel=1e-6)


def test_positional_restraint_openmm_unhalved_flat_bottom_and_pbc():
    """Verify OpenMM unhalved k*dr^2 convention, F = -2*k*dr, flat-bottom radius, and PBC wrapping."""
    k_rest = k_kj_nm2_to_kcal_a2(1000.0)
    assert k_rest == pytest.approx(1000.0 / 418.4, rel=1e-7)

    displacement_fn, _ = space.free()
    ref_pos = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
    pos = jnp.array([[1.5, -0.5, 2.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
    mask = jnp.array([1.0, 1.0], dtype=jnp.float32)

    energy_fn = lambda r: compute_positional_restraint_energy(
        r, ref_pos, mask, displacement_fn, k_kcal_mol_a2=k_rest
    )
    e_val, grad_val = jax.value_and_grad(energy_fn)(pos)
    dr0 = pos[0] - ref_pos[0]
    expected_e = k_rest * float(jnp.sum(dr0 ** 2))
    assert float(e_val) == pytest.approx(expected_e, rel=1e-6)

    # Force F = -grad = -2 * k_rest * dr (unhalved OpenMM convention)
    force = -grad_val
    expected_force_0 = -2.0 * k_rest * dr0
    assert jnp.allclose(force[0], expected_force_0, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(force[1], jnp.zeros(3), atol=1e-6)

    # Flat-bottom restraint: zero energy and zero force inside r_fb
    pos_inside = jnp.array([[0.8, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
    fb_fn = lambda r: compute_positional_restraint_energy(
        r, ref_pos, mask, displacement_fn, k_kcal_mol_a2=k_rest, flat_bottom_radius_angstrom=1.0
    )
    e_in, grad_in = jax.value_and_grad(fb_fn)(pos_inside)
    assert float(e_in) == pytest.approx(0.0, abs=1e-7)
    assert jnp.allclose(grad_in, jnp.zeros_like(grad_in), atol=1e-6)

    pos_outside = jnp.array([[2.5, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)
    e_out = fb_fn(pos_outside)
    assert float(e_out) == pytest.approx(k_rest * ((2.5 - 1.0) ** 2), rel=1e-5)

    # PBC minimum-image wrapping across a 30 A cubic box boundary
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)
    pbc_disp_fn, _ = pbc.create_periodic_space(box)
    ref_pbc = jnp.array([[1.0, 15.0, 15.0]], dtype=jnp.float32)
    # Atom wrapped to x = 29.5 A is only 1.5 A away across the periodic boundary (not 28.5 A)
    pos_pbc = jnp.array([[29.5, 15.0, 15.0]], dtype=jnp.float32)
    mask_pbc = jnp.array([1.0], dtype=jnp.float32)
    e_pbc = compute_positional_restraint_energy(
        pos_pbc, ref_pbc, mask_pbc, pbc_disp_fn, k_kcal_mol_a2=k_rest
    )
    assert float(e_pbc) == pytest.approx(k_rest * (1.5 ** 2), rel=1e-5)


def test_select_restraint_mask_modes_and_solvent_exclusion():
    """Verify backbone, ca, heavy, and none selections and automatic solvent/ion exclusion."""
    atom_names = ["N", "CA", "C", "O", "CB", "HB1", "O", "H1", "NA", "CL"]
    residue_names = ["ALA", "ALA", "ALA", "ALA", "ALA", "ALA", "HOH", "HOH", "NA", "CL"]

    bb_mask = select_restraint_mask(atom_names, residue_names, selection="backbone")
    assert bb_mask.tolist() == [True, True, True, True, False, False, False, False, False, False]

    ca_mask = select_restraint_mask(atom_names, residue_names, selection="ca")
    assert ca_mask.tolist() == [False, True, False, False, False, False, False, False, False, False]

    heavy_mask = select_restraint_mask(atom_names, residue_names, selection="heavy")
    assert heavy_mask.tolist() == [True, True, True, True, True, False, False, False, False, False]

    none_mask = select_restraint_mask(atom_names, residue_names, selection="none")
    assert none_mask.tolist() == [False] * len(atom_names)


def test_staged_softcore_minimization_resolves_severe_steric_clash():
    """Verify staged soft-core sweep resolves an r0 = 0.6 A clash without NaN gradients."""
    sigma = 3.4
    epsilon = 0.2
    sigmas = jnp.array([sigma, sigma], dtype=jnp.float32)
    epsilons = jnp.array([epsilon, epsilon], dtype=jnp.float32)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    pair_mask = jnp.array([1.0], dtype=jnp.float32)
    displacement_fn, _ = space.free()

    # Start with severe steric overlap at r0 = 0.6 A (where standard 12-6 LJ energy is > 10^8 kcal/mol)
    init_pos = jnp.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=jnp.float32)
    ref_pos = init_pos
    # Restrain atom 0 weakly, leave atom 1 free to relax
    restraint_mask = jnp.array([1.0, 0.0], dtype=jnp.float32)

    stage_flags: list[tuple[bool, bool]] = []

    def stage_energy_fn(
        positions,
        *,
        lambda_lj,
        lambda_elec,
        use_pme_reciprocal=False,
        use_dispersion_correction=False,
        softcore_alpha=0.5,
        softcore_exponent=2,
        **kwargs,
    ):
        stage_flags.append((bool(use_pme_reciprocal), bool(use_dispersion_correction)))
        return compute_beutler_softcore_lj_energy(
            positions,
            pairs,
            sigmas,
            epsilons,
            pair_mask,
            displacement_fn,
            lambda_lj=lambda_lj,
            softcore_alpha=softcore_alpha,
            softcore_exponent=softcore_exponent,
        )

    windows = tuple(
        SoftcoreWindow(
            lambda_lj=w.lambda_lj,
            lambda_elec=w.lambda_elec,
            restraint_scale=w.restraint_scale,
            max_iterations=60,
        )
        for w in DEFAULT_NAURMALADE_WINDOWS
    )

    final_pos, history = run_staged_softcore_minimization(
        stage_energy_fn,
        init_pos,
        ref_pos,
        restraint_mask,
        displacement_fn,
        windows=windows,
        step_size=5e-3,
        max_grad_norm=10.0,
    )

    assert len(history) == 5
    assert jnp.all(jnp.isfinite(final_pos))
    # Soft-core stages (first 4 windows) must have PME and dispersion tail disabled
    for stage_info in history[:-1]:
        assert stage_info["pme_enabled"] == 0.0
        assert stage_info["dispersion_tail_enabled"] == 0.0
        assert stage_info["lambda_elec"] == 0.0
    # Final cleanup stage (lambda_lj == 1.0) enables full Hamiltonian
    assert history[-1]["lambda_lj"] == 1.0
    assert history[-1]["lambda_elec"] == 1.0
    assert history[-1]["pme_enabled"] == 1.0
    assert history[-1]["dispersion_tail_enabled"] == 1.0

    # Final interatomic distance must have relaxed out of the steric clash toward r_min = 2^(1/6) * sigma ~ 3.82 A
    final_dist = float(jnp.linalg.norm(final_pos[1] - final_pos[0]))
    assert final_dist > 3.0
    assert history[-1]["final_energy"] < 0.0
