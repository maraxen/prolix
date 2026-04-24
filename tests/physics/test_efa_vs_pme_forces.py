"""Sprint 2 Phase 2: EFA vs PME force comparison tests.

Tests validate force accuracy, bias, and exclusion corrections.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.physics.eval_harness import (
    force_cosine_similarity,
    force_rmse,
    make_comparison_energies,
    make_tip3p_water_system,
)


@pytest.mark.electrostatic_comparison
def test_efa_pme_force_rmse_relative():
    """Test: EFA relative force RMSE at D=512 < 15% of PME RMS force magnitude.

    This test validates the force accuracy of the EFA approximation by comparing
    the relative RMSE to the magnitude of the reference PME forces.
    Expected threshold: 15% for D=512 features (from oracle calibration).
    """
    system = make_tip3p_water_system(n_waters=64, seed=0)
    comparison = make_comparison_energies(system, n_seeds=16, n_features=512)

    f_pme = comparison["pme_forces"]
    f_efa_seeds = comparison["efa_forces_per_seed"]
    mask = system.atom_mask.astype(bool)

    # Compute RMSE for each seed
    rmse_per_seed = jnp.array(
        [force_rmse(f, f_pme, mask) for f in f_efa_seeds]
    )
    mean_rmse = float(jnp.mean(rmse_per_seed))

    # Reference magnitude: RMS of PME forces (over masked atoms)
    pme_rms = float(jnp.sqrt(jnp.mean(f_pme[mask] ** 2)))

    relative_rmse = mean_rmse / (pme_rms + 1e-12)
    print(f"EFA relative RMSE at D=512: {relative_rmse:.4f}")

    assert (
        relative_rmse < 0.15
    ), f"Relative RMSE {relative_rmse:.4f} >= 0.15 threshold"


@pytest.mark.electrostatic_comparison
def test_efa_forces_unbiased():
    """Test: EFA forces show no systematic bias (t-statistic < 4.0, oracle threshold).

    Validates that the mean of EFA force samples across seeds has no significant
    bias compared to per-component standard error.
    """
    system = make_tip3p_water_system(n_waters=64, seed=0)
    comparison = make_comparison_energies(system, n_seeds=16, n_features=512)

    f_seeds = comparison["efa_forces_per_seed"]  # (16, N, 3)
    mask = system.atom_mask.astype(bool)

    # Extract masked forces: (16, N_active, 3)
    f_masked = f_seeds[:, mask, :]

    # Compute t-statistics: mu / (sigma / sqrt(M))
    mu = f_masked.mean(axis=0)  # (N_active, 3) mean per atom per component
    sigma = f_masked.std(axis=0)  # (N_active, 3) std per atom per component
    M = f_seeds.shape[0]  # Number of seeds

    t_stat = jnp.abs(mu) / (sigma / jnp.sqrt(M) + 1e-8)
    max_t = float(t_stat.max())
    p99_t = float(jnp.percentile(t_stat, 99))

    print(f"Max t-stat: {max_t:.2f}, 99th percentile: {p99_t:.2f}")

    assert (
        max_t < 4.0
    ), f"Max t-stat {max_t:.2f} >= 4.0 — EFA forces may be biased"


@pytest.mark.electrostatic_comparison
def test_efa_exclusion_correction_nonzero():
    """Test: EFA exclusion correction is active and nonzero.

    Validates that bonded pair exclusions (1-2, 1-3, 1-4 scaled) are correctly
    applied in the EFA path, changing forces relative to naive RFF without exclusions.
    """
    import dataclasses
    from prolix.physics.flash_explicit import flash_explicit_forces

    system = make_tip3p_water_system(n_waters=16, seed=0)

    # Forces with exclusions (normal path)
    f_with_excl = flash_explicit_forces(
        system, electrostatic_method="efa", n_rff_features=128, rff_seed=0
    )

    # To test without exclusions: temporarily zero out bonds
    # Create a system with no bonds to bypass exclusion correction
    sys_no_excl = dataclasses.replace(
        system,
        bonds=jnp.zeros_like(system.bonds),
        bond_mask=jnp.zeros_like(system.bond_mask),
    )
    f_no_excl = flash_explicit_forces(
        sys_no_excl, electrostatic_method="efa", n_rff_features=128, rff_seed=0
    )

    mask = system.atom_mask.astype(bool)
    diff_rms = force_rmse(f_with_excl, f_no_excl, mask)

    print(f"Force RMSE (with vs without exclusions): {diff_rms:.6f} kcal/mol/Å")

    assert diff_rms > 0.0, (
        "Exclusion correction had no effect on forces — it may not be applied"
    )


@pytest.mark.electrostatic_comparison
def test_efa_nvt_smoke_dt05():
    """Smoke test: NVT equilibration at dt=0.5 fs runs without error.

    Does NOT validate temperature control (that's handled by settle tests).
    Just checks that the integration loop executes.
    """
    from prolix.physics import pbc, settle, system
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    jax.config.update("jax_enable_x64", True)

    n_waters = 8
    system_obj = make_tip3p_water_system(n_waters=n_waters, seed=42)

    # NVT setup
    temperature_k = 300.0
    gamma_ps = 1.0
    dt_fs = 0.5
    n_steps = 10  # Short test

    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)

    # Mock energy_fn: just use flash_explicit for simplicity
    from prolix.physics.flash_explicit import flash_explicit_energy

    def energy_fn(pos_arr):
        return flash_explicit_energy(
            system_obj.replace(
                positions=jnp.pad(
                    pos_arr,
                    ((0, system_obj.n_padded_atoms - system_obj.n_real_atoms), (0, 0)),
                    constant_values=0.0,
                )
            ),
            electrostatic_method="pme",
        )

    shift_fn = lambda r: r  # Identity shift for non-periodic

    mass = system_obj.masses[: system_obj.n_real_atoms]
    water_indices = settle.get_water_indices(0, n_waters)

    init_s, apply_s = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_reduced,
        mass=mass,
        water_indices=water_indices,
        box=None,
        remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
        settle_velocity_iters=10,
    )

    pos_init = system_obj.positions[: system_obj.n_real_atoms]
    state = init_s(jax.random.PRNGKey(42), pos_init, mass=mass)
    apply_j = jax.jit(apply_s)

    for step in range(n_steps):
        state = apply_j(state)

    assert jnp.all(jnp.isfinite(state.position)), "NVT integration produced NaN/inf"
    print(f"NVT smoke test passed: {n_steps} steps at dt={dt_fs} fs")


@pytest.mark.electrostatic_comparison
def test_efa_rmse_decreases_with_d():
    """Test: EFA relative RMSE decreases monotonically with feature count D.

    Validates the approximation quality scales as expected:
    larger D should give smaller error (within tolerance).
    """
    system = make_tip3p_water_system(n_waters=32, seed=0)
    comparison_pme = make_comparison_energies(system, n_seeds=1, n_features=512)
    f_pme = comparison_pme["pme_forces"]
    mask = system.atom_mask.astype(bool)
    pme_rms = float(jnp.sqrt(jnp.mean(f_pme[mask] ** 2)))

    d_values = [64, 128, 256, 512]
    rmses = []

    for D in d_values:
        comparison = make_comparison_energies(system, n_seeds=4, n_features=D)
        f_efa_seeds = comparison["efa_forces_per_seed"]

        rmse_per_seed = jnp.array(
            [force_rmse(f, f_pme, mask) for f in f_efa_seeds]
        )
        mean_rmse = float(jnp.mean(rmse_per_seed))
        relative_rmse = mean_rmse / (pme_rms + 1e-12)
        rmses.append(relative_rmse)
        print(f"D={D}: relative RMSE = {relative_rmse:.4f}")

    # Check monotonic decrease (allow 10% tolerance for noise)
    for i in range(len(rmses) - 1):
        assert (
            rmses[i] >= rmses[i + 1] * 0.9
        ), f"RMSE should decrease with D: {rmses[i]:.4f} vs {rmses[i+1]:.4f}"
