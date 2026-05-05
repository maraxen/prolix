#!/usr/bin/env python3
"""Sprint B Step 1: NVT Temperature Discriminator Test

Measures T_observed at dt=0.25 fs with 64 waters to test H1 hypothesis:
- H1 (integration error): T_obs improves with smaller dt → H1 is correct
- H3 (force-side): T_obs unchanged at smaller dt → H3 is correct

Baseline (dt=0.5 fs, 8w): T_obs = 334.3 K (target 300K, +11% overshoot)
Discriminator (dt=0.25 fs, 64w): Should reveal if timing-dependent (H1) or not (H3)
"""
import jax
import jax.numpy as jnp
import numpy as np
from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water

def measure_temperature_at_dt(dt_fs: float, n_waters: int, seed: int = 7) -> tuple[float, float]:
    """Measure mean rigid-body temperature after burn-in.

    Args:
        dt_fs: Timestep in fs
        n_waters: Number of water molecules
        seed: Random seed

    Returns:
        (T_observable, T_target_kelvin)
    """
    jax.config.update("jax_enable_x64", True)

    temperature_k = 300.0
    gamma_ps = 1.0
    sim_ps = 100.0

    # Convert to steps
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = max(100, steps // 3)

    print(f"[DISCRIMINATOR] dt={dt_fs} fs, n_waters={n_waters}, total_steps={steps}, burn={burn}")

    # Setup
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    sys_dict = _prolix_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma_reduced,
        mass=mass, water_indices=water_indices, box=box_vec,
        remove_linear_com_momentum=False, project_ou_momentum_rigid=True,
        projection_site="post_o", settle_velocity_iters=10
    )
    apply_j = jax.jit(apply_s)

    dof_rigid = float(6 * n_waters - 3)
    state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

    # Burn-in
    print(f"[DISCRIMINATOR] Burn-in phase ({burn} steps)...")
    for step in range(burn):
        state = apply_j(state)
        if (step + 1) % 10000 == 0:
            print(f"  Burned {step + 1} steps...")

    # Production measurement
    print(f"[DISCRIMINATOR] Production phase ({steps - burn} steps)...")
    temps = []
    for step in range(burn, steps):
        state = apply_j(state)
        ke_r = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))
        temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
        temps.append(temp)

        if (step - burn + 1) % 10000 == 0:
            print(f"  Measured {step - burn + 1}/{steps - burn} steps...")

    mean_t_observable = float(np.mean(temps))
    return mean_t_observable, temperature_k

if __name__ == "__main__":
    print("=" * 70)
    print("Sprint B Step 1: NVT Temperature Discriminator Test")
    print("=" * 70)

    # Baseline: dt=0.5 fs, 8 waters (from previous test)
    print("\n[BASELINE] dt=0.5 fs, 8 waters (reproduced for reference)")
    t_obs_baseline, t_target = measure_temperature_at_dt(dt_fs=0.5, n_waters=8, seed=7)
    print(f"[BASELINE RESULT] T_obs={t_obs_baseline:.1f} K, target={t_target:.1f} K, delta={t_obs_baseline - t_target:+.1f} K")

    # Discriminator test 1: dt=0.25 fs, 64 waters (tests H1)
    print("\n" + "=" * 70)
    print("[DISCRIMINATOR 1] dt=0.25 fs, 64 waters (tests H1: integration error)")
    t_obs_disc1, _ = measure_temperature_at_dt(dt_fs=0.25, n_waters=64, seed=7)
    print(f"[DISCRIMINATOR 1 RESULT] T_obs={t_obs_disc1:.1f} K, target={t_target:.1f} K, delta={t_obs_disc1 - t_target:+.1f} K")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"Baseline (dt=0.5 fs, 8w):    T_obs = {t_obs_baseline:.1f} K ({t_obs_baseline - 300:.1f} K overshoot)")
    print(f"Discriminator (dt=0.25 fs, 64w): T_obs = {t_obs_disc1:.1f} K ({t_obs_disc1 - 300:.1f} K overshoot)")

    delta_t_change = t_obs_disc1 - t_obs_baseline
    print(f"\nTemperature change: {delta_t_change:+.1f} K")

    if abs(delta_t_change) < 5.0:  # Unchanged within noise
        print("✗ H1 FALSIFIED: Smaller timestep does NOT reduce temperature drift")
        print("→ Temperature error is NOT primarily timing-dependent (integration error)")
        print("→ H3 (force-side mechanism, e.g., PME grid) is more likely correct")
    elif delta_t_change < -20.0:  # Significant improvement
        print("✓ H1 CONFIRMED: Smaller timestep SIGNIFICANTLY reduces temperature drift")
        print("→ Temperature error IS primarily timing-dependent (integration error)")
        print("→ Solution: Smaller timesteps or constraint-aware thermostat")
    else:
        print("? INCONCLUSIVE: Intermediate change suggests mixed mechanism")
        print("→ Recommend PME grid scaling test (H3) or higher-order integrator")

    print("\n" + "=" * 70)
