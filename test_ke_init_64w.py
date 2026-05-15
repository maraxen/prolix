#!/usr/bin/env python3
"""Debug script with 64 waters to reproduce the exact issue."""

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics import pbc, settle, system, pressure, stress
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.units import BAR_PER_AKMA_PRESSURE
from prolix.physics.simulate import NVTLangevinState, NPTState
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _proxide_params_pure_water,
)


def main():
    jax.config.update("jax_enable_x64", True)

    # Match the test_npt_20ps_liquid_water setup
    n_waters = 64
    dt_fs = 0.5
    temperature_k = 300.0
    pressure_bar = 1.0
    gamma_ps = 1.0
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    ndof = float(6 * n_waters - 3)
    n_atoms = n_waters * 3

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=3.1)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    sys_dict = _proxide_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=32,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        strict_parameterization=False,
    )

    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(n_protein_atoms=0, n_waters=n_waters)

    # === Phase 1: NVT equilibration ===
    init_nvt, apply_nvt = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_ps,
        mass=mass,
        water_indices=water_indices,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    apply_nvt_j = jax.jit(apply_nvt)
    nvt_state = init_nvt(jax.random.PRNGKey(7), jnp.array(positions_a), mass=mass, box=box_vec)

    print("=== Initial NVT State ===")
    ke_total_nvt_init = rigid_tip3p_box_ke_kcal(
        nvt_state.positions, nvt_state.momentum, nvt_state.mass, n_waters
    )
    T_nvt_init = float(2.0 * ke_total_nvt_init / (BOLTZMANN_KCAL * ndof))
    print(f"Temperature (NVT init): {T_nvt_init:.2f} K")

    # Run NVT for a few steps
    nvt_steps = 100
    for i in range(nvt_steps):
        nvt_state = apply_nvt_j(nvt_state, box=box_vec)

    print(f"\n=== After {nvt_steps} NVT steps ===")
    ke_total_nvt = rigid_tip3p_box_ke_kcal(
        nvt_state.positions, nvt_state.momentum, nvt_state.mass, n_waters
    )
    T_nvt = float(2.0 * ke_total_nvt / (BOLTZMANN_KCAL * ndof))
    print(f"Temperature (NVT after {nvt_steps} steps): {T_nvt:.2f} K")

    # === Phase 2: Handoff to NPT ===
    print("\n=== Creating NPT State from NVT ===")

    # This is what the test does: create NPTState manually, then init_npt
    npt_state_before_init = NPTState(
        positions=nvt_state.positions,
        momentum=nvt_state.momentum,
        force=nvt_state.force,
        mass=nvt_state.mass,
        rng=nvt_state.rng,
        box=box_vec,
    )

    print(f"Momentum from NVT (pre-init_npt):")
    print(f"  Min/max: {npt_state_before_init.momentum.min():.6e} / {npt_state_before_init.momentum.max():.6e}")

    ke_total_before = rigid_tip3p_box_ke_kcal(
        npt_state_before_init.positions,
        npt_state_before_init.momentum,
        npt_state_before_init.mass,
        n_waters,
    )
    T_before = float(2.0 * ke_total_before / (BOLTZMANN_KCAL * ndof))
    print(f"Temperature (before init_npt): {T_before:.2f} K")

    # Initialize NPT
    init_npt, apply_npt = settle.settle_csvr_npt(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        target_pressure_bar=pressure_bar,
        tau_barostat_akma=2000.0,
        tau_thermostat_akma=2000.0,
        mass=mass,
        water_indices=water_indices,
        box_init=box_vec,
    )

    # This is the key step: init_npt() is called on the NPT state
    npt_state = init_npt(
        npt_state_before_init.rng,
        npt_state_before_init.positions,
        mass=mass,
        box=box_vec,
    )

    print(f"\n=== After init_npt() ===")
    print(f"Momentum shape: {npt_state.momentum.shape}")
    print(f"Momentum min/max: {npt_state.momentum.min():.6e} / {npt_state.momentum.max():.6e}")

    ke_total_after_init = rigid_tip3p_box_ke_kcal(
        npt_state.positions, npt_state.momentum, npt_state.mass, n_waters
    )
    T_after_init = float(2.0 * ke_total_after_init / (BOLTZMANN_KCAL * ndof))
    print(f"Temperature (after init_npt): {T_after_init:.2f} K")
    print(f"Temperature ratio (init_npt / NVT): {T_after_init / T_nvt:.1f}x")

    # Now take first NPT step
    apply_npt_j = jax.jit(apply_npt)
    npt_state_step1 = apply_npt_j(npt_state, box=npt_state.box)

    print(f"\n=== After 1 NPT step ===")
    ke_total_step1 = rigid_tip3p_box_ke_kcal(
        npt_state_step1.positions, npt_state_step1.momentum, npt_state_step1.mass, n_waters
    )
    T_step1 = float(2.0 * ke_total_step1 / (BOLTZMANN_KCAL * ndof))
    print(f"Temperature: {T_step1:.2f} K")
    print(f"Temperature ratio (step 1 / init): {T_step1 / T_after_init:.1f}x")

    # Take a few more steps
    for i in range(5):
        npt_state_step1 = apply_npt_j(npt_state_step1, box=npt_state_step1.box)

        ke_total = rigid_tip3p_box_ke_kcal(
            npt_state_step1.positions, npt_state_step1.momentum, npt_state_step1.mass, n_waters
        )
        T = float(2.0 * ke_total / (BOLTZMANN_KCAL * ndof))
        print(f"Step {i+2}: T = {T:.2f} K")


if __name__ == "__main__":
    main()
