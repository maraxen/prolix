#!/usr/bin/env python3
"""Debug script to diagnose NPT KE initialization bug."""

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics import pbc, settle, system, pressure, stress
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.units import BAR_PER_AKMA_PRESSURE
from prolix.physics.simulate import NPTState
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _proxide_params_pure_water,
)


def main():
    jax.config.update("jax_enable_x64", True)

    n_waters = 4
    dt_fs = 0.5
    temperature_k = 300.0
    pressure_bar = 1.0

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL

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

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

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

    # Initialize state
    state = init_npt(jax.random.PRNGKey(42), jnp.array(positions_a), mass=mass, box=box_vec)

    print("=== Initial NPT State ===")
    print(f"Momentum shape: {state.momentum.shape}")
    print(f"Momentum min/max: {state.momentum.min():.6e} / {state.momentum.max():.6e}")
    print(f"Momentum mean: {state.momentum.mean():.6e}")

    # Compute kinetic energy from initial momenta
    ndof = float(6 * n_waters - 3)
    ke_total = rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters)
    T_k = float(2.0 * ke_total / (BOLTZMANN_KCAL * ndof))

    print(f"KE total: {ke_total:.6e} kcal/mol")
    print(f"Temperature (from KE): {T_k:.2f} K")
    print(f"Target temperature: {temperature_k:.2f} K")
    print(f"Temperature ratio: {T_k / temperature_k:.1f}x")

    # Check individual velocity components
    velocity = state.momentum / state.mass
    ke_from_velocity = 0.5 * jnp.sum(state.mass * velocity**2)
    print(f"\nAlternative KE calc (0.5*m*v^2): {ke_from_velocity:.6e} kcal/mol")

    # Now apply one step
    print("\n=== After 1 NPT Step ===")
    state = jax.jit(apply_npt)(state, box=state.box)

    ke_total_step1 = rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters)
    T_k_step1 = float(2.0 * ke_total_step1 / (BOLTZMANN_KCAL * ndof))

    print(f"KE total: {ke_total_step1:.6e} kcal/mol")
    print(f"Temperature: {T_k_step1:.2f} K")
    print(f"Temperature ratio: {T_k_step1 / temperature_k:.1f}x")


if __name__ == "__main__":
    main()
