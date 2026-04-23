#!/usr/bin/env python3
"""Diagnostic: Temperature evolution in constrained thermostat."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _prolix_params_pure_water,
)


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
    return float(6 * n_waters - 3)


def debug_temperature_evolution():
    """Run a short simulation and track temperature step-by-step."""
    jax.config.update("jax_enable_x64", True)

    n_waters = 2
    dt_fs = 1.0
    sim_ps = 5.0  # Only 5 ps for debugging
    steps = int(sim_ps * 1000.0 / dt_fs)
    seed = 601
    temperature_k = 300.0
    gamma_ps = 1.0

    # Setup
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

    sys_dict = _prolix_params_pure_water(n_waters)
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

    # Test with constrained thermostat
    print("\n=== CONSTRAINED THERMOSTAT (project_ou_momentum_rigid=True) ===")
    init_s, apply_s = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_reduced,
        mass=mass,
        water_indices=water_indices,
        box=box_vec,
        remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
        settle_velocity_iters=10,
    )
    apply_j = jax.jit(apply_s)
    dof_rigid = _dof_rigid_tip3p_waters(n_waters)

    state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

    # Check initial temperature
    ke_r_init = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
    t_init = 2.0 * ke_r_init / (dof_rigid * BOLTZMANN_KCAL)
    print(f"Initial KE: {ke_r_init:.6f} kcal/mol, T: {t_init:.1f} K")
    print(f"Expected KE for T=300K: {0.5 * dof_rigid * BOLTZMANN_KCAL:.6f} kcal/mol")

    temps_constrained = [t_init]
    for step in range(steps):
        state = apply_j(state)
        ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
        temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
        temps_constrained.append(temp)

    print(f"Step 0: T = {temps_constrained[0]:.1f} K")
    for step in [steps // 4, steps // 2, 3 * steps // 4, steps]:
        if step < len(temps_constrained):
            print(f"Step {step}: T = {temps_constrained[step]:.1f} K")

    mean_t_constrained = float(np.mean(temps_constrained))
    print(f"Mean T (all steps): {mean_t_constrained:.1f} K")
    print(f"Mean T (excluding first 20%): {float(np.mean(temps_constrained[steps//5:])):.1f} K")

    # Test with unconstrained thermostat for comparison
    print("\n=== UNCONSTRAINED THERMOSTAT (project_ou_momentum_rigid=False) ===")
    init_s2, apply_s2 = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_reduced,
        mass=mass,
        water_indices=water_indices,
        box=box_vec,
        remove_linear_com_momentum=False,
        project_ou_momentum_rigid=False,
        projection_site="post_o",
        settle_velocity_iters=10,
    )
    apply_j2 = jax.jit(apply_s2)

    state2 = init_s2(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

    ke_r_init2 = float(rigid_tip3p_box_ke_kcal(state2.position, state2.momentum, state2.mass, n_waters))
    t_init2 = 2.0 * ke_r_init2 / (dof_rigid * BOLTZMANN_KCAL)
    print(f"Initial KE: {ke_r_init2:.6f} kcal/mol, T: {t_init2:.1f} K")

    temps_unconstrained = [t_init2]
    for step in range(steps):
        state2 = apply_j2(state2)
        ke_r = float(rigid_tip3p_box_ke_kcal(state2.position, state2.momentum, state2.mass, n_waters))
        temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
        temps_unconstrained.append(temp)

    print(f"Step 0: T = {temps_unconstrained[0]:.1f} K")
    for step in [steps // 4, steps // 2, 3 * steps // 4, steps]:
        if step < len(temps_unconstrained):
            print(f"Step {step}: T = {temps_unconstrained[step]:.1f} K")

    mean_t_unconstrained = float(np.mean(temps_unconstrained))
    print(f"Mean T (all steps): {mean_t_unconstrained:.1f} K")
    print(f"Mean T (excluding first 20%): {float(np.mean(temps_unconstrained[steps//5:])):.1f} K")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Constrained initial T:  {temps_constrained[0]:.1f} K")
    print(f"Unconstrained initial T: {temps_unconstrained[0]:.1f} K")
    print(f"Constrained mean T:     {mean_t_constrained:.1f} K")
    print(f"Unconstrained mean T:   {mean_t_unconstrained:.1f} K")
    print(f"Difference: {abs(mean_t_constrained - mean_t_unconstrained):.1f} K")


if __name__ == "__main__":
    debug_temperature_evolution()
