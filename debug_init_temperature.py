#!/usr/bin/env python3
"""Diagnostic: Check initial temperatures for constrained vs unconstrained."""
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


def check_init_temperature():
    """Check initial temperature distribution."""
    jax.config.update("jax_enable_x64", True)

    n_waters = 2
    dt_fs = 1.0
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
    dof_rigid = _dof_rigid_tip3p_waters(n_waters)

    print("=== INITIALIZATION TEMPERATURE CHECK ===\n")
    print(f"System: {n_waters} waters ({n_atoms} atoms)")
    print(f"DOF (rigid): {dof_rigid}")
    print(f"Target kT: {kT:.6f} kcal/mol")
    print(f"Expected KE for T=300K: {0.5 * dof_rigid * kT:.6f} kcal/mol\n")

    # Test with constrained thermostat (multiple seeds to see variance)
    print("CONSTRAINED THERMOSTAT (project_ou_momentum_rigid=True):")
    init_s_const, _ = settle.settle_langevin(
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

    temps_const = []
    for i, s in enumerate(range(seed, seed + 3)):
        state = init_s_const(jax.random.PRNGKey(s), jnp.array(positions_a), mass=mass)
        ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
        t = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
        temps_const.append(t)
        print(f"  Seed {s}: KE={ke_r:.6f}, T={t:.1f} K")

    # Test with unconstrained thermostat
    print("\nUNCONSTRAINED THERMOSTAT (project_ou_momentum_rigid=False):")
    init_s_unconst, _ = settle.settle_langevin(
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

    temps_unconst = []
    for i, s in enumerate(range(seed, seed + 3)):
        state = init_s_unconst(jax.random.PRNGKey(s), jnp.array(positions_a), mass=mass)
        ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
        t = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
        temps_unconst.append(t)
        print(f"  Seed {s}: KE={ke_r:.6f}, T={t:.1f} K")

    print(f"\nMean T (constrained):   {np.mean(temps_const):.1f} K")
    print(f"Mean T (unconstrained): {np.mean(temps_unconst):.1f} K")
    print(f"\nExpected: 300 K")
    print(f"Difference (constrained):   {abs(np.mean(temps_const) - 300.0):.1f} K")
    print(f"Difference (unconstrained): {abs(np.mean(temps_unconst) - 300.0):.1f} K")


if __name__ == "__main__":
    check_init_temperature()
