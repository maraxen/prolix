#!/usr/bin/env python3
"""Test: Full 100ps with fixed O-step (no momentum projection)."""
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


def test_full_100ps():
    """Run full 100ps test with fixed O-step."""
    jax.config.update("jax_enable_x64", True)

    n_waters = 2
    dt_fs = 1.0
    sim_ps = 100.0
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = max(100, steps // 3)
    seed = 601

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = 300.0 * BOLTZMANN_KCAL
    gamma_reduced = 1.0 * float(AKMA_TIME_UNIT_FS) * 1e-3

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

    ke_init = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
    t_init = 2.0 * ke_init / (dof_rigid * BOLTZMANN_KCAL)

    print(f"Testing FIXED O-STEP (100ps, dt=1fs, seed={seed})")
    print(f"Init T: {t_init:.1f} K")
    print(f"Burn-in: {burn} steps\n")

    temps: list[float] = []
    for step in range(steps):
        state = apply_j(state)
        if step >= burn:
            ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
            temp = 2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL)
            temps.append(temp)
            if step % 10000 == 0 and step > burn:
                print(f"Step {step}: T = {temp:.1f} K")

    mean_t = float(np.mean(temps)) if temps else float("nan")
    error = abs(mean_t - 300.0)
    passes = error < 5.0

    print(f"\n=== RESULT ===")
    print(f"Mean T (post-burn): {mean_t:.1f} K")
    print(f"Error: {error:.1f} K")
    print(f"Status: {'✅ PASS' if passes else '❌ FAIL'}")

    return passes


if __name__ == "__main__":
    test_full_100ps()
