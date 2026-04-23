#!/usr/bin/env python3
"""Debug: Check what's actually in the initialized momenta array."""
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


def debug_momenta_values():
    """Check the actual values in initialized momenta."""
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

    print("=== INITIALIZED MOMENTA DEBUG ===\n")
    print(f"System: {n_waters} waters, {n_atoms} atoms")
    print(f"Masses: {mass.reshape(-1)}\n")

    # Initialize with constrained thermostat
    init_s, _ = settle.settle_langevin(
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

    state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

    print("MOMENTUM ARRAY:")
    print(state.momentum)
    print(f"\nMomentum shape: {state.momentum.shape}")
    print(f"Momentum dtype: {state.momentum.dtype}")
    print(f"\nMomentum statistics:")
    print(f"  Min: {jnp.min(state.momentum):.6f}")
    print(f"  Max: {jnp.max(state.momentum):.6f}")
    print(f"  Mean: {jnp.mean(state.momentum):.6f}")
    print(f"  Std: {jnp.std(state.momentum):.6f}")

    # Per-atom KE
    print(f"\nPer-atom KE (p^2 / (2*m)):")
    per_atom_ke = 0.5 * jnp.sum(state.momentum**2, axis=1) / mass.reshape(-1)
    for i in range(n_atoms):
        atom_name = ["O", "H1", "H2"][i % 3]
        water_idx = i // 3
        print(f"  Atom {i} ({atom_name} in water {water_idx}): {per_atom_ke[i]:.6f}")
    print(f"  Total (summed): {jnp.sum(per_atom_ke):.6f}")

    # Rigid-body KE
    dof_rigid = 6 * n_waters - 3
    ke_rigid = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
    t_rigid = 2.0 * ke_rigid / (dof_rigid * BOLTZMANN_KCAL)

    print(f"\nRigid-body decomposition:")
    print(f"  Rigid KE: {ke_rigid:.6f} kcal/mol")
    print(f"  Rigid T:  {t_rigid:.1f} K")
    print(f"  Expected KE (300K): {0.5 * dof_rigid * kT:.6f}")
    print(f"  Ratio: {ke_rigid / (0.5 * dof_rigid * kT):.4f}")

    # Per-water analysis
    print(f"\nPer-water decomposition:")
    mom_w = state.momentum.reshape(n_waters, 3, 3)
    pos_w = state.position.reshape(n_waters, 3, 3)
    for w in range(n_waters):
        ke_w = float(rigid_tip3p_box_ke_kcal(pos_w[w:w+1].reshape(1, 3, 3),
                                             mom_w[w:w+1].reshape(1, 3, 3),
                                             mass[3*w:3*w+3].reshape(3), 1))
        t_w = 2.0 * ke_w / (6.0 * BOLTZMANN_KCAL)
        print(f"  Water {w}: KE={ke_w:.6f}, T={t_w:.1f}K")


if __name__ == "__main__":
    debug_momenta_values()
