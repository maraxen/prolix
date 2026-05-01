#!/usr/bin/env python3
"""Sprint B Discriminator: high-γ test.

If the offset is from slow equilibration (10Å grid → hot initial state,
thermostat too weak to cool fast), then γ=50/ps should drive the system to
300K much faster than γ=1/ps.

If the offset persists at γ=50/ps, it is a genuine steady-state bias
(thermostat formula error or KE estimator bug), not an initialization artifact.
"""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.settle_langevin_potential_propagator import settle_langevin_potential_cached_step
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water

n_waters = 64
dt_fs = 0.5
seed = 7
sim_ps = 50.0
burn_fraction = 1.0 / 3.0

kT = 300.0 * BOLTZMANN_KCAL
dt_akma = dt_fs / AKMA_TIME_UNIT_FS
dof = 6 * n_waters - 3

positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)
displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
sys_dict = _prolix_params_pure_water(n_waters)
energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False,
)
mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_waters * 3)
mass_col = mass[:, None]
water_indices = settle.get_water_indices(0, n_waters)

steps = int(sim_ps * 1000.0 / dt_fs)
burn = int(steps * burn_fraction)
prod = steps - burn

for gamma_ps in [1.0, 10.0, 50.0]:
    gamma = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3
    init_fn, _ = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma, mass=mass,
        water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10,
    )
    state = init_fn(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

    def burn_step(s, _):
        s2, _ = settle_langevin_potential_cached_step(
            s, energy_fn=energy_fn, shift_fn=shift_fn, mass_col=mass_col,
            water_indices=water_indices, box_vec=box_vec, dt_akma=dt_akma,
            gamma_reduced=gamma, kT=kT, project_ou_momentum_rigid=True,
        )
        return s2, None

    def prod_step(s, _):
        s2, _ = settle_langevin_potential_cached_step(
            s, energy_fn=energy_fn, shift_fn=shift_fn, mass_col=mass_col,
            water_indices=water_indices, box_vec=box_vec, dt_akma=dt_akma,
            gamma_reduced=gamma, kT=kT, project_ou_momentum_rigid=True,
        )
        ke = rigid_tip3p_box_ke_kcal(s2.position, s2.momentum, s2.mass, n_waters)
        return s2, 2.0 * ke / (dof * BOLTZMANN_KCAL)

    print(f"\n=== γ={gamma_ps}/ps: compiling burn ({burn} steps)... ===", flush=True)
    state_burned, _ = jax.lax.scan(burn_step, state, None, length=burn)
    state_burned = jax.block_until_ready(state_burned)
    print(f"  Running production ({prod} steps)...", flush=True)
    _, temps = jax.lax.scan(prod_step, state_burned, None, length=prod)
    temps = np.array(jax.block_until_ready(temps))
    t_mean = float(np.mean(temps))
    t_std = float(np.std(temps))
    print(f"  γ={gamma_ps:5.1f}/ps  T_mean={t_mean:.2f} K  T_std={t_std:.2f} K  ΔT={t_mean-300:+.2f} K", flush=True)

print("\n=== INTERPRETATION ===")
print("γ=50/ps drives 50x stronger coupling — if T still high, it is a steady-state bias.")
print("If T_mean(γ=50) ≈ 300K but T_mean(γ=1) ≈ 411K → initialization artifact (slow equilibration).")
