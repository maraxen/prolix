#!/usr/bin/env python3
"""Sprint B Discriminator: project_ou_momentum_rigid=False N-sweep.

If superlinear N-scaling vanishes → rigid-projection noise is the bug.
If it persists → something else (estimator, initial conditions).

Baseline (proj=True, fixed PME):
  n_waters=8:  342.4 K
  n_waters=16: 341.7 K
  n_waters=32: 369.0 K
  n_waters=64: 414.9 K
"""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.settle_langevin_potential_propagator import settle_langevin_potential_cached_step
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water

BASELINE = {8: 342.4, 16: 341.7, 32: 369.0, 64: 414.9}

def run(n_waters, dt_fs=0.5, seed=7, sim_ps=50.0, burn_frac=1/3):
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge]*3, dtype=jnp.float64)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    sys_dict = _prolix_params_pure_water(n_waters)
    energy_fn = system.make_energy_fn(displacement_fn, sys_dict, box=box_vec, use_pbc=True,
        implicit_solvent=False, pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0,
        strict_parameterization=False)
    n_atoms = n_waters * 3
    mass = jnp.array([[15.999],[1.008],[1.008]]*n_waters).reshape(n_atoms)
    mass_col = mass[:,None]
    water_indices = settle.get_water_indices(0, n_waters)
    dof = jnp.asarray(6*n_waters - 3, dtype=jnp.float64)
    init_fn, _ = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma,
        mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False,
        project_ou_momentum_rigid=False,  # KEY: disabled
        projection_site="post_o", settle_velocity_iters=10)
    state0 = init_fn(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = int(steps * burn_frac); prod = steps - burn

    def burn_step(s, _):
        s2, _ = settle_langevin_potential_cached_step(s, energy_fn=energy_fn, shift_fn=shift_fn,
            mass_col=mass_col, water_indices=water_indices, box_vec=box_vec, dt_akma=dt_akma,
            gamma_reduced=gamma, kT=kT, project_ou_momentum_rigid=False)
        return s2, None

    def prod_step(s, _):
        s2, _ = settle_langevin_potential_cached_step(s, energy_fn=energy_fn, shift_fn=shift_fn,
            mass_col=mass_col, water_indices=water_indices, box_vec=box_vec, dt_akma=dt_akma,
            gamma_reduced=gamma, kT=kT, project_ou_momentum_rigid=False)
        ke = rigid_tip3p_box_ke_kcal(s2.position, s2.momentum, s2.mass, n_waters)
        return s2, 2.0 * ke / (dof * BOLTZMANN_KCAL)

    state_b, _ = jax.lax.scan(burn_step, state0, None, length=burn)
    state_b = jax.block_until_ready(state_b)
    _, temps = jax.lax.scan(prod_step, state_b, None, length=prod)
    temps = np.array(jax.block_until_ready(temps))
    return float(np.mean(temps)), float(np.std(temps))

print("Discriminator: project_ou_momentum_rigid=False", flush=True)
results = []
for nw in [8, 16, 32, 64]:
    print(f"\n=== n_waters={nw} ===", flush=True)
    t_mean, t_std = run(nw)
    results.append((nw, t_mean, t_std))
    print(f"  T_mean={t_mean:.2f} K  T_std={t_std:.2f} K  (baseline proj=True: {BASELINE[nw]:.1f} K)", flush=True)

print("\n=== SUMMARY ===")
print(f"{'n_w':>4} | {'T_proj_on':>10} | {'T_proj_off':>11} | {'delta':>8}")
print("-" * 42)
for nw, t, ts in results:
    print(f"{nw:>4} | {BASELINE[nw]:>10.1f} | {t:>11.2f} | {t-BASELINE[nw]:>+8.1f}")

ts = [r[1] for r in results]
spread = max(ts) - min(ts)
mono = all(ts[i] <= ts[i+1] for i in range(len(ts)-1))
if spread < 10 and all(abs(t-300) < 10 for t in ts):
    v = "PROJ_CONFIRMED — scaling collapsed, proj noise is the bug"
elif spread >= 25 and mono:
    v = "PROJ_RULED_OUT — N-scaling persists without rigid projection"
else:
    v = f"PROJ_PARTIAL — spread={spread:.1f} K, monotonic={mono}"
print(f"\nVERDICT: {v}")
