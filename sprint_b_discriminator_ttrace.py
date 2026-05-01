#!/usr/bin/env python3
"""Sprint B Discriminator: T(t) trace for N=64.

Emits T every 1 ps to distinguish:
  - Monotonically rising T → system not equilibrated (initialization issue)
  - T plateaus above 300 K → equilibrium itself is wrong (thermostat/estimator bug)
"""
from __future__ import annotations
import sys, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.settle_langevin_potential_propagator import settle_langevin_potential_cached_step
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water

n_waters = 64; dt_fs = 0.5; seed = 7; sim_ps = 100.0; window_ps = 1.0
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
mass = jnp.array([[15.999],[1.008],[1.008]]*n_waters).reshape(n_waters*3)
mass_col = mass[:,None]
water_indices = settle.get_water_indices(0, n_waters)
dof = jnp.asarray(6*n_waters - 3, dtype=jnp.float64)
init_fn, _ = settle.settle_langevin(energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma,
    mass=mass, water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False,
    project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10)
state = init_fn(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

steps_per_window = int(window_ps * 1000.0 / dt_fs)  # 2000 steps per ps
total_windows = int(sim_ps / window_ps)              # 100 windows

def window_step(s, _):
    s2, _ = settle_langevin_potential_cached_step(s, energy_fn=energy_fn, shift_fn=shift_fn,
        mass_col=mass_col, water_indices=water_indices, box_vec=box_vec, dt_akma=dt_akma,
        gamma_reduced=gamma, kT=kT, project_ou_momentum_rigid=True)
    ke = rigid_tip3p_box_ke_kcal(s2.position, s2.momentum, s2.mass, n_waters)
    return s2, 2.0 * ke / (dof * BOLTZMANN_KCAL)

run_window = jax.jit(lambda s: jax.lax.scan(window_step, s, None, length=steps_per_window))

print(f"T(t) trace: n_waters={n_waters}, dt={dt_fs}fs, window={window_ps}ps", flush=True)
print(f"{'t(ps)':>7} | {'T_window(K)':>12} | {'T_cumul(K)':>12} | trend", flush=True)
print("-" * 50, flush=True)

all_temps = []
for w in range(total_windows):
    state, temps = run_window(state)
    state = jax.block_until_ready(state)
    t_ps = (w + 1) * window_ps
    t_window = float(np.mean(np.array(temps)))
    all_temps.extend(temps.tolist())
    t_cumul = float(np.mean(all_temps))
    # trend: rising/falling/plateau over last 5 windows
    if len(all_temps) >= steps_per_window * 5:
        recent = np.mean(all_temps[-steps_per_window*5:])
        earlier = np.mean(all_temps[-steps_per_window*10:-steps_per_window*5]) if len(all_temps) >= steps_per_window*10 else recent
        trend = f"↑{recent-earlier:+.1f}K" if abs(recent-earlier) > 2 else "plateau"
    else:
        trend = "warmup"
    print(f"{t_ps:>7.1f} | {t_window:>12.2f} | {t_cumul:>12.2f} | {trend}", flush=True)
    sys.stdout.flush()

print("\n=== VERDICT ===")
first_10 = np.mean(all_temps[:steps_per_window*10])
last_10 = np.mean(all_temps[-steps_per_window*10:])
drift = last_10 - first_10
if abs(drift) < 5 and last_10 > 310:
    print(f"PLATEAU_ABOVE_300: T equilibrated at {last_10:.1f} K (drift={drift:+.1f} K over full run) → thermostat/estimator bug")
elif drift > 20:
    print(f"STILL_RISING: T rose {drift:+.1f} K from first to last 10ps → not equilibrated, initialization issue")
elif abs(drift) < 5 and abs(last_10 - 300) < 10:
    print(f"EQUILIBRATED_OK: T={last_10:.1f} K → system converged (earlier windows were transient)")
else:
    print(f"UNCLEAR: drift={drift:+.1f} K, final T={last_10:.1f} K → more investigation needed")
