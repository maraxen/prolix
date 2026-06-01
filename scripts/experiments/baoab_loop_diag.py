#!/usr/bin/env python3
"""Diagnostic: BAOAB Python loop vs scan temperature comparison."""
import json, os, sys, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "physics"))
from test_explicit_langevin_tip3p_parity import _equil_water_positions, _proxide_params_pure_water
from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.temperature_scan import make_jitted_temperature_scan
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

n_waters, seed, dt_fs = 2, 601, 0.5
positions_a, box_edge = _equil_water_positions(n_waters, seed=seed)
box_vec = jnp.array([box_edge]*3, dtype=jnp.float64)
dt_akma = dt_fs / AKMA_TIME_UNIT_FS
kT = 300.0 * BOLTZMANN_KCAL
gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
sys_dict = _proxide_params_pure_water(n_waters)
disp, shift = pbc.create_periodic_space(box_vec)
pme_grid = max(16, round(box_edge / 1.0))
ef = system.make_energy_fn(disp, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=pme_grid, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False)
mass = jnp.array([[15.999],[1.008],[1.008]]*n_waters).reshape(n_waters*3)
wi = settle.get_water_indices(0, n_waters)
init_s, apply_s = settle.settle_langevin(ef, shift, dt=dt_akma, kT=kT, gamma=gamma, mass=mass,
    water_indices=wi, box=box_vec, project_ou_momentum_rigid=True, projection_site="post_o")
apply_j = jax.jit(apply_s)
dof = 6*n_waters - 3

# --- Python loop ---
state_loop = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a, dtype=jnp.float64), mass=mass)
temps_loop = []
n_steps_total, burn = 20000, 10000
for i in range(n_steps_total):
    state_loop = apply_j(state_loop)
    if i >= burn:
        ke = float(rigid_tip3p_box_ke_kcal(state_loop.positions, state_loop.momentum, state_loop.mass, n_waters))
        temps_loop.append(2.0*ke/(dof*BOLTZMANN_KCAL))
T_loop = float(np.mean(temps_loop))
print(f"Python loop: mean_T = {T_loop:.1f} K over {len(temps_loop)} post-burn steps")

# --- Scan ---
state_scan = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a, dtype=jnp.float64), mass=mass)
collect = make_jitted_temperature_scan(apply_s, n_steps=n_steps_total, burn=burn, n_waters=n_waters)
temps_scan = collect(state_scan)
temps_scan.block_until_ready()
T_scan = float(np.nanmean(np.asarray(temps_scan, dtype=np.float64)))
print(f"jax.lax.scan: mean_T = {T_scan:.1f} K over {len(temps_scan)} post-burn steps")

result = {"loop_T_k": T_loop, "scan_T_k": T_scan, "n_waters": n_waters, "dt_fs": dt_fs, "seed": seed}
out = os.environ.get("PROLIX_RESULT_PATH") or os.environ.get("BTH_RESULTS_PATH")
if out:
    with open(out, "w") as f: json.dump(result, f)
else:
    print(json.dumps(result))
