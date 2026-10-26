#!/usr/bin/env python3
"""Diagnostic: print temperature profile for first 500 steps."""
import json, os, sys, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "physics"))
from test_explicit_langevin_tip3p_parity import _equil_water_positions, _proxide_params_pure_water
from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
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
dof = 6*n_waters - 3

init_s, apply_s = settle.settle_langevin(ef, shift, dt=dt_akma, kT=kT, gamma=gamma, mass=mass,
    water_indices=wi, box=box_vec, project_ou_momentum_rigid=True, projection_site="post_o")
apply_j = jax.jit(apply_s)

state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a, dtype=jnp.float64), mass=mass)

# initial energy
ke0 = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))
T0 = 2.0 * ke0 / (dof * BOLTZMANN_KCAL)
force_mag = float(jnp.sqrt(jnp.sum(state.force**2)))
print(f"Step 0: T={T0:.1f} K, KE={ke0:.4f} kcal/mol, |F|={force_mag:.2f}")

# check initial O-O distance
pos = np.asarray(state.positions)
d_oo = float(np.linalg.norm(pos[0] - pos[3]))
print(f"Initial O-O distance: {d_oo:.2f} A")

# first 500 steps
for i in range(1, 501):
    state = apply_j(state)
    if i in [1, 5, 10, 50, 100, 200, 500]:
        ke = float(rigid_tip3p_box_ke_kcal(state.positions, state.momentum, state.mass, n_waters))
        T = 2.0 * ke / (dof * BOLTZMANN_KCAL)
        fm = float(jnp.sqrt(jnp.sum(state.force**2)))
        print(f"Step {i:4d}: T={T:.1f} K, KE={ke:.4f}, |F|={fm:.2f}")

out = os.environ.get("PROLIX_RESULT_PATH")
if out:
    with open(out, "w") as f: json.dump({"T0": T0, "d_oo": d_oo}, f)
