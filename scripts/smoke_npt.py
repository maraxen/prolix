"""L2 smoke gate for NPT 20ps validation — tiny system, minimal steps."""
import time
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax

from prolix.physics import settle, system
from prolix.physics.simulate import NPTState
from prolix.physics import pbc
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "physics"))
from test_npt_barostat import (
    _grid_water_positions, _proxide_params_pure_water,
    AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL,
)

n_waters = 4
dt_akma = 0.5 / AKMA_TIME_UNIT_FS
kT = 300.0 * BOLTZMANN_KCAL

positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=3.1)
box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
sys_dict = _proxide_params_pure_water(n_waters)
displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
energy_fn = system.make_energy_fn(
    displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
    pme_grid_points=16, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
)
mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_waters * 3)
water_indices = settle.get_water_indices(n_protein_atoms=0, n_waters=n_waters)

# NVT
init_nvt, apply_nvt = settle.settle_langevin(
    energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=1.0,
    mass=mass, water_indices=water_indices,
    project_ou_momentum_rigid=True, projection_site="post_o",
)
apply_nvt_j = jax.jit(apply_nvt)
nvt_state = init_nvt(jax.random.PRNGKey(7), jnp.array(positions_a), mass=mass, box=box_vec)
t0 = time.time()
for _ in range(20):
    nvt_state = apply_nvt_j(nvt_state, box=box_vec)
print(f"NVT 20 steps ({n_waters}w): {time.time()-t0:.2f}s  finite={bool(jnp.all(jnp.isfinite(nvt_state.positions)))}")

# NPT
init_npt, apply_npt = settle.settle_csvr_npt(
    energy_fn, shift_fn, dt=dt_akma, kT=kT,
    target_pressure_bar=1.0, tau_barostat_akma=2000.0, tau_thermostat_akma=2000.0,
    mass=mass, water_indices=water_indices, box_init=box_vec,
)
apply_npt_j = jax.jit(apply_npt)
npt_state = init_npt(nvt_state.rng, nvt_state.positions, mass=mass, box=box_vec)
t0 = time.time()
for _ in range(20):
    npt_state = apply_npt_j(npt_state, box=npt_state.box)
print(f"NPT 20 steps ({n_waters}w): {time.time()-t0:.2f}s  finite={bool(jnp.all(jnp.isfinite(npt_state.positions)))}")

assert jnp.all(jnp.isfinite(npt_state.positions)), "NPT smoke: NaN detected"
assert jnp.all(jnp.isfinite(npt_state.box)), "NPT smoke: NaN in box"
print("L2 gate: PASS")
