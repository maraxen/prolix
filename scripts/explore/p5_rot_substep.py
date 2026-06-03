"""Track T_trans / T_rot through each substep of one settle_langevin step.

Init momentum is rigid (has rotation). We replicate apply_fn (min-image dp,
matching the patched settle.py) and print T_trans/T_rot after each substep to
find which operation freezes rotation (post-fix the gate equilibrates with
T_rot ~ 1 K).
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.physics.settle import (
    _langevin_step_a, _langevin_step_b, settle_positions, _langevin_settle_vel,
    _langevin_step_o_constrained, TIP3P_ROH, TIP3P_RHH)
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _equil_water_positions, _proxide_params_pure_water)
from tests.physics.test_p2b_nvt_216water import _make_tip3p_excl_indices

N = 895
pos_a, box_edge = _equil_water_positions(N, seed=42)
box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)
dt = 0.5 / AKMA_TIME_UNIT_FS
half_dt = 0.5 * dt
kT = 300.0 * BOLTZMANN_KCAL
gamma = 10.0 * AKMA_TIME_UNIT_FS * 1e-3

sysd = {k: v for k, v in _proxide_params_pure_water(N).items() if k != "exclusion_mask"}
sysd["excl_indices"] = _make_tip3p_excl_indices(N)
disp, shift = pbc.create_periodic_space(box_vec)
grid = max(16, round(box_edge / 1.0))
energy_fn = system.make_energy_fn(disp, sysd, box=box_vec, use_pbc=True,
    implicit_solvent=False, pme_grid_points=grid, pme_alpha=0.34,
    cutoff_distance=9.0, strict_parameterization=False)
force_fn = lambda r: -jax.grad(lambda x: energy_fn(x).sum())(r)

n_atoms = N * 3
mass = jnp.array([[15.999], [1.008], [1.008]] * N, dtype=jnp.float64).reshape(n_atoms)
m = mass.reshape(-1, 1)
mflat = mass.reshape(-1)
wi = settle.get_water_indices(0, N)
M_water = float(15.999 + 1.008 + 1.008)
DOF_T, DOF_R = 3 * N - 3, 3 * N


def TR(p):
    p3 = p.reshape(N, 3, 3)
    v_com = np.asarray(p3.sum(axis=1)) / M_water
    ke_t = 0.5 * M_water * np.sum(v_com ** 2)
    ke_tot = float(jnp.sum(jnp.sum(p ** 2, axis=-1) / (2.0 * mflat)))
    return (2 * ke_t / (DOF_T * BOLTZMANN_KCAL),
            2 * (ke_tot - ke_t) / (DOF_R * BOLTZMANN_KCAL))


def show(tag, p):
    t, r = TR(p)
    print(f"  {tag:24s} T_trans={t:8.1f}  T_rot={r:8.1f}")


def mi(dx):
    return dx - box_vec * jnp.round(dx / box_vec)


init_s, _ = settle.settle_langevin(energy_fn, shift, dt=dt, kT=kT, gamma=gamma,
    mass=mass, water_indices=wi, box=box_vec, project_ou_momentum_rigid=True,
    projection_site="post_o", settle_velocity_iters=10)
state = init_s(jax.random.key(42), jnp.array(pos_a, dtype=jnp.float64), mass=mass)

show("init", state.momentum)
momentum = _langevin_step_b(state.momentum, state.force, dt)
show("after B", momentum)
x_unc_1 = _langevin_step_a(state.positions, momentum, m, half_dt, shift)
x_con_1 = settle_positions(x_unc_1, state.positions, wi, TIP3P_ROH, TIP3P_RHH, 15.999, 1.008, box_vec)
momentum = momentum + m * mi(x_con_1 - x_unc_1) / half_dt
show("after dp_1 (min-img)", momentum)
position = x_con_1
positions_mid = x_con_1
momentum, _ = _langevin_step_o_constrained(momentum, position, m, gamma, dt, kT, jax.random.key(7), wi)
show("after O", momentum)
x_unc_2 = _langevin_step_a(position, momentum, m, half_dt, shift)
x_con_2 = settle_positions(x_unc_2, positions_mid, wi, TIP3P_ROH, TIP3P_RHH, 15.999, 1.008, box_vec)
momentum = momentum + m * mi(x_con_2 - x_unc_2) / half_dt
show("after dp_2 (min-img)", momentum)
position = x_con_2
force = force_fn(position)
momentum = _langevin_step_b(momentum, force, dt)
show("after final B", momentum)
momentum = _langevin_settle_vel(momentum, positions_mid, position, m, wi, dt, 15.999, 1.008, n_iters=10)
show("after SETTLE_vel", momentum)
