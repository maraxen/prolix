"""Single-step substep trace of settle_langevin apply_fn on the 895-water box.

Replicates apply_fn line-by-line (gamma=0, deterministic) and prints the
max momentum / implied KE after each substep, to localize where the step-1
energy blowup is injected.
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
    _langevin_step_a, _langevin_step_b, settle_positions,
    _langevin_settle_vel, TIP3P_ROH, TIP3P_RHH)
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
dof = 6 * N - 3

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
wi = settle.get_water_indices(0, N)

init_s, _ = settle.settle_langevin(energy_fn, shift, dt=dt, kT=kT, gamma=0.0,
    mass=mass, water_indices=wi, box=box_vec, project_ou_momentum_rigid=True,
    projection_site="post_o", settle_velocity_iters=10)
state = init_s(jax.random.key(42), jnp.array(pos_a, dtype=jnp.float64), mass=mass)

m = state.mass  # (n_atoms,1)
mflat = m.reshape(-1)

def temp(p):
    ke = float(jnp.sum(jnp.sum(p ** 2, axis=-1) / (2.0 * mflat)))
    return 2.0 * ke / (dof * BOLTZMANN_KCAL), float(jnp.max(jnp.abs(p)))

def show(tag, p):
    t, mx = temp(p)
    print(f"  {tag:28s} T={t:12.2f} K   max|p|={mx:.4e}")

print(f"N={N} box={box_edge} half_dt(akma)={half_dt:.5f} dt(akma)={dt:.5f}")
show("init momentum", state.momentum)

momentum = _langevin_step_b(state.momentum, state.force, dt)
show("after B (half kick)", momentum)

x_unc_1 = _langevin_step_a(state.positions, momentum, m, half_dt, shift)
print(f"  R1 drift |x_unc_1-x0| max = {float(jnp.max(jnp.abs(x_unc_1-state.positions))):.4e}")
x_con_1 = settle_positions(x_unc_1, state.positions, wi, TIP3P_ROH, TIP3P_RHH, 15.999, 1.008, box_vec)
print(f"  R1 settle corr |x_con_1-x_unc_1| max = {float(jnp.max(jnp.abs(x_con_1-x_unc_1))):.4e}")
dp_1 = m * (x_con_1 - x_unc_1) / half_dt
print(f"  R1 dp_1 max = {float(jnp.max(jnp.abs(dp_1))):.4e}")
momentum = momentum + dp_1
show("after dp_1", momentum)

position = x_con_1
positions_mid = x_con_1
# O-step gamma=0: identity (skip; still apply constrained projection path is gamma-scaled)
from prolix.physics.settle import _langevin_step_o_constrained
momentum, _ = _langevin_step_o_constrained(momentum, position, m, 0.0, dt, kT, jax.random.key(0), wi)
show("after O (gamma=0)", momentum)

x_unc_2 = _langevin_step_a(position, momentum, m, half_dt, shift)
print(f"  R2 drift |x_unc_2-x_con_1| max = {float(jnp.max(jnp.abs(x_unc_2-position))):.4e}")
x_con_2 = settle_positions(x_unc_2, positions_mid, wi, TIP3P_ROH, TIP3P_RHH, 15.999, 1.008, box_vec)
print(f"  R2 settle corr |x_con_2-x_unc_2| max = {float(jnp.max(jnp.abs(x_con_2-x_unc_2))):.4e}")
dp_2 = m * (x_con_2 - x_unc_2) / half_dt
print(f"  R2 dp_2 max = {float(jnp.max(jnp.abs(dp_2))):.4e}")
momentum = momentum + dp_2
show("after dp_2", momentum)

position = x_con_2
force = force_fn(position)
print(f"  force max = {float(jnp.max(jnp.abs(force))):.4e}")
momentum = _langevin_step_b(momentum, force, dt)
show("after final B", momentum)

momentum_pre_sv = momentum
momentum = _langevin_settle_vel(momentum, positions_mid, position, m, wi, dt, 15.999, 1.008, n_iters=10)
show("after final SETTLE_vel", momentum)
print(f"  SETTLE_vel delta max|dp| = {float(jnp.max(jnp.abs(momentum-momentum_pre_sv))):.4e}")
print(f"  (note final SETTLE_vel uses positions_mid=x_con_1, position=x_con_2, dt=FULL dt)")
