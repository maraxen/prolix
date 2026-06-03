"""Decompose equilibrated rigid-TIP3P kinetic energy into translational vs
rotational temperature, to localize the post-fix 158 K under-thermalization.

If T_trans ~ 300 K and T_rot ~ 0 K, the rigid-body Langevin O-step is failing
to drive the 3 rotational DOF per water (the constraint-aware thermostat bug).
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _equil_water_positions, _proxide_params_pure_water)
from tests.physics.test_p2b_nvt_216water import _make_tip3p_excl_indices

N = 895
steps = 800
window = 400
pos_a, box_edge = _equil_water_positions(N, seed=42)
box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)
dt = 0.5 / AKMA_TIME_UNIT_FS
kT = 300.0 * BOLTZMANN_KCAL
gamma = 10.0 * AKMA_TIME_UNIT_FS * 1e-3

sysd = {k: v for k, v in _proxide_params_pure_water(N).items() if k != "exclusion_mask"}
sysd["excl_indices"] = _make_tip3p_excl_indices(N)
disp, shift = pbc.create_periodic_space(box_vec)
grid = max(16, round(box_edge / 1.0))
energy_fn = system.make_energy_fn(disp, sysd, box=box_vec, use_pbc=True,
    implicit_solvent=False, pme_grid_points=grid, pme_alpha=0.34,
    cutoff_distance=9.0, strict_parameterization=False)

n_atoms = N * 3
mass = jnp.array([[15.999], [1.008], [1.008]] * N, dtype=jnp.float64).reshape(n_atoms)
wi = settle.get_water_indices(0, N)
init_s, apply_s = settle.settle_langevin(energy_fn, shift, dt=dt, kT=kT, gamma=gamma,
    mass=mass, water_indices=wi, box=box_vec, project_ou_momentum_rigid=True,
    projection_site="post_o", settle_velocity_iters=10)
state = init_s(jax.random.key(42), jnp.array(pos_a, dtype=jnp.float64), mass=mass)

mflat = mass.reshape(-1)
M_water = float(15.999 + 1.008 + 1.008)


def decomp(p, r):
    """Return (T_trans, T_rot, T_tot) for rigid waters."""
    p3 = p.reshape(N, 3, 3)         # (water, atom, xyz)
    m3 = mflat.reshape(N, 3)        # (water, atom)
    v3 = p3 / m3[:, :, None]
    # COM velocity per water
    v_com = (p3.sum(axis=1)) / M_water        # (N,3)
    ke_trans = 0.5 * M_water * np.sum(np.asarray(v_com) ** 2)
    ke_tot = float(jnp.sum(jnp.sum(p ** 2, axis=-1) / (2.0 * mflat)))
    ke_rot = ke_tot - ke_trans
    dof_trans = 3 * N - 3
    dof_rot = 3 * N
    T_trans = 2 * ke_trans / (dof_trans * BOLTZMANN_KCAL)
    T_rot = 2 * ke_rot / (dof_rot * BOLTZMANN_KCAL)
    T_tot = 2 * ke_tot / ((6 * N - 3) * BOLTZMANN_KCAL)
    return float(T_trans), float(T_rot), float(T_tot)


apply_jit = jax.jit(apply_s)
tt, tr, to = [], [], []
for s in range(1, steps + 1):
    state = apply_jit(state)
    if s > window:
        a, b, c = decomp(state.momentum, state.positions)
        tt.append(a); tr.append(b); to.append(c)

print(f"window=[{window},{steps}]  N={N}")
print(f"T_trans = {np.mean(tt):7.1f} K   (3N-3 translational DOF)")
print(f"T_rot   = {np.mean(tr):7.1f} K   (3N rotational DOF)")
print(f"T_total = {np.mean(to):7.1f} K   (6N-3, as gate measures)")
print(f"target  = 300.0 K")
