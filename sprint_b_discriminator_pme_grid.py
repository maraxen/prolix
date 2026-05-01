#!/usr/bin/env python3
"""Sprint B PME-grid discriminator.

Hypothesis: fixed pme_grid_points=32 causes coarser effective grid spacing
at larger N (box grows with n_waters), injecting systematic force errors the
thermostat absorbs as heat, producing the observed 342-415 K offset.

Test: scale pme_grid_points ~ box_edge / 1.0 Ang (~1 Ang/cell) and measure
whether the N-dependent temperature offset collapses.

Fixed-grid Sprint B baseline (pme_grid_points=32):
  n_waters=8:  T_mean=342.4 K
  n_waters=16: T_mean=341.7 K
  n_waters=32: T_mean=369.0 K
  n_waters=64: T_mean=414.9 K
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.settle_langevin_potential_propagator import settle_langevin_potential_cached_step
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _prolix_params_pure_water,
)

SPRINT_B_FIXED: dict[int, float] = {8: 342.4, 16: 341.7, 32: 369.0, 64: 414.9}


@dataclass
class Result:
    n_waters: int
    box_edge: float
    grid_fixed: int
    grid_scaled: int
    t_mean_k: float
    t_std_k: float


def run_condition(*, n_waters: int, dt_fs: float, seed: int, sim_ps: float, burn_fraction: float) -> Result:
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 * AKMA_TIME_UNIT_FS * 1e-3
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    pme_grid_points = int(max(16, round(float(box_edge) / 1.0)))
    print(f"  box={box_edge:.1f} Å  pme_grid fixed=32 ({box_edge/32:.2f} Å/cell)  scaled={pme_grid_points} ({box_edge/pme_grid_points:.2f} Å/cell)", flush=True)

    sys_dict = _prolix_params_pure_water(n_waters)
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
        pme_grid_points=pme_grid_points, pme_alpha=0.34, cutoff_distance=9.0,
        strict_parameterization=False,
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    mass_col = mass[:, None]
    water_indices = settle.get_water_indices(0, n_waters)
    dof = jnp.asarray(6 * n_waters - 3, dtype=jnp.float64)

    init_fn, _ = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt_akma, kT=kT, gamma=gamma, mass=mass,
        water_indices=water_indices, box=box_vec, remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True, projection_site="post_o", settle_velocity_iters=10,
    )
    state0 = init_fn(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)

    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = int(steps * burn_fraction)
    prod = steps - burn

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

    print(f"  Compiling + running burn ({burn} steps)...", flush=True)
    state_burned, _ = jax.lax.scan(burn_step, state0, None, length=burn)
    state_burned = jax.block_until_ready(state_burned)

    print(f"  Running production ({prod} steps)...", flush=True)
    _, temps = jax.lax.scan(prod_step, state_burned, None, length=prod)
    temps = np.array(jax.block_until_ready(temps))

    return Result(
        n_waters=n_waters, box_edge=float(box_edge),
        grid_fixed=32, grid_scaled=pme_grid_points,
        t_mean_k=float(np.mean(temps)), t_std_k=float(np.std(temps)),
    )


def main() -> None:
    results = []
    for nw in [8, 16, 32, 64]:
        print(f"\n=== n_waters={nw} ===", flush=True)
        r = run_condition(n_waters=nw, dt_fs=0.5, seed=7, sim_ps=50.0, burn_fraction=1/3)
        results.append(r)
        print(f"  T_mean={r.t_mean_k:.2f} K  T_std={r.t_std_k:.2f} K  (baseline={SPRINT_B_FIXED[nw]:.1f} K)", flush=True)

    print("\n" + "=" * 72)
    print(f"{'n_w':>4} | {'box':>6} | {'fix_grid':>8} | {'scl_grid':>8} | {'T_fix':>8} | {'T_scl':>8} | {'T_std':>7}")
    print("-" * 72)
    for r in results:
        print(f"{r.n_waters:>4} | {r.box_edge:>6.1f} | {r.grid_fixed:>8} | {r.grid_scaled:>8} | "
              f"{SPRINT_B_FIXED[r.n_waters]:>8.1f} | {r.t_mean_k:>8.2f} | {r.t_std_k:>7.2f}")

    scaled = [r.t_mean_k for r in results]
    spread = max(scaled) - min(scaled)
    all_closer = all(abs(r.t_mean_k - 300) < abs(SPRINT_B_FIXED[r.n_waters] - 300) for r in results)
    monotonic = all(scaled[i] <= scaled[i+1] for i in range(len(scaled)-1))

    print("\n" + "=" * 72)
    if spread < 10.0 and all_closer:
        verdict = "PME_CONFIRMED — scaled grid collapses the N-dependent offset"
    elif spread >= 25.0 and monotonic:
        verdict = "PME_RULED_OUT — offset persists with same N-scaling"
    else:
        verdict = "PME_PARTIAL — partial collapse, further investigation needed"
    print(f"VERDICT: {verdict}")
    print(f"  Scaled T spread: {spread:.1f} K  |  Fixed T spread: {SPRINT_B_FIXED[64]-SPRINT_B_FIXED[8]:.1f} K")


if __name__ == "__main__":
    main()
