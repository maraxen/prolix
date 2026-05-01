#!/usr/bin/env python3
"""Sprint B Discriminator 1: SETTLE-off temperature control test.

Runs unconstrained (flexible TIP3P, SETTLE disabled) Langevin NVT across
dt in {0.25, 0.5, 1.0} fs at 16 waters, using lax.scan for efficiency.

Verdict logic:
  T ≈ 300 K at all dt  → bug is in SETTLE+OU interaction, not bare OU.
  T elevated and grows with smaller dt → bug is in OU step/thermometer.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, system
from prolix.physics.integrator_builder import make_integrator
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from tests.physics.test_explicit_langevin_tip3p_parity import (
    _grid_water_positions,
    _prolix_params_pure_water,
)


@dataclass
class DiscriminatorResult:
    n_waters: int
    dt_fs: float
    settle_enabled: bool
    steps: int
    burn: int
    t_target_k: float
    t_mean_k: float
    t_std_k: float
    dof: int
    elapsed_s: float


def run_condition(
    *,
    n_waters: int,
    dt_fs: float,
    sim_ps: float,
    burn_fraction: float,
    seed: int,
    temperature_k: float = 300.0,
) -> DiscriminatorResult:
    kT = float(temperature_k) * BOLTZMANN_KCAL
    gamma_ps = 1.0
    gamma = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)

    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    sys_dict = _prolix_params_pure_water(n_waters)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=32,
        pme_alpha=0.34,
        cutoff_distance=9.0,
        strict_parameterization=False,
    )

    n_atoms = n_waters * 3
    mass_flat = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    mass = mass_flat[:, None]  # (N, 1) for momentum/mass broadcasting

    # Unconstrained BAOAB — water_indices=None disables SETTLE and rigid OU
    init_fn, apply_fn = make_integrator(
        energy_fn,
        shift_fn,
        mass,
        sequence_name="baoab_langevin",
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        water_indices=None,
        box=box_vec,
    )

    key = jax.random.PRNGKey(seed)
    state0 = init_fn(key, jnp.array(positions_a, dtype=jnp.float64))

    # Cartesian DOF: 3*N_atoms - 3 (COM removed)
    dof = 3 * n_atoms - 3

    steps = int(sim_ps * 1000.0 / dt_fs)
    burn = int(steps * burn_fraction)
    prod = steps - burn

    def cartesian_t(state) -> jnp.ndarray:
        mom = state.momentum  # (N, 3)
        ke = 0.5 * jnp.sum(jnp.sum(mom**2, axis=-1) / mass_flat)
        return 2.0 * ke / (dof * BOLTZMANN_KCAL)

    def burn_step(state, _):
        return apply_fn(state), None

    def prod_step(state, _):
        state = apply_fn(state)
        return state, cartesian_t(state)

    print(f"  dt={dt_fs} fs: compiling burn ({burn} steps)...", flush=True)
    t0 = time.perf_counter()
    state_burned, _ = jax.lax.scan(burn_step, state0, None, length=burn)
    state_burned = jax.block_until_ready(state_burned)
    print(f"  dt={dt_fs} fs: burn done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"  dt={dt_fs} fs: running production ({prod} steps)...", flush=True)
    t1 = time.perf_counter()
    _, temps = jax.lax.scan(prod_step, state_burned, None, length=prod)
    temps = jax.block_until_ready(temps)
    elapsed = time.perf_counter() - t1

    temps_np = np.array(temps)
    return DiscriminatorResult(
        n_waters=n_waters,
        dt_fs=dt_fs,
        settle_enabled=False,
        steps=steps,
        burn=burn,
        t_target_k=temperature_k,
        t_mean_k=float(np.mean(temps_np)),
        t_std_k=float(np.std(temps_np)),
        dof=dof,
        elapsed_s=elapsed,
    )


def main() -> None:
    n_waters = 16
    sim_ps = 100.0
    burn_fraction = 1.0 / 3.0
    seed = 7
    dt_values = [1.0, 0.5, 0.25]  # largest dt first (fastest compile, most stable)

    results = []
    for dt_fs in dt_values:
        print(f"\n=== SETTLE-OFF: n_waters={n_waters}, dt={dt_fs} fs ===", flush=True)
        try:
            r = run_condition(
                n_waters=n_waters,
                dt_fs=dt_fs,
                sim_ps=sim_ps,
                burn_fraction=burn_fraction,
                seed=seed,
            )
            results.append(asdict(r))
            print(
                f"  T_mean={r.t_mean_k:.2f} K  T_std={r.t_std_k:.2f} K"
                f"  ΔT={r.t_mean_k - r.t_target_k:+.2f} K  ({r.elapsed_s:.1f}s prod)",
                flush=True,
            )
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}", flush=True)
            traceback.print_exc()
            results.append({"dt_fs": dt_fs, "error": str(exc)})

    print("\n=== SUMMARY ===")
    print(f"{'dt_fs':>8}  {'T_mean (K)':>12}  {'T_std (K)':>10}  {'ΔT (K)':>8}")
    for r in results:
        if "error" not in r:
            delta = r["t_mean_k"] - r["t_target_k"]
            print(
                f"{r['dt_fs']:>8.2f}  {r['t_mean_k']:>12.2f}  {r['t_std_k']:>10.2f}"
                f"  {delta:>+8.2f}"
            )
        else:
            print(f"{r['dt_fs']:>8.2f}  FAILED: {r['error']}")

    out = Path(".praxia/tmp/discriminator1_settle_off.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"experiment": "discriminator1_settle_off", "results": results}, indent=2)
    )
    print(f"\nResults written to {out}")

    ok = [r for r in results if "error" not in r]
    if len(ok) >= 2:
        # Sort by dt descending so index 0 = largest dt
        ok_sorted = sorted(ok, key=lambda r: -r["dt_fs"])
        t_means = [r["t_mean_k"] for r in ok_sorted]
        all_near_300 = all(abs(t - 300.0) < 10.0 for t in t_means)
        grows_with_smaller_dt = t_means[-1] > t_means[0]  # smallest dt is hottest
        if all_near_300:
            print(
                "\nVERDICT: T ≈ 300 K at all dt."
                "\n→ Bare OU step is correct. Bug is in SETTLE+OU interaction"
                "\n  (likely SETTLE_vel reference-frame ordering)."
            )
        elif grows_with_smaller_dt:
            print(
                "\nVERDICT: T grows as dt shrinks (SETTLE disabled)."
                "\n→ Bug is in OU step or thermometer, SETTLE-independent."
                "\n  Check σ formula and noise amplitude scaling."
            )
        else:
            print(
                "\nVERDICT: T elevated but no clear dt trend."
                "\n→ Likely steady-state bias unrelated to discretization."
                "\n  Run OpenMM 3-way comparison next."
            )


if __name__ == "__main__":
    main()
