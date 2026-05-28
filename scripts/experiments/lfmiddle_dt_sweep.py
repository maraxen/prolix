#!/usr/bin/env python3
"""LFMiddle dt-sweep experiment for bathos campaigns.

Writes JSON to $BTH_RESULTS_PATH (bth temp) and optionally --out (persistent).
Production segment uses ``jax.lax.scan`` (no Python step loop).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from prolix.physics import pbc, settle, system
from prolix.physics.temperature_scan import make_jitted_temperature_scan
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "physics"))
from test_explicit_langevin_tip3p_parity import _grid_water_positions, _proxide_params_pure_water  # noqa: E402


def run_sweep(
    *,
    dt_fs: float,
    integrator: str,
    n_waters: int,
    sim_ps: float,
    seed: int,
    burn_ps: float,
) -> dict:
  """Run one dt/integrator cell; return bath result_schema fields."""
  jax.config.update("jax_enable_x64", True)
  temperature_k = 300.0
  gamma_ps = 1.0
  positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3

  sys_dict = _proxide_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
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
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)

  kw = dict(
      energy_or_force_fn=energy_fn,
      shift_fn=shift_fn,
      dt=dt_akma,
      kT=kT,
      gamma=gamma_reduced,
      mass=mass,
      water_indices=water_indices,
      box=box_vec,
      project_ou_momentum_rigid=True,
      projection_site="post_o",
  )
  if integrator == "lfmiddle":
    init_s, apply_s = settle.settle_lfmiddle_langevin(**kw)
  else:
    init_s, apply_s = settle.settle_langevin(**kw)

  n_steps = int(sim_ps * 1000.0 / dt_fs)
  burn = int(burn_ps * 1000.0 / dt_fs)

  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  collect_temps = make_jitted_temperature_scan(
      apply_s, n_steps=n_steps, burn=burn, n_waters=n_waters
  )
  temps = collect_temps(state)
  temps.block_until_ready()

  arr = np.asarray(temps, dtype=np.float64)
  return {
      "dt_fs": dt_fs,
      "integrator": integrator,
      "n_waters": n_waters,
      "mean_T_k": float(np.nanmean(arr)),
      "std_T_k": float(np.nanstd(arr)),
      "max_T_k": float(np.nanmax(arr)),
      "sim_ps": sim_ps,
      "is_nan": bool(np.any(~np.isfinite(arr))),
      "seed": seed,
  }


def main() -> None:
  """CLI entry: parse args, run sweep, write JSON to BTH_RESULTS_PATH and/or --out."""
  p = argparse.ArgumentParser()
  p.add_argument("--dt-fs", type=float, required=True)
  p.add_argument("--integrator", choices=("lfmiddle", "baoab"), default="lfmiddle")
  p.add_argument("--n-waters", type=int, default=2)
  p.add_argument("--sim-ps", type=float, default=50.0)
  p.add_argument("--burn-ps", type=float, default=10.0)
  p.add_argument("--seed", type=int, default=701)
  p.add_argument("--out", default=None, help="Persistent output path (written in addition to BTH_RESULTS_PATH)")
  args = p.parse_args()

  result = run_sweep(
      dt_fs=args.dt_fs,
      integrator=args.integrator,
      n_waters=args.n_waters,
      sim_ps=args.sim_ps,
      seed=args.seed,
      burn_ps=args.burn_ps,
  )
  payload = json.dumps(result)

  bth_path = os.environ.get("BTH_RESULTS_PATH")
  if bth_path:
    with open(bth_path, "w", encoding="utf-8") as f:
      f.write(payload)

  if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(payload)

  if not bth_path and not args.out:
    print(payload)


if __name__ == "__main__":
  main()
