#!/usr/bin/env python3
"""Local knob scan for TIP3P rigid thermometer (Prolix SETTLE+Langevin).

Examples::

  JAX_ENABLE_X64=1 uv run python scripts/benchmarks/tip3p_ke_isolate.py \\
    --n-waters 8 --steps 4000 --burn 2000 --sample-every 20 --matrix

  JAX_ENABLE_X64=1 uv run python scripts/benchmarks/tip3p_ke_isolate.py \\
    --projection-site post_settle_vel --settle-velocity-iters 20 --project-ou true
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
  sys.path.insert(0, str(_SRC))

from prolix.physics import pbc, settle, system  # noqa: E402
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME  # noqa: E402
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal  # noqa: E402
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL  # noqa: E402


def _tip3p_grid_positions(n_waters: int, spacing_angstrom: float) -> tuple[np.ndarray, float]:
  o = np.zeros(3, dtype=np.float64)
  r_oh = 0.9572
  theta = math.radians(104.52)
  h1 = np.array([r_oh, 0.0, 0.0], dtype=np.float64)
  h2 = np.array([r_oh * math.cos(theta), r_oh * math.sin(theta), 0.0], dtype=np.float64)
  sites: list[tuple[int, int, int]] = []
  n = int(math.ceil(n_waters ** (1.0 / 3.0))) + 3
  for ix in range(n):
    for iy in range(n):
      for iz in range(n):
        sites.append((ix, iy, iz))
        if len(sites) >= n_waters:
          break
      if len(sites) >= n_waters:
        break
    if len(sites) >= n_waters:
      break
  base = np.array([3.0, 3.0, 3.0], dtype=np.float64)
  pos: list[np.ndarray] = []
  for ix, iy, iz in sites[:n_waters]:
    wo = base + np.array([ix * spacing_angstrom, iy * spacing_angstrom, iz * spacing_angstrom])
    pos.extend([wo + o, wo + h1, wo + h2])
  arr = np.vstack(pos)
  span = np.max(arr, axis=0) - np.min(arr, axis=0)
  box_edge = float(np.max(span) + 16.0)
  return arr, box_edge


def _dof_rigid_tip3p_waters(n_waters: int) -> float:
  return float(6 * n_waters - 3)


def _prolix_params_pure_water(n_waters: int) -> dict:
  from prolix.physics.water_models import WaterModelType, get_water_params

  tip = get_water_params(WaterModelType.TIP3P)
  qo, qh = float(tip.charge_O), float(tip.charge_H)
  sig_o = float(tip.sigma_O)
  eps_o = float(tip.epsilon_O)
  n = n_waters * 3
  charges: list[float] = []
  sigmas: list[float] = []
  epsilons: list[float] = []
  for _ in range(n_waters):
    charges.extend([qo, qh, qh])
    sigmas.extend([sig_o, 1.0, 1.0])
    epsilons.extend([eps_o, 0.0, 0.0])
  mask = jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64)
  for w in range(n_waters):
    b = w * 3
    for i, j in [(0, 1), (0, 2), (1, 2)]:
      a, c = b + i, b + j
      mask = mask.at[a, c].set(0.0).at[c, a].set(0.0)
  return {
    "charges": jnp.array(charges, dtype=jnp.float64),
    "sigmas": jnp.array(sigmas, dtype=jnp.float64),
    "epsilons": jnp.array(epsilons, dtype=jnp.float64),
    "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
    "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "angles": jnp.zeros((0, 3), dtype=jnp.int32),
    "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
    "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
    "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
    "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
    "exclusion_mask": mask,
  }


def _run_prolix_short(
  *,
  n_waters: int,
  seed: int,
  steps: int,
  burn: int,
  sample_every: int,
  dt_fs: float,
  temperature_k: float,
  gamma_ps: float,
  remove_linear_com_momentum: bool,
  project_ou_momentum_rigid: bool,
  projection_site: str,
  settle_velocity_iters: int,
) -> dict:
  jax.config.update("jax_enable_x64", True)
  positions_a, box_edge = _tip3p_grid_positions(n_waters, spacing_angstrom=10.0)
  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  pme = dict(REGRESSION_EXPLICIT_PME)
  alpha = float(pme["pme_alpha_per_angstrom"])
  grid = int(pme["pme_grid_points"])
  cutoff = float(pme["cutoff_angstrom"])
  dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
  kT = float(temperature_k) * BOLTZMANN_KCAL
  gamma_reduced = float(gamma_ps) * float(AKMA_TIME_UNIT_FS) * 1e-3
  sys_dict = _prolix_params_pure_water(n_waters)
  displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
  energy_fn = system.make_energy_fn(
    displacement_fn,
    sys_dict,
    box=box_vec,
    use_pbc=True,
    implicit_solvent=False,
    pme_grid_points=grid,
    pme_alpha=alpha,
    cutoff_distance=cutoff,
    strict_parameterization=False,
  )
  n_atoms = n_waters * 3
  mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
  water_indices = settle.get_water_indices(0, n_waters)
  init_s, apply_s = settle.settle_langevin(
    energy_fn,
    shift_fn,
    dt=dt_akma,
    kT=kT,
    gamma=gamma_reduced,
    mass=mass,
    water_indices=water_indices,
    box=box_vec,
    remove_linear_com_momentum=bool(remove_linear_com_momentum),
    project_ou_momentum_rigid=bool(project_ou_momentum_rigid),
    projection_site=projection_site,
    settle_velocity_iters=int(settle_velocity_iters),
  )
  apply_j = jax.jit(apply_s)
  dof_rigid = _dof_rigid_tip3p_waters(n_waters)
  state = init_s(jax.random.PRNGKey(seed), jnp.array(positions_a), mass=mass)
  temps: list[float] = []
  for step in range(steps):
    state = apply_j(state)
    if step >= burn and (step - burn) % sample_every == 0:
      ke_r = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
      temps.append(2.0 * ke_r / (dof_rigid * BOLTZMANN_KCAL))
  arr = np.asarray(temps, dtype=np.float64)
  return {
    "mean_T_K": float(arr.mean()) if arr.size else float("nan"),
    "std_T_K": float(arr.std(ddof=0)) if arr.size else float("nan"),
    "n_samples": int(arr.size),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n-waters", type=int, default=8)
  ap.add_argument("--seed", type=int, default=4242)
  ap.add_argument("--steps", type=int, default=4000)
  ap.add_argument("--burn", type=int, default=2000)
  ap.add_argument("--sample-every", type=int, default=20)
  ap.add_argument("--dt-fs", type=float, default=2.0)
  ap.add_argument("--temperature-k", type=float, default=300.0)
  ap.add_argument("--gamma-ps", type=float, default=1.0)
  ap.add_argument("--remove-linear-com-momentum", action=argparse.BooleanOptionalAction, default=False)
  ap.add_argument("--project-ou", action=argparse.BooleanOptionalAction, default=True)
  ap.add_argument("--projection-site", choices=("post_o", "post_settle_vel", "both"), default="post_o")
  ap.add_argument("--settle-velocity-iters", type=int, default=10)
  ap.add_argument("--matrix", action="store_true", help="Scan projection_site × settle_iters × project_ou")
  args = ap.parse_args()

  if args.matrix:
    rows = []
    for site in ("post_o", "post_settle_vel", "both"):
      for iters in (10, 20):
        for pou in (True, False):
          if site == "both" and not pou:
            continue
          r = _run_prolix_short(
            n_waters=args.n_waters,
            seed=args.seed,
            steps=args.steps,
            burn=args.burn,
            sample_every=args.sample_every,
            dt_fs=args.dt_fs,
            temperature_k=args.temperature_k,
            gamma_ps=args.gamma_ps,
            remove_linear_com_momentum=args.remove_linear_com_momentum,
            project_ou_momentum_rigid=pou,
            projection_site=site,
            settle_velocity_iters=iters,
          )
          rows.append({"projection_site": site, "settle_velocity_iters": iters, "project_ou": pou, **r})
    print(json.dumps({"schema": "tip3p_ke_isolate/matrix/v1", "rows": rows}, indent=2))
    return 0

  out = _run_prolix_short(
    n_waters=args.n_waters,
    seed=args.seed,
    steps=args.steps,
    burn=args.burn,
    sample_every=args.sample_every,
    dt_fs=args.dt_fs,
    temperature_k=args.temperature_k,
    gamma_ps=args.gamma_ps,
    remove_linear_com_momentum=args.remove_linear_com_momentum,
    project_ou_momentum_rigid=args.project_ou,
    projection_site=args.projection_site,
    settle_velocity_iters=args.settle_velocity_iters,
  )
  print(
    json.dumps(
      {
        "schema": "tip3p_ke_isolate/single/v1",
        "config": {k: getattr(args, k) for k in vars(args) if k != "matrix"},
        **out,
      },
      indent=2,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
