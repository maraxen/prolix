#!/usr/bin/env python3
"""One-shot diagnostics for the ~few-K Prolix–OpenMM rigid-thermometer offset (investigation B).

At **fixed** geometry, compares OpenMM Reference vs Prolix ``make_energy_fn`` (same regression
PME as Slurm / ``REGRESSION_EXPLICIT_PME``). **Interpretation:** If static force RMSE is large,
fix PME/parameterization/constraints modeling before spending time on the BAOAB+SETTLE schedule.
If forces match but long-run T differs, treat the offset as **dynamics** (thermostat, operator
split, constraint iteration, or sampling).

  JAX_ENABLE_X64=1 uv run --extra openmm python scripts/benchmarks/tip3p_bias_investigation_snapshot.py \\
    --n-waters 33
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
  sys.path.insert(0, str(_SRC))
_BENCH = _REPO / "scripts" / "benchmarks"
if str(_BENCH) not in sys.path:
  sys.path.insert(0, str(_BENCH))

from prolix.physics import pbc, system  # noqa: E402
from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME  # noqa: E402

import tip3p_ke_compare as tkc  # noqa: E402


def _load_tip3p_langevin() -> object:
  path = _REPO / "scripts" / "benchmarks" / "tip3p_langevin_tightening.py"
  spec = importlib.util.spec_from_file_location("tip3p_langevin_tightening", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n-waters", type=int, default=4, help="Box size (use 33 to match Tier-1 benchmark).")
  ap.add_argument("--pdb", type=Path, default=None, help="Optional PDB; default builds grid WAT box.")
  args = ap.parse_args()
  pme = dict(REGRESSION_EXPLICIT_PME)
  n_w = int(args.n_waters)
  if args.pdb is not None:
    raise SystemExit("custom PDB not implemented; use default grid only")

  t3 = _load_tip3p_langevin()

  import jax
  import jax.numpy as jnp

  jax.config.update("jax_enable_x64", True)

  import openmm
  from openmm import unit as omm_unit
  from openmm.app import ForceField, HBonds, PDBFile, PME

  from tempfile import TemporaryDirectory

  positions_a, box_edge = tkc._tip3p_grid_positions(n_w, spacing_angstrom=10.0)
  n_atoms = n_w * 3
  alpha = float(pme["pme_alpha_per_angstrom"])
  grid = int(pme["pme_grid_points"])
  cutoff = float(pme["cutoff_angstrom"])
  platform_name = str(pme["openmm_platform"])

  with TemporaryDirectory() as tmp:
    pdb_path = Path(tmp) / "w.pdb"
    tkc._write_tip3p_pdb(pdb_path, positions_a, box_edge)
    pdb = PDBFile(str(pdb_path))
    ff = ForceField("amber14/tip3p.xml")
    omm_system = ff.createSystem(
      pdb.topology,
      nonbondedMethod=PME,
      nonbondedCutoff=cutoff / 10.0 * omm_unit.nanometer,
      constraints=HBonds,
      rigidWater=True,
    )
    for fi in range(omm_system.getNumForces()):
      f = omm_system.getForce(fi)
      if isinstance(f, openmm.NonbondedForce):
        f.setPMEParameters(alpha * 10.0, grid, grid, grid)
        f.setUseDispersionCorrection(bool(pme["use_dispersion_correction"]))

    integrator = openmm.VerletIntegrator(0.001 * omm_unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    ctx = openmm.Context(omm_system, integrator, platform)
    ctx.setPositions(
      [
        openmm.Vec3(positions_a[i, 0] / 10.0, positions_a[i, 1] / 10.0, positions_a[i, 2] / 10.0)
        for i in range(n_atoms)
      ]
    )
    st = ctx.getState(getForces=True)
    f_omm = st.getForces(asNumpy=True).value_in_unit(omm_unit.kilocalories_per_mole / omm_unit.angstrom)

  box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
  sys_dict = t3._prolix_params_pure_water(n_w)
  displacement_fn, _shift = pbc.create_periodic_space(box_vec)
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
  pos_jax = jnp.array(positions_a, dtype=jnp.float64)
  f_plx = -jax.grad(energy_fn)(pos_jax)
  f_omm_f = np.asarray(f_omm, dtype=np.float64)
  f_plx_f = np.asarray(f_plx, dtype=np.float64)
  diff = f_omm_f - f_plx_f
  out = {
    "schema": "tip3p_bias_investigation_snapshot/v1",
    "n_waters": n_w,
    "n_atoms": n_atoms,
    "regression_pme": pme,
    "note": "Total potential is not reported: OpenMM vs Prolix may use different energy zeros / self terms; for B use static force error as the model-sanity signal (see also tests/physics/test_explicit_langevin_tip3p_parity.py::test_openmm_prolix_tip3p_force_rmse_one_step).",
    "force_kcal_mol_A": {
      "rmse": float(np.sqrt(np.mean(diff**2))),
      "max_abs": float(np.max(np.abs(diff))),
      "mean_abs": float(np.mean(np.abs(diff))),
    },
  }
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
