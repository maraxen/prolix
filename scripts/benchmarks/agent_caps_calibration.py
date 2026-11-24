"""Calibrate the in-process caps of ``prolix.agent`` (spec section 4, Task 2).

PROVISIONAL. This script is a sketch, not a calibration, and it has not been run. Nothing in the
repo claims that the shipped caps (MAX_ATOMS, MAX_MIN_STEPS, MAX_MD_STEPS, MAX_DT_FS) were
validated. The validating run is a calibration on a real tiled protein with exclusions, run under
``bth run`` against the committed ``agent_caps_calibration.bth.toml`` pre-registration.

Times each capped operation at a given size on a synthetic free-boundary system (a 3-D grid of
atoms, alternating charges, one bond per consecutive pair), using the same ``ops`` functions the
CLI and MCP call. The intended rule is that caps are lowered until the largest allowed call takes
about 60 s; that rule is the pre-registered target, not a result.

Caps are overridden here per run (``--atoms``, ``--sd-steps``, ``--md-steps``, ``--dt-fs``), so a
size above the shipped cap can be measured. Thread count is recorded from the environment; run
with the same ``OMP_NUM_THREADS``/``OPENBLAS_NUM_THREADS``/``MKL_NUM_THREADS`` the table quotes.

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu \
      uv run --extra dev --extra agent python scripts/benchmarks/agent_caps_calibration.py \
      --out-dir "$TMPDIR/calib" --json "$TMPDIR/calib/table.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger("agent_caps_calibration")


def synthetic_system(n_atoms: int):
    """Free-boundary chain on a cubic grid: 3.5 A spacing, one bond per consecutive pair."""
    side = math.ceil(n_atoms ** (1.0 / 3.0))
    grid = np.array([(i, j, k) for i in range(side) for j in range(side) for k in range(side)], float)
    pos = grid[:n_atoms] * 3.5
    charges = np.zeros(n_atoms)
    charges[: n_atoms - n_atoms % 2] = np.tile([0.2, -0.2], (n_atoms - n_atoms % 2) // 2)
    bonds = np.array([[i, i + 1] for i in range(n_atoms - 1)], dtype=int)
    return SimpleNamespace(
        positions=pos,
        masses=np.full(n_atoms, 12.0),
        bonds=bonds,
        bond_params=np.tile([300.0, 3.5], (len(bonds), 1)),
        angles=None,
        angle_params=None,
        dihedrals=None,
        dihedral_params=None,
        impropers=None,
        improper_params=None,
        charges=charges,
        sigmas=np.full(n_atoms, 3.15),
        epsilons=np.full(n_atoms, 0.1),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the prolix.agent in-process caps on a synthetic system.")
    parser.add_argument("--atoms", type=int, default=1000)
    parser.add_argument("--sd-steps", type=int, default=200)
    parser.add_argument("--md-steps", type=int, default=500)
    parser.add_argument("--dt-fs", type=float, default=2.0)
    parser.add_argument("--out-dir", required=True, help="scratch directory for records and side files")
    parser.add_argument("--json", required=True, help="where to write the calibration table")
    parser.add_argument("--target-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PROLIX_AGENT_OUT_ROOT"] = str(out)

    import jax

    from prolix.agent import _runtime as rt
    from prolix.agent import ops
    from prolix.physics.system import make_bundle_from_system

    rt.MAX_ATOMS = max(rt.MAX_ATOMS, args.atoms)
    rt.MAX_MIN_STEPS = max(rt.MAX_MIN_STEPS, args.sd_steps)
    rt.MAX_MD_STEPS = max(rt.MAX_MD_STEPS, args.md_steps)

    thread_env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    table: dict = {
        "atoms": args.atoms,
        "sd_steps": args.sd_steps,
        "md_steps": args.md_steps,
        "dt_fs": args.dt_fs,
        "threads_env": thread_env,
        "cpu_count": os.cpu_count(),
        "jax_backend": jax.default_backend(),
        "target_s": args.target_s,
    }
    t0 = time.perf_counter()
    bundle = make_bundle_from_system(synthetic_system(args.atoms), boundary_condition="free")
    table["bundle_build_s"] = round(time.perf_counter() - t0, 3)
    logger.info("bundle built in %.2f s", table["bundle_build_s"])

    def timed(name: str, fn, **kw) -> dict:
        t = time.perf_counter()
        res = fn(out_dir=str(out / name), bundle=bundle, overwrite=True, **kw)
        wall = time.perf_counter() - t
        logger.info("%s finished in %.2f s", name, wall)
        return {"wall_s": round(wall, 3), "result": res["result"]}

    table["load_system"] = timed("load_system", ops.load_system)
    table["energy_breakdown"] = timed("energy_breakdown", ops.energy_breakdown)
    table["minimize"] = timed("minimize", ops.minimize, sd_steps=args.sd_steps)
    table["md_nve"] = timed("md_nve", ops.md_smoke, steps=args.md_steps, dt_fs=args.dt_fs, ensemble="nve")
    table["passes_target"] = bool(
        table["minimize"]["wall_s"] <= args.target_s and table["md_nve"]["wall_s"] <= args.target_s
    )
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(table, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: table[k] for k in ("atoms", "sd_steps", "md_steps", "passes_target")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
