#!/usr/bin/env python3
"""Debt 750/3516 verification: does the debt-841 no-water fallback fix change
the 1AKE (vacuum) NaN fragility under batched B=16 vmap?

Background: job 17905437 (B1-full prereg run, all 4 shape classes) found
5/64 systems non-finite at trajectory end, all within the 1AKE shape-class
group (B=16, vacuum, no explicit water). ``partition_bundles_by_shape``
groups 1AKE's 16 identical-topology replicas into their own homogeneous
group, separate from 1ubq/2gb1/water. ``_setup_integrator`` routes
``has_real_water == False`` bundles (1AKE's case) through
``settle_langevin``'s "no water" fallback -- the exact code path debt 841
fixed (atom_mask-aware COM-momentum recentering in
``masked_init_fn``, settle.py:1104-1135). This script isolates 1AKE alone
(no 1ubq/2gb1/water cost) at the same B=16, prereg step count, to check
whether the fix changed the finite fraction.

Usage (local L1 dry-run)::

    uv run python scripts/experiments/verify_debt750_1ake_nan.py --dry-run

Usage (local L2 smoke, few steps)::

    uv run python scripts/experiments/verify_debt750_1ake_nan.py \\
        --smoke --out /tmp/verify_debt750_smoke.json

Usage (cluster L3, prereg scale via bathos)::

    uv run bth run python scripts/experiments/verify_debt750_1ake_nan.py \\
        --tag debt750 --tag b1-1ake-nan --campaign <id> \\
        --out outputs/bench/verify_debt750_1ake_nan.json -- \\
        --replicas 16 --ps 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDB_DIR = ROOT / "data" / "pdb"

# Must match scripts/benchmarks/b1_init_exec.py prereg pins exactly --
# this is a same-conditions before/after comparison, not a new protocol.
DT_FS = 0.5
KT_KCAL = 0.596  # ~300 K
GAMMA_PS = 50.0  # vacuum-safe (XR-VACUUM-DT)
STEPS_PER_PS = int(round(1000.0 / DT_FS))


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _block_trajs(trajectories) -> None:
    import jax

    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    for traj in trajectories:
        jax.block_until_ready(traj.positions)


def _finite_stats(trajectories) -> tuple[bool, float, int, int]:
    import jax

    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    n_sys = len(trajectories)
    n_finite = 0
    for traj in trajectories:
        pos = traj.positions[-1]
        jax.block_until_ready(pos)
        if bool(jax.numpy.all(jax.numpy.isfinite(pos))):
            n_finite += 1
    frac = n_finite / n_sys if n_sys else 1.0
    return (n_finite == n_sys), frac, n_finite, n_sys


def _build_1ake_bundles(replicas: int):
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from _b1_paramize import paramize_pdb_to_bundle

    path = PDB_DIR / "1AKE.pdb"
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture: {path}")
    template = paramize_pdb_to_bundle(str(path), periodic=False)
    return [template] * replicas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--ps", type=float, default=100.0)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="B=2, 200 steps")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.smoke:
        replicas, n_steps = 2, 200
    else:
        replicas = args.replicas
        n_steps = args.n_steps if args.n_steps is not None else int(
            round(args.ps * STEPS_PER_PS)
        )

    bundles = _build_1ake_bundles(replicas)
    result = {
        "replicas": replicas,
        "n_steps": n_steps,
        "seed": args.seed,
        "has_real_water": bool(bundles[0].shape_spec.has_real_water),
        "git_hash": _git_hash(),
    }

    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0

    from prolix.api import EnsemblePlan

    plan = EnsemblePlan.from_bundles(bundles)

    t0 = time.perf_counter()
    plan.run(
        n_steps=1, dt=DT_FS, kT=KT_KCAL, seed=args.seed, gamma=GAMMA_PS,
        run_mode="inference",
    )
    t_first = time.perf_counter() - t0

    t1 = time.perf_counter()
    last = plan.run(
        n_steps=max(n_steps - 1, 1), dt=DT_FS, kT=KT_KCAL, seed=args.seed + 1,
        gamma=GAMMA_PS, run_mode="inference",
    )
    _block_trajs(last)
    t_ss = time.perf_counter() - t1

    all_finite, finite_frac, n_finite, n_sys = _finite_stats(last)

    result.update(
        {
            "t_first_step": t_first,
            "t_steady_state": t_ss,
            "finite_positions": all_finite,
            "finite_fraction": finite_frac,
            "n_finite": n_finite,
            "n_systems": n_sys,
        }
    )

    print(json.dumps(result, indent=2))

    out_path = args.out
    if out_path is None:
        env_path = os.environ.get("BTH_RESULTS_PATH")
        if env_path:
            out_path = Path(env_path)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")

    return 0 if all_finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
