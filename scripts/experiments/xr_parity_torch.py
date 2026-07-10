#!/usr/bin/env python3
"""XR-PARITY-TORCH: rewire regression via prolix Scope A (external_baseline).

Re-homes ``bench_prolix`` as a confirmation-mode experiment. Does **not**
require TorchMD/torch — comparator benches stay under scripts/benchmarks/.

Usage::

    uv run bth run python scripts/experiments/xr_parity_torch.py \\
      --tag xr-parity-torch --campaign <id> \\
      --out outputs/xr_parity_torch.json \\
      -- --probe all --out outputs/xr_parity_torch.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", choices=("planner", "prolix_smoke", "all"), default="all")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset-dir", type=Path, default=_REPO / "data/ani1x_subset")
    ap.add_argument("--n-mols", type=int, default=4)
    ap.add_argument("--n-conf-cap", type=int, default=8)
    ap.add_argument("--n-warmup", type=int, default=1)
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _json_safe(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _probe_planner() -> dict:
    from prolix.tiling.planner import BatchPlanner

    plan_src = BatchPlanner.plan
    xtrax_src = BatchPlanner.plan_with_xtrax
    # plan() must call plan_with_xtrax (alias body)
    import inspect

    plan_body = inspect.getsource(plan_src)
    ok = "plan_with_xtrax" in plan_body and plan_src is not xtrax_src
    return {
        "probe": "planner",
        "planner_pass": int(ok),
        "prolix_smoke_ran": 0,
        "gate_pass": int(ok),
        "per_mol_step_seconds": None,
        "final_loss": None,
        "n_mols": 0,
        "skip_reason": None if ok else "BatchPlanner.plan does not delegate to plan_with_xtrax",
    }


def _probe_prolix_smoke(args: argparse.Namespace) -> dict:
    lane = args.subset_dir / "lane_a"
    if not lane.exists():
        return {
            "probe": "prolix_smoke",
            "planner_pass": 1,
            "prolix_smoke_ran": 0,
            "gate_pass": 0,
            "per_mol_step_seconds": None,
            "final_loss": None,
            "n_mols": args.n_mols,
            "skip_reason": f"missing fixture dir {lane}",
        }

    bench_path = _REPO / "scripts/benchmarks/external_baseline/bench_prolix.py"
    spec = importlib.util.spec_from_file_location("bench_prolix_xr", bench_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ns = argparse.Namespace(
        subset_dir=str(args.subset_dir),
        n_mols=args.n_mols,
        n_conf_cap=args.n_conf_cap,
        precision="float32",
        n_warmup=args.n_warmup,
        n_trials=args.n_trials,
        seed=args.seed,
        hardware_tag="cpu-local-rewire",
        out=None,
    )
    row = mod.bench_one_step(ns)
    per = float(row["per_mol_step_seconds"])
    loss = float(row["final_loss"])
    ok = per > 0.0 and 0.0 < loss < 1e10 and math.isfinite(loss)
    return {
        "probe": "prolix_smoke",
        "planner_pass": 1,
        "prolix_smoke_ran": 1,
        "gate_pass": int(ok),
        "per_mol_step_seconds": per,
        "final_loss": loss,
        "n_mols": int(row["n_mols"]),
        "skip_reason": None,
        "device": row.get("device"),
        "git_hash": row.get("git_hash"),
    }


def main() -> int:
    args = _parse_args()
    results: list[dict] = []
    if args.probe in ("planner", "all"):
        results.append(_probe_planner())
    if args.probe in ("prolix_smoke", "all"):
        results.append(_probe_prolix_smoke(args))

    primary = results[-1]
    if args.probe == "all":
        planner = next(r for r in results if r["probe"] == "planner")
        smoke = next(r for r in results if r["probe"] == "prolix_smoke")
        primary = {
            **smoke,
            "probe": "all",
            "planner_pass": planner["planner_pass"],
            "gate_pass": int(planner["planner_pass"] == 1 and smoke["gate_pass"] == 1),
        }

    primary["campaign_slug"] = "xr-parity-torch"
    primary["seed"] = args.seed
    primary = _json_safe(primary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(primary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(primary, indent=2, allow_nan=False))
    return 0 if primary.get("gate_pass", 0) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
