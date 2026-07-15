#!/usr/bin/env python3
"""Reproduce B1 prereg EnsemblePlan (B=64) and report per-system finite."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from b1_init_exec import (  # noqa: E402
    DT_FS,
    GAMMA_PS,
    KT_KCAL,
    _build_prolix_full_bundles,
)
from prolix.api import EnsemblePlan  # noqa: E402
from prolix.api.ensemble_dedup import partition_bundles_by_shape  # noqa: E402


def main() -> None:
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    replicas = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    bundles, *_ = _build_prolix_full_bundles(replicas)
    groups = partition_bundles_by_shape(bundles)
    print(
        json.dumps(
            {
                "n": len(bundles),
                "groups": [len(g) for g in groups],
                "n_steps": n_steps,
                "seed": seed,
                "hw": str(jax.devices()[0]),
            }
        ),
        flush=True,
    )

    plan = EnsemblePlan.from_bundles(bundles)
    t0 = time.perf_counter()
    trajs = plan.run(
        n_steps=n_steps,
        dt=DT_FS,
        kT=KT_KCAL,
        seed=seed,
        gamma=GAMMA_PS,
        run_mode="inference",
    )
    if not isinstance(trajs, list):
        trajs = [trajs]

    rows = []
    for i, t in enumerate(trajs):
        pos = jnp.asarray(t.positions[-1])
        jax.block_until_ready(pos)
        fin = bool(jnp.all(jnp.isfinite(pos)))
        n_nan = int(jnp.sum(~jnp.isfinite(pos)))
        mx = float(jnp.nanmax(jnp.abs(pos))) if pos.size else float("nan")
        b = bundles[i]
        row = {
            "i": i,
            "n_atoms": int(b.n_atoms),
            "n_waters": int(b.n_waters),
            "finite": fin,
            "n_nan": n_nan,
            "max_abs": mx,
            "shape": list(pos.shape),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    bad = [r for r in rows if not r["finite"]]
    by_class: dict[str, list[float]] = {}
    for r in rows:
        key = f"n{r['n_atoms']}_w{r['n_waters']}"
        by_class.setdefault(key, []).append(r["max_abs"])

    out = {
        "wall_s": time.perf_counter() - t0,
        "n_bad": len(bad),
        "all_finite": len(bad) == 0,
        "bad": bad,
        "max_abs_by_class": {
            k: {"max": max(v), "min": min(v), "n": len(v)} for k, v in by_class.items()
        },
    }
    path = ROOT / "outputs" / "explore" / "b1_finite_full64.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "phase": "done",
                "n_bad": out["n_bad"],
                "all_finite": out["all_finite"],
                "wall_s": out["wall_s"],
                "max_abs_by_class": out["max_abs_by_class"],
                "out": str(path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
