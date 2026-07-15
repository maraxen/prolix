#!/usr/bin/env python3
"""Diagnose B1 inference non-finite finals per topology class (read-only probe).

Runs each shape class independently with a step ladder so we can see *which*
class first goes non-finite, and contrasts water single (SETTLE on) vs stacked
(SETTLE off via integration_prefix).

Usage:
  uv run python scripts/explore/b1_finite_diag.py [max_steps] [replicas]
  # defaults: max_steps=200000, replicas=2
"""

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
    _four_water_bundle,
)
from prolix.api import EnsemblePlan  # noqa: E402
from prolix.api.bundle_stack import (  # noqa: E402
    can_jit_vmap_n_mols,
    integration_prefix_for_bundles,
)
from prolix.api.ensemble_dedup import partition_bundles_by_shape  # noqa: E402

# Ladder denser early (vacuum gate is 1k); then decade-ish toward prereg 200k.
DEFAULT_LADDER = (1, 100, 1000, 5000, 20000, 50000, 100000, 200000)


def _finite_row(label: str, traj) -> dict:
    pos = jnp.asarray(traj.positions[-1])
    fin = bool(jnp.all(jnp.isfinite(pos)))
    n_nan = int(jnp.sum(~jnp.isfinite(pos)))
    mx = float(jnp.nanmax(jnp.abs(pos))) if pos.size else float("nan")
    return {
        "label": label,
        "finite": fin,
        "max_abs": mx,
        "n_nan": n_nan,
        "shape": list(pos.shape),
    }


def _run(plan: EnsemblePlan, n_steps: int, seed: int = 0):
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
    for t in trajs:
        jax.block_until_ready(t.positions)
    return trajs, time.perf_counter() - t0


def ladder_until_fail(plan: EnsemblePlan, ladder: tuple[int, ...], label: str) -> dict:
    """Independent runs at each step count; stop after first non-finite."""
    rows = []
    first_fail = None
    for n in ladder:
        trajs, wall = _run(plan, n, seed=0)
        reps = [_finite_row(f"{label}_r{j}", t) for j, t in enumerate(trajs)]
        all_fin = all(r["finite"] for r in reps)
        row = {
            "n_steps": n,
            "wall_s": wall,
            "all_finite": all_fin,
            "replicas": reps,
        }
        rows.append(row)
        print(json.dumps({"phase": "ladder", "label": label, **row}), flush=True)
        if not all_fin:
            first_fail = n
            break
    return {"label": label, "first_fail_steps": first_fail, "ladder": rows}


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    replicas = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ladder = tuple(n for n in DEFAULT_LADDER if n <= max_steps)
    if not ladder or ladder[-1] != max_steps:
        ladder = (*ladder, max_steps)

    bundles, t_ff, t_b = _build_prolix_full_bundles(replicas)
    groups = partition_bundles_by_shape(bundles)
    meta = {
        "n_bundles": len(bundles),
        "n_groups": len(groups),
        "replicas": replicas,
        "ladder": list(ladder),
        "dt_fs": DT_FS,
        "gamma_ps": GAMMA_PS,
        "t_ff_load": t_ff,
        "t_bundle": t_b,
        "hardware": str(jax.devices()[0]),
        "groups": [],
    }
    for gi, idxs in enumerate(groups):
        g = [bundles[i] for i in idxs]
        b0 = g[0]
        meta["groups"].append(
            {
                "gi": gi,
                "B": len(g),
                "n_atoms": int(b0.n_atoms),
                "n_waters": int(b0.n_waters),
                "prefix": integration_prefix_for_bundles(g),
                "can_vmap": can_jit_vmap_n_mols(g) if len(g) > 1 else True,
                "boundary": b0.shape_spec.boundary_condition,
            }
        )
    print(json.dumps({"phase": "meta", **meta}), flush=True)

    report: dict = {"meta": meta, "per_group": [], "controls": {}}

    # Per-class stacked ladder (production path for B≥2)
    for gi, idxs in enumerate(groups):
        group = [bundles[i] for i in idxs]
        b0 = group[0]
        label = f"stack_g{gi}_n{int(b0.n_atoms)}_w{int(b0.n_waters)}_B{len(group)}"
        plan = EnsemblePlan.from_bundles(group)
        report["per_group"].append(ladder_until_fail(plan, ladder, label))

    # Water controls: single (SETTLE on) vs stacked B=2 (SETTLE off)
    water = _four_water_bundle()
    report["controls"]["water_single_settle_on"] = ladder_until_fail(
        EnsemblePlan.from_bundle(water), ladder, "water_single_SETTLE_on"
    )
    report["controls"]["water_stack_settle_off"] = ladder_until_fail(
        EnsemblePlan.from_bundles([water, water]),
        ladder,
        "water_stack_SETTLE_off",
    )

    # Smallest protein single vs stack (2GB1-class if present)
    protein_singles = []
    for gi, idxs in enumerate(groups):
        b = bundles[idxs[0]]
        if int(b.n_waters) == 0:
            lab = f"single_n{int(b.n_atoms)}"
            # Only probe up to min(5k, max) for large proteins to save walltime
            p_ladder = tuple(n for n in ladder if n <= min(max_steps, 20_000))
            if not p_ladder:
                p_ladder = (min(1000, max_steps),)
            protein_singles.append(
                ladder_until_fail(EnsemblePlan.from_bundle(b), p_ladder, lab)
            )
    report["controls"]["protein_singles"] = protein_singles

    out = ROOT / "outputs" / "explore" / "b1_finite_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"phase": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
