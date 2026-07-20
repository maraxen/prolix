#!/usr/bin/env python3
"""B1 step-scaling profiler: regression-based JIT compile overhead correction.

This script measures the two-call sequence (first step + steady-state) across
varying n_steps values, then fits a linear model to extract per-step and
fixed compile costs:

    t_steady_state = compile_fixed + per_step * (n_steps - 1)

This allows Phase 4 profiling (Claim-1 per-step cost) to correct for hidden
compile costs in EnsemblePlan._run_stacked_dispatch.

Usage (dry-run sanity check, no GPU)::

    uv run python scripts/profile_b1_step_scaling.py --dry-run

Usage (profile per-class, e.g., 1AKE protein)::

    uv run python scripts/profile_b1_step_scaling.py \\
        --mode per-class --class 1ake --n-steps-list "2,10,50" \\
        --replicas 4 --out /tmp/profile_1ake.json

Usage (profile full 4-class aggregate, cluster submission)::

    uv run bth run python scripts/profile_b1_step_scaling.py \\
        --tag phase4-scaling --out outputs/profile_full.json -- \\
        --mode full --replicas 16 --n-steps-list "2,10,50,200,800"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_b1_step_scaling")

ROOT = Path(__file__).resolve().parents[2]
PDB_DIR = ROOT / "data" / "pdb"

# Prereg pins (must match b1_init_exec.py)
DT_FS = 0.5
KT_KCAL = 0.596  # ~300 K
GAMMA_PS = 50.0

PROTEIN_CLASSES = (
    ("1ake", "1AKE.pdb"),
    ("1ubq", "1UBQ.pdb"),
    ("2gb1", "2GB1.pdb"),
)


def _parse_n_steps_list(arg: str) -> list[int]:
    """Parse comma-separated int list."""
    return [int(x.strip()) for x in arg.split(",")]


def _import_b1_helpers():
    """Import the shared B1 benchmark helpers."""
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from b1_init_exec import (
        _block_trajs,
        _build_prolix_full_bundles,
        _four_water_bundle,
    )

    return _block_trajs, _build_prolix_full_bundles, _four_water_bundle


def _import_paramize():
    """Import paramize_pdb_to_bundle."""
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from _b1_paramize import paramize_pdb_to_bundle

    return paramize_pdb_to_bundle


def _build_bundles(mode: str, class_name: str | None, replicas: int):
    """Build bundle list based on mode and class."""
    _, _build_prolix_full_bundles, _four_water_bundle = _import_b1_helpers()
    paramize_pdb_to_bundle = _import_paramize()

    if mode == "full":
        bundles, t_ff, t_bundle = _build_prolix_full_bundles(replicas)
        return bundles
    elif mode == "per-class":
        if class_name is None:
            raise ValueError("--class required for --mode per-class")
        if class_name == "water":
            water = _four_water_bundle()
            return [water] * replicas
        else:
            # Find the PDB file for this class
            pdb_fname = None
            for short_name, fname in PROTEIN_CLASSES:
                if short_name == class_name:
                    pdb_fname = fname
                    break
            if pdb_fname is None:
                raise ValueError(f"Unknown protein class: {class_name}")
            path = PDB_DIR / pdb_fname
            if not path.exists():
                raise FileNotFoundError(f"Missing PDB: {path}")
            bundle = paramize_pdb_to_bundle(str(path), periodic=False)
            return [bundle] * replicas
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B1 step-scaling profiler for Phase 4 compile-regression analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sanity-check bundle construction without running steps",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "per-class"),
        default="full",
        help="Bundle construction mode: full=4-class aggregate, per-class=single class",
    )
    parser.add_argument(
        "--class",
        dest="bundle_class",
        choices=("water", "1ake", "1ubq", "2gb1"),
        default=None,
        help="Class name for --mode per-class (required if per-class, ignored if full)",
    )
    parser.add_argument(
        "--n-steps-list",
        default="2,10,50,200,800",
        help="Comma-separated step counts to profile",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=16,
        help="Replicas per class (default 16, matching B1 prereg)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args()

    # Validate mode vs. class
    if args.mode == "per-class" and args.bundle_class is None:
        parser.error("--class is required when --mode is per-class")

    # Log JAX x64 config at startup
    try:
        import jax

        log.info("jax.config.x64_enabled = %s", jax.config.x64_enabled)
    except Exception as e:
        log.warning("Could not read jax.config.x64_enabled: %s", e)

    n_steps_list = _parse_n_steps_list(args.n_steps_list)
    log.info(
        "Profile mode=%s, class=%s, replicas=%d, n_steps=%s",
        args.mode,
        args.bundle_class or "n/a",
        args.replicas,
        n_steps_list,
    )

    # Build bundles
    try:
        bundles = _build_bundles(args.mode, args.bundle_class, args.replicas)
    except Exception as e:
        log.error("Bundle construction failed: %s", e)
        return 1

    bundle_info = {
        "n_bundles": len(bundles),
        "dtype": str(bundles[0].positions.dtype) if bundles else "unknown",
    }
    log.info("Built %d bundles, dtype=%s", len(bundles), bundle_info["dtype"])

    if args.dry_run:
        summary = {
            "mode": "dry-run",
            "shape_class": args.bundle_class or "all-4-aggregate",
            "bundle_info": bundle_info,
        }
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2) + "\n")
        return 0

    # Regression profiling
    from prolix.api import EnsemblePlan

    _block_trajs, _, _ = _import_b1_helpers()

    results = []
    for n_steps in n_steps_list:
        try:
            # Fresh EnsemblePlan per iteration
            plan = EnsemblePlan.from_bundles(bundles)

            # First step (proxy for compile)
            t_first0 = time.perf_counter()
            first = plan.run(
                n_steps=1,
                dt=DT_FS,
                kT=KT_KCAL,
                seed=0,
                gamma=GAMMA_PS,
                run_mode="inference",
            )
            _block_trajs(first)
            t_first = time.perf_counter() - t_first0
            del first

            # Steady-state (only if n_steps > 1)
            t_ss = 0.0
            if n_steps > 1:
                t_ss0 = time.perf_counter()
                last = plan.run(
                    n_steps=n_steps - 1,
                    dt=DT_FS,
                    kT=KT_KCAL,
                    seed=1,
                    gamma=GAMMA_PS,
                    run_mode="inference",
                )
                _block_trajs(last)
                t_ss = time.perf_counter() - t_ss0
                del last

            result = {
                "n_steps": n_steps,
                "t_first_step": t_first,
                "t_steady_state": t_ss,
            }
            results.append(result)
            log.info(
                "n_steps=%d: t_first=%.4f s, t_steady_state=%.4f s",
                n_steps,
                t_first,
                t_ss,
            )
        except Exception as e:
            log.error("Failed to run n_steps=%d: %s", n_steps, e)
            return 1

    # Fit regression: t_steady_state = compile_fixed + per_step * (n_steps - 1)
    x_vals = []
    y_vals = []
    for r in results:
        if r["n_steps"] > 1:
            x_vals.append(r["n_steps"] - 1)
            y_vals.append(r["t_steady_state"])

    slope = float("nan")
    intercept = float("nan")
    r_squared = float("nan")

    if len(x_vals) >= 2:
        try:
            x_arr = np.array(x_vals, dtype=np.float64)
            y_arr = np.array(y_vals, dtype=np.float64)
            coeffs = np.polyfit(x_arr, y_arr, 1)
            slope = float(coeffs[0])  # per_step
            intercept = float(coeffs[1])  # compile_fixed

            # Compute R²
            y_pred = slope * x_arr + intercept
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            if ss_tot > 0:
                r_squared = float(1.0 - ss_res / ss_tot)
            else:
                r_squared = float("nan")

            log.info(
                "Regression: per_step=%.6e s, compile_fixed=%.6e s, R²=%.6f",
                slope,
                intercept,
                r_squared,
            )
        except Exception as e:
            log.warning("Regression fit failed: %s", e)
    else:
        log.warning("Insufficient data points (< 2) for regression fit")

    summary = {
        "mode": args.mode,
        "shape_class": args.bundle_class or "all-4-aggregate",
        "replicas": args.replicas,
        "per_step_corrected_s": slope,
        "compile_fixed_s": intercept,
        "r_squared": r_squared,
        "raw_results": results,
    }

    # Output
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        log.info("Wrote summary to %s", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
