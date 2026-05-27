"""Scope A bench: prolix v1.2 Bundle code — forward+backward of bonded energy
across N mols (mixed sizes), one Adam step.

This is the prolix entry to the §7.1 external-baseline campaign. Each comparator
(bench_dmff.py, bench_torchmd.py, bench_espaloma.py, bench_pytorch_scratch.py)
emits a JSON row with the same schema, and the campaign analysis joins on
(tool, n_mols, precision) to produce the comparison table.

Usage:
    uv run python scripts/benchmarks/external_baseline/bench_prolix.py \\
        --n-mols 64 --n-conf-cap 100 --precision float32 \\
        --n-warmup 1 --n-trials 3 --out results.json

Emits a single-line JSON dict matching the result_schema in
hp4_s71_external_baseline.bth.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        return "unknown"


def _hardware_tag() -> str:
    try:
        d = jax.devices()[0]
        return f"{d.device_kind}"
    except Exception:
        return "cpu"


def _peak_memory_mib() -> float:
    try:
        stats = jax.devices()[0].memory_stats()
        # Use peak_bytes_in_use if available, else fall back to current
        peak = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
        return float(peak) / (1024 * 1024)
    except Exception:
        return -1.0


def load_base_mols(subset_dir: Path, n_conf_cap: int | None = None) -> list[dict]:
    """Load the 16 base mols from data/ani1x_subset/lane_a/."""
    from prolix.fitting.init import load_params_init_json

    lane_a = subset_dir / "lane_a"
    mol_files = sorted(lane_a.glob("mol_*.params_init.json"))
    mols = []
    for f in mol_files:
        params, topology = load_params_init_json(f)
        h5_path = lane_a / f.name.replace(".params_init.json", ".h5")
        with h5py.File(h5_path, "r") as h:
            n = h["positions"].shape[0]
            cap = min(n, n_conf_cap) if n_conf_cap else n
            positions = jnp.asarray(h["positions"][:cap])
            forces = jnp.asarray(h["forces"][:cap])
            energies = jnp.asarray(h["energy"][:cap])
        mols.append({
            "params": params,
            "topology": topology,
            "positions": positions,
            "forces": forces,
            "energies": energies,
            "n_atoms": positions.shape[1],
            "n_conf": positions.shape[0],
        })
    return mols


def build_batched_bundle(base_mols: list[dict], n_mols_target: int):
    """Tile base mols to reach n_mols_target. Same methodology as fit_bonded_hp4."""
    from prolix.fitting import (
        BatchedFittingBundle,
        build_fitting_bundle,
    )

    n_base = len(base_mols)
    tile_idx = [i % n_base for i in range(n_mols_target)]

    bundles = []
    for i in tile_idx:
        m = base_mols[i]
        bundle = build_fitting_bundle(
            positions_all=m["positions"],
            forces_all=m["forces"],
            energies_all=m["energies"],
            params=m["params"],
            topology=m["topology"],
            n_conf_real=m["n_conf"],
        )
        bundles.append(bundle)
    return BatchedFittingBundle.stack(bundles)


def bench_one_step(args) -> dict:
    """Run Scope A primitive: warmup + n_trials, return JSON row."""
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    from prolix.fitting import (
        FittingConfig,
        make_fitting_plan,
    )
    from prolix.fitting.bundles import TrainState

    base_mols = load_base_mols(Path(args.subset_dir), n_conf_cap=args.n_conf_cap)
    print(f"loaded {len(base_mols)} base mols")
    batched_bundle = build_batched_bundle(base_mols, args.n_mols)
    B = batched_bundle.conformers_batched.n_mols
    n_atoms_max = batched_bundle.conformers_batched.max_n_atoms
    n_conf_max = batched_bundle.conformers_batched.max_n_conf
    print(f"batched: B={B}, n_atoms_max={n_atoms_max}, n_conf_max={n_conf_max}")

    config = FittingConfig(lr=1e-3, n_steps=1, alpha=0.25, w_reg=0.01)
    plan = make_fitting_plan(config)
    opt_state = plan.optimizer.init(batched_bundle.params_batched)
    state = TrainState(
        params=batched_bundle.params_batched,
        opt_state=opt_state,
        key=jax.random.PRNGKey(args.seed),
        step_count=jnp.zeros((), dtype=jnp.int32),
    )

    # Warmup: compiles + discarded for timing
    warmup_times = []
    for w in range(args.n_warmup):
        t0 = time.perf_counter()
        cidx = jnp.asarray(0, dtype=jnp.int32)
        s, m = plan.step(batched_bundle, state, cidx)
        jax.block_until_ready(s.params.k_bond)
        t1 = time.perf_counter()
        warmup_times.append(t1 - t0)
    compile_seconds = warmup_times[0] if warmup_times else 0.0

    # Trials: each trial is ONE forward+backward+update (Scope A = primitive)
    trial_times = []
    final_loss = float("nan")
    for trial in range(args.n_trials):
        t0 = time.perf_counter()
        cidx = jnp.asarray(0, dtype=jnp.int32)
        s, m = plan.step(batched_bundle, state, cidx)
        jax.block_until_ready(s.params.k_bond)
        t1 = time.perf_counter()
        trial_times.append(t1 - t0)
        final_loss = float(m.loss)

    trial_median = float(sorted(trial_times)[len(trial_times) // 2])
    per_mol_step = trial_median / B  # one step, B mols

    row = {
        "tool": "prolix",
        "tool_version": "v1.2-bundle",
        "scope": "A",
        "n_mols": args.n_mols,
        "n_conformers_per_mol": args.n_conf_cap or n_conf_max,
        "n_atoms_max": int(n_atoms_max),
        "precision": args.precision,
        "device": _hardware_tag(),
        "hardware_tag": args.hardware_tag,
        "trial_seconds": trial_median,
        "trial_min_seconds": float(min(trial_times)),
        "trial_max_seconds": float(max(trial_times)),
        "per_mol_step_seconds": per_mol_step,
        "peak_memory_mib": _peak_memory_mib(),
        "compile_seconds": compile_seconds,
        "final_loss": final_loss,
        "git_hash": _git_hash(),
        "n_warmup": args.n_warmup,
        "n_trials": args.n_trials,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=str, default="data/ani1x_subset")
    parser.add_argument("--n-mols", type=int, default=64)
    parser.add_argument("--n-conf-cap", type=int, default=100)
    parser.add_argument("--precision", type=str, choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware-tag", default=os.environ.get("HARDWARE_TAG", "rtx-pro-6000-blackwell"))
    parser.add_argument("--out", type=str, default=None, help="JSON output path; default stdout")
    args = parser.parse_args()

    row = bench_one_step(args)
    print(json.dumps(row, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
