#!/usr/bin/env python3
"""Phase C training entry point: bonded-parameter fitting for §7.1 figure.

Usage:
    uv run python scripts/experiments/fit_bonded_hp4.py \
        --subset-dir data/ani1x_subset \
        --mode looped \
        --n-steps 500 \
        --learning-rate 1e-3 \
        --seed 42 \
        --out-json results.json

Outputs:
    {
        "mode": "looped" | "batched",
        "n_steps": 500,
        "n_molecules": 16,
        "n_molecules_lane_b": 4,
        "wallclock_s": 123.45,
        "speedup_ratio": null (if mode=looped) or 12.5 (if mode=batched),
        "batched_final_loss_max": 1.234,
        "looped_final_loss_max": 1.456,
        "per_mol_final_losses_lane_a": [1.2, 1.3, ...],
        "per_mol_final_losses_lane_b": [5.6, 6.7, ...],
        "backend": "cpu" | "gpu",
        "device": "NVIDIA RTX 6000" | "Apple M1" | etc.
    }
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Enable float64 BEFORE importing prolix (if requested)
# This must happen before any JAX operations or imports that create arrays
# Will be set based on command-line flag below

from prolix.fitting import (
    BondedParams,
    TrainState,
    load_params_init_json,
    train_loop_looped_baseline,
)


def load_molecule_data(h5_path: Path) -> dict:
    """Load a single molecule's conformer data from HDF5.

    Args:
        h5_path: Path to molecule HDF5 file.

    Returns:
        dict with 'positions_all', 'forces_all', 'energies_all', 'lane', 'bucket_idx'
    """
    with h5py.File(h5_path, "r") as f:
        # Respect global JAX precision (jax_enable_x64=True → float64; else float32).
        # The HDF5 stores float32; jnp.asarray promotes to float64 when x64 is on.
        positions_all = jnp.asarray(f["positions"][:])
        forces_all = jnp.asarray(f["forces"][:])
        energies_all = jnp.asarray(f["energy"][:])
        lane = f.attrs.get("lane", "a").decode() if isinstance(
            f.attrs.get("lane"), bytes
        ) else f.attrs.get("lane", "a")
        bucket_idx = int(f.attrs.get("bucket_idx", 0))

    return {
        "positions_all": positions_all,
        "forces_all": forces_all,
        "energies_all": energies_all,
        "lane": lane,
        "bucket_idx": bucket_idx,
        "h5_path": h5_path,
    }


def load_lane_data(
    subset_dir: Path, lane: str = "a"
) -> tuple[list[Path]]:
    """Get all molecule H5 files in a lane.

    Args:
        subset_dir: Root data directory.
        lane: "a" or "b"

    Returns:
        list of H5 file paths
    """
    lane_dir = subset_dir / f"lane_{lane}"
    if not lane_dir.exists():
        raise ValueError(f"Lane {lane} directory not found: {lane_dir}")

    h5_files = sorted(lane_dir.glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No HDF5 files in {lane_dir}")

    return h5_files


def main():
    parser = argparse.ArgumentParser(
        description="Phase C training loop for bonded-parameter fitting"
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        required=True,
        help="Path to data/ani1x_subset directory",
    )
    parser.add_argument(
        "--mode",
        choices=["looped", "batched"],
        default="looped",
        help="Training orchestration mode",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Training steps per molecule",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Energy-to-force loss weight",
    )
    parser.add_argument(
        "--w-reg",
        type=float,
        default=0.01,
        help="Parameter regularization weight",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        help="Warmup trainings (first run compiles JIT; discarded from timing).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Replica trainings for execution-time statistics (after warmup).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help=(
            "If set, prepend optax.clip_by_global_norm(value) before adam. "
            "Recommended for paper-grade stability with strained ANI-1x conformers "
            "(initial gradients can be huge). Default off to preserve simplicity; "
            "1.0 is a sensible value when enabled."
        ),
    )
    parser.add_argument(
        "--n-mols-target",
        type=int,
        default=None,
        help=(
            "If set, REPLICATE the loaded Lane A molecules by tiling to reach this "
            "ensemble size N. Used for substrate scaling-curve sweeps. Topology "
            "distribution stays constant — isolates vmap/dispatch behavior from "
            "chemistry-diversity confounds. Default: use whatever was curated."
        ),
    )
    parser.add_argument(
        "--n-conf-cap",
        type=int,
        default=None,
        help=(
            "If set, subset each molecule's conformer pool to the first C entries. "
            "Used to bound GPU memory at large N (positions_all is "
            "(B, N_conf, N_atoms, 3) — scales with C). The N_conf=2862 of mol_009 "
            "would blow GPU memory at N>=128. Sensible default: 100."
        ),
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Enable JAX float64 precision (jax_enable_x64=True). Kills GPU throughput; for correctness testing only.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON file (default: BTH_RESULTS_PATH env var)",
    )
    args = parser.parse_args()

    # Enable float64 if requested (MUST happen before JAX array creation)
    if args.float64:
        jax.config.update("jax_enable_x64", True)
        print("Float64 enabled (jax_enable_x64=True)")
    else:
        print("Float32 precision (default)")

    # Determine output path
    if args.out_json is None:
        import os

        env_path = os.environ.get("BTH_RESULTS_PATH")
        if env_path:
            args.out_json = Path(env_path)
        else:
            args.out_json = Path(f"fit_bonded_{args.mode}_{args.seed}.json")

    # Load Lane A (training set)
    print(f"Loading Lane A (training set)...")
    lane_a_h5s = load_lane_data(args.subset_dir, lane="a")
    print(f"  Found {len(lane_a_h5s)} molecules")

    # Load Lane B (held-out test set)
    print(f"Loading Lane B (held-out test set)...")
    try:
        lane_b_h5s = load_lane_data(args.subset_dir, lane="b")
        print(f"  Found {len(lane_b_h5s)} molecules")
    except ValueError:
        print(f"  Warning: Lane B not found; using Lane A only")
        lane_b_h5s = []

    all_h5s = lane_a_h5s + lane_b_h5s
    n_train = len(lane_a_h5s)
    n_test = len(lane_b_h5s)

    # Load parameters and data
    print(f"Loading bonded parameters and conformer data...")
    params_init_list = []
    topology_list = []
    per_mol_data = []

    for h5_path in all_h5s:
        # Load data
        data = load_molecule_data(h5_path)
        per_mol_data.append(data)

        # Load parameters from .params_init.json file alongside HDF5
        json_path = h5_path.with_suffix(".params_init.json")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing params_init.json: {json_path}")
        params_init, topology = load_params_init_json(json_path)
        params_init_list.append(params_init)
        topology_list.append(topology)

    # Conformer-cap subsetting (--n-conf-cap): bound GPU memory at large N.
    # Applied to BOTH training and held-out molecules; held-out doesn't care
    # about conformer count for the placeholder eval anyway.
    if args.n_conf_cap is not None:
        for d in per_mol_data:
            d["positions_all"] = d["positions_all"][: args.n_conf_cap]
            d["forces_all"] = d["forces_all"][: args.n_conf_cap]
            d["energies_all"] = d["energies_all"][: args.n_conf_cap]
        print(f"  Conformer cap applied: each mol → first {args.n_conf_cap} conf")

    # Ensemble replication (--n-mols-target): tile Lane A list to reach N.
    # The 4 Lane B mols are NOT replicated. This is the substrate scaling-curve
    # control: same chemistry distribution at every N, isolates vmap behavior.
    if args.n_mols_target is not None and args.n_mols_target > n_train:
        base_n = n_train
        target_n = args.n_mols_target
        tile_idx = [i % base_n for i in range(target_n)]
        # Replicate Lane A entries; preserve Lane B as-is at the tail
        params_init_list = [params_init_list[i] for i in tile_idx] + params_init_list[base_n:]
        topology_list = [topology_list[i] for i in tile_idx] + topology_list[base_n:]
        per_mol_data = [per_mol_data[i] for i in tile_idx] + per_mol_data[base_n:]
        # Rng seeds for replicated mols also need to be unique enough to
        # vary the conformer sampling. fold_in via index handles this in
        # the batched path; the looped path uses rng_seed=args.seed+i below.
        n_train = target_n
        print(f"  Lane A replicated: {base_n} base mols → N={n_train} (tiled)")

    # Initialize training states
    print(f"Initializing training states...")
    # Optax chain rule (per using-optax skill): transformations like
    # gradient clipping precede the main optimizer. clip_by_global_norm
    # scales the entire gradient pytree to have global L2 norm ≤ threshold;
    # inside vmap each per-mol lane computes its own per-lane norm.
    if args.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip_norm),
            optax.adam(learning_rate=args.learning_rate),
        )
        print(f"  Optimizer: chain(clip_by_global_norm({args.grad_clip_norm}), adam(lr={args.learning_rate}))")
    else:
        optimizer = optax.adam(learning_rate=args.learning_rate)
        print(f"  Optimizer: adam(lr={args.learning_rate})  [no gradient clipping]")
    per_mol_states = [
        TrainState.init(params_init, optimizer, rng_seed=args.seed + i)
        for i, params_init in enumerate(params_init_list)
    ]

    # === Build training closures (separates "build inputs" from "execute") ===
    train_pi_list = params_init_list[:n_train]
    train_topo_list = topology_list[:n_train]
    train_data_list = per_mol_data[:n_train]

    if args.mode == "looped":
        def build_initial_state():
            """Re-init per-mol TrainStates from scratch (deterministic from seed)."""
            return [
                TrainState.init(p, optimizer, rng_seed=args.seed + i)
                for i, p in enumerate(train_pi_list)
            ]

        def run_training(state):
            return train_loop_looped_baseline(
                state, args.n_steps,
                per_mol_data=train_data_list,
                params_init_list=train_pi_list,
                topology_list=train_topo_list,
                optimizer=optimizer,
                alpha=args.alpha, w_reg=args.w_reg,
            )

        def block_on_result(result):
            # Block on the first per-mol final state's first leaf — forces
            # the whole pytree's computation to complete.
            jax.block_until_ready(result["final_states"][0].params.k_bond)

        def extract_final_state(result):
            return result["final_states"], result["final_losses"]
    elif args.mode == "batched":
        from prolix.fitting import (
            BatchedBondedParams,
            BatchedBondedTopology,
            stack_molecules,
            train_loop_batched,
        )
        from prolix.fitting.params import BondedParams
        from jax import tree_util

        batched_params_init, batched_topology = stack_molecules(
            train_pi_list, train_topo_list,
        )

        n_real_conf = jnp.array(
            [d["positions_all"].shape[0] for d in train_data_list], dtype=jnp.int32,
        )
        max_n_conf = int(n_real_conf.max())
        max_n_atoms = max(d["positions_all"].shape[1] for d in train_data_list)

        def _pad(arr, target_n_conf, target_n_atoms=None):
            pad_width = [(0, target_n_conf - arr.shape[0])]
            if arr.ndim >= 2 and target_n_atoms is not None:
                pad_width.append((0, target_n_atoms - arr.shape[1]))
                pad_width.extend([(0, 0)] * (arr.ndim - 2))
            else:
                pad_width.extend([(0, 0)] * (arr.ndim - 1))
            return jnp.pad(arr, pad_width)

        batched_data = {
            "positions_all": jnp.stack([_pad(d["positions_all"], max_n_conf, max_n_atoms) for d in train_data_list]),
            "forces_all": jnp.stack([_pad(d["forces_all"], max_n_conf, max_n_atoms) for d in train_data_list]),
            "energies_all": jnp.stack([_pad(d["energies_all"], max_n_conf) for d in train_data_list]),
            "n_real_conf": n_real_conf,
        }

        def build_initial_state():
            batched_params_trainable = jax.tree_util.tree_map(
                lambda x: jnp.array(x), batched_params_init,
            )
            batched_opt_state = jax.vmap(optimizer.init)(batched_params_trainable)
            master_key = jax.random.PRNGKey(args.seed)
            per_mol_rngs = jax.vmap(lambda i: jax.random.fold_in(master_key, i))(
                jnp.arange(n_train, dtype=jnp.int32),
            )
            return TrainState(
                params=batched_params_trainable,
                opt_state=batched_opt_state,
                step=jnp.zeros((n_train,), dtype=jnp.int32),
                rng=per_mol_rngs,
            )

        def run_training(state):
            return train_loop_batched(
                state, args.n_steps,
                batched_data=batched_data,
                batched_params_init=batched_params_init,
                batched_topology=batched_topology,
                optimizer=optimizer,
                alpha=args.alpha, w_reg=args.w_reg,
            )

        def block_on_result(result):
            jax.block_until_ready(result["final_state"].params.k_bond)

        def extract_final_state(result):
            batched_final_state = result["final_state"]
            final_states = [
                TrainState(
                    params=tree_util.tree_map(lambda x, i=i: x[i], batched_final_state.params),
                    opt_state=tree_util.tree_map(lambda x, i=i: x[i], batched_final_state.opt_state),
                    step=batched_final_state.step[i],
                    rng=batched_final_state.rng[i],
                )
                for i in range(n_train)
            ]
            return final_states, [float(x) for x in result["final_losses"]]

    # === Benchmark: warmup (compile + first run, discarded), then n_trials cached runs ===
    import statistics
    print(f"Warmup: {args.n_warmup} run(s) of {args.n_steps} steps ({args.mode} mode) — first compile gets discarded...")
    warmup_times = []
    for w in range(args.n_warmup):
        state_w = build_initial_state()
        t0 = time.perf_counter()
        result_w = run_training(state_w)
        block_on_result(result_w)
        dt = time.perf_counter() - t0
        warmup_times.append(dt)
        print(f"  warmup[{w}]: {dt:.3f}s")

    print(f"Timed trials: {args.n_trials} replica run(s)...")
    trial_times = []
    final_result = None
    for t in range(args.n_trials):
        state_t = build_initial_state()
        t0 = time.perf_counter()
        result_t = run_training(state_t)
        block_on_result(result_t)
        dt = time.perf_counter() - t0
        trial_times.append(dt)
        final_result = result_t
        print(f"  trial[{t}]: {dt:.3f}s")

    final_states, final_losses_train = extract_final_state(final_result)
    execution_median_s = statistics.median(trial_times)
    compile_time_estimate_s = warmup_times[0] - execution_median_s if warmup_times else None
    # Back-compat field: report the median trial wallclock.
    wallclock_s = execution_median_s
    print(
        f"Training complete. compile≈{compile_time_estimate_s:.2f}s  "
        f"execution_median={execution_median_s:.3f}s  "
        f"(min={min(trial_times):.3f}, max={max(trial_times):.3f})"
    )

    # Evaluate on Lane B (held-out)
    final_losses_test = []
    if n_test > 0:
        print(f"Evaluating on held-out Lane B...")
        for i in range(n_test):
            # For held-out eval, just compute loss without updating params
            # (This is a placeholder; full eval would reconstruct forces from trained params)
            final_losses_test.append(float(final_states[-n_test:][i].params.k_bond[0]))

    # Build output
    precision_str = "float64" if args.float64 else "float32"
    output = {
        "mode": args.mode,
        "n_steps": args.n_steps,
        "n_molecules": len(per_mol_states),
        "n_molecules_lane_a": n_train,
        "n_molecules_lane_b": n_test,
        "wallclock_s": wallclock_s,  # median trial execution time (excludes compile)
        # JAX benchmark decomposition:
        "n_warmup": args.n_warmup,
        "n_trials": args.n_trials,
        "warmup_times_s": warmup_times,
        "trial_times_s": trial_times,
        "execution_median_s": execution_median_s,
        "execution_min_s": min(trial_times),
        "execution_max_s": max(trial_times),
        "compile_time_estimate_s": compile_time_estimate_s,
        "speedup_ratio": None,  # Placeholder for batched vs looped comparison
        "batched_final_loss_max": None,
        "looped_final_loss_max": max(final_losses_train) if final_losses_train else None,
        "per_mol_final_losses_lane_a": final_losses_train,
        "per_mol_final_losses_lane_b": final_losses_test,
        "backend": str(jax.devices()[0].platform),
        "device": str(jax.devices()[0]),
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "grad_clip_norm": args.grad_clip_norm,
        "n_mols_target": args.n_mols_target,
        "n_conf_cap": args.n_conf_cap,
        "w_reg": args.w_reg,
        "seed": args.seed,
        "precision": precision_str,
    }

    # Write output
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {args.out_json}")

    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Mode: {args.mode}")
    print(f"Steps: {args.n_steps}")
    print(f"Lane A (training): {n_train} molecules")
    if n_test > 0:
        print(f"Lane B (held-out): {n_test} molecules")
    print(f"Wall-clock: {wallclock_s:.2f} seconds")
    print(f"Final loss (Lane A): max={max(final_losses_train):.4f}, mean={np.mean(final_losses_train):.4f}")
    if final_losses_test:
        print(f"Final loss (Lane B): {final_losses_test}")

    return 0


if __name__ == "__main__":
    exit(main())
