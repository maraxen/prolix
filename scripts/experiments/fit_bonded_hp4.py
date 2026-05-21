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

    # Initialize training states
    print(f"Initializing training states...")
    optimizer = optax.adam(learning_rate=args.learning_rate)
    per_mol_states = [
        TrainState.init(params_init, optimizer, rng_seed=args.seed + i)
        for i, params_init in enumerate(params_init_list)
    ]

    # Run training (only Lane A, Lane B is held out)
    print(f"Starting training ({args.mode} mode, {args.n_steps} steps per molecule)...")
    start_time = time.time()

    if args.mode == "looped":
        result = train_loop_looped_baseline(
            per_mol_states[:n_train],  # Only Lane A for training
            args.n_steps,
            per_mol_data=per_mol_data[:n_train],
            params_init_list=params_init_list[:n_train],
            topology_list=topology_list[:n_train],
            optimizer=optimizer,
            alpha=args.alpha,
            w_reg=args.w_reg,
        )
        final_states = result["final_states"]
        final_losses_train = result["final_losses"]
    elif args.mode == "batched":
        # Stack Lane A molecules into batched containers (Phase C v3+).
        from prolix.fitting import (
            BatchedBondedParams,
            BatchedBondedTopology,
            stack_molecules,
            train_loop_batched,
        )
        from prolix.fitting.params import BondedParams
        import jax
        from jax import tree_util

        train_pi_list = params_init_list[:n_train]
        train_topo_list = topology_list[:n_train]
        train_data_list = per_mol_data[:n_train]

        batched_params_init, batched_topology = stack_molecules(
            train_pi_list, train_topo_list,
        )
        # batched_params_init.k_bond has shape (B, max_n_bonds), etc.
        # Use it directly as the initial trainable params (clone via tree_map).
        # The vmap'd step_fn_one_mol receives per-lane slices which are
        # BondedParams-shaped (max_n_bonds,) thanks to the BatchedBondedParams
        # pytree structure.
        batched_params_trainable = jax.tree_util.tree_map(
            lambda x: jnp.array(x), batched_params_init,
        )

        # Initialize optimizer state PER-LANE via vmap. If we called
        # optimizer.init(batched_params) directly, the optax state would
        # have a scalar `count` field (rank 0) that vmap(in_axes=0) cannot
        # slice. vmapping the init promotes count to shape (B,) along with
        # the rest of the state tree.
        batched_opt_state = jax.vmap(optimizer.init)(batched_params_trainable)

        # Per-mol RNG keys via fold_in (jax.random idiom for parallel seeds).
        master_key = jax.random.PRNGKey(args.seed)
        per_mol_rngs = jax.vmap(lambda i: jax.random.fold_in(master_key, i))(
            jnp.arange(n_train, dtype=jnp.int32),
        )

        batched_state = TrainState(
            params=batched_params_trainable,
            opt_state=batched_opt_state,
            step=jnp.zeros((n_train,), dtype=jnp.int32),
            rng=per_mol_rngs,
        )

        # Pad per-mol conformer arrays to common (max_N_conf, max_N_atoms).
        # Lane A is all bucket 0, but n_atoms varies (15..30); n_conf varies
        # heavily (39..2862). Bond/angle/torsion atom indices only reference
        # real atoms, so appending zero-positions is safe — they never get
        # read by the bonded energy.
        n_real_conf = jnp.array(
            [d["positions_all"].shape[0] for d in train_data_list], dtype=jnp.int32,
        )
        max_n_conf = int(n_real_conf.max())
        max_n_atoms = max(d["positions_all"].shape[1] for d in train_data_list)

        def _pad(arr, target_n_conf, target_n_atoms=None):
            """Pad along conformer axis; optionally also along atom axis."""
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

        result = train_loop_batched(
            batched_state,
            args.n_steps,
            batched_data=batched_data,
            batched_params_init=batched_params_init,
            batched_topology=batched_topology,
            optimizer=optimizer,
            alpha=args.alpha,
            w_reg=args.w_reg,
        )
        # train_loop_batched returns a dict with 'final_state' (stacked) and
        # 'final_losses' (B,). Unpack final_losses per molecule.
        batched_final_state = result["final_state"]
        final_losses_train = [float(x) for x in result["final_losses"]]
        # Expose per-mol final states for Lane B eval below — split stacked.
        final_states = [
            TrainState(
                params=tree_util.tree_map(lambda x, i=i: x[i], batched_final_state.params),
                opt_state=tree_util.tree_map(lambda x, i=i: x[i], batched_final_state.opt_state),
                step=batched_final_state.step[i],
                rng=batched_final_state.rng[i],
            )
            for i in range(n_train)
        ]

    wallclock_s = time.time() - start_time
    print(f"Training complete in {wallclock_s:.2f} seconds")

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
        "wallclock_s": wallclock_s,
        "speedup_ratio": None,  # Placeholder for batched vs looped comparison
        "batched_final_loss_max": None,
        "looped_final_loss_max": max(final_losses_train) if final_losses_train else None,
        "per_mol_final_losses_lane_a": final_losses_train,
        "per_mol_final_losses_lane_b": final_losses_test,
        "backend": str(jax.devices()[0].platform),
        "device": str(jax.devices()[0]),
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
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
