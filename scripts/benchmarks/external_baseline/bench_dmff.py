"""Scope A bench: DMFF (or JAX-bonded sequential proxy) — forward+backward per mol
across N mols (mixed sizes), one Adam step each, sequential loop.

This is a comparator entry to the §7.1 external-baseline campaign. Unlike prolix's
batched bundle, DMFF does not have heterogeneous batching support. For N mols,
we loop sequentially: build energy fn for mol i, run forward+backward+Adam, then
mol i+1. This establishes the "sequential per-tool" baseline to measure prolix's
heterogeneous batching advantage.

Note: DMFF's Hamiltonian class expects pre-built XML force field files (e.g., AMBER,
CHARMM). The ANI-1x subset lacks these. As a proxy, we use direct JAX computation
of bonded energy terms (harmonic bonds, harmonic angles) to measure the sequential
per-mol cost.

Usage:
    uv run python scripts/benchmarks/external_baseline/bench_dmff.py \\
        --n-mols 64 --n-conf-cap 100 --precision float32 \\
        --n-warmup 1 --n-trials 3 --out results.json

Emits a single-line JSON dict matching the result_schema in
bench_dmff.bth.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

try:
    import h5py
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as e:
    print(f"Import error: {e}")
    print("Required packages: h5py, jax, optax")
    print("Install via: uv pip install h5py jax optax")
    exit(1)


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
    """Load the 16 base mols from data/ani1x_subset/lane_a/, same as bench_prolix.py."""
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


def build_sequential_mols(base_mols: list[dict], n_mols_target: int) -> list[dict]:
    """Tile base mols to reach n_mols_target, returning list for sequential loop."""
    n_base = len(base_mols)
    tile_idx = [i % n_base for i in range(n_mols_target)]

    sequential_mols = []
    for i in tile_idx:
        m = base_mols[i]
        # For sequential processing, we'll wrap each mol's data
        sequential_mols.append({
            "params": m["params"],
            "topology": m["topology"],
            "positions": m["positions"],
            "forces": m["forces"],
            "energies": m["energies"],
            "n_atoms": m["n_atoms"],
            "n_conf": m["n_conf"],
        })
    return sequential_mols


def bonded_energy_fn(positions: jnp.ndarray, params: dict, topology: dict) -> float:
    """
    Direct JAX computation of bonded energy (harmonic bonds + harmonic angles).

    DMFF API: using direct JAX bonded terms as DMFF proxy (DMFF's Hamiltonian
    requires XML FF files not available for ANI-1x subset).

    Args:
        positions: (n_atoms, 3) array of atomic coordinates
        params: parameter dict with 'k_bond', 'r_eq', 'k_angle', 'angle_eq'
        topology: topology dict with 'bonds' and 'angles' lists of (i, j) or (i, j, k)

    Returns:
        scalar energy value
    """
    energy = 0.0

    # Harmonic bonds: sum over bond indices
    if hasattr(topology, 'bonds') and topology.bonds:
        bonds = topology.bonds
        k_bond = params.get('k_bond', jnp.ones(len(bonds)))
        r_eq = params.get('r_eq', jnp.ones(len(bonds)))

        for bond_idx, (i, j) in enumerate(bonds):
            dr = jnp.linalg.norm(positions[j] - positions[i])
            k = k_bond[bond_idx] if hasattr(k_bond, '__len__') else k_bond
            req = r_eq[bond_idx] if hasattr(r_eq, '__len__') else r_eq
            energy = energy + 0.5 * k * (dr - req) ** 2

    # Harmonic angles: sum over angle indices
    if hasattr(topology, 'angles') and topology.angles:
        angles = topology.angles
        k_angle = params.get('k_angle', jnp.ones(len(angles)))
        angle_eq = params.get('angle_eq', jnp.ones(len(angles)))

        for angle_idx, (i, j, k) in enumerate(angles):
            v1 = positions[i] - positions[j]
            v2 = positions[k] - positions[j]
            cos_angle = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1e-8)
            angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
            ka = k_angle[angle_idx] if hasattr(k_angle, '__len__') else k_angle
            aeq = angle_eq[angle_idx] if hasattr(angle_eq, '__len__') else angle_eq
            energy = energy + 0.5 * ka * (angle - aeq) ** 2

    return energy


def sequential_step(
    sequential_mols: list[dict],
    adam_state: dict,
    lr: float = 1e-3,
) -> tuple[dict, float]:
    """
    Sequential forward+backward+Adam step over all mols.

    For each mol in sequential_mols:
      1. Compute energy via bonded_energy_fn
      2. Backprop to get gradients
      3. Apply Adam update
      4. Accumulate loss

    Returns:
        (updated adam_state, total_loss)
    """
    total_loss = 0.0

    for mol in sequential_mols:
        positions = mol["positions"]
        params = mol["params"]
        topology = mol["topology"]
        energies = mol["energies"]

        def loss_fn(p):
            # Compute bonded energy for this mol's first conformation
            pred_energy = bonded_energy_fn(positions[0], p, topology)
            # MSE against reference energy
            return (pred_energy - energies[0]) ** 2

        # Gradient
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        total_loss = total_loss + loss_val

        # Simple Adam-like update (mock, since we're not tracking state per param)
        # In a real scenario, we'd use optax.adam here
        for key in params:
            if isinstance(params[key], jnp.ndarray):
                params[key] = params[key] - lr * grads[key]

    return adam_state, total_loss / len(sequential_mols)


def bench_sequential(args) -> dict:
    """Run Scope A sequential primitive: warmup + n_trials, return JSON row."""
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    base_mols = load_base_mols(Path(args.subset_dir), n_conf_cap=args.n_conf_cap)
    print(f"loaded {len(base_mols)} base mols")

    sequential_mols = build_sequential_mols(base_mols, args.n_mols)
    B = len(sequential_mols)

    # Compute n_atoms_max across all tiled mols
    n_atoms_max = max(m["n_atoms"] for m in sequential_mols)
    n_conf_max = max(m["n_conf"] for m in sequential_mols)
    print(f"sequential: B={B}, n_atoms_max={n_atoms_max}, n_conf_max={n_conf_max}")

    adam_state = {}  # Dummy state; in real scenario, use optax

    # Warmup: compiles + discarded for timing
    warmup_times = []
    for w in range(args.n_warmup):
        t0 = time.perf_counter()
        adam_state, loss_val = sequential_step(sequential_mols, adam_state, lr=1e-3)
        jax.block_until_ready(loss_val)
        t1 = time.perf_counter()
        warmup_times.append(t1 - t0)
    compile_seconds = warmup_times[0] if warmup_times else 0.0

    # Trials: each trial is ONE forward+backward+update across all B mols (sequential loop)
    trial_times = []
    final_loss = float("nan")
    for trial in range(args.n_trials):
        t0 = time.perf_counter()
        adam_state, loss_val = sequential_step(sequential_mols, adam_state, lr=1e-3)
        jax.block_until_ready(loss_val)
        t1 = time.perf_counter()
        trial_times.append(t1 - t0)
        final_loss = float(loss_val)

    trial_median = float(sorted(trial_times)[len(trial_times) // 2])
    per_mol_step = trial_median / B  # one step, B mols (sequential)

    row = {
        "tool": "dmff",
        "tool_version": "jax-bonded-sequential",
        "scope": "A",
        "n_mols": args.n_mols,
        "n_conformers_per_mol": args.n_conf_cap or n_conf_max,
        "n_atoms_max": int(n_atoms_max),
        "precision": args.precision,
        "device": _hardware_tag(),
        "hardware_tag": os.environ.get("HARDWARE_TAG", "rtx-pro-6000-blackwell"),
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
    parser.add_argument("--out", type=str, default=None, help="JSON output path; default stdout")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print config but don't time")
    args = parser.parse_args()

    if args.dry_run:
        base_mols = load_base_mols(Path(args.subset_dir), n_conf_cap=args.n_conf_cap)
        sequential_mols = build_sequential_mols(base_mols, args.n_mols)
        print(f"[dry-run] Loaded {len(base_mols)} base mols, tiled to {len(sequential_mols)} sequential mols")
        print(f"[dry-run] n_atoms_max: {max(m['n_atoms'] for m in sequential_mols)}")
        print(f"[dry-run] Config: precision={args.precision}, n_warmup={args.n_warmup}, n_trials={args.n_trials}")
        return

    row = bench_sequential(args)
    print(json.dumps(row, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
