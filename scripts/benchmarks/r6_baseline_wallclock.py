"""R6 baseline wall-clock smoke (HP4 §9 R6 / roadmap §7.1 falsification trigger).

Measures the looped-baseline-vs-jit(vmap) wall-clock ratio for a same-bucket
ensemble of N MolecularBundles, using a synthetic harmonic-pairwise energy
(O(N_atoms^2) compute per bundle, matching real force-field cost scaling).

Reports:
    - looped_wallclock_s: median of M trials of N sequential jax.grad calls
    - batched_wallclock_s: median of M trials of one jit(vmap(jax.grad)) call
    - speedup_ratio = looped / batched

§7.1 falsification trigger (roadmap §2.7 RR7):
    speedup_ratio < 10x → §7.1 multiplicativity story collapses.
    HP4 R6 mitigation: if ratio < 10x on N=16, escalate ensemble to N=64
    before §7.1 implementation begins.

Usage (local CPU smoke, --synthetic for portability):
    uv run python scripts/cluster/r6_baseline_wallclock.py \\
        --synthetic --n-molecules 16 --n-atoms 30 --n-trials 5

Usage (cluster GPU, post-rsync of HP4 curated subset):
    uv run python scripts/cluster/r6_baseline_wallclock.py \\
        --data-dir data/ani1x_subset/lane_a --n-trials 5 \\
        --out-json outputs/results/r6_baseline.json

Exit codes:
    0  benchmark complete, speedup >= 10x
    4  speedup < 10x — R6 falsification trigger, scale ensemble before §7.1
    1  data error (missing files, atoms vary across data-dir)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Falsification trigger from roadmap §2.7 RR7 (HP4 R6 mitigation).
SPEEDUP_FLOOR = 10.0


def _stub_energy(positions: jax.Array) -> jax.Array:
    """O(N_atoms^2) harmonic pairwise energy. Matches real-FF compute scaling."""
    diff = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    return 0.5 * jnp.sum(r2)


def _make_stub_system(positions: jax.Array, species: jax.Array):
    """Duck-typed system for make_bundle_from_system."""
    from types import SimpleNamespace

    return SimpleNamespace(
        positions=positions, species=species,
        masses=jnp.ones(positions.shape[0], dtype=jnp.float32) * 12.0,
        bonds=None, bond_params=None,
        angles=None, angle_params=None,
        dihedrals=None, dihedral_params=None,
        impropers=None, improper_params=None,
        urey_bradley_bonds=None, urey_bradley_params=None,
        water_indices=None, excl_indices=None, box_size=None,
        charges=None, sigmas=None, epsilons=None,
        radii=None, scaled_radii=None,
    )


def _build_synthetic_bundles(n_molecules: int, n_atoms: int, seed: int = 0):
    """Build N same-bucket synthetic bundles for local R6 smoke.

    All bundles share identical shape_spec content (same n_atoms, same boundary
    condition, zero bonds/angles/etc.), satisfying HP3's gating-test contract
    for stacked jit(vmap) cache hits.
    """
    from prolix.physics.system import make_bundle_from_system

    bundles = []
    base_key = jax.random.PRNGKey(seed)
    for i in range(n_molecules):
        key = jax.random.fold_in(base_key, i)
        positions = jax.random.normal(key, (n_atoms, 3)).astype(jnp.float32) * 2.0
        species = jnp.ones(n_atoms, dtype=jnp.int8) * 6
        bundles.append(
            make_bundle_from_system(
                _make_stub_system(positions, species), boundary_condition="free"
            )
        )
    return bundles


def _load_real_bundles(data_dir: Path, n_molecules: int):
    """Load HP4 curated bundles from data_dir. Must all share atom_bucket_idx.

    Loads up to `n_molecules` from data_dir's mol_NNN.h5 files; filters to the
    bundles in the most-common atom_bucket_idx (so the stacked-vmap path is
    valid per HP3 gating contract).
    """
    import h5py

    from prolix.physics.system import make_bundle_from_system

    h5_files = sorted(data_dir.glob("mol_*.h5"))
    if not h5_files:
        log.error("No mol_*.h5 in %s", data_dir)
        sys.exit(1)

    raw = []
    for f in h5_files[:n_molecules * 2]:
        with h5py.File(f) as h:
            raw.append({
                "path": f,
                "positions0": jnp.asarray(h["positions"][0], dtype=jnp.float32),
                "species": jnp.asarray(h["species"][:], dtype=jnp.int8),
                "bucket_idx": int(h.attrs.get("bucket_idx", -1)),
                "n_atoms": int(h["positions"].shape[1]),
            })

    # Group by bucket; pick most-populous bucket.
    by_bucket: dict[int, list] = {}
    for r in raw:
        by_bucket.setdefault(r["bucket_idx"], []).append(r)
    chosen_bucket, members = max(by_bucket.items(), key=lambda kv: len(kv[1]))
    log.info("Chose atom_bucket_idx=%d with %d candidate molecules", chosen_bucket, len(members))

    if len(members) < n_molecules:
        log.warning("Only %d molecules in bucket %d (asked for %d); using all available",
                    len(members), chosen_bucket, n_molecules)
    members = members[:n_molecules]

    # All bundles in a bucket pad to the same bucketed n_atoms.
    bundles = []
    for r in members:
        bundles.append(
            make_bundle_from_system(
                _make_stub_system(r["positions0"], r["species"]),
                boundary_condition="free",
            )
        )
    return bundles, chosen_bucket


def _time_looped(bundles, n_trials: int, n_warmup: int) -> list[float]:
    grad_fns = [jax.jit(jax.grad(_stub_energy)) for _ in bundles]

    # Warmup
    for _ in range(n_warmup):
        for g, b in zip(grad_fns, bundles):
            jax.block_until_ready(g(b.positions))

    samples = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for g, b in zip(grad_fns, bundles):
            jax.block_until_ready(g(b.positions))
        samples.append(time.perf_counter() - t0)
    return samples


def _time_batched(bundles, n_trials: int, n_warmup: int) -> list[float]:
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *bundles)
    # Gradient over each bundle's positions; vmap over the leading batch axis.
    batched_grad = jax.jit(jax.vmap(jax.grad(_stub_energy)))

    # Warmup
    for _ in range(n_warmup):
        jax.block_until_ready(batched_grad(stacked.positions))

    samples = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(batched_grad(stacked.positions))
        samples.append(time.perf_counter() - t0)
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--data-dir", type=Path, help="HP4 lane_a/ or lane_b/ directory")
    p.add_argument("--n-molecules", type=int, default=16)
    p.add_argument("--n-atoms", type=int, default=30, help="(synthetic only)")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="(synthetic only)")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    # bath outcome-eval contract: prefer $BTH_RESULTS_PATH (set by `bth run`) over
    # explicit --out-json if both are unset; explicit --out-json always wins if given.
    if args.out_json is None:
        bth_path = os.environ.get("BTH_RESULTS_PATH")
        if bth_path:
            args.out_json = Path(bth_path)

    log.info("JAX backend=%s devices=%s", jax.default_backend(), jax.devices())

    if args.synthetic:
        log.info("Building %d synthetic bundles (n_atoms=%d)", args.n_molecules, args.n_atoms)
        bundles = _build_synthetic_bundles(args.n_molecules, args.n_atoms, args.seed)
        bucket_idx = bundles[0].shape_spec.atom_bucket_idx
    else:
        bundles, bucket_idx = _load_real_bundles(args.data_dir, args.n_molecules)

    log.info("N=%d bundles, atom_bucket_idx=%d, n_trials=%d, n_warmup=%d",
             len(bundles), bucket_idx, args.n_trials, args.n_warmup)

    log.info("Timing looped baseline (sequential jax.grad per bundle)...")
    looped = _time_looped(bundles, args.n_trials, args.n_warmup)

    log.info("Timing batched (jit(vmap(jax.grad)) on stacked pytree)...")
    batched = _time_batched(bundles, args.n_trials, args.n_warmup)

    looped_med = statistics.median(looped)
    batched_med = statistics.median(batched)
    ratio = looped_med / batched_med if batched_med > 0 else float("inf")

    log.info("looped median: %.4f s", looped_med)
    log.info("batched median: %.4f s", batched_med)
    log.info("speedup ratio: %.2fx", ratio)

    summary = {
        "mode": "synthetic" if args.synthetic else "data-dir",
        "data_dir": str(args.data_dir) if args.data_dir else None,
        "n_molecules": len(bundles),
        "atom_bucket_idx": int(bucket_idx),
        "n_trials": args.n_trials,
        "n_warmup": args.n_warmup,
        "looped_wallclock_s": {
            "median": looped_med,
            "min": min(looped),
            "max": max(looped),
            "samples": looped,
        },
        "batched_wallclock_s": {
            "median": batched_med,
            "min": min(batched),
            "max": max(batched),
            "samples": batched,
        },
        "speedup_ratio": ratio,
        "speedup_floor_falsification": SPEEDUP_FLOOR,
        "passes_falsification_trigger": ratio >= SPEEDUP_FLOOR,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }
    log.info("Summary:\n%s", json.dumps(summary, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        log.info("Wrote %s", args.out_json)

    if ratio < SPEEDUP_FLOOR:
        log.error(
            "R6 falsification trigger fired: ratio=%.2fx < %.1fx. "
            "Scale ensemble to N=64 before §7.1 begins (roadmap §2.7 RR7).",
            ratio, SPEEDUP_FLOOR,
        )
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
