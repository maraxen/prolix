#!/usr/bin/env python
"""GPU Benchmark: Batched vs Sequential Protein Energy Evaluation.

WS-D of #1854. Measures GPU efficiency of batched cross-topology
simulation (via make_batched_energy_fn + padding) vs sequential
per-protein evaluation (via make_energy_fn).

Requires GPU for meaningful results.

Usage:
    # Full benchmark (3 proteins, all tiers):
    uv run python prolix/benchmarks/benchmark_batched.py

    # Quick test (K-Ras only, fewer repeats):
    uv run python prolix/benchmarks/benchmark_batched.py --quick

    # With JAX profiler output:
    uv run python prolix/benchmarks/benchmark_batched.py --profile
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

# Enable x64 for physics accuracy
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    """Benchmark parameters."""
    n_repeats: int = 10
    homo_batch_sizes: list[int] = field(default_factory=lambda: [2, 4, 8])
    quick: bool = False
    profile: bool = False
    output_csv: str = "benchmark_batched_results.csv"


# Target systems from Ting benchmark (references/pdb/)
BENCHMARK_SYSTEMS = {
    "K-Ras": "references/pdb/8T71_chainA_fixed.pdb",
    "Src": "references/pdb/2SRC_chainA_fixed.pdb",
    "B2AR": "references/pdb/2RH1_chainA_fixed.pdb",
}

QUICK_SYSTEMS = {
    "K-Ras": "references/pdb/8T71_chainA_fixed.pdb",
}


# ---------------------------------------------------------------------------
# Protein loading (matches ensemble_test.py pattern)
# ---------------------------------------------------------------------------
def load_protein(pdb_path: str):
    """Load and parameterize a protein from a PDBFixer-fixed PDB."""
    import dataclasses as dc
    from proxide import OutputSpec, CoordFormat
    from proxide.io.parsing.backend import parse_structure
    from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors

    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        parameterize_md=True,
        force_field="proxide/src/proxide/assets/protein.ff19SB.xml",
        add_hydrogens=False,
        remove_solvent=True,
        remove_hetatm=True,
    )
    protein = parse_structure(pdb_path, spec)
    
    if protein.radii is None:
        _radii = assign_mbondi2_radii(list(protein.atom_names), protein.bonds)
        _scaled = assign_obc2_scaling_factors(list(protein.atom_names))
        protein = dc.replace(protein, radii=jnp.asarray(_radii), scaled_radii=jnp.asarray(_scaled))

    return protein


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
def time_fn(fn, n_repeats: int) -> tuple[float, float, float]:
    """Time a JAX function: returns (jit_time, mean_exec, std_exec)."""
    # JIT warmup (first call includes compilation)
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    jit_time = time.perf_counter() - t0

    # Timed repeats
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    return jit_time, np.mean(times), np.std(times)


# ---------------------------------------------------------------------------
# Tier 2B: Sequential baseline
# ---------------------------------------------------------------------------
def benchmark_sequential(proteins: dict, config: BenchmarkConfig) -> list[dict]:
    """Benchmark each protein independently via make_energy_fn."""
    from prolix.physics import system as physics_system
    from prolix.physics import neighbor_list as nl

    results = []
    displacement_fn, _ = space.free()

    for name, protein in proteins.items():
        positions = jnp.array(protein.coordinates).reshape(-1, 3)
        n_atoms = positions.shape[0]

        # Build energy function (canonical path)
        exclusion_spec = nl.ExclusionSpec.from_protein(protein)
        energy_fn = physics_system.make_energy_fn(
            displacement_fn, protein,
            exclusion_spec=exclusion_spec,
            implicit_solvent=True,
        )

        # Time energy evaluation
        fn = lambda: energy_fn(positions)
        jit_s, mean_s, std_s = time_fn(fn, config.n_repeats)

        result = {
            "scenario": "sequential",
            "system": name,
            "batch_size": 1,
            "bucket_size": "",
            "jit_compile_s": f"{jit_s:.4f}",
            "exec_mean_s": f"{mean_s:.6f}",
            "exec_std_s": f"{std_s:.6f}",
            "n_real_atoms": n_atoms,
            "n_padded_atoms": n_atoms,
            "padding_waste_pct": "0.0",
            "throughput_atoms_per_s": f"{n_atoms / mean_s:.0f}",
        }
        results.append(result)
        print(f"  {name} ({n_atoms} atoms): {mean_s:.4f}s ± {std_s:.4f}s "
              f"(JIT: {jit_s:.2f}s)")

        # Clear JIT cache between proteins to avoid cross-contamination
        jax.clear_caches()

    return results


# ---------------------------------------------------------------------------
# Tier 1: Homogeneous batched (same protein × N seeds)
# ---------------------------------------------------------------------------
def benchmark_homogeneous(proteins: dict, config: BenchmarkConfig) -> list[dict]:
    """Benchmark same protein replicated N times with positional noise."""
    from prolix.batched_energy import make_batched_energy_fn
    from prolix.padding import bucket_proteins, collate_batch

    results = []
    displacement_fn, _ = space.free()

    for name, protein in proteins.items():
        n_atoms = protein.coordinates.shape[0]

        for batch_size in config.homo_batch_sizes:
            # Create N copies with slight positional noise
            protein_copies = []
            for i in range(batch_size):
                import dataclasses as dc
                noisy_coords = np.array(protein.coordinates) + \
                    np.random.RandomState(i).randn(*protein.coordinates.shape) * 0.05
                p_copy = dc.replace(protein, coordinates=noisy_coords)
                protein_copies.append(p_copy)

            # Bucket and collate
            try:
                buckets = bucket_proteins(protein_copies)
            except Exception as e:
                print(f"  {name} × {batch_size}: bucket_proteins failed: {e}")
                continue

            for bucket_size, padded_list in buckets.items():
                batch = collate_batch(padded_list)
                n_real = sum(int(s.n_real_atoms) for s in padded_list)
                n_padded = batch_size * bucket_size

                batched_energy = make_batched_energy_fn(
                    displacement_fn, implicit_solvent=True
                )

                fn = lambda: batched_energy(batch)
                jit_s, mean_s, std_s = time_fn(fn, config.n_repeats)

                waste = (n_padded - n_real) / n_padded * 100
                result = {
                    "scenario": "homo_batch",
                    "system": name,
                    "batch_size": batch_size,
                    "bucket_size": bucket_size,
                    "jit_compile_s": f"{jit_s:.4f}",
                    "exec_mean_s": f"{mean_s:.6f}",
                    "exec_std_s": f"{std_s:.6f}",
                    "n_real_atoms": n_real,
                    "n_padded_atoms": n_padded,
                    "padding_waste_pct": f"{waste:.1f}",
                    "throughput_atoms_per_s": f"{n_real / mean_s:.0f}",
                }
                results.append(result)
                print(f"  {name} × {batch_size} (bucket {bucket_size}): "
                      f"{mean_s:.4f}s ± {std_s:.4f}s | "
                      f"waste: {waste:.1f}% (JIT: {jit_s:.2f}s)")

            jax.clear_caches()

    return results


# ---------------------------------------------------------------------------
# Tier 2A: Heterogeneous batched (cross-topology)
# ---------------------------------------------------------------------------
def benchmark_heterogeneous(proteins: dict, config: BenchmarkConfig) -> list[dict]:
    """Benchmark different proteins batched via bucket_proteins."""
    from prolix.batched_energy import make_batched_energy_fn
    from prolix.padding import bucket_proteins, collate_batch

    results = []
    displacement_fn, _ = space.free()

    protein_list = list(proteins.values())
    names = list(proteins.keys())

    try:
        buckets = bucket_proteins(protein_list)
    except Exception as e:
        print(f"  bucket_proteins failed: {e}")
        return results

    for bucket_size, padded_list in buckets.items():
        batch = collate_batch(padded_list)
        batch_size = len(padded_list)
        n_real = sum(int(s.n_real_atoms) for s in padded_list)
        n_padded = batch_size * bucket_size

        batched_energy = make_batched_energy_fn(
            displacement_fn, implicit_solvent=True
        )

        fn = lambda: batched_energy(batch)
        jit_s, mean_s, std_s = time_fn(fn, config.n_repeats)

        waste = (n_padded - n_real) / n_padded * 100
        system_label = "+".join(
            n for n, p in zip(names, protein_list)
            if p.coordinates.shape[0] <= bucket_size
        )

        result = {
            "scenario": "hetero_batch",
            "system": system_label,
            "batch_size": batch_size,
            "bucket_size": bucket_size,
            "jit_compile_s": f"{jit_s:.4f}",
            "exec_mean_s": f"{mean_s:.6f}",
            "exec_std_s": f"{std_s:.6f}",
            "n_real_atoms": n_real,
            "n_padded_atoms": n_padded,
            "padding_waste_pct": f"{waste:.1f}",
            "throughput_atoms_per_s": f"{n_real / mean_s:.0f}",
        }
        results.append(result)
        print(f"  Bucket {bucket_size} ({system_label}, B={batch_size}): "
              f"{mean_s:.4f}s ± {std_s:.4f}s | "
              f"waste: {waste:.1f}% (JIT: {jit_s:.2f}s)")

    jax.clear_caches()
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(results: list[dict], path: str):
    """Write results to CSV."""
    if not results:
        return
    fieldnames = results[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {path}")


def print_summary(seq_results: list[dict], homo_results: list[dict],
                  hetero_results: list[dict]):
    """Print a formatted summary with speedup calculations."""
    print("\n" + "=" * 70)
    print("  GPU Benchmark: Batched vs Sequential Energy Evaluation")
    print("=" * 70)
    print(f"  Hardware: {jax.devices()}")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Backend: {jax.default_backend()}")

    # Sequential baseline
    print(f"\n--- Sequential Baseline (Tier 2B) ---")
    total_seq = 0.0
    for r in seq_results:
        mean = float(r["exec_mean_s"])
        std = float(r["exec_std_s"])
        total_seq += mean
        print(f"  {r['system']:8s} ({r['n_real_atoms']} atoms): "
              f"{mean:.4f}s ± {std:.4f}s")
    print(f"  {'Total':8s}:                   {total_seq:.4f}s")

    # Homogeneous
    if homo_results:
        print(f"\n--- Homogeneous Batch (Tier 1) ---")
        for r in homo_results:
            mean = float(r["exec_mean_s"])
            std = float(r["exec_std_s"])
            # Compare against batch_size × sequential for same protein
            seq_match = [s for s in seq_results if s["system"] == r["system"]]
            if seq_match:
                seq_time = float(seq_match[0]["exec_mean_s"]) * int(r["batch_size"])
                speedup = seq_time / mean if mean > 0 else 0
                print(f"  {r['system']} × {r['batch_size']} (bucket {r['bucket_size']}): "
                      f"{mean:.4f}s ± {std:.4f}s | "
                      f"speedup: {speedup:.1f}x | waste: {r['padding_waste_pct']}%")
            else:
                print(f"  {r['system']} × {r['batch_size']}: {mean:.4f}s ± {std:.4f}s")

    # Heterogeneous
    if hetero_results:
        print(f"\n--- Heterogeneous Batch (Tier 2A) ---")
        for r in hetero_results:
            mean = float(r["exec_mean_s"])
            std = float(r["exec_std_s"])
            speedup = total_seq / mean if mean > 0 else 0
            print(f"  Bucket {r['bucket_size']} ({r['system']}, B={r['batch_size']}): "
                  f"{mean:.4f}s ± {std:.4f}s | "
                  f"speedup vs seq sum: {speedup:.1f}x | waste: {r['padding_waste_pct']}%")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPU Benchmark: Batched vs Sequential (WS-D of #1854)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (K-Ras only, 3 repeats)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable JAX profiler trace output")
    parser.add_argument("--output", default="benchmark_batched_results.csv",
                        help="Output CSV path")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Override number of timed repeats")
    args = parser.parse_args()

    config = BenchmarkConfig(
        quick=args.quick,
        profile=args.profile,
        output_csv=args.output,
    )
    if args.quick:
        config.n_repeats = 3
        config.homo_batch_sizes = [2, 4]
    if args.repeats is not None:
        config.n_repeats = args.repeats

    systems = QUICK_SYSTEMS if args.quick else BENCHMARK_SYSTEMS

    print(f"=== WS-D Benchmark: {len(systems)} proteins, "
          f"{config.n_repeats} repeats ===")
    print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")

    if jax.default_backend() == "cpu":
        print("WARNING: Running on CPU. Results are not representative "
              "of GPU performance.")

    # Load proteins
    print(f"\n--- Loading proteins ---")
    proteins = {}
    for name, path in systems.items():
        if not os.path.exists(path):
            print(f"  SKIP {name}: {path} not found")
            continue
        try:
            protein = load_protein(path)
            n_atoms = protein.coordinates.shape[0]
            proteins[name] = protein
            print(f"  {name}: {n_atoms} atoms")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    if not proteins:
        print("No proteins loaded. Exiting.")
        sys.exit(1)

    all_results = []

    # Tier 2B: Sequential
    print(f"\n--- Tier 2B: Sequential Baseline ---")
    seq_results = benchmark_sequential(proteins, config)
    all_results.extend(seq_results)

    # Tier 1: Homogeneous
    print(f"\n--- Tier 1: Homogeneous Batch ---")
    homo_results = benchmark_homogeneous(proteins, config)
    all_results.extend(homo_results)

    # Tier 2A: Heterogeneous (only if ≥2 proteins)
    hetero_results = []
    if len(proteins) >= 2:
        print(f"\n--- Tier 2A: Heterogeneous Batch ---")
        hetero_results = benchmark_heterogeneous(proteins, config)
        all_results.extend(hetero_results)
    else:
        print(f"\n--- Tier 2A: Skipped (need ≥2 proteins) ---")

    # Output
    write_csv(all_results, config.output_csv)
    print_summary(seq_results, homo_results, hetero_results)


if __name__ == "__main__":
    main()
