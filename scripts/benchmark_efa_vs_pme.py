"""Sprint 2 Phase 4: Benchmark wall-clock times for EFA vs PME.

Profiling script that measures per-step wall time and JIT compilation overhead.
Outputs results to a JSON file with configurable system sizes.

Usage:
    python benchmark_efa_vs_pme.py --n-waters 32 64 128 256 --output results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from prolix.physics.electrostatic_methods import ElectrostaticMethod
from prolix.physics.eval_harness import make_tip3p_water_system
from prolix.physics.flash_explicit import flash_explicit_energy


def benchmark_method(
    system, method: str, n_features: int | None = None, n_runs: int = 10
) -> dict:
    """Benchmark a single method.

    Args:
        system: PaddedSystem.
        method: "pme" or "efa".
        n_features: Number of RFF features (only for EFA).
        n_runs: Number of timed runs.

    Returns:
        Dictionary with timing results.
    """
    if method == "pme":
        energy_fn = lambda sys: flash_explicit_energy(
            sys, electrostatic_method=ElectrostaticMethod.PME
        )
    elif method == "efa":
        assert n_features is not None
        energy_fn = lambda sys: flash_explicit_energy(
            sys,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=n_features,
            rff_seed=0,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Warm-up (unjitted)
    _ = energy_fn(system)

    # JIT compilation + first run
    jit_fn = jax.jit(energy_fn)
    t0 = time.perf_counter()
    result = jax.block_until_ready(jit_fn(system))
    t1 = time.perf_counter()
    jit_time_ms = (t1 - t0) * 1000.0

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = jax.block_until_ready(jit_fn(system))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert to ms

    times = jnp.array(times)
    wall_time_median = float(jnp.median(times))
    wall_time_std = float(jnp.std(times))

    return {
        "wall_time_ms_per_step_median": wall_time_median,
        "wall_time_ms_per_step_std": wall_time_std,
        "jit_compile_ms": jit_time_ms,
        "peak_memory_mb": None,  # Would require more instrumentation
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark EFA vs PME energy evaluation times."
    )
    parser.add_argument(
        "--n-waters",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Water system sizes to benchmark.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_waters": [],
        "results": [],
    }

    for n_waters in args.n_waters:
        print(f"\n=== Benchmarking n_waters={n_waters} ===")

        system = make_tip3p_water_system(n_waters=n_waters, seed=0)

        # Benchmark PME
        print("  PME...", end="", flush=True)
        pme_result = benchmark_method(system, "pme", n_runs=10)
        print(f" {pme_result['wall_time_ms_per_step_median']:.2f} ms/step")

        results["results"].append({
            "method": "pme",
            "n_features": None,
            "n_waters": n_waters,
            **pme_result,
        })

        # Benchmark EFA D=256
        print("  EFA (D=256)...", end="", flush=True)
        efa_256_result = benchmark_method(system, "efa", n_features=256, n_runs=10)
        print(f" {efa_256_result['wall_time_ms_per_step_median']:.2f} ms/step")

        results["results"].append({
            "method": "efa",
            "n_features": 256,
            "n_waters": n_waters,
            **efa_256_result,
        })

        # Benchmark EFA D=512
        print("  EFA (D=512)...", end="", flush=True)
        efa_512_result = benchmark_method(system, "efa", n_features=512, n_runs=10)
        print(f" {efa_512_result['wall_time_ms_per_step_median']:.2f} ms/step")

        results["results"].append({
            "method": "efa",
            "n_features": 512,
            "n_waters": n_waters,
            **efa_512_result,
        })

        results["n_waters"].append(n_waters)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
