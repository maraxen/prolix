#!/usr/bin/env python3
"""PME grid-size sweep profiler for B1 water benchmark.

Measures the performance impact of varying pme_grid_points on energy/force
evaluation for the 4-water B1 system in a 30x30x30 Angstrom periodic box.

The current hardcoded grid_points=64 is suspected to be oversized for such a
tiny system; this script measures potential speedup from smaller grids before
committing to production changes (measurement-only, no modifications to source).

Usage:
    # Dry run (no GPU, just shape validation)
    uv run python scripts/profile_b1_pme_grid_sweep.py --dry-run

    # Full benchmark (requires GPU)
    uv run python scripts/profile_b1_pme_grid_sweep.py \\
        --grid-points-list 64,32,16,8 \\
        --replicas 16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

# JAX configuration (f32 matching GPU profile test convention)
os.environ["JAX_ENABLE_X64"] = "False"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax_md import space

# Add scripts/ to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pme_grid_sweep")


# ================================================================
# Benchmark harness (from tests/test_gpu_profile_components.py)
# ================================================================

def bench(fn, name, n_warmup=3, n_trials=20, n_inner=5):
    """Benchmark with block_until_ready, warmup, multiple trials.

    Pattern matches tests/test_gpu_profile_components.py exactly.
    """
    for _ in range(n_warmup):
        r = fn()
        jax.block_until_ready(r)

    times = []
    for _ in range(n_trials):
        jax.block_until_ready(jnp.zeros(1))  # sync
        t0 = time.perf_counter()
        for _ in range(n_inner):
            r = fn()
            jax.block_until_ready(r)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000 / n_inner)

    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    log.info(f"  {name:55s} {avg:8.3f} ± {std:5.3f} ms  (min={mn:.3f})")
    return avg


# ================================================================
# Main sweep
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PME grid-size sweep profiler for B1 water system",
    )
    parser.add_argument(
        "--grid-points-list",
        type=str,
        default="64,32,16,8",
        help="Comma-separated grid point sizes to sweep (default: 64,32,16,8)",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=16,
        help="Number of replicas for vmap benchmark (default: 16, matching B1 width)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build everything, print shapes, skip bench() calls (GPU-free)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of benchmark trials (default: 20)",
    )
    parser.add_argument(
        "--n-inner",
        type=int,
        default=5,
        help="Inner loop iterations per trial (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "b1_pme_grid_sweep.json",
        help="Result JSON path (bathos --out convention: must be a persistent path, not tmp/)",
    )
    args = parser.parse_args()

    # Parse grid points
    grid_points_list = [int(x.strip()) for x in args.grid_points_list.split(",")]
    n_trials = 1 if args.dry_run else args.n_trials
    n_inner = 1 if args.dry_run else args.n_inner

    log.info("=" * 80)
    log.info("PME GRID-SIZE SWEEP — B1 Water Benchmark")
    log.info("=" * 80)
    log.info(f"Grid points: {grid_points_list}")
    log.info(f"Replicas (vmap width): {args.replicas}")
    log.info(f"Trials: {n_trials}, inner loops: {n_inner}")
    log.info(f"Dry-run: {args.dry_run}")
    log.info(f"x64 enabled: {jax.config.x64_enabled}")
    log.info(f"Devices: {[str(d) for d in jax.devices()]}")
    log.info("")

    # ================================================================
    # 1. Build bundle and system
    # ================================================================
    log.info("Building 4-water bundle...")
    from b1_init_exec import _four_water_bundle

    bundle = _four_water_bundle()
    log.info(f"  Bundle: n_atoms={int(jnp.asarray(bundle.n_atoms))}, "
             f"n_waters={int(jnp.asarray(bundle.n_waters))}")

    # Create displacement function (periodic, from bundle's box)
    from prolix.api.bundle_md import displacement_fn_for_bundle, active_positions

    disp_fn, _ = displacement_fn_for_bundle(bundle)
    positions = active_positions(bundle)
    log.info(f"  Positions shape: {positions.shape}, dtype: {positions.dtype}")

    # ================================================================
    # 2. Create baseline PhysicsSystem
    # ================================================================
    log.info("Creating PhysicsSystem...")
    from prolix.api.bundle_md import physics_system_from_bundle

    sys_baseline = physics_system_from_bundle(bundle, positions)
    log.info(f"  System positions: {sys_baseline.positions.shape}, "
             f"pme_grid_points={sys_baseline.pme_grid_points} (hardcoded baseline)")
    log.info(f"  Periodic box: {sys_baseline.box_size}, pme_alpha={sys_baseline.pme_alpha}")

    # ================================================================
    # 3. Benchmark each grid size
    # ================================================================
    results = []
    baseline_times = {}

    for grid_points in grid_points_list:
        log.info("")
        log.info("=" * 80)
        log.info(f"Grid points: {grid_points}")
        log.info("-" * 80)

        # Modify system for this grid size
        sys = dataclasses.replace(sys_baseline, pme_grid_points=grid_points)

        if args.dry_run:
            log.info(f"  [DRY-RUN] Positions: {sys.positions.shape}, "
                     f"pme_grid_points={sys.pme_grid_points}")
            results.append({
                "grid_points": grid_points,
                "energy_ms": None,
                "grad_ms": None,
                "full_step_ms": None,
                "full_step_vmap_ms": None,
            })
            continue

        # Benchmark 1: Energy only (single_padded_energy)
        log.info("\n1. Energy forward pass (single_padded_energy)")
        from prolix.batched_energy import single_padded_energy

        @jax.jit
        def energy_fn():
            return single_padded_energy(sys, disp_fn, implicit_solvent=False)

        energy_ms = bench(energy_fn, f"Energy @ {grid_points}",
                         n_warmup=3, n_trials=n_trials, n_inner=n_inner)

        # Benchmark 2: Energy gradient (jax.grad of energy)
        log.info("\n2. Energy gradient (jax.grad)")

        def energy_wrapper(r):
            sys_moved = dataclasses.replace(sys, positions=r)
            return single_padded_energy(sys_moved, disp_fn, implicit_solvent=False)

        @jax.jit
        def grad_fn():
            return jax.grad(energy_wrapper)(sys.positions)

        grad_ms = bench(grad_fn, f"Gradient @ {grid_points}",
                       n_warmup=3, n_trials=n_trials, n_inner=n_inner)

        # Benchmark 3: Full step (settle_langevin + 1 integration step)
        log.info("\n3. Full integration step (settle_langevin)")
        try:
            from prolix.physics.settle import settle_langevin
            from prolix.api.bundle_md import water_indices_for_integration, masses_for_bundle

            water_indices = water_indices_for_integration(bundle)
            masses = masses_for_bundle(bundle)

            if water_indices is None or water_indices.shape[0] == 0:
                log.warning("  No water indices; skipping settle_langevin")
                full_step_ms = None
            else:
                # Create integrator (reuse energy_wrapper as the force source)
                init_fn, apply_fn = settle_langevin(
                    energy_or_force_fn=energy_wrapper,
                    shift_fn=disp_fn,
                    dt=0.5,
                    kT=1.0,
                    gamma=1.0,
                    mass=masses,
                    water_indices=water_indices,
                )

                # Initialize state
                key = jax.random.key(42)
                initial_state = init_fn(key, sys.positions, mass=masses)

                @jax.jit
                def step_fn():
                    return apply_fn(initial_state)

                full_step_ms = bench(step_fn, f"Full step @ {grid_points}",
                                    n_warmup=3, n_trials=n_trials, n_inner=n_inner)
        except Exception as e:
            log.warning(f"  Full step skipped: {e}")
            full_step_ms = None

        # Benchmark 4: Vmapped full step (over replicas)
        log.info(f"\n4. Vmapped full step (vmap over {args.replicas} replicas)")
        try:
            if full_step_ms is not None and water_indices is not None:
                # Create replicated positions + velocities
                key = jax.random.key(42)
                keys = jax.random.split(key, args.replicas)

                # Tile positions with small per-replica jitter
                pos_replicas = jnp.tile(sys.positions[None, :, :], (args.replicas, 1, 1))
                jitter_scale = 0.001
                pos_jitter = jax.random.normal(key, pos_replicas.shape) * jitter_scale
                pos_replicas = pos_replicas + pos_jitter

                # Create systems for each replica
                def make_sys_replica(pos_r):
                    return dataclasses.replace(sys, positions=pos_r)

                sys_replicas = jax.vmap(make_sys_replica)(pos_replicas)

                # Initialize states via vmap
                def init_replica(key_r, sys_r):
                    return init_fn(key_r, sys_r.positions, mass=masses)

                initial_states = jax.vmap(init_replica)(keys, sys_replicas)

                @jax.jit
                def step_vmap_fn():
                    return jax.vmap(apply_fn)(initial_states)

                full_step_vmap_ms = bench(step_vmap_fn,
                                         f"Vmapped full step @ {grid_points} ({args.replicas} replicas)",
                                         n_warmup=3, n_trials=n_trials, n_inner=n_inner)
            else:
                log.warning("  Vmapped step skipped (no full_step baseline)")
                full_step_vmap_ms = None
        except Exception as e:
            log.warning(f"  Vmapped step skipped: {e}")
            full_step_vmap_ms = None

        # Record result
        result = {
            "grid_points": grid_points,
            "energy_ms": energy_ms,
            "grad_ms": grad_ms,
            "full_step_ms": full_step_ms,
            "full_step_vmap_ms": full_step_vmap_ms,
        }
        results.append(result)
        baseline_times[grid_points] = {
            "energy": energy_ms,
            "grad": grad_ms,
            "full_step": full_step_ms,
            "full_step_vmap": full_step_vmap_ms,
        }

    # ================================================================
    # 5. Compute speedups and output JSON
    # ================================================================
    log.info("")
    log.info("=" * 80)
    log.info("SPEEDUP SUMMARY vs. baseline (grid_points=64)")
    log.info("-" * 80)

    # Add speedup fields
    baseline_64 = baseline_times.get(64, {})
    for result in results:
        G = result["grid_points"]
        result["energy_speedup"] = (
            baseline_64.get("energy") / result["energy_ms"]
            if result["energy_ms"] is not None and baseline_64.get("energy") is not None
            else None
        )
        result["grad_speedup"] = (
            baseline_64.get("grad") / result["grad_ms"]
            if result["grad_ms"] is not None and baseline_64.get("grad") is not None
            else None
        )
        result["full_step_speedup"] = (
            baseline_64.get("full_step") / result["full_step_ms"]
            if result["full_step_ms"] is not None and baseline_64.get("full_step") is not None
            else None
        )
        result["full_step_vmap_speedup"] = (
            baseline_64.get("full_step_vmap") / result["full_step_vmap_ms"]
            if result["full_step_vmap_ms"] is not None and baseline_64.get("full_step_vmap") is not None
            else None
        )

        # Log summary
        if result["energy_speedup"] is not None:
            line = (
                f"  Grid {G}: energy {result['energy_speedup']:.2f}x, "
                f"grad {result['grad_speedup']:.2f}x"
            )
            if result["full_step_speedup"] is not None:
                line += f", full_step {result['full_step_speedup']:.2f}x"
            if result["full_step_vmap_speedup"] is not None:
                line += f", vmap {result['full_step_vmap_speedup']:.2f}x"
            log.info(line)

    # Flat top-level fields for bathos's DuckDB-based outcome gate (result_schema
    # in profile_b1_pme_grid_sweep.bth.toml expects flat scalars, not the nested
    # config/system/results structure below, which remains for direct analysis).
    baseline_result = next((r for r in results if r["grid_points"] == 64), None)
    n_grid_points_measured = sum(1 for r in results if r.get("energy_ms") is not None)
    best_full_step_speedup = max(
        (r["full_step_speedup"] for r in results if r.get("full_step_speedup") is not None),
        default=None,
    )

    output = {
        "dry_run": args.dry_run,
        "replicas": args.replicas,
        "x64_enabled": bool(jax.config.x64_enabled),
        "n_grid_points_swept": len(grid_points_list),
        "n_grid_points_measured": n_grid_points_measured,
        "baseline_energy_ms": baseline_result["energy_ms"] if baseline_result else None,
        "best_full_step_speedup": best_full_step_speedup,
        "config": {
            "grid_points_list": grid_points_list,
            "replicas": args.replicas,
            "dry_run": args.dry_run,
            "x64_enabled": jax.config.x64_enabled,
        },
        "system": {
            "n_atoms": int(jnp.asarray(bundle.n_atoms)),
            "n_waters": int(jnp.asarray(bundle.n_waters)),
            "box_size": [float(x) for x in sys_baseline.box_size],
            "pme_alpha": float(sys_baseline.pme_alpha),
        },
        "results": results,
    }

    args.out.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    log.info("")
    log.info("=" * 80)
    log.info(f"Output written to {args.out}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
