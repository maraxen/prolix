#!/usr/bin/env python3
"""Combined memory + performance benchmark: N² vs neighbor-list energy.

Sweeps multiple NL cutoffs and tests jax.checkpoint to find optimal config.

Usage:
    uv run python scripts/benchmark_nlvsdense.py
    uv run python scripts/benchmark_nlvsdense.py --cutoffs 8 10 12 16 20
    uv run python scripts/benchmark_nlvsdense.py --chunk-sizes 1 2 4 8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from jax_md import dataclasses as jax_dataclasses

sys.path.insert(0, os.path.dirname(__file__))
from run_batched_pipeline import (
    SYSTEM_CATALOG,
    load_and_parameterize,
    prepare_batches,
)

logger = logging.getLogger("bench_nlvsdense")


def setup_logging() -> str:
    os.makedirs("outputs/logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/logs/bench_nlvsdense_{ts}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_file


def get_gpu_mem() -> dict[str, int]:
    try:
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        if stats:
            return {
                "used": stats.get("bytes_in_use", 0),
                "peak": stats.get("peak_bytes_in_use", 0),
                "total": stats.get("bytes_limit", 0),
            }
    except Exception:
        pass
    return {"used": -1, "peak": -1, "total": -1}


def mb(n: int) -> str:
    return "N/A" if n < 0 else f"{n / (1024**2):.1f}"


def clean_gpu():
    gc.collect()
    jax.clear_caches()
    gc.collect()


def benchmark_fn(f, *args, warmup=1, trials=5, **kwargs):
    import timeit as _timeit
    clean_gpu()
    mem_before = get_gpu_mem()
    t0 = time.perf_counter()
    for _ in range(warmup):
        result = f(*args, **kwargs)
        jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0
    mem_post = get_gpu_mem()
    # Use timeit.repeat for GC-free, precise per-trial timing
    trial_times = _timeit.repeat(
        lambda: jax.block_until_ready(f(*args, **kwargs)),
        repeat=trials,
        number=1,
    )
    result = f(*args, **kwargs)
    jax.block_until_ready(result)
    mem_after = get_gpu_mem()
    return {
        "compile_s": round(compile_time, 3),
        "avg_ms": round(sum(trial_times) / len(trial_times) * 1000, 3),
        "min_ms": round(min(trial_times) * 1000, 3),
        "max_ms": round(max(trial_times) * 1000, 3),
        "trials": trials,
        "mem_before_mb": round(mem_before["used"] / 1024**2, 1),
        "mem_post_compile_mb": round(mem_post["used"] / 1024**2, 1),
        "mem_after_mb": round(mem_after["used"] / 1024**2, 1),
        "mem_peak_mb": round(mem_after["peak"] / 1024**2, 1),
        "result": result,
    }


def build_neighbor_list(positions, n_real, n_padded, cutoff=12.0):
    pos_np = np.array(positions)
    dists = np.linalg.norm(
        pos_np[:n_real, None, :] - pos_np[None, :n_real, :], axis=-1,
    )
    max_k = 0
    for i in range(n_real):
        k = int(np.sum((dists[i] < cutoff) & (dists[i] > 0)))
        max_k = max(max_k, k)
    max_k = min(max_k + 32, n_padded)
    neighbor_idx = np.full((n_padded, max_k), n_padded, dtype=np.int32)
    for i in range(n_real):
        nbrs = np.where((dists[i] < cutoff) & (dists[i] > 0))[0]
        neighbor_idx[i, : len(nbrs)] = nbrs
    return jnp.array(neighbor_idx), max_k


def bench_nl_at_cutoff(sys0, n_real, N, cutoff, displacement_fn, dt, kT, gamma,
                        variant="plain", key=None):
    """Benchmark NL energy/gradient/step at a given cutoff."""
    from prolix.batched_energy import single_padded_energy_nl, single_padded_energy_nl_cvjp
    from prolix.batched_simulate import make_langevin_step_nl, LangevinState

    neighbor_idx, max_k = build_neighbor_list(sys0.positions, n_real, N, cutoff=cutoff)
    sparsity = max_k / N * 100

    if variant == "plain":
        energy_fn = single_padded_energy_nl
        label_suffix = ""
    elif variant == "ckpt":
        # Close over displacement_fn *before* checkpointing so JAX only
        # traces the JAX-traceable args (sys, nbr) through the boundary.
        def _ckpt_energy(sys, nbr, disp_fn):
            return jax.checkpoint(
                lambda s, n: single_padded_energy_nl(s, n, disp_fn)
            )(sys, nbr)
        energy_fn = _ckpt_energy
        label_suffix = "-ckpt"
    elif variant == "cvjp":
        energy_fn = single_padded_energy_nl_cvjp
        label_suffix = "-cvjp"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    label = f"NL-{cutoff:.0f}Å{label_suffix}"

    @jax.jit
    def energy_nl(sys, nbr):
        return energy_fn(sys, nbr, displacement_fn)

    @jax.jit
    def grad_nl(sys, nbr):
        def e_fn(r):
            s = jax_dataclasses.replace(sys, positions=r)
            return energy_fn(s, nbr, displacement_fn)
        return jax.grad(e_fn)(sys.positions)

    # Make step fn with custom energy function
    step_fn_nl = make_langevin_step_nl(dt, kT, gamma, energy_fn=energy_fn)

    @jax.jit
    def step_nl(sys, state, nbr):
        return step_fn_nl(sys, state, nbr)

    r_e = benchmark_fn(energy_nl, sys0, neighbor_idx, warmup=1, trials=5)
    r_g = benchmark_fn(grad_nl, sys0, neighbor_idx, warmup=1, trials=5)

    if key is None:
        key = jax.random.PRNGKey(0)
    init_f = -r_g["result"]
    jax.block_until_ready(init_f)
    state = LangevinState(
        positions=sys0.positions,
        momentum=jnp.zeros_like(sys0.positions),
        force=init_f,
        mass=sys0.masses,
        key=key,
        cap_count=jnp.array(0, dtype=jnp.int32),
    )
    state = step_nl(sys0, state, neighbor_idx)
    jax.block_until_ready(state.positions)

    r_s = benchmark_fn(step_nl, sys0, state, neighbor_idx, warmup=1, trials=5)

    logger.info(
        "  %-16s K=%-5d (%.0f%%) | E=%6.2f ms  G=%6.2f ms  S=%6.2f ms | "
        "peak=%s MB  compile=%.1fs",
        label, max_k, sparsity,
        r_e["avg_ms"], r_g["avg_ms"], r_s["avg_ms"],
        mb(int(r_e["mem_peak_mb"] * 1024**2)), r_e["compile_s"],
    )

    return {
        "cutoff": cutoff,
        "variant": variant,
        "max_k": max_k,
        "sparsity_pct": round(sparsity, 1),
        "energy_ms": r_e["avg_ms"],
        "gradient_ms": r_g["avg_ms"],
        "step_ms": r_s["avg_ms"],
        "energy_compile_s": r_e["compile_s"],
        "gradient_compile_s": r_g["compile_s"],
        "step_compile_s": r_s["compile_s"],
        "energy_peak_mb": r_e["mem_peak_mb"],
        "gradient_peak_mb": r_g["mem_peak_mb"],
        "step_peak_mb": r_s["mem_peak_mb"],
        "neighbor_idx": neighbor_idx,
        "step_fn": step_fn_nl,
    }


def main():
    parser = argparse.ArgumentParser(description="NL vs N² benchmark sweep")
    parser.add_argument("--systems", nargs="+", default=["1X2G", "1Y57"])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--cutoffs", nargs="+", type=float, default=[8.0, 12.0, 16.0, 20.0])
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--output", type=str, default="outputs/logs/bench_nlvsdense.json")
    args = parser.parse_args()

    log_file = setup_logging()
    logger.info("Log file: %s", log_file)

    gpu_mem = get_gpu_mem()
    results = {
        "timestamp": datetime.now().isoformat(),
        "devices": [str(d) for d in jax.devices()],
        "gpu_total_mb": round(gpu_mem["total"] / 1024**2, 1),
        "systems": args.systems,
        "cutoffs": args.cutoffs,
        "chunk_sizes": args.chunk_sizes,
        "buckets": {},
    }

    logger.info("=" * 70)
    logger.info("NL vs N² Cutoff Sweep + Checkpoint Benchmark")
    logger.info("=" * 70)
    logger.info("Devices: %s", jax.devices())
    logger.info("GPU total: %s MB", mb(gpu_mem["total"]))
    logger.info("Cutoffs: %s Å", args.cutoffs)

    proteins = []
    for name in args.systems:
        proteins.append(load_and_parameterize(SYSTEM_CATALOG[name]))
    batches = prepare_batches(proteins, args.systems, n_replicas=args.replicas)

    for bucket_size, (batch, names) in sorted(batches.items()):
        leaves = jax.tree.flatten(batch)[0]
        B = leaves[0].shape[0]
        N = batch.positions.shape[1]
        logger.info("\n" + "=" * 70)
        logger.info("BUCKET %d: B=%d, N=%d", bucket_size, B, N)
        logger.info("=" * 70)

        from prolix.batched_energy import single_padded_energy
        from prolix.batched_simulate import make_langevin_step, LangevinState

        displacement_fn, _ = space.free()
        AKMA = 48.88821
        dt = 2.0 / AKMA
        kB = 0.001987204
        kT = kB * 310.15
        gamma = 1.0 / AKMA
        key = jax.random.PRNGKey(0)

        sys0 = jax.tree.map(lambda x: x[0], batch)
        n_real = int(sys0.n_real_atoms)

        bucket_results = {"B": B, "N": N, "n_real": n_real}

        # ── Dense baseline ──
        logger.info("\n--- Dense N² baseline ---")
        step_fn_dense = make_langevin_step(dt, kT, gamma)

        @jax.jit
        def energy_dense(sys):
            return single_padded_energy(sys, displacement_fn)

        @jax.jit
        def grad_dense(sys):
            def e_fn(r):
                return single_padded_energy(
                    jax_dataclasses.replace(sys, positions=r), displacement_fn)
            return jax.grad(e_fn)(sys.positions)

        @jax.jit
        def step_dense(sys, state):
            return step_fn_dense(sys, state)

        r_e_d = benchmark_fn(energy_dense, sys0, warmup=1, trials=5)
        r_g_d = benchmark_fn(grad_dense, sys0, warmup=1, trials=5)

        init_f = -r_g_d["result"]
        jax.block_until_ready(init_f)
        state_d = LangevinState(
            positions=sys0.positions,
            momentum=jnp.zeros_like(sys0.positions),
            force=init_f,
            mass=sys0.masses,
            key=key,
            cap_count=jnp.array(0, dtype=jnp.int32),
        )
        state_d = step_dense(sys0, state_d)
        jax.block_until_ready(state_d.positions)
        r_s_d = benchmark_fn(step_dense, sys0, state_d, warmup=1, trials=5)

        logger.info(
            "  %-16s          | E=%6.2f ms  G=%6.2f ms  S=%6.2f ms | peak=%s MB",
            "Dense-N²",
            r_e_d["avg_ms"], r_g_d["avg_ms"], r_s_d["avg_ms"],
            mb(int(r_e_d["mem_peak_mb"] * 1024**2)),
        )

        bucket_results["dense"] = {
            "energy_ms": r_e_d["avg_ms"],
            "gradient_ms": r_g_d["avg_ms"],
            "step_ms": r_s_d["avg_ms"],
            "energy_peak_mb": r_e_d["mem_peak_mb"],
        }

        # ── NL cutoff sweep ──
        logger.info("\n--- NL cutoff sweep ---")
        nl_results = {}
        best_step_ms = r_s_d["avg_ms"]
        best_config = "dense"

        for cutoff in args.cutoffs:
            clean_gpu()
            res = bench_nl_at_cutoff(
                sys0, n_real, N, cutoff, displacement_fn, dt, kT, gamma,
                variant="plain", key=key,
            )
            label = f"{cutoff:.0f}"
            nl_results[label] = {k: v for k, v in res.items()
                                 if k not in ("neighbor_idx", "step_fn")}
            if res["step_ms"] < best_step_ms:
                best_step_ms = res["step_ms"]
                best_config = f"NL-{label}Å"

        # ── Checkpoint & CVJP variants (test at all cutoffs for thoroughness) ──
        logger.info("\n--- Checkpoint & CVJP variants ---")
        for cutoff in args.cutoffs:
            for var in ["ckpt", "cvjp"]:
                clean_gpu()
                res = bench_nl_at_cutoff(
                    sys0, n_real, N, cutoff, displacement_fn, dt, kT, gamma,
                    variant=var, key=key,
                )
                label = f"{cutoff:.0f}_{var}"
                nl_results[label] = {k: v for k, v in res.items()
                                     if k not in ("neighbor_idx", "step_fn")}
                if res["step_ms"] < best_step_ms:
                    best_step_ms = res["step_ms"]
                    best_config = f"NL-{cutoff:.0f}Å-{var}"

        bucket_results["nl_sweep"] = nl_results

        # ── Summary ──
        logger.info("\n--- RESULTS TABLE ---")
        logger.info(
            "  %-20s %-8s %-8s %-10s %-10s %-10s %-10s",
            "Config", "max_K", "sparse%", "E(ms)", "G(ms)", "S(ms)", "peak(MB)",
        )
        logger.info("  " + "-" * 78)
        logger.info(
            "  %-20s %-8s %-8s %-10.2f %-10.2f %-10.2f %-10.0f",
            "Dense-N²", "N²", "100%",
            r_e_d["avg_ms"], r_g_d["avg_ms"], r_s_d["avg_ms"],
            r_e_d["mem_peak_mb"],
        )
        for label, data in sorted(nl_results.items()):
            logger.info(
                "  %-20s %-8d %-8.1f %-10.2f %-10.2f %-10.2f %-10.0f",
                f"NL-{label}Å",
                data.get("max_k", 0),
                data.get("sparsity_pct", 0),
                data["energy_ms"], data["gradient_ms"], data["step_ms"],
                data.get("energy_peak_mb", 0),
            )

        ns_per_step = 2e-6
        rate_dense = 3600 / (r_s_d["avg_ms"] / 1000) * ns_per_step
        rate_best = 3600 / (best_step_ms / 1000) * ns_per_step
        logger.info("\n  BEST CONFIG: %s (%.2f ms/step, %.4f ns/hr)",
                     best_config, best_step_ms, rate_best)
        logger.info("  DENSE:       (%.2f ms/step, %.4f ns/hr)",
                     r_s_d["avg_ms"], rate_dense)
        logger.info("  SPEEDUP:     %.2fx", r_s_d["avg_ms"] / best_step_ms)

        bucket_results["best_config"] = best_config
        bucket_results["best_step_ms"] = best_step_ms
        bucket_results["best_rate_ns_hr"] = round(rate_best, 4)
        bucket_results["dense_rate_ns_hr"] = round(rate_dense, 4)

        results["buckets"][str(bucket_size)] = bucket_results

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nJSON saved to %s", output_path)


if __name__ == "__main__":
    main()
