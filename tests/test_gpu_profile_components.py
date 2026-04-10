#!/usr/bin/env python3
"""GPU Component Profiler: Full Simulation Step Breakdown.

Dissects every component of a simulation step on GPU to identify the
actual bottleneck. Uses f32 (confirmed safe for implicit solvent GB)
and proper JAX benchmarking (block_until_ready, warmup, multiple trials).

Usage:
    python tests/test_gpu_profile_components.py
    python tests/test_gpu_profile_components.py --trace   # Perfetto trace
    python tests/test_gpu_profile_components.py --quick   # Fewer trials
"""

import os
import sys
import logging
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["JAX_ENABLE_X64"] = "False"

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
from jax_md import space

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_batched_pipeline import SYSTEM_CATALOG, load_and_parameterize, prepare_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gpu_profile")


# ================================================================
# Benchmark harness
# ================================================================

def bench(fn, name, n_warmup=3, n_trials=20, n_inner=5):
    """Benchmark with block_until_ready, warmup, multiple trials."""
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


def build_static_nl(positions, n_real, N, cutoff):
    """Build static neighbor list at given cutoff (CPU)."""
    pos_np = np.array(positions[:n_real])
    dists = np.linalg.norm(pos_np[:, None, :] - pos_np[None, :, :], axis=-1)
    max_k = 0
    for i in range(n_real):
        k = int(np.sum((dists[i] < cutoff) & (dists[i] > 0)))
        max_k = max(max_k, k)
    max_k = min(max_k + 16, N)
    neighbor_idx = np.full((N, max_k), N, dtype=np.int32)
    for i in range(n_real):
        nbrs = np.where((dists[i] < cutoff) & (dists[i] > 0))[0]
        neighbor_idx[i, :len(nbrs)] = nbrs
    return jnp.array(neighbor_idx), max_k


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--system", default="1X2G")
    args = parser.parse_args()

    N_WARMUP = 3
    N_TRIALS = 20 if not args.quick else 5
    N_INNER = 5 if not args.quick else 1
    bkw = dict(n_warmup=N_WARMUP, n_trials=N_TRIALS, n_inner=N_INNER)

    log.info("=" * 75)
    log.info("GPU Component Profiler — Step Breakdown")
    log.info("=" * 75)
    log.info(f"Devices: {[str(d) for d in jax.devices()]}")
    log.info(f"Backend: {jax.default_backend()}")
    log.info(f"x64 enabled: {jax.config.x64_enabled}")
    log.info(f"Trials: {N_TRIALS}, inner: {N_INNER}, warmup: {N_WARMUP}")

    # Load system
    protein = load_and_parameterize(SYSTEM_CATALOG[args.system])
    batches = prepare_batches([protein], [args.system], n_replicas=1)

    for bucket_size, (batch, names) in batches.items():
        sys0 = jax.tree.map(lambda x: x[0], batch)
        N = sys0.positions.shape[0]
        n_real = int(sys0.n_real_atoms)

        # Force f32
        sys0 = jax.tree.map(
            lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x,
            sys0,
        )

        log.info(f"System: N={N} padded, n_real={n_real}, dtype={sys0.positions.dtype}")
        displacement_fn, _ = space.free()

        from prolix.physics.generalized_born import (
            compute_born_radii,
            compute_born_radii_neighbor_list,
            compute_gb_energy,
            compute_gb_energy_neighbor_list,
        )
        from prolix.batched_energy import single_padded_energy_nl_cvjp
        from prolix.fused_energy import (
            fused_energy_and_forces_nl,
            _bonded_energy_from_positions,
            _gb_energy_from_positions,
        )
        from jax_md import dataclasses as jax_dataclasses

        # Build NLs
        CUTOFFS = [10.0, 12.0, 15.0, 20.0]
        nls = {}
        for c in CUTOFFS:
            nl, K = build_static_nl(sys0.positions, n_real, N, c)
            nls[c] = (nl, K)
            log.info(f"  NL {c:5.1f}Å: K={K}")

        # Real-atom arrays (f32)
        pos_r = sys0.positions[:n_real]
        rad_r = sys0.radii[:n_real]
        chg_r = sys0.charges[:n_real]
        sr_r = (
            sys0.scaled_radii[:n_real]
            if hasattr(sys0, 'scaled_radii') and sys0.scaled_radii is not None
            else None
        )

        # Optional trace
        if args.trace:
            trace_dir = "/tmp/jax_profile_gb"
            os.makedirs(trace_dir, exist_ok=True)
            log.info(f"Profiler trace → {trace_dir}")
            jax.profiler.start_trace(trace_dir, create_perfetto_trace=True)

        results = {}  # name → ms

        # ==============================================================
        # 1. BORN RADII
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("1. BORN RADII")
        log.info("-" * 75)

        @jax.jit
        def born_dense():
            return compute_born_radii(pos_r, rad_r, scaled_radii=sr_r)

        results["born_dense"] = bench(born_dense, "Born radii (dense N², n_real only)", **bkw)

        for c in CUTOFFS:
            nl, K = nls[c]
            @jax.jit
            def born_nl(nl=nl):
                return compute_born_radii_neighbor_list(
                    sys0.positions, sys0.radii, nl,
                    scaled_radii=sys0.scaled_radii if hasattr(sys0, 'scaled_radii') else None,
                )
            results[f"born_nl_{c}"] = bench(born_nl, f"Born radii (NL {c}Å, K={K})", **bkw)

        # ==============================================================
        # 2. GB+COULOMB ENERGY (forward only)
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("2. GB+COULOMB ENERGY (forward)")
        log.info("-" * 75)

        @jax.jit
        def gb_e_dense():
            return compute_gb_energy(pos_r, chg_r, rad_r, scaled_radii=sr_r)

        results["gb_e_dense"] = bench(gb_e_dense, "GB energy (dense N²)", **bkw)

        for c in CUTOFFS:
            nl, K = nls[c]
            @jax.jit
            def gb_e_nl(nl=nl):
                return compute_gb_energy_neighbor_list(
                    sys0.positions, sys0.charges, sys0.radii, nl,
                )
            results[f"gb_e_nl_{c}"] = bench(gb_e_nl, f"GB energy (NL {c}Å, K={K})", **bkw)

        # ==============================================================
        # 3. GB+COULOMB GRADIENT
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("3. GB+COULOMB GRADIENT (jax.grad)")
        log.info("-" * 75)

        @jax.jit
        def gb_g_dense():
            def e_fn(r):
                e, _ = compute_gb_energy(r, chg_r, rad_r, scaled_radii=sr_r)
                return e
            return jax.grad(e_fn)(pos_r)

        results["gb_g_dense"] = bench(gb_g_dense, "GB grad (dense N²)", **bkw)

        for c in CUTOFFS:
            nl, K = nls[c]
            @jax.jit
            def gb_g_nl(nl=nl):
                gb_fn = jax.checkpoint(
                    lambda pos: _gb_energy_from_positions(
                        pos, sys0.charges, sys0.radii, sys0.atom_mask, nl,
                    )
                )
                return jax.grad(gb_fn)(sys0.positions)
            results[f"gb_g_nl_{c}"] = bench(gb_g_nl, f"GB grad (NL CVJP {c}Å, K={K})", **bkw)

        # ==============================================================
        # 4. LJ FORCES
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("4. LJ FUSED (energy + analytical forces)")
        log.info("-" * 75)

        for c in CUTOFFS:
            nl, K = nls[c]
            @jax.jit
            def lj_fused(nl=nl):
                return fused_energy_and_forces_nl(sys0, nl, displacement_fn)
            results[f"lj_{c}"] = bench(lj_fused, f"LJ fused E+F (NL {c}Å, K={K})", **bkw)

        # ==============================================================
        # 5. BONDED TERMS ONLY
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("5. BONDED ENERGY (bonds + angles + dihedrals)")
        log.info("-" * 75)

        @jax.jit
        def bonded_e():
            return _bonded_energy_from_positions(sys0.positions, sys0)

        try:
            results["bonded"] = bench(bonded_e, "Bonded energy only", **bkw)
        except Exception as e:
            log.warning(f"  Bonded benchmark skipped: {e}")

        @jax.jit
        def bonded_g():
            return jax.grad(_bonded_energy_from_positions)(sys0.positions, sys0)

        try:
            results["bonded_g"] = bench(bonded_g, "Bonded gradient (jax.grad)", **bkw)
        except Exception as e:
            log.warning(f"  Bonded grad benchmark skipped: {e}")

        # ==============================================================
        # 6. FULL STEP: E + grad (all terms)
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("6. FULL STEP (E + grad, all terms combined)")
        log.info("-" * 75)

        for c in CUTOFFS:
            nl, K = nls[c]
            @jax.jit
            def full_step(nl=nl):
                def e_fn(r):
                    s = jax_dataclasses.replace(sys0, positions=r)
                    return single_padded_energy_nl_cvjp(s, nl, displacement_fn)
                e = e_fn(sys0.positions)
                g = jax.grad(e_fn)(sys0.positions)
                return e, g
            results[f"full_{c}"] = bench(full_step, f"Full E+grad (NL {c}Å, K={K})", **bkw)

        # ==============================================================
        # 7. LANGEVIN STEP
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("7. LANGEVIN STEP (complete sim step)")
        log.info("-" * 75)

        try:
            from prolix.batched_simulate import make_langevin_step_nl
            key = jax.random.PRNGKey(42)

            for c in [10.0, 12.0, 20.0]:
                nl, K = nls[c]
                step_fn = make_langevin_step_nl(sys0, nl, displacement_fn)
                @jax.jit
                def lang_step(key=key, step_fn=step_fn):
                    return step_fn(sys0.positions, sys0.velocities, key)
                results[f"langevin_{c}"] = bench(
                    lang_step, f"Langevin step (NL {c}Å, K={K})", **bkw
                )
        except Exception as e:
            log.warning(f"  Langevin benchmark skipped: {e}")

        # Stop trace
        if args.trace:
            jax.profiler.stop_trace()
            log.info(f"\nTrace saved to {trace_dir}")

        # ==============================================================
        # SUMMARY TABLE
        # ==============================================================
        log.info("\n" + "=" * 75)
        log.info("SUMMARY — Component Breakdown at 10Å cutoff")
        log.info("=" * 75)

        full_10 = results.get("full_10.0", 1.0)
        components_10 = [
            ("Born radii (NL 10Å)", results.get("born_nl_10.0")),
            ("GB grad (NL CVJP 10Å)", results.get("gb_g_nl_10.0")),
            ("LJ fused E+F (10Å)", results.get("lj_10.0")),
            ("Bonded gradient", results.get("bonded_g")),
        ]
        log.info(f"\n  {'Component':45s} {'Time (ms)':>10s} {'% of step':>10s}")
        log.info(f"  {'-'*45} {'-'*10} {'-'*10}")
        total_parts = 0.0
        for name, t in components_10:
            if t is not None:
                pct = t / full_10 * 100
                log.info(f"  {name:45s} {t:10.3f} {pct:9.1f}%")
                total_parts += t
        log.info(f"  {'Sum of components':45s} {total_parts:10.3f} {total_parts/full_10*100:9.1f}%")
        log.info(f"  {'Full E+grad (10Å)':45s} {full_10:10.3f} {'100.0':>10s}%")
        overhead = full_10 - total_parts
        log.info(f"  {'Overhead (XLA fusion/scheduling)':45s} {overhead:10.3f} {overhead/full_10*100:9.1f}%")

        log.info("\n" + "=" * 75)
        log.info("Cutoff impact on full step:")
        log.info("-" * 75)
        for c in CUTOFFS:
            full_c = results.get(f"full_{c}")
            if full_c:
                log.info(f"  {c:5.1f}Å: {full_c:8.3f} ms  "
                         f"(→ {1000/full_c:.0f} steps/s, "
                         f"{1000/full_c * 0.002:.2f} ns/s at 2fs)")

        log.info("\n" + "=" * 75)
        log.info("DONE")
        log.info("=" * 75)


if __name__ == "__main__":
    main()
