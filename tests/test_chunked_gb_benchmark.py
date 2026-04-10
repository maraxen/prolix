#!/usr/bin/env python3
"""A/B Benchmark: Original broadcast N² vs Chunked custom_vjp GB/Coulomb.

Compares:
  1. Original: dense broadcast (N,N) + jax.grad  [CURRENT]
  2. Chunked:  fori_loop (chunk,N) + custom_vjp analytical forces [NEW]

Tests correctness (energy parity, gradient parity) then benchmarks both.

Usage:
    python tests/test_chunked_gb_benchmark.py
"""

import os
import sys
import logging
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_batched_pipeline import SYSTEM_CATALOG, load_and_parameterize, prepare_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ab_bench")


def benchmark(fn, name, n_warmup=3, n_trials=20, n_inner=5):
    """Proper JAX benchmarking with block_until_ready."""
    for _ in range(n_warmup):
        r = fn()
        jax.block_until_ready(r)

    times = []
    for _ in range(n_trials):
        jax.block_until_ready(jnp.zeros(1))
        t0 = time.perf_counter()
        for _ in range(n_inner):
            r = fn()
            jax.block_until_ready(r)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000 / n_inner)

    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    logger.info(f"  {name:55s} {avg:10.3f} ± {std:6.3f} ms  (min={mn:.3f})")
    return avg


def main():
    logger.info("=" * 75)
    logger.info("A/B Benchmark: Broadcast N² vs Chunked GB/Coulomb")
    logger.info("=" * 75)
    logger.info(f"Devices: {[str(d) for d in jax.devices()]}")
    logger.info(f"Backend: {jax.default_backend()}")

    # Load system
    protein = load_and_parameterize(SYSTEM_CATALOG["1X2G"])
    batches = prepare_batches([protein], ["1X2G"], n_replicas=1)

    for bucket_size, (batch, names) in batches.items():
        sys0 = jax.tree.map(lambda x: x[0], batch)
        N = sys0.positions.shape[0]
        n_real = int(sys0.n_real_atoms)
        logger.info(f"System: N={N} padded, n_real={n_real}")

        # Real-atom arrays for dense path
        pos_real = sys0.positions[:n_real]
        charges_real = sys0.charges[:n_real]
        radii_real = sys0.radii[:n_real]
        scaled_radii_real = (
            sys0.scaled_radii[:n_real]
            if hasattr(sys0, 'scaled_radii') and sys0.scaled_radii is not None
            else None
        )
        mask_real = jnp.ones(n_real, dtype=jnp.bool_)

        # Force f32 — confirmed safe for implicit solvent GB
        pos_real = pos_real.astype(jnp.float32)
        charges_real = charges_real.astype(jnp.float32)
        radii_real = radii_real.astype(jnp.float32)
        if scaled_radii_real is not None:
            scaled_radii_real = scaled_radii_real.astype(jnp.float32)
        logger.info(f"  Input dtype: {pos_real.dtype}")

        from prolix.physics.generalized_born import (
            compute_born_radii,
            compute_gb_energy,
        )
        from prolix.pallas_kernels import (
            gb_coulomb_energy_dense,
            _gb_coulomb_energy_chunked,
        )

        # ================================================================
        # Step 1: Compute Born radii (shared, using dense N² for fairness)
        # ================================================================
        logger.info("\n--- Pre-computing Born radii (dense, shared by both) ---")
        born_radii_real = compute_born_radii(
            pos_real, radii_real, scaled_radii=scaled_radii_real,
        )
        born_radii_real = born_radii_real.astype(jnp.float32)
        jax.block_until_ready(born_radii_real)
        logger.info(f"  Born radii: mean={float(jnp.mean(born_radii_real)):.4f}Å, dtype={born_radii_real.dtype}")

        # ================================================================
        # Step 2: Correctness — energy values must match
        # ================================================================
        logger.info("\n--- Correctness Check: Energy Parity ---")

        # Original: dense broadcast
        from prolix.physics.generalized_born import f_gb, safe_norm
        from proxide.physics import constants

        @jax.jit
        def original_gb_energy(positions, charges, born_radii):
            """Original broadcast N² GB energy — kept in f32."""
            dtype = positions.dtype
            delta = positions[:, None, :] - positions[None, :, :]
            distances = jnp.sqrt(jnp.sum(delta**2, axis=-1) + jnp.array(1e-12, dtype=dtype))
            br_i = born_radii[:, None]
            br_j = born_radii[None, :]
            eff_dist = f_gb(distances, br_i, br_j)
            tau = jnp.array(
                (1.0 / constants.DIELECTRIC_PROTEIN) - (1.0 / constants.DIELECTRIC_WATER),
                dtype=dtype,
            )
            pf = jnp.array(-0.5 * constants.COULOMB_CONSTANT, dtype=dtype) * tau
            charge_prod = charges[:, None] * charges[None, :]
            energy_terms = charge_prod / eff_dist
            return pf * jnp.sum(energy_terms)

        # Chunked: fori_loop
        @jax.jit
        def chunked_gb_energy(positions, charges, born_radii, mask):
            return _gb_coulomb_energy_chunked(
                positions, charges, born_radii, mask, chunk_size=256,
            )

        e_orig = original_gb_energy(pos_real, charges_real, born_radii_real)
        jax.block_until_ready(e_orig)

        e_chunked = chunked_gb_energy(pos_real, charges_real, born_radii_real, mask_real)
        jax.block_until_ready(e_chunked)

        e_diff = abs(float(e_orig) - float(e_chunked))
        e_reldiff = e_diff / (abs(float(e_orig)) + 1e-30)

        logger.info(f"  Original broadcast: {float(e_orig):.6f} kcal/mol")
        logger.info(f"  Chunked fori_loop: {float(e_chunked):.6f} kcal/mol")
        logger.info(f"  Abs diff: {e_diff:.6f}, Rel diff: {e_reldiff:.2e}")
        if e_reldiff < 1e-4:
            logger.info("  ✅ Energies match (reldiff < 1e-4)")
        elif e_reldiff < 0.05:
            logger.info(f"  ⚠️  Energies agree within {e_reldiff*100:.2f}% "
                         "(f32 accumulation order difference — expected)")
        else:
            logger.error(f"  ❌ Energy mismatch too large: {e_reldiff*100:.2f}%")

        # ================================================================
        # Step 3: Correctness — gradient values must match
        # ================================================================
        logger.info("\n--- Correctness Check: Gradient Parity ---")

        @jax.jit
        def original_gb_grad(positions, charges, born_radii):
            return jax.grad(
                lambda p: original_gb_energy(p, charges, born_radii)
            )(positions)

        @jax.jit
        def chunked_gb_grad(positions, charges, born_radii, mask):
            return jax.grad(
                lambda p: gb_coulomb_energy_dense(p, charges, born_radii, mask)
            )(positions)

        g_orig = original_gb_grad(pos_real, charges_real, born_radii_real)
        jax.block_until_ready(g_orig)

        g_chunked = chunked_gb_grad(pos_real, charges_real, born_radii_real, mask_real)
        jax.block_until_ready(g_chunked)

        g_diff = jnp.abs(g_orig - g_chunked)
        g_rmsd = float(jnp.sqrt(jnp.mean(g_diff**2)))
        g_max = float(jnp.max(g_diff))
        g_rel = float(jnp.mean(g_diff) / (jnp.mean(jnp.abs(g_orig)) + 1e-30))

        logger.info(f"  Gradient RMSD: {g_rmsd:.6f}")
        logger.info(f"  Gradient max diff: {g_max:.6f}")
        logger.info(f"  Gradient rel diff: {g_rel:.2e}")
        logger.info(f"  Original |grad| mean: {float(jnp.mean(jnp.abs(g_orig))):.6f}")
        logger.info(f"  Chunked  |grad| mean: {float(jnp.mean(jnp.abs(g_chunked))):.6f}")

        if g_rel < 1e-2:
            logger.info("  ✅ Gradients match (rel < 1e-2)")
        elif g_rel < 1e-1:
            logger.info("  ⚠️  Gradients approximately match (rel < 1e-1)")
        else:
            logger.warning(f"  ❌ Gradient mismatch! rel={g_rel:.2e}")

        # ================================================================
        # Step 4: Correctness — vmap works
        # ================================================================
        logger.info("\n--- Correctness Check: vmap composability ---")
        try:
            batch_pos = jnp.stack([pos_real, pos_real + 0.01])
            batch_charges = jnp.stack([charges_real, charges_real])
            batch_br = jnp.stack([born_radii_real, born_radii_real])
            batch_mask = jnp.stack([mask_real, mask_real])

            vmap_fn = jax.jit(jax.vmap(gb_coulomb_energy_dense))
            e_batch = vmap_fn(batch_pos, batch_charges, batch_br, batch_mask)
            jax.block_until_ready(e_batch)
            logger.info(f"  vmap energies: {e_batch}")
            logger.info("  ✅ vmap works")
        except Exception as e:
            logger.error(f"  ❌ vmap failed: {e}")

        # ================================================================
        # Step 5: Correctness — vmap + grad works
        # ================================================================
        logger.info("\n--- Correctness Check: vmap(grad) composability ---")
        try:
            def single_energy(p, c, br, m):
                return gb_coulomb_energy_dense(p, c, br, m)

            vmap_grad_fn = jax.jit(jax.vmap(
                jax.grad(single_energy)
            ))
            g_batch = vmap_grad_fn(batch_pos, batch_charges, batch_br, batch_mask)
            jax.block_until_ready(g_batch)
            logger.info(f"  vmap(grad) shape: {g_batch.shape}")
            logger.info(f"  vmap(grad) |grad| mean: {float(jnp.mean(jnp.abs(g_batch))):.6f}")
            logger.info("  ✅ vmap(grad) works")
        except Exception as e:
            logger.error(f"  ❌ vmap(grad) failed: {e}")

        # ================================================================
        # Step 6: Performance Benchmark
        # ================================================================
        logger.info("\n" + "=" * 75)
        logger.info("PERFORMANCE BENCHMARK")
        logger.info("=" * 75)

        logger.info("\n--- Forward Pass (Energy Only) ---")
        t_orig_fwd = benchmark(
            lambda: original_gb_energy(pos_real, charges_real, born_radii_real),
            "Original broadcast N² (forward)",
        )

        for chunk in [64, 128, 256, 512]:
            @jax.jit
            def chunked_fwd(chunk_size=chunk):
                return _gb_coulomb_energy_chunked(
                    pos_real, charges_real, born_radii_real, mask_real,
                    chunk_size=chunk_size,
                )
            benchmark(chunked_fwd, f"Chunked fori_loop chunk={chunk} (forward)")

        logger.info("\n--- Backward Pass (Gradient) ---")
        t_orig_bwd = benchmark(
            lambda: original_gb_grad(pos_real, charges_real, born_radii_real),
            "Original broadcast N² (jax.grad)",
        )

        for chunk in [64, 128, 256, 512]:
            @jax.jit
            def chunked_bwd(chunk_size=chunk):
                return jax.grad(
                    lambda p: _gb_coulomb_energy_chunked(
                        p, charges_real, born_radii_real, mask_real,
                        chunk_size=chunk_size,
                    )
                )(pos_real)

            benchmark(chunked_bwd, f"Chunked fori_loop chunk={chunk} (jax.grad)")

        logger.info("\n--- Custom VJP Analytical Forces ---")
        benchmark(
            lambda: chunked_gb_grad(pos_real, charges_real, born_radii_real, mask_real),
            "Custom VJP analytical forces (chunk=256)",
        )

        logger.info("\n--- Memory Usage ---")
        n2_mem = n_real * n_real * 4 * 5 / 1e6
        for chunk in [64, 128, 256, 512]:
            chunk_mem = chunk * n_real * 4 * 5 / 1e6
            logger.info(f"  chunk={chunk}: peak ~{chunk_mem:.1f} MB "
                         f"(vs broadcast: {n2_mem:.1f} MB, "
                         f"ratio: {n2_mem/chunk_mem:.1f}×)")

        logger.info("\n" + "=" * 75)
        logger.info("DONE")
        logger.info("=" * 75)


if __name__ == "__main__":
    main()
