#!/usr/bin/env python3
"""Parity test: fused energy+force vs jax.grad of CVJP energy.

Validates that the fused analytical forces match the reference
jax.grad implementation to within numerical tolerance.

Usage:
    uv run python tests/test_fused_parity.py
"""

import os
import sys
import logging

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from jax_md import dataclasses as jax_dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_batched_pipeline import SYSTEM_CATALOG, load_and_parameterize, prepare_batches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("fused_parity")


def main():
    logger.info("Loading 1X2G...")
    protein = load_and_parameterize(SYSTEM_CATALOG["1X2G"])
    batches = prepare_batches([protein], ["1X2G"], n_replicas=1)

    for bucket_size, (batch, names) in batches.items():
        sys0 = jax.tree.map(lambda x: x[0], batch)
        N = sys0.positions.shape[0]
        n_real = int(sys0.n_real_atoms)
        logger.info(f"System: N={N}, n_real={n_real}")

        displacement_fn, _ = space.free()

        # Build a static neighbor list for testing
        NL_CUTOFF = 20.0
        pos_np = np.array(sys0.positions)
        dists = np.linalg.norm(
            pos_np[:n_real, None, :] - pos_np[None, :n_real, :], axis=-1
        )
        max_k = 0
        for i in range(n_real):
            k = int(np.sum((dists[i] < NL_CUTOFF) & (dists[i] > 0)))
            max_k = max(max_k, k)
        max_k = min(max_k + 32, N)
        neighbor_idx_np = np.full((N, max_k), N, dtype=np.int32)
        for i in range(n_real):
            nbrs = np.where((dists[i] < NL_CUTOFF) & (dists[i] > 0))[0]
            neighbor_idx_np[i, :len(nbrs)] = nbrs
        neighbor_idx = jnp.array(neighbor_idx_np)
        logger.info(f"Neighbor list: K={max_k}")

        # === Reference: jax.grad of CVJP energy ===
        from prolix.batched_energy import single_padded_energy_nl_cvjp

        def ref_energy_and_grad(sys):
            def e_fn(r):
                s = jax_dataclasses.replace(sys, positions=r)
                return single_padded_energy_nl_cvjp(s, neighbor_idx, displacement_fn)
            e = e_fn(sys.positions)
            g = jax.grad(e_fn)(sys.positions)
            return e, -g  # force = -grad

        logger.info("Computing reference energy + forces (jax.grad of CVJP)...")
        ref_energy, ref_forces = ref_energy_and_grad(sys0)
        jax.block_until_ready(ref_forces)
        logger.info(f"  Reference energy: {float(ref_energy):.4f} kcal/mol")

        # === Fused: analytical forces ===
        from prolix.fused_energy import fused_energy_and_forces_nl

        @jax.jit
        def fused_call(sys):
            return fused_energy_and_forces_nl(sys, neighbor_idx, displacement_fn)

        logger.info("Computing fused energy + forces...")
        fused_energy, fused_forces = fused_call(sys0)
        jax.block_until_ready(fused_forces)
        logger.info(f"  Fused energy: {float(fused_energy):.4f} kcal/mol")

        # === Parity checks ===
        logger.info(f"Ref forces have NaN?   {bool(jnp.isnan(ref_forces).any())}")
        logger.info(f"Fused forces have NaN? {bool(jnp.isnan(fused_forces).any())}")
        
        energy_diff = abs(float(ref_energy) - float(fused_energy))
        logger.info(f"\nEnergy difference: {energy_diff:.6f} kcal/mol")

        force_diff = np.array(ref_forces - fused_forces)
        force_mae = np.mean(np.abs(force_diff[:n_real]))
        force_max = np.max(np.abs(force_diff[:n_real]))
        force_rms = np.sqrt(np.mean(force_diff[:n_real] ** 2))
        logger.info(f"Force MAE (real atoms): {force_mae:.6f} kcal/mol/Å")
        logger.info(f"Force MAX (real atoms): {force_max:.6f} kcal/mol/Å")
        logger.info(f"Force RMS (real atoms): {force_rms:.6f} kcal/mol/Å")

        # Cross-check: LJ forces only
        from prolix.fused_energy import compute_pairwise_nl, lj_energy_and_force_nl

        @jax.jit
        def lj_fused_only(sys):
            dr, dist, safe_idx = compute_pairwise_nl(sys.positions, neighbor_idx)
            return lj_energy_and_force_nl(
                dr, dist, sys.sigmas, sys.epsilons,
                safe_idx, neighbor_idx,
            )

        from prolix.batched_energy import _make_lj_energy_nl_cvjp

        @jax.jit
        def lj_grad_only(sys):
            lj_fn = _make_lj_energy_nl_cvjp(neighbor_idx)
            e = lj_fn(sys.positions, sys.sigmas, sys.epsilons)
            f = -jax.grad(lambda r: _make_lj_energy_nl_cvjp(neighbor_idx)(
                r, sys.sigmas, sys.epsilons))(sys.positions)
            return e, f

        logger.info("\n--- LJ-only parity ---")
        e_fused_lj, f_fused_lj = lj_fused_only(sys0)
        e_grad_lj, f_grad_lj = lj_grad_only(sys0)
        jax.block_until_ready(f_fused_lj)
        jax.block_until_ready(f_grad_lj)

        lj_e_diff = abs(float(e_fused_lj) - float(e_grad_lj))
        lj_f_diff = np.array(f_fused_lj - f_grad_lj)
        lj_f_mae = np.mean(np.abs(lj_f_diff[:n_real]))
        lj_f_max = np.max(np.abs(lj_f_diff[:n_real]))
        logger.info(f"LJ energy diff: {lj_e_diff:.6f}")
        logger.info(f"LJ force MAE: {lj_f_mae:.6f}")
        logger.info(f"LJ force MAX: {lj_f_max:.6f}")

        # === Timing comparison ===
        import timeit

        logger.info("\n--- Timing comparison ---")

        # Warmup
        _ = ref_energy_and_grad(sys0)
        jax.block_until_ready(_[1])
        _ = fused_call(sys0)
        jax.block_until_ready(_[1])

        t_ref = timeit.repeat(
            lambda: jax.block_until_ready(ref_energy_and_grad(sys0)),
            repeat=5, number=1,
        )
        t_fused = timeit.repeat(
            lambda: jax.block_until_ready(fused_call(sys0)),
            repeat=5, number=1,
        )

        avg_ref = sum(t_ref) / len(t_ref) * 1000
        avg_fused = sum(t_fused) / len(t_fused) * 1000
        speedup = avg_ref / avg_fused

        logger.info(f"Reference (jax.grad): {avg_ref:.2f} ms")
        logger.info(f"Fused:                {avg_fused:.2f} ms")
        logger.info(f"Speedup:              {speedup:.2f}x")

        # Verdict
        e_pass = energy_diff < 1.0
        f_pass = force_mae < 0.5
        logger.info(f"\n{'PASS' if e_pass and f_pass else 'FAIL'}: "
                     f"energy {'✓' if e_pass else '✗'} (<1.0), "
                     f"force MAE {'✓' if f_pass else '✗'} (<0.5)")


if __name__ == "__main__":
    main()
