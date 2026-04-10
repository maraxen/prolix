#!/usr/bin/env python3
"""FIRE Minimization Convergence Diagnostic.

Tests FIRE convergence at different step counts on a single system
to determine the optimal min_steps for production.

Usage:
    uv run python scripts/diagnose_min_convergence.py

Output: convergence table showing rms_grad, max_grad, energy, wall_time
per checkpoint (every 1000 FIRE steps) up to 50,000 total steps.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "0")

import time
import logging
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from jax_md import minimize as jax_md_minimize
import dataclasses

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    from proxide.io.parsing.backend import parse_structure
    from proxide import OutputSpec, CoordFormat
    from prolix.padding import pad_protein
    from prolix.batched_energy import single_padded_energy, _position_restraint_energy

    # --- Load a single test system ---
    pdb = "references/pdb/6XHB_chainA_fixed.pdb"
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        parameterize_md=True,
        force_field="proxide/src/proxide/assets/protein.ff19SB.xml",
        add_hydrogens=False,
        remove_solvent=True,
        remove_hetatm=True,
    )
    protein = parse_structure(pdb, spec)
    sys = pad_protein(protein, target_atoms=4096)

    n_real = int(sys.atom_mask.sum())
    n_constr = int(sys.constraint_mask.sum())
    logger.info(f"System: 6XHB, {n_real} real atoms, {n_constr} constraints")
    logger.info(f"Initial maxcoord: {float(jnp.max(jnp.abs(sys.positions * sys.atom_mask[:, None]))):.1f}")

    # --- Setup energy function (same as batched_minimize) ---
    displacement_fn, shift_fn = space.free()
    pad_mask_3d = sys.atom_mask[:, None]
    r_ref = sys.positions

    def energy_fn(r, soft_core_lambda=jnp.float32(1.0), k_restraint=jnp.float32(0.0)):
        sys_with_r = dataclasses.replace(sys, positions=r)
        e = single_padded_energy(
            sys_with_r, displacement_fn,
            soft_core_lambda=soft_core_lambda,
        )
        e = e + _position_restraint_energy(r, r_ref, k_restraint, sys.atom_mask)
        return e

    # --- Convergence function ---
    @jax.jit
    def compute_grad_stats(positions):
        """Compute gradient statistics at λ=0.9999."""
        val, grad = jax.value_and_grad(
            lambda pos: energy_fn(pos, soft_core_lambda=jnp.float32(0.9999))
        )(positions)
        grad = jnp.where(pad_mask_3d, grad, 0.0)
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        g_per_atom = jnp.sqrt(jnp.sum(grad ** 2, axis=-1))
        g_real = g_per_atom * sys.atom_mask
        n = jnp.sum(sys.atom_mask)
        rms = jnp.sqrt(jnp.sum(g_real ** 2) / jnp.maximum(n, 1.0))
        mx = jnp.max(g_real)
        n_high = jnp.sum(g_real > 100.0)
        return val, rms, mx, n_high

    # --- Run FIRE in increments ---
    stage_lambdas = jnp.array([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.9999])
    stage_caps = jnp.array([10.0, 100.0, 1000.0, 5000.0, 10000.0, 5000.0, 1000.0])
    stage_restraints = jnp.array([100.0, 50.0, 10.0, 1.0, 0.0, 0.0, 0.0])

    dt_start = 0.00005
    dt_max = 0.0005
    fire_init_fn, fire_apply_fn = jax_md_minimize.fire_descent(
        energy_fn, shift_fn, dt_start=dt_start, dt_max=dt_max,
    )

    # Test multiple total step counts
    total_budgets = [3000, 5000, 10000, 20000, 30000, 50000]

    logger.info("\n" + "=" * 80)
    logger.info("FIRE Convergence Sweep")
    logger.info("=" * 80)
    logger.info(f"{'Steps':>8} {'Time(s)':>8} {'Energy':>14} {'rms_grad':>10} {'max_grad':>10} {'n>100':>6}")
    logger.info("-" * 60)

    # Initial stats
    e, rms, mx, n_high = compute_grad_stats(sys.positions)
    logger.info(f"{'init':>8} {'--':>8} {float(e):>14.1f} {float(rms):>10.1f} {float(mx):>10.1f} {int(n_high):>6}")

    for total_steps in total_budgets:
        # Build stage steps for this budget
        remaining = max(total_steps - 4000, 1000)
        stage_steps = jnp.array([500, 500, 500, 500, 1000, remaining, 2000])

        # Run FIRE
        positions = sys.positions

        t0 = time.time()

        for si in range(7):
            sc_lam = stage_lambdas[si]
            fcap = stage_caps[si]
            k_rest = stage_restraints[si]
            n_steps = int(stage_steps[si])

            fire_state = fire_init_fn(
                positions, mass=sys.masses,
                soft_core_lambda=sc_lam, k_restraint=k_rest,
            )

            def capped_apply(state, fcap, sc_lam, k_rest):
                f = state.force * pad_mask_3d
                f_norm = jnp.linalg.norm(f, axis=-1, keepdims=True)
                cap_ratio = jnp.minimum(1.0, fcap / (f_norm + 1e-8))
                f = f * cap_ratio
                f = jnp.where(jnp.isfinite(f), f, 0.0)
                state = dataclasses.replace(state, force=f)
                return fire_apply_fn(state, soft_core_lambda=sc_lam, k_restraint=k_rest)

            @jax.jit
            def run_stage(positions_in, sc_lam, fcap, k_rest, n_steps):
                fire_s = fire_init_fn(
                    positions_in, mass=sys.masses,
                    soft_core_lambda=sc_lam, k_restraint=k_rest,
                )
                def body(i, s):
                    s = capped_apply(s, fcap, sc_lam, k_rest)
                    new_pos = jnp.where(pad_mask_3d, s.position, r_ref)
                    new_mom = jnp.where(pad_mask_3d, s.momentum, 0.0)
                    new_force = jnp.where(pad_mask_3d, s.force, 0.0)
                    return dataclasses.replace(
                        s, position=new_pos, momentum=new_mom, force=new_force,
                    )
                final_s = jax.lax.fori_loop(0, n_steps, body, fire_s)
                return final_s.position

            positions = run_stage(positions, sc_lam, fcap, k_rest, n_steps)
            positions.block_until_ready()

        elapsed = time.time() - t0

        # Evaluate convergence
        e, rms, mx, n_high = compute_grad_stats(positions)
        logger.info(
            f"{total_steps:>8} {elapsed:>8.1f} {float(e):>14.1f} "
            f"{float(rms):>10.1f} {float(mx):>10.1f} {int(n_high):>6}"
        )

    logger.info("=" * 80)
    logger.info("Target: rms_grad < 10, max_grad < 100")
    logger.info("Recommendation: use the lowest step count that achieves the target.")


if __name__ == "__main__":
    main()
