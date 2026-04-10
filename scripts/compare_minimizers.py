#!/usr/bin/env python3
"""Head-to-head minimizer comparison on 6XHB.

Protocols tested:
  A) SD (staged) → L-BFGS
  B) SD (staged) → FIRE (staged) → L-BFGS
  C) FIRE only (staged) → L-BFGS  [current production baseline]

SD = steepest descent with displacement capping (no inertia).
FIRE = momentum-based with dt/velocity adaptation.
L-BFGS = quasi-Newton polish at λ=0.9999.

Reports: rms_grad, max_grad, energy, wall_time, Rg for each protocol.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "0")

import time
import logging
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from jax_md import minimize as jax_md_minimize

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    from proxide.io.parsing.backend import parse_structure
    from proxide import OutputSpec, CoordFormat
    from prolix.padding import pad_protein
    from prolix.batched_energy import single_padded_energy, _position_restraint_energy

    # --- Load test system ---
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
    log.info(f"System: 6XHB, {n_real} real atoms")

    displacement_fn, shift_fn = space.free()
    pad_mask_3d = sys.atom_mask[:, None]
    r_ref = sys.positions

    def energy_fn(r, soft_core_lambda=jnp.float32(1.0),
                  k_restraint=jnp.float32(0.0)):
        sys_r = dataclasses.replace(sys, positions=r)
        e = single_padded_energy(
            sys_r, displacement_fn, soft_core_lambda=soft_core_lambda)
        e = e + _position_restraint_energy(r, r_ref, k_restraint, sys.atom_mask)
        return e

    # --- Gradient stats ---
    @jax.jit
    def grad_stats(positions):
        val, grad = jax.value_and_grad(
            lambda p: energy_fn(p, soft_core_lambda=jnp.float32(0.9999))
        )(positions)
        grad = jnp.where(pad_mask_3d, grad, 0.0)
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        g_per = jnp.sqrt(jnp.sum(grad**2, axis=-1))
        g_real = g_per * sys.atom_mask
        n = jnp.sum(sys.atom_mask)
        rms = jnp.sqrt(jnp.sum(g_real**2) / jnp.maximum(n, 1.0))
        mx = jnp.max(g_real)
        return val, rms, mx

    def compute_rg(positions):
        m = sys.masses * sys.atom_mask
        total_m = jnp.sum(m) + 1e-12
        com = jnp.sum(positions * m[:, None], axis=0) / total_m
        dr = positions - com[None, :]
        rg2 = jnp.sum(m * jnp.sum(dr**2, axis=-1)) / total_m
        return jnp.sqrt(rg2 + 1e-12)

    # === Soft-core stages (shared) ===
    stage_lambdas = [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.9999]
    stage_caps = [10.0, 100.0, 1000.0, 5000.0, 10000.0, 5000.0, 1000.0]
    stage_restraints = [100.0, 50.0, 10.0, 1.0, 0.0, 0.0, 0.0]

    # =========================================================
    # Steepest Descent with displacement capping (no momentum)
    # =========================================================
    def run_sd_stages(positions, steps_per_stage, max_disp=0.05):
        """Steepest descent with per-atom displacement capping.

        No momentum, no velocity — just gradient steps with a hard
        displacement limit. This is what GROMACS/AMBER use for initial
        clash resolution. max_disp in Angstroms.
        """
        for si in range(7):
            sc_lam = jnp.float32(stage_lambdas[si])
            k_rest = jnp.float32(stage_restraints[si])
            n = steps_per_stage[si]

            @jax.jit
            def sd_stage(pos, sc_lam=sc_lam, k_rest=k_rest, n_steps=n):
                def body(i, p):
                    grad = jax.grad(
                        lambda r: energy_fn(r, soft_core_lambda=sc_lam,
                                            k_restraint=k_rest)
                    )(p)
                    # Zero padding gradients
                    grad = jnp.where(pad_mask_3d, grad, 0.0)
                    grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
                    # Displacement capping: limit max per-atom move
                    g_norm = jnp.sqrt(
                        jnp.sum(grad**2, axis=-1, keepdims=True) + 1e-30)
                    scale = jnp.minimum(1.0, max_disp / g_norm)
                    step = -grad * scale  # negative gradient direction
                    new_p = p + step
                    # Reset padding atoms
                    new_p = jnp.where(pad_mask_3d, new_p, r_ref)
                    return new_p
                return jax.lax.fori_loop(0, n_steps, body, pos)

            positions = sd_stage(positions)
            positions.block_until_ready()
        return positions

    # =========================================================
    # FIRE stages (Staged Minimization)
    # =========================================================
    def run_fire_stages(positions, steps_per_stage, use_displacement_capping=False,
                        heavy_atom_restraints=False):
        """FIRE with soft-core stages, optional capping and restraints."""
        fire_init_fn, fire_apply_fn = jax_md_minimize.fire_descent(
            energy_fn, shift_fn, dt_start=0.00005, dt_max=0.0005)

        # Identify heavy atoms if restraints requested
        element_ids = getattr(sys, "element_ids", None)
        if element_ids is not None:
            restraint_mask = (element_ids > 1).astype(jnp.float32)
        else:
            restraint_mask = (sys.masses > 1.1).astype(jnp.float32)

        for si in range(len(steps_per_stage)):
            sc_lam = jnp.float32(stage_lambdas[si])
            fcap = jnp.float32(stage_caps[si])
            k_rest = jnp.float32(stage_restraints[si]) if heavy_atom_restraints else jnp.float32(0.0)
            n = steps_per_stage[si]

            @jax.jit
            def fire_stage(pos, sc_lam=sc_lam, fcap=fcap, k_rest=k_rest,
                           n_steps=n):
                fire_s = fire_init_fn(pos, mass=sys.masses,
                                     soft_core_lambda=sc_lam,
                                     k_restraint=k_rest)
                
                def body(i, s):
                    # 1. Force capping
                    f = s.force * pad_mask_3d
                    f_norm = jnp.linalg.norm(f, axis=-1, keepdims=True)
                    cap_ratio = jnp.minimum(1.0, fcap / (f_norm + 1e-8))
                    f_capped = f * cap_ratio
                    f_safe = jnp.where(jnp.isfinite(f_capped), f_capped, 0.0)
                    s = dataclasses.replace(s, force=f_safe)
                    
                    # 2. Apply FIRE step
                    prev_pos = s.position
                    s = fire_apply_fn(s, soft_core_lambda=sc_lam,
                                     k_restraint=k_rest)
                    
                    # 3. Optional Displacement Capping (|dr| < 0.1 Å)
                    if use_displacement_capping:
                        dr = s.position - prev_pos
                        dr_norm = jnp.linalg.norm(dr, axis=-1, keepdims=True)
                        scale = jnp.minimum(1.0, 0.1 / (dr_norm + 1e-8))
                        new_pos = prev_pos + dr * scale
                        new_vel = s.momentum / (sys.masses[:, None] + 1e-8) * scale
                        new_mom = new_vel * sys.masses[:, None]
                        s = dataclasses.replace(s, position=new_pos, momentum=new_mom)

                    # 4. Padding reset
                    new_pos = jnp.where(pad_mask_3d, s.position, r_ref)
                    new_mom = jnp.where(pad_mask_3d, s.momentum, 0.0)
                    new_force = jnp.where(pad_mask_3d, s.force, 0.0)
                    return dataclasses.replace(
                        s, position=new_pos, momentum=new_mom,
                        force=new_force)
                
                final_s = jax.lax.fori_loop(0, n_steps, body, fire_s)
                return final_s.position
            positions = fire_stage(positions)
            positions.block_until_ready()
        return positions

    # =========================================================
    # L-BFGS polish (shared)
    # =========================================================
    def run_lbfgs(positions, max_iter=200):
        import jaxopt
        real_mask_3d = (sys.masses > 0.0)[:, None]

        def lbfgs_vg(r):
            val, grad = jax.value_and_grad(
                lambda p: energy_fn(p, soft_core_lambda=jnp.float32(0.9999))
            )(r)
            grad = jnp.where(real_mask_3d, grad, 0.0)
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            g_mag = jnp.sqrt(jnp.sum(grad**2, axis=-1, keepdims=True) + 1e-30)
            grad = grad * jnp.minimum(1.0, 100000.0 / g_mag)
            val = jnp.where(jnp.isfinite(val), val, jnp.float32(1e10))
            return val, grad

        solver = jaxopt.LBFGS(
            fun=lbfgs_vg, value_and_grad=True, maxiter=max_iter,
            tol=1e-6, history_size=10, max_stepsize=0.1,
            min_stepsize=1e-10, linesearch="backtracking",
            stop_if_linesearch_fails=True, jit=True, unroll=False)
        result, _ = solver.run(positions)
        result = jnp.where(real_mask_3d, result, positions)
        return result

    # =========================================================
    # Run protocols
    # =========================================================
    # SD step budgets per stage (roughly matching FIRE's budget distribution)
    sd_steps_short = [200, 200, 200, 200, 400, 500, 500]    # ~2200 total
    sd_steps_long  = [500, 500, 500, 500, 1000, 2000, 2000]  # ~7000 total
    fire_steps_10k = [500, 500, 500, 500, 1000, 5000, 2000]  # 10k (best from sweep)

    protocols = [
        ("A: SD(2.2k) → LBFGS(200)", "sd_short_lbfgs"),
        ("B: SD(7k) → LBFGS(200)", "sd_long_lbfgs"),
        ("C: SD(2.2k) → FIRE(10k) → LBFGS(200)", "sd_fire_lbfgs"),
        ("D: FIRE(10k) → LBFGS(200)  [current]", "fire_lbfgs"),
        ("E: FIRE(10k) + Restraints + Capping → LBFGS", "robust_fire_lbfgs"),
    ]

    log.info("\n" + "=" * 90)
    log.info("Minimizer Protocol Comparison — 6XHB (2631 atoms, 4096 padded)")
    log.info("=" * 90)

    rg_ref = float(compute_rg(sys.positions))
    e0, rms0, mx0 = grad_stats(sys.positions)
    log.info(f"Reference: Rg={rg_ref:.1f}Å  E={float(e0):.0f}  "
             f"rms_grad={float(rms0):.1f}  max_grad={float(mx0):.1f}\n")

    log.info(f"{'Protocol':<45} {'Time':>7} {'Energy':>14} {'rms_grad':>10} "
             f"{'max_grad':>10} {'Rg':>7} {'Rg/Ref':>7}")
    log.info("-" * 100)

    for label, key in protocols:
        t0 = time.time()
        pos = sys.positions

        if key == "sd_short_lbfgs":
            pos = run_sd_stages(pos, sd_steps_short)
            pos = run_lbfgs(pos)

        elif key == "sd_long_lbfgs":
            pos = run_sd_stages(pos, sd_steps_long)
            pos = run_lbfgs(pos)

        elif key == "sd_fire_lbfgs":
            pos = run_sd_stages(pos, sd_steps_short)
            pos = run_fire_stages(pos, fire_steps_10k)
            pos = run_lbfgs(pos)

        elif key == "fire_lbfgs":
            pos = run_fire_stages(pos, fire_steps_10k,
                                 use_displacement_capping=False,
                                 heavy_atom_restraints=False)
            pos = run_lbfgs(pos)

        elif key == "robust_fire_lbfgs":
            pos = run_fire_stages(pos, fire_steps_10k,
                                 use_displacement_capping=True,
                                 heavy_atom_restraints=True)
            pos = run_lbfgs(pos)

        pos.block_until_ready()
        elapsed = time.time() - t0

        e, rms, mx = grad_stats(pos)
        rg = float(compute_rg(pos))
        log.info(
            f"{label:<45} {elapsed:>6.1f}s {float(e):>14.0f} "
            f"{float(rms):>10.1f} {float(mx):>10.1f} {rg:>6.1f} "
            f"{rg/rg_ref:>7.3f}")

    log.info("=" * 90)
    log.info("Target: rms_grad < 10, max_grad < 100, Rg/Ref ≈ 1.0")


if __name__ == "__main__":
    main()
