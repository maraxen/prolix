#!/usr/bin/env python3
"""Diagnostic: does the R1/R2 angular-momentum correction re-pin T_rot to a
biased intra-cycle reference point (OpenMM issue #2532 pattern)?

Context (task 260805_p5_rottrans_mechanism_c): NVE bisect (gamma=0) shows
T_rot == T_trans (mechanism A, RATTLE-as-sink, ruled out). NVT (gamma>0)
shows T_rot < T_trans by several K. The O-step's noise+projection machinery
was tested in isolation in June (p5_ostep_full_diagnostic.py,
p5_ou_noise_direct_test.py) and checked out. OpenMM's reference SETTLE
(ReferenceSETTLEAlgorithm.cpp) needs no angular-momentum patch at all --
its velocity constraint is an exact analytic solve. Prolix's settle_velocities
is a generic iterative RATTLE loop that DOES drain rotational KE, requiring a
bolted-on `_remove_angular_momentum_from_impulse` fix -- and a second,
separate AM-restoration (`_r_step_conserve_angular_momentum`) runs twice per
step (R1 mid-step, R2 after O), each pinning L to a momentum reference
(p_pre_a1, p_pre_a2) captured at a DIFFERENT point in the BAOAB cycle.

OpenMM issue #2532 documents that velocities sampled at different points
within one BAOAB cycle carry systematically different apparent temperatures
-- a real superconvergence property (Leimkuhler & Matthews), not a bug.

This script does NOT change any production code. It reimplements
settle_langevin's step sequence by calling the same private primitives
directly, instrumented to snapshot momentum (and its T_trans/T_rot
decomposition) at 5 points in the cycle:
    1. p_pre_a1   -- momentum right after the first half-B kick (R1's L_target ref)
    2. p_post_r1  -- momentum after R1's SETTLE+impulse+AM-correction
    3. p_pre_a2   -- momentum right after the O step (R2's L_target ref)
    4. p_post_r2  -- momentum after R2's SETTLE+impulse+AM-correction
    5. p_final    -- momentum after the final B kick + SETTLE_vel (what
                      production code actually returns as state.momentum,
                      i.e. what every gate script measures)

If p_pre_a1's implied T_rot runs systematically low, and p_final tracks it
more closely than p_pre_a2's, that's direct evidence the R1 reference point
is the biased one propagating into the measured deficit.

Usage:
    uv run python scripts/explore/p5_r1_r2_reference_bias.py \
        --n-waters 16 --steps 5000 --burn-in 500 --gamma 10 --dt-fs 1.0 \
        --out /tmp/p5_r1_r2_bias.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / marker).exists():
            return p
    raise RuntimeError(f"project root not found (looked for {marker!r})")


PROJECT_ROOT = _find_project_root()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("p5_r1_r2_reference_bias")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R1 vs R2 AM-correction reference-point bias diagnostic")
    p.add_argument("--n-waters", type=int, default=16)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--gamma", type=float, default=10.0, help="friction in ps^-1")
    p.add_argument("--dt-fs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _decompose_transrot(momentum, mass_flat, n_waters, M_water, boltzmann_kcal):
    """Leftover-KE decomposition -- matches p5_nve_transrot_bisect.py verbatim.

    This is the SAME quantity gate scripts measure as "T_rot"/"T_trans" from
    the returned state.momentum, so p_final here is directly comparable to
    production gate results.
    """
    import numpy as np
    import jax.numpy as jnp

    p3 = momentum.reshape(n_waters, 3, 3)
    v_com = p3.sum(axis=1) / M_water
    ke_trans = float(0.5 * M_water * np.sum(np.asarray(v_com) ** 2))
    ke_tot = float(jnp.sum(jnp.sum(momentum ** 2, axis=-1) / (2.0 * mass_flat)))
    ke_rot = ke_tot - ke_trans
    dof_trans = 3 * n_waters - 3
    dof_rot = 3 * n_waters
    t_trans = 2.0 * ke_trans / (dof_trans * boltzmann_kcal)
    t_rot = 2.0 * ke_rot / (dof_rot * boltzmann_kcal)
    return float(t_trans), float(t_rot)


def _run(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from prolix.physics import pbc, settle, system
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.physics.test_explicit_langevin_tip3p_parity import _equil_water_positions, _proxide_params_pure_water
    from tests.physics.test_p2b_nvt_216water import _make_tip3p_excl_indices

    n_waters = args.n_waters
    seed = args.seed

    log.info("Loading equilibrated TIP3P water box fixture (n_waters=%d, seed=%d) …", n_waters, seed)
    pos_a, box_edge = _equil_water_positions(n_waters, seed=seed)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)

    dt = args.dt_fs / AKMA_TIME_UNIT_FS
    half_dt = 0.5 * dt
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = args.gamma * AKMA_TIME_UNIT_FS * 1e-3  # ps^-1 -> AKMA^-1

    log.info("gamma=%.4g AKMA (input ps^-1=%.4g), dt=%.6g AKMA (%.2f fs), kT=%.6g",
              gamma, args.gamma, dt, args.dt_fs, kT)

    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items() if k != "exclusion_mask"}
    sys_dict["excl_indices"] = _make_tip3p_excl_indices(n_waters)

    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    grid = max(16, round(box_edge / 1.0))

    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict,
        box=box_vec, use_pbc=True, implicit_solvent=False,
        pme_grid_points=grid, pme_alpha=0.34,
        cutoff_distance=9.0, strict_parameterization=False,
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64).reshape(n_atoms)
    mass_col = mass[:, None]
    wi = settle.get_water_indices(0, n_waters)
    r_OH, r_HH = settle.TIP3P_ROH, settle.TIP3P_RHH
    mass_oxygen, mass_hydrogen = 15.999, 1.008
    M_water = mass_oxygen + 2.0 * mass_hydrogen

    force_fn = settle._make_settle_compatible_force_fn(energy_fn, mass, box_vec)

    log.info("Initialising state via settle_langevin.init_fn (unmodified production init) …")
    init_s, _ = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma, mass=mass, water_indices=wi,
        box=box_vec, project_ou_momentum_rigid=True, projection_site="post_o",
        settle_velocity_iters=10,
    )
    state = init_s(jax.random.key(seed), jnp.array(pos_a, dtype=jnp.float64), mass=mass)

    def instrumented_step(position, momentum, force, key):
        """Reimplements settle_langevin's apply_fn (projection_site='post_o'
        default path) verbatim, calling the same private primitives, with
        snapshot capture at the 5 points described in the module docstring.
        """
        positions_old = position

        # B: half force kick
        momentum = settle._langevin_step_b(momentum, force, dt)
        p_pre_a1 = momentum  # <-- snapshot 1: R1's L_target reference

        # R1: A + SETTLE + dp-correction + AM-correction
        p_pre_a1_ref = momentum
        x_unc_1 = settle._langevin_step_a(positions_old, momentum, mass_col, half_dt, shift_fn)
        x_con_1 = settle.settle_positions(
            x_unc_1, positions_old, wi, r_OH, r_HH, mass_oxygen, mass_hydrogen, box_vec, None
        )
        dx_1 = x_con_1 - x_unc_1
        dx_1 = dx_1 - box_vec * jnp.round(dx_1 / box_vec)
        dp_1 = mass_col * dx_1 / half_dt
        momentum = momentum + dp_1
        momentum = settle._r_step_conserve_angular_momentum(
            momentum, p_pre_a1_ref, x_unc_1, x_con_1, wi, mass_oxygen, mass_hydrogen, box=box_vec, water_mask=None
        )
        p_post_r1 = momentum  # <-- snapshot 2

        position = x_con_1
        positions_mid = x_con_1

        # O: stochastic step
        momentum, key = settle._langevin_step_o_constrained(
            momentum, position, mass_col, gamma, dt, kT, key, wi, None
        )
        p_pre_a2 = momentum  # <-- snapshot 3: R2's L_target reference

        # R2: A + SETTLE + dp-correction + AM-correction
        p_pre_a2_ref = momentum
        x_unc_2 = settle._langevin_step_a(position, momentum, mass_col, half_dt, shift_fn)
        x_con_2 = settle.settle_positions(
            x_unc_2, positions_mid, wi, r_OH, r_HH, mass_oxygen, mass_hydrogen, box_vec, None
        )
        dx_2 = x_con_2 - x_unc_2
        dx_2 = dx_2 - box_vec * jnp.round(dx_2 / box_vec)
        dp_2 = mass_col * dx_2 / half_dt
        momentum = momentum + dp_2
        momentum = settle._r_step_conserve_angular_momentum(
            momentum, p_pre_a2_ref, x_unc_2, x_con_2, wi, mass_oxygen, mass_hydrogen, box=box_vec, water_mask=None
        )
        p_post_r2 = momentum  # <-- snapshot 4

        position = x_con_2

        # Force at new constrained positions
        force = force_fn(position, box=box_vec)

        # B: final half force kick
        momentum = settle._langevin_step_b(momentum, force, dt)

        momentum = settle._langevin_settle_vel(
            momentum, positions_mid, position, mass_col, wi, dt, mass_oxygen, mass_hydrogen,
            n_iters=10, settle_velocity_tol=None, water_mask=None,
        )
        p_final = momentum  # <-- snapshot 5: what production actually measures

        return position, momentum, force, key, (p_pre_a1, p_post_r1, p_pre_a2, p_post_r2, p_final)

    step_jit = jax.jit(instrumented_step)

    log.info("Running %d steps (burn-in %d) …", args.steps, args.burn_in)
    position, momentum, force, key = state.positions, state.momentum, state.force, state.key
    records = {k: [] for k in ("pre_a1", "post_r1", "pre_a2", "post_r2", "final")}
    mass_flat = mass.reshape(-1)
    for s in range(1, args.steps + 1):
        position, momentum, force, key, snaps = step_jit(position, momentum, force, key)
        if s > args.burn_in:
            for label, p in zip(("pre_a1", "post_r1", "pre_a2", "post_r2", "final"), snaps):
                t_trans, t_rot = _decompose_transrot(p, mass_flat, n_waters, M_water, BOLTZMANN_KCAL)
                records[label].append((t_trans, t_rot))
        if s == 1 or s % max(1, args.steps // 5) == 0:
            log.info("  step %d/%d", s, args.steps)

    summary = {}
    for label, vals in records.items():
        arr = np.array(vals)
        summary[label] = {
            "t_trans_mean": float(arr[:, 0].mean()),
            "t_trans_std": float(arr[:, 0].std()),
            "t_rot_mean": float(arr[:, 1].mean()),
            "t_rot_std": float(arr[:, 1].std()),
        }

    log.info("=== Summary (post burn-in, n=%d samples) ===", len(records["final"]))
    for label in ("pre_a1", "post_r1", "pre_a2", "post_r2", "final"):
        s = summary[label]
        log.info(
            "  %-8s  T_trans=%.2f±%.2f K   T_rot=%.2f±%.2f K   split=%.2f K",
            label, s["t_trans_mean"], s["t_trans_std"], s["t_rot_mean"], s["t_rot_std"],
            s["t_rot_mean"] - s["t_trans_mean"],
        )

    return {
        "n_waters": n_waters,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "gamma_ps": args.gamma,
        "dt_fs": args.dt_fs,
        "seed": seed,
        "summary": summary,
    }


def main() -> None:
    args = _parse_args()
    log.info("p5_r1_r2_reference_bias: n_waters=%d steps=%d gamma=%.4g dt_fs=%.2f seed=%d",
              args.n_waters, args.steps, args.gamma, args.dt_fs, args.seed)
    result = _run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("Result written to %s", out)


if __name__ == "__main__":
    main()
