"""Phase 5 dt-scaling — discretization artifact vs implementation bug discriminator.

Context (task 260805_p5_rottrans_mechanism_c): the residual T_rot/T_trans imbalance
under NVT has survived three ruled-out/deprioritized mechanisms (A: RATTLE sink,
ruled out via NVE bisect; B: OU noise covariance, already inertia-weighted in
production code, proven a no-op in June; C: R1/R2 AM-correction reference-point
bias, ruled out as dominant today at both n=16 and n=64 -- ~85% of the split is
already present before any single-cycle correction step runs).

OpenMM research (today) surfaced a fourth, literature-grounded candidate: Asthagiri
& Beck report that rigid-water rot/trans equipartition genuinely breaks down at
dt >= 1fs even in CORRECT integrators, worsening with dt -- i.e. some residual may
be an unavoidable finite-timestep discretization artifact, not an implementation
bug. This is directly testable: a discretization artifact should shrink toward
zero as dt -> 0; a code bug should persist at any dt.

This script runs settle_langevin (production code, unmodified) at several
decreasing dt values, holding n_waters/gamma/seed fixed, and reports the T_rot/
T_trans split at each. Uses n_waters=64 (>= the documented small-N finite-size
warm-bias threshold) to avoid confounding with the ALREADY-KNOWN, SEPARATE
translational finite-size artifact characterized in
.praxia/docs/research/260612_p5-dt1fs-size-crossover.md.

Usage:
    # L1 dry-run
    uv run python scripts/experiments/p5_dt_scaling_transrot.py --dry-run --out /dev/null

    # L2 smoke (small, fast)
    uv run python scripts/experiments/p5_dt_scaling_transrot.py \\
        --n-waters 16 --steps 100 --burn-in 20 --dt-fs-list 0.5,1.0 \\
        --out /tmp/p5_dt_scaling_smoke.json --seed 42

    # Cluster run (production scale)
    uv run bth run python scripts/experiments/p5_dt_scaling_transrot.py \\
        --n-waters 64 --steps 4000 --burn-in 500 --dt-fs-list 0.25,0.5,1.0 \\
        --out outputs/p5_dt_scaling_transrot.json --campaign <campaign-id>
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
log = logging.getLogger("p5_dt_scaling_transrot")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5: dt-scaling of T_rot/T_trans split (discretization artifact vs bug)"
    )
    p.add_argument("--n-waters", type=int, default=64)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--gamma", type=float, default=10.0, help="friction in ps^-1")
    p.add_argument("--dt-fs-list", type=str, default="0.25,0.5,1.0",
                   help="comma-separated dt values in fs, decreasing informativeness order")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _validate_imports_and_fixture(n_waters: int) -> None:
    log.info("Validating imports …")
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    import numpy as np  # noqa: F401

    from prolix.physics import pbc, settle, system  # noqa: F401
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL  # noqa: F401

    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.physics.test_explicit_langevin_tip3p_parity import (  # noqa: F401
        _equil_water_positions,
        _proxide_params_pure_water,
    )
    from tests.physics.test_p2b_nvt_216water import _make_tip3p_excl_indices  # noqa: F401

    log.info("Verifying water box fixture …")
    from prolix.physics.solvation import load_water_box
    import numpy as np
    box = load_water_box()
    n_total = len(np.array(box.positions)) // 3
    if n_waters > n_total:
        raise RuntimeError(f"--n-waters {n_waters} exceeds fixture size {n_total}")
    log.info("Fixture OK: %d waters available, requesting %d", n_total, n_waters)
    log.info("dry-run complete — all imports and fixture OK")


def _decompose_transrot(momentum, mass_flat, n_waters, M_water, boltzmann_kcal):
    """Matches p5_nve_transrot_bisect.py's decomp() verbatim."""
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


def _run_one_dt(n_waters, seed, gamma_ps, dt_fs, steps, burn_in):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from prolix.physics import pbc, settle, system
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.physics.test_explicit_langevin_tip3p_parity import _equil_water_positions, _proxide_params_pure_water
    from tests.physics.test_p2b_nvt_216water import _make_tip3p_excl_indices

    log.info("Loading equilibrated TIP3P water box fixture (n_waters=%d, seed=%d) …", n_waters, seed)
    pos_a, box_edge = _equil_water_positions(n_waters, seed=seed)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)

    dt = dt_fs / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = gamma_ps * AKMA_TIME_UNIT_FS * 1e-3

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
    wi = settle.get_water_indices(0, n_waters)

    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn, dt=dt, kT=kT, gamma=gamma, mass=mass, water_indices=wi,
        box=box_vec, remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True, projection_site="post_o",
        settle_velocity_iters=10,
    )
    state = init_s(jax.random.key(seed), jnp.array(pos_a, dtype=jnp.float64), mass=mass)
    apply_jit = jax.jit(apply_s)

    mass_flat = mass.reshape(-1)
    M_water = float(15.999 + 1.008 + 1.008)

    t_trans_samples, t_rot_samples = [], []
    for s in range(1, steps + 1):
        state = apply_jit(state)
        if s > burn_in:
            t_trans, t_rot = _decompose_transrot(state.momentum, mass_flat, n_waters, M_water, BOLTZMANN_KCAL)
            t_trans_samples.append(t_trans)
            t_rot_samples.append(t_rot)
        if s == 1 or s % max(1, steps // 5) == 0:
            log.info("  dt=%.2ffs step %d/%d", dt_fs, s, steps)

    t_trans_arr = np.array(t_trans_samples)
    t_rot_arr = np.array(t_rot_samples)
    return {
        "dt_fs": dt_fs,
        "n_samples": len(t_trans_arr),
        "t_trans_mean": float(t_trans_arr.mean()),
        "t_trans_std": float(t_trans_arr.std()),
        "t_rot_mean": float(t_rot_arr.mean()),
        "t_rot_std": float(t_rot_arr.std()),
        "split_k": float(t_rot_arr.mean() - t_trans_arr.mean()),
        "split_sem": float(np.sqrt(t_rot_arr.var() / len(t_rot_arr) + t_trans_arr.var() / len(t_trans_arr))),
    }


def _run(args: argparse.Namespace) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)

    dt_values = [float(x) for x in args.dt_fs_list.split(",")]
    log.info("Running dt-scaling sweep: dt_fs in %s (n_waters=%d, steps=%d, burn_in=%d, gamma=%.2f ps^-1)",
             dt_values, args.n_waters, args.steps, args.burn_in, args.gamma)

    per_dt = []
    for dt_fs in dt_values:
        log.info("=== dt=%.3f fs ===", dt_fs)
        result = _run_one_dt(args.n_waters, args.seed, args.gamma, dt_fs, args.steps, args.burn_in)
        log.info("  T_trans=%.2f±%.2f K  T_rot=%.2f±%.2f K  split=%.2f±%.2f K",
                  result["t_trans_mean"], result["t_trans_std"],
                  result["t_rot_mean"], result["t_rot_std"],
                  result["split_k"], result["split_sem"])
        per_dt.append(result)

    # Sort by dt ascending for slope computation
    per_dt_sorted = sorted(per_dt, key=lambda r: r["dt_fs"])
    dts = [r["dt_fs"] for r in per_dt_sorted]
    splits = [abs(r["split_k"]) for r in per_dt_sorted]

    # Ratio of |split| at largest dt vs smallest dt -- the key discriminator.
    # If discretization artifact: ratio >> 1 (split shrinks a lot as dt shrinks).
    # If implementation bug: ratio ~ 1 (split roughly constant regardless of dt).
    ratio_max_min = splits[-1] / max(splits[0], 1e-6)

    return {
        "n_waters": args.n_waters,
        "steps": args.steps,
        "burn_in": args.burn_in,
        "gamma_ps": args.gamma,
        "seed": args.seed,
        "n_dt_values": len(dts),
        "dt_min_fs": dts[0],
        "dt_max_fs": dts[-1],
        "abs_split_at_min_dt": splits[0],
        "abs_split_at_max_dt": splits[-1],
        "split_ratio_maxdt_over_mindt": ratio_max_min,
        "per_dt": per_dt_sorted,
    }


def main() -> None:
    args = _parse_args()
    log.info("p5_dt_scaling_transrot: n_waters=%d dt_fs_list=%s steps=%d seed=%d",
              args.n_waters, args.dt_fs_list, args.steps, args.seed)

    if args.dry_run:
        _validate_imports_and_fixture(args.n_waters)
        sys.exit(0)

    result = _run(args)
    out = Path(args.out)
    if str(out) != "/dev/null":
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("Result written to %s", out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
