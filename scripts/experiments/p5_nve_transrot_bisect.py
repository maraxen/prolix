"""Phase 5 NVE trans/rot bisect — mechanism A vs mechanism B discriminator.

Runs settle_langevin for --steps at --gamma (use 0 for the decisive NVE gate)
and decomposes final KE into translational and rotational temperature.

Mechanism A (RATTLE projection sink): at gamma=0 T_rot drifts DOWN deterministically.
Mechanism B (OU covariance mismatch): at gamma=0 T_trans ≈ T_rot (split appears only at gamma>0).

Usage:
    # L1 dry-run
    uv run python scripts/experiments/p5_nve_transrot_bisect.py --dry-run --out /dev/null

    # L2 smoke (8 waters, 100 steps)
    uv run python scripts/experiments/p5_nve_transrot_bisect.py \\
        --gamma 0 --n-waters 8 --steps 100 --out /tmp/p5_bisect_smoke.json --seed 42

    # Cluster gate (895 waters, 500 steps)
    uv run bth run python scripts/experiments/p5_nve_transrot_bisect.py \\
        --gamma 0 --n-waters 895 --steps 500 --out outputs/p5_bisect_nve.json \\
        --campaign cd2fa896
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path anchoring — never use cwd or relative strings (CLUSTER.md rule)
# ---------------------------------------------------------------------------

def _find_project_root(marker: str = "pyproject.toml") -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / marker).exists():
            return p
    raise RuntimeError(f"project root not found (looked for {marker!r})")


PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------------------
# Logging — io_callback for in-jit traces; logging module elsewhere
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("p5_nve_transrot_bisect")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5: NVE trans/rot temperature bisect (mechanism A vs B)"
    )
    p.add_argument("--gamma", type=float, default=10.0,
                   help="Langevin friction (ps^-1); 0 = NVE for decisive gate")
    p.add_argument("--n-waters", type=int, default=8,
                   help="Number of TIP3P water molecules")
    p.add_argument("--steps", type=int, default=500,
                   help="Number of integration steps")
    p.add_argument("--out", required=True,
                   help="Output JSON path (bth outcome evaluation)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate imports and fixture path, then exit 0")
    p.add_argument("--seed", type=int, default=42,
                   help="PRNG seed for JAX key and water subsampling")
    return p.parse_args()


def _validate_imports_and_fixture(n_waters: int) -> None:
    """Import everything and verify the water fixture is accessible."""
    log.info("Validating imports …")
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    import numpy as np  # noqa: F401

    from prolix.physics import pbc, settle, system  # noqa: F401
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL  # noqa: F401

    # Load fixture via same path as test_p2b_nvt_216water.py
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.physics.test_explicit_langevin_tip3p_parity import (  # noqa: F401
        _equil_water_positions,
        _proxide_params_pure_water,
    )
    from tests.physics.test_p2b_nvt_216water import (  # noqa: F401
        _make_tip3p_excl_indices,
        _dof_rigid_tip3p_waters,
    )

    log.info("Verifying water box fixture (load_water_box) …")
    from prolix.physics.solvation import load_water_box
    import numpy as np
    box = load_water_box()
    n_total = len(np.array(box.positions)) // 3
    if n_waters > n_total:
        raise RuntimeError(
            f"--n-waters {n_waters} exceeds fixture size {n_total}; "
            "tile the box or reduce --n-waters"
        )
    log.info("Fixture OK: %d waters available, requesting %d", n_total, n_waters)
    log.info("dry-run complete — all imports and fixture OK")


def _decompose_transrot(
    momentum,         # jnp array (n_atoms, 3)
    mass_flat,        # np/jnp array (n_atoms,)
    n_waters: int,
    M_water: float,
    boltzmann_kcal: float,
) -> tuple[float, float]:
    """Return (T_trans_K, T_rot_K) for rigid TIP3P waters.

    Matches the decomp() function in p5_transrot_decomp.py verbatim.
    Uses numpy for the final accumulation to avoid extra jit overhead.
    """
    import numpy as np
    import jax.numpy as jnp

    p3 = momentum.reshape(n_waters, 3, 3)          # (water, atom, xyz)

    # COM velocity per water: p_water_total / M_water
    v_com = (p3.sum(axis=1)) / M_water              # (N_waters, 3)

    ke_trans = float(0.5 * M_water * np.sum(np.asarray(v_com) ** 2))
    ke_tot   = float(jnp.sum(jnp.sum(momentum ** 2, axis=-1) / (2.0 * mass_flat)))
    ke_rot   = ke_tot - ke_trans

    dof_trans = 3 * n_waters - 3
    dof_rot   = 3 * n_waters

    t_trans = 2.0 * ke_trans / (dof_trans * boltzmann_kcal)
    t_rot   = 2.0 * ke_rot   / (dof_rot   * boltzmann_kcal)
    return float(t_trans), float(t_rot)


def _run(args: argparse.Namespace) -> dict:
    """Run settle_langevin for args.steps at args.gamma; return result dict."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from prolix.physics import pbc, settle, system
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.physics.test_explicit_langevin_tip3p_parity import (
        _equil_water_positions,
        _proxide_params_pure_water,
    )
    from tests.physics.test_p2b_nvt_216water import (
        _make_tip3p_excl_indices,
        _dof_rigid_tip3p_waters,
    )

    n_waters = args.n_waters
    seed     = args.seed

    log.info("Loading equilibrated TIP3P water box fixture (n_waters=%d, seed=%d) …",
             n_waters, seed)
    pos_a, box_edge = _equil_water_positions(n_waters, seed=seed)
    box_vec = jnp.array([box_edge] * 3, dtype=jnp.float64)

    dt_fs = 0.5
    dt    = dt_fs / AKMA_TIME_UNIT_FS
    kT    = 300.0 * BOLTZMANN_KCAL
    # gamma is in ps^-1; convert to AKMA units (ps^-1 → AKMA^-1)
    gamma = args.gamma * AKMA_TIME_UNIT_FS * 1e-3

    log.info("gamma=%.4g AKMA (input ps^-1=%.4g), dt=%.6g AKMA, kT=%.6g",
             gamma, args.gamma, dt, kT)

    # System parameters — verbatim from test_p2b_nvt_216water._run_nvt_scan
    sys_dict = {k: v for k, v in _proxide_params_pure_water(n_waters).items()
                if k != "exclusion_mask"}
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
    mass    = jnp.array([[15.999], [1.008], [1.008]] * n_waters, dtype=jnp.float64).reshape(n_atoms)
    wi      = settle.get_water_indices(0, n_waters)

    log.info("Building settle_langevin integrator …")
    init_s, apply_s = settle.settle_langevin(
        energy_fn, shift_fn,
        dt=dt, kT=kT, gamma=gamma,
        mass=mass, water_indices=wi,
        box=box_vec,
        remove_linear_com_momentum=False,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
        settle_velocity_iters=10,
    )

    log.info("Initialising state …")
    state = init_s(jax.random.key(seed), jnp.array(pos_a, dtype=jnp.float64), mass=mass)

    apply_jit = jax.jit(apply_s)

    log.info("Running %d steps …", args.steps)
    for s in range(1, args.steps + 1):
        state = apply_jit(state)
        if s == 1 or s % max(1, args.steps // 5) == 0:
            log.info("  step %d/%d", s, args.steps)

    log.info("Decomposing final KE into T_trans / T_rot …")
    mass_flat = mass.reshape(-1)
    M_water   = float(15.999 + 1.008 + 1.008)

    t_trans, t_rot = _decompose_transrot(
        state.momentum, mass_flat, n_waters, M_water, BOLTZMANN_KCAL
    )
    split_k = t_rot - t_trans

    log.info("T_trans = %.2f K  (3N-3 translational DOF)", t_trans)
    log.info("T_rot   = %.2f K  (3N   rotational DOF)",   t_rot)
    log.info("split_k = %.2f K  (T_rot - T_trans)",       split_k)

    return {
        "gamma":     args.gamma,
        "n_waters":  n_waters,
        "steps":     args.steps,
        "t_trans_k": t_trans,
        "t_rot_k":   t_rot,
        "split_k":   split_k,
    }


def main() -> None:
    args = _parse_args()

    log.info("p5_nve_transrot_bisect: gamma=%.4g  n_waters=%d  steps=%d  seed=%d",
             args.gamma, args.n_waters, args.steps, args.seed)

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
