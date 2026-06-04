"""Phase 5 — batched SETTLE divergence repro (platform_precision hypothesis).

Measures max absolute position diff between vmap-batched and loop-unbatched
settle_langevin trajectories. Under the platform_precision hypothesis,
`_settle_water_batch` PBC minimum-image at line ~237 (delta - box * round(delta/box))
is FMA-sensitive near ±0.5, so GPU-accelerated XLA can produce a different image than
CPU float64, displacing one constrained position by a full box-length.

Usage:
    # L1 dry-run
    uv run python scripts/experiments/p5_settle_batch_divergence.py --dry-run --out /dev/null

    # L2 smoke test (default: 2 waters, box 10 Å, 50 steps)
    uv run python scripts/experiments/p5_settle_batch_divergence.py \\
        --batch-size 2 --n-waters 2 --steps 50 --out /tmp/p5_batch_smoke.json

    # Cluster batch
    uv run bth run python scripts/experiments/p5_settle_batch_divergence.py \\
        --batch-size 2 --n-waters 2 --box-size 10.0 --steps 50 \\
        --out outputs/p5_batch_div.json --campaign cd2fa896
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path anchoring (CLUSTER.md: never use cwd / relative strings)
# ---------------------------------------------------------------------------

def _find_project_root(marker: str = "pyproject.toml") -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / marker).exists():
            return p
    raise RuntimeError(f"project root not found (looked for {marker!r})")


PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("p5_settle_batch_divergence")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5: batched SETTLE divergence repro (platform_precision)"
    )
    p.add_argument("--batch-size", type=int, default=2,
                   help="Number of replicas in the vmap batch")
    p.add_argument("--n-waters", type=int, default=2,
                   help="Number of TIP3P water molecules per replica")
    p.add_argument("--box-size", type=float, default=10.0,
                   help="Cubic box side length in Angstrom (0 = free space / no box)")
    p.add_argument("--steps", type=int, default=50,
                   help="Number of integration steps to run")
    p.add_argument("--seed", type=int, default=42,
                   help="PRNG seed")
    p.add_argument("--out", required=True,
                   help="Output JSON path")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate imports and exit 0 without running dynamics")
    return p.parse_args()


def _validate_imports() -> None:
    log.info("Validating imports …")
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    from prolix.physics import settle  # noqa: F401
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL  # noqa: F401
    log.info("dry-run complete — all imports OK")


def _build_water_system(n_waters: int, box_size: float, seed: int):
    """Build a minimal TIP3P-geometry water system with one dummy solute atom.

    Returns (positions, masses, water_indices, box_vec_or_None).
    """
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)

    # TIP3P geometry (Å)
    OH = 0.9572
    angle_rad = np.deg2rad(104.52 / 2)
    hx = OH * np.sin(angle_rad)
    hy = OH * np.cos(angle_rad)

    # Build n_waters molecules, scattered randomly in a box
    atom_positions = []
    for i in range(n_waters):
        if box_size > 0:
            origin = rng.uniform(1.0, box_size - 1.0, size=3)
        else:
            origin = rng.uniform(-5.0, 5.0, size=3)
        atom_positions.append(origin)                               # O
        atom_positions.append(origin + np.array([hx, -hy, 0.0]))   # H1
        atom_positions.append(origin + np.array([-hx, -hy, 0.0]))  # H2

    # Add one dummy solute atom in the middle of the box (or at origin)
    if box_size > 0:
        atom_positions.append(np.array([box_size / 2, box_size / 2, box_size / 2]))
    else:
        atom_positions.append(np.array([0.0, 0.0, 0.0]))

    positions = jnp.array(np.array(atom_positions, dtype=np.float64))

    # Masses
    masses_list = []
    for _ in range(n_waters):
        masses_list += [15.999, 1.008, 1.008]
    masses_list.append(12.0)  # solute C-like atom
    masses = jnp.array(masses_list, dtype=jnp.float64)

    # Water indices: atoms 3i, 3i+1, 3i+2 for water i
    water_indices = jnp.array(
        [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n_waters)],
        dtype=jnp.int32,
    )

    box_vec = (
        jnp.array([box_size, box_size, box_size], dtype=jnp.float64)
        if box_size > 0 else None
    )

    return positions, masses, water_indices, box_vec


def _make_energy_fn(positions_ref):
    """Harmonic restoring energy (keeps system near initial config)."""
    import jax.numpy as jnp

    k = 0.5  # kcal/(mol·Å²)

    def energy_fn(positions, box=None):
        return 0.5 * k * jnp.sum((positions - positions_ref) ** 2)

    return energy_fn


def _shift_fn(dR, box):
    import jax.numpy as jnp
    return dR - box * jnp.round(dR / box)


def _run_comparison(
    n_waters: int,
    box_size: float,
    steps: int,
    seed: int,
) -> tuple[float, float]:
    """Run vmap batch vs loop comparison.

    Returns (max_diff_with_box, max_diff_free_space).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from prolix.physics import settle
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    dt_fs = 0.5
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 / AKMA_TIME_UNIT_FS

    results = {}

    for label, box_s in [("with_box", box_size), ("free_space", 0.0)]:
        log.info("Running %s (box_size=%.1f) …", label, box_s)

        positions, masses, water_indices, box_vec = _build_water_system(n_waters, box_s, seed)
        energy_fn = _make_energy_fn(positions)

        init_fn, apply_fn = settle.settle_langevin(
            energy_fn,
            _shift_fn,
            dt=dt_akma,
            kT=kT,
            gamma=gamma,
            mass=masses,
            water_indices=water_indices,
            box=box_vec,
            project_ou_momentum_rigid=True,
            projection_site="post_o",
        )

        key = jax.random.key(seed)
        kw = {"box": box_vec} if box_vec is not None else {}
        ref_state = init_fn(key, positions, **kw)

        # Build B=2 batch by stacking the same initial state twice
        def _batch(s):
            return jax.tree.map(lambda x: jnp.stack([x, x], axis=0), s)

        batched_state = _batch(ref_state)

        def apply_fn_step(state):
            return apply_fn(state, **kw)

        vmapped_apply = jax.vmap(apply_fn_step)

        # Run loop and compare positions
        unbatched_curr = ref_state
        batched_curr = batched_state
        max_diff = 0.0

        for step in range(steps):
            unbatched_curr = apply_fn_step(unbatched_curr)
            batched_curr = vmapped_apply(batched_curr)

            # Compare unbatched positions to each replica in the batch
            for replica in range(2):
                diff = float(jnp.max(jnp.abs(unbatched_curr.positions - batched_curr.positions[replica])))
                if diff > max_diff:
                    max_diff = diff
                    if diff > 1e-5:
                        log.warning(
                            "  step=%d replica=%d diff=%.3e — divergence detected",
                            step, replica, diff,
                        )

        log.info("  %s max_diff = %.3e Å", label, max_diff)
        results[label] = max_diff

    return results["with_box"], results["free_space"]


def main() -> None:
    args = _parse_args()

    if args.dry_run:
        _validate_imports()
        if args.out != "/dev/null":
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"dry_run": True}))
        sys.exit(0)

    log.info(
        "Starting batch divergence repro: batch_size=%d n_waters=%d box_size=%.1f steps=%d seed=%d",
        args.batch_size, args.n_waters, args.box_size, args.steps, args.seed,
    )

    max_diff_with_box, max_diff_free_space = _run_comparison(
        n_waters=args.n_waters,
        box_size=args.box_size,
        steps=args.steps,
        seed=args.seed,
    )

    result = {
        "batch_size": args.batch_size,
        "n_waters": args.n_waters,
        "box_size": args.box_size,
        "steps": args.steps,
        "max_diff_with_box": max_diff_with_box,
        "max_diff_free_space": max_diff_free_space,
    }
    log.info("Result: %s", json.dumps(result))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Output written to %s", out_path)


if __name__ == "__main__":
    main()
