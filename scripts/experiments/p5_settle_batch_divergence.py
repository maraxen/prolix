"""Phase 5 — A/B PBC-rounding verification with constructed adversarial case (Stage 1).

Compares loop-unbatched vs vmap-batched SETTLE trajectories under floor(x+0.5)
(fixed, Arm B) vs round(x) (pre-fix, Arm A) PBC-rounding idioms. Tests whether the
primitive error at the PBC-rounding sites is truly eliminated by the floor() fix.

Evidence rule:
  - G1 (constructed adversarial case, targets L773/776): passes under TOL in Arm B
  - G2 (>=5 random seeds): all pass under TOL in Arm B
  - Both are necessary and sufficient for GREEN

Metrics:
  - max_abs_atom_position_delta: max over atoms i, components c of |x_loop[i,c] - x_vmap[i,c]|
  - first_divergence_step: smallest step index exceeding TOL = 1e-10 Å

Usage:
    # Local smoke test (50 steps, small system)
    uv run python scripts/experiments/p5_settle_batch_divergence.py \\
        --mode constructed --pbc-round-mode floor --steps 50 --out /tmp/g1_arm_b.json

    # Cluster batch with bth
    uv run bth run python scripts/experiments/p5_settle_batch_divergence.py \\
        --mode constructed --pbc-round-mode floor --steps 20000 \\
        --out outputs/p5c_g1_arm_b.json --campaign <id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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

# Tolerance for detecting divergence (Angstrom)
TOL = 1e-10


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5 Stage 1: A/B PBC-rounding verification (constructed + seeds)"
    )
    p.add_argument(
        "--mode",
        choices=["constructed", "seeds"],
        required=True,
        help="Constructed adversarial case (G1) or random seeded comparison (G2)",
    )
    p.add_argument(
        "--pbc-round-mode",
        choices=["floor", "round"],
        default="floor",
        help="PBC rounding idiom: floor(x+0.5) [fixed] or round(x) [pre-fix]",
    )
    p.add_argument(
        "--n-waters",
        type=int,
        default=64,
        help="Number of TIP3P water molecules per replica",
    )
    p.add_argument(
        "--box-size",
        type=float,
        default=40.0,
        help="Cubic box side length in Angstrom",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="Number of integration steps to run",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed (for seeded mode or constructed case reproducibility)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output JSON path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports and exit 0 without running dynamics",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Initialization & setup
# ---------------------------------------------------------------------------


def _validate_imports() -> None:
    log.info("Validating imports …")
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    from prolix.physics import settle  # noqa: F401
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL  # noqa: F401
    log.info("dry-run complete — all imports OK")


def _build_water_system(n_waters: int, box_size: float, seed: int):
    """Build a random TIP3P-geometry water system with one dummy solute atom.

    Returns (positions, masses, water_indices, box_vec).
    """
    rng = np.random.default_rng(seed)

    # TIP3P geometry (Å)
    OH = 0.9572
    angle_rad = np.deg2rad(104.52 / 2)
    hx = OH * np.sin(angle_rad)
    hy = OH * np.cos(angle_rad)

    # Build n_waters molecules, scattered randomly in a box
    atom_positions = []
    for i in range(n_waters):
        origin = rng.uniform(1.0, box_size - 1.0, size=3)
        atom_positions.append(origin)                               # O
        atom_positions.append(origin + np.array([hx, -hy, 0.0]))   # H1
        atom_positions.append(origin + np.array([-hx, -hy, 0.0]))  # H2

    # Add one dummy solute atom in the middle of the box
    atom_positions.append(np.array([box_size / 2, box_size / 2, box_size / 2]))

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

    box_vec = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)

    return positions, masses, water_indices, box_vec


def _build_constructed_water_system(box_size: float, seed: int):
    """Build adversarial case: water straddling a PBC face targeting L773/L776.

    Oxygen positioned just inside the +x face; hydrogens positioned to create
    a bond vector component dH_x / box near the half-integer 0.5 tie point.

    Returns (positions, masses, water_indices, box_vec).
    """
    rng = np.random.default_rng(seed)

    # TIP3P geometry
    OH = 0.9572
    angle_rad = np.deg2rad(104.52 / 2)
    hx = OH * np.sin(angle_rad)
    hy = OH * np.cos(angle_rad)

    # Construct a single water molecule that straddles the x-face.
    # O positioned just inside the +x face (at x = box - epsilon).
    # H1 and H2 positioned on the "other side" so that shift_fn wraps them
    # to the opposite face, creating a bond vector dH near the half-box tie point.

    epsilon = 0.01  # distance from face
    ox = box_size - epsilon
    oy = box_size / 2
    oz = box_size / 2

    # H1 and H2: position them symmetrically on the "other side" in the unwrapped frame,
    # differentiated in the y-direction to create a proper V-shaped water geometry.
    # After independent shift_fn wrapping per atom, they'll end up ~box away from O,
    # creating a bond vector dH_x ≈ box/2 which hits the tie point.
    h1_x = epsilon  # wraps back to ~box - epsilon after shift_fn
    h1_y = oy - hy  # H1 offset below oxygen
    h1_z = oz

    h2_x = epsilon
    h2_y = oy + hy  # H2 offset above oxygen (symmetric but opposite to H1)
    h2_z = oz

    # Add one dummy solute in the center
    atom_positions = [
        np.array([ox, oy, oz], dtype=np.float64),      # O
        np.array([h1_x, h1_y, h1_z], dtype=np.float64),  # H1
        np.array([h2_x, h2_y, h2_z], dtype=np.float64),  # H2
        np.array([box_size / 2, box_size / 2, box_size / 2], dtype=np.float64),  # solute
    ]

    positions = jnp.array(np.array(atom_positions, dtype=np.float64))

    # Masses: 1 water + 1 solute
    masses = jnp.array([15.999, 1.008, 1.008, 12.0], dtype=jnp.float64)

    # Water indices: only 1 water, atoms [0, 1, 2]
    water_indices = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    box_vec = jnp.array([box_size, box_size, box_size], dtype=jnp.float64)

    return positions, masses, water_indices, box_vec


def _make_energy_fn(positions_ref):
    """Harmonic restoring energy (keeps system near initial config)."""
    k = 0.5  # kcal/(mol·Å²)

    def energy_fn(positions, box=None):
        return 0.5 * k * jnp.sum((positions - positions_ref) ** 2)

    return energy_fn


def _shift_fn(dR, box):
    """PBC minimum-image shift (using round, not the instrumented _PBC_ROUND_FN)."""
    return dR - box * jnp.round(dR / box)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _run_comparison(
    n_waters: int,
    box_size: float,
    steps: int,
    seed: int,
    pbc_round_mode: str,
) -> dict:
    """Run vmap batch vs loop comparison, return metrics and first-divergence info.

    Returns dict with:
      - max_abs_atom_position_delta (float)
      - first_divergence_step (int or None)
      - offending_molecule_index (int or None)
      - o_pos_loop_pre (list or None)
      - o_pos_vmap_pre (list or None)
      - o_pos_loop_post (list or None)
      - o_pos_vmap_post (list or None)
      - delta_over_box_at_five_sites (dict or None)
      - primitive_error_classification (str or None)
      - batch_size (int)
      - n_waters (int)
      - box_size (float)
      - steps (int)
    """
    import os
    os.environ["PROLIX_PBC_ROUND_MODE"] = pbc_round_mode

    # Force reimport of settle to pick up the new env var
    import importlib
    import prolix.physics.settle
    importlib.reload(prolix.physics.settle)

    from prolix.physics import settle
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    jax.config.update("jax_enable_x64", True)

    dt_fs = 0.5
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 / AKMA_TIME_UNIT_FS

    log.info("Running comparison: n_waters=%d box=%.1f steps=%d pbc_mode=%s",
             n_waters, box_size, steps, pbc_round_mode)

    positions, masses, water_indices, box_vec = _build_water_system(n_waters, box_size, seed)
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
    ref_state = init_fn(key, positions, box=box_vec)

    # Build B=2 batch by stacking the same initial state twice
    def _batch(s):
        return jax.tree.map(lambda x: jnp.stack([x, x], axis=0), s)

    batched_state = _batch(ref_state)

    def apply_fn_step(state):
        return apply_fn(state, box=box_vec)

    vmapped_apply = jax.vmap(apply_fn_step)

    # Run loop and compare positions
    unbatched_curr = ref_state
    batched_curr = batched_state
    max_diff = 0.0
    first_divergence_step = None
    offending_molecule_index = None
    o_pos_loop_pre = None
    o_pos_vmap_pre = None
    o_pos_loop_post = None
    o_pos_vmap_post = None
    delta_over_box_at_five_sites = None
    primitive_error_classification = None

    for step in range(steps):
        unbatched_curr = apply_fn_step(unbatched_curr)
        batched_curr = vmapped_apply(batched_curr)

        # Compare positions per atom
        pos_loop = unbatched_curr.positions
        pos_vmap = batched_curr.positions[0]  # use first replica

        diff = jnp.abs(pos_loop - pos_vmap)
        max_diff_step = float(jnp.max(diff))

        if max_diff_step > max_diff:
            max_diff = max_diff_step

        if max_diff > TOL and first_divergence_step is None:
            first_divergence_step = step

            # Find offending molecule and dump state
            max_per_atom = jnp.max(diff, axis=1)
            offending_atom = int(jnp.argmax(max_per_atom))
            offending_molecule_index = offending_atom // 3

            # Gather pre/post positions for the offending water (O, H1, H2)
            water_idx = offending_molecule_index
            o_idx = 3 * water_idx
            h1_idx = 3 * water_idx + 1
            h2_idx = 3 * water_idx + 2

            o_pos_loop_pre = [float(x) for x in pos_loop[o_idx]]
            o_pos_vmap_pre = [float(x) for x in pos_vmap[o_idx]]
            h1_pos_loop_pre = [float(x) for x in pos_loop[h1_idx]]
            h1_pos_vmap_pre = [float(x) for x in pos_vmap[h1_idx]]
            h2_pos_loop_pre = [float(x) for x in pos_loop[h2_idx]]
            h2_pos_vmap_pre = [float(x) for x in pos_vmap[h2_idx]]

            # Note: we could compute post-step positions by calling settle_positions
            # directly, but for simplicity we just record the current post-step values
            o_pos_loop_post = [float(x) for x in unbatched_curr.positions[o_idx]]
            o_pos_vmap_post = [float(x) for x in batched_curr.positions[0][o_idx]]
            h1_pos_loop_post = [float(x) for x in unbatched_curr.positions[h1_idx]]
            h1_pos_vmap_post = [float(x) for x in batched_curr.positions[0][h1_idx]]
            h2_pos_loop_post = [float(x) for x in unbatched_curr.positions[h2_idx]]
            h2_pos_vmap_post = [float(x) for x in batched_curr.positions[0][h2_idx]]

            # Compute delta/box at the two recoverable sites (L773/L776: dH bond vectors).
            # Other sites (L171/L318/L1276/L1328) require internal settle instrumentation.
            box = float(box_vec[0]) if box_vec is not None else 1.0
            dh1_loop = np.array(h1_pos_loop_post) - np.array(o_pos_loop_post)
            dh1_vmap = np.array(h1_pos_vmap_post) - np.array(o_pos_vmap_post)
            dh2_loop = np.array(h2_pos_loop_post) - np.array(o_pos_loop_post)
            dh2_vmap = np.array(h2_pos_vmap_post) - np.array(o_pos_vmap_post)
            delta_over_box_at_five_sites = {
                "L171_delta_o_over_box": "not_recoverable_without_settle_instrumentation",
                "L318_delta_over_box": "not_recoverable_without_settle_instrumentation",
                "L773_dH1_over_box": {
                    "loop": [float(x / box) for x in dh1_loop],
                    "vmap": [float(x / box) for x in dh1_vmap],
                },
                "L776_dH2_over_box": {
                    "loop": [float(x / box) for x in dh2_loop],
                    "vmap": [float(x / box) for x in dh2_vmap],
                },
                "L1276_dx_1_over_box": "not_recoverable_without_settle_instrumentation",
                "L1328_dx_2_over_box": "not_recoverable_without_settle_instrumentation",
            }

            # Classify primitive error magnitude
            max_err = float(jnp.max(diff))
            if max_err > 1.0:
                primitive_error_classification = "box_quantised"
            elif max_err > 1e-8:
                primitive_error_classification = "large_arbitrary"
            else:
                primitive_error_classification = "small_arbitrary"

    log.info("  max_diff = %.3e Å (step %s)", max_diff, first_divergence_step)

    result = {
        "max_abs_atom_position_delta": float(max_diff),
        "first_divergence_step": first_divergence_step,
        "offending_molecule_index": offending_molecule_index,
        "o_pos_loop_pre": o_pos_loop_pre,
        "o_pos_vmap_pre": o_pos_vmap_pre,
        "o_pos_loop_post": o_pos_loop_post,
        "o_pos_vmap_post": o_pos_vmap_post,
        "h1_pos_loop_pre": None if first_divergence_step is None else h1_pos_loop_pre,
        "h1_pos_vmap_pre": None if first_divergence_step is None else h1_pos_vmap_pre,
        "h1_pos_loop_post": None if first_divergence_step is None else h1_pos_loop_post,
        "h1_pos_vmap_post": None if first_divergence_step is None else h1_pos_vmap_post,
        "h2_pos_loop_pre": None if first_divergence_step is None else h2_pos_loop_pre,
        "h2_pos_vmap_pre": None if first_divergence_step is None else h2_pos_vmap_pre,
        "h2_pos_loop_post": None if first_divergence_step is None else h2_pos_loop_post,
        "h2_pos_vmap_post": None if first_divergence_step is None else h2_pos_vmap_post,
        "delta_over_box_at_five_sites": delta_over_box_at_five_sites,
        "primitive_error_classification": primitive_error_classification,
        "batch_size": 2,
        "n_waters": n_waters,
        "box_size": float(box_size),
        "steps": steps,
    }

    return result


def _run_constructed_case(
    box_size: float,
    steps: int,
    seed: int,
    pbc_round_mode: str,
) -> dict:
    """Run the constructed adversarial case (G1).

    Returns the same dict as _run_comparison.
    """
    import os
    os.environ["PROLIX_PBC_ROUND_MODE"] = pbc_round_mode

    import importlib
    import prolix.physics.settle
    importlib.reload(prolix.physics.settle)

    from prolix.physics import settle
    from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

    jax.config.update("jax_enable_x64", True)

    dt_fs = 0.5
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 0.0  # Deterministic for constructed case

    log.info("Running constructed adversarial case: box=%.1f steps=%d pbc_mode=%s",
             box_size, steps, pbc_round_mode)

    positions, masses, water_indices, box_vec = _build_constructed_water_system(box_size, seed)
    energy_fn = _make_energy_fn(positions)

    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        _shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma,  # No thermostat noise
        mass=masses,
        water_indices=water_indices,
        box=box_vec,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    key = jax.random.key(seed)
    ref_state = init_fn(key, positions, box=box_vec)

    # Build B=2 batch by stacking
    def _batch(s):
        return jax.tree.map(lambda x: jnp.stack([x, x], axis=0), s)

    batched_state = _batch(ref_state)

    def apply_fn_step(state):
        return apply_fn(state, box=box_vec)

    vmapped_apply = jax.vmap(apply_fn_step)

    unbatched_curr = ref_state
    batched_curr = batched_state
    max_diff = 0.0
    first_divergence_step = None
    offending_molecule_index = None
    o_pos_loop_pre = None
    o_pos_vmap_pre = None
    o_pos_loop_post = None
    o_pos_vmap_post = None
    delta_over_box_at_five_sites = None
    primitive_error_classification = None

    for step in range(steps):
        unbatched_curr = apply_fn_step(unbatched_curr)
        batched_curr = vmapped_apply(batched_curr)

        pos_loop = unbatched_curr.positions
        pos_vmap = batched_curr.positions[0]

        diff = jnp.abs(pos_loop - pos_vmap)
        max_diff_step = float(jnp.max(diff))

        if max_diff_step > max_diff:
            max_diff = max_diff_step

        if max_diff > TOL and first_divergence_step is None:
            first_divergence_step = step
            offending_molecule_index = 0  # Only 1 water in constructed case

            o_idx = 0
            h1_idx = 1
            h2_idx = 2

            o_pos_loop_pre = [float(x) for x in pos_loop[o_idx]]
            o_pos_vmap_pre = [float(x) for x in pos_vmap[o_idx]]
            h1_pos_loop_pre = [float(x) for x in pos_loop[h1_idx]]
            h1_pos_vmap_pre = [float(x) for x in pos_vmap[h1_idx]]
            h2_pos_loop_pre = [float(x) for x in pos_loop[h2_idx]]
            h2_pos_vmap_pre = [float(x) for x in pos_vmap[h2_idx]]

            o_pos_loop_post = [float(x) for x in unbatched_curr.positions[o_idx]]
            o_pos_vmap_post = [float(x) for x in batched_curr.positions[0][o_idx]]
            h1_pos_loop_post = [float(x) for x in unbatched_curr.positions[h1_idx]]
            h1_pos_vmap_post = [float(x) for x in batched_curr.positions[0][h1_idx]]
            h2_pos_loop_post = [float(x) for x in unbatched_curr.positions[h2_idx]]
            h2_pos_vmap_post = [float(x) for x in batched_curr.positions[0][h2_idx]]

            # Compute delta/box at the two recoverable sites (L773/L776: dH bond vectors).
            box = float(box_vec[0])
            dh1_loop = np.array(h1_pos_loop_post) - np.array(o_pos_loop_post)
            dh1_vmap = np.array(h1_pos_vmap_post) - np.array(o_pos_vmap_post)
            dh2_loop = np.array(h2_pos_loop_post) - np.array(o_pos_loop_post)
            dh2_vmap = np.array(h2_pos_vmap_post) - np.array(o_pos_vmap_post)
            delta_over_box_at_five_sites = {
                "L171_delta_o_over_box": "not_recoverable_without_settle_instrumentation",
                "L318_delta_over_box": "not_recoverable_without_settle_instrumentation",
                "L773_dH1_over_box": {
                    "loop": [float(x / box) for x in dh1_loop],
                    "vmap": [float(x / box) for x in dh1_vmap],
                },
                "L776_dH2_over_box": {
                    "loop": [float(x / box) for x in dh2_loop],
                    "vmap": [float(x / box) for x in dh2_vmap],
                },
                "L1276_dx_1_over_box": "not_recoverable_without_settle_instrumentation",
                "L1328_dx_2_over_box": "not_recoverable_without_settle_instrumentation",
            }

            max_err = float(jnp.max(diff))
            if max_err > 1.0:
                primitive_error_classification = "box_quantised"
            elif max_err > 1e-8:
                primitive_error_classification = "large_arbitrary"
            else:
                primitive_error_classification = "small_arbitrary"

    log.info("  constructed case max_diff = %.3e Å (step %s)", max_diff, first_divergence_step)

    result = {
        "max_abs_atom_position_delta": float(max_diff),
        "first_divergence_step": first_divergence_step,
        "offending_molecule_index": offending_molecule_index,
        "o_pos_loop_pre": o_pos_loop_pre,
        "o_pos_vmap_pre": o_pos_vmap_pre,
        "o_pos_loop_post": o_pos_loop_post,
        "o_pos_vmap_post": o_pos_vmap_post,
        "h1_pos_loop_pre": None if first_divergence_step is None else h1_pos_loop_pre,
        "h1_pos_vmap_pre": None if first_divergence_step is None else h1_pos_vmap_pre,
        "h1_pos_loop_post": None if first_divergence_step is None else h1_pos_loop_post,
        "h1_pos_vmap_post": None if first_divergence_step is None else h1_pos_vmap_post,
        "h2_pos_loop_pre": None if first_divergence_step is None else h2_pos_loop_pre,
        "h2_pos_vmap_pre": None if first_divergence_step is None else h2_pos_vmap_pre,
        "h2_pos_loop_post": None if first_divergence_step is None else h2_pos_loop_post,
        "h2_pos_vmap_post": None if first_divergence_step is None else h2_pos_vmap_post,
        "delta_over_box_at_five_sites": delta_over_box_at_five_sites,
        "primitive_error_classification": primitive_error_classification,
        "batch_size": 2,
        "n_waters": 1,
        "box_size": float(box_size),
        "steps": steps,
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.dry_run:
        _validate_imports()
        if args.out != "/dev/null":
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"dry_run": True}))
        sys.exit(0)

    log.info("Starting P5c Stage 1 A/B test: mode=%s pbc_round=%s steps=%d",
             args.mode, args.pbc_round_mode, args.steps)

    if args.mode == "constructed":
        result = _run_constructed_case(
            box_size=args.box_size,
            steps=args.steps,
            seed=args.seed,
            pbc_round_mode=args.pbc_round_mode,
        )
    else:  # seeded
        result = _run_comparison(
            n_waters=args.n_waters,
            box_size=args.box_size,
            steps=args.steps,
            seed=args.seed,
            pbc_round_mode=args.pbc_round_mode,
        )

    log.info("Result: %s", json.dumps(result, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Output written to %s", out_path)


if __name__ == "__main__":
    main()
