#!/usr/bin/env python3
"""Phase 4 redirect (task 260715_b1_physics_parity): does EnsemblePlan actually
share ONE compiled program across genuinely different real proteins, when they
are solvated (SETTLE+PME) and their shape_spec buckets are deliberately aligned?

Background: B1's original 4 shape classes (3 vacuum/GB proteins + 1 isolated
4-water bundle) never tested this -- each class is, by construction, a single
topology tiled to `replicas` copies of itself, never a genuine MIX of different
real systems. The step-scaling profiling (scripts/experiments/
profile_b1_step_scaling.py) found 1AKE (vacuum/GB) dominates per-step cost --
orthogonal to the PME+SETTLE+heterogeneous-batching question this project's
thesis actually rests on (see memory project_prolix_identity_and_thesis).

1VII (villin headpiece, 1963 real atoms) and 2GB1 (protein G B1 domain, 1987
real atoms) are two genuinely different real proteins that, once solvated with
the SAME target_box_size=[32,32,32] via solvate_protein_to_bundle
(src/prolix/physics/solvation.py), land in the IDENTICAL MolecularShapeSpec
across all 8 bucket axes (atom/bond/angle/dihedral/water/excl/cmap/exception) --
confirmed structurally via partition_bundles_by_shape returning 1 group for a
mixed [1vii, 2gb1, 1vii, 2gb1] list. This is NOT automatic for arbitrary
protein pairs -- 1VII vs 2JOF and 1VII vs 1CRN do NOT bucket-match (their
protein-topology-derived bond/angle/dihedral bucket indices differ even
though their water-dominated atom-count bucket matches).

This script measures whether that structural grouping translates into a real,
single shared JIT compile cost: --protein {1vii,2gb1} times ONE protein tiled
to `replicas` copies (control, matching profile_b1_step_scaling.py's
methodology); --mode combined times an EQUAL split of both proteins
interleaved. If combined's compile_fixed_s ~= either single-protein run's
compile_fixed_s (not their sum), that's direct evidence of one shared compile
across genuinely different real solvated proteins.

Debt 756 end-to-end validation (--mode three_way): 1UAO (Chignolin) does NOT
naturally bucket-match 1vii/2gb1 on the bond/angle/dihedral axes -- a
genuinely mismatched third protein. `target_bucket_counts` (built in Tasks
1-2 of this plan) forces all three into ONE shared MolecularShapeSpec,
proving the mechanism works end-to-end with real proteins/real solvation,
not just in the fast unit test
(tests/physics/test_flash_dense_parity.py::
test_solvate_protein_to_bundle_target_bucket_counts_forces_match). Note:
1UAO alone hits a pre-existing, unrelated CMAP-padding crash in
make_bundle_from_system (debt #1170) -- worked around uniformly for all
three proteins via `target_bucket_counts["cmap"] = 512`.

Usage::

    # L1 dry-run (shapes only, no GPU)
    uv run python scripts/experiments/profile_b1_heterogeneous_solvated_compile.py --dry-run --protein 1vii
    uv run python scripts/experiments/profile_b1_heterogeneous_solvated_compile.py --dry-run --mode three_way

    # L2 local CPU smoke
    JAX_PLATFORMS=cpu uv run python scripts/experiments/profile_b1_heterogeneous_solvated_compile.py \\
        --protein 1vii --replicas 2 --n-steps-list 2,5 --out /tmp/smoke.json
    JAX_PLATFORMS=cpu uv run python scripts/experiments/profile_b1_heterogeneous_solvated_compile.py \\
        --mode three_way --replicas 3 --n-steps-list 2,5 --out /tmp/smoke_three_way.json

    # L3 cluster (via bth run, see campaign 32d6574e)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_b1_hetero_solvated")

ROOT = Path(__file__).resolve().parents[2]

DT_FS = 0.5
KT_KCAL = 0.596
GAMMA_PS = 50.0

# Same box for every protein -- this is what makes bucket-matching possible
# (water atom count dominates the total, so a fixed box equalizes the
# dominant atom-count bucket; the protein-topology-derived buckets still need
# to independently coincide, which is why 1vii/2gb1 was found by trying
# several candidates, not assumed).
TARGET_BOX_SIZE = [32.0, 32.0, 32.0]
PADDING = 8.0

PROTEIN_PDBS = {
    "1vii": "1VII.pdb",
    "2gb1": "2GB1.pdb",
    "1uao": "1UAO.pdb",  # debt 756 validation: does NOT naturally bucket-match
                          # 1vii/2gb1 on bond/angle/dihedral axes; forced via
                          # target_bucket_counts below.
}


def _resolve_ff_path() -> str:
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from _b1_paramize import _resolve_ff_path as _resolve

    return _resolve("protein.ff19SB.xml")


def _load_and_solvate(protein_key: str, target_bucket_counts: dict | None = None):
    """Parse + solvate one protein into a MolecularBundle.

    ``target_bucket_counts`` (debt 756) is threaded straight through to
    ``solvate_protein_to_bundle`` -- passing None reproduces the pre-756
    behavior byte-for-byte (natural smallest-fit bucket selection).
    """
    import jax.numpy as jnp
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.solvation import solvate_protein_to_bundle
    from prolix.physics.water_models import WaterModelType

    ff_path = _resolve_ff_path()
    pdb_path = ROOT / "data" / "pdb" / PROTEIN_PDBS[protein_key]
    spec = OutputSpec(parameterize_md=True, force_field=ff_path, coord_format=CoordFormat.Full)
    protein = parse_structure(str(pdb_path), spec)
    return solvate_protein_to_bundle(
        protein,
        padding=PADDING,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array(TARGET_BOX_SIZE),
        target_bucket_counts=target_bucket_counts,
    )


def _build_bundles(mode: str, protein_key: str | None, replicas: int):
    if mode == "single":
        tmpl = _load_and_solvate(protein_key)
        return [tmpl] * replicas
    elif mode == "combined":
        b_1vii = _load_and_solvate("1vii")
        b_2gb1 = _load_and_solvate("2gb1")
        if b_1vii.shape_spec != b_2gb1.shape_spec:
            raise RuntimeError(
                f"1vii/2gb1 shape_spec mismatch -- bucket-matching assumption "
                f"broke (upstream data/code change?): {b_1vii.shape_spec} vs "
                f"{b_2gb1.shape_spec}"
            )
        half = replicas // 2
        bundles = [b_1vii] * half + [b_2gb1] * (replicas - half)
        return bundles
    elif mode == "three_way":
        # Debt 756 end-to-end validation: 1UAO (Chignolin, 138 real
        # pre-solvation atoms per proxide parameterization -- far smaller
        # than 1vii/2gb1's ~596) does NOT naturally bucket-match 1vii/2gb1.
        # Force all three into ONE shared MolecularShapeSpec via
        # target_bucket_counts, proving the mechanism built in Tasks 1-2
        # works end-to-end with real proteins/real solvation, not just in a
        # fast unit test (tests/physics/test_flash_dense_parity.py::
        # test_solvate_protein_to_bundle_target_bucket_counts_forces_match).
        from prolix.types.bundles import (
            ANGLE_BUCKETS,
            ATOM_BUCKETS,
            BOND_BUCKETS,
            DIHEDRAL_BUCKETS,
        )

        # Determine the shared target ceiling PER AXIS from the NATURAL
        # (unforced) shape_spec of all three proteins, taking the max bucket
        # index across all three for each axis. The brief anticipated only
        # bond/angle/dihedral would need forcing, reasoning that 1vii's own
        # natural bucket sizes would already be the ceiling (1uao's real
        # bond/angle/dihedral counts are far smaller than 1vii/2gb1's, given
        # its far smaller residue count) -- confirmed true empirically below
        # (max reduces to 1vii/2gb1's value on those three axes, a no-op vs.
        # reading from b_1vii alone).
        #
        # DEVIATION from the brief: the ATOM axis ALSO needs forcing, and in
        # the OPPOSITE direction (1uao is the ceiling, not 1vii/2gb1). This
        # module's docstring claims a fixed target_box_size equalizes the
        # atom bucket because water dominates the atom count -- true for the
        # 1vii/2gb1 pair it was calibrated on, but NOT for 1uao: its much
        # smaller protein volume leaves more room for solvent water in the
        # SAME box, so its total solvated atom count crosses an ATOM_BUCKETS
        # rung 1vii/2gb1 do not (empirically: 1vii/2gb1 land at bucket 2048,
        # 1uao lands at bucket 5000). Taking the per-axis MAX natural bucket
        # across all three proteins (rather than hardcoding to b_1vii) is
        # correct regardless of which protein happens to be the ceiling for
        # a given axis, and was verified necessary via the L1 dry-run before
        # landing this code (see task-3-report.md).
        # NOTE: even this "natural" (bond/angle/dihedral/atom-unforced) probe
        # load must pass cmap:512 for all three -- 1uao crashes in
        # make_bundle_from_system's CMAP padding (debt #1170) regardless of
        # whether any target_bucket_counts override is in play at all. The
        # cmap override doesn't touch the atom/bond/angle/dihedral axes being
        # read here, so the "natural" values inspected below are unaffected.
        _cmap_workaround = {"cmap": 512}
        b_1vii_natural = _load_and_solvate("1vii", _cmap_workaround)
        b_2gb1_natural = _load_and_solvate("2gb1", _cmap_workaround)
        b_1uao_natural = _load_and_solvate("1uao", _cmap_workaround)
        naturals = (b_1vii_natural, b_2gb1_natural, b_1uao_natural)
        target_bucket_counts = {
            "atom": ATOM_BUCKETS[max(b.shape_spec.atom_bucket_idx for b in naturals)],
            "bond": BOND_BUCKETS[max(b.shape_spec.bond_bucket_idx for b in naturals)],
            "angle": ANGLE_BUCKETS[max(b.shape_spec.angle_bucket_idx for b in naturals)],
            "dihedral": DIHEDRAL_BUCKETS[
                max(b.shape_spec.dihedral_bucket_idx for b in naturals)
            ],
            # cmap:512 works around a pre-existing, unrelated crash in
            # make_bundle_from_system's CMAP padding -- confirmed present for
            # 1UAO specifically (ValueError: index can't contain negative
            # values) even before target_bucket_counts existed at all. See
            # debt #1170. Included uniformly for all three proteins here
            # (1vii/2gb1 don't need it -- their natural cmap_bucket_idx
            # already matches) rather than only for 1uao -- simpler, and
            # inert/safe wherever the real cmap count doesn't require it.
            "cmap": 512,
        }

        # Apply the SAME dict uniformly to all three loads (including 1vii,
        # whose natural bucket sizes are what defined most of the dict in
        # the first place -- this re-load should be a no-op for 1vii's own
        # bond/angle/dihedral/cmap axes but exercises the override code path
        # identically across all three proteins, and is NOT a no-op for
        # 1vii/2gb1's atom axis, which gets forced up from bucket 2048 to
        # 5000 to match 1uao's larger natural requirement).
        b_1vii = _load_and_solvate("1vii", target_bucket_counts)
        b_2gb1 = _load_and_solvate("2gb1", target_bucket_counts)
        b_1uao = _load_and_solvate("1uao", target_bucket_counts)

        if not (b_1vii.shape_spec == b_2gb1.shape_spec == b_1uao.shape_spec):
            raise RuntimeError(
                f"three_way bucket-forcing failed -- 1vii/2gb1/1uao "
                f"shape_spec mismatch even after target_bucket_counts "
                f"override {target_bucket_counts}: "
                f"1vii={b_1vii.shape_spec} 2gb1={b_2gb1.shape_spec} "
                f"1uao={b_1uao.shape_spec}"
            )
        log.info(
            "three_way: forced shape_spec match confirmed for 1vii/2gb1/1uao "
            "via target_bucket_counts=%s",
            target_bucket_counts,
        )
        third = replicas // 3
        bundles = (
            [b_1vii] * third
            + [b_2gb1] * third
            + [b_1uao] * (replicas - 2 * third)
        )
        return bundles
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile-sharing test: heterogeneous real solvated proteins"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("single", "combined", "three_way"), default="single")
    parser.add_argument("--protein", choices=("1vii", "2gb1", "1uao"), default=None,
                         help="Required for --mode single, ignored for --mode combined/three_way")
    parser.add_argument("--n-steps-list", default="2,10,50,200,800")
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--use-neighbor-list", action="store_true",
        help=(
            "Exercise EnsemblePlan.run(use_neighbor_list=True) instead of the dense "
            "path. Originally (Phase 6 steps 4-5) this only worked for a single "
            "solvated protein (_run_single_inference); debt 802 extended "
            "use_neighbor_list=True to _run_stacked_dispatch (the vmapped "
            "multi-bundle path, commit 3950f4c), so this now also works with "
            "--mode combined -- the actual cross-protein compile-sharing question "
            "this script measures."
        ),
    )
    parser.add_argument("--nl-update-every", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "single" and args.protein is None:
        parser.error("--protein is required when --mode single")

    import jax

    log.info("jax.config.x64_enabled = %s", jax.config.x64_enabled)
    log.info("Mode=%s protein=%s replicas=%d", args.mode, args.protein, args.replicas)

    try:
        bundles = _build_bundles(args.mode, args.protein, args.replicas)
    except Exception as e:
        log.error("Bundle construction failed: %s", e)
        return 1

    n_real = int(bundles[0].n_atoms)
    log.info(
        "Built %d bundles, n_real_atoms[0]=%d, shape_spec=%s",
        len(bundles), n_real, bundles[0].shape_spec,
    )

    protein_label = args.protein
    if args.mode == "combined":
        protein_label = "combined-1vii-2gb1"
    elif args.mode == "three_way":
        protein_label = "three_way-1vii-2gb1-1uao"

    # ATOM_BUCKETS is imported here (not module-level) because its ladder
    # membership can change across commits (debt 773 inserted a 2048 rung
    # between 1024 and 5000, 2026-08-05) -- emitting the resolved bucket
    # *size* alongside the raw index means catalog rows recorded before and
    # after such a ladder change remain distinguishable, rather than only
    # recording an index whose real-world meaning silently shifted.
    from prolix.types.bundles import ATOM_BUCKETS

    bundle_info = {
        "n_bundles": len(bundles),
        "n_real_atoms_first": n_real,
        "shape_spec_atom_bucket_idx": bundles[0].shape_spec.atom_bucket_idx,
        "shape_spec_atom_bucket_size": ATOM_BUCKETS[bundles[0].shape_spec.atom_bucket_idx],
    }
    if args.mode == "three_way":
        # Explicit shape_spec match confirmation for all three proteins --
        # _build_bundles already raises RuntimeError if this doesn't hold, so
        # reaching here means it's True; surfaced explicitly in the JSON
        # output (dry-run and full run) as direct evidence, not just an
        # absence of a raised exception.
        bundle_info["shape_spec_bond_bucket_idx"] = bundles[0].shape_spec.bond_bucket_idx
        bundle_info["shape_spec_angle_bucket_idx"] = bundles[0].shape_spec.angle_bucket_idx
        bundle_info["shape_spec_dihedral_bucket_idx"] = bundles[0].shape_spec.dihedral_bucket_idx
        bundle_info["three_way_shape_spec_match"] = all(
            b.shape_spec == bundles[0].shape_spec for b in bundles
        )

    if args.dry_run:
        summary = {"mode": "dry-run", "protein": protein_label, **bundle_info}
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2))
        return 0

    from prolix.api import EnsemblePlan
    from prolix.api.ensemble_dedup import partition_bundles_by_shape

    n_groups = len(partition_bundles_by_shape(bundles))
    log.info("partition_bundles_by_shape -> %d group(s)", n_groups)

    def _block_trajs(trajectories) -> None:
        if not isinstance(trajectories, list):
            trajectories = [trajectories]
        for traj in trajectories:
            jax.block_until_ready(traj.positions)

    run_kwargs = {}
    if args.use_neighbor_list:
        run_kwargs = {
            "use_neighbor_list": True,
            "nl_update_every": args.nl_update_every,
        }

    n_steps_list = [int(x) for x in args.n_steps_list.split(",")]
    results = []
    for n_steps in n_steps_list:
        try:
            plan = EnsemblePlan.from_bundles(bundles)

            t_first0 = time.perf_counter()
            first = plan.run(n_steps=1, dt=DT_FS, kT=KT_KCAL, seed=0, gamma=GAMMA_PS, run_mode="inference", **run_kwargs)
            _block_trajs(first)
            t_first = time.perf_counter() - t_first0
            del first

            t_ss = 0.0
            if n_steps > 1:
                t_ss0 = time.perf_counter()
                last = plan.run(n_steps=n_steps - 1, dt=DT_FS, kT=KT_KCAL, seed=1, gamma=GAMMA_PS, run_mode="inference", **run_kwargs)
                _block_trajs(last)
                t_ss = time.perf_counter() - t_ss0
                del last

            results.append({"n_steps": n_steps, "t_first_step": t_first, "t_steady_state": t_ss})
            log.info("n_steps=%d: t_first=%.4f s, t_steady_state=%.4f s", n_steps, t_first, t_ss)
        except Exception as e:
            log.error("Failed to run n_steps=%d: %s", n_steps, e)
            return 1

    x_vals = [r["n_steps"] - 1 for r in results if r["n_steps"] > 1]
    y_vals = [r["t_steady_state"] for r in results if r["n_steps"] > 1]
    slope = intercept = r_squared = float("nan")
    if len(x_vals) >= 2:
        x_arr, y_arr = np.array(x_vals, dtype=np.float64), np.array(y_vals, dtype=np.float64)
        coeffs = np.polyfit(x_arr, y_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        y_pred = slope * x_arr + intercept
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        log.info("Regression: per_step=%.6e s, compile_fixed=%.6e s, R^2=%.6f", slope, intercept, r_squared)

    summary = {
        "mode": args.mode,
        "protein": protein_label,
        "replicas": args.replicas,
        "n_groups_from_partition": n_groups,
        "per_step_corrected_s": slope,
        "compile_fixed_s": intercept,
        "r_squared": r_squared,
        **bundle_info,
        "raw_results": results,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log.info("Wrote summary to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
