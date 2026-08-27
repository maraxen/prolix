#!/usr/bin/env python3
"""Discovery: 1vii flash / PME-on @ 1.0 Å host timer vs tile size × remat.

Ungated. Same bonded+flash composition as single_padded_force(use_flash=True).
Command buffers stay on (do not set empty --xla_gpu_enable_command_buffer=).
No profiler traces. OOM is recorded as status=oom with exit 0.

  PYTHONPATH=. uv run python scripts/explore/prof_flash_tile_checkpoint.py \\
      --tile 256 --remat on --out out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prof_flash_tile_checkpoint")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _git_sha() -> str | None:
    env = os.environ.get("PROLIX_GIT_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "resource_exhausted",
        "out of memory",
        "oom",
        "cudaerror",
        "failed to allocate",
        "insufficient memory",
    )
    return any(n in text for n in needles)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protein", choices=("1vii", "2gb1"), default="1vii")
    p.add_argument("--pme", choices=("on", "off"), default="on")
    p.add_argument("--grid-spacing", type=float, default=1.0)
    p.add_argument("--tile", type=int, required=True)
    p.add_argument("--remat", choices=("on", "off"), required=True)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-inner", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--use-size-buckets",
        choices=("on", "off"),
        default="on",
        help="on: ATOM_BUCKETS pad (Claim-1 JIT sharing). off: exact N for one homogeneous system.",
    )
    p.add_argument(
        "--xla-recipe",
        choices=("default", "intensive"),
        default="default",
        help="intensive: extra GPU compile autotune; must apply before JAX import.",
    )
    args = p.parse_args(argv)

    if args.xla_recipe == "intensive":
        from prolix.xla_flags import INTENSIVE_GPU_FLAGS, merge_xla_flags

        os.environ["XLA_FLAGS"] = merge_xla_flags(
            os.environ.get("XLA_FLAGS", ""), INTENSIVE_GPU_FLAGS
        )

    remat = args.remat == "on"
    tile = int(args.tile)
    use_size_buckets = args.use_size_buckets == "on"
    git_sha = _git_sha()
    xla_flags = os.environ.get("XLA_FLAGS", "")

    from scripts.experiments.profile_b1_flash_vs_autodiff_forces import (
        PADDING,
        PROTEIN_PDBS,
        TARGET_BOX_SIZE,
        _apply_pme_config,
        _resolve_ff_path,
        bench,
    )

    import jax.numpy as jnp
    from proxide import CoordFormat, OutputSpec, parse_structure

    from prolix.physics.solvation import solvate_protein_to_bundle
    from prolix.physics.water_models import WaterModelType

    ff_path = _resolve_ff_path()
    pdb_path = ROOT / "data" / "pdb" / PROTEIN_PDBS[args.protein]
    spec = OutputSpec(parameterize_md=True, force_field=ff_path, coord_format=CoordFormat.Full)
    protein = parse_structure(str(pdb_path), spec)
    bundle = solvate_protein_to_bundle(
        protein,
        padding=PADDING,
        model_type=WaterModelType.TIP3P,
        ionic_strength=0.0,
        neutralize=True,
        target_box_size=jnp.array(TARGET_BOX_SIZE),
        use_size_buckets=use_size_buckets,
    )
    n_real = int(bundle.n_atoms)
    n_padded = int(bundle.positions.shape[0])
    n_tiles = math.ceil(n_padded / tile)

    base = {
        "status": "ok",
        "protein": args.protein,
        "pme": args.pme,
        "grid_spacing": args.grid_spacing,
        "tile": tile,
        "remat": remat,
        "use_size_buckets": use_size_buckets,
        "xla_recipe": args.xla_recipe,
        "n_real_atoms": n_real,
        "n_padded_atoms": n_padded,
        "n_tiles": n_tiles,
        "flash_forces_ms": None,
        "std_ms": None,
        "git_sha": git_sha,
        "xla_flags": xla_flags,
        "error": None,
    }

    import jax

    from prolix.api.bundle_md import displacement_fn_for_bundle, physics_system_from_bundle
    from prolix.batched_energy import (
        _angle_energy_masked,
        _bond_energy_masked,
        _cmap_energy_masked,
        _dihedral_energy_masked,
    )
    from prolix.physics.flash_explicit import flash_explicit_forces

    disp_fn, _ = displacement_fn_for_bundle(bundle)
    sys_obj = physics_system_from_bundle(bundle, bundle.positions)
    sys_obj, _grid_key = _apply_pme_config(sys_obj, args.pme, args.grid_spacing)
    r = sys_obj.positions

    def bonded_energy(positions):
        e_bond = _bond_energy_masked(
            positions, sys_obj.bonds, sys_obj.bond_params, sys_obj.bond_mask, disp_fn,
        )
        e_ub = _bond_energy_masked(
            positions, sys_obj.urey_bradley_bonds, sys_obj.urey_bradley_params,
            sys_obj.urey_bradley_mask, disp_fn,
        )
        e_angle = _angle_energy_masked(
            positions, sys_obj.angles, sys_obj.angle_params, sys_obj.angle_mask, disp_fn,
        )
        e_dih = _dihedral_energy_masked(
            positions, sys_obj.dihedrals, sys_obj.dihedral_params,
            sys_obj.dihedral_mask, disp_fn,
        )
        e_imp = _dihedral_energy_masked(
            positions, sys_obj.impropers, sys_obj.improper_params,
            sys_obj.improper_mask, disp_fn,
        )
        e_cmap = _cmap_energy_masked(
            positions, sys_obj.cmap_torsions, sys_obj.cmap_indices, sys_obj.cmap_mask,
            sys_obj.cmap_coeffs, disp_fn,
        )
        return e_bond + e_ub + e_angle + e_dih + e_imp + e_cmap

    def flash_forces():
        bonded_force = -jax.grad(bonded_energy)(r)
        f_nonbonded = flash_explicit_forces(
            sys_obj, T=tile, remat=remat,
        )
        return (bonded_force + f_nonbonded) * sys_obj.atom_mask[:, None]

    flash_jit = jax.jit(flash_forces)
    try:
        timed = bench(
            flash_jit,
            f"bonded+flash T={tile} remat={args.remat} buckets={args.use_size_buckets} xla={args.xla_recipe}",
            args.n_warmup,
            args.n_trials,
            args.n_inner,
        )
    except Exception as exc:
        if _is_oom(exc):
            base["status"] = "oom"
            base["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            log.error("OOM (recorded): %s", base["error"])
            _write(args.out, base)
            return 0
        base["status"] = "error"
        base["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        log.error("benchmark failed: %s", base["error"])
        _write(args.out, base)
        return 1

    base["flash_forces_ms"] = timed["mean_ms"]
    base["std_ms"] = timed["std_ms"]
    _write(args.out, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
