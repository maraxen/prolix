#!/usr/bin/env python3
"""Phase 5 scoping (task 260715_b1_physics_parity): is debt 761 (wire
single_padded_force's flash path into EnsemblePlan) worth any engineering
investment at all?

Background: EnsemblePlan's dispatch path (energy_fn_from_bundle,
src/prolix/api/bundle_md.py:446) calls single_padded_energy + naive jax.grad
to get forces -- a dense O(N^2) autodiff pass. A separate, already-built,
already-production-used force path exists: single_padded_force
(src/prolix/batched_energy.py:604), whose default use_flash=True branch
(flash_explicit_forces, src/prolix/physics/flash_explicit.py) tiles the
direct-space nonbonded pass into 256x256 blocks under jax.checkpoint instead
of building the full dense (N,N) exclusion matrix.

Two corrections found during Phase 5 scoping (read-only investigation,
.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md) mean this
is NOT a free win to assume:
  1. flash_explicit_forces is NOT closed-form analytical -- it still calls
     jax.grad (via eqx.filter_grad) internally. The only difference from
     today's path is the tiled/checkpointed memory-locality pattern.
  2. The PME custom_vjp benefit (make_spme_energy_fn) is already present in
     today's single_padded_energy+jax.grad path -- it is not unique to flash.

So the ONLY thing this script measures is whether flash's tiled/checkpointed
direct-space pass is actually faster than the current dense pass, on a real
bundle-derived PhysicsSystem (the 1VII/2GB1 solvated-protein bundles built by
solvate_protein_to_bundle this session -- same construction path
profile_b1_heterogeneous_solvated_compile.py already validated for compile-
sharing). This is measurement only -- no EnsemblePlan wiring happens here.

Usage::

    # L1 dry-run (shapes only, no GPU)
    uv run python scripts/experiments/profile_b1_flash_vs_autodiff_forces.py --dry-run --protein 1vii

    # L2 local CPU smoke
    JAX_PLATFORMS=cpu uv run python scripts/experiments/profile_b1_flash_vs_autodiff_forces.py \\
        --protein 1vii --n-warmup 1 --n-trials 2 --n-inner 1 --out /tmp/smoke.json

    # L3 cluster (via bth run, see campaign 32d6574e)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_b1_flash_vs_autodiff")

ROOT = Path(__file__).resolve().parents[2]

PADDING = 8.0
TARGET_BOX_SIZE = [32.0, 32.0, 32.0]

PROTEIN_PDBS = {
    "1vii": "1VII.pdb",
    "2gb1": "2GB1.pdb",
}


def _resolve_ff_path() -> str:
    sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
    from _b1_paramize import _resolve_ff_path as _resolve

    return _resolve("protein.ff19SB.xml")


def _load_and_solvate(protein_key: str):
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
    )


def bench(fn, name, n_warmup=3, n_trials=20, n_inner=5):
    """Same convention as tests/test_gpu_profile_components.py::bench."""
    import jax

    for _ in range(n_warmup):
        r = fn()
        jax.block_until_ready(r)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            r = fn()
            jax.block_until_ready(r)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000 / n_inner)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    log.info("  %-45s %8.3f ms ± %5.3f ms  (min=%.3f)", name, avg, std, mn)
    return {"mean_ms": avg, "std_ms": std, "min_ms": mn}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure single_padded_energy+jax.grad vs single_padded_force(use_flash=True) wall-clock"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protein", choices=("1vii", "2gb1"), default="1vii")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-inner", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import jax

    log.info("jax.config.x64_enabled = %s", jax.config.x64_enabled)
    log.info("protein=%s", args.protein)

    try:
        bundle = _load_and_solvate(args.protein)
    except Exception as e:
        log.error("Bundle construction failed: %s", e)
        return 1

    n_real = int(bundle.n_atoms)
    n_padded = int(bundle.positions.shape[0])
    log.info("Built bundle: n_real_atoms=%d, n_padded_atoms=%d, shape_spec=%s", n_real, n_padded, bundle.shape_spec)

    bundle_info = {"n_real_atoms": n_real, "n_padded_atoms": n_padded}

    if args.dry_run:
        summary = {"mode": "dry-run", "protein": args.protein, **bundle_info}
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2))
        return 0

    from prolix.api.bundle_md import displacement_fn_for_bundle, physics_system_from_bundle
    from prolix.batched_energy import single_padded_energy, single_padded_force

    disp_fn, _ = displacement_fn_for_bundle(bundle)
    positions = bundle.positions
    sys_obj = physics_system_from_bundle(bundle, positions)

    def energy_only():
        return single_padded_energy(sys_obj, disp_fn, implicit_solvent=False)

    def autodiff_forces():
        return jax.grad(
            lambda r: single_padded_energy(
                physics_system_from_bundle(bundle, r), disp_fn, implicit_solvent=False
            )
        )(positions)

    def flash_forces():
        return single_padded_force(
            sys_obj, disp_fn, implicit_solvent=False, explicit_solvent=True, use_flash=True,
        )

    energy_jit = jax.jit(energy_only)
    autodiff_jit = jax.jit(autodiff_forces)
    flash_jit = jax.jit(flash_forces)

    log.info("Benchmarking (n_warmup=%d, n_trials=%d, n_inner=%d)...", args.n_warmup, args.n_trials, args.n_inner)
    results = {
        "energy_only": bench(energy_jit, "single_padded_energy (forward only)", args.n_warmup, args.n_trials, args.n_inner),
        "autodiff_forces": bench(autodiff_jit, "single_padded_energy + jax.grad (current EnsemblePlan path)", args.n_warmup, args.n_trials, args.n_inner),
    }

    flash_error = None
    try:
        results["flash_forces"] = bench(flash_jit, "single_padded_force(use_flash=True) (debt 761 candidate)", args.n_warmup, args.n_trials, args.n_inner)
    except Exception as e:
        # KNOWN BLOCKING BUG (debt 765, filed 2026-07-17): flash_explicit_forces
        # expects sys.excl_indices/excl_scales_vdw/excl_scales_elec in a
        # per-atom-row layout (N, max_excl_per_atom); physics_system_from_bundle
        # populates the pair-list layout (n_pairs, 2)/(n_pairs,) that
        # single_padded_energy's dense path and the PME exclusion correction
        # both correctly consume. This crashes on any bundle-derived
        # PhysicsSystem, independent of jit/device -- reported here as a null
        # result rather than a hard script failure, so this benchmark degrades
        # gracefully once/if 765 is fixed rather than needing a rewrite.
        flash_error = f"{type(e).__name__}: {str(e)[:300]}"
        log.error("flash_forces benchmark failed (see debt 765): %s", flash_error)

    flash_ms = results.get("flash_forces", {}).get("mean_ms") if "flash_forces" in results else None
    speedup = (results["autodiff_forces"]["mean_ms"] / flash_ms) if flash_ms else None
    if speedup is not None:
        log.info("Flash speedup over autodiff: %.3fx", speedup)

    summary = {
        "mode": "bench",
        "protein": args.protein,
        **bundle_info,
        "energy_only_ms": results["energy_only"]["mean_ms"],
        "autodiff_forces_ms": results["autodiff_forces"]["mean_ms"],
        "flash_forces_ms": flash_ms,
        "flash_speedup": speedup,
        "flash_error": flash_error,
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
