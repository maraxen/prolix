#!/usr/bin/env python3
"""Phase 5 scoping (task 260715_b1_physics_parity) + P6/P7 probe.

P6: CPU smoke, --stage 1, --emit-record.
P7: GPU Stage-2 sweep flags --mode/--pme/--grid-spacing, --stage 2, traces.

The frozen sidecar profile_b1_flash_vs_autodiff_forces.bth.toml is not
edited; extra axes live on ProbeRecord.config.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
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

KNOWN_LABELS = frozenset(
    {
        "dense_bonded_angle", "dense_bonded_bond", "dense_bonded_cmap",
        "dense_bonded_dihedral", "dense_bonded_improper",
        "dense_bonded_urey_bradley", "dense_coulomb_direct",
        "dense_lj_direct", "dense_pme_reciprocal",
        "flash_bonded", "flash_grad", "flash_lj_only",
        "flash_nonbonded_tiles", "pme_bwd_gather", "pme_charge_spread",
        "pme_fft_forward", "pme_fft_inverse", "pme_greens_setup",
        "pme_greens_setup_standalone", "settle_cramers_rule",
        "settle_pos_eigh", "settle_rattle_loop",
        "nl_lj_forward", "nl_lj_backward", "nl_coulomb_forward", "nl_coulomb_backward",
    }
)


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


def _apply_pme_config(sys_obj, pme: str, grid_spacing: float):
    import dataclasses

    import numpy as np

    # pme_alpha / pme_grid_points are eqx.field(static=True) on PhysicsSystem,
    # so eqx.tree_at cannot replace them (cluster 20707425).
    mean_l = float(np.mean(np.asarray(sys_obj.box_size)))
    if pme == "off":
        sys_obj = dataclasses.replace(sys_obj, pme_alpha=0.0)
        return sys_obj, "na"
    ngrid = max(int(math.ceil(mean_l / float(grid_spacing))), 1)
    sys_obj = dataclasses.replace(sys_obj, pme_grid_points=ngrid)
    return sys_obj, str(grid_spacing)


def bench(fn, name, n_warmup=3, n_trials=20, n_inner=5):
    """Same convention as tests/test_gpu_profile_components.py::bench."""
    import jax

    t_first = time.perf_counter()
    r = fn()
    jax.block_until_ready(r)
    first_s = time.perf_counter() - t_first
    log.info("  %s first_call_seconds=%.3f (compile + first warmup)", name, first_s)
    for _ in range(max(n_warmup - 1, 0)):
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


def _capture_trace(fn, trace_dir: Path) -> tuple[dict, dict[str, float]]:
    """One post-warmup traced call; return (scopes, dispatch metrics)."""
    import jax

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from xtrax.profiling.trace import parse_dispatch_counts, parse_scopes, scope_map_from_hlo_text

    hlo_text = fn.lower().compile().as_text()
    scope_map = scope_map_from_hlo_text(hlo_text, KNOWN_LABELS)
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        r = fn()
        jax.block_until_ready(r)
    gz_files = list(trace_dir.rglob("*.trace.json.gz"))
    if not gz_files:
        log.warning("no *.trace.json.gz under %s", trace_dir)
        return {label: None for label in KNOWN_LABELS}, {}
    with gzip.open(gz_files[0], "rt") as fh:
        events = json.load(fh)["traceEvents"]
    scopes_raw = parse_scopes(events, scope_map)
    scopes = {label: scopes_raw.get(label) for label in KNOWN_LABELS}
    dispatch = {k: float(v) for k, v in parse_dispatch_counts(events).items()}
    recovered = [k for k, v in scopes.items() if v is not None]
    log.info("trace recovered %d labels: %s", len(recovered), recovered)
    return scopes, dispatch


def _timed_seconds(results: dict) -> float:
    candidates = []
    for key in ("autodiff_forces", "flash_forces", "nl_forces", "energy_only"):
        if key in results:
            candidates.append(results[key]["mean_ms"] / 1000.0)
    if not candidates:
        raise RuntimeError("no timed results to stamp total_step_seconds")
    return max(candidates)


def _emit_probe_record(
    *,
    protein: str,
    stage: int,
    n_real: int,
    n_padded: int,
    n_warmup: int,
    n_trials: int,
    n_inner: int,
    results: dict,
    path: Path,
    force_mode: str,
    pme: str,
    grid_spacing: str,
    scopes_arm: str,
    platform: str,
    scopes: dict | None,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from xtrax.profiling.record import ProbeRecord

    metrics: dict[str, float] = {"total_step_seconds": _timed_seconds(results)}
    if "energy_only" in results:
        metrics["energy_only_ms"] = results["energy_only"]["mean_ms"]
    if "autodiff_forces" in results:
        metrics["autodiff_forces_ms"] = results["autodiff_forces"]["mean_ms"]
    if "flash_forces" in results:
        metrics["flash_forces_ms"] = results["flash_forces"]["mean_ms"]
        if "autodiff_forces" in results:
            metrics["flash_speedup"] = (
                results["autodiff_forces"]["mean_ms"] / results["flash_forces"]["mean_ms"]
            )
    if extra_metrics:
        metrics.update(extra_metrics)

    attribution = None
    if scopes is not None:
        # Keep {} when every scope value is None (missing labels). `or None`
        # made ProbeRecord reject the record (cluster 20762713).
        attribution = {
            label: "named_scope" for label, value in scopes.items() if value is not None
        }

    record_mode = force_mode if force_mode in ("dense", "flash") else "both"
    probe_id = f"stage{stage}_{protein}_{record_mode}_pme{pme}_g{grid_spacing}_sc{scopes_arm}"
    record = ProbeRecord(
        probe_id=probe_id,
        stage=stage,
        n_atoms=n_real,
        platform=platform,
        metrics=metrics,
        scopes=scopes,
        attribution_method=attribution,
        config={
            "protein": protein,
            "mode": record_mode if record_mode != "both" else "dense+flash",
            "pme": pme,
            "grid_spacing": grid_spacing,
            "scopes": scopes_arm,
            "n_padded_atoms": str(n_padded),
            "n_warmup": str(n_warmup),
            "n_trials": str(n_trials),
            "n_inner": str(n_inner),
        },
    )
    record.write(path)
    log.info("Wrote ProbeRecord to %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure single_padded_energy+jax.grad vs single_padded_force(use_flash=True) wall-clock"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protein", choices=("1vii", "2gb1"), default="1vii")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-inner", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--emit-record", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("dense", "flash", "nl", "both"),
        default="both",
        help="which force path to time (P7 array passes dense|flash); "
        "'both' also times 'nl' for a 3-way comparison",
    )
    parser.add_argument("--pme", choices=("on", "off"), default="on")
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument(
        "--scopes",
        choices=("on", "off"),
        default="on",
        help="config stamp only; the scopes_off patch lives in prof_stage2_scope_ab.py",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    import jax

    log.info("jax.config.x64_enabled = %s", jax.config.x64_enabled)
    log.info(
        "protein=%s mode=%s pme=%s grid=%s stage=%s",
        args.protein, args.mode, args.pme, args.grid_spacing, args.stage,
    )

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
        summary = {
            "mode": "dry-run",
            "protein": args.protein,
            "force_mode": args.mode,
            "pme": args.pme,
            **bundle_info,
        }
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2))
        return 0

    import equinox as eqx

    from prolix.api.bundle_md import displacement_fn_for_bundle, physics_system_from_bundle
    from prolix.batched_energy import single_padded_energy, single_padded_force
    from prolix.physics.neighbor_list import make_neighbor_list_fn

    disp_fn, _ = displacement_fn_for_bundle(bundle)
    positions = bundle.positions
    sys_obj = physics_system_from_bundle(bundle, positions)
    sys_obj, grid_key = _apply_pme_config(sys_obj, args.pme, args.grid_spacing)

    def energy_only():
        return single_padded_energy(sys_obj, disp_fn, implicit_solvent=False)

    def autodiff_forces():
        return jax.grad(
            lambda r: single_padded_energy(
                eqx.tree_at(lambda s: s.positions, sys_obj, r),
                disp_fn,
                implicit_solvent=False,
            )
        )(positions)

    def flash_forces():
        return single_padded_force(
            sys_obj, disp_fn, implicit_solvent=False, explicit_solvent=True, use_flash=True,
        )

    # NL comparison (task #4793 follow-up): same dense energy path, but
    # direct-space LJ/Coulomb via O(N*K) chunked_*_energy_nl instead of
    # O(N^2) masked -- matches EnsemblePlan's use_neighbor_list=True branch
    # (bundle_md.py force_fn_from_bundle uses flash, not NL, for forces;
    # NL only exists on the energy/jax.grad side per single_padded_energy's
    # own docstring -- debt 760).
    import jax.numpy as jnp

    cutoff = float(getattr(bundle, "cutoff_distance", 9.0))
    box_vec = jnp.diag(bundle.box)
    neighbor_fn = make_neighbor_list_fn(disp_fn, box_vec, cutoff)
    nbr = neighbor_fn.allocate(positions)
    log.info("NL allocated: k_max=%d (n_padded=%d, k/n=%.3f)", nbr.idx.shape[-1], positions.shape[0], nbr.idx.shape[-1] / positions.shape[0])

    def nl_forces():
        return jax.grad(
            lambda r: single_padded_energy(
                eqx.tree_at(lambda s: s.positions, sys_obj, r),
                disp_fn,
                implicit_solvent=False,
                neighbor=nbr,
            )
        )(positions)

    energy_jit = jax.jit(energy_only)
    autodiff_jit = jax.jit(autodiff_forces)
    flash_jit = jax.jit(flash_forces)
    nl_jit = jax.jit(nl_forces)

    log.info("Benchmarking (n_warmup=%d, n_trials=%d, n_inner=%d)...", args.n_warmup, args.n_trials, args.n_inner)
    results: dict = {}
    flash_error = None
    timed_fn = None
    timed_name = None

    if args.mode == "both":
        results["energy_only"] = bench(
            energy_jit, "single_padded_energy (forward only)", args.n_warmup, args.n_trials, args.n_inner,
        )
        results["autodiff_forces"] = bench(
            autodiff_jit, "single_padded_energy + jax.grad (current EnsemblePlan path)",
            args.n_warmup, args.n_trials, args.n_inner,
        )
        try:
            results["flash_forces"] = bench(
                flash_jit, "single_padded_force(use_flash=True) (debt 761 candidate)",
                args.n_warmup, args.n_trials, args.n_inner,
            )
        except Exception as e:
            flash_error = f"{type(e).__name__}: {str(e)[:300]}"
            log.error("flash_forces benchmark failed: %s", flash_error)
        results["nl_forces"] = bench(
            nl_jit, "single_padded_energy(neighbor=nbr) + jax.grad (#4793 NL comparison)",
            args.n_warmup, args.n_trials, args.n_inner,
        )
        timed_fn, timed_name = autodiff_jit, "autodiff_forces"
    elif args.mode == "dense":
        results["autodiff_forces"] = bench(
            autodiff_jit, "dense autodiff forces", args.n_warmup, args.n_trials, args.n_inner,
        )
        timed_fn, timed_name = autodiff_jit, "autodiff_forces"
    elif args.mode == "nl":
        results["nl_forces"] = bench(
            nl_jit, "NL forces", args.n_warmup, args.n_trials, args.n_inner,
        )
        timed_fn, timed_name = nl_jit, "nl_forces"
    else:
        try:
            results["flash_forces"] = bench(
                flash_jit, "flash forces", args.n_warmup, args.n_trials, args.n_inner,
            )
        except Exception as e:
            flash_error = f"{type(e).__name__}: {str(e)[:300]}"
            log.error("flash_forces benchmark failed: %s", flash_error)
            return 1
        timed_fn, timed_name = flash_jit, "flash_forces"

    flash_ms = results.get("flash_forces", {}).get("mean_ms") if "flash_forces" in results else None
    autodiff_ms = results.get("autodiff_forces", {}).get("mean_ms") if "autodiff_forces" in results else None
    energy_ms = results.get("energy_only", {}).get("mean_ms") if "energy_only" in results else None
    nl_ms = results.get("nl_forces", {}).get("mean_ms") if "nl_forces" in results else None
    speedup = (autodiff_ms / flash_ms) if (autodiff_ms and flash_ms) else None
    if speedup is not None:
        log.info("Flash speedup over autodiff: %.3fx", speedup)
    nl_speedup = (autodiff_ms / nl_ms) if (autodiff_ms and nl_ms) else None
    if nl_speedup is not None:
        log.info("NL speedup over dense autodiff: %.3fx", nl_speedup)

    summary = {
        "mode": args.mode,
        "protein": args.protein,
        **bundle_info,
        "energy_only_ms": energy_ms,
        "autodiff_forces_ms": autodiff_ms,
        "flash_forces_ms": flash_ms,
        "flash_speedup": speedup,
        "flash_error": flash_error,
        "nl_forces_ms": nl_ms,
        "nl_speedup": nl_speedup,
        "nl_k_max": int(nbr.idx.shape[-1]) if "nl_forces" in results or args.mode == "nl" else None,
        "pme": args.pme,
        "grid_spacing": grid_key,
        "raw_results": results,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log.info("Wrote summary to %s", args.out)

    scopes = None
    extra_metrics: dict[str, float] = {}
    if args.emit_record is not None and args.stage >= 2 and timed_fn is not None:
        trace_dir = args.trace_dir or (args.emit_record.parent / f"{args.emit_record.stem}_trace")
        scopes, extra_metrics = _capture_trace(timed_fn, trace_dir)
        log.info("captured trace for %s under %s", timed_name, trace_dir)

    backend = jax.default_backend()
    platform = "gpu" if backend == "gpu" else "cpu"
    if args.emit_record is not None:
        _emit_probe_record(
            protein=args.protein,
            stage=args.stage,
            n_real=n_real,
            n_padded=n_padded,
            n_warmup=args.n_warmup,
            n_trials=args.n_trials,
            n_inner=args.n_inner,
            results=results,
            path=args.emit_record,
            force_mode=args.mode,
            pme=args.pme,
            grid_spacing=grid_key,
            scopes_arm=args.scopes,
            platform=platform,
            scopes=scopes,
            extra_metrics=extra_metrics or None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
