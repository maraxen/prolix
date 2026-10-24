#!/usr/bin/env python3
"""Stage-2 GPU probe: force-only flash control vs jitted settle_langevin apply_fn.

Discovery only (ungated). Times a warmed single ``apply_fn`` step with the
P1–P8 stack (``bench`` + ``ProbeRecord(stage=2)`` + profiler + HLO ops).
Does not time ``EnsemblePlan.run()`` or a fused ``fori_loop`` as the primary
metric. Optional amortized ``fori_loop(n=32)/n`` is stamped in ``config``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prof_ensemble_apply_probe")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.prof_ensemble_step import (  # noqa: E402
    _build_1vii,
    _build_dhfr,
    _run_kwargs,
)
from scripts.experiments.profile_b1_flash_vs_autodiff_forces import (  # noqa: E402
    KNOWN_LABELS,
    bench,
)


def _require_gpu() -> None:
    import jax

    devices = jax.devices()
    backend = jax.default_backend()
    log.info("jax.default_backend=%s jax.devices=%s", backend, devices)
    if backend != "gpu":
        raise SystemExit(
            f"Stage-2 probe requires JAX GPU, got default_backend={backend!r} "
            f"devices={devices}"
        )


def _allocate_neighbor(bundle, prefix: int):
    import jax.numpy as jnp

    from prolix.api.bundle_md import (
        _host_float,
        displacement_fn_for_bundle,
        positions_with_prefix,
    )
    from prolix.physics import neighbor_list as nl_mod

    displacement_fn, _ = displacement_fn_for_bundle(bundle)
    box_vec = jnp.diag(bundle.box)
    cutoff = _host_float(bundle.cutoff_distance, 9.0)
    neighbor_fn = nl_mod.make_neighbor_list_fn(displacement_fn, box_vec, cutoff)
    return neighbor_fn.allocate(positions_with_prefix(bundle, prefix))


def _force_only_fn(bundle):
    from prolix.api.bundle_md import displacement_fn_for_bundle, physics_system_from_bundle
    from prolix.batched_energy import single_padded_force

    disp_fn, _ = displacement_fn_for_bundle(bundle)
    sys_obj = physics_system_from_bundle(bundle, bundle.positions)

    def flash_forces():
        return single_padded_force(
            sys_obj,
            disp_fn,
            implicit_solvent=False,
            explicit_solvent=True,
            use_flash=True,
        )

    import jax

    return jax.jit(flash_forces)


def _apply_step_fn(bundle, *, nonbonded: str, seed: int, dt: float, kT: float, gamma: float):
    import jax

    from prolix.api import EnsemblePlan
    from prolix.api.ensemble_plan import _integrator_positions

    plan = EnsemblePlan.from_bundle(bundle)
    run_kw = _run_kwargs(nonbonded)
    use_nl = bool(run_kw.get("use_neighbor_list"))
    use_flash = bool(run_kw.get("use_flash_forces"))
    prefix = int(bundle.positions.shape[0]) if use_nl else None
    nbr0 = _allocate_neighbor(bundle, int(bundle.positions.shape[0])) if use_nl else None

    state, apply_fn, dt_s, kT_s = plan._setup_integrator(
        bundle,
        dt,
        kT,
        seed,
        integration_prefix=prefix,
        dt_unit="fs",
        gamma=gamma,
        use_flash_forces=use_flash,
        initial_neighbor=nbr0,
    )

    if use_nl:

        def step(s):
            return apply_fn(s, neighbor=nbr0, kT=kT_s, dt=dt_s)

        def update_step(s):
            new_n = nbr0.update(_integrator_positions(s))
            return apply_fn(s, neighbor=new_n, kT=kT_s, dt=dt_s)

        def fori_body(i, s):
            return apply_fn(s, neighbor=nbr0, kT=kT_s, dt=dt_s)

    else:

        def step(s):
            return apply_fn(s, kT=kT_s, dt=dt_s)

        update_step = None

        def fori_body(i, s):
            return apply_fn(s, kT=kT_s, dt=dt_s)

    step_jit = jax.jit(step)
    update_jit = jax.jit(update_step) if update_step is not None else None

    @jax.jit
    def many32(s):
        return jax.lax.fori_loop(0, 32, fori_body, s)

    # 0-arg jit so bench() and fn.lower().compile() share one program.
    timed = jax.jit(lambda: step_jit(state))

    return timed, state, update_jit, many32


def _capture_trace(fn, trace_dir: Path) -> tuple[dict, dict[str, float], list[dict], str]:
    import jax

    from xtrax.profiling.trace import (
        parse_dispatch_counts,
        parse_hlo_op_times,
        parse_scopes,
        scope_map_from_hlo_text,
    )

    hlo_text = fn.lower().compile().as_text()
    scope_map = scope_map_from_hlo_text(hlo_text, KNOWN_LABELS)
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        r = fn()
        jax.block_until_ready(r)
    gz_files = list(trace_dir.rglob("*.trace.json.gz"))
    if not gz_files:
        log.warning("no *.trace.json.gz under %s", trace_dir)
        empty_scopes = {label: None for label in KNOWN_LABELS}
        return empty_scopes, {}, [], hlo_text
    with gzip.open(gz_files[0], "rt") as fh:
        events = json.load(fh)["traceEvents"]
    scopes_raw = parse_scopes(events, scope_map)
    scopes = {label: scopes_raw.get(label) for label in KNOWN_LABELS}
    dispatch = {k: float(v) for k, v in parse_dispatch_counts(events).items()}
    hlo_ops = parse_hlo_op_times(events)
    ranked = sorted(hlo_ops.items(), key=lambda kv: -kv[1][0])
    total_hlo = sum(s for s, _ in hlo_ops.values())
    rows = []
    for op, (seconds, n) in ranked[:30]:
        rows.append(
            {
                "hlo_op": op,
                "exclusive_seconds": seconds,
                "n_occurrences": n,
                "pct_of_hlo_ops": (seconds / total_hlo) if total_hlo else None,
            }
        )
    recovered = [k for k, v in scopes.items() if v is not None]
    log.info("trace recovered %d labels: %s", len(recovered), recovered)
    if rows:
        log.info("top hlo_op=%s exclusive_seconds=%.6f", rows[0]["hlo_op"], rows[0]["exclusive_seconds"])
        if str(rows[0]["hlo_op"]).startswith("command_buffer_"):
            log.warning(
                "top hlo_op is command_buffer_*; named_scope ranking is not "
                "the production program. Do not mix a cmdbuf-off replay."
            )
    return scopes, dispatch, rows, hlo_text


def _write_hlo_dump(path: Path, rows: list[dict], host_step: float) -> None:
    top = rows[0]["hlo_op"] if rows else ""
    payload = {
        "n_distinct_in_dump": len(rows),
        "host_step_seconds": host_step,
        "top_hlo_op": top,
        "command_buffer_top": bool(str(top).startswith("command_buffer_")),
        "top": rows,
        "note": (
            "If the top row is command_buffer_*, named_scope cannot pierce "
            "the CUDA graph. cmdbuf-off is a different program; do not mix "
            "into the production ranking."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("Wrote HLO op dump to %s", path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("force", "md"), default="md")
    p.add_argument("--system", choices=("1vii", "dhfr"), default="1vii")
    p.add_argument("--nonbonded", choices=("flash", "nl", "dense"), default="flash")
    p.add_argument("--use-size-buckets", choices=("on", "off"), default="on")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-inner", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument(
        "--cmdbuf-off",
        action="store_true",
        help="Diagnostic: empty XLA command buffers (different program than production).",
    )
    p.add_argument(
        "--pallas-pme",
        action="store_true",
        help="Opt-in Pallas SPME spread/gather (PROLIX_USE_PALLAS_PME=1). Not the default.",
    )
    p.add_argument("--emit-record", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if args.pallas_pme:
        os.environ["PROLIX_USE_PALLAS_PME"] = "1"
        log.warning("Pallas SPME opt-in: not the production default (ADR-006)")

    if args.cmdbuf_off:
        # Must run before the first JAX import in this process.
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
        ).strip()
        log.warning("cmdbuf-off diagnostic: this is not the production program")

    log.info("JAX_PLATFORMS=%s", os.environ.get("JAX_PLATFORMS", "") or "(unset)")

    import jax
    import jax.numpy as jnp

    from prolix.simulate import BOLTZMANN_KCAL
    from xtrax.profiling.record import ProbeRecord

    _require_gpu()

    buckets = args.use_size_buckets == "on"
    if args.system == "dhfr":
        bundle = _build_dhfr(buckets)
    else:
        bundle = _build_1vii(buckets, jnp.float32)
    n_real = int(bundle.n_atoms)
    n_padded = int(bundle.positions.shape[0])
    log.info("bundle n_real=%d n_padded=%d arm=%s nonbonded=%s", n_real, n_padded, args.arm, args.nonbonded)

    kT = 300.0 * BOLTZMANN_KCAL
    update_ms = None
    amortized_ms = None

    if args.arm == "force":
        timed = _force_only_fn(bundle)
        bench_name = "flash_forces_control"
        mode = "force_flash"
        nonbonded_stamp = "flash"
    else:
        timed, state, update_jit, many32 = _apply_step_fn(
            bundle,
            nonbonded=args.nonbonded,
            seed=args.seed,
            dt=args.dt,
            kT=kT,
            gamma=args.gamma,
        )
        bench_name = f"apply_fn_{args.nonbonded}_no_nl_update"
        mode = f"md_{args.nonbonded}"
        nonbonded_stamp = args.nonbonded

    results = bench(timed, bench_name, args.n_warmup, args.n_trials, args.n_inner)
    host_s = results["mean_ms"] / 1000.0

    if args.arm == "md" and update_jit is not None:
        r = update_jit(state)
        jax.block_until_ready(r)
        t0 = time.perf_counter()
        r = update_jit(state)
        jax.block_until_ready(r)
        update_ms = (time.perf_counter() - t0) * 1000.0
        log.info("nl update-step trial (not in total_step_seconds): %.3f ms", update_ms)

    if args.arm == "md":
        r = many32(state)
        jax.block_until_ready(r)
        t0 = time.perf_counter()
        r = many32(state)
        jax.block_until_ready(r)
        amortized_ms = (time.perf_counter() - t0) * 1000.0 / 32.0
        log.info("amortized fori_loop(32)/32 (config only): %.3f ms", amortized_ms)

    trace_dir = args.trace_dir or (args.emit_record.parent / f"{args.emit_record.stem}_trace")
    scopes, dispatch, hlo_rows, hlo_text = _capture_trace(timed, trace_dir)
    dump_path = args.emit_record.parent / f"hlo_op_dump_{args.emit_record.stem}.json"
    _write_hlo_dump(dump_path, hlo_rows, host_s)
    hlo_text_path = args.emit_record.parent / f"hlo_as_text_{args.emit_record.stem}.txt"
    hlo_text_path.write_text(hlo_text)
    log.info("Wrote HLO as_text (%d chars) to %s", len(hlo_text), hlo_text_path)

    top_op = hlo_rows[0]["hlo_op"] if hlo_rows else ""
    cmdbuf_top = str(top_op).startswith("command_buffer_")

    metrics: dict[str, float] = {
        "total_step_seconds": host_s,
        "mean_ms": results["mean_ms"],
        "std_ms": results["std_ms"],
        "min_ms": results["min_ms"],
    }
    metrics.update(dispatch)
    if args.arm == "force":
        metrics["flash_forces_ms"] = results["mean_ms"]

    attribution = {label: "named_scope" for label, value in scopes.items() if value is not None}

    config = {
        "protein": args.system,
        "arm": args.arm,
        "mode": mode,
        "nonbonded": nonbonded_stamp,
        "use_size_buckets": args.use_size_buckets,
        "cmdbuf_off": "true" if args.cmdbuf_off else "false",
        "use_pallas_pme": "true" if args.pallas_pme else "false",
        "n_padded_atoms": str(n_padded),
        "n_warmup": str(args.n_warmup),
        "n_trials": str(args.n_trials),
        "n_inner": str(args.n_inner),
        "top_hlo_op": top_op or "none",
        "command_buffer_top": "true" if cmdbuf_top else "false",
        "hlo_op_dump": str(dump_path),
        "nl_update_in_timed_fn": "false",
        "primary_metric": "jitted_single_apply_fn" if args.arm == "md" else "force_only_flash",
    }
    if update_ms is not None:
        config["nl_update_step_ms"] = f"{update_ms:.6f}"
    if amortized_ms is not None:
        config["amortized_fori32_ms_per_step"] = f"{amortized_ms:.6f}"

    probe_id = f"stage2_{args.system}_{mode}_b{args.use_size_buckets}"
    if args.cmdbuf_off:
        probe_id += "_cmdbufoff"
    if args.pallas_pme:
        probe_id += "_pallas"
    record = ProbeRecord(
        probe_id=probe_id,
        stage=2,
        n_atoms=n_real,
        platform="gpu",
        metrics=metrics,
        scopes=scopes,
        attribution_method=attribution,
        config=config,
    )
    record.write(args.emit_record)
    log.info("Wrote ProbeRecord to %s", args.emit_record)
    print(
        json.dumps(
            {
                "probe_id": probe_id,
                "mean_ms": results["mean_ms"],
                "n_padded": n_padded,
                "top_hlo_op": top_op,
                "command_buffer_top": cmdbuf_top,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
