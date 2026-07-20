#!/usr/bin/env python3
"""Phase 4 Step 3a/3b: HLO structural check + jax.profiler trace of B1's water-class program.

Background: B1's water shape-class (4 TIP3P waters, periodic PME, real SETTLE
constraint-solving) is the leading suspect for prolix's ~7.28x steady-state
slowdown vs OpenMM. Before spending GPU time on a device trace, this script
first answers a free, CPU-only structural question: does XLA hoist the PME
Green's-function/B-spline-modulus setup (named_scope "pme_greens_setup",
added to src/prolix/physics/pme.py in this same Phase 4 pass) out of the
compiled while_loop body, or does it get recomputed every one of the ~200k
steps? This directly informs whether that block is a real optimization target.

Step 3a (--hlo-only): faithfully reconstructs the exact jax.jit closure that
EnsemblePlan._run_stacked_dispatch builds internally for run_mode="inference"
(one XLA program: vmap over replicas x lax.while_loop over steps), but exposes
.lower().compile().as_text() instead of executing it. CPU-only, no GPU needed.

Step 3b (--trace): captures a real jax.profiler Perfetto trace of the same
water-class program under execution, with the named_scope labels (pme.py /
settle.py) visible as trace regions. Requires a GPU (myxcel gpu/l40s preset)
to produce a representative device trace.

Usage::

    # Free, CPU-only structural check
    JAX_PLATFORMS=cpu uv run python scripts/profile_b1_water_trace.py --hlo-only --n-steps 20

    # Device trace (GPU required)
    uv run python scripts/profile_b1_water_trace.py --trace --n-steps 100 --n-trials 5

KNOWN LIMITATION (--hlo-only): at tiny toy scale (replicas=2, n_steps=5), the
compiled HLO text shows zero occurrences of every pme_* named_scope label,
even though this script's own reconstruction was directly verified (byte-
identical final positions) against EnsemblePlan._run_stacked_dispatch itself,
and PME reciprocal energy is independently confirmed to execute correctly at
that scale (research record
260717_pme_reciprocal_silently_disabled_under_stacked_dispatch's post-fix
verification). Root cause not yet pinned down -- likely XLA constant-folding
or op-fusion at this toy scale drops the named_scope metadata rather than
dropping the computation itself (settle_* labels DO survive at this scale,
serving as a positive control that the grep methodology itself works). Re-
check at production scale (replicas=16, n_steps>=200) before trusting a
"not_found_in_hlo" verdict for any pme_* label -- it may just mean "not
visible at this trace size," not "not computed."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_b1_water_trace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

# Prereg pins (must match scripts/benchmarks/b1_init_exec.py)
DT_FS = 0.5
KT_KCAL = 0.596  # ~300 K
GAMMA_PS = 50.0

# named_scope labels added to pme.py / settle.py in this same Phase 4 pass.
NAMED_SCOPES = [
    "pme_greens_setup",
    "pme_greens_setup_standalone",
    "pme_fft_forward",
    "pme_fft_inverse",
    "pme_charge_spread",
    "pme_bwd_gather",
    "settle_pos_eigh",
    "settle_cramers_rule",
    "settle_rattle_loop",
]


def _build_water_plan(replicas: int):
    """Water-only EnsemblePlan at B1's real vmap width (16 at prereg scale)."""
    from b1_init_exec import _four_water_bundle
    from prolix.api import EnsemblePlan

    water = _four_water_bundle()
    bundles = [water] * replicas
    plan = EnsemblePlan.from_bundles(bundles)
    return plan, bundles


def _build_batched_callable(plan, bundles, n_steps: int, seed: int = 0):
    """Reconstruct EnsemblePlan._run_stacked_dispatch's inference-mode jit
    closure (src/prolix/api/ensemble_plan.py, run_mode == "inference" branch)
    line for line, but return the jitted callable + its args instead of
    calling it immediately -- so callers can use .lower().compile().as_text()
    (HLO inspection, no execution) or call it directly under a
    jax.profiler.trace() context (device trace).

    NOTE: this intentionally calls EnsemblePlan's private (single-underscore)
    internals (plan._run_single_inference, plan.batch_plan) to stay
    byte-for-byte faithful to the real dispatch path -- if
    EnsemblePlan._run_stacked_dispatch's structure changes, update this to
    match. See the real method for the source of truth.
    """
    import jax
    import jax.numpy as jnp

    from prolix.api.bundle_stack import (
        integration_prefix_for_bundles,
        stack_molecular_bundles,
    )
    from prolix.api.ensemble_dispatch import dispatch_n_mols

    stacked = stack_molecular_bundles(bundles)
    n_systems = len(bundles)
    seeds = jnp.arange(n_systems, dtype=jnp.int32) + seed
    integration_prefix = integration_prefix_for_bundles(bundles)
    batch_plan = plan.batch_plan

    def run_one(bundle, seed_i):
        return plan._run_single_inference(
            bundle,
            n_steps,
            DT_FS,
            KT_KCAL,
            seed_i,
            observables=None,
            integration_prefix=integration_prefix,
            trim_output=False,
            dt_unit="fs",
            gamma=GAMMA_PS,
        )

    @jax.jit
    def _batched(sb, sd):
        return dispatch_n_mols(batch_plan, n_systems, run_one, sb, sd)

    return _batched, (stacked, seeds)


def _analyze_hlo_named_scopes(hlo_text: str, scopes: list[str]) -> dict:
    """Classify each named_scope's occurrences as inside a while-loop body
    (recomputed every iteration) vs. entry-only (hoisted, computed once).

    XLA's HLO pretty-printer always places a computation's opening/closing
    braces at column 0 (nested shape/layout annotations like f32[64]{0} use
    braces inline, never at line-start), so a column-anchored regex reliably
    isolates computation blocks without being confused by those.
    """
    while_refs = re.findall(
        r"while\([^)]*\),\s*condition=(%?[\w.]+),\s*body=(%?[\w.]+)", hlo_text
    )
    body_names = {b.lstrip("%") for _, b in while_refs}
    log.info(
        "Found %d while() instruction(s); body computation name(s): %s",
        len(while_refs), sorted(body_names),
    )

    block_pattern = re.compile(r"^([^\n{]*)\{\n(.*?)\n\}\s*$", re.MULTILINE | re.DOTALL)
    blocks = []
    for m in block_pattern.finditer(hlo_text):
        header, body = m.group(1), m.group(2)
        name_match = re.search(r"(%[\w.]+)", header)
        name = name_match.group(1).lstrip("%") if name_match else None
        blocks.append({
            "name": name,
            "is_entry": "ENTRY" in header,
            "is_while_body": name in body_names,
            "body": body,
        })
    log.info(
        "Parsed %d computation block(s) from HLO text (%d identified as while-loop bodies)",
        len(blocks), sum(1 for b in blocks if b["is_while_body"]),
    )

    findings = {}
    for scope in scopes:
        total = len(re.findall(re.escape(scope), hlo_text))
        in_while_body = sum(
            len(re.findall(re.escape(scope), b["body"])) for b in blocks if b["is_while_body"]
        )
        in_entry = sum(
            len(re.findall(re.escape(scope), b["body"])) for b in blocks if b["is_entry"]
        )
        if in_while_body > 0:
            verdict = "recomputed_every_iteration"
        elif total > 0:
            verdict = "hoisted_or_entry_only"
        else:
            verdict = "not_found_in_hlo"
        findings[scope] = {
            "total_occurrences": total,
            "occurrences_in_while_body": in_while_body,
            "occurrences_in_entry": in_entry,
            "verdict": verdict,
        }
        log.info(
            "  %-30s total=%-4d in_while_body=%-4d in_entry=%-4d verdict=%s",
            scope, total, in_while_body, in_entry, verdict,
        )

    f64_hits = len(re.findall(r"\bf64\b", hlo_text))
    convert_hits = len(re.findall(r"\bconvert\(", hlo_text))
    log.info("HLO f64 dtype mentions: %d, convert() ops: %d", f64_hits, convert_hits)

    return {
        "n_blocks_found": len(blocks),
        "n_while_bodies_found": len(body_names),
        "named_scope_findings": findings,
        "f64_mentions": f64_hits,
        "convert_op_count": convert_hits,
    }


def run_hlo_only(replicas: int, n_steps: int) -> dict:
    log.info("Building water-only plan (replicas=%d)...", replicas)
    plan, bundles = _build_water_plan(replicas)
    batched_fn, args = _build_batched_callable(plan, bundles, n_steps)

    log.info("Lowering + compiling (no execution)...")
    compiled = batched_fn.lower(*args).compile()
    hlo_text = compiled.as_text()

    out_dir = ROOT / "outputs" / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    hlo_path = out_dir / f"b1_water_hlo_n{n_steps}_r{replicas}.txt"
    hlo_path.write_text(hlo_text)
    log.info("Wrote raw HLO text (%d bytes) to %s", len(hlo_text), hlo_path)

    analysis = _analyze_hlo_named_scopes(hlo_text, NAMED_SCOPES)
    return {
        "mode": "hlo-only",
        "n_steps": n_steps,
        "replicas": replicas,
        "hlo_path": str(hlo_path),
        "hlo_bytes": len(hlo_text),
        **analysis,
    }


def run_trace(replicas: int, n_steps: int, n_trials: int) -> dict:
    import jax

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    trace_dir = ROOT / "outputs" / "profiling" / f"b1_water_trace_{job_id}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building water-only plan (replicas=%d)...", replicas)
    plan, bundles = _build_water_plan(replicas)
    batched_fn, args = _build_batched_callable(plan, bundles, n_steps)

    log.info("Warming up (outside trace, triggers compile)...")
    result = batched_fn(*args)
    jax.block_until_ready(result)
    log.info("Warmup done.")

    log.info("Capturing %d trial(s) under jax.profiler.trace -> %s", n_trials, trace_dir)
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for i in range(n_trials):
            result = batched_fn(*args)
            jax.block_until_ready(result)
            log.info("  trial %d/%d done", i + 1, n_trials)

    log.info("Trace written to %s", trace_dir)
    log.info(
        "View: pull this directory locally (myxcel pull_project / bth pull), "
        "then drag the .trace.json.gz into https://ui.perfetto.dev "
        "(named_scope labels appear as nested tracks)."
    )
    return {
        "mode": "trace",
        "n_steps": n_steps,
        "replicas": replicas,
        "n_trials": n_trials,
        "trace_dir": str(trace_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B1 water-class HLO structural check + jax.profiler trace capture"
    )
    parser.add_argument("--hlo-only", action="store_true", help="CPU-only HLO structural check")
    parser.add_argument("--trace", action="store_true", help="GPU device trace capture")
    parser.add_argument("--n-steps", type=int, default=50, help="Steps in the compiled while_loop")
    parser.add_argument("--replicas", type=int, default=16, help="vmap width (default matches B1 prereg)")
    parser.add_argument("--n-trials", type=int, default=5, help="Repeated calls captured in the trace")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write JSON summary")
    args = parser.parse_args()

    if not args.hlo_only and not args.trace:
        parser.error("pass --hlo-only and/or --trace")

    import jax

    log.info("jax.config.x64_enabled = %s", jax.config.x64_enabled)
    log.info("jax.devices() = %s", jax.devices())

    hlo_result = run_hlo_only(args.replicas, args.n_steps) if args.hlo_only else None
    trace_result = run_trace(args.replicas, args.n_steps, args.n_trials) if args.trace else None

    # Flat top-level fields for bathos's DuckDB-based outcome gate (result_schema
    # in profile_b1_water_trace.bth.toml expects flat scalars) -- nested
    # hlo_only/trace blocks remain for direct analysis/synthesis-doc use.
    mode = "both" if (hlo_result and trace_result) else ("hlo_only" if hlo_result else "trace")
    summary = {
        "mode": mode,
        "n_steps": args.n_steps,
        "replicas": args.replicas,
        "n_while_bodies_found": hlo_result["n_while_bodies_found"] if hlo_result else None,
        "n_blocks_found": hlo_result["n_blocks_found"] if hlo_result else None,
        "n_trials": trace_result["n_trials"] if trace_result else None,
        "trace_dir": trace_result["trace_dir"] if trace_result else None,
        "hlo_only": hlo_result,
        "trace": trace_result,
    }

    text = json.dumps(summary, indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log.info("Wrote summary to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
