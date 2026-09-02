#!/usr/bin/env python3
"""Stage 1 probe: executed CPU micro-probe with trace capture.

See .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section
P4. Executes real integrator steps (reusing
scripts/experiments/profile_b1_water_trace.py's water-plan builder -- do not
re-derive it) under jax.profiler.trace, parses the trace via
xtrax.profiling.trace into per-named_scope wall times + dispatch
counts, and emits ProbeRecord(stage=1, platform="cpu", ...).

Placed in scripts/explore/ (ad-hoc, no bathos sidecar required per the
project's bathos hard rules).

Usage::

    uv run python scripts/explore/prof_stage1_cpu_micro.py
    uv run python scripts/explore/prof_stage1_cpu_micro.py --n-steps 20
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys as _sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from xtrax.profiling.claims import (  # noqa: E402
    ClaimClass,
    ClaimValidityError,
    assert_claim_supported,
)
from xtrax.profiling.emitters import attribution_from_scopes  # noqa: E402
from xtrax.profiling.record import ProbeRecord  # noqa: E402
from xtrax.profiling.trace import (  # noqa: E402
    parse_dispatch_counts,
    parse_scopes,
    scope_map_from_hlo_text,
)

# Every named_scope label currently wrapped anywhere in src/prolix/ (P2's
# dense_*/flash_* additions plus pme.py's/settle.py's pre-existing ones).
# See tests/profiling/test_trace_parse.py's real-fixture leg and backlog
# #4330 for why settle_*/pme_* are not expected to recover non-zero
# exclusive time at this scale even though dense_bonded_* reliably does.
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
    }
)


def _load_reference_module():
    spec = importlib.util.spec_from_file_location(
        "b1_water_trace_stage1",
        _ROOT / "scripts" / "experiments" / "profile_b1_water_trace.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _probe(n_steps: int, trace_dir: Path) -> ProbeRecord:
    import jax

    mod = _load_reference_module()
    # replicas=1 fails stack_molecular_bundles's stack-compatibility check
    # (a length-1 list edge case unrelated to this probe); replicas=2 is the
    # smallest replica count that stacks cleanly and matches
    # tests/profiling/fixtures/b1_water_stage1.trace.json.gz's own params.
    plan, bundles = mod._build_water_plan(replicas=2)
    batched_fn, args = mod._build_batched_callable(plan, bundles, n_steps)

    hlo_text = batched_fn.lower(*args).compile().as_text()
    scope_map = scope_map_from_hlo_text(hlo_text, KNOWN_LABELS)

    # Warm-up outside the trace (matches profile_b1_water_trace.py's own
    # convention: compile cost must never enter the timed region).
    result = batched_fn(*args)
    jax.block_until_ready(result)

    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        t0 = time.perf_counter()
        result = batched_fn(*args)
        jax.block_until_ready(result)
        total_step_seconds = time.perf_counter() - t0

    gz_files = list(trace_dir.rglob("*.trace.json.gz"))
    if not gz_files:
        raise RuntimeError(f"no *.trace.json.gz produced under {trace_dir}")
    with gzip.open(gz_files[0], "rt") as fh:
        trace_json = json.load(fh)
    events = trace_json["traceEvents"]

    scopes_raw = parse_scopes(events, scope_map)
    # Fallback per P4's spec: any P2 (or pre-existing) label absent from the
    # executed trace is recorded as scopes[label] = None, not silently
    # omitted -- a missing label is a recorded outcome, never a silent zero.
    scopes: dict[str, tuple[float, int] | None] = {
        label: scopes_raw.get(label) for label in KNOWN_LABELS
    }
    # attribution_from_scopes returns {} (never None) when every label is
    # absent -- "trace captured, all labels missing" must stay distinguishable
    # from "no trace captured" (scopes=None). See #4330.
    attribution_method = attribution_from_scopes(scopes)

    dispatch_counts = parse_dispatch_counts(events, fn_name="_batched")
    metrics: dict[str, float] = {
        "total_step_seconds": total_step_seconds,
        **dispatch_counts,
    }

    return ProbeRecord(
        probe_id=f"stage1_cpu_micro_nsteps{n_steps}",
        stage=1,
        n_atoms=64,
        platform="cpu",
        metrics=metrics,
        scopes=scopes,
        attribution_method=attribution_method,
        config={"n_steps": str(n_steps), "replicas": "1"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument(
        "--out-dir", default=str(_ROOT / "outputs" / "profiling" / "stage1")
    )
    parser.add_argument(
        "--trace-dir", default=None, help="Defaults to a subdir of --out-dir"
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    trace_dir = Path(args.trace_dir) if args.trace_dir else out_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    record = _probe(args.n_steps, trace_dir)

    out_path = out_dir / f"{record.probe_id}.json"
    record.write(out_path)
    print(f"wrote {out_path}")
    recovered = [k for k, v in record.scopes.items() if v is not None] if record.scopes else []
    print(f"scopes recovered (non-None): {recovered}")
    dispatch_only = {k: v for k, v in record.metrics.items() if k != "total_step_seconds"}
    print(f"dispatch metrics: {dispatch_only}")
    print(f"total_step_seconds: {record.metrics['total_step_seconds']:.6f}")

    # Gate smoke assertion: TERM_RANKING must raise with the STAGE-FLOOR
    # message on Stage-1 records -- unlike P3 (whose records carry neither
    # total_step_seconds nor scopes and hit the metric/scopes filter first),
    # Stage-1 records DO carry total_step_seconds and (given >=2 recovered
    # labels) >=2 non-None scopes, so select_sources' fail-closed filter
    # passes and the stage>=2 floor is what actually fires. This is where
    # the stage rule is demonstrated on real records (P3's gate cannot
    # reach guard (a) at all).
    try:
        assert_claim_supported([record], ClaimClass.TERM_RANKING)
    except ClaimValidityError as exc:
        message = str(exc)
        assert "stage>=2" in message, (
            f"expected the stage floor to fire on this Stage-1 record, got: {message}"
        )
        print(f"smoke check OK: TERM_RANKING raised with the stage floor: {message}")
    else:
        raise AssertionError(
            "assert_claim_supported([record], ClaimClass.TERM_RANKING) was "
            "expected to raise (stage floor) on a Stage-1 record"
        )


if __name__ == "__main__":
    main()
