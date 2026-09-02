#!/usr/bin/env python3
"""Dump exclusive time by executed HLO op from a jax.profiler Perfetto gzip.

Discovery only (ungated). Use when named_scope recovery is a few percent of
host total_step_seconds — GPU command buffers hide kernels as one thunk.

  PYTHONPATH=. uv run python scripts/explore/prof_dump_hlo_ops.py TRACE.json.gz \\
      --out outputs/profiling/stage2/hlo_op_dump.json --top 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xtrax.profiling.trace import parse_hlo_op_times  # noqa: E402


def load_events(path: Path) -> list[dict]:
    with gzip.open(path, "rt") if path.suffix == ".gz" or path.name.endswith(".gz") else path.open() as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "traceEvents" in payload:
        return payload["traceEvents"]
    if isinstance(payload, list):
        return payload
    raise SystemExit(f"{path}: expected Perfetto JSON with traceEvents")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top", type=int, default=30)
    p.add_argument(
        "--host-step-seconds",
        type=float,
        default=None,
        help="optional host total_step_seconds to stamp as a ratio",
    )
    args = p.parse_args(argv)
    events = load_events(args.trace)
    times = parse_hlo_op_times(events)
    ranked = sorted(times.items(), key=lambda kv: -kv[1][0])
    total_hlo = sum(s for s, _ in times.values())
    rows = []
    for op, (seconds, n) in ranked[: args.top]:
        row = {
            "hlo_op": op,
            "exclusive_seconds": seconds,
            "n_occurrences": n,
            "pct_of_hlo_ops": (seconds / total_hlo) if total_hlo else None,
        }
        if args.host_step_seconds:
            row["pct_of_host_step"] = seconds / args.host_step_seconds
        rows.append(row)
    payload = {
        "trace": str(args.trace),
        "n_distinct_hlo_ops": len(times),
        "sum_hlo_op_seconds": total_hlo,
        "host_step_seconds": args.host_step_seconds,
        "top": rows,
        "note": (
            "If the top row is command_buffer_*, named_scope cannot pierce "
            "the CUDA graph. Diagnostic replay: "
            "XLA_FLAGS=--xla_gpu_enable_command_buffer= (empty). That is a "
            "different program (P5); compare host step on vs off."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
