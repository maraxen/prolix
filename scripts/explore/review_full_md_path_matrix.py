#!/usr/bin/env python3
"""Campaign E review: P_nl vs P_flash. Lesson 379 — do not default Flash from a race."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "outputs" / "profiling" / "full_md_path_matrix"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=MATRIX)
    args = p.parse_args()
    if not args.dir.is_dir():
        print(f"no matrix dir yet: {args.dir}")
        print("investigation gated on campaign E cells; NL cells already emit k_over_n")
        return 0
    rows = []
    for summary in sorted(args.dir.glob("*/summary.json")):
        data = json.loads(summary.read_text())
        rows.append(data)
        print(
            f"{data.get('system')} {data.get('nonbonded')} buckets={data.get('use_size_buckets')} "
            f"status={data.get('status')} s/step={data.get('per_step_corrected_s')} "
            f"k/n={data.get('k_over_n')}"
        )
    by = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        by.setdefault(r["system"], {})[r["nonbonded"]] = r.get("per_step_corrected_s")
    for system, costs in by.items():
        flash = costs.get("flash")
        nl = costs.get("nl")
        if flash is None or nl is None:
            continue
        if nl > flash:
            print(
                f"INVESTIGATE {system}: P_nl={nl} slower than P_flash={flash}. "
                "Do not default flash. Check k_over_n, rebuild_every, init_neighbor_bound."
            )
        else:
            print(f"{system}: P_nl={nl} <= P_flash={flash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
