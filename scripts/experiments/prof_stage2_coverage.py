#!/usr/bin/env python3
"""P7 post-array coverage aggregator. Gates coverage, not flash_speedup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CELLS = [
    ("1vii", "off", "na"),
    ("1vii", "on", "1.0"),
    ("1vii", "on", "1.2"),
    ("2gb1", "off", "na"),
    ("2gb1", "on", "1.0"),
    ("2gb1", "on", "1.2"),
]


def _bth_sql(campaign_tag: str) -> list[dict]:
    sql = (
        "SELECT id, status, exit_code, tags FROM "
        "read_parquet('~/.bth/catalog/runs/prolix/run_*.parquet') "
        f"WHERE list_contains(tags, '{campaign_tag}')"
    )
    proc = subprocess.run(
        ["uv", "run", "bth", "sql", "--json", sql],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        proc = subprocess.run(
            ["uv", "run", "bth", "sql", sql],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(f"bathos catalog unreachable: {proc.stderr or proc.stdout}")
    text = proc.stdout.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append({
            "id": parts[0],
            "status": parts[1],
            "exit_code": parts[2],
            "tags": parts[3] if len(parts) > 3 else "",
        })
    return rows


def _cell_key(cfg: dict) -> tuple[str, str, str] | None:
    if cfg.get("scopes") == "off":
        return None
    protein = cfg.get("protein")
    pme = cfg.get("pme")
    grid = cfg.get("grid_spacing")
    mode = cfg.get("mode")
    if protein is None or pme is None or grid is None or mode not in ("dense", "flash"):
        return None
    return protein, pme, grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "outputs" / "profiling" / "stage2")
    parser.add_argument("--campaign-tag", default="p7-stage2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from scripts.profiling.record import ProbeRecord

    paths = sorted(args.records_dir.glob("*.json"))
    records = []
    for path in paths:
        if path.name.endswith("_summary.json") or path.parent.name == "scope_ab":
            continue
        try:
            records.append(ProbeRecord.read(path))
        except Exception:
            continue

    if records:
        flags = {r.xla_flags for r in records}
        if len(flags) > 1:
            raise SystemExit(f"xla_flags not unanimous: {sorted(map(repr, flags))}")

    catalog = _bth_sql(args.campaign_tag)
    ok_tasks = set()
    for row in catalog:
        tags = row.get("tags", "")
        exit_code = row.get("exit_code")
        status = str(row.get("status", ""))
        code_ok = str(exit_code) in ("0", "0.0") or exit_code == 0
        if not code_ok or status not in ("completed", "success", ""):
            continue
        if isinstance(tags, list):
            tag_list = tags
        else:
            tag_list = str(tags)
        ok_tasks.add(str(row.get("id")))

    by_cell: dict[tuple[str, str, str], set[str]] = {c: set() for c in CELLS}
    speedups: dict[str, float | None] = {}
    for rec in records:
        key = _cell_key(rec.config)
        if key is None or key not in by_cell:
            continue
        by_cell[key].add(rec.config["mode"])
        if rec.config["mode"] == "flash" and "flash_speedup" in rec.metrics:
            speedups[f"flash_speedup_{key[0]}_pme{key[1]}_g{key[2]}"] = rec.metrics["flash_speedup"]

    def _slug(cell: tuple[str, str, str]) -> str:
        protein, pme, grid = cell
        return f"cell_{protein}_pme{pme}_g{grid.replace('.', 'p')}"

    cell_ok = {}
    n_ok = 0
    for cell in CELLS:
        ok = by_cell[cell] >= {"dense", "flash"}
        cell_ok[_slug(cell)] = ok
        if ok:
            n_ok += 1

    summary = {
        "n_configs_with_valid_dense_flash_pair": n_ok,
        "n_records": len(records),
        "n_catalog_ok": len(ok_tasks),
        **{k: int(v) for k, v in cell_ok.items()},
        **{k: v for k, v in speedups.items()},
    }
    text = json.dumps(summary, indent=2)
    print(text)
    out = args.out or (args.records_dir / "coverage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
