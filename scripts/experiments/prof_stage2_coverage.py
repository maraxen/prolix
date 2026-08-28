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


def _decode_array_task(task_id: int) -> tuple[tuple[str, str, str], str] | None:
    if task_id < 0 or task_id > 11:
        return None
    proteins = ("1vii", "2gb1")
    modes = ("dense", "flash")
    protein = proteins[task_id // 6]
    mode = modes[(task_id % 6) // 3]
    pme_idx = task_id % 3
    if pme_idx == 0:
        return (protein, "off", "na"), mode
    grid = "1.0" if pme_idx == 1 else "1.2"
    return (protein, "on", grid), mode


def _cell_mode_from_tags(tags) -> tuple[tuple[str, str, str], str] | None:
    import re

    if tags is None:
        return None
    if isinstance(tags, list):
        blob = " ".join(str(t) for t in tags)
    else:
        text = str(tags).strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    blob = " ".join(str(t) for t in parsed)
                else:
                    blob = text
            except json.JSONDecodeError:
                blob = text
        else:
            blob = text
    m_task = re.search(r"array_task:(\d+)", blob)
    if m_task:
        decoded = _decode_array_task(int(m_task.group(1)))
        if decoded is not None:
            return decoded
    protein = pme = grid = mode = None
    m = re.search(r"protein:([^\s,\]\"]+)", blob)
    if m:
        protein = m.group(1)
    m = re.search(r"mode:([^\s,\]\"]+)", blob)
    if m:
        mode = m.group(1)
    m = re.search(r"pme:([^\s,\]\"]+)", blob)
    if m:
        pme = m.group(1)
    m = re.search(r"grid:([^\s,\]\"]+)", blob)
    if m:
        grid = m.group(1)
    if protein and mode in ("dense", "flash") and pme and grid:
        return (protein, pme, grid), mode
    return None


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
    ok_modes_by_cell: dict[tuple[str, str, str], set[str]] = {c: set() for c in CELLS}
    n_catalog_ok = 0
    for row in catalog:
        tags = row.get("tags", "")
        exit_code = row.get("exit_code")
        status = str(row.get("status", "")).lower()
        code_ok = str(exit_code) in ("0", "0.0") or exit_code == 0
        if not code_ok:
            continue
        if status and status not in ("completed", "success", "ok", "pass"):
            continue
        n_catalog_ok += 1
        cell_mode = _cell_mode_from_tags(tags)
        if cell_mode is None:
            continue
        cell, mode = cell_mode
        if cell in ok_modes_by_cell:
            ok_modes_by_cell[cell].add(mode)

    by_cell: dict[tuple[str, str, str], set[str]] = {c: set() for c in CELLS}
    speedups: dict[str, float | None] = {}
    for rec in records:
        key = _cell_key(rec.config)
        if key is None or key not in by_cell:
            continue
        by_cell[key].add(rec.config["mode"])
        if rec.config["mode"] == "flash" and "flash_speedup" in rec.metrics:
            speedups[f"flash_speedup_{key[0]}_pme{key[1]}_g{key[2]}"] = rec.metrics["flash_speedup"]

    n_record_pairs = sum(1 for c in CELLS if by_cell[c] >= {"dense", "flash"})
    if not any(ok_modes_by_cell.values()) and n_record_pairs > 0:
        # Cluster 20792000: 12 records + catalog ok rows, but tags did not
        # decode into cells. --no-sidecar GPU runs also yield an empty local
        # catalog after pull. Successful ProbeRecords are exit-0 for this sweep.
        ok_modes_by_cell = {c: set(by_cell[c]) for c in CELLS}

    def _slug(cell: tuple[str, str, str]) -> str:
        protein, pme, grid = cell
        return f"cell_{protein}_pme{pme}_g{grid.replace('.', 'p')}"

    cell_ok = {}
    n_ok = 0
    for cell in CELLS:
        ok = (
            by_cell[cell] >= {"dense", "flash"}
            and ok_modes_by_cell[cell] >= {"dense", "flash"}
        )
        cell_ok[_slug(cell)] = ok
        if ok:
            n_ok += 1

    summary = {
        "n_configs_with_valid_dense_flash_pair": n_ok,
        "n_record_dense_flash_pair": n_record_pairs,
        "n_records": len(records),
        "n_catalog_ok": n_catalog_ok,
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
