"""P8 bottleneck report: assert_claim_supported, then a Markdown ranking table.

No prolix imports. Discovers ProbeRecords under outputs/profiling/stage*/*.json
or an explicit path list. Never reads a remote path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.profiling.claims import (
    CONTRACT_VERSION,
    ClaimClass,
    assert_claim_supported,
    select_sources,
)
from scripts.profiling.record import ProbeRecord

_SKIP_SUFFIXES = ("_summary.json",)
_SKIP_NAMES = {"coverage.json"}

_COLUMNS = (
    "scope",
    "exclusive_seconds",
    "n_occurrences",
    "pct_of_total",
    "stage",
    "n_atoms",
    "platform",
    "device_kind",
    "attribution_method",
    "probe_id",
)

_MIXED_BANNER = (
    "> MIXED ATTRIBUTION: this ranking combines named_scope and op_name "
    "attribution; per-row method is in the attribution_method column."
)


def discover_records(
    paths: list[Path] | None = None,
    *,
    root: Path | None = None,
) -> list[ProbeRecord]:
    """Load ProbeRecords from explicit paths or outputs/profiling/stage*/*.json."""
    if paths:
        files = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files.extend(sorted(p.glob("*.json")))
            else:
                files.append(p)
    else:
        base = (root or Path.cwd()) / "outputs" / "profiling"
        files = sorted(base.glob("stage*/*.json"))
    records: list[ProbeRecord] = []
    for path in files:
        if path.name in _SKIP_NAMES or path.name.endswith(_SKIP_SUFFIXES):
            continue
        try:
            records.append(ProbeRecord.read(path))
        except Exception:
            continue
    return records


def _row_sort_key(row: dict[str, str]) -> tuple[int, float, str]:
    raw = row["exclusive_seconds"]
    if raw == "absent":
        return (1, 0.0, row["scope"])
    return (0, -float(raw), row["scope"])


def _rows_from_sources(sources: list[ProbeRecord]) -> list[dict[str, str]]:
    attributed: list[tuple[ProbeRecord, str, float, int, str]] = []
    absent: list[tuple[ProbeRecord, str, str]] = []
    for rec in sources:
        methods = rec.attribution_method or {}
        scopes = rec.scopes or {}
        for name, value in scopes.items():
            method = methods.get(name, "")
            if value is None:
                absent.append((rec, name, method))
            else:
                exclusive, n_occ = value
                attributed.append((rec, name, float(exclusive), int(n_occ), method))
    total_exc = sum(item[2] for item in attributed)
    rows: list[dict[str, str]] = []
    for rec, name, exclusive, n_occ, method in attributed:
        pct = (100.0 * exclusive / total_exc) if total_exc > 0 else 0.0
        rows.append(
            {
                "scope": name,
                "exclusive_seconds": f"{exclusive:.12g}",
                "n_occurrences": str(n_occ),
                "pct_of_total": f"{pct:.4g}",
                "stage": str(rec.stage),
                "n_atoms": str(rec.n_atoms),
                "platform": rec.platform,
                "device_kind": rec.device_kind or "",
                "attribution_method": method,
                "probe_id": rec.probe_id,
            }
        )
    for rec, name, method in absent:
        rows.append(
            {
                "scope": name,
                "exclusive_seconds": "absent",
                "n_occurrences": "absent",
                "pct_of_total": "absent",
                "stage": str(rec.stage),
                "n_atoms": str(rec.n_atoms),
                "platform": rec.platform,
                "device_kind": rec.device_kind or "",
                "attribution_method": method,
                "probe_id": rec.probe_id,
            }
        )
    rows.sort(key=_row_sort_key)
    return rows


def _markdown_table(rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    body = [
        "| " + " | ".join(row[c] for c in _COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def render_report(paths: list[Path] | None = None, *, root: Path | None = None) -> str:
    """Assert TERM_RANKING is supported, then render the bottleneck table."""
    records = discover_records(paths, root=root)
    assert_claim_supported(records, ClaimClass.TERM_RANKING)
    sources = select_sources(records, ClaimClass.TERM_RANKING)
    rows = _rows_from_sources(sources)
    methods = {row["attribution_method"] for row in rows if row["attribution_method"]}
    parts: list[str] = []
    if methods == {"named_scope", "op_name"} or (
        "named_scope" in methods and "op_name" in methods
    ):
        parts.append(_MIXED_BANNER)
        parts.append("")
    parts.append(_markdown_table(rows))
    git_sha = sources[0].git_sha
    xla_flags = sources[0].xla_flags
    parts.append("")
    parts.append(
        f"contract_version={CONTRACT_VERSION} git_sha={git_sha} "
        f"xla_flags={xla_flags!r}"
    )
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="ProbeRecord JSON files or directories (default: outputs/profiling/stage*/*.json)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = render_report(list(args.paths) or None)
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
