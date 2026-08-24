"""Idempotently insert a campaign row into this host's bathos catalog.

`bth sync` only rsyncs cool-tier run fragments, not `bathos.db`, so a campaign
created on a laptop is invisible to `bth run --campaign` on Engaging. This
script plants the same UUID on the cluster catalog.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

_CAMPAIGNS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    project_slug TEXT NOT NULL,
    name TEXT NOT NULL,
    mode TEXT NOT NULL,
    question TEXT,
    hypothesis TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    concluded_at TEXT,
    conclusion TEXT,
    outcome_label TEXT,
    parent_campaign_id TEXT,
    stopping_threshold REAL,
    claim_path TEXT,
    claim_sha256 TEXT,
    claim_mode TEXT,
    negative_check TEXT
)
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--project-slug", default="prolix")
    p.add_argument("--question", default=None)
    p.add_argument("--started-at", required=True)
    p.add_argument(
        "--catalog",
        default="",
        help="Override catalog dir (default: ~/.bth/catalog)",
    )
    p.add_argument("--out", default="")
    args = p.parse_args()

    catalog = Path(args.catalog).expanduser() if args.catalog else Path.home() / ".bth" / "catalog"
    catalog.mkdir(parents=True, exist_ok=True)
    db_path = catalog / "bathos.db"
    con = duckdb.connect(str(db_path))
    con.execute(_CAMPAIGNS_TABLE_SCHEMA)
    row = con.execute("SELECT id, name, status FROM campaigns WHERE id = ?", [args.id]).fetchone()
    if row is None:
        con.execute(
            "INSERT INTO campaigns (id, project_slug, name, mode, question, hypothesis, "
            "status, started_at) VALUES (?, ?, ?, ?, ?, NULL, 'open', ?)",
            [args.id, args.project_slug, args.name, args.mode, args.question, args.started_at],
        )
        action = "inserted"
    else:
        action = f"exists name={row[1]} status={row[2]}"
    check = con.execute("SELECT id FROM campaigns WHERE id = ?", [args.id]).fetchone()
    line = f"action={action} id={check[0]} catalog={db_path}"
    print(line)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
