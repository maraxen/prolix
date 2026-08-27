"""Plant campaign E in the bth-sync catalog and make campaign resolve read-only.

Cluster array tasks were:
1. Writing cool-tier parquet to ~/.bth/catalog, while `bth sync` rsyncs
   ~/projects/prolix/.bth/catalog/runs/prolix/.
2. Taking an exclusive DuckDB lock on bathos.db at --campaign resolve.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

CID = "bd1f47e8-46c5-40ba-abaf-20bf77d05a67"
QUESTION = (
    "On real EnsemblePlan.run inference steps, what are P_flash / P_nl / "
    "P_dense costs, and does NL survive init+step?"
)
STARTED = "2026-08-21T13:17:17.643957+00:00"

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

_OLD = """        campaign_db = duckdb.connect(str(db_path))
        try:
            resolved_campaign_id = _resolve_campaign_id(campaign_db, campaign_id)
"""
_NEW = """        campaign_db = duckdb.connect(str(db_path), read_only=True)
        try:
            resolved_campaign_id = _resolve_campaign_id(campaign_db, campaign_id)
"""


def _seed_catalog(catalog: Path) -> None:
    catalog.mkdir(parents=True, exist_ok=True)
    (catalog / "runs" / "prolix").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(catalog / "bathos.db"))
    con.execute(_CAMPAIGNS_TABLE_SCHEMA)
    row = con.execute("SELECT id FROM campaigns WHERE id = ?", [CID]).fetchone()
    if row is None:
        con.execute(
            "INSERT INTO campaigns (id, project_slug, name, mode, question, "
            "hypothesis, status, started_at) VALUES (?, ?, ?, ?, ?, NULL, 'open', ?)",
            [CID, "prolix", "full-md-path-matrix", "exploration", QUESTION, STARTED],
        )
        action = "inserted"
    else:
        action = "exists"
    print(f"catalog={catalog} campaign={action} id={CID}")


def _patch_runner(path: Path) -> None:
    if not path.is_file():
        print(f"runner_missing {path}")
        return
    text = path.read_text()
    if "duckdb.connect(str(db_path), read_only=True)" in text and "resolved_campaign_id" in text:
        print(f"runner_already_patched {path}")
        return
    if _OLD not in text:
        print(f"runner_pattern_missing {path}")
        return
    path.write_text(text.replace(_OLD, _NEW, 1))
    print(f"runner_patched {path}")


def main() -> int:
    _seed_catalog(Path.home() / "projects" / "prolix" / ".bth" / "catalog")
    # Keep the home catalog in sync so older jobs still resolve the campaign.
    _seed_catalog(Path.home() / ".bth" / "catalog")
    for runner in (
        Path.home() / "projects" / "bathos" / "src" / "bathos" / "runner.py",
        Path("/orcd/pool/008/so3_shared/marielle/projects/bathos/src/bathos/runner.py"),
    ):
        _patch_runner(runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
