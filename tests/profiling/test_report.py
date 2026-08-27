"""P8 report.py: claim-gated bottleneck table over committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.profiling.claims import ClaimValidityError
from scripts.profiling.record import ProbeRecord

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "records"


def _paths(*names: str) -> list[Path]:
    return [FIXTURE_DIR / n for n in names]


def test_every_committed_fixture_roundtrips_probe_record_read():
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    assert paths, f"expected committed fixtures under {FIXTURE_DIR}"
    for path in paths:
        rec = ProbeRecord.read(path)
        assert rec.probe_id
        ProbeRecord.from_json(rec.to_json())


def test_report_over_stage0_and_stage1_fixtures_raises_on_term_ranking():
    from scripts.profiling.report import render_report

    with pytest.raises(ClaimValidityError, match="TERM_RANKING"):
        render_report(_paths("stage0_structural.json", "stage1_cpu_micro.json"))


def test_report_over_stage0_stage1_and_stage2_fixtures_emits_ranking_table():
    from scripts.profiling.report import render_report

    text = render_report(
        _paths(
            "stage0_structural.json",
            "stage1_cpu_micro.json",
            "stage2_named_scope.json",
        )
    )
    header = (
        "| scope | exclusive_seconds | n_occurrences | pct_of_total | "
        "stage | n_atoms | platform | device_kind | attribution_method | probe_id |"
    )
    assert header in text
    assert "pme_fft_forward" in text
    assert "flash_nonbonded_tiles" in text
    assert "MIXED ATTRIBUTION" not in text
    assert "contract_version=" in text
    assert "git_sha=" in text
    assert "xla_flags=" in text


def test_mixed_attribution_fixture_set_emits_banner():
    from scripts.profiling.report import render_report

    text = render_report(
        _paths("stage2_named_scope.json", "stage2_op_name.json")
    )
    assert (
        "> MIXED ATTRIBUTION: this ranking combines named_scope and op_name "
        "attribution; per-row method is in the attribution_method column."
    ) in text


def test_single_method_fixture_set_omits_mixed_banner():
    from scripts.profiling.report import render_report

    text = render_report(_paths("stage2_named_scope.json"))
    assert "MIXED ATTRIBUTION" not in text


def test_none_scope_renders_as_absent_not_zero():
    from scripts.profiling.report import render_report

    text = render_report(_paths("stage2_named_scope.json"))
    assert "absent" in text
    for line in text.splitlines():
        if "dense_bonded_bond" in line:
            assert "0.0" not in line.split("|")[2]
            assert "absent" in line
