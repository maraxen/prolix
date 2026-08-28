"""Coverage tag decode: JSON / DuckDB list blobs from bth sql."""

from __future__ import annotations

from scripts.experiments.prof_stage2_coverage import _cell_mode_from_tags


def test_decode_json_list_tags():
    tags = '["p7-stage2", "array_task:4", "protein:1vii", "mode:flash"]'
    cell, mode = _cell_mode_from_tags(tags)
    assert cell == ("1vii", "on", "1.0")
    assert mode == "flash"


def test_decode_python_list_tags():
    tags = ["p7-stage2", "array_task:0"]
    cell, mode = _cell_mode_from_tags(tags)
    assert cell == ("1vii", "off", "na")
    assert mode == "dense"
