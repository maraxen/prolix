"""Claim 2 W4 (#278): browser smoke demo — 100-step trace + static HTML.

Builds a self-contained ``index.html`` with energy/temperature charts from the
export diagnostics path (TIP3P one-water solvated smoke fixture).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from prolix.api.browser_demo import build_browser_smoke_demo

_b1_path = Path(__file__).resolve().parent.parent / "bench" / "test_b1_smoke.py"
_spec = importlib.util.spec_from_file_location("test_b1_smoke", _b1_path)
assert _spec and _spec.loader
_b1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b1)
_make_bundle = _b1._make_bundle

W4_N_STEPS = 100


@pytest.mark.fast
def test_w4_browser_smoke_demo_trace_and_html(tmp_path):
    """100-step solvated smoke run produces finite traces and renderable HTML."""
    bundle = _make_bundle(10, seed=278)
    meta = build_browser_smoke_demo(
        tmp_path,
        bundle,
        n_steps=W4_N_STEPS,
        dt=0.5,
        kT=0.596,
        seed=278,
        compile_wasm=False,
        system_label="B=1 smoke export (10-atom bucket, Claim 2 path)",
    )

    assert meta["n_steps"] == W4_N_STEPS
    assert meta["n_atoms"] == 10
    assert len(meta["temperatures"]) == W4_N_STEPS
    assert len(meta["energies"]) == W4_N_STEPS
    assert meta["positions_shape"] == (W4_N_STEPS, 10, 3)

    for t in meta["temperatures"]:
        assert math.isfinite(t) and t > 0.0
    for e in meta["energies"]:
        assert math.isfinite(e)

    html_path = Path(meta["html_path"])
    trace_path = Path(meta["trace_path"])
    assert html_path.is_file()
    assert trace_path.is_file()

    html = html_path.read_text()
    assert "temp-chart" in html
    assert "energy-chart" in html
    assert "trace-data" in html

    trace = json.loads(trace_path.read_text())
    assert trace["meta"]["n_steps"] == W4_N_STEPS
    assert len(trace["temperatures"]) == W4_N_STEPS


@pytest.mark.wasm
@pytest.mark.fast
def test_w4_browser_smoke_bundles_wasm_when_iree_available(tmp_path):
    """When iree-compile is present, demo also bundles trajectory.wasm under cap."""
    from prolix import export

    if export.find_iree_compile() is None:
        pytest.skip("iree-compile not found; install with: uv sync --extra wasm")

    bundle = _make_bundle(10, seed=278)
    meta = build_browser_smoke_demo(
        tmp_path,
        bundle,
        n_steps=4,
        compile_wasm=True,
    )
    assert meta["wasm_bytes"] is not None
    assert meta["wasm_bytes"] > 0
    assert Path(meta["wasm_path"]).is_file()
