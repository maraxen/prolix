"""Unit tests for B1-AOT-RATIO / B1-FINITE-GATE harness helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "b1_init_exec",
    _ROOT / "scripts" / "benchmarks" / "b1_init_exec.py",
)
assert _SPEC is not None and _SPEC.loader is not None
b1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(b1)


def _traj(pos):
    return SimpleNamespace(positions=jnp.asarray(pos))


def test_aot_ratio_is_compile_over_total():
    """Prereg H0: aot_ratio = t_aot_compile / t_total (not cold-warm)."""
    t_aot, t_total = 15.0, 2755.0
    ratio = t_aot / t_total
    assert ratio < 0.5
    # Old buggy metric would fail:
    t_cold, t_warm = 15.0, 3.0
    old = (t_cold - t_warm) / t_cold
    assert old > 0.5


def test_finite_stats_fraction():
    ok = _traj([[[1.0, 2.0, 3.0]]])
    bad = _traj([[[float("nan"), 0.0, 0.0]]])
    all_fin, frac, n_fin, n_sys = b1._last_positions_finite_stats([ok, ok, bad, ok])
    assert all_fin is False
    assert n_sys == 4
    assert n_fin == 3
    assert frac == pytest.approx(0.75)
    assert frac < b1.FINITE_FRACTION_MIN


def test_finite_fraction_min_passes_diag_17905437():
    """5/64 nonfinite → 59/64 = 0.921875 ≥ 0.9; 6/64 fails."""
    assert (59 / 64) >= b1.FINITE_FRACTION_MIN
    assert (58 / 64) >= b1.FINITE_FRACTION_MIN  # 0.90625 still passes
    assert (57 / 64) < b1.FINITE_FRACTION_MIN
