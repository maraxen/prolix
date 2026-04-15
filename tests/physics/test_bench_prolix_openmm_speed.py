"""Smoke tests for ``scripts/benchmarks/prolix_vs_openmm_speed.py`` (import + helpers)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "scripts" / "benchmarks" / "prolix_vs_openmm_speed.py"


def _load_bench():
  spec = importlib.util.spec_from_file_location("prolix_vs_openmm_speed", _BENCH)
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  # Required so dataclasses can resolve TimingRow.__module__ during import.
  sys.modules["prolix_vs_openmm_speed"] = mod
  spec.loader.exec_module(mod)
  return mod


def test_list_openmm_platform_names_returns_list():
  mod = _load_bench()
  names = mod.list_openmm_platform_names()
  assert isinstance(names, list)
  for n in names:
    assert isinstance(n, str)
    assert len(n) > 0


def test_iter_openmm_platforms_in_order_is_subset():
  mod = _load_bench()
  have = set(mod.list_openmm_platform_names())
  ordered = mod.iter_openmm_platforms_in_order()
  assert set(ordered) == have
  if "CUDA" in have and "Reference" in have:
    assert ordered.index("CUDA") < ordered.index("Reference")


def test_minimal_protein_dict_has_radii():
  mod = _load_bench()
  import numpy as np

  d = mod._minimal_protein_dict(np.array([1.0, -1.0]))
  assert "radii" in d and "scaled_radii" in d
