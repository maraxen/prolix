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
  # Must match the module name passed to spec_from_file_location.
  sys.modules["prolix_vs_openmm_speed"] = mod
  spec.loader.exec_module(mod)
  return mod


def test_openmm_platform_order_is_nonempty_tuple():
  mod = _load_bench()
  assert isinstance(mod.OPENMM_PLATFORM_ORDER, tuple)
  assert len(mod.OPENMM_PLATFORM_ORDER) > 0
  assert "CUDA" in mod.OPENMM_PLATFORM_ORDER
  assert mod.OPENMM_PLATFORM_ORDER.index("CUDA") < mod.OPENMM_PLATFORM_ORDER.index("Reference")


def test_timing_row_dataclass_fields():
  mod = _load_bench()
  row = mod.TimingRow(
    engine="Prolix",
    backend="jax",
    n_atoms=100,
    mean_ms=1.0,
    std_ms=0.1,
    calls_per_s=1000.0,
  )
  assert row.engine == "Prolix"
  assert row.n_atoms == 100
  assert row.vram_gb == 0.0


def test_build_openmm_system_smoke():
  mod = _load_bench()
  omm_system = mod.build_openmm_system(n_atoms=4, box_angstrom=50.0)
  assert omm_system.getNumParticles() == 4
