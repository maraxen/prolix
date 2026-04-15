"""Fast checks that benchmark PDB paths from ``scripts/run_batched_pipeline.py`` exist."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
  "run_batched_pipeline",
  _ROOT / "scripts" / "run_batched_pipeline.py",
)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
SYSTEM_CATALOG = _MOD.SYSTEM_CATALOG
load_and_parameterize = _MOD.load_and_parameterize


def test_system_catalog_pdb_files_exist() -> None:
  for key, path in SYSTEM_CATALOG.items():
    assert path.is_file(), f"SYSTEM_CATALOG[{key!r}] missing file: {path}"


@pytest.mark.slow
def test_load_and_parameterize_1x2g_catalog_smoke() -> None:
  """End-to-end parameterization (requires FF XML + proxide)."""
  pdb = SYSTEM_CATALOG["1X2G"]
  protein = load_and_parameterize(pdb)
  n = int(protein.coordinates.reshape(-1, 3).shape[0])
  assert n > 0
