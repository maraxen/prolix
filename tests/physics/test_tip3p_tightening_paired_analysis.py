"""Tests for ``scripts/benchmarks/tip3p_tightening_paired_analysis.py``."""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

_SCRIPT_PATH = (
  Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "tip3p_tightening_paired_analysis.py"
)
_FIXTURES_AGG = Path(__file__).resolve().parents[1] / "fixtures" / "tip3p_tightening_aggregate"


def _mod():
  spec = importlib.util.spec_from_file_location("tip3p_tightening_paired_analysis", _SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _load(name: str) -> dict:
  return json.loads((_FIXTURES_AGG / name).read_text(encoding="utf-8"))


def test_paired_analysis_schema_and_advancement_flag() -> None:
  m = _mod()
  incumbent = _load("fail_r_prolix.json")
  candidate = _load("pass_r_both_x.json")
  for row in candidate["aggregated"]:
    row["per_replica_mean_T_K"] = row["per_replica_mean_T_K"][:3]
    row["per_replica_std_T_K"] = row["per_replica_std_T_K"][:3]
    row["n_replicas"] = 3
    row["mean_T_K_mean"] = float(sum(row["per_replica_mean_T_K"]) / 3.0)
    row["mean_T_K_pstdev_across_replicas"] = float(statistics.pstdev(row["per_replica_mean_T_K"]))
  out = m.analyze(incumbent, candidate)
  assert out["schema"] == "tip3p_tightening_paired_analysis/v1"
  assert out["source"]["n_pairs"] == 3
  assert out["advancement"]["passes"] is True
  assert out["metrics"]["M1_prolix_r_relative_error_reduction"]["summary"]["lower_95_one_sided"] > 0.0


def test_paired_analysis_includes_m3_m4_m5_when_present() -> None:
  m = _mod()
  incumbent = _load("pass_r_both_x.json")
  candidate = _load("pass_r_both_x.json")
  for row in incumbent["aggregated"]:
    if row["engine"] == "prolix":
      row["per_replica_diag_projection_residual_p95"] = [0.20, 0.22, 0.21, 0.20, 0.19]
      row["per_replica_diag_bond_residual_max_abs_p95"] = [0.010, 0.011, 0.010, 0.009, 0.010]
      row["per_replica_diag_com_metric_p95"] = [0.030, 0.029, 0.031, 0.030, 0.030]
  for row in candidate["aggregated"]:
    if row["engine"] == "prolix":
      row["per_replica_diag_projection_residual_p95"] = [0.15, 0.16, 0.15, 0.16, 0.15]
      row["per_replica_diag_bond_residual_max_abs_p95"] = [0.008, 0.008, 0.009, 0.008, 0.008]
      row["per_replica_diag_com_metric_p95"] = [0.020, 0.021, 0.020, 0.021, 0.020]
  out = m.analyze(incumbent, candidate)
  assert out["metrics"]["M3_projection_residual_reduction"]["summary"] is not None
  assert out["metrics"]["M4_bond_residual_reduction"]["summary"] is not None
  assert out["metrics"]["M5_com_metric_reduction"]["summary"] is not None
  assert out["metrics"]["M3_projection_residual_reduction"]["summary"]["mean"] > 0.0


def test_paired_analysis_v2_accepted() -> None:
  m = _mod()
  incumbent = _load("v2_missing_optional_diag.json")
  candidate = _load("v2_pass_r_both_x.json")
  for row in candidate["aggregated"]:
    row["per_replica_mean_T_K"] = row["per_replica_mean_T_K"][:3]
    row["per_replica_std_T_K"] = row["per_replica_std_T_K"][:3]
    row["n_replicas"] = 3
    row["mean_T_K_mean"] = float(sum(row["per_replica_mean_T_K"]) / 3.0)
    row["mean_T_K_pstdev_across_replicas"] = float(statistics.pstdev(row["per_replica_mean_T_K"]))
  out = m.analyze(incumbent, candidate)
  assert out["source"]["incumbent_schema"] == "tip3p_tightening_aggregate/v2"
  assert out["source"]["candidate_schema"] == "tip3p_tightening_aggregate/v2"
  assert out["metrics"]["M3_projection_residual_reduction"]["summary"] is None


def test_bath_mismatch_raises() -> None:
  m = _mod()
  incumbent = _load("pass_r_both_x.json")
  candidate = _load("pass_r_both_x.json")
  candidate["meta"]["benchmark_policy"]["temperature_k"] = 310.0
  with pytest.raises(ValueError, match="bath temperature mismatch"):
    m.analyze(incumbent, candidate)

