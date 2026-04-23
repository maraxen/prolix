"""Unit tests for ``scripts/benchmarks/tip3p_ke_gates.py`` (P2a-B2-R / P2a-B2-X)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tip3p_ke_compare"
_FIXTURES_AGG = Path(__file__).resolve().parents[1] / "fixtures" / "tip3p_tightening_aggregate"
_GATES_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "tip3p_ke_gates.py"
_AGGREGATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "aggregate_tip3p_tightening_logs.py"


def _gates():
  spec = importlib.util.spec_from_file_location("tip3p_ke_gates", _GATES_PATH)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _load(name: str) -> dict:
  return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _load_agg(name: str) -> dict:
  return json.loads((_FIXTURES_AGG / name).read_text(encoding="utf-8"))


def _aggregate_mod():
  spec = importlib.util.spec_from_file_location("aggregate_tip3p_tightening_logs", _AGGREGATE_PATH)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_tip3p_ke_gates_constants() -> None:
  g = _gates()
  assert g.P2A_B2_R_TOL_REL == pytest.approx(0.0167)


def test_p2a_b2_r_x_pass_fixture() -> None:
  g = _gates()
  out = g.evaluate_payload(_load("p2a_b2_r_x_pass.json"))
  assert out["p2a_b2_r_both_pass"] is True
  assert out["p2a_b2_x"]["pass"] is True


def test_p2a_b2_r_fail_openmm_fixture() -> None:
  g = _gates()
  out = g.evaluate_payload(_load("p2a_b2_r_fail_openmm.json"))
  assert out["openmm"]["p2a_b2_r_pass"] is False
  assert out["prolix"]["p2a_b2_r_pass"] is True


def test_skip_error_row_before_success() -> None:
  g = _gates()
  out = g.evaluate_payload(_load("error_then_success_shape.json"))
  assert out["p2a_b2_r_both_pass"] is True


def test_find_engine_summary_missing_raises() -> None:
  g = _gates()
  bad = {"meta": {"benchmark_policy": {"temperature_k": 300.0}}, "config": {"temperature_k": 300.0}, "diagnostics": []}
  with pytest.raises(ValueError, match="no successful diagnostics"):
    g.evaluate_payload(bad)


def test_bath_temperature_mismatch_raises() -> None:
  g = _gates()
  p = _load("p2a_b2_r_x_pass.json")
  p["config"]["temperature_k"] = 299.0
  with pytest.raises(ValueError, match="temperature_k mismatch"):
    g.evaluate_payload(p)


def test_tightening_aggregate_schema_constant() -> None:
  g = _gates()
  assert g.TIGHTENING_AGGREGATE_SCHEMA == "tip3p_tightening_aggregate/v1"
  assert g.TIGHTENING_AGGREGATE_SCHEMA_V2 == "tip3p_tightening_aggregate/v2"


def test_evaluate_tightening_aggregate_pass_fixture() -> None:
  g = _gates()
  out = g.evaluate_tightening_aggregate(_load_agg("pass_r_both_x.json"))
  assert out["p2a_b2_r_both_pass"] is True
  assert out["p2a_b2_x"]["pass"] is True
  assert out["schema"] == g.TIGHTENING_AGGREGATE_SCHEMA


def test_evaluate_tightening_aggregate_fail_r_prolix() -> None:
  g = _gates()
  out = g.evaluate_tightening_aggregate(_load_agg("fail_r_prolix.json"))
  assert out["openmm"]["p2a_b2_r_pass"] is True
  assert out["prolix"]["p2a_b2_r_pass"] is False
  assert out["p2a_b2_r_both_pass"] is False


def test_evaluate_tightening_wrong_schema_raises() -> None:
  g = _gates()
  p = _load_agg("pass_r_both_x.json")
  p["schema"] = "tip3p_tightening_aggregate/v0"
  with pytest.raises(ValueError, match="expected schema"):
    g.evaluate_tightening_aggregate(p)


def test_evaluate_tightening_aggregate_v2_pass_fixture() -> None:
  g = _gates()
  out = g.evaluate_tightening_aggregate(_load_agg("v2_pass_r_both_x.json"))
  assert out["p2a_b2_r_both_pass"] is True
  assert out["schema"] == g.TIGHTENING_AGGREGATE_SCHEMA_V2


def test_evaluate_tightening_aggregate_v2_missing_optional_diag() -> None:
  g = _gates()
  out = g.evaluate_tightening_aggregate(_load_agg("v2_missing_optional_diag.json"))
  assert out["p2a_b2_r_both_pass"] is False
  assert out["prolix"]["p2a_b2_r_pass"] is False


def test_evaluate_tightening_mean_mismatch_raises() -> None:
  g = _gates()
  with pytest.raises(ValueError, match="mean_T_K_mean"):
    g.evaluate_tightening_aggregate(_load_agg("mean_mismatch.json"))


def test_main_tightening_aggregate_exit_codes() -> None:
  g = _gates()
  assert g.main([str(_FIXTURES_AGG / "pass_r_both_x.json")]) == 0
  assert g.main([str(_FIXTURES_AGG / "fail_r_prolix.json")]) == 1


def test_main_compare_fixture_exit_code() -> None:
  g = _gates()
  assert g.main([str(_FIXTURES / "p2a_b2_r_x_pass.json")]) == 0


def test_aggregate_script_outputs_schema_and_gates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  block = {
    "replica_id": 0,
    "runs": [
      {
        "engine": "openmm",
        "mean_T_K": 300.0,
        "std_T_K": 0.1,
        "target_T_K": 300.0,
        "profile_id": "openmm_ref_linear_com_on",
      },
      {
        "engine": "prolix",
        "mean_T_K": 300.0,
        "std_T_K": 0.1,
        "target_T_K": 300.0,
        "profile_id": "openmm_ref_linear_com_on",
      },
    ],
  }
  log = tmp_path / "tip3p_tightening_r0_99.log"
  log.write_text("warning line\n" + json.dumps(block) + "\n", encoding="utf-8")
  agg = _aggregate_mod()
  assert agg.main([str(log)]) == 0
  raw = capsys.readouterr().out
  payload = json.loads(raw)
  assert payload["schema"] == agg.TIGHTENING_AGGREGATE_SCHEMA
  assert payload["meta"]["benchmark_policy"]["temperature_k"] == 300.0
  g = _gates()
  out = g.evaluate_tightening_aggregate(payload)
  assert out["p2a_b2_r_both_pass"] is True


def test_aggregate_inconsistent_target_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  b0 = {
    "replica_id": 0,
    "runs": [
      {"engine": "openmm", "mean_T_K": 300.0, "std_T_K": 0.1, "target_T_K": 300.0, "profile_id": "x"},
      {"engine": "prolix", "mean_T_K": 300.0, "std_T_K": 0.1, "target_T_K": 300.0, "profile_id": "x"},
    ],
  }
  b1 = {
    "replica_id": 1,
    "runs": [
      {"engine": "openmm", "mean_T_K": 300.0, "std_T_K": 0.1, "target_T_K": 310.0, "profile_id": "x"},
      {"engine": "prolix", "mean_T_K": 300.0, "std_T_K": 0.1, "target_T_K": 310.0, "profile_id": "x"},
    ],
  }
  p0 = tmp_path / "tip3p_tightening_r0_a.log"
  p1 = tmp_path / "tip3p_tightening_r1_a.log"
  p0.write_text(json.dumps(b0), encoding="utf-8")
  p1.write_text(json.dumps(b1), encoding="utf-8")
  agg = _aggregate_mod()
  assert agg.main([str(p0), str(p1)]) == 2
  assert "inconsistent" in capsys.readouterr().err.lower()
