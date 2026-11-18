# ruff: noqa: E402
"""Tests for `bench_v36_cupotential_ab` (`bathos` schema + `--cpu-smoke`) and `aggregate_v35_cross_arch_report`."""

from __future__ import annotations

import copy
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKTREE_ROOT) not in sys.path:
  sys.path.insert(0, str(WORKTREE_ROOT))

import scripts.benchmarks.aggregate_v35_cross_arch_report as agg_mod
import scripts.benchmarks.bench_v36_cupotential_ab as bench_v36_mod
from scripts.benchmarks.aggregate_v35_cross_arch_report import (
  ARCH_SPECS,
  REQUIRED_PROVENANCE_FIELDS,
  ProvenanceValidationError,
  aggregate_v35_reports,
  collect_provenance_records,
  render_markdown_summary,
)
from scripts.benchmarks.bench_v36_cupotential_ab import run_v36_ab_benchmark

pytestmark = pytest.mark.fast

REAL_BENCH_DIR = WORKTREE_ROOT / "docs" / "benchmarks"
REAL_L40S_JSON = REAL_BENCH_DIR / "261111_l40s_pallas_v35_release_ladder.json"
BTH_TOML_PATH = WORKTREE_ROOT / "scripts" / "benchmarks" / "bench_v36_cupotential_ab.bth.toml"
DOC_GUIDE_PATH = (
  WORKTREE_ROOT / "docs" / "source" / "guides" / "cupotential_parity_and_openmm_fastpath.md"
)


def test_bth_toml_schema_and_single_residual_outcome() -> None:
  """Verify `bench_v36_cupotential_ab.bth.toml` conforms to bathos DuckDB experiment schema."""
  assert BTH_TOML_PATH.is_file()
  cfg = tomllib.loads(BTH_TOML_PATH.read_text(encoding="utf-8"))

  assert "experiment" in cfg
  assert isinstance(cfg["experiment"].get("hypothesis"), str)
  assert len(cfg["experiment"]["hypothesis"].strip()) > 40

  outcomes = cfg.get("outcomes")
  assert isinstance(outcomes, dict)
  assert len(outcomes) >= 3

  residual_names: list[str] = []
  for name, spec in outcomes.items():
    assert isinstance(spec.get("condition"), str)
    assert spec["condition"].strip()
    assert isinstance(spec.get("decision"), str)
    assert spec["decision"].strip()
    assert isinstance(spec.get("reasoning"), str)
    assert spec["reasoning"].strip()
    if spec.get("is_residual") is True:
      residual_names.append(name)

  assert len(residual_names) == 1, f"Expected exactly 1 residual outcome, got {residual_names}"

  schema = cfg.get("result_schema")
  assert isinstance(schema, dict)
  for req_key in (
    "schema_version",
    "gpu_tag",
    "suites_requested",
    "device_kind",
    "jax_version",
    "cpu_smoke",
    "quick_mode",
    "total_wall_s",
  ):
    assert req_key in schema


def test_bench_v36_cpu_smoke_all_five_suites(tmp_path: Path) -> None:
  """Execute `bench_v36_cupotential_ab.py --cpu-smoke` across S1..S5 and verify invariants."""
  out_json = tmp_path / "v36_ab_cpu_smoke.json"
  rc = bench_v36_mod.main(["--cpu-smoke", "--suites", "S1,S2,S3,S4,S5", "--out", str(out_json)])
  assert rc == 0
  assert out_json.is_file()

  payload = json.loads(out_json.read_text(encoding="utf-8"))
  assert payload["schema_version"] == "pallas_v36_ab_suite_v1"
  assert payload["cpu_smoke"] is True
  assert payload["quick_mode"] is False
  assert payload["total_wall_s"] > 0.0

  # Match every key & declared type in bench_v36_cupotential_ab.bth.toml [result_schema]
  bth_cfg = tomllib.loads(BTH_TOML_PATH.read_text(encoding="utf-8"))
  type_map = {"str": str, "bool": bool, "float": (int, float)}
  for k, t_name in bth_cfg["result_schema"].items():
    assert k in payload
    assert isinstance(payload[k], type_map[t_name])

  # S1: Reciprocal pressure (+353 bar legacy bias eliminated) + NPT liquid density
  s1 = payload["s1_npt_tip3p_pressure_and_density"]
  assert abs(s1["legacy_bias_bar"]) > 200.0
  assert s1["v36_abs_error_bar"] < 1.0e-3
  assert 0.5 < s1["prolix_mean_density_g_cm3"] < 1.5
  assert 0.5 < s1["openmm_mean_density_g_cm3"] < 1.5

  # S2: Symplectic RATTLE rotational KE & post-B2 orthogonality + NVT solute-water delta_T
  s2 = payload["s2_nvt_hbonds_solute_water_delta_t"]
  assert s2["rotational_ke_rel_err"] < 1.0e-4
  assert s2["post_b2_rad_vel_err"] < 1.0e-6
  assert math.isfinite(s2["t_solute_k"])
  assert math.isfinite(s2["t_water_k"])
  assert math.isfinite(s2["delta_t_solute_minus_water_k"])

  # S3: Suite D ProlixContext Morton fast-path + ProlixSimulation reporter execution
  s3 = payload["s3_suite_d_openmm_fastpath_and_reporters"]
  assert s3["morton_fastpath_active"] is True
  assert s3["v36_fastpath_ms_per_step"] > 0.0
  assert s3["staged_simulation_ms_per_step"] > 0.0
  assert s3["n_reports_emitted"] >= 2

  # S4: nb >= 64 3D cell-grid + 8-octant sub-AABB tile pruning
  s4 = payload["s4_dhfr_morton_tile_pruning_and_throughput"]
  assert s4["n_tiles"] >= 64
  assert 0 < s4["v36_octant_cell_grid_tile_pairs"] < s4["all_upper_tri_tile_pairs"]
  assert s4["v36_octant_cell_grid_tile_pairs"] <= s4["coarse_aabb_only_tile_pairs"]
  assert s4["pruned_vs_all_pairs_pct"] > 20.0
  assert s4["prolix_ms_per_step"] > 0.0

  # S5: fltflt mixed/single ratio & rigid SETTLE bond constraint accuracy
  s5 = payload["s5_fltflt_mixed_vs_single_ratio"]
  assert s5["single_ms_per_step"] > 0.0
  assert s5["mixed_fltflt_ms_per_step"] > 0.0
  assert s5["mixed_over_single_ratio"] > 0.0
  assert s5["bond_constraint_max_err_mixed"] < 1.0e-4


def test_bench_v36_gpu_mode_enforces_anti_fallback_guard() -> None:
  """Verify GPU mode (`cpu_smoke=False`) refuses to run without JAX GPU / OpenMM CUDA."""
  fake_cpu_dev = type("FakeDev", (), {"platform": "cpu", "device_kind": "cpu"})()
  with (
    patch("scripts.benchmarks.bench_v36_cupotential_ab.jax.devices", return_value=[fake_cpu_dev]),
    pytest.raises(RuntimeError, match="GPU mode requires a JAX GPU backend"),
  ):
    run_v36_ab_benchmark(suites=("S1",), cpu_smoke=False)


def test_v35_aggregator_against_real_l40s_release_ladder(tmp_path: Path) -> None:
  """Verify `aggregate_v35_cross_arch_report.py` on real `261111_l40s_pallas_v35_release_ladder.json`."""
  assert REAL_L40S_JSON.is_file()
  out_json = tmp_path / "v35_summary.json"
  out_md = tmp_path / "v35_summary.md"

  rc = agg_mod.main(
    [
      "--bench-dir",
      str(REAL_BENCH_DIR),
      "--out-json",
      str(out_json),
      "--out-md",
      str(out_md),
    ]
  )
  assert rc == 0
  assert out_json.is_file()
  assert out_md.is_file()

  summary = json.loads(out_json.read_text(encoding="utf-8"))
  assert summary["schema_version"] == "pallas_v35_cross_arch_summary_v1"
  assert "l40s" in summary["verified_gpus"]

  l40s = summary["architectures"]["l40s"]
  assert l40s["status"] == "verified"
  assert l40s["device_name"] == "NVIDIA L40S"
  assert l40s["n_provenance_records"] >= 20
  assert "libOpenMMCUDA" in l40s["lib_openmm_cuda_path"]

  metrics = l40s["metrics"]
  assert metrics["gbsa_5dfr_b1_ms"] == pytest.approx(0.2267367, rel=1e-4)
  assert metrics["dhfr_24k_b1_ms"] == pytest.approx(0.6635139, rel=1e-4)
  assert metrics["apoa1_92k_b1_ms"] == pytest.approx(1.4563838, rel=1e-4)
  assert metrics["cellulose_408k_b1_ms"] == pytest.approx(7.1615131, rel=1e-4)
  assert metrics["ragged_xla_compile_count"] == 1

  md_text = out_md.read_text(encoding="utf-8")
  assert "| `l40s` | `verified` | NVIDIA L40S |" in md_text
  assert "`Cellulose` (`408.6k`, PME) Prolix `B=1`" in md_text


def test_v35_aggregator_require_all_and_full_four_arch_pass(tmp_path: Path) -> None:
  """Verify `--require-all` fails when GPUs are pending and passes when all 4 are verified."""
  l40s_raw = json.loads(REAL_L40S_JSON.read_text(encoding="utf-8"))
  (tmp_path / "261111_l40s_pallas_v35_release_ladder.json").write_text(
    json.dumps(l40s_raw), encoding="utf-8"
  )

  # Without --require-all, missing GPUs are gracefully marked pending
  partial = aggregate_v35_reports(tmp_path, require_all=False)
  assert partial["verified_gpus"] == ["l40s"]
  assert set(partial["pending_gpus"]) == {"a100", "h200", "blackwell"}
  for pending_tag in ("a100", "h200", "blackwell"):
    assert partial["architectures"][pending_tag]["status"] == "pending"

  # With --require-all, missing GPUs raise FileNotFoundError
  with pytest.raises(FileNotFoundError, match="--require-all failed"):
    aggregate_v35_reports(tmp_path, require_all=True)

  # Populate all 4 canonical architectures (including Blackwell fallback filename test)
  for tag, dev_name, fname in (
    ("a100", "NVIDIA A100-SXM4-80GB", "261111_a100_pallas_v35_release_ladder.json"),
    ("h200", "NVIDIA H200", "261111_h200_pallas_v35_release_ladder.json"),
    (
      "blackwell",
      "NVIDIA RTX PRO 6000 Blackwell",
      "261111_rtx_pro_6000_blackwell_pallas_v35_release_ladder.json",
    ),
  ):
    cloned = copy.deepcopy(l40s_raw)
    cloned["gpu_tag"] = tag
    cloned["device_kind"] = dev_name
    for _, prov in collect_provenance_records(cloned):
      prov["device_name"] = dev_name
    (tmp_path / fname).write_text(json.dumps(cloned), encoding="utf-8")

  full = aggregate_v35_reports(tmp_path, require_all=True)
  assert full["all_present"] is True
  assert full["pending_gpus"] == []
  assert [spec[0] for spec in ARCH_SPECS] == full["verified_gpus"]
  md_full = render_markdown_summary(full)
  assert "pending" not in md_full


@pytest.mark.parametrize(
  ("bad_field", "bad_value"),
  [
    ("platform_name", "CPU"),
    ("platform_name", "OpenCL"),
    ("device_name", ""),
    ("device_index", None),
    ("precision", "fp16_unknown"),
    ("cuda_Compiler", ""),
    ("CudaDriverVersion", None),
    ("lib_openmm_cuda_path", "/usr/lib/libOpenMMCPU.so"),
  ],
)
def test_v35_aggregator_rejects_corrupted_cuda_provenance(
  tmp_path: Path,
  bad_field: str,
  bad_value: Any,
) -> None:
  """Every one of the 7 OpenMM CUDA provenance fields must trip `ProvenanceValidationError` on corruption."""
  assert bad_field in REQUIRED_PROVENANCE_FIELDS
  corrupted = json.loads(REAL_L40S_JSON.read_text(encoding="utf-8"))
  records = collect_provenance_records(corrupted)
  assert len(records) > 5
  # Corrupt a deep nested provenance record (not just the first one)
  records[-1][1][bad_field] = bad_value
  (tmp_path / "261111_l40s_pallas_v35_release_ladder.json").write_text(
    json.dumps(corrupted), encoding="utf-8"
  )
  with pytest.raises(ProvenanceValidationError):
    aggregate_v35_reports(tmp_path, require_all=False)


def test_sphinx_guide_has_zero_proprietary_or_tmp_references() -> None:
  """Ensure the v3.6/v3.7 Sphinx guide contains only clean scientific prose and no local scratch paths."""
  assert DOC_GUIDE_PATH.is_file()
  text = DOC_GUIDE_PATH.read_text(encoding="utf-8").lower()
  for prefix, suffix in (("/" + "tmp/", "cupotential"), ("include/", "cupo"), ("lib", "cupo")):
    assert (prefix + suffix) not in text
