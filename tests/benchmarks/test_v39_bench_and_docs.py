"""Tests for Pallas v3.9 benchmark suite (`bench_v39_triclinic_alchem_prolate`) and Sphinx docs."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import openmm as mm
import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKTREE_ROOT) not in sys.path:
  sys.path.insert(0, str(WORKTREE_ROOT))
if str(WORKTREE_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(WORKTREE_ROOT / "src"))

pytestmark = pytest.mark.fast

BTH_TOML_PATH = (
  WORKTREE_ROOT / "scripts" / "benchmarks" / "bench_v39_triclinic_alchem_prolate.bth.toml"
)
BENCH_PY_PATH = (
  WORKTREE_ROOT / "scripts" / "benchmarks" / "bench_v39_triclinic_alchem_prolate.py"
)
GUIDE_MD_PATH = (
  WORKTREE_ROOT / "docs" / "source" / "guides" / "cupotential_parity_and_openmm_fastpath.md"
)
API_DOCS_DIR = WORKTREE_ROOT / "docs" / "source" / "api"

PROLATE_PUBLIC_FNS = (
  "prolate_p1_short_range_value_and_d_dr2",
  "prolate_p1_long_range_value_and_d_dr2",
  "prolate_p6_short_range_value_and_d_dr2",
  "prolate_p6_long_range_value_and_d_dr2",
  "prolate_p1_fourier_kernel_and_d_dk2",
  "prolate_p6_fourier_kernel_and_d_dk2",
  "prolate_p1_self_energy_coeff",
  "prolate_p6_self_energy_coeff",
  "prolate_p1_dc_coeff",
  "prolate_spme_electrostatics",
  "prolate_dispersion_pme",
)

DLPACK_JAX_METHODS = (
  "getPositionsJax",
  "setPositionsJax",
  "getVelocitiesJax",
  "setVelocitiesJax",
  "getForcesJax",
)


def test_v39_bth_toml_schema_and_single_residual_outcome() -> None:
  """Verify `bench_v39_triclinic_alchem_prolate.bth.toml` conforms to bathos DuckDB schema."""
  assert BTH_TOML_PATH.is_file()
  cfg = tomllib.loads(BTH_TOML_PATH.read_text(encoding="utf-8"))

  assert "experiment" in cfg
  hypothesis = cfg["experiment"].get("hypothesis")
  assert isinstance(hypothesis, str)
  assert len(hypothesis.strip()) > 60

  outcomes = cfg.get("outcomes")
  assert isinstance(outcomes, dict)
  assert len(outcomes) >= 3

  residual_names: list[str] = []
  for name, spec in outcomes.items():
    assert isinstance(spec.get("condition"), str)
    assert spec["condition"].strip()
    assert "pallas_v39_complete_suite_v1" in spec["condition"]
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


def test_v39_bench_cpu_smoke_all_five_suites(tmp_path: Path) -> None:
  """Execute `bench_v39_triclinic_alchem_prolate.py --cpu-smoke` across S1..S5."""
  import scripts.benchmarks.bench_v39_triclinic_alchem_prolate as bench_v39_mod

  out_json = tmp_path / "v39_complete_cpu_smoke.json"
  rc = bench_v39_mod.main(["--cpu-smoke", "--suites", "S1,S2,S3,S4,S5", "--out", str(out_json)])
  assert rc == 0
  assert out_json.is_file()

  payload: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
  assert payload["schema_version"] == "pallas_v39_complete_suite_v1"
  assert payload["cpu_smoke"] is True
  assert payload["quick_mode"] is False
  assert payload["total_wall_s"] > 0.0

  bth_cfg = tomllib.loads(BTH_TOML_PATH.read_text(encoding="utf-8"))
  type_map: dict[str, Any] = {"str": str, "bool": bool, "float": (int, float)}
  for key, t_name in bth_cfg["result_schema"].items():
    assert key in payload
    assert isinstance(payload[key], type_map[t_name])

  # S1: Reduced-form rhombic dodecahedron & truncated octahedron parity + holonomic bonds
  s1 = payload["s1_triclinic_explicit_solvent"]
  assert s1["max_rel_energy_err"] < 5.0e-3
  assert s1["max_force_rel_err"] < 2.5e-2
  assert s1["max_holonomic_bond_err_a"] < 1.0e-4
  assert s1["rhombic_dodecahedron"]["morton_fastpath_active"] is True
  assert s1["truncated_octahedron"]["morton_fastpath_active"] is True
  assert s1["rhombic_dodecahedron"]["prolix_ms_per_step"] > 0.0
  assert s1["truncated_octahedron"]["prolix_ms_per_step"] > 0.0

  # S2: Cross-shape 1D ragged batching + deterministic forces
  s2 = payload["s2_ragged_cross_shape_batch"]
  assert s2["batch_size"] == 3
  assert s2["replica_shapes"] == ["cubic", "rhombic_dodecahedron", "truncated_octahedron"]
  assert s2["deterministic_force_max_diff"] <= 1.0e-6
  assert s2["prolix_batch_ms_per_step"] > 0.0
  assert len(s2["per_replica_potential_energies_kj_mol"]) == 3
  assert all(math.isfinite(val) for val in s2["per_replica_potential_energies_kj_mol"])

  # S3: Clean-room C^2 Prolate Spheroidal Ewald splittings (P=1 & P=6)
  s3 = payload["s3_prolate_c2_splitting"]
  assert s3["p1_val_at_rc"] < 1.0e-9
  assert s3["p1_d_dr2_at_rc"] < 1.0e-9
  assert s3["p6_val_at_rc"] < 1.0e-9
  assert s3["p6_d_dr2_at_rc"] < 1.0e-9
  assert s3["p6_k0_val"] > 0.0
  assert abs(s3["p6_k0_grad"]) < 1.0e-12
  assert s3["p1_rel_energy_err"] < 5.0e-3
  assert s3["p1_grad_finite"] is True
  assert s3["p6_grad_finite"] is True
  assert s3["prolate_p1_ms"] > 0.0
  assert s3["prolate_p6_ms"] > 0.0

  # S4: Sparse InteractionGroups + dynamic alchemical lambda sweep
  s4 = payload["s4_sparse_interaction_groups_alchemical"]
  assert s4["is_gpu_runner_fast"] is True
  assert s4["sparse_exclusion_omitted"] is True
  assert s4["sparse_ig_rel_err"] < 1.0e-2
  n_win = len(s4["lambda_windows"])
  assert n_win >= 2
  assert len(s4["window_potential_energies_kj_mol"]) == n_win
  assert len(s4["du_dlambda_per_window"]) == n_win
  assert all(math.isfinite(val) for val in s4["window_potential_energies_kj_mol"])
  assert all(math.isfinite(val) for val in s4["du_dlambda_per_window"])
  assert s4["alchem_batch_ms_per_step"] > 0.0

  # S5: GPU anisotropic/membrane barostats, 4.0 fs HMR, and zero-copy JAX/DLPack
  s5 = payload["s5_barostats_hmr_and_dlpack"]
  assert s5["aniso_ms_per_step"] > 0.0
  assert s5["aniso_volume_nm3"] > 1.0
  assert s5["membrane_ms_per_step"] > 0.0
  assert s5["membrane_volume_nm3"] > 1.0
  assert s5["hmr_dt_fs"] == 4.0
  assert s5["hmr_bond_max_err_a"] < 1.0e-4
  assert s5["dlpack_roundtrip_max_err"] <= 1.0e-7
  assert s5["zerocopy_jax_us"] > 0.0
  assert s5["host_numpy_us"] > 0.0


def test_v39_gpu_mode_enforces_openmm_cuda_and_jax_gpu_guards() -> None:
  """Verify GPU mode (`cpu_smoke=False`) rejects CPU JAX backends and wires `require_openmm_cuda_context`."""
  import scripts.benchmarks.bench_v39_triclinic_alchem_prolate as bench_v39_mod

  fake_cpu_dev = type("FakeDev", (), {"platform": "cpu", "device_kind": "cpu"})()
  with (
    patch.object(bench_v39_mod.jax, "devices", return_value=[fake_cpu_dev]),
    pytest.raises(RuntimeError, match="GPU mode requires a JAX GPU backend"),
  ):
    bench_v39_mod.run_v39_complete_benchmark(suites=("S1",), cpu_smoke=False)

  dummy_sys = mm.System()
  dummy_sys.addParticle(12.0)
  dummy_integ = mm.VerletIntegrator(0.001)
  sentinel_ctx = MagicMock(spec=mm.Context)
  sentinel_prov = {"platform_name": "CUDA"}
  with patch.object(
    bench_v39_mod,
    "require_openmm_cuda_context",
    return_value=(sentinel_ctx, sentinel_prov),
  ) as mock_guard:
    ctx_out, prov_out = bench_v39_mod._openmm_context_for_mode(
      dummy_sys, dummy_integ, expected_precision="mixed", cpu_smoke=False
    )
    assert ctx_out is sentinel_ctx
    assert prov_out == sentinel_prov
    mock_guard.assert_called_once_with(
      dummy_sys,
      dummy_integ,
      expected_precision="mixed",
      allow_opencl_fallback=False,
      allow_cpu_fallback=False,
    )


def test_v39_docs_have_zero_transitions_zero_proprietary_refs_and_full_api_coverage() -> None:
  """Verify zero `---` docutils transitions, zero proprietary strings, and complete v3.9 API coverage."""
  transition_re = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
  pages_to_check = [
    GUIDE_MD_PATH,
    API_DOCS_DIR / "api.md",
    API_DOCS_DIR / "export.md",
    API_DOCS_DIR / "knob_surface.md",
    API_DOCS_DIR / "openmm.md",
    API_DOCS_DIR / "physics.md",
  ]
  for page_path in pages_to_check:
    assert page_path.is_file(), f"Missing doc page: {page_path}"
    text = page_path.read_text(encoding="utf-8")
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
      if line.strip().startswith("```"):
        in_fence = not in_fence
        continue
      if not in_fence:
        assert not transition_re.match(line), (
          f"{page_path.name}:{lineno} contains docutils horizontal transition {line!r}"
        )

  forbidden_tokens = ("cu" + "Potential", "cu" + "po/", "LicenseRef-Nvidia" + "Proprietary")
  for clean_path in (
    GUIDE_MD_PATH,
    API_DOCS_DIR / "physics.md",
    API_DOCS_DIR / "openmm.md",
    BENCH_PY_PATH,
    BTH_TOML_PATH,
  ):
    content = clean_path.read_text(encoding="utf-8")
    for bad in forbidden_tokens:
      assert bad not in content, f"Forbidden token {bad!r} found in {clean_path.name}"

  guide_text = GUIDE_MD_PATH.read_text(encoding="utf-8")
  for expected_topic in (
    "2505.09727",
    "prolate_splitting",
    "rhombic_dodecahedron",
    "truncated_octahedron",
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
    "getPositionsJax",
    "bench_v39_triclinic_alchem_prolate.py",
  ):
    assert expected_topic in guide_text, f"Missing topic {expected_topic!r} in guide"

  physics_md = (API_DOCS_DIR / "physics.md").read_text(encoding="utf-8")
  prolate_mod = importlib.import_module("prolix.physics.prolate_splitting")
  assert inspect.getdoc(prolate_mod), "prolix.physics.prolate_splitting missing module docstring"
  for fn_name in PROLATE_PUBLIC_FNS:
    assert fn_name in physics_md, f"{fn_name} missing from docs/source/api/physics.md"
    fn_obj = getattr(prolate_mod, fn_name)
    assert callable(fn_obj)
    assert inspect.getdoc(fn_obj), f"prolix.physics.prolate_splitting.{fn_name} missing docstring"

  openmm_md = (API_DOCS_DIR / "openmm.md").read_text(encoding="utf-8")
  omm_pkg = importlib.import_module("prolix.openmm")
  for cls_name in ("ProlixContext", "ProlixSimulation", "ProlixBatchSimulation"):
    cls_obj = getattr(omm_pkg, cls_name)
    for m_name in DLPACK_JAX_METHODS:
      assert m_name in openmm_md, f"{m_name} missing from docs/source/api/openmm.md"
      assert callable(getattr(cls_obj, m_name, None)), f"{cls_name}.{m_name} missing or not callable"


def test_v39_sphinx_html_build_zero_errors_when_sphinx_installed(tmp_path: Path) -> None:
  """Run a full Sphinx HTML build into `tmp_path` when `sphinx` + `myst_parser` are installed."""
  if (
    importlib.util.find_spec("sphinx") is None
    or importlib.util.find_spec("myst_parser") is None
  ):
    pytest.skip("sphinx / myst_parser not installed in active pytest environment")

  from sphinx.cmd.build import build_main

  html_out = tmp_path / "html"
  warn_log = tmp_path / "sphinx_warnings.log"
  mock_list = (
    "iree,kups,dmff,torch,torchmd,espaloma,dgl,openmmtools,parmed,"
    "pymbar,netCDF4,mpiplus,numba,h5py,termcolor"
  )
  ext_list = (
    "sphinx.ext.autodoc,sphinx.ext.autosummary,sphinx.ext.napoleon,"
    "sphinx.ext.viewcode,sphinx.ext.intersphinx,sphinx.ext.mathjax,myst_parser"
  )
  rc = build_main(
    [
      "-b",
      "html",
      "-D",
      f"extensions={ext_list}",
      "-D",
      f"autodoc_mock_imports={mock_list}",
      "-w",
      str(warn_log),
      str(WORKTREE_ROOT / "docs" / "source"),
      str(html_out),
    ]
  )
  assert rc == 0
  assert (html_out / "api" / "physics.html").is_file()
  assert (html_out / "api" / "openmm.html").is_file()
  assert (html_out / "guides" / "cupotential_parity_and_openmm_fastpath.html").is_file()
  if warn_log.is_file():
    warn_text = warn_log.read_text(encoding="utf-8")
    error_lines = [ln for ln in warn_text.splitlines() if "ERROR:" in ln]
    assert error_lines == [], f"Sphinx HTML build emitted ERROR lines: {error_lines}"
