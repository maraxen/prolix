"""Cross-architecture aggregator and provenance verifier for Pallas v3.5 release ladders.

Ingests `docs/benchmarks/261111_{l40s,a100,h200,blackwell}_pallas_v35_release_ladder.json`,
validates all 7 mandatory OpenMM CUDA anti-fallback provenance fields across every OpenMM
measurement in each present artifact, marks pending GPUs gracefully unless `--require-all`
is set, and emits both a structured JSON report and a GitHub-flavored Markdown summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_DIR = ROOT / "docs" / "benchmarks"

EXPECTED_SCHEMA_VERSION = "pallas_v35_release_ladder_v1"

ARCH_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
  ("l40s", "NVIDIA L40S (sm_89)", ("261111_l40s_pallas_v35_release_ladder.json",)),
  ("a100", "NVIDIA A100-SXM4-80GB (sm_80)", ("261111_a100_pallas_v35_release_ladder.json",)),
  ("h200", "NVIDIA H200 (sm_90)", ("261111_h200_pallas_v35_release_ladder.json",)),
  (
    "blackwell",
    "NVIDIA RTX PRO 6000 Blackwell (sm_120)",
    (
      "261111_blackwell_pallas_v35_release_ladder.json",
      "261111_rtx_pro_6000_blackwell_pallas_v35_release_ladder.json",
    ),
  ),
)

REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
  "platform_name",
  "device_name",
  "device_index",
  "precision",
  "cuda_Compiler",
  "CudaDriverVersion",
  "lib_openmm_cuda_path",
)

VALID_PRECISIONS: frozenset[str] = frozenset({"single", "mixed", "double"})


class ProvenanceValidationError(ValueError):
  """Raised when an OpenMM benchmark record violates CUDA anti-fallback provenance."""


def collect_provenance_records(
  node: Any,
  path: str = "",
) -> list[tuple[str, dict[str, Any]]]:
  """Recursively locate all `provenance` and `openmm_provenance` dicts in a ladder JSON."""
  found: list[tuple[str, dict[str, Any]]] = []
  if isinstance(node, dict):
    for key, val in node.items():
      child_path = f"{path}.{key}" if path else key
      if key in ("provenance", "openmm_provenance") and isinstance(val, dict):
        found.append((child_path, val))
      else:
        found.extend(collect_provenance_records(val, child_path))
  elif isinstance(node, list):
    for idx, item in enumerate(node):
      found.extend(collect_provenance_records(item, f"{path}[{idx}]"))
  return found


def validate_openmm_cuda_provenance(
  prov: dict[str, Any],
  *,
  location: str,
) -> None:
  """Validate all 7 mandatory OpenMM CUDA anti-fallback provenance fields on one record."""
  for field in REQUIRED_PROVENANCE_FIELDS:
    if field not in prov:
      raise ProvenanceValidationError(
        f"{location}: missing required OpenMM CUDA provenance field '{field}'"
      )
    val = prov[field]
    if not isinstance(val, str) or not val.strip():
      raise ProvenanceValidationError(
        f"{location}: OpenMM CUDA provenance field '{field}' must be a non-empty string, got {val!r}"
      )

  if prov["platform_name"] != "CUDA":
    raise ProvenanceValidationError(
      f"{location}: expected platform_name == 'CUDA', got {prov['platform_name']!r}"
    )

  prec = prov["precision"].strip().lower()
  if prec not in VALID_PRECISIONS:
    raise ProvenanceValidationError(
      f"{location}: invalid OpenMM CUDA precision {prov['precision']!r}; "
      f"expected one of {sorted(VALID_PRECISIONS)}"
    )

  lib_path = prov["lib_openmm_cuda_path"]
  if "libOpenMMCUDA" not in lib_path:
    raise ProvenanceValidationError(
      f"{location}: lib_openmm_cuda_path must reference libOpenMMCUDA.so, got {lib_path!r}"
    )


def _find_explicit_system(
  data: dict[str, Any],
  system_name: str,
) -> dict[str, Any] | None:
  suites_ab = data.get("suites_a_and_b")
  if not isinstance(suites_ab, dict):
    return None
  ladder = suites_ab.get("explicit_ladder")
  if not isinstance(ladder, list):
    return None
  for entry in ladder:
    if isinstance(entry, dict) and entry.get("system") == system_name:
      return entry
  return None


def _extract_batch_ms(
  entry: dict[str, Any] | None,
  batch_size: int,
  *,
  scaling_key: str = "batch_scaling",
) -> float | None:
  if not isinstance(entry, dict):
    return None
  rows = entry.get(scaling_key)
  if not isinstance(rows, list):
    return None
  for row in rows:
    if isinstance(row, dict) and row.get("B") == batch_size:
      for ms_key in ("ms_per_step_per_replica", "prolix_ms_per_step_per_replica", "ms_per_eval_per_replica"):
        val = row.get(ms_key)
        if isinstance(val, (int, float)):
          return float(val)
  return None


def _extract_openmm_single_b1_ms(entry: dict[str, Any] | None) -> float | None:
  if not isinstance(entry, dict):
    return None
  omm = entry.get("openmm")
  if not isinstance(omm, dict):
    return None
  b1 = omm.get("openmm_cuda_single_b1")
  if isinstance(b1, dict) and isinstance(b1.get("ms_per_step"), (int, float)):
    return float(b1["ms_per_step"])
  return None


def _extract_arch_metrics(data: dict[str, Any]) -> dict[str, Any]:
  gate = data.get("v35_gate_summary")
  gate_d = gate if isinstance(gate, dict) else {}

  suites_ab = data.get("suites_a_and_b")
  gbsa = suites_ab.get("suite_a_gbsa") if isinstance(suites_ab, dict) else None
  uao = _find_explicit_system(data, "1UAO_solvated_2k")
  ubq = _find_explicit_system(data, "1UBQ_solvated_7k")
  dhfr = _find_explicit_system(data, "DHFR_24k_pme_openmm_grid")
  apoa1 = _find_explicit_system(data, "ApoA1_92k_pme")
  cellulose = _find_explicit_system(data, "Cellulose_408k_pme")

  suite_c = data.get("suite_c")
  ragged = suite_c.get("cross_size_ragged_screen") if isinstance(suite_c, dict) else None

  suite_f = data.get("suite_f_differentiable_moat")
  diff_systems = suite_f.get("differentiable_systems") if isinstance(suite_f, dict) else None
  dhfr_vjp_ms: float | None = None
  if isinstance(diff_systems, list):
    for ds in diff_systems:
      if isinstance(ds, dict) and "DHFR" in str(ds.get("system", "")):
        dhfr_vjp_ms = _extract_batch_ms(ds, 1)
        break

  return {
    "gbsa_5dfr_b1_ms": gate_d.get("MK-67_5DFR_gbsa_B1_ms") or _extract_batch_ms(gbsa, 1),
    "gbsa_5dfr_b32_ms_rep": gate_d.get("MK-67_5DFR_gbsa_B32_ms_rep") or _extract_batch_ms(gbsa, 32),
    "gbsa_5dfr_omm_b1_ms": _extract_openmm_single_b1_ms(gbsa),
    "uao_2k_b1_ms": _extract_batch_ms(uao, 1),
    "uao_2k_omm_b1_ms": _extract_openmm_single_b1_ms(uao),
    "ubq_7k_b1_ms": _extract_batch_ms(ubq, 1),
    "dhfr_24k_b1_ms": gate_d.get("MK-66_DHFR_60grid_B1_ms") or _extract_batch_ms(dhfr, 1),
    "dhfr_24k_b32_ms_rep": gate_d.get("MK-65_DHFR_60grid_B32_ms_rep") or _extract_batch_ms(dhfr, 32),
    "dhfr_24k_omm_b1_ms": _extract_openmm_single_b1_ms(dhfr),
    "apoa1_92k_b1_ms": gate_d.get("MK-65_ApoA1_92k_B1_ms") or _extract_batch_ms(apoa1, 1),
    "apoa1_92k_omm_b1_ms": _extract_openmm_single_b1_ms(apoa1),
    "cellulose_408k_b1_ms": gate_d.get("MK-71_Cellulose_408k_B1_ms") or _extract_batch_ms(cellulose, 1),
    "ragged_suite_c2_ms_rep": (
      gate_d.get("MK-68_SuiteC2_ragged_ms_rep")
      or (
        float(ragged["prolix_ms_per_step_per_replica"])
        if isinstance(ragged, dict)
        and isinstance(ragged.get("prolix_ms_per_step_per_replica"), (int, float))
        else None
      )
    ),
    "ragged_xla_compile_count": (
      gate_d.get("MK-68_SuiteC2_xla_compile_count")
      or (
        int(ragged["xla_executable_count"])
        if isinstance(ragged, dict) and isinstance(ragged.get("xla_executable_count"), int)
        else None
      )
    ),
    "dhfr_24k_vjp_b1_ms": dhfr_vjp_ms,
  }


def _resolve_arch_file(bench_dir: Path, filenames: tuple[str, ...]) -> Path | None:
  for name in filenames:
    cand = bench_dir / name
    if cand.is_file():
      return cand
  return None


def aggregate_v35_reports(
  bench_dir: Path = DEFAULT_BENCH_DIR,
  *,
  require_all: bool = False,
) -> dict[str, Any]:
  """Ingest, validate OpenMM CUDA provenance, and aggregate 4-GPU v3.5 ladder artifacts."""
  architectures: dict[str, dict[str, Any]] = {}
  verified_tags: list[str] = []
  pending_tags: list[str] = []

  for gpu_tag, display_label, candidate_files in ARCH_SPECS:
    resolved = _resolve_arch_file(bench_dir, candidate_files)
    if resolved is None:
      pending_tags.append(gpu_tag)
      architectures[gpu_tag] = {
        "gpu_tag": gpu_tag,
        "display_label": display_label,
        "status": "pending",
        "artifact_path": str(bench_dir / candidate_files[0]),
        "device_name": None,
        "driver_version": None,
        "openmm_version": None,
        "jax_version": None,
        "quick_mode": None,
        "total_wall_s": None,
        "n_provenance_records": 0,
        "metrics": {},
      }
      continue

    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != EXPECTED_SCHEMA_VERSION:
      pending_tags.append(gpu_tag)
      architectures[gpu_tag] = {
        "gpu_tag": gpu_tag,
        "display_label": display_label,
        "status": "pending",
        "reason": f"schema_version != {EXPECTED_SCHEMA_VERSION!r}",
        "artifact_path": str(resolved),
        "device_name": raw.get("device_kind") if isinstance(raw, dict) else None,
        "driver_version": None,
        "openmm_version": None,
        "jax_version": raw.get("jax_version") if isinstance(raw, dict) else None,
        "quick_mode": raw.get("quick_mode") if isinstance(raw, dict) else None,
        "total_wall_s": raw.get("total_wall_s") if isinstance(raw, dict) else None,
        "n_provenance_records": 0,
        "metrics": {},
      }
      continue

    inv = raw.get("openmm_inventory")
    if isinstance(inv, dict):
      if not inv.get("has_cuda_platform"):
        raise ProvenanceValidationError(
          f"{resolved.name}.openmm_inventory: has_cuda_platform must be True"
        )
      inv_lib = inv.get("lib_openmm_cuda_path")
      if not isinstance(inv_lib, str) or "libOpenMMCUDA" not in inv_lib:
        raise ProvenanceValidationError(
          f"{resolved.name}.openmm_inventory: lib_openmm_cuda_path invalid: {inv_lib!r}"
        )

    prov_records = collect_provenance_records(raw)
    if not prov_records:
      raise ProvenanceValidationError(
        f"{resolved.name}: zero OpenMM CUDA provenance records found in artifact"
      )

    for loc, prov in prov_records:
      validate_openmm_cuda_provenance(prov, location=f"{resolved.name}:{loc}")

    first_prov = prov_records[0][1]
    verified_tags.append(gpu_tag)
    architectures[gpu_tag] = {
      "gpu_tag": gpu_tag,
      "display_label": display_label,
      "status": "verified",
      "artifact_path": str(resolved),
      "device_name": first_prov["device_name"],
      "driver_version": first_prov["CudaDriverVersion"],
      "cuda_compiler": first_prov["cuda_Compiler"],
      "lib_openmm_cuda_path": first_prov["lib_openmm_cuda_path"],
      "openmm_version": inv.get("openmm_version") if isinstance(inv, dict) else None,
      "jax_version": raw.get("jax_version"),
      "quick_mode": bool(raw.get("quick_mode", False)),
      "total_wall_s": float(raw.get("total_wall_s", 0.0)),
      "n_provenance_records": len(prov_records),
      "metrics": _extract_arch_metrics(raw),
    }

  if require_all and pending_tags:
    raise FileNotFoundError(
      f"--require-all failed: missing or pending v3.5 release ladder artifacts for GPUs: {pending_tags}"
    )

  return {
    "schema_version": "pallas_v35_cross_arch_summary_v1",
    "bench_dir": str(bench_dir),
    "required_provenance_fields": list(REQUIRED_PROVENANCE_FIELDS),
    "verified_gpus": verified_tags,
    "pending_gpus": pending_tags,
    "all_present": len(pending_tags) == 0,
    "architectures": architectures,
  }


def _fmt_ms(val: Any) -> str:
  if isinstance(val, (int, float)):
    return f"{float(val):.4f}"
  return "pending"


def render_markdown_summary(summary: dict[str, Any]) -> str:
  """Render a GitHub-flavored Markdown cross-architecture provenance & latency report."""
  archs = summary["architectures"]
  tags = [spec[0] for spec in ARCH_SPECS]

  lines: list[str] = [
    "# Pallas v3.5 Cross-Architecture Release Ladder Summary",
    "",
    "## 1. OpenMM CUDA Anti-Fallback Provenance Status",
    "",
    "| GPU Tag | Status | Hardware Device | CUDA Driver | OpenMM | JAX | Verified CUDA Records | Wall Time (s) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |",
  ]
  for tag in tags:
    rec = archs[tag]
    if rec["status"] == "verified":
      wall_s = f"{rec['total_wall_s']:.1f}"
      lines.append(
        f"| `{tag}` | `verified` | {rec['device_name']} | `{rec['driver_version']}` | "
        f"`{rec['openmm_version'] or 'n/a'}` | `{rec['jax_version'] or 'n/a'}` | "
        f"{rec['n_provenance_records']} | {wall_s} |"
      )
    else:
      lines.append(
        f"| `{tag}` | `pending` | {rec['display_label']} | pending | pending | pending | 0 | pending |"
      )

  lines.extend(
    [
      "",
      "## 2. Cross-Architecture Kernel & Frontend Latency Matrix (`ms/step` or `ms/eval`)",
      "",
      "| Metric | `l40s` | `a100` | `h200` | `blackwell` |",
      "| :--- | ---: | ---: | ---: | ---: |",
    ]
  )

  metric_rows: tuple[tuple[str, str], ...] = (
    ("`5DFR_GBSA` (`2.5k`) Prolix `B=1`", "gbsa_5dfr_b1_ms"),
    ("`5DFR_GBSA` (`2.5k`) Prolix `B=32` (`ms/rep`)", "gbsa_5dfr_b32_ms_rep"),
    ("`5DFR_GBSA` (`2.5k`) OpenMM CUDA `B=1`", "gbsa_5dfr_omm_b1_ms"),
    ("`1UAO` (`2.3k`, PME) Prolix `B=1`", "uao_2k_b1_ms"),
    ("`1UAO` (`2.3k`, PME) OpenMM CUDA `B=1`", "uao_2k_omm_b1_ms"),
    ("`1UBQ` (`7.1k`, PME) Prolix `B=1`", "ubq_7k_b1_ms"),
    ("`DHFR` (`23.6k`, `60^3` PME) Prolix `B=1`", "dhfr_24k_b1_ms"),
    ("`DHFR` (`23.6k`, `60^3` PME) Prolix `B=32` (`ms/rep`)", "dhfr_24k_b32_ms_rep"),
    ("`DHFR` (`23.6k`, `60^3` PME) OpenMM CUDA `B=1`", "dhfr_24k_omm_b1_ms"),
    ("`ApoA1` (`92.2k`, PME) Prolix `B=1`", "apoa1_92k_b1_ms"),
    ("`ApoA1` (`92.2k`, PME) OpenMM CUDA `B=1`", "apoa1_92k_omm_b1_ms"),
    ("`Cellulose` (`408.6k`, PME) Prolix `B=1`", "cellulose_408k_b1_ms"),
    ("Suite C2 Cross-Size Ragged (`ms/rep`)", "ragged_suite_c2_ms_rep"),
    ("Suite F `DHFR` Differentiable VJP (`ms/eval`)", "dhfr_24k_vjp_b1_ms"),
  )

  for label, m_key in metric_rows:
    cells = [
      _fmt_ms(archs[t]["metrics"].get(m_key)) if archs[t]["status"] == "verified" else "pending"
      for t in tags
    ]
    lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

  xla_cells = [
    str(archs[t]["metrics"].get("ragged_xla_compile_count"))
    if archs[t]["status"] == "verified"
    and archs[t]["metrics"].get("ragged_xla_compile_count") is not None
    else "pending"
    for t in tags
  ]
  lines.append(
    f"| Suite C2 Ragged XLA Executable Count | {xla_cells[0]} | {xla_cells[1]} | {xla_cells[2]} | {xla_cells[3]} |"
  )
  lines.append("")
  return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Aggregate Pallas v3.5 4-GPU release ladder JSON artifacts and verify OpenMM CUDA provenance."
  )
  parser.add_argument(
    "--bench-dir",
    type=Path,
    default=DEFAULT_BENCH_DIR,
    help="Directory containing 261111_*_pallas_v35_release_ladder.json files.",
  )
  parser.add_argument(
    "--require-all",
    action="store_true",
    help="Fail if any of l40s, a100, h200, or blackwell is missing or pending.",
  )
  parser.add_argument(
    "--out-json",
    type=Path,
    default=None,
    help="Optional path to write the aggregated JSON summary.",
  )
  parser.add_argument(
    "--out-md",
    type=Path,
    default=None,
    help="Optional path to write the aggregated Markdown summary table.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_arg_parser()
  args = parser.parse_args(argv)
  summary = aggregate_v35_reports(args.bench_dir, require_all=args.require_all)
  md_text = render_markdown_summary(summary)

  if args.out_json is not None:
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

  if args.out_md is not None:
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")

  sys.stdout.write(md_text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
