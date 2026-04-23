#!/usr/bin/env python3
"""Paired-delta analysis on tightening aggregate artifacts.

Produces ``tip3p_tightening_paired_analysis/v1`` from two
``tip3p_tightening_aggregate/v1|v2`` payloads (incumbent vs candidate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMAS = {"tip3p_tightening_aggregate/v1", "tip3p_tightening_aggregate/v2"}
PAIRED_SCHEMA = "tip3p_tightening_paired_analysis/v1"

# One-sided 95% t critical values (alpha=0.05).
_T95_ONE_SIDED = {
  1: 6.314,
  2: 2.92,
  3: 2.353,
  4: 2.132,
  5: 2.015,
  6: 1.943,
  7: 1.895,
  8: 1.86,
  9: 1.833,
  10: 1.812,
}


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  schema = payload.get("schema")
  if schema not in SUPPORTED_SCHEMAS:
    msg = f"unsupported schema {schema!r}; expected one of {sorted(SUPPORTED_SCHEMAS)!r}"
    raise ValueError(msg)
  return payload


def _row(payload: dict[str, Any], engine: str) -> dict[str, Any]:
  for row in payload.get("aggregated", []):
    if row.get("engine") == engine:
      return row
  msg = f"missing aggregated row for engine={engine!r}"
  raise ValueError(msg)


def _paired_values(payload: dict[str, Any], engine: str) -> list[float]:
  row = _row(payload, engine)
  vals = row.get("per_replica_mean_T_K")
  if not isinstance(vals, list) or not vals:
    msg = f"engine={engine} missing per_replica_mean_T_K"
    raise ValueError(msg)
  return [float(x) for x in vals]


def _paired_row_metric(payload: dict[str, Any], engine: str, key: str) -> list[float] | None:
  row = _row(payload, engine)
  vals = row.get(key)
  if vals is None:
    return None
  if not isinstance(vals, list):
    raise ValueError(f"row field {key!r} must be a list when present")
  return [float(x) for x in vals]


def _bath_k(payload: dict[str, Any]) -> float:
  t = ((payload.get("meta") or {}).get("benchmark_policy") or {}).get("temperature_k")
  if t is None:
    raise ValueError("meta.benchmark_policy.temperature_k missing")
  t = float(t)
  if t <= 0:
    raise ValueError("temperature_k must be positive")
  return t


def _paired_summary(deltas: list[float]) -> dict[str, Any]:
  if not deltas:
    return {"n": 0, "mean": None, "sem": None, "lower_95_one_sided": None, "upper_95_one_sided": None}
  n = len(deltas)
  mean = float(statistics.fmean(deltas))
  sem = float(statistics.pstdev(deltas) / math.sqrt(n)) if n > 1 else 0.0
  tcrit = _T95_ONE_SIDED.get(n - 1, 1.645)
  lower = mean - tcrit * sem
  upper = mean + tcrit * sem
  return {
    "n": n,
    "mean": mean,
    "sem": sem,
    "lower_95_one_sided": lower,
    "upper_95_one_sided": upper,
  }


def _hash_baseline_tuple(payload: dict[str, Any]) -> str:
  basis = {
    "schema": payload.get("schema"),
    "meta": payload.get("meta"),
    "aggregated": payload.get("aggregated"),
  }
  b = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(b).hexdigest()


def analyze(incumbent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
  bath_i = _bath_k(incumbent)
  bath_c = _bath_k(candidate)
  if abs(bath_i - bath_c) > 1e-9:
    raise ValueError(f"bath temperature mismatch: incumbent={bath_i} candidate={bath_c}")
  bath = bath_i

  inc_omm = _paired_values(incumbent, "openmm")
  inc_plx = _paired_values(incumbent, "prolix")
  can_omm = _paired_values(candidate, "openmm")
  can_plx = _paired_values(candidate, "prolix")
  n = len(inc_plx)
  if not (len(inc_omm) == len(inc_plx) == len(can_omm) == len(can_plx)):
    raise ValueError("replica count mismatch across incumbent/candidate rows")

  # M1: prolix R-relative error reduction (incumbent - candidate), larger is better.
  m1_inc = [abs(t - bath) / bath for t in inc_plx]
  m1_can = [abs(t - bath) / bath for t in can_plx]
  m1_delta = [a - b for a, b in zip(m1_inc, m1_can, strict=True)]

  # M2: cross-engine absolute delta reduction (incumbent - candidate), larger is better.
  m2_inc = [abs(o - p) for o, p in zip(inc_omm, inc_plx, strict=True)]
  m2_can = [abs(o - p) for o, p in zip(can_omm, can_plx, strict=True)]
  m2_delta = [a - b for a, b in zip(m2_inc, m2_can, strict=True)]

  def _diagnostic_delta(key: str) -> tuple[list[float] | list[None], dict[str, Any] | None]:
    inc = _paired_row_metric(incumbent, "prolix", key)
    can = _paired_row_metric(candidate, "prolix", key)
    if inc is None or can is None:
      return [None] * n, None
    if len(inc) != n or len(can) != n:
      raise ValueError(f"diagnostic replica length mismatch for {key}")
    deltas = [a - b for a, b in zip(inc, can, strict=True)]  # reduction: incumbent-candidate
    return deltas, _paired_summary(deltas)

  m3_delta, m3_summary = _diagnostic_delta("per_replica_diag_projection_residual_p95")
  m4_delta, m4_summary = _diagnostic_delta("per_replica_diag_bond_residual_max_abs_p95")
  m5_delta, m5_summary = _diagnostic_delta("per_replica_diag_com_metric_p95")
  metrics = {
    "M1_prolix_r_relative_error_reduction": {
      "deltas": m1_delta,
      "summary": _paired_summary(m1_delta),
    },
    "M2_cross_engine_abs_delta_reduction_K": {
      "deltas": m2_delta,
      "summary": _paired_summary(m2_delta),
    },
    "M3_projection_residual_reduction": {
      "deltas": m3_delta,
      "summary": m3_summary,
    },
    "M4_bond_residual_reduction": {
      "deltas": m4_delta,
      "summary": m4_summary,
    },
    "M5_com_metric_reduction": {
      "deltas": m5_delta,
      "summary": m5_summary,
    },
  }
  m1_lower = metrics["M1_prolix_r_relative_error_reduction"]["summary"]["lower_95_one_sided"]
  return {
    "schema": PAIRED_SCHEMA,
    "source": {
      "incumbent_schema": incumbent.get("schema"),
      "candidate_schema": candidate.get("schema"),
      "temperature_k": bath,
      "n_pairs": n,
      "baseline_tuple_hash": _hash_baseline_tuple(incumbent),
    },
    "metrics": metrics,
    "advancement": {
      "criterion": "one_sided_95_lower_bound_gt_zero_on_M1",
      "passes": bool(m1_lower is not None and m1_lower > 0.0),
      "m1_lower_95_one_sided": m1_lower,
    },
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--incumbent", type=Path, required=True, help="Incumbent aggregate JSON")
  ap.add_argument("--candidate", type=Path, required=True, help="Candidate aggregate JSON")
  args = ap.parse_args(argv)
  out = analyze(_load(args.incumbent), _load(args.candidate))
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

