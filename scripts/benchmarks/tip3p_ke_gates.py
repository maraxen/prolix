#!/usr/bin/env python3
"""P2a-B2-R / P2a-B2-X (G4) gates on benchmark JSON summaries.

Supports:

- ``tip3p_ke_compare/v1``-shaped payloads (``diagnostics`` / ``replicate_temperature``).
- ``tip3p_tightening_aggregate/v1`` from ``aggregate_tip3p_tightening_logs.py``.

Normative definitions live in ``docs/source/explicit_solvent/tip3p_benchmark_policy.md``.
This module is the single implementation used by tests and offline checkers.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

P2A_B2_R_TOL_REL = 0.0167  # ±1.67% of bath temperature (stretch gate, P2a-B2 narrative)
TIGHTENING_AGGREGATE_SCHEMA = "tip3p_tightening_aggregate/v1"
TIGHTENING_AGGREGATE_SCHEMA_V2 = "tip3p_tightening_aggregate/v2"
TIGHTENING_AGGREGATE_SCHEMAS = {TIGHTENING_AGGREGATE_SCHEMA, TIGHTENING_AGGREGATE_SCHEMA_V2}


def _sem_replicate_means(values: list[float]) -> float:
  """SEM of replicate-level means (matches ``tip3p_ke_compare._sem``)."""
  if len(values) < 2:
    return 0.0
  return float(statistics.pstdev(values) / math.sqrt(len(values)))


def find_engine_summary(diagnostics: list[dict[str, Any]], engine: str) -> dict[str, Any]:
  """Return the first successful summary block for ``engine`` (skip error-only rows)."""
  for block in diagnostics:
    if block.get("engine") != engine:
      continue
    if "replicate_temperature" not in block:
      continue
    if block.get("error"):
      continue
    return block
  msg = f"no successful diagnostics block for engine={engine!r}"
  raise ValueError(msg)


def bath_temperature_k(payload: dict[str, Any]) -> float:
  """Bath temperature (K) from benchmark metadata; cross-check ``config`` when present."""
  meta = payload.get("meta") or {}
  cfg = payload.get("config") or {}
  t_meta = (meta.get("benchmark_policy") or {}).get("temperature_k")
  t_cfg = cfg.get("temperature_k")
  if t_meta is None:
    msg = "meta.benchmark_policy.temperature_k is required"
    raise ValueError(msg)
  t_m = float(t_meta)
  if t_cfg is not None and float(t_cfg) != t_m:
    msg = f"temperature_k mismatch: meta={t_m} vs config={t_cfg}"
    raise ValueError(msg)
  return t_m


def p2a_b2_r_passes(*, mean_t: float, t_bath_k: float, tol_rel: float = P2A_B2_R_TOL_REL) -> bool:
  """Per-engine rigid-thermometer mean (replicate-mean series) within ±tol_rel of bath."""
  if t_bath_k <= 0:
    return False
  return abs(float(mean_t) - float(t_bath_k)) / float(t_bath_k) <= tol_rel


def p2a_b2_x_g4_passes(
  *,
  mean_omm: float,
  mean_plx: float,
  sem_omm: float,
  sem_plx: float,
) -> tuple[bool, float, float]:
  """Cross-engine G4 (soft): |ΔT| <= 2 * sqrt(SEM_omm^2 + SEM_plx^2). Returns (pass, diff, bound)."""
  diff = abs(float(mean_omm) - float(mean_plx))
  bound = 2.0 * math.sqrt(float(sem_omm) ** 2 + float(sem_plx) ** 2)
  return (diff <= bound, diff, bound)


def _find_aggregate_engine_row(aggregated: list[dict[str, Any]], engine: str) -> dict[str, Any]:
  for row in aggregated:
    if row.get("engine") == engine:
      return row
  msg = f"no aggregated row for engine={engine!r}"
  raise ValueError(msg)


def _aggregate_row_mean_sem(row: dict[str, Any]) -> tuple[float, float]:
  xs = row.get("per_replica_mean_T_K")
  if not isinstance(xs, list) or not xs:
    msg = "aggregated row must include non-empty per_replica_mean_T_K list"
    raise ValueError(msg)
  vals = [float(x) for x in xs]
  mean = float(statistics.fmean(vals))
  declared = float(row["mean_T_K_mean"])
  tol = max(1e-6, 1e-9 * abs(mean))
  if abs(mean - declared) > tol:
    msg = f"mean_T_K_mean {declared} != fmean(per_replica_mean_T_K) {mean}"
    raise ValueError(msg)
  return mean, _sem_replicate_means(vals)


def evaluate_tightening_aggregate(
  payload: dict[str, Any],
  *,
  tol_rel: float = P2A_B2_R_TOL_REL,
) -> dict[str, Any]:
  """Evaluate R and X on ``aggregate_tip3p_tightening_logs.py`` output (schema v1/v2)."""
  schema = payload.get("schema")
  if schema not in TIGHTENING_AGGREGATE_SCHEMAS:
    msg = f"expected schema in {sorted(TIGHTENING_AGGREGATE_SCHEMAS)!r}, got {schema!r}"
    raise ValueError(msg)
  if schema == TIGHTENING_AGGREGATE_SCHEMA_V2:
    run_meta = (payload.get("meta") or {}).get("run_metadata")
    if not isinstance(run_meta, dict):
      msg = "v2 aggregate requires meta.run_metadata object"
      raise ValueError(msg)
  t_bath = bath_temperature_k(payload)
  aggregated = payload.get("aggregated")
  if not isinstance(aggregated, list):
    msg = "payload.aggregated must be a list"
    raise ValueError(msg)
  omm_row = _find_aggregate_engine_row(aggregated, "openmm")
  plx_row = _find_aggregate_engine_row(aggregated, "prolix")
  mo, so = _aggregate_row_mean_sem(omm_row)
  mp, sp = _aggregate_row_mean_sem(plx_row)
  r_omm = p2a_b2_r_passes(mean_t=mo, t_bath_k=t_bath, tol_rel=tol_rel)
  r_plx = p2a_b2_r_passes(mean_t=mp, t_bath_k=t_bath, tol_rel=tol_rel)
  x_pass, x_diff, x_bound = p2a_b2_x_g4_passes(mean_omm=mo, mean_plx=mp, sem_omm=so, sem_plx=sp)
  return {
    "schema": schema,
    "t_bath_k": t_bath,
    "openmm": {"mean_T_K": mo, "sem_T_K": so, "p2a_b2_r_pass": r_omm},
    "prolix": {"mean_T_K": mp, "sem_T_K": sp, "p2a_b2_r_pass": r_plx},
    "p2a_b2_r_both_pass": bool(r_omm and r_plx),
    "p2a_b2_x": {"pass": x_pass, "abs_delta_mean_T_K": x_diff, "g4_bound_K": x_bound},
  }


def evaluate_payload(
  payload: dict[str, Any],
  *,
  tol_rel: float = P2A_B2_R_TOL_REL,
) -> dict[str, Any]:
  """Evaluate R and X gates; fail closed on missing engines."""
  t_bath = bath_temperature_k(payload)
  diagnostics = payload.get("diagnostics")
  if not isinstance(diagnostics, list):
    msg = "payload.diagnostics must be a list"
    raise ValueError(msg)
  omm = find_engine_summary(diagnostics, "openmm")
  plx = find_engine_summary(diagnostics, "prolix")
  rt_o = omm["replicate_temperature"]
  rt_p = plx["replicate_temperature"]
  mo, so = float(rt_o["mean"]), float(rt_o["sem"])
  mp, sp = float(rt_p["mean"]), float(rt_p["sem"])
  r_omm = p2a_b2_r_passes(mean_t=mo, t_bath_k=t_bath, tol_rel=tol_rel)
  r_plx = p2a_b2_r_passes(mean_t=mp, t_bath_k=t_bath, tol_rel=tol_rel)
  x_pass, x_diff, x_bound = p2a_b2_x_g4_passes(mean_omm=mo, mean_plx=mp, sem_omm=so, sem_plx=sp)
  return {
    "t_bath_k": t_bath,
    "openmm": {"mean_T_K": mo, "sem_T_K": so, "p2a_b2_r_pass": r_omm},
    "prolix": {"mean_T_K": mp, "sem_T_K": sp, "p2a_b2_r_pass": r_plx},
    "p2a_b2_r_both_pass": bool(r_omm and r_plx),
    "p2a_b2_x": {"pass": x_pass, "abs_delta_mean_T_K": x_diff, "g4_bound_K": x_bound},
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "json_path",
    type=Path,
    help="Path to tip3p_ke_compare JSON or tip3p_tightening_aggregate/v1 JSON",
  )
  args = ap.parse_args(argv)
  payload = json.loads(args.json_path.read_text(encoding="utf-8"))
  schema = payload.get("schema")
  if schema in TIGHTENING_AGGREGATE_SCHEMAS:
    out = evaluate_tightening_aggregate(payload)
  else:
    out = evaluate_payload(payload)
  print(json.dumps(out, indent=2))
  return 0 if out["p2a_b2_r_both_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
