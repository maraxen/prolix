#!/usr/bin/env python3
"""Aggregate final JSON summaries from ``tip3p_langevin_tightening.py`` tee logs (one replica per file)."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
  sys.path.insert(0, str(_BENCH))
import tip3p_ke_profile  # noqa: E402

TIGHTENING_AGGREGATE_SCHEMA = "tip3p_tightening_aggregate/v1"
TIGHTENING_AGGREGATE_SCHEMA_V2 = "tip3p_tightening_aggregate/v2"


def _infer_bath_k(blocks: list[dict]) -> float:
  targets: list[float] = []
  for b in blocks:
    for run in b["runs"]:
      targets.append(float(run["target_T_K"]))
  if not targets:
    msg = "no target_T_K found in run summaries"
    raise ValueError(msg)
  u = {round(t, 9) for t in targets}
  if len(u) != 1:
    msg = f"inconsistent target_T_K across runs: {u}"
    raise ValueError(msg)
  return float(targets[0])


def _infer_profile_id(blocks: list[dict]) -> str | None:
  ids: list[str] = []
  for b in blocks:
    for run in b["runs"]:
      pid = run.get("profile_id")
      if pid is not None and str(pid) != "":
        ids.append(str(pid))
  if not ids:
    return None
  u = set(ids)
  if len(u) != 1:
    msg = f"inconsistent profile_id across runs: {u}"
    raise ValueError(msg)
  return next(iter(u))


def _extract_summary_json(text: str) -> dict:
  """Parse the last top-level JSON object in ``text`` (warnings may precede the block)."""
  decoder = json.JSONDecoder()
  last: dict | None = None
  i = 0
  n = len(text)
  while i < n:
    while i < n and text[i].isspace():
      i += 1
    if i >= n:
      break
    try:
      obj, j = decoder.raw_decode(text, i)
    except json.JSONDecodeError:
      i += 1
      continue
    if isinstance(obj, dict) and "replica_id" in obj and "runs" in obj:
      last = obj
    i = j
  if last is None:
    raise ValueError("no summary JSON with keys replica_id and runs found")
  return last


def _collect(paths: list[Path]) -> list[dict]:
  out: list[dict] = []
  for p in paths:
    block = _extract_summary_json(p.read_text(encoding="utf-8", errors="replace"))
    out.append(block)
  return out


def _newest_per_replica(
  log_dir: Path,
  pattern: str = "tip3p_tightening_r*_*.log",
  *,
  basename_substr: str | None = None,
  after_mtime: float | None = None,
) -> list[Path]:
  """One log per replica index (newest mtime), for SLURM array tee names that embed task job id."""
  base = str(log_dir)
  by_rep: dict[int, list[Path]] = {}
  for pstr in glob.glob(f"{base}/{pattern}"):
    p = Path(pstr)
    name = p.name
    if basename_substr is not None and basename_substr not in name:
      continue
    if after_mtime is not None and p.stat().st_mtime < after_mtime:
      continue
    if not name.startswith("tip3p_tightening_r"):
      continue
    mid = name.removeprefix("tip3p_tightening_r").split("_", 1)[0]
    try:
      rep = int(mid)
    except ValueError:
      continue
    by_rep.setdefault(rep, []).append(p)
  chosen = [max(paths, key=lambda x: x.stat().st_mtime) for _, paths in sorted(by_rep.items())]
  return chosen


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "logs",
    nargs="*",
    type=Path,
    help="Log files (default: stdin as a single blob if no paths given)",
  )
  ap.add_argument(
    "--from-dir",
    type=Path,
    default=None,
    help="Directory of tee logs: pick newest file per replica (tip3p_tightening_rN_*.log)",
  )
  ap.add_argument(
    "--basename-substr",
    default=None,
    metavar="SUBSTR",
    help="When using --from-dir: only consider logs whose filename contains this substring",
  )
  ap.add_argument(
    "--after-mtime",
    type=float,
    default=None,
    metavar="UNIX_TIME",
    help="When using --from-dir: only consider logs with mtime >= this Unix timestamp",
  )
  ap.add_argument(
    "--require-profile-id",
    action="store_true",
    help="Fail unless every run in every block has the same non-null profile_id (gate hygiene).",
  )
  ap.add_argument(
    "--schema-version",
    choices=("v1", "v2"),
    default="v1",
    help="Aggregate output schema version. Keep v1 as default for compatibility.",
  )
  args = ap.parse_args(argv)
  chosen_paths: list[Path] | None = None

  if args.from_dir is not None:
    chosen_paths = _newest_per_replica(
      args.from_dir,
      basename_substr=args.basename_substr,
      after_mtime=args.after_mtime,
    )
    blocks = _collect(chosen_paths)
  elif args.logs:
    blocks = _collect(args.logs)
  else:
    stdin = sys.stdin.read()
    blocks = [_extract_summary_json(stdin)]

  if args.require_profile_id:
    for b in blocks:
      tip3p_ke_profile.assert_runs_profile_consistency(b["runs"], context=f"replica_id={b.get('replica_id')}")

  by_engine: dict[str, list[float]] = {}
  for b in blocks:
    for run in b["runs"]:
      eng = str(run["engine"])
      by_engine.setdefault(eng, []).append(float(run["mean_T_K"]))
      by_engine.setdefault(f"{eng}_std", []).append(float(run["std_T_K"]))
      if eng == "prolix":
        mat = run.get("mean_T_atomic_K")
        by_engine.setdefault("prolix_mean_T_atomic_K", []).append(
          float(mat) if mat is not None else float("nan")
        )
        sat = run.get("std_T_atomic_K")
        by_engine.setdefault("prolix_std_T_atomic_K", []).append(
          float(sat) if sat is not None else float("nan")
        )
        for key in (
          "diag_projection_residual_p95",
          "diag_bond_residual_max_abs_p95",
          "diag_com_metric_p95",
        ):
          val = run.get(key)
          by_engine.setdefault(f"prolix_{key}", []).append(float(val) if val is not None else float("nan"))

  rows = []
  for eng in sorted({k for k in by_engine if not k.endswith("_std") and not k.startswith("prolix_")}):
    xs = by_engine[eng]
    ss = by_engine.get(f"{eng}_std", [])
    mean_x = float(sum(xs) / len(xs)) if xs else float("nan")
    pst = (
      float(statistics.pstdev(xs))
      if len(xs) > 1
      else 0.0
    )
    row: dict = {
      "engine": eng,
      "n_replicas": len(xs),
      "mean_T_K_mean": mean_x,
      "mean_T_K_pstdev_across_replicas": pst,
      "per_replica_mean_T_K": xs,
      "per_replica_std_T_K": ss,
    }
    if eng == "prolix":
      ats = by_engine.get("prolix_mean_T_atomic_K", [])
      ast = by_engine.get("prolix_std_T_atomic_K", [])
      if ats and len(ats) == len(xs):
        finite = np.asarray(ats, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
          row["mean_T_atomic_K_mean"] = float(np.mean(finite))
          row["mean_T_atomic_K_pstdev_across_replicas"] = (
            float(statistics.pstdev([float(x) for x in finite]))
            if finite.size > 1
            else 0.0
          )
        row["per_replica_mean_T_atomic_K"] = ats
        row["per_replica_std_T_atomic_K"] = ast if ast and len(ast) == len(xs) else []
      for diag_key in (
        "diag_projection_residual_p95",
        "diag_bond_residual_max_abs_p95",
        "diag_com_metric_p95",
      ):
        vals = by_engine.get(f"prolix_{diag_key}", [])
        if vals and len(vals) == len(xs):
          finite = np.asarray(vals, dtype=np.float64)
          finite = finite[np.isfinite(finite)]
          row[f"{diag_key}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
          row[f"per_replica_{diag_key}"] = vals
    rows.append(row)

  try:
    t_bath = _infer_bath_k(blocks)
    profile_id = _infer_profile_id(blocks)
  except ValueError as e:
    print(f"aggregate_tip3p_tightening_logs: {e}", file=sys.stderr)
    return 2

  meta: dict = {"benchmark_policy": {"temperature_k": t_bath}}
  if profile_id is not None:
    meta["profile_id"] = profile_id

  payload: dict = {
    "schema": TIGHTENING_AGGREGATE_SCHEMA if args.schema_version == "v1" else TIGHTENING_AGGREGATE_SCHEMA_V2,
    "meta": meta,
    "aggregated": rows,
  }
  if args.schema_version == "v2":
    payload["meta"]["run_metadata"] = {
      "git_sha": None,
      "jax_backend": None,
      "profile_id": profile_id,
      "gamma_ps": None,
      "projection_site": None,
      "settle_velocity_iters": None,
      "diagnostics_level": None,
      "diagnostics_decimation": None,
      "effective_integrator_config": None,
    }
  if chosen_paths is not None:
    payload["chosen_log_files"] = [str(p) for p in chosen_paths]
    if args.basename_substr is not None:
      payload["basename_substr"] = args.basename_substr
    if args.after_mtime is not None:
      payload["after_mtime"] = args.after_mtime
  print(json.dumps(payload, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main(None))
