#!/usr/bin/env python3
"""Write ``outputs/logs/engaging/<date>/run_manifest.json`` for cluster benchmark runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _try_run(cmd: list[str]) -> str:
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return (r.stdout or "").strip()
  except (OSError, subprocess.TimeoutExpired):
    return ""


def main() -> None:
  root = Path(os.environ.get("PROLIX_ROOT", os.getcwd())).resolve()
  date = os.environ.get(
    "ENGAGING_LOG_DATE",
    datetime.now(timezone.utc).strftime("%Y%m%d"),
  )
  out_dir = root / "outputs" / "logs" / "engaging" / date
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "slurm").mkdir(exist_ok=True)
  (out_dir / "app").mkdir(exist_ok=True)

  sha = _try_run(["git", "-C", str(root), "rev-parse", "HEAD"])
  branch = _try_run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"])

  manifest: dict = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_sha": sha or None,
    "git_branch": branch or None,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "engaging_log_date": date,
    "prolix_root": str(root),
  }

  jax_v = _try_run(
    [
      sys.executable,
      "-c",
      "import jax, jaxlib; print(jax.__version__); print(jaxlib.__version__)",
    ]
  )
  if jax_v:
    lines = jax_v.splitlines()
    manifest["jax_version"] = lines[0] if lines else None
    manifest["jaxlib_version"] = lines[1] if len(lines) > 1 else None

  omm = _try_run([sys.executable, "-c", "import openmm; print(openmm.__version__)"])
  manifest["openmm_version"] = omm or None

  out_path = out_dir / "run_manifest.json"
  out_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(out_path)


if __name__ == "__main__":
  main()
