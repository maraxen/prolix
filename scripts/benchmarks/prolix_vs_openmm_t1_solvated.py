#!/usr/bin/env python3
"""T1 explicit throughput: same entry point and PME defaults as T0, larger periodic workload.

This does **not** load a full solvated peptide+water system (that path is parity-tested elsewhere).
For throughput comparability it scales the **minimal neutral PME box** via ``--charge-copies`` and
writes ``tier: "T1"`` in ``--json`` output (see ``docs/source/explicit_solvent/schemas/benchmark_run.schema.json``).

Defaults (override on the CLI after this wrapper's name)::

  --tier T1 --charge-copies 4 --box 60.0

Example::

  uv run python scripts/benchmarks/prolix_vs_openmm_t1_solvated.py --json t1.json --skip-reference
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _with_t1_defaults(argv: list[str]) -> list[str]:
  """Prepend T1-oriented defaults when the user did not set the flags."""
  out = list(argv)
  if "--tier" not in out:
    out = ["--tier", "T1"] + out
  if "--charge-copies" not in out:
    out = ["--charge-copies", "4"] + out
  if "--box" not in out:
    out = ["--box", "60.0"] + out
  return out


def main() -> int:
  here = Path(__file__).resolve().parent
  speed_path = here / "prolix_vs_openmm_speed.py"
  if not speed_path.is_file():
    print("Could not find prolix_vs_openmm_speed.py", file=sys.stderr)
    return 2
  cmd = [sys.executable, str(speed_path), *_with_t1_defaults(sys.argv[1:])]
  return int(subprocess.call(cmd))


if __name__ == "__main__":
  raise SystemExit(main())
