"""P2b NVT gate runner — wraps pytest and emits a bth-compatible result JSON.

Runs tests/physics/test_p2b_nvt_216water.py and extracts structured outcome
from the pytest JSON report. Outputs a result JSON for bathos outcome evaluation.

Usage (via bth run in SLURM script):
    uv run bth run python scripts/experiments/p2b_nvt_gate.py --out outputs/p2b_nvt_gate.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output JSON path for bth outcome evaluation")
    p.add_argument("--smoke", action="store_true", help="Dry-run smoke (import check only)")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        out.write_text(json.dumps({"gate_pass": 1, "n_passed": 0, "n_xfailed": 0, "n_failed": 0,
                                   "smoke": True}))
        print("smoke ok")
        return

    report = Path("tmp/pytest_p2b_nvt.json")
    report.parent.mkdir(parents=True, exist_ok=True)

    ret = subprocess.run(
        ["uv", "run", "pytest", "tests/physics/test_p2b_nvt_216water.py",
         "-m", "slow", "-v", f"--json-report", f"--json-report-file={report}",
         "--timeout=1800"],
        cwd=Path(__file__).resolve().parents[2],
    )

    result = {"n_passed": 0, "n_xfailed": 0, "n_failed": 0, "gate_pass": 0, "exit_code": ret.returncode}

    if report.exists():
        data = json.loads(report.read_text())
        summary = data.get("summary", {})
        result["n_passed"] = summary.get("passed", 0)
        result["n_xfailed"] = summary.get("xfailed", 0)
        result["n_failed"] = summary.get("failed", 0)

    # Gate passes if pytest exits 0 (1 passed + 1 xfailed counts as success)
    result["gate_pass"] = 1 if ret.returncode == 0 else 0

    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
