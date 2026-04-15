#!/usr/bin/env bash
# Local dry-run for Engaging chignolin benchmark steps (no SSH/sbatch).
# Run from the repository root before first cluster submit:
#   bash scripts/verify_engaging_local.sh
#
# Uses `uv run`; with a UV workspace (venv under the parent of prolix+proxide), that
# environment is picked up automatically. Otherwise use pip/venv and `python -m pytest` etc.
#
# Records the same log layout as SLURM templates under outputs/logs/engaging/<date>/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PROLIX_ROOT="$ROOT"
export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

echo "Manifest: $(head -c 200 "${LOG_ROOT}/run_manifest.json")..."
echo "--- pytest (CPU, not slow) ---"
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" uv run pytest tests/physics/test_pbc_end_to_end.py -m "not slow" -q --tb=short
if [ "${ENGAGING_LOCAL_QUICK:-}" = "1" ]; then
  echo "OK (quick): manifest + pytest only. Unset ENGAGING_LOCAL_QUICK for full GPU benchmarks."
  exit 0
fi
echo "--- prolix_vs_openmm_speed ---"
uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --skip-reference
echo "--- benchmark_nlvsdense ---"
uv run python scripts/benchmark_nlvsdense.py --systems CHIGNOLIN 1X2G --replicas 1
echo "OK: local engaging pipeline finished. Logs under ${LOG_ROOT}"
