#!/bin/bash
# P2b cross-validation: slow pytest gates on cluster.
#
# Runs test_p2b_nvt_216water.py::test_nvt_216water_temperature_stability
# (and any other -m slow tests listed below).
#
# Partitions:
#   mit_normal_gpu — 6h limit (GPU), default for this job
#   pi_so3         — 48h limit, override: sbatch --partition=pi_so3 ...
#
# Usage (from repo root after pushing to cluster):
#   sbatch scripts/slurm/p2b_slow_gate.slurm
#   # or with GPU override:
#   sbatch --partition=pi_so3 --gres=gpu:1 scripts/slurm/p2b_slow_gate.slurm
#
# Output: logs to outputs/logs/engaging/<date>/slurm/<jobid>.out

#SBATCH --job-name=plx-p2b-slow
#SBATCH --partition=mit_preemptable
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/engaging/%j.out
#SBATCH --error=outputs/logs/engaging/%j.err

set -euo pipefail

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

# shellcheck source=/dev/null
source "scripts/slurm/_common_env.sh"

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"
echo "job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local} workspace_root=${WORKSPACE_ROOT}"

if [[ "${P2B_SKIP_UV_SYNC:-0}" != "1" ]]; then
  # shellcheck source=/dev/null
  source "scripts/slurm/_workspace_uv_sync_inner.sh"
  _workspace_uv_sync_run
fi

cd "${PROLIX_ROOT}"

# Blackwell XLA workaround (node4007/node4008, SM120)
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
  echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# Create tmp/ for JSON report (required by addopts)
mkdir -p tmp

echo "=== P2b slow gates: NVT 216-water ==="
uv run pytest tests/physics/test_p2b_nvt_216water.py \
  -m slow \
  -v \
  --timeout=1800 \
  2>&1 | tee "${LOG_ROOT}/app/p2b_slow_gate_${SLURM_JOB_ID:-local}.log"

echo "=== Done: exit $? ==="
