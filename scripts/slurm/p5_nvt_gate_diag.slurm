#!/bin/bash
# P5 NVT gate diagnostic — multi-seed stability + T_trans/T_rot decomp + OU comparison.
#
# Answers three questions in one run:
#   Q1  Is 293.83 K statistical (5-seed mean) or systematic?
#   Q2  T_trans vs T_rot decomposition (which DOF are underpowered?)
#   Q3  Constrained vs unconstrained OU (projection-drain hypothesis)
#
# Bathos campaign: 933024dc (p5-nvt-gate-diag)
#
# Usage (from repo root after pushing to cluster):
#   sbatch scripts/slurm/p5_nvt_gate_diag.slurm

#SBATCH --job-name=plx-p5-diag
#SBATCH --partition=mit_preemptable
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/engaging/%j.out
#SBATCH --error=outputs/logs/engaging/%j.err

set -euo pipefail

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Validate: myxcel wraps scripts in a /tmp sbatch so SLURM_SUBMIT_DIR may be /tmp.
# Fall back to canonical cluster path, then export back so _common_env.sh agrees.
if [[ ! -f "${_PROJ}/scripts/slurm/_common_env.sh" ]]; then
  _PROJ="${HOME}/projects/prolix"
fi
export SLURM_SUBMIT_DIR="${_PROJ}"
cd "${_PROJ}"

# shellcheck source=/dev/null
source "scripts/slurm/_common_env.sh"

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"
echo "job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local} workspace_root=${WORKSPACE_ROOT}"

if [[ "${P5_DIAG_SKIP_UV_SYNC:-0}" != "1" ]]; then
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

# shellcheck source=/dev/null
source "scripts/slurm/_bth_env.sh"

mkdir -p outputs/p5_nvt_gate_diag

CAMPAIGN_ID="${CAMPAIGN_ID:-dbb2cc17}"
OUT="outputs/p5_nvt_gate_diag/result_${SLURM_JOB_ID:-local}.json"

echo "=== P5 NVT gate diagnostic (campaign=${CAMPAIGN_ID}) ==="
uv run bth run python scripts/experiments/p5_nvt_gate_diag.py \
  --tag p5-nvt-gate-diag \
  --tag "slurm_job=${SLURM_JOB_ID:-local}" \
  --campaign "${CAMPAIGN_ID}" \
  --out "${OUT}" \
  -- --out "${OUT}" \
  2>&1 | tee "${LOG_ROOT}/app/p5_nvt_gate_diag_${SLURM_JOB_ID:-local}.log"

echo "=== Done: exit $? ==="
