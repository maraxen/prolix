#!/bin/bash
#SBATCH --job-name=prolix_verify_debt750
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/logs/slurm/%A_%a.verify_debt750.out
#SBATCH --error=outputs/logs/slurm/%A_%a.verify_debt750.err
#
# Debt 750/3516 verification: 1AKE-only B=16, prereg step count (200_000
# steps @ dt=0.5fs, seed=0), same conditions as job 17905437's original
# 5/64-non-finite diagnostic -- isolated to 1AKE alone so no 1ubq/2gb1/water
# cost is paid. See scripts/experiments/verify_debt750_1ake_nan.bth.toml.
#
# Submission:
#   export CAMPAIGN_ID=<from `bth campaign create`>
#   sbatch --export=ALL,CAMPAIGN_ID scripts/slurm/verify_debt750_1ake_nan.slurm

set -euo pipefail

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f "${_PROJ}/scripts/slurm/_common_env.sh" ]]; then
  _PROJ="${HOME}/projects/prolix"
fi
export SLURM_SUBMIT_DIR="${_PROJ}"
cd "${_PROJ}"
PROLIX_ROOT="$(pwd -P)"
cd "${PROLIX_ROOT}"
export PROLIX_ROOT

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-$(dirname "${PROLIX_ROOT}")}"
export UV_PROJECT="${WORKSPACE_ROOT}"
export UV_PYTHON="${UV_PYTHON:-3.13}"

SEED="${SEED:-0}"
REPLICAS="${REPLICAS:-16}"
PS="${PS:-100}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"

# shellcheck source=/dev/null
source "scripts/slurm/_bth_env.sh"
cd "${PROLIX_ROOT}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
  echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

echo "=== Debt 750/3516 verification: 1AKE B=16 NaN fragility post-debt-841 ==="
echo "Host: $(hostname)"
echo "PROLIX_ROOT=${PROLIX_ROOT}"
echo "uv python: $(uv run --python "${UV_PYTHON}" python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Replicas: ${REPLICAS}  ps: ${PS}  Seed: ${SEED}"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

mkdir -p outputs/bench outputs/logs/slurm

OUT="outputs/bench/verify_debt750_1ake_nan_seed${SEED}_job${SLURM_JOB_ID:-local}.json"

CAMPAIGN_FLAG=""
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"

# shellcheck disable=SC2086
uv run --python "${UV_PYTHON}" bth run python scripts/experiments/verify_debt750_1ake_nan.py \
    --tag cluster --tag debt750 --tag debt841 --tag "seed:${SEED}" \
    ${CAMPAIGN_FLAG} \
    --out "${OUT}" \
    -- \
    --replicas "${REPLICAS}" \
    --ps "${PS}" \
    --seed "${SEED}" \
    --out "${OUT}"

EXIT=$?
echo "=== exit=${EXIT} out=${OUT} ==="
exit ${EXIT}
