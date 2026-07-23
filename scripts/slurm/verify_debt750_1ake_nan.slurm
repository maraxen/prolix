#!/bin/bash
#SBATCH --job-name=prolix_verify_debt750
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
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
# The shared workspace's `prolix` member is an editable install of whatever
# branch happens to be checked out at WORKSPACE_ROOT/prolix (same inode as
# ~/projects/prolix) -- NOT this job's PROLIX_ROOT when PROLIX_ROOT is an
# isolated worktree. Forcing UV_PROJECT to WORKSPACE_ROOT in that case makes
# `uv run` import prolix from the wrong branch/commit (see job 18523653:
# AttributeError on MolecularShapeSpec.has_real_water because the shared
# checkout predated debt 841's fix). Only use the shared workspace when
# PROLIX_ROOT actually IS its `prolix` member; otherwise let uv discover the
# project from cwd (this worktree's own pyproject.toml/uv.lock).
if [[ "$(cd "${WORKSPACE_ROOT}/prolix" 2>/dev/null && pwd -P)" == "${PROLIX_ROOT}" ]]; then
  export UV_PROJECT="${WORKSPACE_ROOT}"
else
  echo "Isolated worktree detected (PROLIX_ROOT=${PROLIX_ROOT} != shared ${WORKSPACE_ROOT}/prolix) -- using worktree-local uv project, not the shared workspace."
  unset UV_PROJECT || true
fi
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
UV_SYNC_EXTRAS=()
if [[ -z "${UV_PROJECT:-}" ]]; then
  # Isolated worktree: freshly-synced local venv has no cached cuda install to
  # inherit (unlike the shared workspace, synced once via `uv sync --extra cuda
  # --extra dev --package prolix`) -- request the same extras explicitly so we
  # don't silently get a CPU-only jax on a GPU allocation.
  UV_SYNC_EXTRAS=(--extra cuda --extra dev)
fi
echo "uv python: $(uv run --python "${UV_PYTHON}" "${UV_SYNC_EXTRAS[@]}" python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Replicas: ${REPLICAS}  ps: ${PS}  Seed: ${SEED}"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

mkdir -p outputs/bench outputs/logs/slurm

OUT="outputs/bench/verify_debt750_1ake_nan_seed${SEED}_job${SLURM_JOB_ID:-local}.json"

CAMPAIGN_FLAG=""
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"

# shellcheck disable=SC2086
uv run --python "${UV_PYTHON}" "${UV_SYNC_EXTRAS[@]}" bth run python scripts/experiments/verify_debt750_1ake_nan.py \
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
