#!/bin/bash
#SBATCH --job-name=prolix_fit_hp4
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4007,node4008
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%A_%a.fit_bonded_hp4.out
#SBATCH --error=outputs/logs/slurm/%A_%a.fit_bonded_hp4.err

# §7.1 ensemble scaling sweep — job array on pinned hardware.
#
# Hardware pin: pi_so3, node4007/node4008 ONLY (4× RTX Pro 6000 Blackwell).
# Exclude node4009 (H200) — different GPU class would confound the comparison.
#
# Array indexing (0..23):
#   N_VALUES   = (16 32 64 128 256 512)  → 6 values
#   MODES      = (looped batched)         → 2 values
#   PRECISIONS = (float32 float64)        → 2 values
#   Index = (N_idx * 4) + (mode_idx * 2) + prec_idx
#
# Submission:
#   sbatch --array=0-23%4 \
#          --export=ALL,CAMPAIGN_ID=<uuid>,N_CONF_CAP=100 \
#          scripts/slurm/fit_bonded_hp4.slurm
#   (%4 = max 4 concurrent tasks, matching 4 GPUs across the two nodes)
#
# Single-cell submission (legacy, no array — set MODE/PRECISION/etc via --export):
#   sbatch --export=ALL,MODE=batched,PRECISION=float32,N_MOLS_TARGET=128,... \
#       scripts/slurm/fit_bonded_hp4.slurm

set -euo pipefail

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

# ===== Decode array task ID into (N, MODE, PRECISION) =====
N_VALUES=(16 32 64 128 256 512)
MODES=(looped batched)
PRECISIONS=(float32 float64)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    T=${SLURM_ARRAY_TASK_ID}
    N_IDX=$(( T / 4 ))
    MODE_IDX=$(( (T % 4) / 2 ))
    PREC_IDX=$(( T % 2 ))
    if [[ $N_IDX -ge ${#N_VALUES[@]} ]]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$T out of range (max 23)"
        exit 1
    fi
    N_MOLS_TARGET="${N_VALUES[$N_IDX]}"
    MODE="${MODES[$MODE_IDX]}"
    PRECISION="${PRECISIONS[$PREC_IDX]}"
fi

# ===== Defaults (legacy single-cell path; also fallback for env-set values) =====
MODE="${MODE:-batched}"
PRECISION="${PRECISION:-float32}"
N_STEPS="${N_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEED="${SEED:-42}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
N_MOLS_TARGET="${N_MOLS_TARGET:-}"
N_CONF_CAP="${N_CONF_CAP:-}"

# Validate
if [[ "${MODE}" != "looped" && "${MODE}" != "batched" ]]; then
    echo "ERROR: MODE must be 'looped' or 'batched' (got ${MODE})"; exit 1
fi
if [[ "${PRECISION}" != "float32" && "${PRECISION}" != "float64" ]]; then
    echo "ERROR: PRECISION must be 'float32' or 'float64' (got ${PRECISION})"; exit 1
fi

source "scripts/slurm/_bth_env.sh"

echo "=== §7.1 fit_bonded_hp4 ==="
echo "Host: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-N/A}  task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Mode: ${MODE}  Precision: ${PRECISION}  N_mols: ${N_MOLS_TARGET:-<none>}  N_conf_cap: ${N_CONF_CAP:-<none>}  N_steps: ${N_STEPS}"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

mkdir -p outputs/results outputs/logs/slurm

DATA_DIR="data/ani1x_subset"
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: ${DATA_DIR} not found"; exit 1
fi

# Build flags
PRECISION_FLAG=""
[[ "${PRECISION}" == "float64" ]] && PRECISION_FLAG="--float64"

CAMPAIGN_FLAG=""
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"

SWEEP_FLAGS=""
[[ -n "${N_MOLS_TARGET}" ]] && SWEEP_FLAGS+=" --n-mols-target ${N_MOLS_TARGET}"
[[ -n "${N_CONF_CAP}" ]] && SWEEP_FLAGS+=" --n-conf-cap ${N_CONF_CAP}"

# shellcheck disable=SC2086
uv run bth run python scripts/experiments/fit_bonded_hp4.py \
    --tag cluster --tag hp4 --tag s71 --tag fit \
    --tag "mode:${MODE}" --tag "precision:${PRECISION}" \
    ${N_MOLS_TARGET:+--tag "N:${N_MOLS_TARGET}"} \
    ${SLURM_ARRAY_TASK_ID:+--tag "array_task:${SLURM_ARRAY_TASK_ID}"} \
    ${CAMPAIGN_FLAG} \
    -- \
    --subset-dir "${DATA_DIR}" \
    --mode "${MODE}" \
    --n-steps "${N_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    ${PRECISION_FLAG} \
    ${SWEEP_FLAGS}

EXIT=$?
echo "=== exit=${EXIT} ==="
exit ${EXIT}
