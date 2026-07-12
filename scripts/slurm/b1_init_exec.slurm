#!/bin/bash
#SBATCH --job-name=prolix_b1_init_exec
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%A_%a.b1_init_exec.out
#SBATCH --error=outputs/logs/slurm/%A_%a.b1_init_exec.err
# gres tag only (no --nodelist). Override e.g. --gres=gpu:a100:1 if needed.
# Secondary (pi_so3): sbatch -p pi_so3 --gres=gpu:1 --time=8:00:00 ...

# B1-full Claim-1 init-bound campaign (prereg 260528).
#
# Array tasks map to (backend, seed):
#   0..2  → prolix  seeds 0,1,2
#   3..5  → openmm  seeds 0,1,2
#
# Submission (from repo root after campaign create):
#   export CAMPAIGN_ID=<id from bth campaign create>
#   mkdir -p outputs/logs/slurm outputs/bench
#   sbatch --array=0-2%3 --export=ALL,CAMPAIGN_ID \
#       scripts/slurm/b1_init_exec.slurm          # prolix wave
#   sbatch --array=3-5%3 --export=ALL,CAMPAIGN_ID \
#       scripts/slurm/b1_init_exec.slurm          # openmm wave (if CUDA OpenMM ok)
#
# Single-cell override:
#   sbatch --export=ALL,CAMPAIGN_ID,BACKEND=prolix,SEED=0 \
#       scripts/slurm/b1_init_exec.slurm

set -euo pipefail

# Resolve prolix checkout (realpath — ~/projects/prolix → so3_shared tree)
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f "${_PROJ}/scripts/slurm/_common_env.sh" ]]; then
  _PROJ="${HOME}/projects/prolix"
fi
export SLURM_SUBMIT_DIR="${_PROJ}"
cd "${_PROJ}"
PROLIX_ROOT="$(pwd -P)"
cd "${PROLIX_ROOT}"
export PROLIX_ROOT

# Engaging UV workspace lives at ~/projects (parent of prolix). Pin 3.13 to match
# prolix requires-python ==3.13.* (parent .python-version must not force 3.12).
WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-$(dirname "${PROLIX_ROOT}")}"
export UV_PROJECT="${WORKSPACE_ROOT}"
export UV_PYTHON="${UV_PYTHON:-3.13}"

# Array → backend/seed (override via env wins)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${BACKEND:-}" ]]; then
    TASK="${SLURM_ARRAY_TASK_ID}"
    if (( TASK <= 2 )); then
        BACKEND="prolix"
        SEED="${TASK}"
    else
        BACKEND="openmm"
        SEED="$((TASK - 3))"
    fi
fi

BACKEND="${BACKEND:-prolix}"
SEED="${SEED:-0}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
REPLICAS="${REPLICAS:-16}"
PS="${PS:-100}"
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

# shellcheck source=/dev/null
source "scripts/slurm/_bth_env.sh"
# _common_env cds via SLURM_SUBMIT_DIR; re-assert prolix cwd for the bench
cd "${PROLIX_ROOT}"

# Prereg XLA pins (GPU). Opt-in flag for triton softmax (may be unknown on some jaxlibs).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export B1_XLA_TRITON_FLAG="${B1_XLA_TRITON_FLAG:-0}"

echo "=== B1-full b1_init_exec ==="
echo "Host: $(hostname)"
echo "cwd: $(pwd)"
echo "PROLIX_ROOT=${PROLIX_ROOT}"
echo "UV_PROJECT=${UV_PROJECT} UV_PYTHON=${UV_PYTHON}"
echo "uv python: $(uv run --python "${UV_PYTHON}" python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-N/A}  task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Backend: ${BACKEND}  Seed: ${SEED}  Replicas: ${REPLICAS}  PS: ${PS}"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

mkdir -p outputs/bench outputs/logs/slurm

OUT="outputs/bench/b1_full_${BACKEND}_seed${SEED}_job${SLURM_JOB_ID:-local}.json"

CAMPAIGN_FLAG=""
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"

# shellcheck disable=SC2086
uv run --python "${UV_PYTHON}" bth run python scripts/benchmarks/b1_init_exec.py \
    --tag cluster --tag b1-full --tag "backend:${BACKEND}" --tag "seed:${SEED}" \
    ${SLURM_ARRAY_TASK_ID:+--tag "array_task:${SLURM_ARRAY_TASK_ID}"} \
    ${CAMPAIGN_FLAG} \
    --out "${OUT}" \
    -- \
    --backend "${BACKEND}" \
    --seed "${SEED}" \
    --replicas "${REPLICAS}" \
    --ps "${PS}" \
    ${SMOKE_FLAG}

EXIT=$?
echo "=== exit=${EXIT} out=${OUT} ==="
exit ${EXIT}
