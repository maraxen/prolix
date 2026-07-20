#!/bin/bash
#SBATCH --job-name=prolix_b1_init_exec
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/slurm/%A_%a.b1_init_exec.out
#SBATCH --error=outputs/logs/slurm/%A_%a.b1_init_exec.err
#
# Host RAM (not GPU) killed prereg on pi_so3: ~67 GiB RSS vs --mem=64G.
# Defaults: mit_preemptable + h200 + 256G + submit with --array=%1.
#
# GRES overrides (tag only, no --nodelist):
#   --gres=gpu:h200:1              # best VRAM; often ≥1–2 TiB host (default)
#   --gres=gpu:rtx_pro_6000:1      # same class as pi_so3 4007/4008 (~96 GiB)
#   --gres=gpu:h100:1              # 80 GiB class; OK if host mem ≥256G
#   --gres=gpu:l40s:1              # first-test FOOTPRINT=l40s only (48 GiB VRAM)
#   A100 40GB is too small for prereg VRAM (~54 GiB seen on Blackwell)
#
# Footprint tiers (b1_init_exec.py FOOTPRINTS):
#   l40s   — B=8, 1 ps, chunk=50     (first measure; not Claim-1 headline)
#   a100   — B=32, 10 ps, chunk=100
#   h200   — B=64, 100 ps, chunk=200
#   prereg — same as h200 (full protocol)
#
# Path (B1-INFER): default B1_PATH=inference (while_loop carry-only, jitted).
#   B1_PATH=baseline keeps pathological scan+traj A/B.
#
# Array: 0..2 prolix seeds; 3..5 openmm. Use %1 so two seeds never share a node.
#
# Submission:
#   export CAMPAIGN_ID=2115b4dd
#   # Claim-1 prereg (recommended — inference path):
#   sbatch --array=0-2%1 --export=ALL,CAMPAIGN_ID,FOOTPRINT=prereg,B1_PATH=inference \
#       scripts/slurm/b1_init_exec.slurm
#   # Pathological baseline A/B:
#   sbatch --array=0-2%1 --export=ALL,CAMPAIGN_ID,FOOTPRINT=prereg,B1_PATH=baseline \
#       scripts/slurm/b1_init_exec.slurm
#   # Same on Blackwell-class preemptable:
#   sbatch --gres=gpu:rtx_pro_6000:1 --array=0-2%1 \
#       --export=ALL,CAMPAIGN_ID,FOOTPRINT=prereg,B1_PATH=inference scripts/slurm/b1_init_exec.slurm
#   # L40S first-test tier:
#   sbatch -p mit_normal_gpu --gres=gpu:l40s:1 --mem=64G --time=2:00:00 \
#       --array=0-2%3 --export=ALL,CAMPAIGN_ID,FOOTPRINT=l40s,B1_PATH=inference \
#       scripts/slurm/b1_init_exec.slurm
#   # pi_so3 pin (only if asked): -p pi_so3 --nodelist=node4007,node4008 --gres=gpu:1 --mem=256G --array=%1

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
FOOTPRINT="${FOOTPRINT:-l40s}"
# Never use env name PATH — that clobbers the shell executable search path.
PATH_MODE="${B1_PATH:-inference}"
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi

# OpenMM CUDA baseline: conda-forge openmm (PyPI wheel has no libOpenMMCUDA).
# Override with B1_OPENMM_ENV if needed.
B1_OPENMM_ENV="${B1_OPENMM_ENV:-/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm}"
PYTHON_RUN="python"
if [[ "${BACKEND}" == "openmm" ]]; then
    if [[ ! -x "${B1_OPENMM_ENV}/bin/python" ]]; then
        echo "ERROR: OpenMM CUDA env missing: ${B1_OPENMM_ENV}/bin/python" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    module load cuda/12.9.1 2>/dev/null || true
    export OPENMM_PLUGIN_DIR="${B1_OPENMM_ENV}/lib/plugins"
    export LD_LIBRARY_PATH="${B1_OPENMM_ENV}/lib${CUDA_HOME:+:${CUDA_HOME}/lib64}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    PYTHON_RUN="${B1_OPENMM_ENV}/bin/python"
    echo "OpenMM CUDA env: ${B1_OPENMM_ENV}"
    echo "OPENMM_PLUGIN_DIR=${OPENMM_PLUGIN_DIR}"
fi

# shellcheck source=/dev/null
source "scripts/slurm/_bth_env.sh"
cd "${PROLIX_ROOT}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export B1_XLA_TRITON_FLAG="${B1_XLA_TRITON_FLAG:-0}"

# Blackwell (node4007/4008, SM120) — same pin as fit_bonded_hp4 / p5 diag
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
  echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

echo "=== B1-full b1_init_exec ==="
echo "Host: $(hostname)"
echo "cwd: $(pwd)"
echo "PROLIX_ROOT=${PROLIX_ROOT}"
echo "UV_PROJECT=${UV_PROJECT} UV_PYTHON=${UV_PYTHON}"
echo "uv python: $(uv run --python "${UV_PYTHON}" python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-N/A}  task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Backend: ${BACKEND}  Seed: ${SEED}  Footprint: ${FOOTPRINT}  Path: ${PATH_MODE}"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

mkdir -p outputs/bench outputs/logs/slurm

OUT="outputs/bench/b1_full_${FOOTPRINT}_${BACKEND}_${PATH_MODE}_seed${SEED}_job${SLURM_JOB_ID:-local}.json"

CAMPAIGN_FLAG=""
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"

# shellcheck disable=SC2086
uv run --python "${UV_PYTHON}" bth run "${PYTHON_RUN}" scripts/benchmarks/b1_init_exec.py \
    --tag cluster --tag b1-full --tag "backend:${BACKEND}" --tag "seed:${SEED}" \
    --tag "footprint:${FOOTPRINT}" --tag "path:${PATH_MODE}" \
    ${SLURM_ARRAY_TASK_ID:+--tag "array_task:${SLURM_ARRAY_TASK_ID}"} \
    ${CAMPAIGN_FLAG} \
    --out "${OUT}" \
    -- \
    --backend "${BACKEND}" \
    --seed "${SEED}" \
    --footprint "${FOOTPRINT}" \
    --path "${PATH_MODE}" \
    --out-json "${OUT}" \
    ${SMOKE_FLAG}

EXIT=$?
echo "=== exit=${EXIT} out=${OUT} ==="
exit ${EXIT}
