#!/bin/bash
#SBATCH --job-name=plx-lfmiddle-dt
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2%3
#SBATCH --output=outputs/logs/slurm/%A_%a.lfmiddle_dt_sweep.out
#SBATCH --error=outputs/logs/slurm/%A_%a.lfmiddle_dt_sweep.err

# LFMiddle dt falsification sweep (bathos campaign lfmiddle-dt-sweep).
# GPU + jax.lax.scan production loop (no Python step driver).
#
# Array: 0 → 0.25 fs, 1 → 0.5 fs, 2 → 1.0 fs
#
#   export CAMPAIGN_ID=89c9a900
#   sbatch --array=0-2%3 \
#     --export=ALL,CAMPAIGN_ID=${CAMPAIGN_ID},N_WATERS=2,SIM_PS=50,BURN_PS=10,SEED=701 \
#     scripts/slurm/lfmiddle_dt_sweep.slurm

set -euo pipefail

# Anchor paths to this script — safe from any submission cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if PROLIX_ROOT="$(git -C "${_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  PROLIX_ROOT="${HOME}/projects/prolix"
fi
cd "${PROLIX_ROOT}"

DT_VALUES=(0.25 0.5 1.0)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  DT_FS="${DT_VALUES[$SLURM_ARRAY_TASK_ID]}"
else
  DT_FS="${DT_FS:-0.5}"
fi

N_WATERS="${N_WATERS:-2}"
SIM_PS="${SIM_PS:-50.0}"
BURN_PS="${BURN_PS:-10.0}"
SEED="${SEED:-701}"
INTEGRATOR="${INTEGRATOR:-lfmiddle}"
CAMPAIGN_ID="${CAMPAIGN_ID:-89c9a900}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# JAX Blackwell workaround — XLA autotuning hangs on SM120 (jaxlib 0.10.0, NVIDIA JAX 25.04)
# See ~/.claude/rules/CLUSTER.md §6 and praxia lesson #41.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

source "${SCRIPT_DIR}/_bth_env.sh"

RESULTS_DIR="${PROLIX_ROOT}/outputs/results/lfmiddle_dt_sweep"
mkdir -p "${RESULTS_DIR}" outputs/logs/slurm

OUT="${RESULTS_DIR}/lfmiddle_dt${DT_FS}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.json"

echo "=== LFMiddle dt sweep ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "JAX_PLATFORMS=${JAX_PLATFORMS}  JAX_ENABLE_X64=${JAX_ENABLE_X64}"
echo "dt_fs=${DT_FS}  n_waters=${N_WATERS}  sim_ps=${SIM_PS}  burn_ps=${BURN_PS}"
echo "Campaign: ${CAMPAIGN_ID}"
echo "OUT: ${OUT}"

# PROLIX_RESULT_PATH used instead of --out to avoid collision with bth's own --out flag
export PROLIX_RESULT_PATH="${OUT}"

uv run bth run \
  --tag cluster --tag lfmiddle-dt-sweep --tag sprint5 \
  --tag "dt_fs:${DT_FS}" --tag "integrator:${INTEGRATOR}" \
  --campaign "${CAMPAIGN_ID}" \
  --out "${OUT}" \
  python scripts/experiments/lfmiddle_dt_sweep.py \
  --dt-fs "${DT_FS}" \
  --integrator "${INTEGRATOR}" \
  --n-waters "${N_WATERS}" \
  --sim-ps "${SIM_PS}" \
  --burn-ps "${BURN_PS}" \
  --seed "${SEED}"

EXIT=$?
echo "=== output: ${OUT} ==="
cat "${OUT}" 2>/dev/null || echo "(result file not found)"
echo "=== exit=${EXIT} ==="
exit "${EXIT}"
