#!/bin/bash
#SBATCH --job-name=p5_nvt_gate_dt1fs
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/slurm/%j.p5_nvt_gate_dt1fs.out
#SBATCH --error=outputs/logs/slurm/%j.p5_nvt_gate_dt1fs.err

# P5 NVT gate at dt=1.0 fs — extended timestep validation.
# Tests whether settle_langevin can achieve stable T_rot at dt=1.0 fs
# on a 895-water system with gamma=10 ps^-1 coupling strength.
#
# Submission with campaign ID (required):
#   sbatch --export=ALL,CAMPAIGN_ID=<campaign-id> scripts/slurm/p5_nvt_gate_dt1fs.sbatch

set -euo pipefail

# Use SLURM_SUBMIT_DIR for durable project root (survives transient worktree submit dirs).
_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if PROLIX_ROOT="$(git -C "${_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  PROLIX_ROOT="${HOME}/projects/prolix"
fi
cd "${PROLIX_ROOT}"

echo "=== P5 NVT gate dt=1.0 fs ==="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

RESULTS_DIR="${PROLIX_ROOT}/outputs/results"
LOGS_DIR="${PROLIX_ROOT}/outputs/logs/slurm"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

OUT="${RESULTS_DIR}/p5_nvt_gate_dt1fs_${SLURM_JOB_ID}.json"

# JAX on Blackwell (node4007/node4008, SM120): XLA autotuning hangs without this flag.
# Documented in NVIDIA JAX 25.04 release notes; confirmed 1170x speedup on job 15294138.
# Key on hostname/nodelist, not partition — pi_so3 may gain non-Blackwell nodes.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active (${_NODE}): XLA_FLAGS=${XLA_FLAGS}"
fi

export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

# bathos provenance: per-script sidecar at scripts/experiments/p5_nvt_gate_dt1fs.bth.toml
source "scripts/slurm/_bth_env.sh"

CAMPAIGN_ID="${CAMPAIGN_ID:?CAMPAIGN_ID env var required — pass via sbatch --export=ALL,CAMPAIGN_ID=<id>}"

uv run bth run python scripts/experiments/p5_nvt_gate_dt1fs.py \
    --tag p5 --tag nvt-gate --tag dt1fs \
    --campaign "${CAMPAIGN_ID}" \
    --out "${OUT}"

EXIT=$?
echo "=== output: ${OUT} ==="
cat "${OUT}" 2>/dev/null
echo
echo "=== exit=${EXIT} ==="
exit ${EXIT}
