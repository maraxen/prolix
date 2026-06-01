#!/bin/bash
#SBATCH --job-name=plx-T-profile
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.temp_profile_diag.out
#SBATCH --error=outputs/logs/slurm/%j.temp_profile_diag.err
set -euo pipefail
_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
PROLIX_ROOT="$(git -C "${_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null || echo "${HOME}/projects/prolix")"
cd "${PROLIX_ROOT}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PROLIX_RESULT_PATH="${PROLIX_ROOT}/outputs/results/temp_profile_${SLURM_JOB_ID}.json"
mkdir -p "${PROLIX_ROOT}/outputs/results"
uv run python scripts/experiments/temp_profile_diag.py
