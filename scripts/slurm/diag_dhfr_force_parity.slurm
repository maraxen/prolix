#!/bin/bash
#SBATCH --job-name=dhfr_fparity
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/logs/slurm/%j.dhfr_fparity.out
#SBATCH --error=outputs/logs/slurm/%j.dhfr_fparity.err

# T0: localise #4953 by measurement (task 260903_dhfr_divergence_parity).
#
# Single-point energy + per-atom force parity, prolix vs OpenMM, on the identical
# DHFR configuration. Two legs, because agreement BETWEEN them is itself a datum:
#   dense -- O(N^2) direct-space kernel
#   nl    -- production neighbor-list kernel (the path the benchmark times)
#
# Reading:
#   nonbonded cosine ~1, ratio ~1   -> forces are right; #4953 is in the integrator
#   cosine ~1, ratio far from 1     -> right direction, wrong scale: units/prefactor
#   cosine << 1                     -> wrong forces; the term named is the culprit
#   dense and nl differ from each other -> a kernel bug on top of everything else
#
# A leg that raises still records its exception rather than killing the job -- a
# failure is a datum here too.

set -uo pipefail
# SLURM_SUBMIT_DIR is where sbatch was INVOKED, not the job cwd. Validate each
# candidate rather than trusting either (job 21804954 ran entirely outside the uv
# project by trusting it).
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
_is_repo_root "${_PROJ}" || { echo "FATAL: no prolix repo root found (tried PWD=${PWD}, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2; exit 1; }
cd "${_PROJ}" || exit 1
echo "repo root: $(pwd)"
mkdir -p outputs/logs/slurm outputs/dhfr_force_parity
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    # XLA autotuning hangs on SM120 (Blackwell); 1170x difference. See CLUSTER.md.
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="outputs/dhfr_force_parity"

run_leg() {
    local path="$1"
    echo "=== leg ${path} ==="
    uv run --extra cuda python scripts/explore/diag_dhfr_force_parity.py \
        --force-path "${path}" \
        --out "${OUT_DIR}/${path}_${TAG}.json"
    echo "--- leg ${path} exit=$? ---"
}

run_leg nl
run_leg dense

echo "=== done: results in ${OUT_DIR}/*_${TAG}.json ==="
