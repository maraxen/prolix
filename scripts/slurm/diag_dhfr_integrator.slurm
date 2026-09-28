#!/bin/bash
#SBATCH --job-name=dhfr_integ
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/logs/slurm/%j.dhfr_integ.out
#SBATCH --error=outputs/logs/slurm/%j.dhfr_integ.err

# #4953: does one integrator step move atoms by 0.5*(F/m)*dt^2?
#
# Forces are verified correct (0.039% vs OpenMM, job 21967760) and the box bug is
# fixed (61de3d2), yet PE reaches 2.7e15 kcal/mol within 20 steps. This checks the
# integrator against Newton directly -- no second engine needed -- from rest with
# the thermostat fully off (kT=0, gamma=0). The deviation is a pure number and its
# value names the defect (mass-valued -> momentum/velocity confusion; constant ->
# a dt or unit factor).
#
# Both force paths, and 1 vs 2 steps: a defect in the very first step shows up as a
# ratio that changes with step count, a persistent one as a ratio that does not.

set -uo pipefail
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
_is_repo_root "${_PROJ}" || { echo "FATAL: no prolix repo root found (PWD=${PWD}, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2; exit 1; }
cd "${_PROJ}" || exit 1
echo "repo root: $(pwd)"
mkdir -p outputs/logs/slurm outputs/dhfr_integrator
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
OUT="outputs/dhfr_integrator"

run_leg() {
    local path="$1" steps="$2"
    echo "=== leg ${path} steps=${steps} ==="
    uv run --extra cuda python scripts/explore/diag_dhfr_integrator_step.py \
        --force-path "${path}" --dt 1.0 --steps "${steps}" \
        --out "${OUT}/${path}_s${steps}_${TAG}.json"
    echo "--- leg ${path} steps=${steps} exit=$? ---"
}

run_leg nl 1
run_leg nl 2
run_leg flash 1

echo "=== done: ${OUT}/*_${TAG}.json ==="
