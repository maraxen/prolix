#!/bin/bash
#SBATCH --job-name=dhfr_stab
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/logs/slurm/%j.dhfr_stab.out
#SBATCH --error=outputs/logs/slurm/%j.dhfr_stab.err

# Is the DHFR NVT trajectory physical? (task 260903_dhfr_nl_capacity)
#
# Three legs in one job, so the discriminating matrix costs one queue wait:
#   1. NL    @ dt=1.0 fs -- reproduces the observed overflow
#   2. flash @ dt=1.0 fs -- same system, other force path, no neighbor list
#   3. NL    @ dt=0.5 fs -- same path, half the timestep
#
# Reading:
#   flash stable, NL collapses      -> NL-branch force bug (exclusions/PME)
#   both collapse, dt=0.5 stable    -> timestep limit at this scale, not a defect
#   everything collapses            -> the system build is wrong
#
# Each leg runs to completion independently; a leg that raises records the
# exception as its result rather than killing the job, because a failure IS the
# datum here.

set -uo pipefail
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
mkdir -p outputs/logs/slurm outputs/dhfr_stability
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
OUT_DIR="outputs/dhfr_stability"
CHUNK="${CHUNK_STEPS:-20}"
NCHUNK="${N_CHUNKS:-12}"

run_leg() {
    local name="$1" path="$2" dt="$3"
    echo "=== leg ${name}: force_path=${path} dt=${dt} ==="
    uv run --extra cuda python scripts/explore/diag_dhfr_nvt_stability.py \
        --force-path "${path}" \
        --dt "${dt}" \
        --chunk-steps "${CHUNK}" \
        --n-chunks "${NCHUNK}" \
        --out "${OUT_DIR}/${name}_${TAG}.json"
    echo "--- leg ${name} exit=$? ---"
}

run_leg nl_dt1p0    nl    1.0
run_leg flash_dt1p0 flash 1.0
run_leg nl_dt0p5    nl    0.5

echo "=== done: results in ${OUT_DIR}/*_${TAG}.json ==="
