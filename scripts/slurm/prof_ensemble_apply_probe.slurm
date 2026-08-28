#!/bin/bash
#SBATCH --job-name=prolix_ens_s2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-5%4
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_ensemble_apply_probe.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_ensemble_apply_probe.err

# Stage-2 L40S 1vii: force-only flash control + jitted apply_fn (flash/NL/dense).
# Explore / ungated — no bth sidecar.
#
#   0 force-only flash, buckets on (same pad as flash buckets-on MD)
#   1 flash buckets on
#   2 flash buckets off
#   3 nl buckets on
#   4 nl buckets off
#   5 dense buckets on
#
#   myxcel submit engaging prolix scripts/slurm/prof_ensemble_apply_probe.slurm \
#     --partition mit_normal_gpu --time 00:25:00 \
#     --sbatch-arg chdir=/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628 \
#     --worktree wt-20260807-132628

set -euo pipefail

_locate_root() {
    local cand
    for cand in \
        "${PWD}" \
        "${SLURM_SUBMIT_DIR:-}" \
        "${HOME}/projects/.worktrees/prolix/wt-20260807-132628" \
        "${HOME}/projects/prolix"
    do
        if [[ -n "${cand}" && -f "${cand}/scripts/slurm/_bth_env.sh" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    return 1
}
_PROJ="$(_locate_root)" || {
    echo "ERROR: cannot locate prolix root (PWD=${PWD} SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2
    exit 1
}
cd "${_PROJ}"
echo "cwd=${_PROJ}"
if [[ -z "${PROLIX_GIT_SHA:-}" && -f "${_PROJ}/.git_sha" ]]; then
    PROLIX_GIT_SHA="$(head -1 "${_PROJ}/.git_sha" | tr -d '[:space:]')"
    export PROLIX_GIT_SHA
fi
if [[ -z "${PROLIX_GIT_SHA:-}" ]]; then
    PROLIX_GIT_SHA="$(git -C "${_PROJ}" rev-parse HEAD 2>/dev/null || true)"
    export PROLIX_GIT_SHA
fi
echo "PROLIX_GIT_SHA=${PROLIX_GIT_SHA:-unset}"
echo "stage=2 gpu=l40s"

# Pin CUDA before any JAX import. If SLURM exposes multiple GPUs, use the first.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"

ARM=md
NONBONDED=flash
BUCKETS=on

case "${TASK}" in
    0) ARM=force; NONBONDED=flash; BUCKETS=on ;;
    1) ARM=md; NONBONDED=flash; BUCKETS=on ;;
    2) ARM=md; NONBONDED=flash; BUCKETS=off ;;
    3) ARM=md; NONBONDED=nl; BUCKETS=on ;;
    4) ARM=md; NONBONDED=nl; BUCKETS=off ;;
    5) ARM=md; NONBONDED=dense; BUCKETS=on ;;
    *) echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK out of range 0-5" >&2; exit 1 ;;
esac

TAG="l40s_1vii_${ARM}_${NONBONDED}_b${BUCKETS}"
OUTDIR="outputs/profiling/stage2_apply_fn/${TAG}"
RECORD="${OUTDIR}/record.json"
mkdir -p "${OUTDIR}" outputs/logs/slurm

echo "=== apply-fn probe ${TAG} task=${TASK} ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Array: ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "JAX_PLATFORMS=${JAX_PLATFORMS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Date: $(date -Is)"
uv run --extra cuda python -c "import jax; print('jax.default_backend=', jax.default_backend()); print('jax.devices=', jax.devices())"

set -o pipefail
uv run --extra cuda python scripts/explore/prof_ensemble_apply_probe.py \
    --system 1vii \
    --arm "${ARM}" \
    --nonbonded "${NONBONDED}" \
    --use-size-buckets "${BUCKETS}" \
    --emit-record "${RECORD}" \
    --trace-dir "${OUTDIR}/trace" \
    2>&1 | tee "${OUTDIR}/probe.log"

EXIT=${PIPESTATUS[0]}
echo "=== probe exit=${EXIT} ==="
exit ${EXIT}
