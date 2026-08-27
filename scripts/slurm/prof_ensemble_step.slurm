#!/bin/bash
#SBATCH --job-name=prolix_ens_s1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-6%4
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_ensemble_step.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_ensemble_step.err

# Campaign E bd1f47e8 — Stage 1 on L40S (48 GB). 1vii full EnsemblePlan inference.
# DHFR dense is out of scope (OOMs even on H200). DHFR flash/NL is Stage 2
# (separate array later) if Stage 1 NL-vs-flash ranking is clean.
#
#   0 A 1vii flash buckets-on f32 remat-on
#   1 B 1vii flash buckets-off
#   2 C 1vii nl buckets-on
#   3 D 1vii nl buckets-off
#   4 E 1vii dense buckets-on
#   5 F 1vii flash remat-off
#   6 G 1vii flash f64
#
#   myxcel submit engaging prolix scripts/slurm/prof_ensemble_step.slurm \
#     --partition mit_normal_gpu --time 00:25:00 \
#     --sbatch-arg chdir=/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628 \
#     --worktree wt-20260807-132628

set -euo pipefail

CAMPAIGN_ID="${CAMPAIGN_ID:-bd1f47e8}"

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
echo "CAMPAIGN_ID=${CAMPAIGN_ID}"
echo "stage=1 gpu=l40s"

# Pin CUDA before any JAX import. This worktree .venv is often synced without
# the cuda extra (cpu+tpu only); --extra cuda pulls jax[cuda12].
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"

SYSTEM=1vii
NONBONDED=flash
BUCKETS=on
REMAT=on
DTYPE=f32
NSTEPS="2,4,8"

case "${TASK}" in
    0) SYSTEM=1vii; NONBONDED=flash; BUCKETS=on; REMAT=on; DTYPE=f32 ;;
    1) SYSTEM=1vii; NONBONDED=flash; BUCKETS=off; REMAT=on; DTYPE=f32 ;;
    2) SYSTEM=1vii; NONBONDED=nl; BUCKETS=on; REMAT=on; DTYPE=f32 ;;
    3) SYSTEM=1vii; NONBONDED=nl; BUCKETS=off; REMAT=on; DTYPE=f32 ;;
    4) SYSTEM=1vii; NONBONDED=dense; BUCKETS=on; REMAT=on; DTYPE=f32 ;;
    5) SYSTEM=1vii; NONBONDED=flash; BUCKETS=on; REMAT=off; DTYPE=f32 ;;
    6) SYSTEM=1vii; NONBONDED=flash; BUCKETS=on; REMAT=on; DTYPE=f64 ;;
    *) echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK out of range 0-6 (stage 1 1vii/L40S)" >&2; exit 1 ;;
esac

TAG="l40s_${SYSTEM}_${NONBONDED}_b${BUCKETS}_r${REMAT}_${DTYPE}"
OUTDIR="outputs/profiling/full_md_path_matrix/${TAG}"
OUT="${OUTDIR}/summary.json"
mkdir -p "${OUTDIR}" outputs/logs/slurm

echo "=== ensemble-step ${TAG} task=${TASK} ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Array: ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "JAX_PLATFORMS=${JAX_PLATFORMS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Date: $(date -Is)"
uv run --extra cuda python -c "import jax; print('jax.default_backend=', jax.default_backend()); print('jax.devices=', jax.devices())"

PROBE=none
if [[ "${NONBONDED}" == "nl" ]]; then
    PROBE=nl-stats
fi

uv run --extra cuda bth run python scripts/experiments/prof_ensemble_step.py \
    --tag cluster \
    --tag full-md-path-matrix \
    --tag stage1-l40s \
    --campaign "${CAMPAIGN_ID}" \
    --out "${OUT}" \
    -- \
    --system "${SYSTEM}" \
    --nonbonded "${NONBONDED}" \
    --use-size-buckets "${BUCKETS}" \
    --remat "${REMAT}" \
    --dtype "${DTYPE}" \
    --n-steps-list "${NSTEPS}" \
    --probe "${PROBE}" \
    --out "${OUT}"

EXIT=$?
echo "=== probe exit=${EXIT} ==="
exit ${EXIT}
