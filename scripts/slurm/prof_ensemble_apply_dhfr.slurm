#!/bin/bash
#SBATCH --job-name=prolix_s2_dhfr
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_apply_dhfr.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_apply_dhfr.err

# DHFR JAC (~23.5k atoms) Stage-2 apply_fn on H200. No dense (OOM even here).
#   0 force-only flash buckets on
#   1 md flash buckets on
#   2 md nl buckets on
#   3 md nl buckets off

set -euo pipefail

_locate_root() {
    local cand
    for cand in "${PWD}" "${SLURM_SUBMIT_DIR:-}" \
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
_PROJ="$(_locate_root)" || exit 1
cd "${_PROJ}"
if [[ -z "${PROLIX_GIT_SHA:-}" && -f "${_PROJ}/.git_sha" ]]; then
    PROLIX_GIT_SHA="$(head -1 "${_PROJ}/.git_sha" | tr -d '[:space:]')"
    export PROLIX_GIT_SHA
fi
# xtrax.profiling.record reads XTRAX_GIT_SHA, not PROLIX_GIT_SHA -- the var was
# renamed when scripts/profiling/ moved upstream. Export both: the local scripts
# with their own _git_sha() (prof_ensemble_step.py, prof_flash_tile_checkpoint.py)
# still read PROLIX_GIT_SHA. A missing XTRAX_GIT_SHA does not error -- it silently
# records a blank sha on cluster scratch, where there is no .git to fall back to.
export XTRAX_GIT_SHA="${PROLIX_GIT_SHA:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# PROLIX_CMDBUF_OFF=1 explodes the CUDA-graph command buffer into separately
# timed thunks. Without it, XLA folds the whole step into one command_buffer_*
# thunk and no per-component ranking exists to recover -- which is why no
# DHFR-scale breakdown has ever been produced. DIAGNOSTIC ONLY: never pool these
# timings with a normally-compiled run (the probe warns when the top thunk is a
# command_buffer_* entry, and TAG below keeps the records in separate dirs).
CMDBUF_SUFFIX=""
if [[ "${PROLIX_CMDBUF_OFF:-0}" == "1" ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
    CMDBUF_SUFFIX="_cmdbufoff"
    echo "cmdbuf-off diagnostic compile: XLA_FLAGS=${XLA_FLAGS}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi
source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?}"
ARM=md
NONBONDED=flash
BUCKETS=on
case "${TASK}" in
    0) ARM=force; NONBONDED=flash; BUCKETS=on ;;
    1) ARM=md; NONBONDED=flash; BUCKETS=on ;;
    2) ARM=md; NONBONDED=nl; BUCKETS=on ;;
    3) ARM=md; NONBONDED=nl; BUCKETS=off ;;
    *) exit 1 ;;
esac

TAG="h200_dhfr_${ARM}_${NONBONDED}_b${BUCKETS}${CMDBUF_SUFFIX}"
OUTDIR="outputs/profiling/stage2_apply_fn/${TAG}"
RECORD="${OUTDIR}/record.json"
mkdir -p "${OUTDIR}" outputs/logs/slurm
echo "=== DHFR apply-fn ${TAG} ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
uv run --extra cuda python -c "import jax; print('jax.default_backend=', jax.default_backend()); print('jax.devices=', jax.devices())"
set -o pipefail
uv run --extra cuda python scripts/explore/prof_ensemble_apply_probe.py \
    --system dhfr \
    --arm "${ARM}" \
    --nonbonded "${NONBONDED}" \
    --use-size-buckets "${BUCKETS}" \
    --emit-record "${RECORD}" \
    --trace-dir "${OUTDIR}/trace" \
    2>&1 | tee "${OUTDIR}/probe.log"
EXIT=${PIPESTATUS[0]}
echo "=== probe exit=${EXIT} ==="
exit ${EXIT}
