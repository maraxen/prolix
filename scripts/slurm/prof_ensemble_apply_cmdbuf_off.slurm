#!/bin/bash
#SBATCH --job-name=prolix_s2_cmdbuf
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_apply_cmdbuf.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_apply_cmdbuf.err

# Diagnostic: command-buffer off. Different program than production ranking.
#   0 force-only flash 1vii
#   1 md flash 1vii buckets on

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
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi
source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?}"
ARM=force
case "${TASK}" in
    0) ARM=force ;;
    1) ARM=md ;;
    *) exit 1 ;;
esac

TAG="l40s_1vii_${ARM}_flash_bon_cmdbufoff"
OUTDIR="outputs/profiling/stage2_apply_fn/${TAG}"
RECORD="${OUTDIR}/record.json"
mkdir -p "${OUTDIR}" outputs/logs/slurm
echo "=== cmdbuf-off ${TAG} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
uv run --extra cuda python scripts/explore/prof_ensemble_apply_probe.py \
    --system 1vii \
    --arm "${ARM}" \
    --nonbonded flash \
    --use-size-buckets on \
    --cmdbuf-off \
    --emit-record "${RECORD}" \
    --trace-dir "${OUTDIR}/trace" \
    2>&1 | tee "${OUTDIR}/probe.log"
