#!/bin/bash
#SBATCH --job-name=prolix_tile_ckpt
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-5%3
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_flash_tile_checkpoint.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_flash_tile_checkpoint.err

# Discovery 2×3: T ∈ {128,256,512} × remat on/off. Command buffers ON.
# Task 2 (T=256 remat=on) is the control (~1.076 ms cmdbuf-on P7 flash cell).
#
#   myxcel submit engaging prolix scripts/slurm/prof_flash_tile_checkpoint.slurm \
#     --partition mit_preemptable --time 00:15:00 \
#     --sbatch-arg chdir=/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628

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
echo "XLA_FLAGS=${XLA_FLAGS:-}"

source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"
if (( TASK < 0 || TASK > 5 )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK out of range 0-5" >&2
    exit 1
fi
TILES=(128 256 512)
TILE="${TILES[TASK / 2]}"
if (( TASK % 2 == 0 )); then
    REMAT=on
else
    REMAT=off
fi
TAG="t${TILE}_remat_${REMAT}"
OUTDIR="outputs/profiling/stage2/tile_checkpoint/${TAG}"
OUT="${OUTDIR}/summary.json"
mkdir -p "${OUTDIR}" outputs/logs/slurm

echo "=== tile-checkpoint ${TAG} task=${TASK} ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Array: ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

uv run bth run python scripts/explore/prof_flash_tile_checkpoint.py \
    --no-sidecar \
    --tag tile-checkpoint-sweep \
    --tag "protein:1vii" \
    --tag "mode:flash" \
    --tag "pme:on" \
    --tag "grid:1.0" \
    --tag "tile:${TILE}" \
    --tag "remat:${REMAT}" \
    --out "${OUT}" \
    -- \
    --protein 1vii \
    --pme on \
    --grid-spacing 1.0 \
    --tile "${TILE}" \
    --remat "${REMAT}" \
    --n-warmup 3 --n-trials 20 --n-inner 5 \
    --out "${OUT}"

EXIT=$?
echo "=== probe exit=${EXIT} ==="
exit ${EXIT}
