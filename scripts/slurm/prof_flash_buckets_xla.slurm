#!/bin/bash
#SBATCH --job-name=prolix_buck_xla
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3%2
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_flash_buckets_xla.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_flash_buckets_xla.err

# 2×2: size-buckets {on,off} × XLA recipe {default,intensive}.
# T=256 remat=on, 1vii flash PME-on 1.0 Å, command buffers ON.
# Task 0 is the control (buckets on, default XLA).
#
#   myxcel submit engaging prolix scripts/slurm/prof_flash_buckets_xla.slurm \
#     --partition mit_preemptable --time 00:20:00 \
#     --array 0-3%2 \
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
    echo "ERROR: cannot locate prolix root" >&2
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

source "scripts/slurm/_bth_env.sh"

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"
if (( TASK < 0 || TASK > 3 )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK out of range 0-3" >&2
    exit 1
fi
if (( TASK % 2 == 0 )); then
    BUCKETS=on
else
    BUCKETS=off
fi
if (( TASK / 2 == 0 )); then
    XLA=default
else
    XLA=intensive
fi
TAG="buckets_${BUCKETS}_xla_${XLA}"
OUTDIR="outputs/profiling/stage2/buckets_xla/${TAG}"
OUT="${OUTDIR}/summary.json"
mkdir -p "${OUTDIR}" outputs/logs/slurm

echo "=== ${TAG} task=${TASK} ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
echo "XLA_FLAGS_pre=${XLA_FLAGS:-}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"

uv run bth run python scripts/explore/prof_flash_tile_checkpoint.py \
    --no-sidecar \
    --tag buckets-xla-sweep \
    --tag "protein:1vii" \
    --tag "buckets:${BUCKETS}" \
    --tag "xla:${XLA}" \
    --out "${OUT}" \
    -- \
    --protein 1vii \
    --pme on \
    --grid-spacing 1.0 \
    --tile 256 \
    --remat on \
    --use-size-buckets "${BUCKETS}" \
    --xla-recipe "${XLA}" \
    --n-warmup 3 --n-trials 20 --n-inner 5 \
    --out "${OUT}"

EXIT=$?
echo "=== probe exit=${EXIT} ==="
exit ${EXIT}
