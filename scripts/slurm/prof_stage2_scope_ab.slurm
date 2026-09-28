#!/bin/bash
#SBATCH --job-name=prolix_p7_scope_ab
#SBATCH --partition=mit_preemptable,pi_so3
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%j.prof_stage2_scope_ab.out
#SBATCH --error=outputs/logs/slurm/%j.prof_stage2_scope_ab.err

# P7 task 12: scopes on/off A/B (1vii, flash, PME on, 1.0 Å).
# --time is a placeholder; override at submit from the first array-task log:
#   round_up_to_5min(2 * n_pairs * (t_gpu_child + t_compile_child)) + 15 min
# If that exceeds 12h, reduce n_pairs (cap 200) and record INSUFFICIENT_POWER.

set -euo pipefail

# myxcel sbatch cwd is $HOME, so SLURM_SUBMIT_DIR is not the repo.
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
echo "P7 cwd=${_PROJ}"
if [[ -z "${PROLIX_GIT_SHA:-}" && -f "${_PROJ}/.git_sha" ]]; then
    PROLIX_GIT_SHA="$(head -1 "${_PROJ}/.git_sha" | tr -d '[:space:]')"
    export PROLIX_GIT_SHA
fi
if [[ -z "${PROLIX_GIT_SHA:-}" ]]; then
    PROLIX_GIT_SHA="$(git -C "${_PROJ}" rev-parse HEAD 2>/dev/null || true)"
    export PROLIX_GIT_SHA
fi
echo "PROLIX_GIT_SHA=${PROLIX_GIT_SHA:-unset}"
# xtrax.profiling.record reads XTRAX_GIT_SHA, not PROLIX_GIT_SHA -- the var was
# renamed when scripts/profiling/ moved upstream. Export both: the local scripts
# with their own _git_sha() (prof_ensemble_step.py, prof_flash_tile_checkpoint.py)
# still read PROLIX_GIT_SHA. A missing XTRAX_GIT_SHA does not error -- it silently
# records a blank sha on cluster scratch, where there is no .git to fall back to.
export XTRAX_GIT_SHA="${PROLIX_GIT_SHA:-}"
source "scripts/slurm/_bth_env.sh"

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
CAMPAIGN_FLAG=()
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG=(--campaign "${CAMPAIGN_ID}")
N_PAIRS_FLAG=()
[[ -n "${N_PAIRS:-}" ]] && N_PAIRS_FLAG=(--n-pairs "${N_PAIRS}")

mkdir -p outputs/profiling/stage2/scope_ab outputs/logs/slurm

echo "=== P7 scope A/B ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

uv run bth run python scripts/experiments/prof_stage2_scope_ab.py \
    --no-sidecar \
    --tag p7-stage2 --tag scope-ab \
    "${CAMPAIGN_FLAG[@]}" \
    --out outputs/profiling/stage2/scope_ab/summary.json \
    -- \
    --out-dir outputs/profiling/stage2/scope_ab \
    --out outputs/profiling/stage2/scope_ab/summary.json \
    "${N_PAIRS_FLAG[@]}"

EXIT=$?
echo "=== exit=${EXIT} ==="
exit ${EXIT}
