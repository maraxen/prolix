#!/bin/bash
#SBATCH --job-name=prolix_p7_coverage
#SBATCH --partition=mit_normal
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.prof_stage2_coverage.out
#SBATCH --error=outputs/logs/slurm/%j.prof_stage2_coverage.err

# P7 coverage aggregation. Submit with --dependency=afterany:<array_job_id>
# (afterany, not afterok: partial coverage is the metric).

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
source "scripts/slurm/_bth_env.sh"

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
CAMPAIGN_FLAG=()
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG=(--campaign "${CAMPAIGN_ID}")

mkdir -p outputs/profiling/stage2 outputs/logs/slurm

echo "=== P7 coverage ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
echo "Date: $(date -Is)"

uv run bth run python scripts/experiments/prof_stage2_coverage.py \
    --tag p7-stage2 --tag coverage \
    "${CAMPAIGN_FLAG[@]}" \
    --out outputs/profiling/stage2/coverage.json \
    -- \
    --records-dir outputs/profiling/stage2 \
    --campaign-tag p7-stage2 \
    --out outputs/profiling/stage2/coverage.json

EXIT=$?
echo "=== exit=${EXIT} ==="
exit ${EXIT}
