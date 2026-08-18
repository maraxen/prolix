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

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"
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
