#!/bin/bash
#SBATCH --job-name=prolix_l3_smoke
#SBATCH --partition=mit_quicktest
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=outputs/logs/slurm/%j.l3_smoke.out
#SBATCH --error=outputs/logs/slurm/%j.l3_smoke.err

# L3 cluster smoke for HP4 curated subset (CLUSTER.md §7 Gate L3).
#
# Prereqs (must be done before this job is submitted):
#   1. Local L1 + L2 of fetch_ani1x_subset.py (with --dry-run + real run)
#   2. data/ani1x_subset/ rsync'd to ~/projects/prolix/data/ani1x_subset/
#   3. outputs/logs/slurm/ exists on remote
#
# Submission (from local prolix dir, transparent SSH per CLUSTER.md §0):
#   mkdir -p outputs/logs/slurm
#   sbatch scripts/cluster/l3_smoke.sbatch
#
# Expected wallclock: ~30 seconds on mit_quicktest CPU.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

echo "=== L3 smoke ==="
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Date: $(date -Is)"

mkdir -p outputs/results

# Run smoke on Lane A first molecule (deterministic from SELECT_16 seed=42).
DATA_PATH="data/ani1x_subset/lane_a/mol_000.h5"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: ${DATA_PATH} not found on cluster."
    echo "Run scripts/data/fetch_ani1x_subset.py locally, then rsync data/ani1x_subset/ here."
    exit 1
fi

uv run python scripts/cluster/l3_smoke_load_bundle.py \
    --data-path "${DATA_PATH}" \
    --out-json "outputs/results/l3_smoke_${SLURM_JOB_ID:-local}.json"

EXIT=$?
echo "=== L3 smoke exit=${EXIT} ==="
exit ${EXIT}
