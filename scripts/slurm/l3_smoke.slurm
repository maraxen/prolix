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
# bath-tracked: provenance via `bth run`; sidecar at
#   scripts/validation/l3_smoke_load_bundle.bth.toml
#
# Prereqs:
#   1. Local L1 + L2 of fetch_ani1x_subset.py
#   2. data/ani1x_subset/ rsync'd to ~/projects/prolix/data/ani1x_subset/
#   3. outputs/logs/slurm/ exists on remote
#   4. bath catalog accessible (defaults to ~/.bth/catalog/)
#
# Submission (from local prolix dir, transparent SSH per CLUSTER.md §0):
#   mkdir -p outputs/logs/slurm
#   sbatch scripts/slurm/l3_smoke.slurm
#
# Expected wallclock: ~30 seconds on mit_quicktest CPU.

set -euo pipefail

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

# Load bath + cluster environment (BTH_PROJECT_SLUG, LOG_ROOT, etc.)
source "scripts/slurm/_bth_env.sh"

echo "=== L3 smoke ==="
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "bath project: ${BTH_PROJECT_SLUG}"
echo "Date: $(date -Is)"

mkdir -p outputs/results

DATA_PATH="data/ani1x_subset/lane_a/mol_000.h5"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: ${DATA_PATH} not found on cluster."
    echo "Run scripts/data/fetch_ani1x_subset.py locally, then rsync data/ani1x_subset/ here."
    exit 1
fi

# bath run sets BTH_RESULTS_PATH env var for the subprocess; the Python script
# defaults --out-json to that path, enabling sidecar outcome evaluation.
# Don't pass --out-json explicitly — that would override BTH_RESULTS_PATH.
uv run bth run python scripts/validation/l3_smoke_load_bundle.py \
    --tag cluster --tag hp4 --tag l3 \
    -- \
    --data-path "${DATA_PATH}"

EXIT=$?
echo "=== L3 smoke exit=${EXIT} ==="
exit ${EXIT}
