#!/bin/bash
# One-shot Engaging smoke: batched implicit pipeline + dtype-stable loops (CPU).
# Submit after ``just push-engaging`` (or ``push-engaging-clean`` if lock changed).
#
#SBATCH --job-name=plx-batched-smoke
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:14:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Anchor paths to this script — safe from any submission cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common_env.sh"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

LOG="${LOG_ROOT}/app/batched_simulate_smoke_${SLURM_JOB_ID:-local}.log"
echo "log=${LOG} host=$(hostname) date=$(date -Is)"

uv run pytest tests/physics/test_batched_simulate.py \
  tests/physics/test_md_potential_bundle_parity.py \
  tests/physics/test_step_result.py \
  -q --tb=short 2>&1 | tee "${LOG}"

echo "OK batched smoke finished"
