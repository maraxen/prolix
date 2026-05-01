#!/bin/bash
# Same as ``smoke_batched_simulate_cpu.slurm`` but longer wall time on ``mit_preemptable``
# (cold compile headroom, large graphs).
#
#SBATCH --job-name=plx-batched-smoke
#SBATCH --partition=mit_preemptable
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

LOG="${LOG_ROOT}/app/batched_simulate_smoke_preempt_${SLURM_JOB_ID:-local}.log"
echo "log=${LOG} host=$(hostname) date=$(date -Is)"

uv run pytest tests/physics/test_batched_simulate.py \
  tests/physics/test_md_potential_bundle_parity.py \
  tests/physics/test_step_result.py \
  -q --tb=short 2>&1 | tee "${LOG}"

echo "OK batched smoke (preemptable) finished"
