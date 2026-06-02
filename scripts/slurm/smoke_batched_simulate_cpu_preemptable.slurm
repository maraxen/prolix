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

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

# shellcheck source=/dev/null
source "scripts/slurm/_common_env.sh"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

LOG="${LOG_ROOT}/app/batched_simulate_smoke_preempt_${SLURM_JOB_ID:-local}.log"
echo "log=${LOG} host=$(hostname) date=$(date -Is)"

uv run pytest tests/physics/test_batched_simulate.py \
  tests/physics/test_md_potential_bundle_parity.py \
  tests/physics/test_step_result.py \
  -q --tb=short 2>&1 | tee "${LOG}"

echo "OK batched smoke (preemptable) finished"
