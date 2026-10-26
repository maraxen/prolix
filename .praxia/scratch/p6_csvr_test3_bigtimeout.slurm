#!/bin/bash
# Phase 6 VERIFY follow-up: re-run test_equipartition_csvr_dt2fs_chi2 (500ps, the
# largest CSVR xfail test) alone with a much larger pytest timeout -- job 20248769
# hit the 3600s per-test ceiling mid-way through JAX->Python velocity extraction,
# which looked like genuine slowness (past the compute loop), not a hang.
#SBATCH --job-name=p6-csvr-test3-bigtimeout
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

_PROJ="/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628"
cd "${_PROJ}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4 inter_op_parallelism_threads=4"

echo "p6-csvr-test3-bigtimeout job=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$(pwd)"

uv run pytest \
  tests/physics/test_settle_temperature_control.py::test_equipartition_csvr_dt2fs_chi2 \
  -v -rA --tb=long --timeout=18000
