#!/bin/bash
# Phase 6 VERIFY follow-up: capture exact failure reason for
# test_temperature_csvr_dt1fs_near_target, which reported plain FAILED (not XFAIL)
# under xfail(strict=True) in job 20248769 -- almost certainly an unexpected XPASS,
# but need the exact pytest summary line as hard evidence, not inference.
#SBATCH --job-name=p6-csvr-test1-reason
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --time=2:00:00
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

echo "p6-csvr-test1-reason job=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$(pwd)"

uv run pytest \
  tests/physics/test_settle_temperature_control.py::test_temperature_csvr_dt1fs_near_target \
  -v -rA --tb=long --timeout=5400
