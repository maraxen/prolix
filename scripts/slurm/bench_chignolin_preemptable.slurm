#!/bin/bash
# Longer wall time on ``mit_preemptable`` — prefer checkpointed / resumable workflows
# if your site preempts jobs (see explicit solvent implementation plan).
#
# Submit with explicit dated logs (same pattern as ``bench_chignolin_pi_so3.slurm``).
#
#SBATCH --job-name=plx-ch-pre
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

# Same workload as pi_so3 template; extend with sweeps or DHFR when Phase 2 is ready.
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" uv run pytest tests/physics/test_pbc_end_to_end.py -m "not slow" -q --tb=short \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_accuracy_${SLURM_JOB_ID:-local}.log"

uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --skip-reference \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_speed_openmm_${SLURM_JOB_ID:-local}.log"

uv run python scripts/benchmark_nlvsdense.py --systems CHIGNOLIN 1X2G --replicas 2 \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_nlvsdense_${SLURM_JOB_ID:-local}.log"
