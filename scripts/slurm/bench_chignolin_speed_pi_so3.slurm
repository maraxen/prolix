#!/bin/bash
# Throughput-only step: OpenMM speed sweep + NL vs dense (GPU). Pair with accuracy job or
# submit with ``sbatch --dependency=afterok:<accuracy_jobid>`` after ``bench_chignolin_accuracy_pi_so3.slurm``.
#
#SBATCH --job-name=plx-ch-spd
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --skip-reference \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_speed_openmm_${SLURM_JOB_ID:-local}.log"

uv run python scripts/benchmark_nlvsdense.py --systems CHIGNOLIN 1X2G --replicas 1 \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_nlvsdense_${SLURM_JOB_ID:-local}.log"
