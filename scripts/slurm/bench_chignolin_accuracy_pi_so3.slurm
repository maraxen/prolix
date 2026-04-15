#!/bin/bash
# Accuracy-only step: pytest subset on CPU. Submit before ``bench_chignolin_speed_pi_so3.slurm``
# or use ``scripts/slurm/submit_chignolin_split_pi_so3.sh`` for chained jobs.
#
#SBATCH --job-name=plx-ch-acc
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" uv run pytest tests/physics/test_pbc_end_to_end.py -m "not slow" -q --tb=short \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_accuracy_${SLURM_JOB_ID:-local}.log"
