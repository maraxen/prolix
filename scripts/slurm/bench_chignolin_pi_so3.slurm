#!/bin/bash
# Chignolin-phase smoke on the MIT ``pi_so3`` GPU partition (short wall time).
# Combined accuracy + throughput in one allocation. To run pytest (CPU) first then GPU
# benchmarks only after success, use ``bench_chignolin_accuracy_pi_so3.slurm`` +
# ``bench_chignolin_speed_pi_so3.slurm`` or ``scripts/slurm/submit_chignolin_split_pi_so3.sh``.
#
# Recommended submit (dated log tree):
#   export ENGAGING_LOG_DATE=$(date +%Y%m%d)
#   mkdir -p "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm"
#   sbatch -o "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/%x_%j.out" \
#          -e "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/%x_%j.err" \
#          scripts/slurm/bench_chignolin_pi_so3.slurm
#
#SBATCH --job-name=plx-ch-piso3
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

# --- Accuracy (CPU-friendly pytest subset; tune -m / paths for your budget) ---
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" uv run pytest tests/physics/test_pbc_end_to_end.py -m "not slow" -q --tb=short \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_accuracy_${SLURM_JOB_ID:-local}.log"

# --- Throughput / driver checks (GPU) ---
uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --skip-reference \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_speed_openmm_${SLURM_JOB_ID:-local}.log"

uv run python scripts/benchmark_nlvsdense.py --systems CHIGNOLIN 1X2G --replicas 1 \
  2>&1 | tee "${LOG_ROOT}/app/chignolin_nlvsdense_${SLURM_JOB_ID:-local}.log"

# Optional: one JAX profile trace under app/
# uv run python scripts/profile_step.py 2>&1 | tee "${LOG_ROOT}/app/profile_${SLURM_JOB_ID:-local}.log"
