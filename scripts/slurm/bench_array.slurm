#!/bin/bash
# Example array job: encode axes in SLURM_ARRAY_TASK_ID (cap concurrency with %%K).
#
# Submit:
#   export ENGAGING_LOG_DATE=$(date +%Y%m%d)
#   mkdir -p "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm"
#   sbatch --array=0-7%4 \
#     -o "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/%x_%A_%a.out" \
#     -e "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/%x_%A_%a.err" \
#     scripts/slurm/bench_array.slurm
#
#SBATCH --job-name=plx-array
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7%4

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

TASK="${SLURM_ARRAY_TASK_ID:-0}"
echo "array_task=${TASK} job=${SLURM_JOB_ID:-}"

# Example mapping: vary pytest file or benchmark flag by task id.
case "${TASK}" in
  0) uv run python scripts/benchmark_nlvsdense.py --systems CHIGNOLIN --replicas 1 ;;
  1) uv run python scripts/benchmark_nlvsdense.py --systems 1X2G --replicas 1 ;;
  2) uv run python scripts/benchmark_gb_forces.py ;;
  3) uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --skip-reference ;;
  *) echo "Task ${TASK}: add a sweep axis (cutoff, electrostatic method, x64) here." ;;
esac
