#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=sprint_b_step2_impulse
#SBATCH --output=outputs/logs/sprint_b_step2_constraint_impulse_%A_%a.out
#SBATCH --error=outputs/logs/sprint_b_step2_constraint_impulse_%A_%a.err

set -euo pipefail

cd ~/projects/prolix

echo "=== Sprint B Step 2: Constraint-Impulse Scaling ==="
echo "Matrix via job array: n_waters=[8,16,32,64], dt=[0.5,0.25], + 64-water OU controls"
echo "Starting at: $(date)"
echo

# Force CPU backend and float64 for numerically consistent diagnostics.
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

task_id="${SLURM_ARRAY_TASK_ID:-0}"

case "${task_id}" in
  0) n_w=8;  dt=0.5; proj=1 ;;
  1) n_w=8;  dt=0.25; proj=1 ;;
  2) n_w=16; dt=0.5; proj=1 ;;
  3) n_w=16; dt=0.25; proj=1 ;;
  4) n_w=32; dt=0.5; proj=1 ;;
  5) n_w=32; dt=0.25; proj=1 ;;
  6) n_w=64; dt=0.5; proj=1 ;;
  7) n_w=64; dt=0.25; proj=1 ;;
  8) n_w=64; dt=0.5; proj=0 ;;
  9) n_w=64; dt=0.25; proj=0 ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0..9" >&2
    exit 2
    ;;
esac

mkdir -p .praxia/tmp
result_path=".praxia/tmp/sprint_b_step2_constraint_impulse_results_${n_w}w_dt${dt}_proj${proj}.json"
report_path=".praxia/tmp/sprint_b_step2_constraint_impulse_report_${n_w}w_dt${dt}_proj${proj}.md"

echo "Condition: n_waters=${n_w}, dt_fs=${dt}, project_ou_momentum_rigid=${proj}"

uv run python3 sprint_b_step2_constraint_impulse.py \
  --seed 7 \
  --sim-ps 100 \
  --burn-fraction 0.3333333333 \
  --n-waters "${n_w}" \
  --dt-fs "${dt}" \
  --project-ou-momentum-rigid "${proj}" \
  --results-json "${result_path}" \
  --report-md "${report_path}"

echo
echo "Step 2 completed at: $(date)"
