#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=step2_smoke
#SBATCH --output=outputs/logs/sprint_b_step2_smoke_%j.out
#SBATCH --error=outputs/logs/sprint_b_step2_smoke_%j.err

set -euo pipefail

cd ~/projects/prolix

echo "=== Sprint B Step 2: single-node smoke (8w, dt=0.5fs, short trajectory) ==="
echo "Starting at: $(date)"

# CPU only; avoid CUDA init noise; cap threads to reduce LLVM compile parallelism.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

mkdir -p .praxia/tmp
RESULT=".praxia/tmp/sprint_b_step2_smoke_results.json"
REPORT=".praxia/tmp/sprint_b_step2_smoke_report.md"

uv run python3 sprint_b_step2_constraint_impulse.py \
  --seed 7 \
  --sim-ps 0.05 \
  --burn-fraction 0.4 \
  --n-waters 8 \
  --dt-fs 0.5 \
  --project-ou-momentum-rigid 1 \
  --results-json "${RESULT}" \
  --report-md "${REPORT}"

echo "SMOKE_OK wrote ${RESULT} and ${REPORT}"
echo "Finished at: $(date)"
