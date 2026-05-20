#!/bin/bash
#SBATCH --job-name=prolix-fit-hp4-s71
#SBATCH --partition=mit_normal_gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@mit.edu

set -euo pipefail

# Configuration
export PROJECT_ROOT="${PWD}"
export SUBSET_DIR="${PROJECT_ROOT}/data/ani1x_subset"
export N_STEPS=500
export LEARNING_RATE=1e-3
export SEED=42
export OUTPUT_JSON="${PROJECT_ROOT}/outputs/results/fit_bonded_hp4_looped_${SEED}.json"

# Create output directories
mkdir -p "${PROJECT_ROOT}/outputs/logs/slurm"
mkdir -p "${PROJECT_ROOT}/outputs/results"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"

# Load environment
source "${PROJECT_ROOT}/_bth_env.sh" 2>/dev/null || echo "Warning: _bth_env.sh not found"

# Gate L1: Dry-run (verify imports + paths)
echo ""
echo "=== Gate L1: Dry-run (imports + paths) ==="
cd "${PROJECT_ROOT}"
uv run python3 -c "
from prolix.fitting import train_loop_looped_baseline
from pathlib import Path
import h5py

subset_dir = Path('${SUBSET_DIR}')
lane_a = list((subset_dir / 'lane_a').glob('*.h5'))
lane_b = list((subset_dir / 'lane_b').glob('*.h5'))
print(f'Lane A: {len(lane_a)} molecules')
print(f'Lane B: {len(lane_b)} molecules')
print('Imports OK')
"

# Gate L2: Smoke test (n_steps=5 on CPU)
echo ""
echo "=== Gate L2: Smoke test (5 steps, CPU, 1 molecule) ==="
uv run python3 scripts/experiments/fit_bonded_hp4.py \
    --subset-dir "${SUBSET_DIR}" \
    --mode looped \
    --n-steps 5 \
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --out-json /tmp/smoke_test.json

echo "Smoke test output:"
cat /tmp/smoke_test.json

# Gate L3: Production run (500 steps)
echo ""
echo "=== Gate L3: Production run (${N_STEPS} steps) ==="
uv run python3 scripts/experiments/fit_bonded_hp4.py \
    --subset-dir "${SUBSET_DIR}" \
    --mode looped \
    --n-steps "${N_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --out-json "${OUTPUT_JSON}"

echo "Results: ${OUTPUT_JSON}"
echo "End time: $(date)"

exit 0
