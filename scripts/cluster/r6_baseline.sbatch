#!/bin/bash
#SBATCH --job-name=prolix_r6_baseline
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.r6_baseline.out
#SBATCH --error=outputs/logs/slurm/%j.r6_baseline.err

# R6 baseline wall-clock smoke (HP4 §9 R6 / roadmap §2.7 RR7 falsification trigger).
#
# Times jit(vmap(jax.grad)) over a stacked HP4 same-bucket ensemble vs. the
# looped baseline. If the ratio is < 10x, §7.1 implementation must NOT begin
# with the current 20-molecule ensemble — scale to 64 first.
#
# Prereqs (must be done before this job is submitted):
#   1. Local L1 + L2 of fetch_ani1x_subset.py
#   2. data/ani1x_subset/ rsync'd to ~/projects/prolix/data/ani1x_subset/
#   3. outputs/logs/slurm/ exists on remote
#   4. mit_normal_gpu reserve available (or change --partition to pi_so3)
#
# Submission:
#   mkdir -p outputs/logs/slurm
#   sbatch scripts/cluster/r6_baseline.sbatch
#
# Expected wallclock: ~2-5 minutes on a single GPU.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

echo "=== R6 baseline ==="
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

mkdir -p outputs/results

DATA_DIR="data/ani1x_subset/lane_a"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: ${DATA_DIR} not found."
    echo "Run scripts/data/fetch_ani1x_subset.py locally, then rsync data/ani1x_subset/ here."
    exit 1
fi

uv run python scripts/cluster/r6_baseline_wallclock.py \
    --data-dir "${DATA_DIR}" \
    --n-molecules 16 \
    --n-trials 5 \
    --n-warmup 2 \
    --out-json "outputs/results/r6_baseline_${SLURM_JOB_ID:-local}.json"

EXIT=$?
echo "=== R6 baseline exit=${EXIT} ==="
if [[ ${EXIT} -eq 4 ]]; then
    echo "FALSIFICATION TRIGGER FIRED: speedup < 10x"
    echo "Re-run with --n-molecules 64 OR scale Lane A subset before §7.1."
fi
exit ${EXIT}
