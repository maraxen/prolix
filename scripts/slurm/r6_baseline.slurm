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
# bath-tracked: provenance via `bth run`; sidecar at
#   scripts/benchmarks/r6_baseline_wallclock.bth.toml
#
# Outcome thresholds (from sidecar):
#   pass:      speedup_ratio >= 10  → proceed to §7.1 with current ensemble
#   marginal:  5 <= ratio < 10      → scale ensemble to N=64 before §7.1
#   fail:      ratio < 5            → re-think Claim 1 dispatch strategy (BLOCKING)
#
# Prereqs:
#   1. Local L1 + L2 of fetch_ani1x_subset.py
#   2. data/ani1x_subset/ rsync'd to ~/projects/prolix/data/ani1x_subset/
#   3. outputs/logs/slurm/ exists on remote
#   4. mit_normal_gpu reserve available (or change --partition to pi_so3)
#
# Submission:
#   mkdir -p outputs/logs/slurm
#   sbatch scripts/slurm/r6_baseline.slurm
#
# Expected wallclock: ~2-5 minutes on a single GPU.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

source scripts/slurm/_bth_env.sh

echo "=== R6 baseline ==="
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "bath project: ${BTH_PROJECT_SLUG}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

mkdir -p outputs/results

DATA_DIR="data/ani1x_subset/lane_a"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: ${DATA_DIR} not found."
    echo "Run scripts/data/fetch_ani1x_subset.py locally, then rsync data/ani1x_subset/ here."
    exit 1
fi

OUT_JSON="outputs/results/r6_baseline_${SLURM_JOB_ID:-local}.json"

uv run bth run python scripts/benchmarks/r6_baseline_wallclock.py \
    --tag cluster --tag hp4 --tag r6 --tag benchmark \
    --out "${OUT_JSON}" \
    -- \
    --data-dir "${DATA_DIR}" \
    --n-molecules 16 \
    --n-trials 5 \
    --n-warmup 2 \
    --out-json "${OUT_JSON}"

EXIT=$?
echo "=== R6 baseline exit=${EXIT} ==="
if [[ ${EXIT} -eq 4 ]]; then
    echo "FALSIFICATION TRIGGER FIRED: speedup < 10x"
    echo "Check bath outcome via: bth show <run-id> and bth campaign review <id>"
    echo "Marginal (5-10x): scale to N=64. Fail (<5x): re-think Claim 1."
fi
exit ${EXIT}
