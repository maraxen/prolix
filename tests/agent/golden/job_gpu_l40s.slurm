#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=prolix_golden_l40s
#SBATCH --output=scripts/slurm/agent_jobs/golden_l40s/%j.golden_l40s.out
#SBATCH --error=scripts/slurm/agent_jobs/golden_l40s/%j.golden_l40s.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p scripts/slurm/agent_jobs/golden_l40s
source scripts/slurm/_uv_env.sh && prolix_uv_sync
rc=0
uv run python scripts/benchmarks/prolix_vs_openmm_speed.py --sweep '1000; rm -rf /' || rc=$?
echo "=== golden_l40s exit=${rc} ==="
exit "${rc}"
