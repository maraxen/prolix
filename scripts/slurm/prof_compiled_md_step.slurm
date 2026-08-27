#!/bin/bash
#SBATCH --job-name=prolix_compiled_step
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%j.prof_compiled_md_step.out
#SBATCH --error=outputs/logs/slurm/%j.prof_compiled_md_step.err

set -euo pipefail
_PROJ="${PWD}"
cd "${_PROJ}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
source "scripts/slurm/_bth_env.sh"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "JAX_PLATFORMS=${JAX_PLATFORMS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
OUT="outputs/profiling/full_md_path_matrix/compiled_inner_loop.json"
uv run --extra cuda python scripts/explore/prof_compiled_md_step.py --n-inner 256 --out "${OUT}"
echo "=== exit=$? ==="
