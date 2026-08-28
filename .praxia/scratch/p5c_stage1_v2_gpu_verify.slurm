#!/bin/bash
# Phase 5-C Stage 1 v2 GPU verification for backlog #934 (H-DISCONTINUITY).
# The prior run (job 20318383) forced JAX_PLATFORMS=cpu and came back
# completely clean (0.0 divergence, bit-identical, all modes/seeds) -- but
# that never exercised GPU execution, which is the untested path most
# plausibly responsible for the documented cluster-only divergence (the bug
# is defined as "passes on local CPU, diverges on Engaging"). This run
# explicitly requests the H200 GPU on pi_so3 and does NOT force CPU.
#SBATCH --job-name=p5c_stage1_v2_gpu
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

# SLURM_SUBMIT_DIR comes through empty for myxcel-worktree-routed submissions
# (known bug this session) -- hardcode the known myxcel worktree scratch path.
_PROJ="/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628"
cd "${_PROJ}"

# Mandatory JAX-on-Blackwell (SM120) workaround for pi_so3 GPU nodes
# (node4007/node4008) -- CLUSTER.md: XLA autotuning hangs / is catastrophically
# slow (1170x, SLURM job 15294138) without this flag. Must appear before any
# uv run / python invocation. Do NOT set JAX_PLATFORMS -- let JAX use the GPU.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export JAX_ENABLE_X64=1

echo "p5c_stage1_v2_gpu job=${SLURM_JOB_ID:-local} host=$(hostname) node=${_NODE} cwd=$(pwd) start=$(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || echo "nvidia-smi not available"

OUT_DIR="${_PROJ}/outputs/p5c_stage1_v2_gpu"
mkdir -p "${OUT_DIR}"

echo ""
echo "=== G2-SEEDED (original small_water_system, atoms ON discontinuities) -- GPU ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode seeded --steps 10 --seed 42 \
    --out "${OUT_DIR}/g2_seeded.json"

echo ""
echo "=== G1'-TRANSLATED (fixture rigidly moved off discontinuities) -- GPU ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode translated --steps 10 --seed 42 \
    --out "${OUT_DIR}/g1_translated.json"

echo ""
echo "=== G1''-BITEXACT (step-0 ULP dump, both paths) -- GPU ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode bitexact --seed 42 \
    --out "${OUT_DIR}/g1_bitexact.json"

echo ""
echo "=== G2 additional seeds (N>=5 total, seeds 42-46) -- GPU ==="
for seed in 43 44 45 46; do
    echo "--- seed=${seed} ---"
    uv run python scripts/experiments/p5_settle_batch_divergence.py \
        --mode seeded --steps 10 --seed "${seed}" \
        --out "${OUT_DIR}/g2_seeded_seed${seed}.json"
done

echo ""
echo "P5-C Stage 1 v2 GPU runs completed at $(date)"
ls -lh "${OUT_DIR}/"
