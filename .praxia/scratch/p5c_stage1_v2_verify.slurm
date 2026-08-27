#!/bin/bash
# Phase 5-C Stage 1 v2 verification for backlog #934 (H-DISCONTINUITY): run the
# small 4-atom/10-step test's seeded (original fixture), translated (off
# discontinuities), and bitexact (step-0 ULP dump) modes on real cluster
# hardware. Locally all three now show max_diff=0.0 / bit_identical=true
# post-fix (see commits 92f1bee, 9e3c500) -- this job is what actually tests
# whether the divergence still manifests on Engaging.
#SBATCH --job-name=p5c_stage1_v2
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

# SLURM_SUBMIT_DIR comes through empty for myxcel-worktree-routed submissions
# (known bug this session) -- hardcode the known myxcel worktree scratch path.
_PROJ="/home/maarxaru/projects/.worktrees/prolix/wt-20260807-132628"
cd "${_PROJ}"

# pi_so3 nodes (node4007/4008) carry an H200 this job doesn't request via
# --gres; force CPU explicitly. Thread caps prevent oversubscription on
# shared Engaging nodes (confirmed necessary earlier this session).
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4 inter_op_parallelism_threads=4"

echo "p5c_stage1_v2 job=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$(pwd) start=$(date)"

OUT_DIR="${_PROJ}/outputs/p5c_stage1_v2"
mkdir -p "${OUT_DIR}"

echo ""
echo "=== G2-SEEDED (original small_water_system, atoms ON discontinuities) ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode seeded --steps 10 --seed 42 \
    --out "${OUT_DIR}/g2_seeded.json"

echo ""
echo "=== G1'-TRANSLATED (fixture rigidly moved off discontinuities) ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode translated --steps 10 --seed 42 \
    --out "${OUT_DIR}/g1_translated.json"

echo ""
echo "=== G1''-BITEXACT (step-0 ULP dump, both paths) ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode bitexact --seed 42 \
    --out "${OUT_DIR}/g1_bitexact.json"

echo ""
echo "=== G2 additional seeds (N>=5 total, seeds 42-46) ==="
for seed in 43 44 45 46; do
    echo "--- seed=${seed} ---"
    uv run python scripts/experiments/p5_settle_batch_divergence.py \
        --mode seeded --steps 10 --seed "${seed}" \
        --out "${OUT_DIR}/g2_seeded_seed${seed}.json"
done

echo ""
echo "P5-C Stage 1 v2 runs completed at $(date)"
ls -lh "${OUT_DIR}/"
