#!/bin/bash

#SBATCH --job-name=p5c_stage1_ab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Thread limits to prevent oversubscription on Engaging
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export JAX_FLAGS="--xla_force_host_platform_device_count=4"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

# Project root (worktree-safe anchoring)
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${_PROJ}"

# Output directories
OUT_DIR="${_PROJ}/outputs/p5c_stage1_ab"
mkdir -p "${OUT_DIR}"

# Fixed parameters
BOX_SIZE=40.0
SEED_BASE=42

echo "Starting P5-C Stage 1 A/B verification"
echo "  Project: ${_PROJ}"
echo "  Output: ${OUT_DIR}"
echo "  Time: $(date)"

# Arm A / Arm B / Constructed case / Seeded cases
# Run in order: A_constructed, B_constructed, A_seed0, B_seed0, A_seed1, B_seed1, ...

# --- Constructed adversarial case (G1) ---
echo ""
echo "=== Constructed case: Arm A (round) ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode constructed \
    --pbc-round-mode round \
    --box-size ${BOX_SIZE} \
    --steps 100 \
    --seed ${SEED_BASE} \
    --out "${OUT_DIR}/g1_arm_a_constructed.json" \
    2>&1 | head -100

echo ""
echo "=== Constructed case: Arm B (floor) ==="
uv run python scripts/experiments/p5_settle_batch_divergence.py \
    --mode constructed \
    --pbc-round-mode floor \
    --box-size ${BOX_SIZE} \
    --steps 100 \
    --seed ${SEED_BASE} \
    --out "${OUT_DIR}/g1_arm_b_constructed.json" \
    2>&1 | head -100

# --- Seeded cases (G2): 5+ random seeds ---
for seed_offset in 0 1 2 3 4; do
    seed=$((SEED_BASE + seed_offset))

    echo ""
    echo "=== Seeded case seed=${seed}: Arm A (round) ==="
    uv run python scripts/experiments/p5_settle_batch_divergence.py \
        --mode seeds \
        --pbc-round-mode round \
        --n-waters 64 \
        --box-size ${BOX_SIZE} \
        --steps 100 \
        --seed ${seed} \
        --out "${OUT_DIR}/g2_arm_a_seed${seed}.json" \
        2>&1 | head -100

    echo ""
    echo "=== Seeded case seed=${seed}: Arm B (floor) ==="
    uv run python scripts/experiments/p5_settle_batch_divergence.py \
        --mode seeds \
        --pbc-round-mode floor \
        --n-waters 64 \
        --box-size ${BOX_SIZE} \
        --steps 100 \
        --seed ${seed} \
        --out "${OUT_DIR}/g2_arm_b_seed${seed}.json" \
        2>&1 | head -100
done

echo ""
echo "P5-C Stage 1 A/B runs completed at $(date)"
echo "Results written to ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
