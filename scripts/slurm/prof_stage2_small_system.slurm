#!/bin/bash
#SBATCH --job-name=prolix_p7_stage2
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%A_%a.prof_stage2.out
#SBATCH --error=outputs/logs/slurm/%A_%a.prof_stage2.err

# P7 Stage-2 small-system probe (NOT DHFR).
#
# Walltime derivation (spec P7):
#   t_cpu = 0.557 s   (P6 1vii autodiff, n_trials_smoke=3)
#   n_trials_gpu = 20
#   3 * t_cpu * n_trials_gpu / n_trials_smoke = 11.14 s
#   round_up_to_5min(11.14s) = 5 min + 10 min XLA compile allowance = 00:15:00
#   (also the floor). 3x = GPU-vs-CPU / contention margin; 10 min = compile bound.
#
# Array T=0..11:
#   protein = T / 6          (1vii, 2gb1)
#   mode    = (T % 6) / 3    (dense, flash)
#   pme_idx = T % 3          (off, on@1.0, on@1.2)
#
# Partition is pi_so3 only: gres=gpu:h200:1 + mit_normal_gpu is rejected
# ("Requested node configuration is not available").
# Do NOT pin --nodelist=node4007,node4008 (excludes the H200).
# %k throttle is a courtesy cap, not placement.
#
# Submit from worktree:
#   myxcel submit engaging prolix scripts/slurm/prof_stage2_small_system.slurm \
#     --array 0-11%<k> --export=ALL,CAMPAIGN_ID=450f3583-c8dd-43f7-a59f-0642810746a5

set -euo pipefail

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

source "scripts/slurm/_bth_env.sh"

# CLUSTER.md §2 — inert on H200 (--gres=gpu:h200:1 cannot land on 4007/4008).
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

PROTEINS=(1vii 2gb1)
MODES=(dense flash)

T="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"
if [[ "${T}" -lt 0 || "${T}" -gt 11 ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$T out of range 0-11"
    exit 1
fi
PROTEIN="${PROTEINS[$(( T / 6 ))]}"
MODE="${MODES[$(( (T % 6) / 3 ))]}"
PME_IDX=$(( T % 3 ))
GRID_FLAGS=()
if [[ "${PME_IDX}" -eq 0 ]]; then
    PME=off
    GRID=na
else
    PME=on
    if [[ "${PME_IDX}" -eq 1 ]]; then
        GRID=1.0
    else
        GRID=1.2
    fi
    GRID_FLAGS=(--grid-spacing "${GRID}")
fi

PROBE_ID="stage2_${PROTEIN}_${MODE}_pme${PME}_g${GRID}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"
CAMPAIGN_FLAG=()
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG=(--campaign "${CAMPAIGN_ID}")

mkdir -p outputs/profiling/stage2 outputs/logs/slurm

echo "=== P7 Stage-2 ${PROBE_ID} ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Array: ${SLURM_ARRAY_JOB_ID:-N/A}/${T}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

uv run bth run python scripts/experiments/profile_b1_flash_vs_autodiff_forces.py \
    --tag p7-stage2 \
    --tag "array_task:${T}" \
    --tag "protein:${PROTEIN}" \
    --tag "mode:${MODE}" \
    --tag "pme:${PME}" \
    --tag "grid:${GRID}" \
    "${CAMPAIGN_FLAG[@]}" \
    --out "outputs/profiling/stage2/${PROBE_ID}_summary.json" \
    -- \
    --protein "${PROTEIN}" \
    --mode "${MODE}" \
    --pme "${PME}" \
    "${GRID_FLAGS[@]}" \
    --n-warmup 3 --n-trials 20 --n-inner 5 \
    --stage 2 --scopes on \
    --emit-record "outputs/profiling/stage2/${PROBE_ID}.json" \
    --out "outputs/profiling/stage2/${PROBE_ID}_summary.json"

EXIT=$?
echo "=== exit=${EXIT} ==="
exit ${EXIT}
