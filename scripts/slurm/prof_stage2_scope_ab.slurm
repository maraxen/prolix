#!/bin/bash
#SBATCH --job-name=prolix_p7_scope_ab
#SBATCH --partition=pi_so3,mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%j.prof_stage2_scope_ab.out
#SBATCH --error=outputs/logs/slurm/%j.prof_stage2_scope_ab.err

# P7 task 12: scopes on/off A/B (1vii, flash, PME on, 1.0 Å).
# --time is a placeholder; override at submit from the first array-task log:
#   round_up_to_5min(2 * n_pairs * (t_gpu_child + t_compile_child)) + 15 min
# If that exceeds 12h, reduce n_pairs (cap 200) and record INSUFFICIENT_POWER.

set -euo pipefail

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"
source "scripts/slurm/_bth_env.sh"

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
CAMPAIGN_FLAG=()
[[ -n "${CAMPAIGN_ID}" ]] && CAMPAIGN_FLAG=(--campaign "${CAMPAIGN_ID}")
N_PAIRS_FLAG=()
[[ -n "${N_PAIRS:-}" ]] && N_PAIRS_FLAG=(--n-pairs "${N_PAIRS}")

mkdir -p outputs/profiling/stage2/scope_ab outputs/logs/slurm

echo "=== P7 scope A/B ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

uv run bth run python scripts/experiments/prof_stage2_scope_ab.py \
    --tag p7-stage2 --tag scope-ab \
    "${CAMPAIGN_FLAG[@]}" \
    --out outputs/profiling/stage2/scope_ab/summary.json \
    -- \
    --out-dir outputs/profiling/stage2/scope_ab \
    --out outputs/profiling/stage2/scope_ab/summary.json \
    "${N_PAIRS_FLAG[@]}"

EXIT=$?
echo "=== exit=${EXIT} ==="
exit ${EXIT}
