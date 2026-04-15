#!/usr/bin/env bash
# Submit accuracy then speed with SLURM dependency (saves GPU time vs combined job).
# Usage (from repo root on login node):
#   bash scripts/slurm/submit_chignolin_split_pi_so3.sh
# Optional: export ENGAGING_LOG_DATE=20260410 PARTITION=pi_so3
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
PART="${ENGAGING_PARTITION:-${PARTITION:-pi_so3}}"
LR="outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm"
mkdir -p "${LR}"

JID="$(sbatch --partition="${PART}" -o "${LR}/accuracy_%x_%j.out" -e "${LR}/accuracy_%x_%j.err" \
  scripts/slurm/bench_chignolin_accuracy_pi_so3.slurm | awk '{print $4}')"
echo "Submitted accuracy job ${JID}"
sbatch --partition="${PART}" --dependency=afterok:"${JID}" \
  -o "${LR}/speed_%x_%j.out" -e "${LR}/speed_%x_%j.err" \
  scripts/slurm/bench_chignolin_speed_pi_so3.slurm
