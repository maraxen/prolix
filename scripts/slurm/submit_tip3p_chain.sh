#!/usr/bin/env bash
# Chain ``bench_workspace_uv_sync.slurm`` → TIP3P array (``TIP3P_SKIP_UV_SYNC=1``).
#
# Usage (from prolix repo root on Engaging):
#   export ENGAGING_LOG_DATE=$(date +%Y%m%d)
#   export TIP3P_SBATCH_OPTS='--array=0-7%4'   # optional
#   export TIP3P_DT_FS=1.0                      # optional (default 2.0 in slurm)
#   export TIP3P_TOTAL_STEPS=20000 TIP3P_BURN_IN=5000
#   bash scripts/slurm/submit_tip3p_chain.sh pi_so3
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

PART="${1:?usage: submit_tip3p_chain.sh <partition>}"
shift || true

export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
mkdir -p "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm"

if [[ -z "${TIP3P_SBATCH_OPTS:-}" ]]; then
  EXTRA=( --array=0-7%4 )
else
  # shellcheck disable=SC2206
  EXTRA=( ${TIP3P_SBATCH_OPTS} )
fi

SYNC="$(sbatch --parsable --partition="${PART}" \
  -o "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/uvsync_%j.out" \
  -e "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/uvsync_%j.err" \
  scripts/slurm/bench_workspace_uv_sync.slurm | cut -d';' -f1)"

echo "uv_sync_job=${SYNC}"
[[ -n "${SYNC}" ]] || exit 1

sbatch --dependency="afterok:${SYNC}" --partition="${PART}" --export=ALL,TIP3P_SKIP_UV_SYNC=1 \
  "${EXTRA[@]}" \
  -o "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/%x_%A_%a.out" \
  -e "outputs/logs/engaging/${ENGAGING_LOG_DATE}/slurm/%x_%A_%a.err" \
  scripts/slurm/bench_tip3p_langevin_preemptable.slurm
