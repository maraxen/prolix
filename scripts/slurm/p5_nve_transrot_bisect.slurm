#!/bin/bash
# P5-A NVE trans/rot bisect — gamma=0 discriminator between mechanism A
# (RATTLE projection sink, settle.py:settle_velocities) and mechanism B
# (OU noise-covariance mismatch, settle.py:_project_one_water_momentum_rigid).
#
# Bathos campaign: 72edf0c7 (p5-nve-transrot-bisect)
#
# Usage (from repo root after pushing to cluster):
#   sbatch scripts/slurm/p5_nve_transrot_bisect.slurm

#SBATCH --job-name=plx-p5-nve-bisect
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:14:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/engaging/%j.out
#SBATCH --error=outputs/logs/engaging/%j.err

set -euo pipefail

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f "${_PROJ}/scripts/slurm/_common_env.sh" ]]; then
  _PROJ="${HOME}/projects/prolix"
fi
export SLURM_SUBMIT_DIR="${_PROJ}"
cd "${_PROJ}"

# shellcheck source=/dev/null
source "scripts/slurm/_common_env.sh"

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"
echo "job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local} workspace_root=${WORKSPACE_ROOT}"

if [[ "${P5_BISECT_SKIP_UV_SYNC:-0}" != "1" ]]; then
  # shellcheck source=/dev/null
  source "scripts/slurm/_workspace_uv_sync_inner.sh"
  _workspace_uv_sync_run
fi

cd "${PROLIX_ROOT}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# shellcheck source=/dev/null
source "scripts/slurm/_bth_env.sh"

mkdir -p outputs/p5_nve_transrot_bisect

CAMPAIGN_ID="${CAMPAIGN_ID:-72edf0c7}"
OUT="outputs/p5_nve_transrot_bisect/result_${SLURM_JOB_ID:-local}.json"

echo "=== P5-A NVE trans/rot bisect (campaign=${CAMPAIGN_ID}) ==="
uv run bth run python scripts/experiments/p5_nve_transrot_bisect.py \
  --tag p5-nve-transrot-bisect \
  --tag "slurm_job=${SLURM_JOB_ID:-local}" \
  --campaign "${CAMPAIGN_ID}" \
  --out "${OUT}" \
  -- --gamma 0 --n-waters 895 --steps 500 --seed 42 --out "${OUT}" \
  2>&1 | tee "${LOG_ROOT}/app/p5_nve_transrot_bisect_${SLURM_JOB_ID:-local}.log"

echo "=== Done: exit $? ==="
