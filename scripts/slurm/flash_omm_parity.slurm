#!/bin/bash
#SBATCH --job-name=flash_omm_parity
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/slurm/%A.flash_omm_parity.out
#SBATCH --error=outputs/logs/slurm/%A.flash_omm_parity.err

set -euo pipefail
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"
# shellcheck source=_bth_env.sh
source scripts/slurm/_bth_env.sh
OUT="${OUT:-outputs/nonbonded_omm_parity/flash_${SLURM_JOB_ID:-local}.json}"
mkdir -p "$(dirname "${OUT}")"
uv run --extra openmm bth run python scripts/experiments/flash_omm_parity.py \
  --tag nonbonded-omm-parity \
  ${CAMPAIGN_ID:+--campaign "$CAMPAIGN_ID"} \
  --out "${OUT}" \
  -- --probe "${PROBE:-water}"
