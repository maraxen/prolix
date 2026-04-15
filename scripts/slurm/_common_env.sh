# Source from SLURM scripts: ``source scripts/slurm/_common_env.sh``
# Expects: repository root as current working directory (``sbatch`` from repo root).
set -euo pipefail
export PROLIX_ROOT="${PROLIX_ROOT:-$(pwd)}"
export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
export LOG_ROOT="${PROLIX_ROOT}/outputs/logs/engaging/${ENGAGING_LOG_DATE}"
mkdir -p "${LOG_ROOT}/slurm" "${LOG_ROOT}/app"
uv run python scripts/write_engaging_manifest.py
