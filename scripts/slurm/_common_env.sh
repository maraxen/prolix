# Source from SLURM scripts: ``source "${SCRIPT_DIR}/_common_env.sh``"
# Expects: repository root as current working directory (``sbatch`` from repo root).
set -euo pipefail

# Anchor paths to this script — safe from any submission cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROLIX_ROOT="${PROLIX_ROOT:-$(pwd)}"
export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
export LOG_ROOT="${PROLIX_ROOT}/outputs/logs/engaging/${ENGAGING_LOG_DATE}"
mkdir -p "${LOG_ROOT}/slurm" "${LOG_ROOT}/app"
uv run python scripts/write_engaging_manifest.py
