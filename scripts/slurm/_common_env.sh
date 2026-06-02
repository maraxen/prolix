# Source from SLURM scripts: ``source "scripts/slurm/_common_env.sh``"
# Expects: repository root as current working directory (``sbatch`` from repo root).
set -euo pipefail

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"
export PROLIX_ROOT="${PROLIX_ROOT:-$(pwd)}"
export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
export LOG_ROOT="${PROLIX_ROOT}/outputs/logs/engaging/${ENGAGING_LOG_DATE}"
mkdir -p "${LOG_ROOT}/slurm" "${LOG_ROOT}/app"
uv run python scripts/write_engaging_manifest.py
