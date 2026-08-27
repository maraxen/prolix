# Source from SLURM scripts: ``source "scripts/slurm/_common_env.sh``"
# Expects: repository root as current working directory (``sbatch`` from repo root).
set -euo pipefail

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
#
# Wrapper-submitted jobs (e.g. myxcel submit_job) already `cd` into the correct
# worktree scratch dir inside the generated sbatch script body before this file is
# sourced — but SLURM_SUBMIT_DIR reflects wherever the submitting process's OWN cwd
# was when it invoked `sbatch`, which can differ from that in-script `cd` target.
# Trust an already-correct inherited pwd first; only fall back to SLURM_SUBMIT_DIR /
# BASH_SOURCE when pwd doesn't look like a real repo root.
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -f "$1/scripts/write_engaging_manifest.py" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${_PROJ}"
export PROLIX_ROOT="${PROLIX_ROOT:-$(pwd)}"
export ENGAGING_LOG_DATE="${ENGAGING_LOG_DATE:-$(date +%Y%m%d)}"
export LOG_ROOT="${PROLIX_ROOT}/outputs/logs/engaging/${ENGAGING_LOG_DATE}"
mkdir -p "${LOG_ROOT}/slurm" "${LOG_ROOT}/app"
uv run python scripts/write_engaging_manifest.py
