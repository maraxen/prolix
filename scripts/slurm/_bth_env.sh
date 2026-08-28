# Source from SLURM scripts wanting bath provenance tracking:
#   source "${SCRIPT_DIR}/_bth_env.sh"
# Expects: repository root as CWD (sbatch from repo root).
#
# Wraps the project's existing _common_env.sh (which sets up
# PROLIX_ROOT, LOG_ROOT, etc.) and adds the bath provenance env.
# bath auto-detects project_slug from .bth.toml; no env override needed
# unless running outside the project root.

# shellcheck source=_common_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common_env.sh"

# bath conventions
export BTH_PROJECT_SLUG="${BTH_PROJECT_SLUG:-prolix}"
# SLURM_JOB_ID auto-captured by `bth run`.
#
# `bth sync <remote>` rsyncs cool-tier fragments to
#   {host}:{remote_root}/.bth/catalog/runs/{slug}/
# (see bathos.sync.sync_catalog). Default `bth run` writes ~/.bth/catalog, which
# that rsync never sees (rsync 11 / empty pull). On SLURM, pin the catalog to
# this project's remote_root from .bth.toml (`~/projects/prolix`).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export BTH_CATALOG_DIR="${BTH_CATALOG_DIR:-${HOME}/projects/prolix/.bth/catalog}"
    mkdir -p "${BTH_CATALOG_DIR}/runs/${BTH_PROJECT_SLUG}"
fi
