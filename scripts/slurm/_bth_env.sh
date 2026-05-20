# Source from SLURM scripts wanting bath provenance tracking:
#   source scripts/slurm/_bth_env.sh
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
# BTH_CATALOG_DIR defaults to ~/.bth/catalog/ — no override unless shared cluster catalog
# SLURM_JOB_ID auto-captured by `bth run`
