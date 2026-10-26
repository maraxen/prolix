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
# Do NOT pin BTH_CATALOG_DIR here. Since the bathos run-log cut-over (Engaging
# 2026-10-17), `bth run` appends to the project-local log (.bth/log/) of the
# project named by .bth.toml's [project] id. A per-project catalog dir never gets
# a cutover marker, so runs written there stay legacy-mode and are invisible to
# the run log (bathos 0.13.0a5 "foreign catalog" split brain). The old pin's
# ~/projects/prolix/.bth/catalog holds 21 pre-cut-over runs, all already in the
# run log. Pull cluster runs home with `bth sync engaging --pull`.
