#!/bin/bash
# One-shot: create/update the workspace venv at ``~/projects`` (``uv sync --package prolix``).
# **proxide** is installed from PyPI (manylinux wheels); no Rust/cmake on the node.
#
#SBATCH --job-name=plx-uvsync
#SBATCH --partition=mit_preemptable
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# SLURM copies scripts to spool dir — BASH_SOURCE[0] points there, not the repo.
# Use SLURM_SUBMIT_DIR (set by sbatch) when running as a job; BASH_SOURCE fallback for
# direct execution. Must submit from project root: cd ~/projects/<proj> && sbatch ...
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

# shellcheck source=/dev/null
source "scripts/slurm/_common_env.sh"

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"

echo "workspace_root=${WORKSPACE_ROOT} job=${SLURM_JOB_ID:-} host=$(hostname)"

# shellcheck source=/dev/null
source "scripts/slurm/_workspace_uv_sync_inner.sh"
_workspace_uv_sync_run

echo "uv sync finished OK"
