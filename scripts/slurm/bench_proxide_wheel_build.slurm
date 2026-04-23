#!/bin/bash
# **Optional / legacy:** build a ``proxide`` wheel locally when PyPI wheels are unavailable.
# Default Engaging path uses **PyPI proxide** (see ``workspace/pyproject.toml``); you normally do **not**
# need this job.
#
# Build output: ``${WORKSPACE_ROOT}/wheels/``. Requires Lmod + ``cmake`` (``_engaging_modules.sh``).
#
# Output: ``wheels/proxide-*.whl`` (newest used by sync). Rebuild after proxide changes:
#   ``REBUILD_PROXIDE_WHEEL=1`` in the submit chain or delete old wheels.
#
#SBATCH --job-name=plx-pxd-whl
#SBATCH --partition=mit_preemptable
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh
# shellcheck source=/dev/null
source scripts/slurm/_engaging_modules.sh

engaging_load_modules

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"
OUT_DIR="${PROXIDE_WHEEL_OUT_DIR:-${WORKSPACE_ROOT}/wheels}"

echo "workspace_root=${WORKSPACE_ROOT} out_dir=${OUT_DIR} job=${SLURM_JOB_ID:-} host=$(hostname)"

mkdir -p "${OUT_DIR}"
cd "${WORKSPACE_ROOT}"

uv python install
command -v cmake
echo "CMAKE=${CMAKE:-}"

# ``--no-build-isolation-package`` is not enough on uv 0.10.x: maturin still used an isolated
# interpreter under ``~/.cache/uv/builds-v0``. Full ``--no-build-isolation`` fixes ``cmake`` exec.
UV_NO_BUILD_ISOLATION=1 uv build --wheel --package proxide --out-dir "${OUT_DIR}" --no-build-isolation

echo -n "Built: "
ls -t "${OUT_DIR}"/proxide-*.whl | head -1
echo "proxide wheel build OK"
