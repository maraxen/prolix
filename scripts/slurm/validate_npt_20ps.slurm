#!/bin/bash
# Sprint 14/15 validation: NPT 20 ps liquid-water trajectory gate.
#
# Runs test_npt_20ps_liquid_water (64 TIP3P waters, 4000 NVT + 40000 NPT steps):
#   - Phase 1 NVT equilibration (2 ps, 300 K)
#   - Phase 2 NPT production (20 ps, 300 K, 1 bar)
#   - Asserts: T_mean in [295,305] K, no thermal drift, P_mean in [-99,101] bar,
#              no NaN/Inf, box volume varies smoothly
#
# CPU-only: JAX_ENABLE_X64=1 required for float64 SETTLE+PME.
#
#SBATCH --job-name=plx-npt-20ps
#SBATCH --partition=mit_preemptable
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/engaging/%x_%j.out
#SBATCH --error=outputs/logs/engaging/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

echo "npt_20ps job=${SLURM_JOB_ID:-local} host=$(hostname) workspace=${PROLIX_ROOT}"

[[ -d "${PROLIX_ROOT}/.venv" ]] || {
  echo "missing ${PROLIX_ROOT}/.venv; run bench_workspace_uv_sync.slurm first" >&2
  exit 2
}

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

cd "${PROLIX_ROOT}"

LOGFILE="${LOG_ROOT}/app/npt_20ps_${SLURM_JOB_ID:-local}.log"
mkdir -p "${LOG_ROOT}/app"

echo "JAX_PLATFORMS=${JAX_PLATFORMS} JAX_ENABLE_X64=${JAX_ENABLE_X64}" | tee -a "${LOGFILE}"

uv run pytest \
  tests/physics/test_npt_barostat.py::test_npt_20ps_liquid_water \
  -v --tb=short --timeout=5400 2>&1 | tee -a "${LOGFILE}"

echo "npt_20ps finished; see ${LOGFILE}"
