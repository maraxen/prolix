#!/bin/bash
# Sprint 3 validation: CSVR thermostat temperature and equipartition tests.
#
# Runs three compute-intensive pytest cases that gate the dt=2.0 fs claim:
#   - test_temperature_csvr_dt1fs_near_target  (100 ps, 2 waters, ±15K)
#   - test_temperature_csvr_dt2fs_near_target  (100 ps, 2 waters, ±5K)  ← critical
#   - test_equipartition_csvr_dt2fs_chi2       (500 ps, 8 waters, KS p>0.05)
#
# CPU-only: tests require float64 (JAX_ENABLE_X64=1). CUDA float64 is gated
# per-GPU; CPU is reliable and avoids downgrade warnings. These are validation
# simulations, not throughput benchmarks, so CPU suffices.
#
#SBATCH --job-name=plx-csvr-validate
#SBATCH --partition=mit_preemptable
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"

echo "csvr_validate job=${SLURM_JOB_ID:-local} host=$(hostname) workspace=${WORKSPACE_ROOT}"

[[ -d "${WORKSPACE_ROOT}/.venv" ]] || {
  echo "missing ${WORKSPACE_ROOT}/.venv; run bench_workspace_uv_sync.slurm first" >&2
  exit 2
}

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

cd "${PROLIX_ROOT}"

LOGFILE="${LOG_ROOT}/app/csvr_validate_${SLURM_JOB_ID:-local}.log"
mkdir -p "${LOG_ROOT}/app"

echo "JAX_PLATFORMS=${JAX_PLATFORMS} JAX_ENABLE_X64=${JAX_ENABLE_X64}" | tee -a "${LOGFILE}"

uv run pytest \
  tests/physics/test_settle_temperature_control.py::test_temperature_csvr_dt1fs_near_target \
  tests/physics/test_settle_temperature_control.py::test_temperature_csvr_dt2fs_near_target \
  tests/physics/test_settle_temperature_control.py::test_equipartition_csvr_dt2fs_chi2 \
  -v --tb=short 2>&1 | tee -a "${LOGFILE}"

echo "csvr_validate finished; see ${LOGFILE}"
