#!/bin/bash
# P2a-B2 investigation: ``dt_fs`` × ``gamma_ps`` matrix (OpenMM + Prolix) via ``tip3p_dt_gamma_matrix.py``.
# One job runs the full matrix; output is one JSON document in the app log.
#
# Environment (optional):
#   TIP3P_MATRIX_DT_LIST     default 1,2
#   TIP3P_MATRIX_GAMMA_LIST  default 0.5,1,2
#   TIP3P_MATRIX_N_WATERS    default 8
#   TIP3P_MATRIX_STEPS       default 12000
#   TIP3P_MATRIX_BURN        default 4000
#   TIP3P_MATRIX_REPLICAS    default 3
#   TIP3P_MATRIX_PROJECTION_SITE  default post_settle_vel
#   TIP3P_MATRIX_SETTLE_ITERS     default 20
#
#SBATCH --job-name=plx-tip3p-dt-g
#SBATCH --partition=mit_preemptable
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"

DT_LIST="${TIP3P_MATRIX_DT_LIST:-1,2}"
G_LIST="${TIP3P_MATRIX_GAMMA_LIST:-0.5,1,2}"
NW="${TIP3P_MATRIX_N_WATERS:-8}"
ST="${TIP3P_MATRIX_STEPS:-12000}"
BR="${TIP3P_MATRIX_BURN:-4000}"
REP="${TIP3P_MATRIX_REPLICAS:-3}"
PSITE="${TIP3P_MATRIX_PROJECTION_SITE:-post_settle_vel}"
SITERS="${TIP3P_MATRIX_SETTLE_ITERS:-20}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

[[ -d "${WORKSPACE_ROOT}/.venv" ]] || {
  echo "missing ${WORKSPACE_ROOT}/.venv; run bench_workspace_uv_sync.slurm first" >&2
  exit 2
}

mkdir -p "${LOG_ROOT}/app"
echo "dt_gamma_matrix job=${SLURM_JOB_ID:-} dt_list=${DT_LIST} gamma_list=${G_LIST} nw=${NW} steps=${ST} burn=${BR} replicas=${REP} projection_site=${PSITE}"

uv run python scripts/benchmarks/tip3p_dt_gamma_matrix.py \
  --dt-list "${DT_LIST}" \
  --gamma-list "${G_LIST}" \
  --n-waters "${NW}" \
  --steps "${ST}" \
  --burn "${BR}" \
  --replicas "${REP}" \
  --projection-site "${PSITE}" \
  --settle-velocity-iters "${SITERS}" \
  2>&1 | tee "${LOG_ROOT}/app/tip3p_dt_gamma_matrix_${SLURM_JOB_ID:-local}.log"

echo "tip3p_dt_gamma_matrix finished OK"
