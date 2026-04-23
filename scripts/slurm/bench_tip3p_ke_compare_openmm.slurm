#!/bin/bash
# TIP3P KE diagnostic — **OpenMM only** (``Reference``): ``getKineticEnergy`` vs Σ½m|v|².
# CPU-only; no JAX. Expects workspace venv at ``~/projects``.
#
#SBATCH --job-name=plx-tip3p-ke-omm
#SBATCH --partition=mit_preemptable
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=node4009

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"

NW="${TIP3P_KE_N_WATERS:-4}"
ST="${TIP3P_KE_STEPS:-3000}"
BR="${TIP3P_KE_BURN:-1200}"
SE="${TIP3P_KE_SAMPLE:-20}"

echo "engine=openmm workspace_root=${WORKSPACE_ROOT} job=${SLURM_JOB_ID:-} host=$(hostname) nwaters=${NW} steps=${ST} burn=${BR}"

[[ -d "${WORKSPACE_ROOT}/.venv" ]] || {
  echo "missing ${WORKSPACE_ROOT}/.venv; run bench_workspace_uv_sync.slurm first" >&2
  exit 2
}

mkdir -p "${LOG_ROOT}/app"
uv run python scripts/benchmarks/tip3p_ke_compare.py \
  --engine openmm \
  --n-waters "${NW}" \
  --steps "${ST}" \
  --burn "${BR}" \
  --sample-every "${SE}" \
  2>&1 | tee "${LOG_ROOT}/app/tip3p_ke_compare_openmm_${SLURM_JOB_ID:-local}.log"

echo "tip3p_ke_compare_openmm finished OK"
