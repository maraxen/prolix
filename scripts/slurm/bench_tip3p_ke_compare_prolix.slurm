#!/bin/bash
# TIP3P KE diagnostic — **Prolix only**: JAX SETTLE+Langevin; atomic vs rigid-body KE ratio.
# Uses **GPU** (CUDA) to avoid XLA CPU JIT OOM on shared nodes.
#
#SBATCH --job-name=plx-tip3p-ke-plx
#SBATCH --partition=mit_preemptable
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
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

echo "engine=prolix workspace_root=${WORKSPACE_ROOT} job=${SLURM_JOB_ID:-} host=$(hostname) nwaters=${NW} steps=${ST} burn=${BR}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

[[ -d "${WORKSPACE_ROOT}/.venv" ]] || {
  echo "missing ${WORKSPACE_ROOT}/.venv; run bench_workspace_uv_sync.slurm first" >&2
  exit 2
}

mkdir -p "${LOG_ROOT}/app"
uv run python scripts/benchmarks/tip3p_ke_compare.py \
  --engine prolix \
  --n-waters "${NW}" \
  --steps "${ST}" \
  --burn "${BR}" \
  --sample-every "${SE}" \
  2>&1 | tee "${LOG_ROOT}/app/tip3p_ke_compare_prolix_${SLURM_JOB_ID:-local}.log"

echo "tip3p_ke_compare_prolix finished OK"
