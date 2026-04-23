#!/bin/bash
# P2a-B2 tightening: checkpointed TIP3P Langevin (OpenMM / Prolix).
#
# **Partitions**
#   * **mit_preemptable** — default in the ``#SBATCH`` lines below; cheap throughput; jobs may be
#     preempted. Re-submit with the same ``TIP3P_RUN_ROOT`` / array task; ``--resume`` continues MD.
#   * **pi_so3** — pass ``sbatch --partition=pi_so3`` (see ``just submit-tip3p-tightening-pi-so3``)
#     when you need the lab partition and **want to avoid preemption** (site policy permitting).
#     Command-line ``sbatch`` options override these directives.
#
# Replicas = SLURM array tasks (independent RNG seeds).
#
# **Venv:** Prefer ``bench_workspace_uv_sync.slurm`` once, then run with ``TIP3P_SKIP_UV_SYNC=1``
# (set by Just when jobs are chained). ``proxide`` is from PyPI; no cmake/Rust on compute nodes.
#
# **JAX:** Defaults to CUDA (``JAX_PLATFORMS=cuda``) so Prolix avoids XLA-CPU JIT RAM spikes on
# shared nodes. Override with ``JAX_PLATFORMS=cpu`` for debugging. OpenMM platform is separate
# (see ``OPENMM_PLATFORM`` / script defaults).
#
# **fp64:** ``JAX_ENABLE_X64=1`` (default here) matches ``tip3p_ke_compare --jax-x64 on`` / Tier-1
# policy. Set ``JAX_ENABLE_X64=0`` only for exploratory f32 runs. If CUDA still downgrades dtypes,
# try ``JAX_PLATFORMS=cpu`` with x64 enabled.
#
# **Timestep:** ``TIP3P_DT_FS`` (default ``2.0``) is passed to ``--dt-fs`` for OpenMM and Prolix.
# **Integrator knobs:** optional env passthrough:
#   ``TIP3P_GAMMA_PS`` (default ``1.0``), ``TIP3P_PROJECT_OU_MOMENTUM_RIGID`` (default ``true``),
#   ``TIP3P_PROJECTION_SITE`` (default ``post_o``), ``TIP3P_SETTLE_VELOCITY_ITERS`` (default ``10``),
#   ``TIP3P_DIAGNOSTICS_LEVEL`` (default ``off``), ``TIP3P_DIAGNOSTICS_DECIMATION`` (default ``10``).
#
#SBATCH --job-name=plx-tip3p-T
#SBATCH --partition=mit_preemptable
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%4

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# shellcheck source=/dev/null
source scripts/slurm/_common_env.sh

WORKSPACE_ROOT="${ENGAGING_WORKSPACE_ROOT:-${HOME}/projects}"
export UV_PROJECT="${WORKSPACE_ROOT}"
echo "workspace_root=${WORKSPACE_ROOT} replica=${SLURM_ARRAY_TASK_ID:-0} job=${SLURM_JOB_ID:-} skip_uv_sync=${TIP3P_SKIP_UV_SYNC:-0} dt_fs=${TIP3P_DT_FS:-2.0}"

if [[ "${TIP3P_SKIP_UV_SYNC:-0}" != "1" ]]; then
  # shellcheck source=/dev/null
  source scripts/slurm/_workspace_uv_sync_inner.sh
  _workspace_uv_sync_run
else
  [[ -d "${WORKSPACE_ROOT}/.venv" ]] || {
    echo "TIP3P_SKIP_UV_SYNC=1 but ${WORKSPACE_ROOT}/.venv missing; run bench_workspace_uv_sync.slurm first." >&2
    exit 2
  }
fi

cd "${PROLIX_ROOT}"

REPLICA="${SLURM_ARRAY_TASK_ID:-0}"
# Optional: TIP3P_RUN_TAG=x64 (or a date suffix) avoids ``--resume`` picking up an older fp32 run
# under the same ENGAGING_LOG_DATE. TIP3P_RUN_ROOT overrides entirely if set.
if [[ -n "${TIP3P_RUN_ROOT:-}" ]]; then
  RUN_ROOT="${TIP3P_RUN_ROOT}"
elif [[ -n "${TIP3P_RUN_TAG:-}" ]]; then
  RUN_ROOT="${PROLIX_ROOT}/outputs/tip3p_tightening/${ENGAGING_LOG_DATE}/${TIP3P_RUN_TAG}/replica_${REPLICA}"
else
  RUN_ROOT="${PROLIX_ROOT}/outputs/tip3p_tightening/${ENGAGING_LOG_DATE}/replica_${REPLICA}"
fi
mkdir -p "${RUN_ROOT}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# CLI ``--jax-x64`` must match env so printed JSON ``jax.*`` matches Tier-1 policy.
case "${JAX_ENABLE_X64}" in
  1|true|yes|on) TIP3P_JAX_X64_ARG=on ;;
  *) TIP3P_JAX_X64_ARG=off ;;
esac

ENGINE="${TIP3P_ENGINE:-both}"
CHK_EVERY="${TIP3P_CHECKPOINT_EVERY:-500}"
N_WAT="${TIP3P_N_WATERS:-33}"
STEPS="${TIP3P_TOTAL_STEPS:-50000}"
BURN="${TIP3P_BURN_IN:-10000}"
SAMPLE="${TIP3P_SAMPLE_EVERY:-10}"
DT_FS="${TIP3P_DT_FS:-2.0}"
GAMMA_PS="${TIP3P_GAMMA_PS:-1.0}"
PROJECT_OU="${TIP3P_PROJECT_OU_MOMENTUM_RIGID:-true}"
PROJECTION_SITE="${TIP3P_PROJECTION_SITE:-post_o}"
SETTLE_ITERS="${TIP3P_SETTLE_VELOCITY_ITERS:-10}"
DIAG_LEVEL="${TIP3P_DIAGNOSTICS_LEVEL:-off}"
DIAG_DECIM="${TIP3P_DIAGNOSTICS_DECIMATION:-10}"

uv run python scripts/benchmarks/tip3p_langevin_tightening.py \
  --engine "${ENGINE}" \
  --checkpoint-dir "${RUN_ROOT}" \
  --checkpoint-every "${CHK_EVERY}" \
  --resume \
  --replica-id "${REPLICA}" \
  --n-waters "${N_WAT}" \
  --gamma-ps "${GAMMA_PS}" \
  --dt-fs "${DT_FS}" \
  --total-steps "${STEPS}" \
  --burn-in "${BURN}" \
  --sample-every "${SAMPLE}" \
  --jax-x64 "${TIP3P_JAX_X64_ARG}" \
  --project-ou-momentum-rigid "${PROJECT_OU}" \
  --projection-site "${PROJECTION_SITE}" \
  --settle-velocity-iters "${SETTLE_ITERS}" \
  --diagnostics-level "${DIAG_LEVEL}" \
  --diagnostics-decimation "${DIAG_DECIM}" \
  2>&1 | tee "${LOG_ROOT}/app/tip3p_tightening_r${REPLICA}_${SLURM_JOB_ID:-local}.log"
