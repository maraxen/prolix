#!/bin/bash
# GPU disambiguation run for the settle_langevin half_dt propagator fix (cfd2b85):
# rerun the fast test suite on GPU to determine whether the reproducible CPU-only
# segfault in test_settle_preserves_water_geometry (jobs 19848713, 19849325) is a
# pre-existing CPU-JAX-backend issue or a real regression from the merged fix.
#
#SBATCH --job-name=p5-settle-fix-fastsuite-gpu
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# Submitted via `myxcel submit --sbatch-arg="chdir=<worktree path>"`, which sets the
# job's initial cwd correctly (unlike SLURM_SUBMIT_DIR, which reflects myxcel's own
# ssh-session cwd on the remote host, not the worktree root — a mismatch that made the
# usual SLURM_SUBMIT_DIR/BASH_SOURCE fallback resolve to the wrong directory here).
# Trust pwd instead of re-deriving it. _common_env.sh does its own SLURM_SUBMIT_DIR
# fallback internally — unset it here so that logic falls through to a BASH_SOURCE-based
# resolution that's correct relative to our already-correct pwd, instead of re-cd'ing to
# the same wrong (but non-empty) directory this script just worked around.
_PROJ="$(pwd)"
unset SLURM_SUBMIT_DIR

# shellcheck source=/dev/null
source "scripts/slurm/_common_env.sh"

# Blackwell (SM120) workaround — node4007/4008 currently run rtx_pro_6000 (Blackwell);
# safe no-op elsewhere (e.g. node4009's H200/Hopper).
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

export JAX_PLATFORMS="cuda"
export JAX_ENABLE_X64="1"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# This worktree's local .venv was synced without the cuda extra (pyproject.toml:
# cuda = ["jax[cuda12]"]), so jax only sees ['cpu', 'tpu'] backends. Sync it in
# before running — idempotent, and cheap if already present.
uv sync --extra cuda --extra dev

LOG="${LOG_ROOT}/app/p5_settle_fix_fastsuite_gpu_${SLURM_JOB_ID:-local}.log"
echo "log=${LOG} host=$(hostname) date=$(date -Is)" | tee "${LOG}"

uv run pytest -m "not slow" -v --tb=short -ra \
  --json-report --json-report-file=tmp/pytest.json \
  2>&1 | tee -a "${LOG}"
