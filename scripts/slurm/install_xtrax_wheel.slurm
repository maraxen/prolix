#!/bin/bash
#SBATCH --job-name=xtrax_wheel
#SBATCH --partition=mit_normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=outputs/logs/slurm/%j.xtrax_wheel.out
#SBATCH --error=outputs/logs/slurm/%j.xtrax_wheel.err

# Install a staged xtrax wheel into a venv, offline.
#
# READ THIS FIRST: THE INSTALL IS NOT DURABLE IN A uv-MANAGED WORKSPACE VENV
# --------------------------------------------------------------------------
# Verified the hard way on 2026-09-01. Job 21783065 installed 0.4.0a7 here and
# proved it imported. Job 21784691, 27 minutes later, loaded 0.4.0a6 from the
# very same site-packages -- because it ran `uv run`, and `uv run` first syncs
# the environment to the lockfile. The shared workspace lock at
# ${ENGAGING_WORKSPACE_ROOT}/uv.lock pins xtrax 0.4.0a6, so uv dutifully
# reinstalled a6 over the a7 that was sitting there.
#
# Nothing warns you. The install exits 0, the import check passes, and the
# revert happens silently inside an unrelated later job. So:
#
#   * `uv pip install` into a uv-managed venv survives only until the next
#     `uv run` / `uv sync` in that workspace.
#   * The durable fix is the LOCKFILE, not the venv: bump the pin and re-lock,
#     then materialize with scripts/slurm/bench_workspace_uv_sync.slurm.
#   * That lock is SHARED (biosensors and protamer resolve against it too), so
#     re-locking it is a cross-project decision, not a prolix-local one.
#
# This script therefore remains useful for exactly two things: priming a venv
# that is NOT uv-managed, and pairing with `uv run --no-sync` for a one-shot
# check. Do not treat it as "upgrading the cluster".
#
# WHY A BATCH JOB AT ALL
# ----------------------
# Compute nodes have no outbound internet, so the wheel is staged as a FILE
# (built by xtrax CI, downloaded locally, pushed here) rather than resolved from
# PyPI. The login node is not an option either: installs there are blocked.
#
# WHY --no-deps
# -------------
# a6 -> a7 changes exactly one core pin and it WIDENS it (jax/jaxlib <0.11 ->
# <0.12), so everything a7 needs is already satisfied. A resolving install could
# decide to touch jax, and this venv carries the CUDA jax build that previously
# took a dedicated pass to fix after it silently fell back to CPU. Pinning the
# blast radius to the xtrax directory keeps concurrent jobs in the venv safe.
#
# NOTE ON SLURM_SUBMIT_DIR
# ------------------------
# Every path below is absolute on purpose. SLURM_SUBMIT_DIR is the directory
# sbatch was INVOKED from, not the job working directory set by `sbatch -D`, so
# a job submitted over ssh from a home directory runs with the wrong cwd. This
# script is immune to that; scripts that resolve paths relatively are not.

set -uo pipefail
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
cd "${_PROJ}" || exit 1
mkdir -p outputs/logs/slurm
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"

VENV="${XTRAX_TARGET_VENV:-/orcd/pool/008/so3_shared/marielle/projects/.venv}"
WHEEL="${XTRAX_WHEEL:-$HOME/projects/prolix/outputs/xtrax-0.4.0a7-py3-none-any.whl}"

if [[ ! -f "${WHEEL}" ]]; then
    echo "ERROR: wheel not staged: ${WHEEL}" >&2
    exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: target venv has no interpreter: ${VENV}/bin/python" >&2
    exit 1
fi

echo "=== before ==="
ls -d "${VENV}"/lib/*/site-packages/xtrax-*.dist-info 2>/dev/null || echo "(no xtrax installed)"

echo "=== installing ${WHEEL##*/} ==="
uv pip install --python "${VENV}/bin/python" --no-deps --reinstall "${WHEEL}"
RC=$?
echo "--- install exit=${RC} ---"

echo "=== after ==="
ls -d "${VENV}"/lib/*/site-packages/xtrax-*.dist-info 2>/dev/null || echo "(no xtrax installed)"

# The whole point of a7 is that it ships xtrax.profiling, which a6 did not.
# Verify the subpackage actually imports rather than trusting the version string.
echo "=== import check: xtrax.profiling ==="
"${VENV}/bin/python" - <<'PYEOF'
import sys
import xtrax
print("xtrax", xtrax.__version__, "from", xtrax.__file__)
import xtrax.profiling as p
print("xtrax.profiling OK from", p.__file__)
from xtrax.profiling.emitters import attribution_from_scopes
from xtrax.profiling.trace import parse_hlo_op_times
print("emitters.attribution_from_scopes OK")
print("trace.parse_hlo_op_times OK")
if not xtrax.__version__.startswith("0.4.0a7"):
    print("FAIL: expected 0.4.0a7, got", xtrax.__version__)
    sys.exit(1)
print("VERIFY OK")
PYEOF
RC2=$?
echo "=== install=${RC} verify=${RC2} ==="
exit $(( RC != 0 || RC2 != 0 ))
