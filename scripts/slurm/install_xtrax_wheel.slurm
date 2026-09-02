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

# Install a staged xtrax wheel into the shared workspace venv.
#
# WHY A BATCH JOB
# ---------------
# Compute nodes have no outbound internet, so the wheel is staged as a FILE
# (built by xtrax CI, downloaded locally, pushed here) rather than resolved from
# PyPI. The login node is not an option either: installs there are blocked.
#
# WHY --no-deps
# -------------
# a6 -> a7 changes exactly one core pin, and it WIDENS it:
#   jax/jaxlib  >=0.10.2,<0.11   ->   >=0.10.2,<0.12
# Everything a7 requires is therefore already satisfied by the a6 install. A
# dependency-resolving install could decide to touch jax itself, and this venv
# carries the CUDA jax build that took a dedicated pass to get right (see
# reference_engaging_uv_workspace_cpu_fallback) -- so we pin the blast radius to
# the xtrax package directory alone. The new `export` extra (iree, hf-hub) is
# optional and deliberately not installed.
#
# WHY THIS IS SAFE WHILE OTHER JOBS RUN
# -------------------------------------
# Concurrent jobs in this venv (naurmalade_group_vft_apo x21) do not import
# xtrax, and --no-deps touches nothing outside site-packages/xtrax*.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/prolix}"
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
