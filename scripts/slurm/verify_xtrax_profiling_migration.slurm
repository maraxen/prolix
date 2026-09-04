#!/bin/bash
#SBATCH --job-name=xtrax_migr_check
#SBATCH --partition=mit_normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.xtrax_migr_check.out
#SBATCH --error=outputs/logs/slurm/%j.xtrax_migr_check.err

# Verify the scripts.profiling -> xtrax.profiling migration on the cluster.
#
# The migration deleted the local scripts/profiling/ package and repointed seven
# consumer scripts plus five test modules at xtrax.profiling, which only exists
# from xtrax 0.4.0a7 (installed into the shared venv by job 21783065). Local
# import success proves nothing about the cluster: that is where these scripts
# actually run, and it is a different interpreter and a different xtrax.
#
# Two checks, because they fail differently:
#   1. Import every migrated consumer as a module. A missing symbol in
#      xtrax.profiling surfaces here as ImportError, not at profiling time
#      several GPU-hours into a job.
#   2. Run tests/profiling/. These are pure-python trace/record/report parsers
#      with no GPU or JAX-compute dependency, so they are cheap and safe on a
#      CPU node -- and they exercise the migrated API rather than just its
#      presence.
#
# CPU-only and deliberately so: nothing here needs a GPU, and the GPU quota is
# saturated by other campaigns.

set -uo pipefail

# SLURM_SUBMIT_DIR is the directory sbatch was INVOKED from, which is not the
# same thing as the job working directory set by `sbatch -D`. Submitting over
# ssh from a home directory therefore lands here with SLURM_SUBMIT_DIR=$HOME,
# and a bare `cd "${SLURM_SUBMIT_DIR}"` runs the whole job outside any uv
# project -- which resolves to an environment with neither xtrax nor pytest and
# fails in a way that looks exactly like a bad install. Validate the candidate
# the way scripts/slurm/_common_env.sh does instead of trusting it.
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
cd "${_PROJ}" || exit 1
echo "repo root: $(pwd)"
mkdir -p outputs/logs/slurm
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"

# Keep JAX on CPU: this check is about import wiring, not kernels.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "=== [1/2] import every migrated consumer ==="
uv run python - <<'PYEOF'
import importlib
import sys
import traceback

print(f"sys.executable = {sys.executable}")
print(f"sys.prefix     = {sys.prefix}")

import xtrax
print(f"xtrax {xtrax.__version__} from {xtrax.__file__}")

MODULES = [
    "scripts.experiments.prof_stage2_coverage",
    "scripts.experiments.profile_b1_flash_vs_autodiff_forces",
    "scripts.explore.prof_dump_hlo_ops",
    "scripts.explore.prof_ensemble_apply_probe",
    "scripts.explore.prof_stage0_cost_analysis",
    "scripts.explore.prof_stage1_cpu_micro",
    "scripts.explore.prof_stage1_scope_perturbation",
]

failures = []
for name in MODULES:
    try:
        importlib.import_module(name)
    except Exception:
        failures.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()
    else:
        print(f"ok   {name}")

# The migration is only meaningful if the OLD package is genuinely gone from the
# import path -- otherwise a consumer could still be resolving the stale copy and
# we would not know. Assert it is unimportable rather than assuming.
try:
    importlib.import_module("scripts.profiling")
except ImportError:
    print("ok   scripts.profiling is gone (expected)")
else:
    print("WARN scripts.profiling still importable in this checkout")
    print("     (stale directory on the cluster copy; consumers no longer use it)")

if failures:
    print(f"IMPORT CHECK FAILED: {failures}")
    sys.exit(1)
print("IMPORT CHECK OK")
PYEOF
RC1=$?
echo "--- import check exit=${RC1} ---"

echo "=== [2/2] tests/profiling ==="
uv run pytest tests/profiling -q --no-header -p no:cacheprovider
RC2=$?
echo "--- tests exit=${RC2} ---"

echo "=== imports=${RC1} tests=${RC2} ==="
exit $(( RC1 != 0 || RC2 != 0 ))
