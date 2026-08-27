#!/bin/bash
# Phase 5-C historical-vs-HEAD A/B for backlog #934 (P5-C): the "8.975 A" figure
# was traced by an Opus oracle investigation to a 2026-06-04 Certify run on
# mit_quicktest (CPU-only -- GPU was never actually implicated) against commit
# d6025aa (HEAD at Certify time). Nobody has re-run the actual failing pytest
# node against that historical commit since. This job runs the SAME test node
# (test_settle_batched_vs_unbatched, body confirmed byte-identical between
# d6025aa and current HEAD) against both trees in one job, same node, so any
# difference is attributable to code/environment drift between the two commits
# and not to machine variance.
#SBATCH --job-name=p5c_hist_ab
#SBATCH --partition=mit_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -uo pipefail  # NOT -e: both arms must run even if one fails/asserts

_ROOT="/home/maarxaru/projects/.worktrees/prolix"
HIST_DIR="${_ROOT}/p5c_hist_d6025aa"
HEAD_DIR="${_ROOT}/wt-20260807-132628"

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export JAX_PLATFORMS=cpu

echo "p5c_hist_ab job=${SLURM_JOB_ID:-local} host=$(hostname) start=$(date)"
echo "HIST_DIR=${HIST_DIR}"
echo "HEAD_DIR=${HEAD_DIR}"

echo ""
echo "=== HISTORICAL ARM (commit d6025aa, HEAD at 2026-06-04 Certify time) ==="
cd "${HIST_DIR}"
git rev-parse HEAD 2>&1 || echo "no .git (rsync-only tree, expected)"
uv run python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" 2>&1
uv run pytest tests/physics/test_settle_batched.py::test_settle_batched_vs_unbatched \
    -s -p no:randomly --no-header 2>&1
HIST_EXIT=$?
echo "HIST_EXIT=${HIST_EXIT}"

echo ""
echo "=== HEAD ARM (current worktree HEAD) ==="
cd "${HEAD_DIR}"
git rev-parse HEAD 2>&1 || echo "no .git"
uv run python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" 2>&1
uv run pytest tests/physics/test_settle_batched.py::test_settle_batched_vs_unbatched \
    -s -p no:randomly --no-header 2>&1
HEAD_EXIT=$?
echo "HEAD_EXIT=${HEAD_EXIT}"

echo ""
echo "=== HEAD ARM SIDE CHECK: test_batching_settle_vs_unbatched (slow-marked, never run on cluster) ==="
uv run pytest tests/physics/test_batching_settle_smoke.py::test_batching_settle_vs_unbatched \
    -s -p no:randomly --no-header -m slow 2>&1
SMOKE_EXIT=$?
echo "SMOKE_EXIT=${SMOKE_EXIT}"

echo ""
echo "hostname=$(hostname)"
lscpu | grep 'Model name' || true
echo "SUMMARY: HIST_EXIT=${HIST_EXIT} HEAD_EXIT=${HEAD_EXIT} SMOKE_EXIT=${SMOKE_EXIT}"
echo "P5-C historical A/B completed at $(date)"
