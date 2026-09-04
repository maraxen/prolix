#!/bin/bash
#SBATCH --job-name=settle_ab
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%j.settle_ab.out
#SBATCH --error=outputs/logs/slurm/%j.settle_ab.err

# Is test_post_settle_vel_rigid_mean_t_near_post_o's failure caused by the #4971
# fix, or was it already failing?
#
# Job 21969534 gave 1 failed / 48 passed, the failure being
#   post_o temperature too high: 484.43022718467955 K (threshold 350)
# on a 2-water, 600-step system. That file was in no recent gate, so there is no
# evidence either way about its prior state, and assuming it is a regression is as
# wrong as assuming it is not.
#
# So run the SAME test twice in one job, with and without the fix, changing nothing
# else -- same node, same seed (701), same everything. The fix is a purely additive
# block (33 lines, 0 deletions) delimited by its own marker comment, so stripping it
# reproduces the pre-fix file exactly.
#
# Reading:
#   both fail        -> pre-existing failure; the fix is exonerated and the test is
#                       a separate (real) problem
#   only WITH fails  -> genuine regression from the fix; investigate before shipping
#   only WITHOUT     -> the fix repaired it
#
# The original file is restored by a trap, so an interrupted job cannot leave the
# tree modified.

set -uo pipefail
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
_is_repo_root "${_PROJ}" || { echo "FATAL: no prolix repo root (PWD=${PWD})" >&2; exit 1; }
cd "${_PROJ}" || exit 1
echo "repo root: $(pwd)"
mkdir -p outputs/logs/slurm tmp

SETTLE="src/prolix/physics/settle.py"
BACKUP="tmp/settle_with_fix_${SLURM_JOB_ID:-local}.py"
cp "${SETTLE}" "${BACKUP}" || exit 1
# Guarantee restoration on any exit path, including scancel.
trap 'cp -f "${BACKUP}" "${SETTLE}"; echo "restored ${SETTLE} from ${BACKUP}"' EXIT INT TERM

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

TEST="tests/physics/test_settle_langevin_projection_site.py::test_post_settle_vel_rigid_mean_t_near_post_o"

echo "############ A: WITH the #4971 fix ############"
uv run --extra cuda python -m pytest "${TEST}" -v -p no:cacheprovider 2>&1 | tail -25
echo "############ A exit=$? ############"

echo "############ stripping the fix ############"
# Delete the additive block: from its marker comment through the second re-imaging
# call. Verified by a line-count delta of exactly 33.
sed -i '/# Make each molecule WHOLE before the rigid-body fit below (#4971)./,/pos_h2_new = to_oxygen_image(pos_h2_new, pos_oxygen_new)/d' "${SETTLE}"
echo "line delta: $(( $(wc -l < "${BACKUP}") - $(wc -l < "${SETTLE}") )) (expect 33)"
grep -c "to_oxygen_image" "${SETTLE}" || echo "to_oxygen_image fully removed (expected)"

echo "############ B: WITHOUT the fix (pre-#4971) ############"
uv run --extra cuda python -m pytest "${TEST}" -v -p no:cacheprovider 2>&1 | tail -25
echo "############ B exit=$? ############"

echo "=== done; trap restores ${SETTLE} ==="
