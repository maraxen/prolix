#!/bin/bash
#SBATCH --job-name=settle_gate
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=outputs/logs/slurm/%j.settle_gate.out
#SBATCH --error=outputs/logs/slurm/%j.settle_gate.err

# Regression gate for the #4971 SETTLE intramolecular-imaging fix.
#
# This changes CORE PHYSICS -- every NVT/NPT production path goes through
# _settle_water_batch -- so the question that matters is not whether the new test
# passes but whether anything that passed before still does. Phase 5's dt=1.0 fs
# cap-lift, the C3 angular-momentum correction and the projection-site work all
# rest on this code.
#
# Included:
#   * the SETTLE suites (constraints, batching, violation detection, projection
#     site, RATTLE combination, noise) -- the direct blast radius
#   * both 1VII NL/flash parity suites -- SETTLE must not have moved a force or an
#     energy; these were green in jobs 21771771 / 21895490 and must stay green
#   * the JAX-free bench guards, which cost nothing to include
#
# Deliberately NOT included: the 10 ps 64-water trajectories. They are the right
# follow-up for equipartition/temperature confirmation but take far longer than a
# gate should, and the fix is a no-op for any molecule that is not split across a
# face -- which, in an already-wrapped fixture, is all of them.

set -uo pipefail
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
_is_repo_root "${_PROJ}" || { echo "FATAL: no prolix repo root (PWD=${PWD}, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2; exit 1; }
cd "${_PROJ}" || exit 1
echo "repo root: $(pwd)"
mkdir -p outputs/logs/slurm tmp
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    # XLA autotuning hangs on SM120 (Blackwell); 1170x difference. See CLUSTER.md.
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

REPORT="tmp/pytest_settle_gate_${SLURM_JOB_ID:-local}.json"

uv run --extra cuda python -m pytest \
    tests/physics/test_settle.py \
    tests/physics/test_settle_constraint_violation.py \
    tests/physics/test_settle_batched.py \
    tests/physics/test_settle_langevin_projection_site.py \
    tests/physics/test_settle_rattle_combined.py \
    tests/physics/test_settle_noise_direct.py \
    tests/physics/test_1vii_nl_nonbonded_parity.py \
    tests/physics/test_nl_kernel_exclusion_parity.py \
    tests/bench/test_bench_dhfr_backend_split.py \
    -v --json-report --json-report-file="${REPORT}"
RC=$?

echo "=== settle_gate exit=${RC} report=${REPORT} ==="
exit "${RC}"
