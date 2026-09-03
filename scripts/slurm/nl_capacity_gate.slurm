#!/bin/bash
#SBATCH --job-name=prolix_nl_gate
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/%j.nl_gate.out
#SBATCH --error=outputs/logs/slurm/%j.nl_gate.err

# Neighbor-list capacity gate (task 260903_dhfr_nl_capacity).
#
# Gates the single-bundle NL capacity fix before the DHFR benchmark is allowed to
# quote a number. Chain the benchmark behind it with:
#   sbatch --dependency=afterok:<this_jobid> scripts/slurm/bench_dhfr_parity_split.slurm
#
# Covers:
#   * the new capacity-floor + realloc-recovery regressions, and the pre-existing
#     stacked NL tests, in test_b1_stacked_neighbor_list.py
#   * the two NL/flash parity suites, because changing how capacity is sized is a
#     physics-path change, not a refactor: a larger K must not move a force or an
#     energy. Both were green in job 21771771 and must stay green.

set -uo pipefail
# SLURM_SUBMIT_DIR is where sbatch was INVOKED, not the job cwd -- `sbatch -D`
# sets the latter. Validate each candidate rather than trusting either (job
# 21804954 ran an entire job outside the uv project by trusting SLURM_SUBMIT_DIR).
_is_repo_root() { [[ -f "$1/pyproject.toml" ]] && [[ -d "$1/scripts/slurm" ]]; }
if _is_repo_root "${PWD}"; then
    _PROJ="${PWD}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_repo_root "${SLURM_SUBMIT_DIR}"; then
    _PROJ="${SLURM_SUBMIT_DIR}"
else
    _PROJ="${HOME}/projects/prolix"
fi
_is_repo_root "${_PROJ}" || { echo "FATAL: no prolix repo root found (tried PWD=${PWD}, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})" >&2; exit 1; }
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

REPORT="tmp/pytest_nl_gate_${SLURM_JOB_ID:-local}.json"

echo "=== NL capacity + parity gate ==="
uv run --extra cuda python -m pytest \
    tests/api/test_b1_stacked_neighbor_list.py \
    tests/physics/test_1vii_nl_nonbonded_parity.py \
    tests/physics/test_nl_kernel_exclusion_parity.py \
    -v --json-report --json-report-file="${REPORT}"
RC=$?

echo "=== nl_gate exit=${RC} report=${REPORT} ==="
exit "${RC}"
