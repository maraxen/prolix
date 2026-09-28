#!/bin/bash
#SBATCH --job-name=prolix_dhfr_split
#SBATCH --partition=pi_so3,mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/logs/slurm/%j.dhfr_split.out
#SBATCH --error=outputs/logs/slurm/%j.dhfr_split.err

# DHFR ns/day parity (#295), run as two legs in two environments plus a combine.
#
# WHY THREE STEPS INSTEAD OF ONE PROCESS
# --------------------------------------
# The benchmark used to run prolix and OpenMM in one process, and its OpenMM leg
# was silently falling back to CPU -- which flatters prolix, so every ratio it
# ever produced is a CEILING on our standing, not a floor.
#
# The two halves cannot share an environment:
#   * the CUDA-capable OpenMM is a conda-forge env (has libOpenMMCUDA.so) with
#     NO jax, NO cuda python packages and NO prolix;
#   * the workspace venv that has jax+prolix carries the PyPI `openmm` wheel,
#     which ships no CUDA plugin at all.
# So each leg runs under its own interpreter and the results are merged after.
# This mirrors the interpreter switch scripts/slurm/b1_init_exec.slurm already
# uses for the same reason.
#
# The combine step REFUSES to emit a ratio unless the OpenMM leg actually ran on
# CUDA and both legs ran the same step count -- see
# tests/bench/test_bench_dhfr_backend_split.py.

set -uo pipefail
# SLURM_SUBMIT_DIR is where sbatch was INVOKED, not the job cwd -- `sbatch -D`
# sets the latter. Submitting over ssh from $HOME therefore lands outside the uv
# project, where `uv run --extra cuda` degrades to a bare interpreter with no
# prolix on the path (job 21804954: all three legs exit=2, file-not-found).
# Validate each candidate instead of trusting it.
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
mkdir -p outputs/logs/slurm outputs/dhfr_parity
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    # XLA autotuning hangs on SM120 (Blackwell); 1170x difference. See CLUSTER.md.
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="outputs/dhfr_parity"
P_JSON="${OUT_DIR}/prolix_${TAG}.json"
O_JSON="${OUT_DIR}/openmm_${TAG}.json"
C_JSON="${OUT_DIR}/combined_${TAG}.json"

# Keep both legs on the same sweep. The combine step asserts they match, but
# passing one value keeps them from drifting in the first place.
N_STEPS_LIST="${N_STEPS_LIST:-50,200,1000,5000}"
# SMOKE is deliberately NOT supported here. The smoke path builds its OpenMM
# system from sys_data (constructed with jnp), so --backend openmm --smoke needs
# jax -- which the conda-forge OpenMM env does not have -- and the smoke
# comparator hardcodes the CPU platform anyway, so the combine step would refuse
# the result. Use the single-process path for smoke:
#   uv run python scripts/experiments/bench_dhfr_parity.py --backend both --smoke ...
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "ERROR: SMOKE=1 is not supported by the split wrapper." >&2
    echo "       The smoke OpenMM comparator needs jax (absent from the CUDA" >&2
    echo "       OpenMM env) and hardcodes the CPU platform, which the combine" >&2
    echo "       step correctly refuses. Run smoke single-process instead:" >&2
    echo "         uv run python scripts/experiments/bench_dhfr_parity.py \\" >&2
    echo "             --backend both --smoke --out <path>" >&2
    exit 1
fi

echo "=== [1/3] prolix leg (workspace venv, jax+cuda) ==="
uv run --extra cuda python scripts/experiments/bench_dhfr_parity.py \
    --backend prolix \
    --n-steps-list "${N_STEPS_LIST}" \
    --gpu-tag h200 \
    --out "${P_JSON}"
RC1=$?
echo "--- prolix leg exit=${RC1} ---"

echo "=== [2/3] openmm leg (conda-forge CUDA env) ==="
# conda-forge openmm: the PyPI wheel has no libOpenMMCUDA.so. Override with
# B1_OPENMM_ENV if the env moves.
B1_OPENMM_ENV="${B1_OPENMM_ENV:-/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm}"
if [[ ! -x "${B1_OPENMM_ENV}/bin/python" ]]; then
    echo "ERROR: OpenMM CUDA env missing: ${B1_OPENMM_ENV}/bin/python" >&2
    echo "       Without it the OpenMM leg can only run on CPU, which is exactly" >&2
    echo "       the silent fallback this split exists to prevent. Refusing." >&2
    exit 1
fi
module load cuda/12.9.1 2>/dev/null || true
export OPENMM_PLUGIN_DIR="${B1_OPENMM_ENV}/lib/plugins"
export LD_LIBRARY_PATH="${B1_OPENMM_ENV}/lib${CUDA_HOME:+:${CUDA_HOME}/lib64}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "OpenMM CUDA env: ${B1_OPENMM_ENV}"
echo "OPENMM_PLUGIN_DIR=${OPENMM_PLUGIN_DIR}"

"${B1_OPENMM_ENV}/bin/python" scripts/experiments/bench_dhfr_parity.py \
    --backend openmm \
    --n-steps-list "${N_STEPS_LIST}" \
    --gpu-tag h200 \
    --out "${O_JSON}"
RC2=$?
echo "--- openmm leg exit=${RC2} ---"

echo "=== [3/3] combine (validates CUDA + matching step counts) ==="
uv run python scripts/experiments/bench_dhfr_parity.py \
    --combine-prolix "${P_JSON}" \
    --combine-openmm "${O_JSON}" \
    --out "${C_JSON}"
RC3=$?

echo "=== prolix=${RC1} openmm=${RC2} combine=${RC3} ==="
echo "=== combined result: ${C_JSON} ==="
exit $(( RC1 != 0 || RC2 != 0 || RC3 != 0 ))
