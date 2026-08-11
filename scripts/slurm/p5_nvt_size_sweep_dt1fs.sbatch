#!/bin/bash
#SBATCH --job-name=p5_size_sweep_dt1fs
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=11:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --output=outputs/logs/slurm/%A_%a.p5_size_sweep_dt1fs.out
#SBATCH --error=outputs/logs/slurm/%A_%a.p5_size_sweep_dt1fs.err

# P5 NVT system-size sweep at dt=1.0 fs — warm-bias size-crossover characterization.
# One array task = one system size. Maps SLURM_ARRAY_TASK_ID -> n_waters.
#
# Submission with campaign ID (required):
#   sbatch --export=ALL,CAMPAIGN_ID=<campaign-id> scripts/slurm/p5_nvt_size_sweep_dt1fs.sbatch
# L3 smoke (smallest size only):
#   sbatch --array=0-0 --export=ALL,CAMPAIGN_ID=<id> scripts/slurm/p5_nvt_size_sweep_dt1fs.sbatch

set -euo pipefail

# Submitted via `myxcel submit --sbatch-arg="chdir=<worktree path>"`, which sets the
# job's initial cwd correctly (unlike SLURM_SUBMIT_DIR, which reflects myxcel's own
# ssh-session cwd on the remote host, not the worktree root). Trust pwd instead of
# re-deriving it (see scripts/slurm/p5_settle_fix_fastsuite_gpu.slurm for the same fix).
PROLIX_ROOT="$(pwd)"
cd "${PROLIX_ROOT}"

# Sanity check: fail loudly rather than silently running against a wrong
# checkout if chdir wasn't set (e.g. a manual sbatch or scontrol requeue).
if [[ ! -f "${PROLIX_ROOT}/scripts/slurm/_bth_env.sh" ]]; then
    echo "FATAL: ${PROLIX_ROOT} does not look like the prolix worktree root" \
         "(scripts/slurm/_bth_env.sh not found). Submit via" \
         "myxcel submit --sbatch-arg=\"chdir=<worktree path>\"." >&2
    exit 1
fi

# Map array task id -> system size.
SIZES=(2 16 64 216 512 895)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_WATERS="${SIZES[${TASK_ID}]}"

echo "=== P5 NVT size sweep dt=1.0 fs ==="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}  Array: ${SLURM_ARRAY_JOB_ID:-NA}_${TASK_ID}  n_waters=${N_WATERS}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Date: $(date -Is)"

RESULTS_DIR="${PROLIX_ROOT}/outputs/results"
LOGS_DIR="${PROLIX_ROOT}/outputs/logs/slurm"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

OUT="${RESULTS_DIR}/p5_size_sweep_dt1fs_n${N_WATERS}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}.json"

# JAX on Blackwell (node4007/node4008, SM120): XLA autotuning hangs without this flag.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active (${_NODE}): XLA_FLAGS=${XLA_FLAGS}"
fi

export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false

# bathos provenance: per-script sidecar at scripts/experiments/p5_nvt_size_sweep_dt1fs.bth.toml
# _common_env.sh does its own SLURM_SUBMIT_DIR-based re-cd internally — unset it here so
# that logic falls through to a BASH_SOURCE-based resolution correct relative to our
# already-correct pwd, instead of re-cd'ing to the wrong (but non-empty) submit dir.
unset SLURM_SUBMIT_DIR
source "scripts/slurm/_bth_env.sh"

CAMPAIGN_ID="${CAMPAIGN_ID:?CAMPAIGN_ID env var required — pass via sbatch --export=ALL,CAMPAIGN_ID=<id>}"

uv run bth run python scripts/experiments/p5_nvt_size_sweep_dt1fs.py \
    --tag p5 --tag nvt-size-sweep --tag dt1fs --tag "n${N_WATERS}" \
    --campaign "${CAMPAIGN_ID}" \
    --out "${OUT}" \
    -- --out "${OUT}" --n-waters "${N_WATERS}" --seeds 42,43,44 --steps 30000 --burn 10000

EXIT=$?
echo "=== output: ${OUT} ==="
cat "${OUT}" 2>/dev/null
echo
echo "=== exit=${EXIT} ==="
exit ${EXIT}
