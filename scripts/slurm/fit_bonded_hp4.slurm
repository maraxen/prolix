#!/bin/bash
#SBATCH --job-name=prolix_fit_hp4
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.fit_bonded_hp4.out
#SBATCH --error=outputs/logs/slurm/%j.fit_bonded_hp4.err
# Partition NOT pinned here — set per-submission via:
#   sbatch --partition=mit_preemptable ...
#   sbatch --partition=pi_so3 ...
# (Avoid mit_normal_gpu — 6h walltime cap and weaker GPUs per CLUSTER.md §1.)

# §7.1 Phase C cluster submission: bonded-parameter fitting on the HP4
# 16-molecule Lane A ensemble (4 Lane B held out per spec §10.4).
#
# Parameterized via sbatch --export:
#   MODE     : "looped" | "batched"   (default: batched)
#   PRECISION: "float32" | "float64"  (default: float32)
#   N_STEPS  : integer                 (default: 500)
#   CAMPAIGN_ID: optional bath campaign UUID for outcome attribution
#
# Submission examples:
#   sbatch --export=MODE=looped,PRECISION=float32  scripts/slurm/fit_bonded_hp4.slurm
#   sbatch --export=MODE=batched,PRECISION=float64 scripts/slurm/fit_bonded_hp4.slurm
#
# Expected wallclock: ~5-15 min for n_steps=500 on a single GPU (looped
# scales with 16 sequential per-mol scans; batched is single vmap'd scan).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

# Defaults if not provided via --export.
MODE="${MODE:-batched}"
PRECISION="${PRECISION:-float32}"
N_STEPS="${N_STEPS:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEED="${SEED:-42}"
CAMPAIGN_ID="${CAMPAIGN_ID:-}"

# Validate
if [[ "${MODE}" != "looped" && "${MODE}" != "batched" ]]; then
    echo "ERROR: MODE must be 'looped' or 'batched' (got ${MODE})"
    exit 1
fi
if [[ "${PRECISION}" != "float32" && "${PRECISION}" != "float64" ]]; then
    echo "ERROR: PRECISION must be 'float32' or 'float64' (got ${PRECISION})"
    exit 1
fi

# bath + cluster env (BTH_PROJECT_SLUG, LOG_ROOT, etc.)
source scripts/slurm/_bth_env.sh

echo "=== §7.1 fit_bonded_hp4 ==="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Mode: ${MODE}  Precision: ${PRECISION}  N_steps: ${N_STEPS}  Seed: ${SEED}"
echo "Campaign: ${CAMPAIGN_ID:-<none>}"
echo "Date: $(date -Is)"

mkdir -p outputs/results outputs/logs/slurm

DATA_DIR="data/ani1x_subset"
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: ${DATA_DIR} not found on cluster."
    exit 1
fi

# Build the --float64 flag conditional.
PRECISION_FLAG=""
if [[ "${PRECISION}" == "float64" ]]; then
    PRECISION_FLAG="--float64"
fi

# Build the --campaign flag conditional.
CAMPAIGN_FLAG=""
if [[ -n "${CAMPAIGN_ID}" ]]; then
    CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"
fi

# bath run sets BTH_RESULTS_PATH; the Python script defaults --out-json to
# that path. Outcomes get sidecar-evaluated at compact time.
# Use tags so post-hoc filtering can group the 4-way matrix.
# shellcheck disable=SC2086
uv run bth run python scripts/experiments/fit_bonded_hp4.py \
    --tag cluster --tag hp4 --tag s71 --tag fit \
    --tag "mode:${MODE}" --tag "precision:${PRECISION}" \
    ${CAMPAIGN_FLAG} \
    -- \
    --subset-dir "${DATA_DIR}" \
    --mode "${MODE}" \
    --n-steps "${N_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    ${PRECISION_FLAG}

EXIT=$?
echo "=== fit_bonded_hp4 exit=${EXIT} ==="
exit ${EXIT}
