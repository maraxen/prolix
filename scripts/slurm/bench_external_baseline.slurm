#!/bin/bash
#SBATCH --job-name=prolix_bench_ext
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4007,node4008
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.bench_external_baseline.out
#SBATCH --error=outputs/logs/slurm/%j.bench_external_baseline.err

# §7.1 external-baseline single-tool smoke runner.
# Pinned to node4007/4008 (RTX Pro 6000 Blackwell) — same as fit_bonded_hp4.
#
# Submission (per-tool smoke):
#   sbatch --export=ALL,TOOL=prolix,N_MOLS=64,N_CONF_CAP=100,PRECISION=float32 \
#          scripts/slurm/bench_external_baseline.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${HOME}/projects/prolix}"

TOOL="${TOOL:?TOOL env var required (prolix|dmff|torchmd|espaloma|pytorch_scratch)}"
N_MOLS="${N_MOLS:-64}"
N_CONF_CAP="${N_CONF_CAP:-100}"
PRECISION="${PRECISION:-float32}"
N_WARMUP="${N_WARMUP:-1}"
N_TRIALS="${N_TRIALS:-3}"
SEED="${SEED:-42}"

echo "=== §7.1 external-baseline bench ==="
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "Tool: ${TOOL}  N_mols: ${N_MOLS}  N_conf_cap: ${N_CONF_CAP}  Precision: ${PRECISION}"
echo "Date: $(date -Is)"

mkdir -p outputs/results/external_baseline outputs/logs/slurm

OUT="outputs/results/external_baseline/${TOOL}_n${N_MOLS}_${PRECISION}_${SLURM_JOB_ID:-local}.json"

BENCH_SCRIPT="scripts/benchmarks/external_baseline/bench_${TOOL}.py"
if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "ERROR: ${BENCH_SCRIPT} not found"
    exit 1
fi

uv run python "${BENCH_SCRIPT}" \
    --n-mols "${N_MOLS}" \
    --n-conf-cap "${N_CONF_CAP}" \
    --precision "${PRECISION}" \
    --n-warmup "${N_WARMUP}" \
    --n-trials "${N_TRIALS}" \
    --seed "${SEED}" \
    --out "${OUT}"

EXIT=$?
echo "=== output: ${OUT} ==="
cat "${OUT}" 2>/dev/null
echo
echo "=== exit=${EXIT} ==="
exit ${EXIT}
