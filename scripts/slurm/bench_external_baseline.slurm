#!/bin/bash
#SBATCH --job-name=prolix_bench_ext
#SBATCH --partition=pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/%j.bench_external_baseline.out
#SBATCH --error=outputs/logs/slurm/%j.bench_external_baseline.err

# §7.1 external-baseline single-tool smoke runner.
# Default partition: pi_so3 (RTX Pro 6000 Blackwell, sm_120).
# For tools locked to CUDA 12.1 / torch 2.1.0 (espaloma, DMFF):
#   override partition + gres at sbatch time to target A100 (sm_80):
#     sbatch --partition=mit_preemptable --nodelist=<a100-node> --gres=gpu:a100:1 \
#            --export=ALL,TOOL=espaloma,...,HARDWARE_TAG=a100-sm80 \
#            scripts/slurm/bench_external_baseline.slurm
# Hardware compatibility:
#   prolix/TorchMD  → Blackwell sm_120 (CUDA 13, torch 2.12.0+cu130 / JAX 0.8.1)
#   DMFF/espaloma   → A100 sm_80 (CUDA 12.1, JAX 0.4.25+cu121 / torch 2.1.0+cu121)
#
# Submission (per-tool smoke, Blackwell default):
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
HARDWARE_TAG="${HARDWARE_TAG:-rtx-pro-6000-blackwell}"

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

# Each external tool has its own isolated venv; route to the correct interpreter.
# The prolix comparator uses the project uv venv.
BENCH_DIR="scripts/benchmarks/external_baseline"
case "${TOOL}" in
    prolix)
        PYTHON_INTERP="python"  # project uv venv already active via outer 'uv run bth run'
        ;;
    dmff)
        PYTHON_INTERP="${BENCH_DIR}/.venv-dmff/bin/python"
        ;;
    torchmd)
        PYTHON_INTERP="${BENCH_DIR}/.venv-torchmd/bin/python"
        ;;
    espaloma)
        PYTHON_INTERP="micromamba run -n espaloma-bench python"
        ;;
    *)
        echo "ERROR: unknown tool '${TOOL}' — expected prolix|dmff|torchmd|espaloma"
        exit 1
        ;;
esac

echo "Python: ${PYTHON_INTERP}"

# FORCE_CPU=1: hide the GPU for all tools (for apples-to-apples CPU comparison).
# Default (0): each tool uses its native device detection.
FORCE_CPU="${FORCE_CPU:-0}"
if [[ "${FORCE_CPU}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "FORCE_CPU=1: GPU hidden for all tools"
fi

# dmff: jaxlib 0.4.25+cuda12.cudnn89 does not bundle cuDNN; requires libcudnn.so.8
#   from the nvidia-cudnn-cu12 pip package installed in .venv-dmff.
#   libcuda.so.1 is available on A100 compute nodes at /usr/lib64/ (no extra path).
if [[ "${TOOL}" == "dmff" ]]; then
    _DMFF_SP="${BENCH_DIR}/.venv-dmff/lib/python3.11/site-packages"
    _CUDNN_LIB="${_DMFF_SP}/nvidia/cudnn/lib"
    export LD_LIBRARY_PATH="${_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# espaloma: three LD_LIBRARY_PATH layers required in priority order:
#   1. pip torch lib — pip-installed torch 2.1.0+cu121 bundles some CUDA libs;
#      must come BEFORE conda lib/ to prevent the stale conda-installed libtorch
#      (from openff-nagl → torch 2.6.x via conda) from being loaded instead.
#   2. nvidia cuda pip packages — DGL 2.1.0+cu121 links against libcusparse.so.12
#      etc. which live in site-packages/nvidia/*/lib/ (installed as nvidia-*-cu12).
#   3. conda lib/ — provides libstdcxx-ng (GLIBCXX_3.4.29 required by pip numpy).
if [[ "${TOOL}" == "espaloma" ]]; then
    _MAMBA_ENV="${HOME}/.local/share/mamba/envs/espaloma-bench"
    _SP="${_MAMBA_ENV}/lib/python3.11/site-packages"
    _TORCH_LIB="${_SP}/torch/lib"
    _NV_LIBS="${_SP}/nvidia/cusparse/lib:${_SP}/nvidia/cublas/lib:${_SP}/nvidia/cuda_runtime/lib:${_SP}/nvidia/cufft/lib:${_SP}/nvidia/curand/lib:${_SP}/nvidia/cusolver/lib:${_SP}/nvidia/nvjitlink/lib"
    export LD_LIBRARY_PATH="${_TORCH_LIB}:${_NV_LIBS}:${_MAMBA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# bath provenance: per-script sidecar at scripts/benchmarks/external_baseline/bench_${TOOL}.bth.toml
# Campaign: hp4-s71-external-baseline (edbd0b84-c0e2-4d2d-a89f-5f9fe8ae3aff)
# CAMPAIGN_ID can be overridden via --export=ALL,CAMPAIGN_ID=<id> at sbatch time.
source scripts/slurm/_bth_env.sh

CAMPAIGN_ID="${CAMPAIGN_ID:-edbd0b84-c0e2-4d2d-a89f-5f9fe8ae3aff}"
CAMPAIGN_FLAG="--campaign ${CAMPAIGN_ID}"

uv run bth run ${PYTHON_INTERP} "${BENCH_SCRIPT}" \
    --tag s71 --tag external-baseline --tag scope-a \
    --tag "tool:${TOOL}" --tag "n_mols:${N_MOLS}" --tag "precision:${PRECISION}" \
    --tag "hardware:${HARDWARE_TAG}" \
    ${CAMPAIGN_FLAG} \
    --out "${OUT}" \
    --n-mols "${N_MOLS}" \
    --n-conf-cap "${N_CONF_CAP}" \
    --precision "${PRECISION}" \
    --n-warmup "${N_WARMUP}" \
    --n-trials "${N_TRIALS}" \
    --seed "${SEED}" \
    --hardware-tag "${HARDWARE_TAG}" \
    --out "${OUT}"

EXIT=$?
echo "=== output: ${OUT} ==="
cat "${OUT}" 2>/dev/null
echo
echo "=== exit=${EXIT} ==="
exit ${EXIT}
