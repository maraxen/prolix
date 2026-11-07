#!/bin/bash
set -euo pipefail

GPU_TAG="${GPU_TAG:-b300}"
echo "========================================================================"
echo "Engaging Multi-GPU Benchmark: Pallas v3.0 E2E Ensemble Buckets (${GPU_TAG})"
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1 || true
echo "========================================================================"

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh || true
fi
module load cuda/12.9.1 2>/dev/null || true

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_CACHE_DIR="/tmp/uv_cache_prolix_${SLURM_JOB_ID:-$$}"
if [[ -d "/orcd/pool/008/so3_shared/marielle/projects/prolix/tmp_xtrax_pkg" ]]; then
    export PYTHONPATH="${PWD}/src:/orcd/pool/008/so3_shared/marielle/projects/prolix/tmp_xtrax_pkg:${PYTHONPATH:-}"
else
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

if [[ -x "/orcd/pool/008/so3_shared/marielle/projects/.venv/bin/python" ]]; then
    PROLIX_PY="/orcd/pool/008/so3_shared/marielle/projects/.venv/bin/python"
else
    PROLIX_PY="${PWD}/.venv/bin/python"
fi

OMM_ROOT="/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm"
if [[ -x "${OMM_ROOT}/bin/python" ]]; then
    export OPENMM_PYTHON="${OMM_ROOT}/bin/python"
    export OPENMM_PLUGIN_DIR="${OMM_ROOT}/lib/plugins"
    export LD_LIBRARY_PATH="${OMM_ROOT}/lib:${OMM_ROOT}/lib/plugins:/orcd/software/core/001/pkg/cuda/12.9.1/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"
fi

mkdir -p outputs/bench

echo "=== [1/2] Verify Pallas v2.4..v3.1 GPU unit tests + architecture/export/parity guardrails ==="
if [[ "${FAST_PYTEST:-0}" == "1" ]]; then
    "${PROLIX_PY}" -m pytest tests/test_ensemble_plan_pallas_v2.py -k "test_pallas_v31" -v -o addopts=""
elif [[ "${SKIP_PYTEST:-0}" != "1" ]]; then
    "${PROLIX_PY}" -m pytest tests/test_ensemble_plan_pallas_v2.py -k "test_pallas_v31 or test_pallas_v30 or test_pallas_v26 or test_pallas_v25 or test_pallas_v24" -v -o addopts=""
    "${PROLIX_PY}" -m pytest tests/test_import_architecture.py tests/test_stablehlo_export_guardrails.py tests/api/test_v1_ensemble_plan_parity.py -v -o addopts=""
fi

echo "=== [2/2] Run E2E Multi-Bucket / Multi-System / Ragged-Bucket + Sub-Step Profile + OpenMM CUDA Benchmark ==="
"${PROLIX_PY}" scripts/benchmarks/bench_e2e_ensemble_buckets.py \
    --out "outputs/bench/261106_${GPU_TAG}_pallas_v31_e2e_ensemble_buckets.json"

echo "=== PALLAS v3.1 (${GPU_TAG}) BENCHMARK COMPLETE ==="
