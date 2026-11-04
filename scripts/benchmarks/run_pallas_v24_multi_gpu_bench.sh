#!/bin/bash
set -euo pipefail

GPU_TAG="${GPU_TAG:-b300}"
echo "========================================================================"
echo "Engaging Multi-GPU Benchmark: Pallas v2.4 E2E Ensemble Buckets (${GPU_TAG})"
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1 || true
echo "========================================================================"

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

module load cuda/12.9.1 2>/dev/null || true

mkdir -p outputs/bench

echo "=== [1/2] Verify Pallas v2.4 GPU unit tests ==="
"${PROLIX_PY}" -m pytest tests/test_ensemble_plan_pallas_v2.py -k "test_pallas_v24 or test_pallas_v22" -v

echo "=== [2/2] Run E2E Multi-Bucket / Multi-System / Ragged-Bucket Benchmark ==="
"${PROLIX_PY}" scripts/benchmarks/bench_e2e_ensemble_buckets.py \
    --out "outputs/bench/261102_${GPU_TAG}_pallas_v24_e2e_ensemble_buckets.json"

echo "=== PALLAS v2.4 (${GPU_TAG}) BENCHMARK COMPLETE ==="
