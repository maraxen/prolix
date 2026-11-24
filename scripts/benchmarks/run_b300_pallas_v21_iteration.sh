#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "Engaging B300 Benchmark: Pallas v2.1 Iteration vs Pallas v2.0 & OpenMM CUDA"
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
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

B1_OPENMM_ENV="${B1_OPENMM_ENV:-/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm}"
export OPENMM_PYTHON="${B1_OPENMM_ENV}/bin/python"
export OPENMM_PLUGIN_DIR="${B1_OPENMM_ENV}/lib/plugins"
module load cuda/12.9.1 2>/dev/null || true

mkdir -p outputs/bench outputs/dhfr_parity

echo "=== [1/4] Verify Pallas v2.3 GPU tests on B300 Triton GPU ==="
"${PROLIX_PY}" -m pytest tests/test_ensemble_plan_pallas_v2.py -v

echo "=== [2/4] Sub-Step Breakdown & Multi-Replica Ragged Bucket Scaling (v2.0 vs v2.1) ==="
if [[ ! -f outputs/bench/261030_b300_pallas_v21_breakdown_and_buckets.json || "${FORCE_BREAKDOWN:-0}" == "1" ]]; then
    "${PROLIX_PY}" scripts/benchmarks/bench_pallas_v21_breakdown_and_buckets.py \
        --out outputs/bench/261030_b300_pallas_v21_breakdown_and_buckets.json
else
    echo "--- Reusing cached outputs/bench/261030_b300_pallas_v21_breakdown_and_buckets.json ---"
fi

echo "=== [3/4] Solvated DHFR 23,558-Atom EnsemblePlan(use_pallas=True) Rollout (v2.1) ==="
N_STEPS_LIST="${N_STEPS_LIST:-100,500,2000,5000}"

"${PROLIX_PY}" scripts/experiments/bench_dhfr_parity.py \
    --backend prolix \
    --use-pallas \
    --n-steps-list "${N_STEPS_LIST}" \
    --gpu-tag b300 \
    --out outputs/dhfr_parity/prolix_pallas_v21_b300.json

if [[ ! -f outputs/dhfr_parity/openmm_cuda_b300.json ]]; then
    echo "--- Running OpenMM CUDA reference (not cached) ---"
    "${OPENMM_PYTHON}" scripts/experiments/bench_dhfr_parity.py \
        --backend openmm \
        --n-steps-list "${N_STEPS_LIST}" \
        --gpu-tag b300 \
        --out outputs/dhfr_parity/openmm_cuda_b300.json
fi

"${PROLIX_PY}" scripts/experiments/bench_dhfr_parity.py \
    --combine-prolix outputs/dhfr_parity/prolix_pallas_v21_b300.json \
    --combine-openmm outputs/dhfr_parity/openmm_cuda_b300.json \
    --out outputs/dhfr_parity/combined_pallas_v21_b300.json

echo "=== [4/4] Consolidated v2.0 vs v2.1 vs OpenMM CUDA Summary ==="
"${PROLIX_PY}" -c '
import json
from pathlib import Path

comb_v21 = json.loads(Path("outputs/dhfr_parity/combined_pallas_v21_b300.json").read_text())
breakdown = json.loads(Path("outputs/bench/261030_b300_pallas_v21_breakdown_and_buckets.json").read_text())

v20_ms = 18.202
v20_ns_day = 4.747
v21_ms = comb_v21["per_step_corrected_s"] * 1000.0
v21_ns_day = comb_v21["prolix_ns_per_day"]
omm_ms = (comb_v21["openmm_wallclock_s"] / comb_v21["openmm_n_production_steps"]) * 1000.0
omm_ns_day = comb_v21["openmm_ns_per_day"]

report = {
    "dhfr_23558_steady_state": {
        "pallas_v20_ms_per_step": v20_ms,
        "pallas_v20_ns_per_day": v20_ns_day,
        "pallas_v21_ms_per_step": v21_ms,
        "pallas_v21_ns_per_day": v21_ns_day,
        "openmm_cuda_ms_per_step": omm_ms,
        "openmm_cuda_ns_per_day": omm_ns_day,
        "speedup_v21_vs_v20": v20_ms / v21_ms,
        "ratio_v21_vs_openmm_cuda": comb_v21["ratio"],
    },
    "breakdown": breakdown["dhfr_23558_breakdown"],
    "multi_replica_bucket_scaling": breakdown["multi_replica_bucket_scaling"],
}
out_p = Path("outputs/bench/261030_b300_pallas_v21_summary.json")
out_p.write_text(json.dumps(report, indent=2))
print(json.dumps(report["dhfr_23558_steady_state"], indent=2))
'
echo "=== PALLAS v2.1 BENCHMARK COMPLETE ==="
