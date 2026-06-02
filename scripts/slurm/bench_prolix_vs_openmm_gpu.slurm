#!/bin/bash
# GPU throughput benchmark: Prolix (JAX) native vs OpenMM (NGC container).
#
#SBATCH --job-name=plx-bench-gpu
#SBATCH --partition=pi_so3,mit_preemptable,mit_normal_gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

set -euo pipefail

# Anchor paths to this script — safe from any submission cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common_env.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_openmm_ngc_env.sh"

openmm_load_container_module

RUNTIME="$(openmm_container_runtime)"
SIF="$(openmm_ngc_sif_path)"

SWEEP="${PLX_BENCH_SWEEP:-1000,5000,10000}"
OUT_DIR="${LOG_ROOT}/app"
mkdir -p "${OUT_DIR}"

JAX_JSON="${OUT_DIR}/bench_jax_${SLURM_JOB_ID}.json"
OMM_JSON="${OUT_DIR}/bench_omm_${SLURM_JOB_ID}.json"

echo "========================================================================"
echo "GPU Benchmark: Prolix vs OpenMM"
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
echo "Sweep: ${SWEEP}"
echo "========================================================================"

# --- 1. JAX Benchmark (Native) ---
echo -e "\n[1] Running Prolix (JAX) Native..."
export JAX_PLATFORMS=cuda

uv run python scripts/benchmarks/prolix_vs_openmm_speed.py \
    --sweep "${SWEEP}" \
    --skip-openmm \
    --json "${JAX_JSON}"

# --- 2. OpenMM Benchmark (Container) ---
if [[ -f "$SIF" ]]; then
    echo -e "\n[2] Running OpenMM (NGC Container)..."
    # Apptainer inherits host cwd before bind mounts; avoid chdir into host path.
    # Bind project root to /prolix inside container
    "$RUNTIME" exec --nv --pwd /prolix \
        -B "$(pwd):/prolix" \
        "$SIF" \
        python3 scripts/benchmarks/prolix_vs_openmm_speed.py \
            --sweep "${SWEEP}" \
            --skip-prolix \
            --json "${OMM_JSON}"
else
    echo -e "\n[2] Skipping OpenMM (SIF not found: $SIF)"
    echo "Run: sbatch scripts/slurm/pull_openmm_ngc.slurm"
fi

echo -e "\nBenchmark finished."
echo "JAX Results: ${JAX_JSON}"
echo "OMM Results: ${OMM_JSON}"

# --- 3. Merge Results (Optional) ---
FINAL_JSON="${OUT_DIR}/bench_combined_${SLURM_JOB_ID}.json"
uv run python -c "
import json, os
out = {}
if os.path.exists('${JAX_JSON}'):
    with open('${JAX_JSON}') as f: out.update(json.load(f))
if os.path.exists('${OMM_JSON}'):
    with open('${OMM_JSON}') as f:
        omm = json.load(f)
        if not out: out = omm
        else: out['results'].extend(omm.get('results', []))
if out:
    with open('${FINAL_JSON}', 'w') as f: json.dump(out, f, indent=2)
    print(f'Combined Results: {FINAL_JSON}')
"
