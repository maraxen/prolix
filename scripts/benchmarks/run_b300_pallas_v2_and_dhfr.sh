#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "Engaging B300 Benchmark: Pallas v2 Megakernel + Native CUDA OpenMM + DHFR"
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Date: $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
echo "========================================================================"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

B1_OPENMM_ENV="${B1_OPENMM_ENV:-/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm}"
export OPENMM_PYTHON="${B1_OPENMM_ENV}/bin/python"
export OPENMM_PLUGIN_DIR="${B1_OPENMM_ENV}/lib/plugins"
module load cuda/12.9.1 2>/dev/null || true

mkdir -p outputs/bench outputs/dhfr_parity

echo "=== [1/4] Verify tests/test_ensemble_plan_pallas_v2.py on B300 Triton GPU ==="
uv run --extra cuda pytest tests/test_ensemble_plan_pallas_v2.py -v

echo "=== [2/4] Synthetic Scaling vs Native OpenMM CUDA (N=128..16384) ==="
uv run --extra cuda python scripts/benchmarks/pallas_vs_baseline.py \
    --kernel openmm_comparison \
    --sizes 128,512,1024,2048,4096,8192,16384 \
    --repeats 9 \
    --out outputs/bench/261029_b300_openmm_cuda_comparison.json

echo "=== [3/4] Solvated DHFR 23,558-Atom Benchmark (Prolix XLA vs Pallas v2 vs OpenMM CUDA) ==="
N_STEPS_LIST="${N_STEPS_LIST:-100,500,2000,5000}"

echo "--- [3a] Prolix XLA (use_pallas=False) ---"
uv run --extra cuda python scripts/experiments/bench_dhfr_parity.py \
    --backend prolix \
    --n-steps-list "${N_STEPS_LIST}" \
    --gpu-tag b300 \
    --out outputs/dhfr_parity/prolix_xla_b300.json

echo "--- [3b] Prolix Pallas v2 (use_pallas=True) ---"
uv run --extra cuda python scripts/experiments/bench_dhfr_parity.py \
    --backend prolix \
    --use-pallas \
    --n-steps-list "${N_STEPS_LIST}" \
    --gpu-tag b300 \
    --out outputs/dhfr_parity/prolix_pallas_v2_b300.json

echo "--- [3c] Native OpenMM CUDA (b1-openmm env) ---"
"${OPENMM_PYTHON}" scripts/experiments/bench_dhfr_parity.py \
    --backend openmm \
    --n-steps-list "${N_STEPS_LIST}" \
    --gpu-tag b300 \
    --out outputs/dhfr_parity/openmm_cuda_b300.json

echo "--- [3d] Combine XLA + OpenMM CUDA and Pallas v2 + OpenMM CUDA ---"
uv run --no-sync python scripts/experiments/bench_dhfr_parity.py \
    --combine-prolix outputs/dhfr_parity/prolix_xla_b300.json \
    --combine-openmm outputs/dhfr_parity/openmm_cuda_b300.json \
    --out outputs/dhfr_parity/combined_xla_b300.json

uv run --no-sync python scripts/experiments/bench_dhfr_parity.py \
    --combine-prolix outputs/dhfr_parity/prolix_pallas_v2_b300.json \
    --combine-openmm outputs/dhfr_parity/openmm_cuda_b300.json \
    --out outputs/dhfr_parity/combined_pallas_v2_b300.json

echo "=== [4/4] Single-Step DHFR 23,558-Atom Energy/Force Parity & Consolidated Report ==="
uv run --extra cuda python -c '
import json, time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import scripts.experiments.bench_dhfr_parity as B
from prolix.api.bundle_md import displacement_fn_for_bundle, energy_fn_from_bundle
from prolix.api.ensemble_plan import _encode_bundle_nl
import prolix.physics.neighbor_list as nl_mod

ff_path = B._resolve_ff_path()
sys_data = B._build_dhfr_system_dict(ff_path)
bundle = B._build_bundle_from_sys_data(sys_data)
disp_fn, _ = displacement_fn_for_bundle(bundle)
box_vec = jnp.diag(bundle.box)
neighbor_fn = nl_mod.make_neighbor_list_fn(disp_fn, box_vec, 9.0, capacity_multiplier=1.25, dr_threshold=1.0)
nbr_raw = neighbor_fn.allocate(bundle.positions)
nbr_enc = _encode_bundle_nl(nbr_raw, bundle)

e_fn_xla = energy_fn_from_bundle(bundle, build_dense_exclusions=False, use_pallas=False, use_pallas_pme=False, excl_mode="table")
e_fn_pal = energy_fn_from_bundle(bundle, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True, excl_mode="pre_encoded")

vg_xla = jax.jit(jax.value_and_grad(lambda r, n: e_fn_xla(r, neighbor=n)))
vg_pal = jax.jit(jax.value_and_grad(lambda r, n: e_fn_pal(r, neighbor=n)))

e_xla, g_xla = vg_xla(bundle.positions, nbr_raw)
jax.block_until_ready((e_xla, g_xla))
e_pal, g_pal = vg_pal(bundle.positions, nbr_enc)
jax.block_until_ready((e_pal, g_pal))

t_xla = []
t_pal = []
for _ in range(9):
    t0 = time.perf_counter()
    ex, gx = vg_xla(bundle.positions, nbr_raw)
    jax.block_until_ready((ex, gx))
    t_xla.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    ep, gp = vg_pal(bundle.positions, nbr_enc)
    jax.block_until_ready((ep, gp))
    t_pal.append(time.perf_counter() - t0)

med_xla_s = float(np.median(t_xla))
med_pal_s = float(np.median(t_pal))

comb_xla = json.loads(Path("outputs/dhfr_parity/combined_xla_b300.json").read_text())
comb_pal = json.loads(Path("outputs/dhfr_parity/combined_pallas_v2_b300.json").read_text())

summary = {
    "meta": {
        "system": "DHFR_JAC_23558_atoms",
        "n_atoms": int(bundle.n_atoms),
        "nl_capacity": int(nbr_raw.idx.shape[1]),
        "device": str(jax.devices()[0]),
        "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
        "jax_version": jax.__version__,
    },
    "single_step_force_eval": {
        "xla_median_ms": med_xla_s * 1000.0,
        "pallas_v2_median_ms": med_pal_s * 1000.0,
        "force_eval_speedup_pallas_v2_vs_xla": med_xla_s / med_pal_s,
        "e_xla_kcal": float(e_xla),
        "e_pallas_v2_kcal": float(e_pal),
        "rel_dE": float(abs(e_pal - e_xla) / max(abs(float(e_xla)), 1e-12)),
        "max_abs_dF_kcal_mol_A": float(jnp.max(jnp.abs(g_pal - g_xla))),
        "rms_dF_kcal_mol_A": float(jnp.sqrt(jnp.mean((g_pal - g_xla) ** 2))),
    },
    "ensemble_plan_nvt_settle_rollout": {
        "prolix_xla": comb_xla,
        "prolix_pallas_v2": comb_pal,
        "speedup_pallas_v2_vs_xla_steady_state": (
            comb_xla["per_step_corrected_s"] / comb_pal["per_step_corrected_s"]
        ),
        "ratio_pallas_v2_vs_openmm_cuda": comb_pal["ratio"],
        "ratio_xla_vs_openmm_cuda": comb_xla["ratio"],
    },
}
out_p = Path("outputs/bench/261029_b300_dhfr_23558_benchmark.json")
out_p.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["single_step_force_eval"], indent=2))
print("Speedup Pallas v2 vs XLA (steady state):", summary["ensemble_plan_nvt_settle_rollout"]["speedup_pallas_v2_vs_xla_steady_state"])
print("Prolix XLA ns/day:", comb_xla["prolix_ns_per_day"], "Prolix Pallas v2 ns/day:", comb_pal["prolix_ns_per_day"], "OpenMM CUDA ns/day:", comb_pal["openmm_ns_per_day"])
'
echo "=== ALL BENCHMARKS COMPLETE ==="
