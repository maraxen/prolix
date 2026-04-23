#!/usr/bin/env bash
# Epic A: two Tier-1 tip3p_ke_compare runs (replicas=5, x64 Prolix, --engine both).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${TIP3P_KE_OUT:-$ROOT/outputs/logs/local_tip3p_ke}"
mkdir -p "$OUT"
cd "$ROOT"
export OPENMM_INSTALL_MODE="${OPENMM_INSTALL_MODE:-ephemeral}"

COMMON=(python scripts/benchmarks/tip3p_ke_compare.py
  --engine both --jax-x64 on --n-waters 33 --steps 30000 --burn 10000 --sample-every 10
  --replicas 5 --timing-mode both --warmup-steps 100 --measure-steps 500
  --dt-fs 2.0 --temperature-k 300 --gamma-ps 1.0 --openmm-integrator middle)

echo "=== diag_linear_com_off ==="
uv run --with openmm "${COMMON[@]}" --remove-cmmotion false \
  >"$OUT/tier1_both_diag_linear_com_off_x64_r5.json"

echo "=== openmm_ref_linear_com_on ==="
uv run --with openmm "${COMMON[@]}" --remove-cmmotion true \
  >"$OUT/tier1_both_openmm_ref_linear_com_on_x64_r5.json"

( cd "$OUT" && sha256sum tier1_both_diag_linear_com_off_x64_r5.json tier1_both_openmm_ref_linear_com_on_x64_r5.json \
  >tier1_sprint_sha256sums.txt )
echo "Wrote $OUT/tier1_*_r5.json and tier1_sprint_sha256sums.txt"
