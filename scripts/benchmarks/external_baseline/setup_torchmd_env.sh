#!/usr/bin/env bash
# setup_torchmd_env.sh — create .venv-torchmd for bench_torchmd.py (§7.1 external-baseline)
#
# Requirements:
#   - uv >= 0.5  (https://docs.astral.sh/uv/getting-started/installation/)
#   - Python 3.11 reachable by uv
#   - CUDA 13.x toolkit headers accessible (for the cu130 torch wheel)
#   - Run from the prolix project root or scripts/benchmarks/external_baseline/
#
# Tested environment:
#   PyTorch 2.12.0+cu130, TorchMD 1.1.2, moleculekit 1.13.6
#   NumPy 2.4.6, h5py 3.16.0, PyYAML 6.0.3
#
# No venv patches required — all APIs match cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-torchmd"

echo "==> Creating TorchMD venv at $VENV_DIR"
uv venv --python 3.11 "$VENV_DIR"

echo "==> Installing PyTorch (CUDA 13.x)"
# Use the pytorch nightly index for cu130 wheel if needed, otherwise fall back to stable
uv pip install --python "$VENV_DIR/bin/python" \
    "torch>=2.0" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    || uv pip install --python "$VENV_DIR/bin/python" "torch>=2.0"

echo "==> Installing TorchMD + moleculekit"
# torchmd is on PyPI
uv pip install --python "$VENV_DIR/bin/python" \
    "torchmd==1.1.2" \
    "moleculekit>=1.13" \
    "pyyaml>=6.0" \
    "numpy>=1.20" \
    "h5py>=3.0"

echo "==> Smoke test"
"$VENV_DIR/bin/python" - <<'PY'
import torch, yaml, h5py
from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
import moleculekit
print('moleculekit:', moleculekit.__version__)
print('All imports OK')
PY

echo ""
echo "==> .venv-torchmd ready. Run bench_torchmd.py with:"
echo "    $VENV_DIR/bin/python scripts/benchmarks/external_baseline/bench_torchmd.py --help"
