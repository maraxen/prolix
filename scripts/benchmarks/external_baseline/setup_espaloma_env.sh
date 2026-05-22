#!/usr/bin/env bash
# setup_espaloma_env.sh — create micromamba env for bench_espaloma.py (§7.1 external-baseline)
#
# Requirements:
#   - micromamba >= 1.5  (https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
#     Quick install: "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
#   - ~8 GB free disk space (torch, dgl, openff-toolkit are large)
#   - Run from anywhere; env is created in micromamba's envs directory
#
# Why micromamba, not uv?
#   openff-toolkit and openff-units are conda-forge packages. They are yanked
#   from PyPI (0.18.0 uploaded without maintainer coordination). The correct
#   install path is conda-forge. uv cannot install conda packages.
#
# Known DGL/torch compatibility constraint:
#   DGL 2.x requires torchdata.datapipes (removed from torchdata >= 0.9).
#   DGL C++ sparse libs are pre-compiled per torch version. As of 2026-05-22,
#   DGL 2.1.0 is the latest release and has wheels for torch ≤ 2.3.
#   This env pins torch to 2.1.x for DGL 2.1.0 compatibility.
#   When DGL releases wheels for newer torch, update TORCH_VERSION below.
#
# Known numpy/conda conflict:
#   openff-toolkit from conda-forge has a transitive dep on openff-nagl which
#   pulls in pytorch (2.6.x) and numpy >= 2.x via conda. DGL 2.1.0 C extensions
#   were compiled against numpy 1.x and fail with numpy 2.x. Fix: force-install
#   numpy<2 via pip after conda setup (see below).
#   Additionally, the conda-installed libtorch files in $CONDA_PREFIX/lib/
#   conflict with the pip-installed torch 2.1.0+cu121. Fix: set LD_LIBRARY_PATH
#   so that site-packages/torch/lib/ comes BEFORE $CONDA_PREFIX/lib/:
#     export LD_LIBRARY_PATH="${MAMBA_ENV}/lib/python3.11/site-packages/torch/lib:${MAMBA_ENV}/lib:${LD_LIBRARY_PATH}"
#   This is done automatically in bench_external_baseline.slurm for the espaloma case.
#
# Tested environment (smoke-tested 2026-05-22):
#   micromamba 2.6.2, Python 3.11, torch 2.1.0+cu121, torchdata 0.7.0,
#   DGL 2.1.0 (dglteam/label/cu121), espaloma 0.4.0+1.g413eb55,
#   openff-toolkit 0.16.x (conda-forge), libstdcxx-ng (conda-forge)
#   libstdcxx-ng required on Rocky 8 / CentOS 8: system GCC 8 lacks GLIBCXX_3.4.29
#   which numpy 2.x and torch need

set -euo pipefail

ENV_NAME="${1:-espaloma-bench}"
TORCH_VERSION="2.1.0"
CUDA_TAG="cu121"        # match to your CUDA installation; options: cpu, cu118, cu121
# torchdata 0.7.0 is the torch-2.1.x matched release that still ships datapipes
# (DGL 2.1.0 graphbolt imports torchdata.datapipes; removed in torchdata 0.9.0+)
TORCHDATA_VERSION="0.7.0"

echo "==> Checking micromamba"
if ! command -v micromamba &>/dev/null; then
    echo "ERROR: micromamba not found."
    echo "Install with:  \"\${SHELL}\" <(curl -L micro.mamba.pm/install.sh)"
    exit 1
fi

echo "==> Creating micromamba env: $ENV_NAME"
# libstdcxx-ng provides a newer libstdc++ from conda-forge, required on Rocky 8 /
# CentOS 8 (GLIBCXX_3.4.29 is missing from system GCC 8; numpy 2.x needs GCC 12+).
micromamba create -n "$ENV_NAME" -y \
    -c conda-forge \
    python=3.11 \
    libstdcxx-ng \
    rdkit \
    "openmm>=7.6" \
    h5py

echo "==> Installing openff-toolkit from conda-forge"
micromamba install -n "$ENV_NAME" -y \
    -c conda-forge \
    openff-toolkit

echo "==> Installing PyTorch $TORCH_VERSION + torchdata $TORCHDATA_VERSION"
# torchdata must be pinned together with torch; DGL 2.1.0 graphbolt imports
# torchdata.datapipes which was removed in torchdata 0.9.0+ (torch 2.3+).
# numpy must be pinned to <2 before torch: openff-nagl (transitive dep of
# openff-toolkit) conda-installs numpy 2.x, but DGL 2.1.0 requires numpy 1.x.
micromamba run -n "$ENV_NAME" pip install \
    "numpy>=1.24,<2" \
    "torch==${TORCH_VERSION}" \
    "torchdata==${TORCHDATA_VERSION}" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Installing DGL 2.1.0 (dglteam channel, matched to torch ${TORCH_VERSION}/${CUDA_TAG})"
# dglteam distributes pre-compiled DGL wheels matched to each PyTorch version
micromamba run -n "$ENV_NAME" pip install \
    "dgl==2.1.0" \
    --extra-index-url "https://data.dgl.ai/wheels/${CUDA_TAG}/repo.html"

echo "==> Installing espaloma (GitHub HEAD) and remaining deps"
micromamba run -n "$ENV_NAME" pip install \
    "git+https://github.com/choderalab/espaloma.git" \
    qcportal \
    openmmforcefields \
    pandas

echo "==> Smoke test"
micromamba run -n "$ENV_NAME" python - <<'PY'
import torch
print('torch:', torch.__version__)
import dgl
print('dgl:', dgl.__version__)
import espaloma as esp
print('espaloma:', esp.__version__)
from openff.toolkit import Molecule
m = Molecule.from_smiles('CC')
g = esp.Graph(m)
print('ethane n2 nodes:', g.heterograph.number_of_nodes('n2'))
print('All imports OK — espaloma env ready')
PY

echo ""
echo "==> env '$ENV_NAME' ready. Run bench_espaloma.py with:"
echo "    micromamba run -n $ENV_NAME python scripts/benchmarks/external_baseline/bench_espaloma.py --help"
