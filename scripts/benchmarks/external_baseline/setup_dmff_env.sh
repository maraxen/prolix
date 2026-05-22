#!/usr/bin/env bash
# setup_dmff_env.sh — create .venv-dmff for bench_dmff.py (§7.1 external-baseline)
#
# Requirements:
#   - uv >= 0.5  (https://docs.astral.sh/uv/getting-started/installation/)
#   - Python 3.11 reachable by uv (uv python install 3.11 if needed)
#   - Run from the prolix project root or scripts/benchmarks/external_baseline/
#
# Tested environment (2026-05-22):
#   JAX 0.4.25, jaxlib 0.4.25, DMFF 1.0.1.dev (GitHub main), OpenMM 8.1.1
#   NumPy 1.26.x (JAX 0.4.25 supports numpy 1.x; OpenMM 8.1.1 compiled against 1.x)
#   OpenMM 8.1.1 required on Rocky 8 / glibc 2.28; 8.5.1+ requires glibc 2.34
#   No v0.2.7 tag exists in DMFF repo (went v0.2.0 → v1.0.0)
#   On glibc 2.34+ (local/Ubuntu): jax[cpu]==0.10.1 + numpy>=2.0 + openmm==8.5.1 works
#
# Known venv patches applied after install (documented below):
#   1. dmff/settings.py: jax.config.config was removed in JAX 0.4.7+;
#      patched to: import jax; config = jax.config
#   2. dmff/common/nblist.py: np.fromiter(permutations, dtype=np.dtype(int,2))
#      is broken in NumPy 2.x; patched to: np.array(list(...), dtype=np.int32)
#
# Why patches instead of older JAX?
#   JAX 0.10.1 requires numpy>=2.0, so we can't pin numpy 1.x without
#   downgrading JAX to ~0.4.6. The patches are one-liners that forward-port
#   DMFF 0.2.7 to the modern stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-dmff"

echo "==> Creating DMFF venv at $VENV_DIR"
uv venv --python 3.11 "$VENV_DIR"

echo "==> Installing JAX (CPU) + core scientific stack"
# On Rocky 8 / glibc 2.28: pin JAX 0.4.25 + numpy 1.26.x.
# JAX 0.10.1 requires numpy>=2.0, but OpenMM 8.x PyPI wheels available for
# glibc 2.28 (8.1.1, 8.2.0, 8.3.0) were compiled against numpy 1.x and fail
# with NumPy 2.x ABI. On newer systems (glibc 2.34+) with OpenMM 8.5.1,
# upgrading to jax[cpu]==0.10.1 + numpy>=2.0 is safe.
uv pip install --python "$VENV_DIR/bin/python" \
    "jax[cpu]==0.4.25" \
    "jaxlib==0.4.25" \
    "numpy>=1.26,<2.0" \
    "optax>=0.1.4" \
    "h5py>=3.0"

echo "==> Installing OpenMM"
# openmm 8.5.1+ requires glibc 2.34 (manylinux_2_34). Rocky 8 / CentOS 8 has
# glibc 2.28, so we pin 8.1.1 which ships manylinux_2_17 wheels (compatible).
# On newer systems (glibc 2.34+) upgrading to 8.5.1 is safe.
uv pip install --python "$VENV_DIR/bin/python" \
    "openmm>=8.1.1,<8.2.0"

echo "==> Installing DMFF (GitHub main HEAD)"
# DMFF is not on PyPI; there is no 0.2.7 tag — the repo jumped v0.2.0 → v1.0.0.
# main HEAD (1.0.x dev) is what was smoke-tested locally.
uv pip install --python "$VENV_DIR/bin/python" \
    "git+https://github.com/deepmodeling/DMFF.git"

echo "==> Applying venv patches for JAX/NumPy compatibility"

# Patch 1: dmff/settings.py — jax.config.config shim
SETTINGS_FILE="$VENV_DIR/lib/python3.11/site-packages/dmff/settings.py"
if [ -f "$SETTINGS_FILE" ]; then
    # Replace: from jax.config import config
    # With:    import jax; config = jax.config  # compat shim
    python3 - "$SETTINGS_FILE" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
if 'from jax.config import config' in text:
    text = text.replace(
        'from jax.config import config',
        'import jax\nconfig = jax.config  # compat shim: jax.config.config removed in JAX 0.4.7+'
    )
    with open(path, 'w') as f:
        f.write(text)
    print(f"  patched {path}")
else:
    print(f"  {path}: no patch needed (already updated or different version)")
PY
fi

# Patch 2: dmff/common/nblist.py — np.fromiter tuple dtype broken in NumPy 2.x
NBLIST_FILE="$VENV_DIR/lib/python3.11/site-packages/dmff/common/nblist.py"
if [ -f "$NBLIST_FILE" ]; then
    python3 - "$NBLIST_FILE" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
old = 'np.fromiter(permutations(range(natoms), 2), dtype=np.dtype(int, 2))'
new = 'np.array(list(permutations(range(natoms), 2)), dtype=np.int32)  # compat: np.fromiter broken for tuples in NumPy 2.x'
if old in text:
    text = text.replace(old, new)
    with open(path, 'w') as f:
        f.write(text)
    print(f"  patched {path} ({text.count(new)} occurrence(s))")
else:
    print(f"  {path}: no patch needed")
PY
fi

echo "==> Smoke test"
"$VENV_DIR/bin/python" -c "
import jax, dmff, openmm, optax, h5py
print('JAX:', jax.__version__)
print('OpenMM:', openmm.__version__)
print('optax:', optax.__version__)
print('h5py:', h5py.__version__)
print('DMFF: installed (no __version__ attribute)')
print('All imports OK')
" 2>&1 | grep -v "WDDM\|CUDA\|GPU\|Falling back\|warn"

echo ""
echo "==> .venv-dmff ready. Run bench_dmff.py with:"
echo "    $VENV_DIR/bin/python scripts/benchmarks/external_baseline/bench_dmff.py --help"
