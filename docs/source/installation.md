# Installation & Environment Setup

Prolix targets **Python 3.13** (`>=3.13,<3.14`) across Linux `x86_64` / `aarch64` and macOS Silicon/Intel. GPU execution uses **JAX + Pallas/Triton** on NVIDIA Ampere (`sm_80`), Ada (`sm_89`), Hopper (`sm_90`), and Blackwell (`sm_120`) GPUs, while CPU and StableHLO/WebAssembly paths run without any GPU dependencies.

---

## Quick Install

::::{tab-set}

:::{tab-item} NVIDIA GPU (CUDA 12 + OpenMM)
Recommended for single-system `prolix.openmm`, multi-replica `ProlixBatchSimulation`, and Pallas v3.5 megakernels:

```bash
uv pip install "prolix[cuda,openmm]"
```

Or inside a cloned repository with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra cuda --extra openmm
```
:::

:::{tab-item} CPU Only (Development & Export)
Runs pure-JAX physics, `prolix.openmm` on CPU (`use_pallas=False`), unit tests, and StableHLO lowering without CUDA drivers:

```bash
uv pip install "prolix[cpu,openmm]"
```
:::

:::{tab-item} StableHLO → WebAssembly (`iree-compile`)
Adds `iree-base-compiler>=3.11` for compiling StableHLO MLIR artifacts into standalone `wasm32` modules:

```bash
uv pip install "prolix[cpu,wasm]"
```
:::

:::{tab-item} Documentation Builder
Installs Sphinx, `sphinx-book-theme`, `myst-nb`, `sphinx-design`, and OpenMM for building the ReadTheDocs site locally:

```bash
uv pip install "prolix[docs,openmm]"
sphinx-build -W --keep-going -b html docs/source docs/_build/html
```
:::

::::

---

## Optional Dependency Extras

| Extra | Packages Installed | When to Use |
| :--- | :--- | :--- |
| `cuda` | `jax[cuda12]` | NVIDIA GPU acceleration + Pallas/Triton v3.5 kernels (`sm_80`–`sm_120`). |
| `tpu` | `jax[tpu]` | Google Cloud TPU execution via XLA tiled primitives. |
| `cpu` | `jax` | Headless CI, macOS laptops, and CPU reference runs. |
| `openmm` | `openmm` (`8.3.x` on Linux `x86_64` for `manylinux_2_28` HPC compatibility; `>=8.4.0` elsewhere) | `prolix.openmm` drop-in backend, `EnsemblePlan.from_openmm_systems`, and OpenMM parity suites. |
| `wasm` | `iree-base-compiler>=3.11` | `prolix.export.compile_lowered_to_wasm` (`wasm32-unknown-unknown`). |
| `viz` | `py2Dmol>=1.5.0`, `flask>=3.0.0` | Interactive trajectory and structure visualization. |
| `docs` | `sphinx`, `sphinx-book-theme`, `myst-nb`, `sphinx-design`, `sphinx-copybutton`, `sphinxext-rediraffe` | Building HTML documentation under `docs/source/`. |
| `dev` | `ruff`, `pytest`, `pytest-cov`, `pytest-xdist`, `ty` | Contributor linting, static typing, and fast CI gates. |

:::{note}
**Linux `x86_64` OpenMM Wheel Policy:** Upstream OpenMM 8.4+ publishes `manylinux_2_34`-only wheels with no sdist. Prolix pins `openmm>=8.3.1,<8.4.0` on `linux/x86_64` so enterprise and HPC cluster nodes with older `glibc` (`manylinux_2_28`) resolve binary wheels cleanly out of the box.
:::

---

## Precision Modes (`single`, `mixed`, `double`)

Both `ProlixOpenMMConfig(precision=...)` and `EnsemblePlan(..., precision=...)` expose three numerical precision modes matching OpenMM's platform model:

| Mode | Force & SPME Kernels | Coordinate, Velocity & Constraint State | `JAX_ENABLE_X64` Needed? | Recommended Workload |
| :--- | :--- | :--- | :--- | :--- |
| `"single"` *(default)* | `float32` | `float32` | No | Maximum-throughput screening, equilibration, implicit solvent. |
| `"mixed"` | `float32` (with pre-cast inputs & `float32` Philox deviates) | `float64` positions, momenta, SETTLE, and RATTLE accumulators | Yes (`jax.config.update("jax_enable_x64", True)`) | Long-timescale explicit-solvent production NVE/NVT/NPT with zero coordinate drift. |
| `"double"` | `float64` | `float64` | Yes (`jax.config.update("jax_enable_x64", True)`) | High-precision energy/force parity audits and finite-difference gradient checks. |

Enable 64-bit support in JAX **before** importing `jax.numpy` or creating contexts when using `"mixed"` or `"double"` precision:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

---

## Runtime Environment Variables

Prolix automatically selects GPU architecture parameters (`num_warps`, `num_stages`) via {func}`prolix.export.resolve_pallas_compiler_params`. Advanced users and cluster operators can override kernel compilation behavior with environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `JAX_PLATFORMS` | Auto (`gpu` if available) | Set `JAX_PLATFORMS=cpu` on login/doc nodes without GPUs. |
| `JAX_ENABLE_X64` | `0` | Set `1` to unlock `float64` state accumulators for `"mixed"` and `"double"` precision. |
| `PROLIX_PALLAS_NUM_WARPS` | Kernel-dependent (`2` or `4`) | Override Triton `num_warps` per CTA during Pallas lowering. |
| `PROLIX_PALLAS_NUM_STAGES` | `1` (megakernel / `sm_120`), `2` (`sm_80`/`sm_89`/`sm_90` NL/PME) | Override Triton software pipelining stages (`keep 1` on Blackwell `sm_120` to avoid `.local` stack spill). |

---

## Verifying Your Installation

### Verify JAX + Pallas + `prolix.openmm`

```bash
python -c "
import jax, prolix
from prolix.openmm import ProlixPlatform, ProlixOpenMMConfig
print('Prolix:', prolix.__file__)
print('JAX devices:', jax.devices())
print('Default platform:', ProlixPlatform().getName())
"
```

### Verify OpenMM CUDA Provenance (Benchmark Nodes)

When benchmarking against native OpenMM CUDA, run Prolix's anti-fallback guard to confirm `libOpenMMCUDA.so` is loaded and OpenMM never silently falls back to `OpenCL`, `CPU`, or `Reference`:

```bash
python -m scripts.benchmarks.openmm_cuda_guard --check --precision single
```

### Run Static Architecture & Fast Test Gates

```bash
python3 scripts/ci/import_graph.py               # Zero import cycles & leaf invariants
python3 scripts/export_regression_pme.py --check # Explicit-PME regression doc sync
uv run pytest -m "not slow"                      # Fast unit & parity test suite
```
