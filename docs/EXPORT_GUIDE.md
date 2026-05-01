# Prolix Export Guide: StableHLO / WASM Deployment

## Overview

The `prolix.export` module exports energy functions to **StableHLO** intermediate representation via JAX's `.lower()` API, enabling deployment to WASM, WebGPU, and portable execution environments.

**Status**: v1.1 — Energy function export ready. Integrator export deferred to v1.2.

## Why Export?

Prolix models compiled in JAX need a portable, serializable form for:

1. **Web deployment** — Compile once; run in browser via WebAssembly
2. **Vendor-agnostic serving** — Same artifact runs on CPU, GPU, TPU, WASM
3. **Reproducibility** — Archive the exact computation graph alongside a paper
4. **CI validation** — Confirm that refactors don't change the compiled artifact

## Quick Start

```python
import jax.numpy as jnp
from jax_md import space

from prolix.physics.system import make_energy_fn_pure
from prolix.export import export_energy_fn, save_artifact, load_artifact

# 1. Build the explicit-params energy function (required for export)
box_vec = jnp.array([30.0, 30.0, 30.0])
displacement_fn, _ = space.periodic_general(box_vec)

params, fn = make_energy_fn_pure(
    displacement_fn, sys_dict, box_vec,
    cutoff_distance=9.0,
    pme_grid_points=64,
    pme_alpha=0.34,
)

# 2. Lower to StableHLO
positions = jnp.array(...)  # (N, 3)
lowered = export_energy_fn(fn, params, positions)

# 3. Compile and call
compiled = lowered.compile()
energy = compiled(params, positions)

# 4. Inspect the IR
print(lowered.as_text()[:500])

# 5. Save to disk
save_artifact(lowered, "energy.mlir")

# 6. Load later (returns MLIR text; re-compile via export_energy_fn + .compile())
mlir_text = load_artifact("energy.mlir")
```

## API Reference

### `export_energy_fn(fn, params, example_positions) -> Lowered`

Lower an energy function to StableHLO via `jax.jit(...).lower(...)`.

| Arg | Type | Description |
|-----|------|-------------|
| `fn` | callable | `(EnergyParams, positions) -> float` from `make_energy_fn_pure` |
| `params` | `EnergyParams` | charges, sigmas, epsilons — used for shape/dtype inference |
| `example_positions` | `(N, 3)` array | correct dtype/shape |

Returns a `jax._src.stages.Lowered` object with:
- `.compile()` → `CompiledFn` callable
- `.as_text()` → `str` StableHLO/MLIR text
- `.cost_analysis()` → estimated FLOPs / memory

### `save_artifact(artifact, path)`

Write the MLIR text from a `Lowered` artifact to a file.

### `load_artifact(path) -> str`

Read MLIR text from a file. Returns the raw string for inspection.

> **Note**: JAX does not provide a standalone MLIR-text-to-callable deserializer. To re-execute, call `export_energy_fn` again and `.compile()` the new `Lowered` object. `load_artifact` is for inspection and archival only.

### `export_langevin_step(step, example_state, **params)` — v1.2

Raises `NotImplementedError` in v1.1. See **Future Work** below.

## Prerequisites: Explicit-Params API

`jax.jit(...).lower()` requires all traced inputs to be explicit function arguments. The standard `make_energy_fn` closure bakes `charges`, `sigmas`, `epsilons` into the function — these cannot vary at call time and cannot be exported as inputs.

`make_energy_fn_pure` solves this: it returns `(params, fn)` where `params` carries the force-field arrays explicitly.

```python
# ❌ Not exportable — charges/sigmas are captured in closure
fn = make_energy_fn(displacement_fn, system, box=box)
energy = fn(positions)

# ✓ Exportable — params are explicit runtime inputs
params, fn = make_energy_fn_pure(displacement_fn, system, box)
energy = fn(params, positions)
lowered = export_energy_fn(fn, params, positions)
```

## Reproducibility

The MLIR text is deterministic: same JAX version + same inputs → identical artifact.

```python
a1 = export_energy_fn(fn, params, pos).as_text()
a2 = export_energy_fn(fn, params, pos).as_text()
assert a1 == a2  # True
```

Pin versions in your project for long-term reproducibility:

```toml
[project]
dependencies = ["jax>=0.4", "prolix>=1.1"]
```

## Future Work (v1.2)

### Langevin Step Export

`export_langevin_step` will be enabled once:

1. `Step.apply()` is refactored to always return `IntegratorState` (currently returns `IntegratorState | tuple`)
2. Optional `constraint_dofs` is handled as a compile-time parameter, not a runtime optional
3. Integration tests validate round-trip with `make_integrator()`

### Binary Serialization

`save_artifact_binary()` / `load_artifact_binary()` for 10–50× smaller artifacts.

### WASM Runtime

A lightweight WASM runtime for browser-side energy evaluation.

## Known Limitations (v1.1)

| Limitation | Workaround |
|-----------|-----------|
| Integrator export not yet supported | Use `export_energy_fn` for energy only |
| Box is fixed at factory time (NVT) | Re-export if box changes |
| `load_artifact` returns text, not callable | Re-export + `.compile()` to re-execute |
| JAX ≥ 0.4 required | Pin `jax>=0.4` in project dependencies |
