# Tutorial 4: Portable StableHLO & WebGPU/WASM Export (`prolix.export`)

Prolix enforces a strict architectural separation between GPU-specific Pallas/Triton megakernels and portable **StableHLO** lowering (`tests/test_stablehlo_export_guardrails.py`). Whenever JAX traces inside {func}`prolix.export.export_context`, every kernel entry point detects `in_export_context() == True` and routes through pure-JAX tiled primitives—guaranteeing **zero** `triton`, `mosaic`, `pallas`, `xla.gpu`, `cublas`, or `cusolver` custom calls in the emitted MLIR module.

This lets you compile identical physics and `EnsemblePlan` integrators to:
- **Portable StableHLO MLIR** (`.mlir`) for ahead-of-time XLA/IREE deployment on CPU, TPU, Metal, or WebGPU.
- **Standalone WebAssembly (`wasm32-unknown-unknown`)** binaries under the **50 MiB** browser cap (`WASM_ARTIFACT_MAX_BYTES`) via `iree-compile`.

---

## 1. Lowering an Explicit-Parameter Energy Function to StableHLO

Because `jax.jit.lower` requires all runtime tensors as explicit inputs rather than huge Python closures, always pair {func}`~prolix.export.export_energy_fn` with {func}`~prolix.physics.make_energy_fn_pure`:

```python
import jax.numpy as jnp
from jax_md import space
from prolix.export import export_energy_fn, load_artifact, save_artifact
from prolix.physics import make_energy_fn_pure

# 1. Build pure explicit-parameter energy callable
box = jnp.asarray(physics_system.box_size)
displacement_fn, _ = space.periodic(box)
params, energy_fn = make_energy_fn_pure(
    displacement_fn,
    physics_system,
    cutoff_distance=9.0,
    pme_grid_points=32,
)

# 2. Lower inside export_context() -> pure StableHLO LoweringResult (no Triton custom calls)
example_pos = jnp.zeros((params.charges.shape[0], 3), dtype=jnp.float32)
lowered = export_energy_fn(energy_fn, params, example_pos)

# 3. Inspect cost analysis & persist StableHLO MLIR text to disk
print("StableHLO cost estimate:", lowered.cost_analysis())
save_artifact(lowered, "artifacts/protein_energy.mlir")

mlir_text = load_artifact("artifacts/protein_energy.mlir")
assert "triton" not in mlir_text.lower()

# 4. Compile & execute locally without retracing
compiled_fn = lowered.compile()
e_val = compiled_fn(params, positions_f32)
print(f"Compiled StableHLO energy: {float(e_val):.4f} kcal/mol")
```

---

## 2. Compiling StableHLO to WebAssembly (`wasm32`) with IREE

With the optional `wasm` extra installed (`uv pip install "prolix[cpu,wasm]"`), {func}`~prolix.export.compile_lowered_to_wasm` invokes `iree-compile` targeting `wasm32-unknown-unknown` and {func}`~prolix.export.assert_wasm_artifact_under_limit` enforces the 50 MiB browser deployment budget:

```python
from prolix.export import (
    WASM_ARTIFACT_MAX_BYTES,
    assert_wasm_artifact_under_limit,
    compile_lowered_to_wasm,
    find_iree_compile,
)

if find_iree_compile() is not None:
    wasm_path = compile_lowered_to_wasm(lowered, "artifacts/protein_energy.wasm")
    size_bytes = assert_wasm_artifact_under_limit(
        wasm_path, max_bytes=WASM_ARTIFACT_MAX_BYTES
    )
    print(f"Built browser-ready WASM module: {wasm_path} ({size_bytes / 1024:.1f} KiB)")
```

---

## 3. Exporting Complete Trajectory Runners (`prolix.api.export_run`)

To run interactive molecular dynamics in browser/edge runtimes where users dynamically adjust **temperature** (`kT`) and **timestep** (`dt`) without recompiling the WASM module, use the export-shaped factories in `prolix.api.export_run`:

- {func}`~prolix.api.export_run.make_single_trajectory_fn`: `(seed, dt_fs, kT_kcal) -> (n_steps, n_atoms, 3)` positions in `Å`.
- {func}`~prolix.api.export_run.make_hetero_trajectory_fn`: `(seed_base, dt_fs, kT_kcal) -> tuple[(n_steps, n_atoms_i, 3), ...]` across multi-system ensembles.
- {func}`~prolix.api.export_run.make_smoke_diagnostics_fn`: `(seed, dt_fs, kT_kcal) -> (positions, temperatures_K, kinetic_energies_kcal)` for live browser telemetry traces.

```python
import jax
import jax.numpy as jnp
from prolix.api.export_run import make_single_trajectory_fn, make_smoke_diagnostics_fn
from prolix.export import compile_lowered_to_wasm, export_context, save_artifact
from prolix.simulate import BOLTZMANN_KCAL

# Build a 100-step trajectory runner with dynamic (seed, dt_fs, kT_kcal) scalar inputs
traj_fn = make_single_trajectory_fn(bundle, n_steps=100)
diag_fn = make_smoke_diagnostics_fn(bundle, n_steps=100, gamma=10.0)

seed_ex = jnp.uint32(0)
dt_ex = jnp.float32(1.0)                       # femtoseconds
kT_ex = jnp.float32(300.0 * BOLTZMANN_KCAL)    # kcal/mol

with export_context():
    lowered_traj = jax.jit(traj_fn).lower(seed_ex, dt_ex, kT_ex)
    lowered_diag = jax.jit(diag_fn).lower(seed_ex, dt_ex, kT_ex)

save_artifact(lowered_traj, "artifacts/trajectory_100steps.mlir")
save_artifact(lowered_diag, "artifacts/diagnostics_100steps.mlir")
```

---

## 4. Wrapping Custom Step Functions (`export_langevin_step`)

If you construct custom low-level integrators over flat `LangevinState` PyTrees (which avoid `Optional` fields for strict `jax.export` compatibility), wrap the step function with {func}`~prolix.export.export_langevin_step` so every force evaluation inside the step automatically executes under `export_context()`:

```python
import jax
from prolix.export import export_langevin_step
from prolix.types.integrators import IntegratorConfig, LangevinState

cfg = IntegratorConfig(
    thermostat="langevin",
    has_pbc=False,
    dt=1.0,
    kT=0.59616,
    gamma=1.0,
)

portable_step = export_langevin_step(raw_step_fn, cfg)
lowered_step = jax.jit(portable_step).lower(initial_langevin_state)
```
