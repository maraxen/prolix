# `prolix.export` — Portable StableHLO & WebAssembly

Hardware-agnostic StableHLO MLIR lowering context managers, artifact persistence helpers, GPU Triton `CompilerParams` resolvers, and IREE `wasm32` compilers.

```{eval-rst}
.. currentmodule:: prolix.export

.. autosummary::
   :nosignatures:

   export_context
   in_export_context
   export_energy_fn
   export_langevin_step
   save_artifact
   load_artifact
   find_iree_compile
   compile_stablehlo_mlir_to_wasm
   compile_lowered_to_wasm
   assert_wasm_artifact_under_limit
   resolve_pallas_compiler_params
```

---

## Export Context & Lowering Entry Points

```{eval-rst}
.. autofunction:: prolix.export.export_context

.. autofunction:: prolix.export.in_export_context

.. autofunction:: prolix.export.export_energy_fn

.. autofunction:: prolix.export.export_langevin_step
```

---

## Artifact Persistence & IREE `wasm32` Compilation

```{eval-rst}
.. autofunction:: prolix.export.save_artifact

.. autofunction:: prolix.export.load_artifact

.. autofunction:: prolix.export.find_iree_compile

.. autofunction:: prolix.export.compile_stablehlo_mlir_to_wasm

.. autofunction:: prolix.export.compile_lowered_to_wasm

.. autofunction:: prolix.export.assert_wasm_artifact_under_limit

.. autodata:: prolix.export.WASM_ARTIFACT_MAX_BYTES
```

---

## Pallas Compiler Tuning

```{eval-rst}
.. autofunction:: prolix.export.resolve_pallas_compiler_params
```
