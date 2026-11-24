# Import architecture hardening (Phase 1) — 2026-10-25

Goal (GEMINI.md Phase 1): stabilise the package by removing cyclic imports and the lazy-import
smells that existed only to dodge them, and keep shared types in a flat `prolix.typing`.

## Method

`scripts/ci/import_graph.py` (pure AST, no JAX needed) builds the `prolix.*` import graph and
separates **top-level** edges (executed at import time, excluding `TYPE_CHECKING`) from **lazy**
edges (inside functions). It reports module-level cycles (Tarjan SCCs) and *cycle-dodging* lazy
imports (the lazy target reaches the importer through top-level imports).
`scripts/ci/hoist_lazy_imports.py` hoists function-local `prolix.*` imports to module level
(with a `--keep` list for optional backends); run the graph check afterwards.

## Findings (before)

* 0 module-level cycles, but **133 lazy intra-package imports, 10 of them cycle-dodging** —
  every one a direct two-module cycle that was only avoided by deferring an import.

| Lazy edge (importer -> target) | Real cause | Fix (leaf module) |
|---|---|---|
| `constraints` -> `settle` (x3) | `settle` needed the SHAKE/RATTLE projectors that lived in `constraints` | `physics/constraint_projection.py` (`project_positions`, `project_momenta`); both depend on it |
| `step_system` -> `barostat`, `virtual_sites_step` | both subclass `Step`, which lived in `step_system` | `physics/step_base.py` (`Step`) |
| `tiling.planner` -> `xtrax_adapter` | adapter needs `AxisSpec/AxisDecision/BatchPlan` from `planner` | `tiling/plan_types.py` (types + `estimate_memory_theoretical`) |
| `padding` -> `batched_energy` | `padding` needed `_build_dense_exclusion_scales` | moved to `physics/neighbor_list.py` as `build_dense_exclusion_scales` (private alias kept) |
| `flash_explicit` <-> `batched_energy` (mutual lazy, exposed by hoisting) | `flash_explicit` used the four masked bonded terms | `physics/bonded_masked.py` |
| `types.bundles` -> `physics.system` (x2) | `MolecularBundle` convenience constructors forward to the higher-level factory | **kept**, allowlisted: late binding from a lower layer to a higher-layer factory is correct layering |

Every old import path still works: the original modules re-export the moved names.

## Result (after)

* 0 module-level cycles; cycle-dodging lazy imports **10 -> 2** (the documented `bundles`
  exceptions); lazy intra-package imports **133 -> 85 repo-wide**; in `prolix.physics` only the
  **4 optional Pallas** imports remain lazy (`jax.experimental.pallas` stays off the default
  import path): `flash_explicit`/`pme` -> `pallas_kernels`, `pme` -> `pme_pallas`.
* 30 existing physics/batched/tiling modules had their local `prolix.*` imports hoisted and
  their import blocks isort-ed (ruff `I`, changed files only).

## Task 1.1 (flat `typing.py`)

`prolix.typing` is already a zero-dependency leaf (fan-in 22): jaxtyping aliases, index
constants, the parameter NamedTuples, `PhysicsSystem` (+ `PaddedSystem` alias), `EnergyParams`,
`IntegratorParams/State`. The other widely shared types (`WaterModelType`, `ElectrostaticMethod`)
are already dependency-free leaves, and the remaining cross-module classes are domain types
(fitting, tiling, observables) that sit in correct layers. No further extraction was warranted;
instead the property is now **enforced** (below).

## Guardrails (`tests/test_import_architecture.py`)

1. no module-level cycle among `prolix.*`;
2. no *new* cycle-dodging lazy import (allowlist: the two `types.bundles` edges);
3. no unreviewed lazy `prolix.*` import in `prolix.physics` (allowlist: the 3 Pallas edges);
4. leaf modules stay leaves: `prolix.typing`, `water_models`, `electrostatic_methods`,
   `constraint_projection`, `tiling.plan_types` import nothing from prolix; `step_base` only
   `prolix.typing`.

Allowlists carry reasons and are meant to only shrink.

## Rules of thumb for contributors

* A lazy `prolix.*` import to "avoid a circular import" means a shared piece wants to be a leaf
  module — extract it, do not defer the import.
* Optional heavy backends (Pallas) may stay lazy, but must be added to the allowlist with a reason.
* Re-export moved names from the old module so public import paths do not break.
