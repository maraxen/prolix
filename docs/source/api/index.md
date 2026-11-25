# API Reference

Public Python API documentation for Prolix's simulation planners, drop-in OpenMM execution backend, `KnobSurface` configuration governance, portable StableHLO/WASM export utilities, and core JAX physics primitives.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} `prolix.api` — Ensemble Planning & Observables
:link: api
:link-type: doc

`EnsemblePlan`, `Trajectory`, `Observable`, `Temperature`, `Energy`, `KineticEnergy`, `RMSD`, `Pressure`, and `prolix.api.export_run`.
:::

:::{grid-item-card} `prolix.openmm` — Drop-In OpenMM Backend
:link: openmm
:link-type: doc

`ProlixPlatform`, `ProlixOpenMMConfig`, `ProlixContext`, `ProlixState`, `ProlixSimulation`, `ProlixBatchSimulation`, `from_openmm_system`, and unit helpers.
:::

:::{grid-item-card} `prolix.knob_surface` — Configuration & Governance
:link: knob_surface
:link-type: doc

`KnobSurface`, all 9 `*Controls` sub-surface dataclasses, and `prolix.governance.redsox_adapter` U1/U5 static audit functions.
:::

:::{grid-item-card} `prolix.export` — StableHLO & WebAssembly
:link: export
:link-type: doc

`export_context`, `export_energy_fn`, `export_langevin_step`, `save_artifact`, `load_artifact`, and IREE `wasm32` compilation helpers.
:::

:::{grid-item-card} `prolix.physics` — Potentials, Integrators & Bundles
:link: physics
:link-type: doc

`make_energy_fn_pure`, `DifferentiableParams`, `PhysicsSystem`, `MolecularBundle`, `MolecularShapeSpec`, NPT integrators, and `kups_adapter`.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

api
openmm
knob_surface
export
physics
```
