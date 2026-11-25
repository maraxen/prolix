# Tutorials

Hands-on, end-to-end workflows covering drop-in OpenMM execution, multi-replica and ragged congeneric ligand screening, analytical force-field differentiation, and portable StableHLO/WebAssembly export.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1. Drop-In OpenMM Acceleration
:link: openmm_acceleration
:link-type: doc

Run existing OpenMM `System`, `Topology`, and `Integrator` scripts on Prolix's JAX/Pallas backend using `ProlixPlatform`, `ProlixContext`, and `ProlixSimulation` — including GPU L-BFGS minimization, NPT barostats, virtual sites, and custom Lepton forces.
:::

:::{grid-item-card} 2. Multi-Replica & Ragged Ligand Screening
:link: batch_screening
:link-type: doc

Scale from single trajectories to hundreds of parallel replicas, congeneric ligand series (`stride=0` topology deduplication), and cross-size ragged protein panels using `ProlixBatchSimulation` and `EnsemblePlan.from_openmm_systems`.
:::

:::{grid-item-card} 3. Differentiable MD & Force-Field Optimization
:link: differentiable_md
:link-type: doc

Compute exact $O(N \cdot K)$ gradients with respect to coordinates, partial charges, Lennard-Jones $\sigma/\varepsilon$, torsions, and OpenMM global context parameters using `jax.value_and_grad`, `make_energy_fn_pure`, and `optax`.
:::

:::{grid-item-card} 4. Portable StableHLO & WebGPU/WASM Export
:link: stablehlo_export
:link-type: doc

Lower pure-JAX energy functions and `EnsemblePlan` trajectory runners into hardware-agnostic StableHLO MLIR (`prolix.export`) and compile browser-deployable `wasm32` artifacts with IREE.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

openmm_acceleration
batch_screening
differentiable_md
stablehlo_export
```
