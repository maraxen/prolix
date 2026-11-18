# Prolix Documentation

**Prolix** is a GPU-native molecular dynamics and differentiable protein physics engine built on **JAX** and **Pallas/Triton**. It combines a zero-friction, drop-in **OpenMM** execution backend (`prolix.openmm`) with fused multi-replica and ragged congeneric batch kernels (`EnsemblePlan`, `ProlixBatchSimulation`), $O(N \cdot K)$ analytical `@jax.custom_vjp` coordinate and force-field gradients, and portable **StableHLO / WebAssembly** deployment (`prolix.export`).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Drop-In OpenMM Acceleration
:link: tutorials/openmm_acceleration
:link-type: doc

Replace `openmm.app.Simulation` or `openmm.Context` in one import (`from prolix.openmm import Simulation, ProlixPlatform`). Supports explicit PME, OBC2 GBSA, SETTLE/SHAKE/RATTLE, virtual sites, Lepton custom forces, and NPT barostats.
:::

:::{grid-item-card} High-Throughput & Ragged Screening
:link: tutorials/batch_screening
:link-type: doc

Simulate hundreds of replicas, congeneric ligand series (`stride=0` topology deduplication at ~45 MB/replica vs ~450–580 MB/replica under `CUDA MPS`), and cross-size ragged batches on a single GPU via `ProlixBatchSimulation` and `EnsemblePlan`.
:::

:::{grid-item-card} Differentiable MD & Force Fields
:link: tutorials/differentiable_md
:link-type: doc

Differentiate energies and forces with respect to atomic coordinates, partial charges, Lennard-Jones $\sigma/\varepsilon$, torsions, and OpenMM global context parameters using memory-safe $O(N \cdot K)$ analytical `@jax.custom_vjp` rules.
:::

:::{grid-item-card} Portable StableHLO & WebGPU/WASM
:link: tutorials/stablehlo_export
:link-type: doc

Lower pure-JAX energy and trajectory programs (`prolix.export`) into hardware-agnostic StableHLO MLIR with zero GPU-locked custom calls, ready for IREE WebAssembly (`wasm32`) and browser deployment under 50 MiB.
:::

::::

---

## Five-Line Quickstart

### 1. Drop-In Single-System OpenMM Simulation

```python
import openmm
from openmm import app, unit
from prolix.openmm import ProlixPlatform, Simulation

pdb = app.PDBFile("system.pdb")
ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.9 * unit.nanometer,
    constraints=app.HBonds,
)
integrator = openmm.LangevinMiddleIntegrator(
    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
)

platform = ProlixPlatform()
props = {"UsePallas": "true", "NonbondedEngine": "auto", "Precision": "mixed"}
sim = Simulation(pdb.topology, system, integrator, platform, props)
sim.context.setPositions(pdb.positions)
sim.minimizeEnergy()
sim.step(10_000)
state = sim.context.getState(getPositions=True, getEnergy=True)
```

### 2. Multi-Replica or Congeneric Ligand Batch (`ProlixBatchSimulation`)

```python
from openmm import unit
from prolix.openmm import ProlixBatchSimulation, ProlixOpenMMConfig

batch = ProlixBatchSimulation(
    systems=ligand_systems,           # Homogeneous, congeneric, or ragged OpenMM Systems
    integrators=integrator_template,  # Broadcast or per-replica OpenMM Integrators
    positions_list=initial_positions, # List of (N_i, 3) nm arrays or Quantities
    config=ProlixOpenMMConfig(use_pallas=True, nonbonded_engine="auto", precision="mixed"),
)
batch.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)
batch.step(5_000)
states = batch.getStates(getPositions=True, getEnergy=True)
```

---

## Architecture at a Glance

```{mermaid}
flowchart LR
    subgraph Frontends["User Frontends"]
        OMM["prolix.openmm<br/>Simulation / Context / ProlixBatchSimulation"]
        API["prolix.api<br/>EnsemblePlan & Observables"]
        PURE["prolix.physics<br/>make_energy_fn_pure & DifferentiableParams"]
    end

    subgraph Governance["Zero-Cycle Leaf Schema"]
        KNOB["KnobSurface & ProlixOpenMMConfig<br/>redsox U1 Reachability + U5 Superset"]
    end

    subgraph Backends["Execution & Export Backends"]
        PALLAS["Pallas v3.5 GPU Megakernels<br/>Morton 32x32 Tiles • SRAM SPME<br/>Fused BAOAB+SETTLE • 2-Pass OBC2 GBSA<br/>1D Ragged CTA Queue"]
        VJP["Analytical @jax.custom_vjp<br/>O(N·K) Coordinate & FF Gradients"]
        HLO["prolix.export<br/>Pure-JAX StableHLO & IREE wasm32"]
    end

    OMM --> KNOB
    API --> KNOB
    PURE --> KNOB
    KNOB --> PALLAS
    KNOB --> VJP
    KNOB --> HLO
```

---

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User Guides

guides/index
guides/cupotential_parity_and_openmm_fastpath
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
