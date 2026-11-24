# Prolix

**Prolix** is a GPU-native molecular dynamics and differentiable protein physics engine built on **JAX** and **Pallas/Triton**. It pairs a drop-in **OpenMM** execution backend (`prolix.openmm`) with high-throughput multi-replica and ragged congeneric batch kernels (`ProlixBatchSimulation`, `EnsemblePlan`), $O(N \cdot K)$ analytical `@jax.custom_vjp` coordinate and force-field gradients, and portable **StableHLO / WebAssembly** export (`prolix.export`).

---

## Key Capabilities

- **Drop-In OpenMM Acceleration (`prolix.openmm`)**: Run unmodified OpenMM `System`, `Topology`, and `Integrator` workflows via `ProlixPlatform`, `ProlixContext`, and `ProlixSimulation` — supporting explicit-solvent SPME, OBC2 GBSA + ACE SASA, analytical water `SETTLE` + solute `SHAKE/RATTLE`, virtual sites, Lepton custom forces, GPU L-BFGS/FIRE minimization, and Monte Carlo / SCR NPT barostats.
- **Pallas v3.5 GPU Megakernels**:
  - **Morton $32 \times 32$ Newton-Pair Tiles (`MK-51`, `MK-65`, `MK-71`)**: 30-bit Z-order spatial sorting with $O(N_{\text{tiles}})$ GPU AABB skin-rebuild scaling past 408k atoms.
  - **Tile-Local SRAM SPME (`MK-54`, `MK-70`)**: Shared-memory B-spline charge spreading + smooth $2^a 3^b 5^c$ batched 3D FFTs + fused gather.
  - **Persistent Inter-Step Fused Integrator (`MK-46`, `MK-58`, `MK-66`)**: Multi-step `BAOAB` + `SETTLE` + `RATTLE` + `Philox-2x32` noise fused across `nl_update_every` chunks in `single`, `mixed` (`float64` state/constraints + `float32` force/PME kernels), and `double` precision.
  - **Two-Pass Tile OBC2 GBSA (`MK-67`)**: Symmetric $32 \times 32$ tile Generalized Born + ACE nonpolar SASA with Debye-Hückel salt screening and analytical Born-radius chain-rule forces.
- **Multi-Replica, Congeneric & Ragged Screening (`ProlixBatchSimulation`, `EnsemblePlan`)**:
  - **`stride = 0` Congeneric Deduplication (`MK-64`)**: Shares receptor topology/exclusion tables across ligand or mutant panels at **~45–55 MB/replica** marginal VRAM (vs **~450–580 MB/replica** under `CUDA MPS`, scaling past `R = 128` on 48 GB GPUs).
  - **1D Prefix-Sum Ragged CTA Queue (`MK-68`)**: Executes heterogeneous systems spanning `2k` to `50k+` atoms in a single flat kernel launch with cost-modeled multi-tier SPME.
- **Analytical Differentiability (`@jax.custom_vjp`)**: Exact $O(N \cdot K)$ gradients with respect to coordinates, partial charges, Lennard-Jones $(\sigma, \varepsilon)$, torsions (`make_energy_fn_pure`, `DifferentiableParams`), and OpenMM global context parameters (`getEnergyParameterDerivatives`).
- **Portable StableHLO & WebAssembly (`prolix.export`)**: Pure-JAX lowering inside `export_context()` with zero GPU-locked custom calls, compiling under 50 MiB to `wasm32` via IREE (`compile_lowered_to_wasm`).
- **Static Knob Governance (`KnobSurface` & `redsox`)**: Zero-cycle 9-domain configuration schema verified by U5 superset (`check_superset`) and U1 consumer AST reachability (`check_prolix_reachability`) gates.

---

## Installation

Prolix requires **Python 3.13** (`>=3.13,<3.14`).

```bash
# NVIDIA GPU (CUDA 12 + OpenMM drop-in backend)
uv pip install "prolix[cuda,openmm]"

# CPU-only development & OpenMM parity
uv pip install "prolix[cpu,openmm]"

# StableHLO -> WebAssembly (iree-compile)
uv pip install "prolix[cpu,wasm]"

# ReadTheDocs / Sphinx documentation suite
uv pip install "prolix[docs,openmm]"
```

### Editable Source & Sibling `proxide` Workspace

```bash
git clone https://github.com/maraxen/prolix.git
cd prolix
uv sync --extra cuda --extra openmm --group dev
```

When developing `prolix` alongside a sibling `../proxide` Rust/Python checkout:

```bash
bash scripts/sync_workspace_lock.sh
cd .. && uv sync --extra cuda --extra openmm --group dev --package prolix
```

---

## Quickstart

### 1. Drop-In Single-System OpenMM Simulation

```python
import jax
jax.config.update("jax_enable_x64", True)  # Required for precision="mixed" or "double"

import openmm
from openmm import app, unit
from prolix.openmm import ProlixOpenMMConfig, ProlixSimulation

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

sim = ProlixSimulation(
    pdb.topology,
    system,
    integrator,
    config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="auto", use_pallas=True, use_pallas_pme=True),
)
sim.context.setPositions(pdb.positions)
sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
sim.step(10_000)
state = sim.context.getState(getPositions=True, getEnergy=True)
```

### 2. Multi-Replica & Congeneric Ligand Screening (`ProlixBatchSimulation`)

```python
from openmm import unit
from prolix.openmm import ProlixBatchSimulation, ProlixOpenMMConfig

batch = ProlixBatchSimulation(
    systems=ligand_systems,           # Homogeneous, congeneric, or ragged OpenMM Systems
    integrators=integrator,           # Broadcast or per-replica OpenMM Integrators
    positions_list=initial_positions, # List of (N_i, 3) nm arrays or Quantities
    config=ProlixOpenMMConfig(precision="mixed", nonbonded_engine="auto", use_pallas=True, use_pallas_pme=True),
)
batch.setVelocitiesToTemperature(300.0 * unit.kelvin, randomSeed=42)
batch.step(25_000)
states = batch.getStates(getPositions=True, getForces=True, getEnergy=True)
```

### 3. Differentiable Force Fields & Portable StableHLO Export

```python
import jax
from jax_md import space
from prolix.export import export_energy_fn, save_artifact
from prolix.physics import make_energy_fn_pure

displacement_fn, _ = space.periodic(physics_system.box_size)
params, energy_fn = make_energy_fn_pure(displacement_fn, physics_system, cutoff_distance=9.0)

# Simultaneous O(N·K) analytical gradients wrt force-field parameters & coordinates
energy_kcal, (grad_params, grad_pos) = jax.value_and_grad(energy_fn, argnums=(0, 1))(params, positions)

# Pure-JAX StableHLO MLIR lowering (zero Triton/CUDA custom calls)
lowered = export_energy_fn(energy_fn, params, positions)
save_artifact(lowered, "artifacts/energy.mlir")
```

---

### 4. Agent CLI and MCP server

Install the extra (`uv sync --extra agent` or `pip install "prolix[agent]"`), then:

```bash
prolix load_system --pdb protein.pdb --out-dir runs/a --json     # parameterise and validate (CPU)
prolix minimize --pdb protein.pdb --out-dir runs/a --sd-steps 50
prolix render_job --script scripts/benchmarks/prolix_vs_openmm_speed.py --name speed1 --preset gpu_l40s
prolix submit_job --name speed1                                  # dry run; confirm needs PROLIX_AGENT_ALLOW_SUBMIT=1
prolix-mcp                                                       # the same tools over stdio MCP
```

Quick operations run in-process on CPU and write `<tool>.json` run records under `--out-dir`. Long
runs are rendered as Slurm scripts under `scripts/slurm/agent_jobs/` and submitted with myxcel.
Exit codes: 0 success, 1 extra missing, 2 usage or jax cache configuration error, 3 refused, 4 input error.

## Documentation

Full Sphinx + MyST documentation (ReadTheDocs configuration in `.readthedocs.yaml`) lives under [`docs/source/`](docs/source/index.md):

- **[Installation & Precision Modes](docs/source/installation.md)**
- **Tutorials**:
  1. [Drop-In OpenMM Acceleration (`prolix.openmm`)](docs/source/tutorials/openmm_acceleration.md)
  2. [Multi-Replica & Ragged Ligand Screening (`ProlixBatchSimulation` & `EnsemblePlan`)](docs/source/tutorials/batch_screening.md)
  3. [Differentiable MD & Force-Field Optimization (`jax.grad` / `@jax.custom_vjp`)](docs/source/tutorials/differentiable_md.md)
  4. [Portable StableHLO & WebGPU/WASM Export (`prolix.export`)](docs/source/tutorials/stablehlo_export.md)
- **User Guides**:
  - [`KnobSurface` & `ProlixOpenMMConfig` Governance Reference](docs/source/guides/knob_governance.md)
  - [Pallas v3.5 Megakernel Architecture](docs/source/guides/pallas_megakernel.md)
  - [Physics, Units, & Multi-Engine Validation Summary](docs/source/guides/physics_validation.md)
- **[API Reference (`prolix.api`, `prolix.openmm`, `prolix.knob_surface`, `prolix.export`, `prolix.physics`)](docs/source/api/index.md)**

Build the HTML documentation locally with strict warning-as-error checking:

```bash
sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

---

## Development & Verification Gates

```bash
python3 scripts/ci/import_graph.py               # AST zero-cycle & leaf-module gate
python3 scripts/export_regression_pme.py --check # Explicit-PME protocol sentinel check
uv run ruff check src/                           # Linter
uv run ty check                                  # Static type checker
uv run pytest -m "not slow"                      # Fast CI & parity suite
```

## License

MIT License — see [LICENSE](LICENSE) for details.
