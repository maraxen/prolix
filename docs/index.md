# Prolix Documentation Index

The canonical ReadTheDocs Sphinx + MyST documentation suite is located in [`docs/source/index.md`](source/index.md) and configured via [`.readthedocs.yaml`](../.readthedocs.yaml).

## Building HTML Docs

```bash
sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

## Quick Navigation

- **[Overview & Quickstart](source/index.md)**
- **[Installation & Precision Modes](source/installation.md)**
- **Tutorials**:
  - [1. Drop-In OpenMM Acceleration (`prolix.openmm`)](source/tutorials/openmm_acceleration.md)
  - [2. Multi-Replica & Ragged Ligand Screening (`ProlixBatchSimulation` & `EnsemblePlan`)](source/tutorials/batch_screening.md)
  - [3. Differentiable MD & Force-Field Optimization](source/tutorials/differentiable_md.md)
  - [4. Portable StableHLO & WebGPU/WASM Export](source/tutorials/stablehlo_export.md)
- **User Guides**:
  - [`KnobSurface` & `ProlixOpenMMConfig` Reference](source/guides/knob_governance.md)
  - [Pallas v3.5 Megakernel Architecture](source/guides/pallas_megakernel.md)
  - [Physics, Units, & Multi-Engine Validation](source/guides/physics_validation.md)
  - [Explicit-Solvent OpenMM Comparison Protocol](source/explicit_solvent/openmm_comparison_protocol.md)
- **[API Reference](source/api/index.md)**
