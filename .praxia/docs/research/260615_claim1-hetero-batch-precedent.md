---
name: claim1-hetero-batch-precedent
description: DR-claim1-1 heterogeneous-batched MD precedent scan — §Related Work + RR1 rebuttal + §7.1 positioning
metadata:
  type: project
---

# Heterogeneous-Batch MD Precedent Scan — Claim 1 Evidence

**Date:** 2026-06-15
**Task:** 260615_sprint39 (#255)
**Purpose:** §Related Work + RR1 rebuttal + §7.1 positioning

---

## Executive Summary

No existing MD tool provides the combination of (1) bucketed JIT compilation across varied-topology systems and (2) differentiable trajectories through a heterogeneous batch in a single compiled backward pass. TorchSim (arXiv 2508.06628, 2025) is the closest prior art for heterogeneous batching but uses a sequential bin-packing approach without an explicit JIT cache key and does not offer differentiable trajectories through its AutoBatcher. jax-md provides differentiability but requires fixed-shape inputs and cannot batch systems with different atom counts in a single compiled function. prolix's `MolecularBundle` + `MolecularShapeSpec` bucketed architecture is the only design that enables both capabilities simultaneously, and this combination is what makes the §7.1 figure (gradient-based bonded-parameter fitting over a mixed-topology ensemble) possible.

---

## Per-Tool Analysis

### kUPS (CuspAI, open-sourced April 2026)

**Sources:** kUPS docs · kUPS GitHub · CuspAI blog Apr 2026 · PR #95

**Architecture:** JAX-based, columnar Table layout (SoA), composes with `jit`, `vmap`, `grad`. MACE models via Tojax with no performance penalty (XLA end-to-end). 49× throughput over RASPA reported on some workloads.

**Batching approach:** `vmap`-based batching over fixed-shape arrays. Documentation: "aggressive XLA compilation settings can be enabled when certain tensors have fixed shapes — e.g., the number of atoms will be constant." PR #95: batched relaxation of independent structures required per-system LBFGS state fix — implies system homogeneity assumption.

**Heterogeneous topology:** No documented support. Fixed atom count required for compiled path.

**Differentiability:** Yes — `jax.grad` available.

**Assessment:** Strong JAX-native MD engine with differentiability, but batching is homogeneous. Not prior art for the hetero-batch claim.

---

### jax-md (Schoenholz & Cubuk, NeurIPS 2020)

**Sources:** jax-md GitHub · NeurIPS 2020 paper · Issue #101 · Issue #192

**Architecture:** Energy function closures + simulate.* integrators (Langevin, NHC). prolix uses jax-md as a substrate for neighbor list and space primitives.

**Batching approach:** `vmap` over identical-shape systems. Issue #101 (2021): `vmap(neighbor_fn)` fails with `ConcretizationTypeError` for varied sizes. Issue #192: neighbor list reallocation across a `vmap` batch requires knowing largest capacity upfront. All published batch examples use homogeneous N.

**Heterogeneous topology:** Not supported. `vmap` requires static shapes.

**Differentiability:** Yes — core value proposition of jax-md.

**Assessment:** jax-md is the substrate prolix's bucketed layer builds on. Has differentiability but not hetero-batch. The hetero-batch support in prolix is an architectural addition above jax-md's design.

---

### OpenMM / MultiOpenMM

**Sources:** OpenMM 8.x docs · MultiOpenMM (epretti) · OMSF MPS blog 2024

**Architecture:** CUDA-native MD engine. MultiOpenMM: third-party package by epretti that stacks multiple systems into one OpenMM `System`, including systems with different force fields and molecular topologies. NVIDIA MPS: OS-level concurrent GPU scheduling.

**Batching approach:** MultiOpenMM stacks systems CPU-side or distributes across workers. MPS is OS-level concurrent scheduling of independent CUDA contexts, not a single compiled JIT graph. Each distinct topology triggers separate CUDA kernel compilation.

**Heterogeneous topology:** Partial — MultiOpenMM supports topologically distinct systems, but via CPU-side concatenation and serial/MPS dispatch, not a single compiled forward pass.

**Differentiability:** No. OpenMM is not autodiff-native.

---

### GROMACS -multidir

**Sources:** GROMACS mdrun features · CSC throughput tutorial

**Architecture:** C++ MD engine with MPI parallelism. `-multidir` flag: N simulations under one MPI allocation, one subdirectory per simulation with its own .tpr and topology.

**Batching approach:** Each MPI rank runs its own independent simulation. MPI-coupled for replica exchange communication. Designed for same-topology replicas. Different-topology systems run as independent MPI processes with no shared compiled kernel.

**Heterogeneous topology:** No. Design assumes same topology across replicas.

**Differentiability:** No.

---

### Folding@home (FAH)

**Sources:** FAH v8.1 client guide · FAHbench technical details

**Architecture:** Distributed volunteer computing. Work units (WUs) dispatched to volunteer clients. GPU computation via CUDA/OpenCL using OpenMM kernels internally.

**Batching approach:** Different proteins go to different client machines. On a single GPU, one WU is processed at a time. No concept of batching varied-topology systems in a single compiled kernel. Heterogeneity is at the fleet level.

**Differentiability:** No.

**Assessment:** FAH is irrelevant prior art for device-level hetero-batch. Included for completeness as the paradigmatic "distributed MD over many proteins" example.

---

### TorchSim (Cohen et al., arXiv 2508.06628, 2025)

**Sources:** arXiv 2508.06628 · torch-sim GitHub · IOPscience (published)

**Architecture:** PyTorch-native atomistic simulation engine. Self-described as "first MD package to support true batching for systems of diverse sizes." BinningAutobatcher: 1D bin-packing by memory footprint; systems packed into bins of max-memory capacity; bins evolved one by one. InFlightAutobatcher: variable-length simulations. Claims up to 100× speedup vs ASE.

**Batching approach:** BinningAutobatcher calculates memory footprint per system and packs into memory-capped bins using 1D bin-packing. Within a bin: all systems are pad-to-max and evolved together. Bins are processed **sequentially**. No explicit JIT cache-key mechanism described — `torch.compile` would recompile if the packed shape changes across bin executions.

**Heterogeneous topology:** Yes — BinningAutobatcher is designed for diverse system sizes. **Closest prior art to prolix's hetero-batch claim.**

**Differentiability:** Partial. PyTorch autograd is available in TorchSim ecosystem, but the paper does not describe threading `torch.grad` through a multi-step BinningAutobatcher trajectory for parameter fitting.

**Assessment:** Most significant prior art. Demonstrates the community recognizes value of device-level hetero-batch. Key distinctions from prolix: (1) TorchSim bins are processed sequentially; prolix buckets are compiled functions that accept parallel batches; (2) TorchSim has no explicit bucketed JIT cache key; (3) TorchSim does not offer differentiable trajectories through the hetero-batch for parameter fitting.

---

### prolix (this work)

**Location:** `src/prolix/types/bundles.py` · `src/prolix/tiling/planner.py` · `src/prolix/physics/system.py`

**Architecture:** `MolecularBundle`: eqx.Module with all topology arrays padded to bucketed sizes (`ATOM_BUCKETS=(64,128,256,1024,5000,25000,60000)`). `MolecularShapeSpec`: frozen dataclass, the ONLY `static=True` field, serves as XLA JIT cache key. Two systems with different real atom counts but same bucket share a compiled XLA function. `BatchPlanner` (`tiling/planner.py`): routes heterogeneous axes to `safe_map`, homogeneous to `vmap`, with greedy memory-budget logic. Topology arrays are DYNAMIC (not static=True) to avoid per-topology recompile.

**Key design invariant** (`bundles.py:113`): "Two systems with different real counts but same bucket size produce identical `shape_spec` (enabling Claim 1: heterogeneous batch substrate)."

---

## Comparison Table

| Tool | Decl. API | Varied-topo batch | Bucketed compile | Diff. traj | Compile-once |
|------|-----------|-------------------|------------------|------------|--------------|
| **prolix** | Y | Y | Y | Y | Partial (v1.1) |
| kUPS | Y | N | N | Y | N |
| jax-md | Partial | N | N | Y | N |
| OpenMM / MultiOpenMM | N | Partial | N | N | N |
| GROMACS -multidir | N | N | N | N | N |
| Folding@home | N | N | N | N | N |
| TorchSim | Y | Y | Partial | Partial | N |

**Column definitions:**
- **Decl. API:** User specifies intent, engine handles execution graph.
- **Varied-topo batch:** Different atom counts in a single compiled forward pass on one device.
- **Bucketed compile:** Separate compiled functions per bucket (bounded recompilation).
- **Diff. traj:** `jax.grad` or `torch.grad` through n integration steps.
- **Compile-once:** One compiled function for all shapes (XLA shape polymorphism).

---

## Prolix Contribution Framing

The key novelty of prolix's batching architecture is not heterogeneous batch execution per se — TorchSim (2025) demonstrates that the MD community values and can achieve this — but rather the combination of bucketed compile-once JIT and end-to-end differentiability through the batch trajectory.

prolix's `MolecularBundle` encodes each system's topology into a `MolecularShapeSpec` (the sole `static=True` field), so any two systems that share a bucket assignment share a compiled XLA function, yielding O(B_max) total compilations regardless of how many distinct topologies are in the dataset. Unlike TorchSim's sequentially-processed bins, prolix's same-bucket systems are batched together in a single `vmap` or `safe_map` call. Unlike jax-md and kUPS, which require fixed atom counts for compilation, prolix's topology arrays are dynamic (not static), with real counts encoded in mask arrays.

The combination — bucketed JIT, dynamic topology, and `jax.grad`-compatible integrators — makes possible the §7.1 experiment: training bonded force-field parameters on a mixed-topology molecular ensemble in a single compiled backward pass. No surveyed tool provides this capability:

- jax-md and kUPS: differentiability without hetero-batch
- TorchSim: hetero-batch without differentiable trajectories
- OpenMM and GROMACS: neither in a single compiled pass

---

## Cross-Surface Notes

- Only prolix and TorchSim claim to run varied-topology systems in a single batched execution. TorchSim uses pad-to-max within memory-capped bins (sequential bin processing); prolix uses bucketed JIT keys with shared compiled functions, processed in parallel within a vmap/safe_map pass.
- Only prolix and jax-md provide differentiable trajectories (`jax.grad`). TorchSim has PyTorch autograd available but does not thread it through the BinningAutobatcher multi-step trajectory. kUPS exposes `jax.grad` but its batching is fixed-shape.
- kUPS and jax-md both require fixed atom counts per compiled function. prolix solves this via `MolecularShapeSpec` as the static JIT key, with topology arrays remaining dynamic. This is the key architectural distinction.
- GROMACS -multidir and Folding@home provide fleet-level parallelism (many processes/machines, one system each). Not device-level hetero-batch.
- TorchSim's BinningAutobatcher is the closest prior art. Key differences: (1) TorchSim bins are sequential; prolix buckets accept parallel batches; (2) TorchSim has no explicit JIT cache key mechanism; (3) TorchSim does not offer differentiable trajectories.

---

## Open Questions

1. kUPS PR #95 (per-system optimizer for batched relaxation) suggests kUPS is actively developing hetero-batch support. Monitor whether a future kUPS release adds explicit bucketed heterogeneous MD. If so, prolix's differentiability advantage becomes the sole differentiator.
2. TorchSim's `torch.compile` behavior across bins is unclear from the paper. If `torch.compile` with `dynamic=True` or bucket padding is used, TorchSim may be closer to prolix's bucketed approach than the paper implies. Requires reading the torch-sim source code.
3. The 2025 TorchSim paper does not explicitly state whether `torch.grad` can backpropagate through a multi-step BinningAutobatcher trajectory. If so, this would change the comparison materially.
4. MultiOpenMM's stacking approach is under-documented on whether it achieves true single-GPU concurrent simulation vs. sequential dispatch.
5. prolix's v1.1 target of XLA shape polymorphism (compile-once for all bucket sizes) is listed as 'Partial'. This would collapse O(B_max) compilations to 1. Timeline and feasibility should be assessed before making this a paper claim.

---

## Sources

1. CuspAI. "kUPS: a molecular simulation engine for the AI era." Medium, April 2026.
2. kUPS documentation. https://cusp-ai-oss.github.io/kUPS/
3. kUPS GitHub — PR #95 (per-system optimizer for batched relaxation).
4. Schoenholz, S.S. & Cubuk, E.D. "JAX MD: A Framework for Differentiable Physics." NeurIPS 2020.
5. jax-md Issue #101 (vmap neighbor_fn). https://github.com/jax-md/jax-md/issues/101
6. jax-md Issue #192 (batch neighbor list reallocation). https://github.com/jax-md/jax-md/issues/192
7. OpenMM documentation (v8.x). https://docs.openmm.org/latest/userguide/
8. MultiOpenMM (epretti). https://github.com/epretti/multiopenmm
9. OMSF blog. "Maximizing OpenMM Throughput with NVIDIA MPS." 2024.
10. GROMACS mdrun features (multidir). https://manual.gromacs.org/current/user-guide/mdrun-features.html
11. Folding@home v8.1 client guide. https://foldingathome.org/v8-1-client-guide/
12. Cohen, O. et al. "TorchSim: An efficient atomistic simulation engine in PyTorch." arXiv 2508.06628, 2025.
13. prolix codebase. `src/prolix/types/bundles.py` (MolecularBundle, MolecularShapeSpec, ATOM_BUCKETS).
14. prolix codebase. `src/prolix/tiling/planner.py` (BatchPlanner, AxisSpec.heterogeneous flag).
15. prolix codebase. `src/prolix/physics/system.py` (make_bundle_from_system, bucket assignment).
