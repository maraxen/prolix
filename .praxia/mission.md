# Prolix — Project Mission

**Prolix** is a JAX-based molecular dynamics engine for protein folding and dynamics.

## Core Thesis

Prolix enables **heterogeneous batching** of molecular systems with different sizes and topologies in a single GPU kernel — without sacrificing differentiability or portability. The engine thesis is:

> A single XLA compilation boundary can execute a batch of structurally heterogeneous protein systems, using bucketed padding and a static JIT cache key (`MolecularShapeSpec`) to amortize compilation cost across diverse topologies.

This is **Claim 1 (RR1)**: compile-level sharing across heterogeneous topologies, distinct from per-system JIT (OpenMM, GROMACS) and shape-homogeneous batching (TorchSim, NequIP).

## Engine Paper Scope (v1.0)

Three claims, two supporting claims:

| Label | Claim |
|-------|-------|
| RR1 | Heterogeneous topology batching via `MolecularShapeSpec` JIT key |
| RR2 | Differentiable trajectories (end-to-end gradients through MD steps) |
| RR7 | Portability: CPU/GPU/TPU via XLA, no CUDA lock-in |
| RR4 | Memory efficiency: O(bucket) not O(max_size) padding overhead |
| RR5 | Throughput parity vs. OpenMM on DHFR benchmark |

## Two-Lane Architecture

- **Lane A (engine paper)**: Claim validation (V-series gates), benchmarks (B-series), §7.1 figure
- **Lane B (science substrate)**: HP4 ANI-1x curation → differentiable bonded parameter fitting figure

## Current Focus (as of 2026-06-15)

Sprint 41 pending approval. Immediate paper-critical path:
1. **#283** — `EnsemblePlan.from_bundle()` + `EnsemblePlan.from_bundles()` API (unblocks V1–V7)
2. **#260** — HP4 ANI-1x curation (cluster-scale; requires dedicated session)
3. **#259** — §7.1 differentiable bonded-parameter fitting figure (paper gate)

## Key Invariants

- `MolecularShapeSpec` is the **sole** `eqx.field(static=True)` in `MolecularBundle` — it stores bucket indices as Python ints and serves as the XLA JIT cache key. Never add a second static field without understanding this invariant.
- Timestep cap: dt ≤ 1.0 fs at production scale (n ≳ 16, γ ≈ 10 ps⁻¹). dt ≤ 0.5 fs for n ≲ 16 or weak friction.
- NPT long-trajectory (> ~10 ps) still diverges — use NVT for production runs.
- `NPTState` lives in `prolix.typing`, not `prolix.physics.simulate`.
