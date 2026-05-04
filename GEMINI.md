# Workspace GEMINI.md: Prolix OODA v2 Orchestration

**Current Phase**: Epic: v0.1.0-alpha Release Preparation
**Strategic Goal**: Transition from verified features to a production-ready, trustworthy toolkit for MD and ML communities.

---

## Current Status (2026-05-03)

- **Sprint 9: Neighbor List Performance**: DONE. $O(N \cdot K)$ kernels + buffer-aware updates.
- **Sprint 10: NPT Ensemble**: DONE. Stochastic Cell Rescaling barostat + SETTLE integration.
- **Sprint 11: E2E Optimization Path**: DONE.
    - **Differentiable Infrastructure**: Shipped `DifferentiableParams` Pytree and pure energy factories.
    - **Analytical VJPs**: Shipped manual gradients for LJ and PME with <1e-6 parity vs autodiff.
    - **StableHLO Export**: Verified end-to-end export path for structural refinement.

---

## Epic: v0.1.0-alpha Release Preparation

### Phase 1: Architectural Hardening
*Strategic Intent*: Eliminate circular dependencies and lazy-import smells to stabilize the package.

| Task | Description | Agent Needs | Status |
| :--- | :--- | :--- | :--- |
| **1.1** | **Core Type Extraction** | `recon` (map), `fixer` (extract) | Pending |
| **1.2** | **Resolve Circularities** | `fixer` (refactor) | Pending |
| **1.3** | **StableHLO Guardrails** | `reviewer` (CI tests) | Pending |

### Phase 2: Validation Storytelling
*Strategic Intent*: Establish community trust via rigorous, reproducible evidence.

| Task | Description | Agent Needs | Status |
| :--- | :--- | :--- | :--- |
| **2.1** | **Validation Whitepaper** | `librarian` (ref), `summarize` | Pending |
| **2.2** | **User Tutorials** | `designer` (UX), `fixer` | Pending |
| **2.3** | **API Documentation** | `auditor` (docs) | Pending |

### Phase 3: "Fair Comparison" Benchmarking
*Strategic Intent*: Demonstrate the $O(N)$ memory moat and throughput parity.

| Task | Description | Agent Needs | Status |
| :--- | :--- | :--- | :--- |
| **3.1** | **Apples-to-Apples Harness**| `staff` (coord), `librarian` | Pending |
| **3.2** | **Scaling Proofs** | `generalist` (batch run) | Pending |

---

## Strategic Priorities (Backlog)

| Priority | Feature | Description | Status |
| :--- | :--- | :--- | :--- |
| **P0** | **Architectural Hardening** | Clean up `prolix.physics` package structure. | **Next** |
| **P0** | **Release Validation** | Formalize the "Three-Tier" parity report. | Pending |
| **P1** | **kUPS Benchmark** | High-precision comparison vs kUPS. | Pending |
| **P2** | **NL Differentiability** | Analytical VJP for neighbor list updates. | Debt |

---

## Engineering Standards

- **Units**: Internal AKMA. Conversion via `kups_adapter`.
- **Differentiability**: All kernels MUST support `jax.grad`.
- **Export**: All energy functions MUST be exportable via `prolix.export` (StableHLO).
- **Tiling**: Use `prolix.physics.tiling` primitives for all $O(N^2)$ and $O(N \cdot K)$ operations.
- **Imports**: Avoid lazy-loading; use absolute imports and a flat `typing.py` for shared state.
