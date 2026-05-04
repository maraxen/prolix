# Workspace GEMINI.md: Prolix OODA v2 Orchestration

**Current Phase**: Sprint 9: Neighbor List Performance Tuning
**Strategic Goal**: Deliver production-grade MD throughput via JIT-optimized $O(N \cdot K)$ kernels and buffer-aware updates.

---

## Current Status (2026-05-03)

- **LJ Tail Corrections**: DONE. Prefactors corrected to $8\pi$. Achieved -0.14 kcal/mol parity with OpenMM for 1805-atom system.
- **Impulsive Pressure**: DONE. Integrated into `instantaneous_pressure_akma`. Verified via volume-derivative check.
- **Neighbor List Tracking**: DONE. Added `did_overflow` to `LangevinState` and integrated `did_buffer_overflow` checks into MD steps.
- **NL Performance**: DONE. Implemented nested-scanning $O(N \cdot K)$ kernels in `optimization.py` and buffer-aware conditional NL updates in simulation loop. Achieved high force parity ($10^{-7}$).

---

## Roadmap ahead

- **Sprint 9: Performance Tuning** (Done)
    - [x] Implement `lax.scan` for tiled reductions in `optimization.py` to reduce VRAM pressure.
    - [x] Optimize `tile_size` and capacity heuristics.
    - [x] JIT-optimize `NeighborList.update` frequency using `jax.lax.cond`.
- **Sprint 10: NPT Ensemble** (Done)
    - [x] Implement Stochastic Cell Rescaling barostat for pressure control.
    - [x] Integrate barostat with modular `StepSequence` architecture.
    - [x] Implement `SETTLE_Position_Step` for modular constraint handling.
- **Sprint 11: E2E Optimization Path (High-Res)**
    - [ ] **Gradient-of-Gradients Validation**: Verify Hessian-vector products and higher-order autodiff for TIP3P/CHARMM parameter sensitivity.
    - [ ] **Force-Field Refinement**: Implement differentiability for all bonded parameters (bonds, angles, dihedrals) to allow training against OpenMM force baselines.
    - [ ] **Differentiable PME**: Ensure `pme_alpha` and grid parameters are JAX-differentiable for self-consistent field tuning.
- **Sprint 12: Strategic Future (Post-Production)**
    - [ ] **Librarian Deep Dive**: Invoke `@librarian` to explore publications on JAX-MD, kUPS, OpenMM, and Amber to identify state-of-the-art differentiable MD methods.
    - [ ] **Oracle Direction**: Based on librarian report, `@oracle` to define next major architectural shift (e.g., neural force fields, multi-scale integration, or enhanced sampling).

---

## Strategic Priorities (Backlog)

| Priority | Feature | Description | Status |
| :--- | :--- | :--- | :--- |
| **P0** | **Tiled Neighbor List** | Implement $O(N \cdot K)$ tiled kernels over neighbor lists. | **Done (Logic)** |
| **P0** | **Equivalence Benchmarks**| Establish high-fidelity parity against OpenMM Reference. | **Done (0.1 kcal/mol)** |
| **P0** | **NL Performance** | Optimize kernels and update logic for GPU throughput. | **In Progress** |
| **P1** | **NPT Ensemble** | Stochastic Cell Rescaling and pressure virial corrections. | Pending |

---

## Engineering Standards

- **Units**: Internal AKMA. Conversion via `kups_adapter`.
- **Differentiability**: All kernels MUST support `jax.grad`.
- **Export**: All energy functions MUST be exportable via `prolix.export` (StableHLO).
- **Tiling**: Use `prolix.physics.tiling` primitives for all $O(N^2)$ and $O(N \cdot K)$ operations.
- **Overflow**: Always propagate `did_overflow` flags from dynamic neighbor lists.
