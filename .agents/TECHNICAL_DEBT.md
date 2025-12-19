# Prolix Technical Debt

## Active Issues

### 1. Test Standardization

- **Status**: Planning
- **Goal**: Create reusable test fixtures and utilities for parity testing across different systems and conditions.

### 2. CI Pipeline Enhancement

- **Status**: Planning
- **Goal**: Integrate Rust builds for `oxidize` into GitHub Actions and add smoke/integration test splits.

## Completed Items ✅

### 1. Hydrogen Addition

- ✅ Resolved hanging issue.
- ✅ Validated geometry placement and energy relaxation.

### 2. Explicit Solvent Stability

- ✅ Stable simulations with SETTLE and Langevin integrator.
- ✅ Molecule-aware PBC wrapping implemented.

### 3. OpenMM Parity (Basic)

- ✅ Bonded energy parity (bonds, angles, dihedrals).
- ✅ Basic nonbonded validation.

## Deferred Items

### 1. Full Nonbonded Parity

- Need full Coulomb + LJ + 1-4 comparison vs OpenMM.

### 2. Implicit Solvent (GBSA) Parity

- Need comparison of Born radii and polar/nonpolar energies.
