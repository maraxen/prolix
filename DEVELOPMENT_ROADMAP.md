# Prolix & Proxide Development Roadmap

This document outlines the next stage of development after achieving stable explicit solvent simulations.

---

## Phase 1: Test Standardization

**Goal:** Create consistent, reusable test infrastructure across repositories.

- **Standard Protein Set:** Use 1UAO (small), 1CRN (medium), and chignolin (miniprotein) as standard benchmarks.
- **Tolerance Thresholds:**
  - Energy atol: 0.01 kcal/mol
  - Force RMSE: 3.0 kcal/mol/Å
  - Position rtol: 1e-4
- **Shared Fixtures:** Consolidate fixtures in `tests/physics/conftest.py`.

## Phase 2: Visualization Improvements

**Goal:** Enhance the premium visualization experience.

- **Viewer Documentation:** Complete documentation for the `viewer/` module.
- **GIF Generation:** Standardize `scripts/generate_simulation_movie.py` for easy conversion of trajectories to high-quality GIFs.
- **Micro-animations:** Add subtle micro-animations to browser-based visualizations.

## Phase 3: Coverage & Parity Expansion

**Goal:** Deepen validation and support for more simulation conditions.

- **Full Nonbonded Parity:** Implement detailed Coulomb and LJ comparison tests against OpenMM.
- **GBSA Validation:** Expand parity tests to include Born radii and polar/nonpolar energy components.
- **Forcefield Support:** Validate `ff14SB` and alternative water models like `SPC/E`.
- **Constraint Geometry:** Add explicit verification of SETTLE and SHAKE geometry preservation.

## Phase 4: CI/CD & Maintenance

**Goal:** Improve reliability and developer experience.

- **Integrated CI:** Build the `oxidize` Rust extension within Prolix's GitHub Actions.
- **Test Tiers:** Separate "Smoke Tests" (fast) from "Integration Tests" (full dynamics).
- **Maintenance:** Standardize artifact storage in the `outputs/` directory (git-ignored) for all agents.

---

## Implementation Protocol

- **Ground Truth:** `proxide` remains the authoritative source for physics and parsing logic.
- **Verification:** All new physics features must include an equivalence test against OpenMM.
