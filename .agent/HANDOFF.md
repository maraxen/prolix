# Prolix MD - Development Handoff

## Project Status

**OpenMM Physics Parity:** ✅ Achieved (Dec 2025)  
**Position Stability:** ✅ Fixed (terminal topology and minimization)  
**Test Protein:** 1UAO (Trp-cage, 138 atoms)  
**Force Field:** ff19SB + GBSA/OBC2

---

## Current Goals

### 1. OpenMM Equivalence & Plausibility Test Suite

> **Priority: High** | Look at [OpenMM's own test suite](https://github.com/openmm/openmm/tree/master/tests) for inspiration.

**Objective:** Build a comprehensive test suite comparing JAX MD to OpenMM across:

- [x] **Energy decomposition tests** - per-component energy comparison (bond, angle, torsion, CMAP) in `test_openmm_parity.py`
- [x] **Force comparison tests** - gradient accuracy against OpenMM forces
- [x] **Position plausibility tests** ✓ - already implemented in `tests/physics/test_position_plausibility.py`
- [x] **Trajectory stability tests** - long-time dynamics stability (minimization + NVT)
- [x] **Ensemble property tests** - NVE conservation, NVT temperature
- [x] **Conservation tests** - energy conservation in NVE

**Code coverage goals:**

- [x] Test all energy components with multiple proteins (parametrized for 1UAO, extensible)
- [x] Test all ensemble types (NVE, NVT)
- [ ] Test edge cases (single residue, capping groups, unusual bonds)

**Key files:**

- `tests/physics/test_position_plausibility.py` - existing position tests
- `benchmarks/verify_end_to_end_physics.py` - energy comparison benchmark
- *NEW:* `tests/physics/test_openmm_parity.py` - comprehensive OpenMM comparison

---

### 2. Visualization Suite Completion

> **Priority: Medium**

**Objective:** Complete the visualization tools for trajectory analysis.

- [x] `prolix.visualization.animate_trajectory()` - GIF/video generation
- [x] `prolix.visualization.TrajectoryReader` - efficient trajectory loading
- [x] RMSD plotting over trajectory
- [x] Contact map visualization
- [x] Ramachandran plots
- [x] Energy vs. time plots
- [x] structure viewer integration (py3Dmol) with support for both browser and Jupyter/Colab

**Key files:**

- `src/prolix/visualization/__init__.py`
- `src/prolix/analysis.py` - trajectory analysis functions

---

### 3. Explicit Solvent with Pre-Equilibrated Water Boxes

> **Priority: High** | Study [OpenMM's modeller](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/modeller.py) for reference.

**Objective:** Support explicit solvation using pre-equilibrated water boxes (TIP3P, TIP4P, SPC/E).
Keep this in the JAX ecosystem and independent of
outside dependencies as much as possible.

- [x] **Water box initialization** - load pre-equilibrated boxes from GROMACS/AMBER format
- [x] **Solvation workflow** - `Modeller.addSolvent()` equivalent
- [x] **Ion placement** - neutralization and ionic strength control
- [x] **PME electrostatics** - already partially implemented in `src/prolix/physics/pme.py`
- [ ] **Long-range corrections** - LJ tail corrections

**OpenMM reference files:**

- `openmm/app/modeller.py` - water box placement
- `openmm/app/forcefield.py` - water templates
- Pre-equilibrated box sources: `openmm/app/data/`

**Key prolix files:**

- `src/prolix/physics/pme.py` - PME implementation
- `src/prolix/physics/system.py` - energy function with PBC
- *NEW:* `src/prolix/solvation.py` - water box utilities

---

### 4. Demo Jupyter/Colab Notebook ✅ Completed (Dec 2025)

> **Priority: Medium**

**Objective:** Create a polished notebook demonstrating all supported features.

**Contents:**

1. **Setup** - installation, imports
2. **Loading structures** - PDB parsing, force field loading
3. **Energy minimization** - robust multi-stage approach
4. **MD simulation** - NVT Langevin, trajectory saving
5. **Analysis** - RMSD, contacts, energy visualization
6. **Advanced features** - implicit solvent, constraints, multiple proteins, parallel tempering

**Target location:** `notebooks/prolix_tutorial.ipynb`

---

### 5. Force Field Support Certification

> **Priority: High**

**Objective:** Determine which force fields are fully supported through stress testing.

| Force Field | Status | Notes |
|-------------|--------|-------|
| ff19SB | ✅ Supported | Primary development target |
| ff14SB | ⚠️ To test | Amber family |
| ff99SB | ⚠️ To test | Legacy Amber |
| CHARMM36 | ⚠️ To test | May need Urey-Bradley |
| GAFF/GAFF2 | ⚠️ Partial | Ligand support in progress |
| SMIRNOFF | ❌ Not started | Requires OpenFF integration |

**Certification criteria:**

- [ ] All energy components match OpenMM (< 0.1 kcal/mol difference)
- [ ] Minimization converges without position explosion
- [ ] 100 ps NVT simulation remains stable
- [ ] Forces match OpenMM (RMSE < 5 kcal/mol/Å)

---

## Repository Structure

```
prolix/
├── src/prolix/
│   ├── physics/          # Energy functions
│   │   ├── system.py     # Main energy factory
│   │   ├── bonded.py     # Bond/Angle/Torsion
│   │   ├── cmap.py       # CMAP with periodic splines
│   │   ├── generalized_born.py  # GBSA/OBC2
│   │   ├── pme.py        # Particle Mesh Ewald
│   │   └── simulate.py   # Minimization, dynamics
│   ├── visualization/    # Trajectory viz
│   ├── analysis.py       # RMSD, contacts, etc.
│   └── simulate.py       # SimulationSpec, trajectory writing
├── priox/src/priox/
│   ├── md/bridge/        # PDB+FF → SystemParams
│   │   ├── core.py       # parameterize_system
│   │   └── types.py      # SystemParams TypedDict
│   └── physics/force_fields/  # .eqx loader
├── tests/physics/
│   ├── test_position_plausibility.py  # NEW
│   └── test_ensembles.py
├── benchmarks/
│   └── verify_end_to_end_physics.py
├── notebooks/            # Tutorials
└── data/
    ├── pdb/              # Test structures
    └── force_fields/     # Pre-converted .eqx
```

---

## Running Tests

```bash
# Position plausibility tests (includes OpenMM comparison)
uv run pytest tests/physics/test_position_plausibility.py -v

# Full physics benchmark
uv run python benchmarks/verify_end_to_end_physics.py

# Run simulation demo
uv run python scripts/simulate_chignolin_gif.py
```

---

## Recent Fixes (Dec 2025)

1. **Terminal topology fix** - H bonds now properly inferred from naming conventions (bonds: 73→136)
2. **Robust minimization** - steepest descent pre-conditioning prevents position explosion
3. **rattle_langevin fix** - manual momenta initialization for shape compatibility

---

## Known Limitations

1. **N² non-bonded scaling** - neighbor lists available but O(N²) default
2. **GBSA approximation** - uses scaled radii, matches OpenMM ~0.0003 kcal/mol
3. **No explicit solvent** - implicit only (explicit in development)
4. **Force RMSE ~2.7 kcal/mol/Å** - acceptable for sampling
