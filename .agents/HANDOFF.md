# Prolix MD - Development Handoff

## Project Status

**Proxide Migration:** ✅ COMPLETE (Dec 2025)  
**OpenMM Physics Parity:** ✅ Achieved (Dec 2025)  
**Test Protein:** 1UAO (Trp-cage, 138 atoms)  
**Force Field:** ff19SB + GBSA/OBC2 or Explicit Solvent (TIP3P)

---

## Recent Achievements

- ✅ **Hydrogen Addition Resolved:** No longer hangs, geometry is validated.
- ✅ **Explicit Solvent Stable:** Simulations run correctly with SETTLE constraints and Langevin integrator.
- ✅ **Proxide Integration:** Fully transitioned from local `priox` to `proxide` submodule.

---

## Roadmap: Next Steps

1. **Standardize Simulation Tests:**
   - Create consistent fixtures for implicit/explicit systems.
   - Expand parity tests to include force vector comparison.

2. **Visualization:**
   - Document the `viewer/` module.
   - Standardize artifact storage in `outputs/`.

3. **CI/CD Integration:**
   - Build `oxidize` extension in GitHub Actions.
   - Run smoke tests on every push.

---

## Repository Structure

```
prolix/
├── src/prolix/
│   ├── physics/          # Energy functions
│   │   ├── system.py     # Main energy factory
│   │   ├── bonded.py     # Bond/Angle/Torsion
│   │   ├── settle.py     # SETTLE constraint logic
│   │   └── simulate.py   # Integrators, dynamics
│   ├── visualization/    # Trajectory viz
│   └── analysis.py       # RMSD, contacts, etc.
├── proxide/              # Git submodule (ground truth)
│   ├── src/proxide/      # Python package
│   └── oxidize/          # Rust extension
├── tests/physics/        # Physics and parity tests
├── scripts/              # Debug and utility scripts
├── data/                 # Test structures
└── outputs/              # Git-ignored debugging artifacts
```

---

## Running Tests

```bash
# Verify imports
uv run python -c "import proxide; import oxidize; print('OK')"

# Run physics tests
uv run pytest tests/physics/ -v
```
