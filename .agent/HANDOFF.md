# Prolix MD - Development Handoff

## Project Status

**Proxide Migration:** 🔄 In Progress (Dec 2025)  
**OpenMM Physics Parity:** ✅ Achieved (Dec 2025)  
**Test Protein:** 1UAO (Trp-cage, 138 atoms)  
**Force Field:** ff19SB + GBSA/OBC2

---

## Migration Status

### Completed

- [x] Removed local `priox/` folder
- [x] Added `proxide` as git submodule from github.com/maraxen/proxide
- [x] Updated `pyproject.toml` dependencies (proxide, oxidize)
- [x] Updated `uv.sources` paths
- [x] Replaced all `from priox` → `from proxide` imports
- [x] Replaced all `priox_rs` → `oxidize` imports
- [x] Built oxidize Rust extension with maturin
- [x] Added `.pth` file for proxide Python package import
- [x] Installed missing dependency (hydride)

### Remaining

- [ ] **Refactor tests to use new proxide API** - `jax_md_bridge.parameterize_system` is removed
- [ ] Remove/update legacy `data/force_fields/*.eqx` references
- [ ] Update scripts that reference old force field paths
- [ ] Update `.agent/` documentation
- [ ] Commit migration changes

### Breaking Changes in Proxide

1. **`jax_md_bridge` module removed** - MD parameterization now happens via Rust
2. **No more `.eqx` files** - Force fields loaded from XML via `oxidize.load_forcefield()`
3. **`AtomicSystem` is now parameterized directly** - Use `parse_structure` with `spec.parameterize_md = True`

### New API for MD Parameterization

**OLD (removed):**

```python
from proxide.md import jax_md_bridge
from proxide.physics.force_fields import load_force_field

ff = load_force_field("protein19SB.eqx")
params = jax_md_bridge.parameterize_system(ff, residues, atom_names)
```

**NEW:**

```python
from proxide.io.parsing.rust import parse_structure, OutputSpec

spec = OutputSpec()
spec.parameterize_md = True
spec.force_field = "protein.ff19SB.xml"  # Looks in assets

protein = parse_structure("protein.pdb", spec)
# protein.charges, protein.sigmas, protein.epsilons are already set
# protein.bonds, protein.bond_params, etc. are already set
```

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
├── proxide/              # Git submodule (ground truth)
│   ├── src/proxide/
│   │   ├── assets/       # Force field XMLs
│   │   ├── core/         # AtomicSystem, Protein
│   │   ├── io/parsing/   # rust.py, dispatch.py
│   │   ├── md/bridge/    # types.py (SystemParams TypedDict)
│   │   └── physics/      # electrostatics, vdw
│   └── oxidize/          # Rust extension
├── tests/physics/
│   ├── test_position_plausibility.py
│   └── test_ensembles.py
├── benchmarks/
│   └── verify_end_to_end_physics.py
├── notebooks/            # Tutorials
└── data/                 # Test structures (legacy .eqx to be removed)
```

---

## Running Tests

```bash
# Basic import verification
uv run python -c "import proxide; import oxidize; print('OK')"

# Run tests (currently failing due to API changes)
uv run pytest tests/ -x --tb=short

# Position plausibility tests
uv run pytest tests/physics/test_position_plausibility.py -v
```

---

## Ground Truth Priority

**`proxide` is the authoritative source.**

If tests fail:

1. **First:** Modify prolix to match proxide's new API
2. **Exception:** If proxide has a documented inaccuracy, file an issue
3. **Document:** Any workarounds in `.agent/PROXIDE_DISCREPANCIES.md`

---

## Known Limitations

1. **N² non-bonded scaling** - neighbor lists available but O(N²) default
2. **GBSA approximation** - uses scaled radii, matches OpenMM ~0.0003 kcal/mol
3. **No explicit solvent** - implicit only (explicit in development)
4. **Force RMSE ~2.7 kcal/mol/Å** - acceptable for sampling
