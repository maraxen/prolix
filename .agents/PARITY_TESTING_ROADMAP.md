# Prolix Parity Testing Roadmap

## Current Status: 13/13 Tests Passing

All current parity tests pass with `add_hydrogens=False`.

## Phase 1: Fix Hydrogen Addition (BLOCKED)

### Goal

Enable `add_hydrogens=True` in `parse_structure()` without hanging.

### Actions

1. Merge `origin/optimization/split_lib_rs` branch in proxide
2. Use `maturin develop --release` for builds (bypasses uv cache issue)
3. Verify fragment library loads correctly
4. Trace exact hang location

### Testing

```bash
uv run python -c "
from oxidize import parse_structure, OutputSpec, CoordFormat, HydrogenSource
spec = OutputSpec()
spec.add_hydrogens = True
spec.hydrogen_source = HydrogenSource.FragmentLibrary
result = parse_structure('data/pdb/1CRN.pdb', spec)
print('Success:', len(result['atom_mask']))
"
```

## Phase 2: Explicit Solvent Parity (IN PROGRESS)

### Goal

Full energy parity between Prolix and OpenMM for explicit solvent.

### Current Status

- Solvation logic implemented and verified ✅
- PME electrostatics implemented ✅
- **Blocker**: Simulation instability at step 0 (investigating)

### Tests to Add

- [ ] `test_coulomb_energy_parity` - Compare electrostatic energy
- [ ] `test_lj_energy_parity` - Compare Lennard-Jones energy
- [ ] `test_14_nonbonded_parity` - Compare 1-4 scaled interactions
- [ ] `test_total_explicit_energy_parity` - Full system comparison

### Files

- `tests/physics/test_explicit_parity.py`
- `scripts/simulate_explicit_rust.py` (instability reproduction)

## Phase 3: Implicit Solvent (GBSA) Parity

### Goal

GBSA energy parity between Prolix and OpenMM.

### Tests to Add

- [ ] `test_gbsa_born_radii_parity`
- [ ] `test_gbsa_polar_energy_parity`
- [ ] `test_gbsa_nonpolar_energy_parity`
- [ ] `test_gbsa_total_energy_parity`

### Files

- `tests/physics/test_implicit_parity.py` (new)

## Phase 4: End-to-End Simulation

### Goal

Run short MD simulations in both Prolix (JAX) and OpenMM, compare trajectories.

### Tests

- [ ] Energy conservation
- [ ] Forces comparison at each step
- [ ] RMS displacement correlation

## Key Files

| File | Description |
|------|-------------|
| `tests/physics/test_explicit_parity.py` | Explicit solvent tests |
| `proxide/oxidize/src/lib.rs` | Main Rust entry point |
| `proxide/oxidize/src/geometry/hydrogens.rs` | H addition |
| `proxide/oxidize/src/spec.rs` | OutputSpec, HydrogenSource |
