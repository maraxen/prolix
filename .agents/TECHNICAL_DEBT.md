# Prolix/Proxide Technical Debt

## Active Issues

### 1. Hydrogen Addition Hangs (HIGH PRIORITY)

- **Status**: Blocked
- **File**: `proxide/oxidize/src/lib.rs`, `geometry/hydrogens.rs`
- **Issue**: `parse_structure()` with `add_hydrogens=True` hangs indefinitely
- **Root Cause**: `uv run` overwrites manually installed wheels with cached build from editable path
- **See**: `.agents/HYDROGEN_ADDITION_DEBUG.md`

### 2. lib.rs Needs Refactoring

- **Status**: Partially done on `origin/optimization/split_lib_rs` branch
- **Issue**: `lib.rs` is 1700+ lines, should be split into modules

### 3. Explicit Solvent Simulation Instability (HIGH PRIORITY)

- **Status**: Active
- **File**: `scripts/simulate_explicit_rust.py`, `src/prolix/physics/system.py`
- **Issue**: MD simulation crashes at step 0 with NaN/Inf positions for explicit solvent
- **Possible Causes**:
  - Missing bond constraints (SETTLE for water, SHAKE for H-bonds)
  - PME parameters or grid size mismatch
  - Float32 precision limitations on GPU
  - Integration timesteps (need 0.5-1.0fs or constraints)

## Completed Items ✅

### Explicit Solvation Logic

- Water box tiling fixed (symmetric tiling, proper bounds check)
- Solvation integration in simulation pipeline
- PME/PBC support added to energy function (with neighbor lists pending)

### Hydrogen Source Enum

- `HydrogenSource` enum implemented and exported
- Variants: `ForceFieldFirst`, `FragmentLibrary`, `ForceFieldOnly`
- Currently all variants use FragmentLibrary (FF coords TBD)

### OpenMM Parity Tests

- Bond energy parity ✅
- Angle energy parity ✅  
- Dihedral energy parity ✅
- Nonbonded validation ✅

### Cell List Optimization

- Replaced O(n²) neighbor search with O(n) cell list algorithm
- File: `geometry/cell_list.rs`, `geometry/topology.rs`

### Fragment Library

- Lazy_static replaced with `once_cell::OnceCell`
- Eager initialization in pymodule init
- Rayon parallelization removed (GIL issues)

## Deferred Items

### 1. Force-Field Based Hydrogen Coordinates

- Currently all `HydrogenSource` variants use geometric placement
- True FF-based coordinate generation not implemented

### 2. Implicit Solvent (GBSA) Parity Tests

- Need to add GBSA energy comparison tests

### 3. Full Nonbonded Parity

- Current test only validates finiteness
- Need full Coulomb + LJ + 1-4 comparison

## Environment Notes

```bash
# Build oxidize correctly (bypasses uv cache)
cd proxide/oxidize
maturin develop --release

# OR clear uv cache
rm -rf ~/.cache/uv/builds-v0/oxidize*
uv sync
```
