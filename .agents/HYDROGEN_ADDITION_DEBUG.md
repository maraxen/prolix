# Hydrogen Addition Debugging - Current Status

## Summary

Hydrogen addition via `add_hydrogens=True` in `parse_structure()` is **hanging** when called from Python. All other functionality (MD parameterization, bonded/nonbonded energy parity tests) works correctly.

## Branch Information

- **Current branch**: `main` (prolix) / `main` (proxide submodule)
- **lib.rs split branch**: ✅ **MERGED** `origin/optimization/split_lib_rs`
  - lib.rs now split into modular files: `py_chemistry.rs`, `py_forcefield.rs`, `py_hdf5.rs`, `py_parsers.rs`, `py_trajectory.rs`
  - Hydrogen addition work preserved via stash resolution

## What Works ✅

1. **9/13 parity tests pass** (`tests/physics/test_explicit_parity.py`)
   - Position plausibility tests
   - Bonded energy tests (bond, angle, dihedral)
   - Nonbonded energy validation
   - OpenMM parity tests (4 skipped - OpenMM not installed)

2. **HydrogenSource enum** - Implemented and exported
   - `ForceFieldFirst` (default)
   - `FragmentLibrary`
   - `ForceFieldOnly`

3. **MD parameterization** - Works correctly with `add_hydrogens=False`

4. **Fragment library** - Rust tests pass (`cargo test fragment --release`)

5. **Maturin build** - Successfully builds (`maturin develop --release`)

## What's Broken ❌

### Hydrogen Addition Hangs

When `spec.add_hydrogens = True`, `parse_structure()` hangs indefinitely.

**Root Cause Analysis** (UPDATED):

Previous hypothesis about uv cache overwriting wheels was **INCORRECT**.

Debug log confirms Rust code IS being called:

```
[parse_structure] path=data/pdb/1CRN.pdb, add_hydrogens=true
[parse_structure] Inside with_gil
```

The hang occurs **INSIDE** the Rust code after entering `with_gil`. Likely locations:

1. Fragment library initialization (`init_fragment_library()`)
2. Bond inference step (`geometry::topology::infer_bonds`)
3. Hydrogen placement algorithm (`geometry::hydrogens::add_hydrogens`)

**The Original Hypothesis Was Wrong**: The editable install issue was NOT causing the hang. The Rust code executes but blocks somewhere inside the hydrogen addition logic.

### Fix Required

1. Add more `eprintln!` statements to narrow exact hang location
2. Check if the `OnceCell<FragmentLibrary>` initialization deadlocks
3. Test fragment library loading in isolation
4. Consider if nested `Python::with_gil` calls cause issues

## Attempted Fixes

1. **Rayon parallelization removed** - Changed `into_par_iter()` to `into_iter()` to avoid GIL deadlock
2. **`lazy_static` replaced with `once_cell::OnceCell`** - Eager initialization to avoid GIL issues
3. **Cell list optimization** - Replaced O(n²) neighbor search with O(n) cell list algorithm
4. **Debug logging** - Added `eprintln!` and file-based logging (see `/tmp/oxidize_debug.log`)

## Files Modified

### proxide/oxidize/src/

| File | Changes |
|------|---------|
| `lib.rs` | Added debug logging, HydrogenSource dispatch |
| `spec.rs` | Added `HydrogenSource` enum |
| `geometry/hydrogens.rs` | Replaced lazy_static with OnceCell, removed Rayon |
| `geometry/topology.rs` | Switched to fast cell list neighbor search |

### tests/physics/test_explicit_parity.py

- Added `_extract_full_params()` for complete FF parameter extraction
- Added `test_bond_energy_parity`, `test_angle_energy_parity`, `test_dihedral_energy_parity`
- Uses `CoordFormat.Full` and `add_hydrogens=False` workaround

## Next Steps

1. **Merge with split branch**:

   ```bash
   cd proxide
   git fetch origin
   git merge origin/optimization/split_lib_rs
   ```

2. **Fix the build/install issue**:
   - Use `maturin develop --release` instead of `uv pip install`
   - Or clean uv cache: `rm -rf ~/.cache/uv/builds-v0/oxidize*`

3. **Debug hydrogen addition**:
   - Ensure fresh wheel is actually loaded
   - Trace where exactly the hang occurs
   - Check if fragment library binary data loads correctly

4. **Achieve end-to-end parity**:
   - Implicit solvent (GBSA) parity tests
   - Explicit solvent parity tests with hydrogen addition
   - Full energy comparison (Prolix vs OpenMM)

## Test Commands

```bash
# Run parity tests (currently all pass)
cd /home/marielle/workspace/prolix
JAX_PLATFORMS=cpu uv run pytest tests/physics/test_explicit_parity.py -v

# Build oxidize with maturin
cd proxide/oxidize
maturin develop --release

# Verify oxidize version (check timestamp)
ls -la .venv/lib/python3.13/site-packages/oxidize/oxidize*.so
```
