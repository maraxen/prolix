# PROMPT: Refactor Prolix for Proxide Integration

The user is migrating `prolix` (a JAX MD library) to use `proxide` (a Rust-accelerated I/O and parameterization library) as a git submodule.

## Current State

**Migration Progress:**

- `priox/` replaced with `proxide` submodule.
- `priox` imports replaced with `proxide`.
- `priox_rs` imports replaced with `oxidize`.
- `oxidize` Rust extension built and importable.
- `data/force_fields/*.eqx` DELETED (no longer used).

**Problem:**

1. Tests fail because `proxide.md.jax_md_bridge.parameterize_system()` has been **REMOVED**.
2. Proxide now handles MD parameterization directly in Rust via `parse_structure()`.
3. Prolix code still expects to load `.eqx` files and use `parameterize_system()`.

## Objective

Refactor `prolix` tests and scripts to use the new `proxide` API for loading and parameterizing systems.

**Old Pattern (DEPRECATED):**

```python
from proxide.physics.force_fields import load_force_field
from proxide.md import jax_md_bridge

ff = load_force_field("protein19SB.eqx")
params = jax_md_bridge.parameterize_system(ff, residues, atom_names)
```

**New Pattern (REQUIRED):**

```python
from proxide.io.parsing.rust import parse_structure, OutputSpec

spec = OutputSpec(
    parameterize_md=True,
    force_field="protein.ff19SB.xml",  # Loads from proxide/src/proxide/assets/
    # Optionally specify explicit water model:
    # water_model="tip3p" 
    # Optionally specify explicit solvent model:
    solvent_model="implicit" # validate this is true, you might need to add it to the spec and update proxide. it should have the obc params for the solvent model so that might be the best approach is just have water model be tip3p and solvent model be implicit if obc params are specified instead and default to the most recent version of implicit
)

# Returns atomic_system.AtomicSystem with .charges, .sigmas, .bonds, etc. populated
system = parse_structure("protein.pdb", spec) 
```

## Tasks

1. **Refactor Test Fixtures:**
    Update `tests/conftest.py` and other test setup files to provide parameterized `AtomicSystem` objects using `parse_structure` instead of `jax_md_bridge`.

2. **Fix Test Files:**
    The following files likely need updates:
    - `tests/physics/test_bridge.py`
    - `tests/physics/test_implicit_solvent_md.py`
    - `tests/physics/test_jax_md_energy.py`
    - `tests/physics/debug_forces.py`
    - `tests/physics/debug_topology.py`

    Replace `parameterize_system` calls with `parse_structure` calls.

3. **Update Scripts:**
    Update `scripts/` to stop looking for `.eqx` files and use the new API.

4. **Verify Physics:**
    Running `tests/physics/test_position_plausibility.py` is the best way to verify the new parameterization yields stable physics.

**Constraint:** `proxide` is the ground truth. If the API doesn't fit `prolix`, adapt `prolix` code, do not modify `proxide` unless absolutely necessary (it may be for the implicit/explicit solvent API).
