# Prolix

**Prolix** is a specialized library for Protein Physics and molecular dynamics simulations in JAX. It provides efficient implementations of force fields, energy calculations, and MD integrations for the JAX ecosystem.

**NOTE**: This is a work-in-progress library and is not yet ready for production use. It is currently in active development and subject to change.

## Features

- **JAX MD Integration**: Seamless bridging between protein force fields and JAX MD simulations
- **Implicit Solvent**: Generalized Born (GBSA) implementation with OBC2 model
- **Force Fields**: Support for AMBER force fields (ff14SB, ff19SB) with CMAP corrections
- **Energy Calculations**: Bond, angle, torsion, and non-bonded energy terms
- **MD Simulations**: Stable molecular dynamics with customizable thermostats and integrators

## Installation

### From GitHub (Recommended)

**Prolix** depends on **proxide** (protein I/O and force fields). Install `proxide` first, then `prolix`:

```bash
pip install "git+https://github.com/maraxen/proxide.git@main"
pip install "git+https://github.com/maraxen/prolix.git@main"
```

If `proxide` is already on PyPI for your platform, `pip install "git+https://github.com/maraxen/prolix.git@main"` may resolve it automatically; otherwise use the two-step order above (proxide uses a Rust/`maturin` build).

### From Source

To install Prolix from source:

```bash
git clone https://github.com/maraxen/prolix.git
cd prolix
pip install .
```

For development installation with test dependencies:

```bash
pip install -e ".[dev]"
```

Optional OpenMM parity tooling: `pip install -e ".[dev,openmm]"` (requires a compatible PyPI wheel for your platform; see `pyproject.toml`).

### UV workspace (editable proxide)

`proxide` is pinned for UV as a workspace member. With sibling checkouts `…/prolix` and `…/proxide`, install from their parent (see `workspace/README.md`):

```bash
# from the prolix repo root; ../proxide must exist
bash scripts/sync_workspace_lock.sh
cd .. && uv sync --extra cuda --extra dev --package prolix
```

After changing dependencies in `pyproject.toml`, run `bash scripts/sync_workspace_lock.sh` again and commit `workspace/uv.lock`.

## Usage

### Running a Simple MD Simulation

```python
from proxide.io.parsing import load_structure
from proxide.physics.force_fields import load_force_field
from prolix.physics.simulate import run_simulation

protein = load_structure("protein.pdb")
ff = load_force_field("path/to/ff19SB.eqx")
# Combine the structure and force field into system parameters (see docs / tutorials), then:
# final_positions = run_simulation(system_params, protein.coord, temperature=300.0)
```

### Computing Energy

```python
from jax_md import space
from prolix.physics import system

# Create energy function
displacement_fn, _ = space.free()
energy_fn = system.make_energy_fn(
    displacement_fn=displacement_fn,
    system_params=params,
    implicit_solvent=True
)

# Compute energy
energy = energy_fn(positions)
```

## Development

### Running Tests

Run the full test suite:

```bash
pytest
```

Run fast "smoke" tests to verify core functionality:

```bash
pytest -m smoke
```

### Linting and Typing

Check code quality:

```bash
ruff check .
pyright
```

## Dependencies

Prolix depends on:

- **proxide**: Protein I/O, parsing, and force field loading
- **jax-md**: Molecular dynamics primitives
- **jax**: Automatic differentiation and accelerators
- **equinox**: Modules and serialization

## License

MIT License - see LICENSE file for details.
