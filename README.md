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

To install Prolix, simply run:

```bash
pip install .
```

For development installation with test dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Running a Simple MD Simulation

```python
from prolix.physics import system, simulate
from priox.physics import force_fields
from priox.io.parsing import biotite as bio

# Load structure and force field
structure = bio.load_structure_with_hydride("protein.pdb")
ff = force_fields.load_force_field("path/to/ff19SB.eqx")

# Set up simulation
params = system.parameterize_system(ff, structure)
trajectory = simulate.run_md(
    positions=structure.coord,
    params=params,
    temperature=300.0,
    steps=10000
)
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

- **priox**: For protein I/O and force field loading
- **jax-md**: For molecular dynamics engine
- **jax**: For automatic differentiation and GPU acceleration
- **equinox**: For neural network modules and serialization

## License

MIT License - see LICENSE file for details.
