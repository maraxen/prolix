# Examples

## Basic MD Simulation

A simple example showing how to run a molecular dynamics simulation with prolix:

```python
import jax.numpy as jnp
from jax_md import space
from prolix.physics import system, simulate
from priox.physics import force_fields
from priox.io.parsing import biotite as bio

# Load structure
structure = bio.load_structure_with_hydride("protein.pdb", add_hydrogens=True)

# Load force field
ff = force_fields.load_force_field("ff19SB.eqx")

# Parameterize system
params = system.parameterize_system(ff, structure)

# Create energy function
displacement_fn, _ = space.free()
energy_fn = system.make_energy_fn(
    displacement_fn=displacement_fn,
    system_params=params,
    implicit_solvent=True,
    solvent_dielectric=78.5,
)

# Run minimization
from prolix.physics import simulate
minimized_coords = simulate.minimize_energy(
    structure.coord,
    energy_fn,
    max_iterations=1000
)

# Run MD
trajectory = simulate.run_md(
    minimized_coords,
    params,
    temperature=300.0,
    steps=10000,
    dt=0.002  # 2 fs timestep
)
```

## Energy Calculation

Computing energies for a protein structure:

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

# Compute total energy
total_energy = energy_fn(coords)

# Compute energy components
bond_energy = system.bonded.make_bond_energy_fn(
    displacement_fn, params['bonds'], params['bond_params']
)(coords)

angle_energy = system.bonded.make_angle_energy_fn(
    displacement_fn, params['angles'], params['angle_params']
)(coords)
```

## Working with Force Fields

Loading and using different force fields:

```python
from priox.physics import force_fields

# Load ff14SB
ff14 = force_fields.load_force_field("path/to/ff14SB.eqx")

# Load ff19SB
ff19 = force_fields.load_force_field("path/to/ff19SB.eqx")

# Get charge for specific atom
charge = ff19.get_charge("ALA", "CA")

# Get LJ parameters
sigma, epsilon = ff19.get_lj_params("ALA", "CA")
```
