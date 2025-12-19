
import jax
import jax.numpy as jnp
from proxide.io.parsing.rust import parse_structure
from oxidize import CoordFormat, OutputSpec
import numpy as np
import oxidize
from prolix.physics import system
from prolix.physics.neighbor_list import ExclusionSpec

# Load protein
pdb_path = 'data/pdb/1UAO.pdb'
ff_path = 'proxide/src/proxide/assets/protein.ff19SB.xml'
spec = OutputSpec(
    coord_format=CoordFormat.Full,
    force_field=ff_path,
    parameterize_md=True,
    add_hydrogens=False, # DISABLED GHOST ATOMS
)
protein = parse_structure(pdb_path, spec)

# Assign radii manually (as in simulation script)
print("Assigning radii...")
radii_list = oxidize.assign_mbondi2_radii(protein.atom_names, protein.bonds.tolist())
protein = protein.replace(radii=jnp.array(radii_list))

# Convert to system params
# prolix.simulate.run_simulation does this internally, we replicate it
atomic_sys = protein
n_atoms = len(atomic_sys.coordinates) // 3
print(f"Num atoms: {n_atoms}")

system_params = {
    "charges": jnp.array(atomic_sys.charges),
    "sigmas": jnp.array(atomic_sys.sigmas),
    "epsilons": jnp.array(atomic_sys.epsilons),
    "bonds": jnp.array(atomic_sys.bonds),
    "bond_params": jnp.array(atomic_sys.bond_params),
    "angles": jnp.array(atomic_sys.angles),
    "angle_params": jnp.array(atomic_sys.angle_params),
    "dihedrals": jnp.array(atomic_sys.proper_dihedrals),
    "dihedral_params": jnp.array(atomic_sys.dihedral_params),
    "impropers": jnp.array(atomic_sys.impropers),
    "improper_params": jnp.array(atomic_sys.improper_params),
    "gb_radii": jnp.array(atomic_sys.radii),
}

# Build exclusion spec explicitly to ensure it works
excl_spec = ExclusionSpec.from_system_params(system_params)
print(f"Exclusions: 1-2/1-3: {len(excl_spec.idx_12_13)}, 1-4: {len(excl_spec.idx_14)}")

# Create energy function (Dense / Implicit Solvent)
energy_fn = system.make_energy_fn(
    displacement_fn=None, # Non-periodic
    system_params=system_params,
    exclusion_spec=excl_spec,
    implicit_solvent=True,
    use_pbc=False
)

# Calculate Energy
positions = jnp.array(atomic_sys.coordinates).reshape(-1, 3)
try:
    E = energy_fn(positions)
    print(f"Total Energy: {E}")
except Exception as e:
    print(f"Energy calc failed: {e}")

# Breakdown?
# system.make_energy_fn returns a composed function.
# We can't easily break it down without calling internal functions.
# But if E is huge, we know.
