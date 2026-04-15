# Examples

## Basic MD Simulation

This mirrors the flow in `scripts/simulate_chignolin_stable.py`: load a structure with **proxide**, parameterize with **`parameterize_system`**, then build a **prolix** energy function and run minimization or full MD.

```python
import jax.numpy as jnp
import biotite.structure as struc
from jax_md import space

from proxide.io.parsing import biotite as parsing_biotite
from proxide.md.bridge.core import parameterize_system
from proxide.physics.force_fields.loader import load_force_field
from prolix.physics import system
from prolix.physics.simulate import run_minimization, run_simulation

# 1) Load coordinates (Biotite AtomArray)
atom_array = parsing_biotite.load_structure_with_hydride("protein.pdb", model=1)

# 2) Build residue / atom-name lists (see simulate_chignolin_stable.py for terminal renaming)
residues: list[str] = []
atom_names: list[str] = []
atom_counts: list[int] = []
res_starts = struc.get_residue_starts(atom_array)
for i, start_idx in enumerate(res_starts):
    end_idx = res_starts[i + 1] if i + 1 < len(res_starts) else len(atom_array)
    res_atoms = atom_array[start_idx:end_idx]
    residues.append(res_atoms.res_name[0])
    names = res_atoms.atom_name.tolist()
    atom_names.extend(names)
    atom_counts.append(len(names))

ff = load_force_field("ff19SB.eqx")
system_params = parameterize_system(ff, residues, atom_names, atom_counts)

# 3) Energy + dynamics
displacement_fn, _ = space.free()
energy_fn = system.make_energy_fn(
    displacement_fn,
    system_params,
    implicit_solvent=True,
    solvent_dielectric=78.5,
)
coords = jnp.asarray(atom_array.coord)
coords = run_minimization(energy_fn, coords, steps=500)
# Optional full pipeline (minimize → thermalize → production):
# coords = run_simulation(system_params, coords, temperature=300.0, min_steps=500, therm_steps=1000)
```

## Energy Calculation

With `system_params` and positions `coords` from above:

```python
from jax_md import space
from prolix.physics import system

displacement_fn, _ = space.free()
energy_fn = system.make_energy_fn(
    displacement_fn,
    system_params,
    implicit_solvent=True,
)
total_energy = energy_fn(coords)
```

For bonded/nonbonded breakdowns, use the helpers under `prolix.physics.bonded` and the same `system_params` fields your parameterization produced (see tests and `scripts/`).

## Working with Force Fields

```python
from proxide.physics.force_fields.loader import load_force_field

ff14 = load_force_field("path/to/ff14SB.eqx")
ff19 = load_force_field("path/to/ff19SB.eqx")
# Per-residue charges and LJ parameters are accessed via the loaded force-field object API
# (see proxide docs and loader tests for details).
```
