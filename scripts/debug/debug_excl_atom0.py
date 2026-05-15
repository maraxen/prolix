import os
import openmm.app as app
import openmm
import jax.numpy as jnp
from jax_md import space
from prolix.physics import system, neighbor_list
from proxide import parse_structure, OutputSpec

def debug_exclusions():
    pdb_path = "data/pdb/1UAO.pdb"
    
    pdb = app.PDBFile(pdb_path)
    omm_ff = app.ForceField("amber14/protein.ff14SB.xml", "implicit/obc2.xml")
    omm_system = omm_ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    nb_force = [f for f in omm_system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    
    ff_path = "/home/marielle/projects/prolix/.venv/lib/python3.13/site-packages/openmm/app/data/amber14/protein.ff14SB.xml"
    spec = OutputSpec(parameterize_md=True, force_field=ff_path, add_hydrogens=False)
    protein = parse_structure(pdb_path, spec)
    
    # Check atom 0
    print(f"Atom 0: {protein.atom_types[0] if protein.atom_types is not None else 'N'}")
    
    # OpenMM exceptions for atom 0
    omm_excl = {}
    for i in range(nb_force.getNumExceptions()):
        p1, p2, q, s, e = nb_force.getExceptionParameters(i)
        if p1 == 0: omm_excl[p2] = (q, s, e)
        if p2 == 0: omm_excl[p1] = (q, s, e)
    
    print("\nOpenMM Exceptions for Atom 0:")
    for p, params in sorted(omm_excl.items()):
        q = params[0].value_in_unit(openmm.unit.elementary_charge**2)
        s = params[1].value_in_unit(openmm.unit.nanometer)
        e = params[2].value_in_unit(openmm.unit.kilojoule_per_mole)
        print(f"  Atom {p}: q={q:.4f}, s={s:.4f}, e={e:.4f}")
        
    # JAX exclusions for atom 0
    exclusion_spec = neighbor_list.ExclusionSpec.from_protein(protein)
    excl_idx, excl_sv, excl_se = neighbor_list.map_exclusions_to_dense_padded(exclusion_spec, max_exclusions=32)
    
    print("\nJAX Exclusions for Atom 0:")
    for i in range(32):
        p = int(excl_idx[0, i])
        if p == -1: break
        print(f"  Atom {p}: scale_vdw={excl_sv[0, i]:.4f}, scale_elec={excl_se[0, i]:.4f}")

if __name__ == "__main__":
    debug_exclusions()
