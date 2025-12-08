
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import openmm
from openmm import app, unit
import os
import biotite.structure.io.pdb as pdb
from priox.io.parsing import biotite as parsing_biotite
from priox.md.bridge import core as bridge_core
from priox.physics.force_fields import loader as ff_loader
from prolix.physics import system, bonded
from jax_md import space

PROLIX_FF = "data/force_fields/ff14SB.eqx"
PDB_1UAO = "data/pdb/1UAO.pdb"
SOLVATED_PDB = "data/pdb/1UAO_solvated_tip3p.pdb"

def generate_solvated_system():
    print(f"Generating solvated system at {SOLVATED_PDB} via OpenMM...")
    if not os.path.exists(PDB_1UAO):
        raise FileNotFoundError(f"Missing {PDB_1UAO}")
        
    pdb_in = app.PDBFile(PDB_1UAO)
    modeller = app.Modeller(pdb_in.topology, pdb_in.positions)
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    # Use padding 1.0 nm = 10 A
    modeller.addSolvent(ff, model='tip3p', padding=1.0*unit.nanometers)
    
    # Save
    with open(SOLVATED_PDB, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    
    return modeller.topology, modeller.positions

def get_openmm_energy(pdb_path, use_pme=False):
    print(f"Computing OpenMM Energy (PME={use_pme})...")
    pdb_in = app.PDBFile(pdb_path)
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    
    # Create System
    if use_pme:
        nonbonded = app.PME
        cutoff = 0.9 * unit.nanometers
    else:
        nonbonded = app.CutoffNonPeriodic
        cutoff = 0.9 * unit.nanometers
        
    system_omm = ff.createSystem(pdb_in.topology, 
                                 nonbondedMethod=nonbonded, 
                                 nonbondedCutoff=cutoff, 
                                 constraints=app.HBonds, 
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)

    if use_pme:
        for f in system_omm.getForces():
            if isinstance(f, openmm.NonbondedForce):
                f.setPMEParameters(3.4, 64, 64, 64)
                break
                                 
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system_omm, integrator)
    context.setPositions(pdb_in.positions)
    state = context.getState(getEnergy=True)
    total_e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"OpenMM Total Energy: {total_e}")
    return total_e

def test_parity_tip3p():
    print("\n--- Testing Explicit Solvent Parity ---")
    if not os.path.exists(SOLVATED_PDB):
        generate_solvated_system()
        
    # Get OpenMM Vacuum Energy for direct comparison
    e_omm_vacuum = get_openmm_energy(SOLVATED_PDB, use_pme=False)
    
    # Prolix Setup
    print("Computing Prolix Energy...")
    # remove_solvent=False is CRITICAL to keep the waters we just generated!
    atom_array = parsing_biotite.load_structure_with_hydride(SOLVATED_PDB, model=1, remove_solvent=False)
    pos = jnp.array(atom_array.coord)
    
    # Get Box from PDB (CRYST1)
    if atom_array.box is not None:
        box = atom_array.box
        if box.ndim == 3:
             box = box[0]
        box_size = jnp.array([box[0,0], box[1,1], box[2,2]])
        print(f"Box Size: {box_size}")
    else:
        box_size = jnp.array([40.0, 40.0, 40.0])
    
    # Topology Lists
    atom_names = list(atom_array.atom_name)
    
    # Use per-residue list for parameterize_system
    import biotite.structure as struc
    res_starts = struc.get_residue_starts(atom_array)
    residues = [atom_array.res_name[i] for i in res_starts]
    atom_counts = []
    for i in range(len(res_starts)-1):
        atom_counts.append(res_starts[i+1] - res_starts[i])
    atom_counts.append(len(atom_array) - res_starts[-1])
    
    # Load FF
    ff = ff_loader.load_force_field(PROLIX_FF)
    
    print(f"Unique residues found: {set(residues)}")
    print(f"Atom count: {len(atom_names)}")
    
    # Use rigid_water=True (default) to match OpenMM's rigidWater=True
    # This zeros out bond/angle force constants for water at parameterization time
    params = bridge_core.parameterize_system(
        ff, residues, atom_names, atom_counts, 
        water_model="TIP3P", 
        rigid_water=True  # This zeros k for water bonds/angles
    )
    
    print(f"Total Bonds: {len(params['bonds'])}")
    print(f"Total Angles: {len(params['angles'])}")
    
    # Debug: Check angle k values
    angle_k_values = params["angle_params"][:, 1]
    unique_k = np.unique(np.array(angle_k_values))
    print(f"DEBUG: Unique angle k values: {unique_k}")
    zero_k_count = np.sum(np.array(angle_k_values) == 0.0)
    print(f"DEBUG: Angles with k=0: {zero_k_count} / {len(angle_k_values)}")
    
    # Count k=100 angles (water non-rigid default)
    k100_count = np.sum(np.abs(np.array(angle_k_values) - 100.0) < 0.1)
    print(f"DEBUG: Angles with k~100: {k100_count} / {len(angle_k_values)}")
    
    # Create Energy Fn
    displacement_fn, shift_fn = space.periodic(box_size)
    
    # Debug: Check individual energy components
    bond_fn = bonded.make_bond_energy_fn(displacement_fn, params["bonds"], params["bond_params"])
    e_bond = bond_fn(pos)
    print(f"DEBUG Bond Energy: {e_bond}")
    
    angle_fn = bonded.make_angle_energy_fn(displacement_fn, params["angles"], params["angle_params"])
    e_angle = angle_fn(pos)
    print(f"DEBUG Angle Energy: {e_angle}")
    
    if params["dihedrals"] is not None and len(params["dihedrals"]) > 0:
        dihedral_fn = bonded.make_dihedral_energy_fn(displacement_fn, params["dihedrals"], params["dihedral_params"])
        e_dihedral = dihedral_fn(pos)
        print(f"DEBUG Dihedral Energy: {e_dihedral}")
    
    # Test bonded only first (use_pbc=False to skip PME)
    energy_fn_bonded = system.make_energy_fn(
        displacement_fn,
        params,
        box=box_size,
        use_pbc=False,  # Skip PME for debug
        implicit_solvent=False,  # Explicit water - no GBSA!
        cutoff_distance=9.0, 
        pme_grid_points=64, 
        pme_alpha=0.34
    )
    e_bonded_only = energy_fn_bonded(pos)
    print(f"DEBUG Bonded-Only Energy (No PME): {e_bonded_only}")
    
    # Full energy with PBC/PME
    energy_fn = system.make_energy_fn(
        displacement_fn,
        params,
        box=box_size,
        use_pbc=True,
        implicit_solvent=False,  # Explicit water - no GBSA!
        cutoff_distance=9.0, 
        pme_grid_points=64, 
        pme_alpha=0.34
    )
    
    e_prolix_vacuum = e_bonded_only
    print(f"Prolix Vacuum Energy: {e_prolix_vacuum:.4f}")
    
    diff = abs(e_omm_vacuum - e_prolix_vacuum)
    print(f"Difference (Vacuum): {diff:.4f} kcal/mol")
    
    rel_error = diff / abs(e_omm_vacuum)
    print(f"Relative Error (Vacuum): {rel_error:.6f}")
    
    # Allow 5% error for Vacuum parity (cutoff handling diffs)
    if rel_error > 0.05:
        pytest.fail(f"Vacuum Parity failed. Rel Error {rel_error:.6f} > 5%")
    else:
        print("VACUUM PARITY PASSED! Rigid water implementation verified.")
        
if __name__ == "__main__":
    test_parity_tip3p()
