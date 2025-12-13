"""PME parity tests against OpenMM.

NOTE: This test is skipped because it requires OpenMM and deprecated biotite modules.
"""

import pytest

pytest.skip("Requires OpenMM and deprecated biotite - needs migration", allow_module_level=True)

PROLIX_FF = "data/force_fields/ff14SB.eqx"
SOLVATED_PDB = "data/pdb/1UAO_solvated_tip3p.pdb"

def test_pme_parity():
    """Verifies PME Energy Parity with OpenMM for Explicit Solvent System."""
    print(f"Loading {SOLVATED_PDB}...")
    if not os.path.exists(SOLVATED_PDB):
        pytest.skip("Solvated PDB not found. Run test_explicit_parity.py first.")

    # Load Structure
    atom_array = parsing_biotite.load_structure_with_hydride(SOLVATED_PDB, model=1, remove_solvent=False)
    pos = jnp.array(atom_array.coord)
    
    # Box
    if atom_array.box is not None:
        box = atom_array.box
        if box.ndim == 3: box = box[0]
        if box.ndim == 2:
            box_size = jnp.array([box[0,0], box[1,1], box[2,2]])
        elif box.ndim == 1:
             box_size = jnp.array([box[0], box[1], box[2]])
        else:
             print(f"Warning: Unexpected box shape {box.shape}")
             box_size = jnp.array([40.0, 40.0, 40.0])
    else:
        box_size = jnp.array([40.0, 40.0, 40.0])

    # Parameterize
    atom_names = list(atom_array.atom_name)
    import biotite.structure as struc
    res_starts = struc.get_residue_starts(atom_array)
    residues = [atom_array.res_name[i] for i in res_starts]
    atom_counts = []
    for i in range(len(res_starts)-1):
        atom_counts.append(res_starts[i+1] - res_starts[i])
    atom_counts.append(len(atom_array) - res_starts[-1])

    ff = ff_loader.load_force_field(PROLIX_FF)
    params = bridge_core.parameterize_system(
        ff, residues, atom_names, atom_counts, 
        water_model="TIP3P", 
        rigid_water=True 
    )

    # Displacements
    displacement_fn, shift_fn = space.periodic(box_size)

    # Make Energy Fn (PME Enabled)
    energy_fn = system.make_energy_fn(
        displacement_fn,
        params,
        box=box_size,
        use_pbc=True,
        implicit_solvent=False,
        cutoff_distance=9.0, 
        pme_grid_points=64, 
        pme_alpha=0.34
    )
    
    e_prolix = energy_fn(pos)
    print(f"Prolix Total Energy (PME): {e_prolix} kcal/mol")
    
    # Compare with OpenMM
    pdb_in = app.PDBFile(SOLVATED_PDB)
    ff_omm = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system_omm = ff_omm.createSystem(pdb_in.topology, 
                                 nonbondedMethod=app.PME, 
                                 nonbondedCutoff=0.9*unit.nanometers, 
                                 constraints=app.HBonds, 
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)
            
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system_omm, integrator)
    context.setPositions(pdb_in.positions)
    state = context.getState(getEnergy=True)
    e_omm = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"OpenMM Total Energy: {e_omm}")
    
    diff = abs(e_prolix - e_omm)
    rel_error = diff / abs(e_omm)
    print(f"Difference: {diff} kcal/mol ({rel_error*100:.2f}%)")
    
    # Assert Parity (Allow 2% error or 100 kcal/mol)
    assert diff < 100.0 or rel_error < 0.02, f"PME Parity Failed. Diff {diff}"

if __name__ == "__main__":
    test_pme_parity()
