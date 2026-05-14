
import jax
import jax.numpy as jnp
import numpy as np
import os
import tempfile
from openmm import app, unit
import openmm
from pdbfixer import PDBFixer
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat, assign_mbondi2_radii, assign_obc2_scaling_factors
from prolix.physics import bonded, system, neighbor_list as nl, generalized_born

def debug_parity():
    pdb_path = "data/pdb/1UAO.pdb"
    import proxide
    ff_xml_path = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")
    
    fixer = PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp)
        tmp_path = tmp.name
        
    spec = OutputSpec(coord_format=CoordFormat.Full, add_hydrogens=False, parameterize_md=True, force_field=ff_xml_path)
    protein_system = parse_structure(tmp_path, spec=spec)

    radii = assign_mbondi2_radii(list(protein_system.atom_names), protein_system.bonds)
    scaled_radii = assign_obc2_scaling_factors(list(protein_system.atom_names))
    object.__setattr__(protein_system, 'radii', np.array(radii, dtype=np.float32))
    object.__setattr__(protein_system, 'scaled_radii', np.array(scaled_radii, dtype=np.float32))
    
    exclusion_spec = nl.ExclusionSpec.from_protein(protein_system)
    
    from jax_md import space
    displacement_fn, shift_fn = space.free()
    
    # Use NoCutoff for both
    energy_fns = system.make_energy_fn(displacement_fn, protein_system, implicit_solvent=True, 
                                      exclusion_spec=exclusion_spec, return_decomposed=True,
                                      cutoff_distance=0.0)

    pos = jnp.array(protein_system.coordinates)
    
    # OMM Setup
    pdb_file = app.PDBFile(tmp_path)
    topology = pdb_file.topology
    omm_positions = pdb_file.positions
    os.unlink(tmp_path)

    omm_ff = app.ForceField(ff_xml_path, "implicit/obc2.xml")
    omm_system = omm_ff.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False
    )
    nb_force = [f for f in omm_system.getForces() if isinstance(f, openmm.NonbondedForce)][0]

    # Create separate groups for each force type
    # NonbondedForce will hold LJ and Coulomb
    # We want to identify the index of NonbondedForce
    nb_force_idx = -1
    for i, force in enumerate(omm_system.getForces()):
        force.setForceGroup(i)
        if isinstance(force, openmm.NonbondedForce):
            nb_force_idx = i

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(omm_positions)

    # For NonbondedForce, we can decompose into LJ and Coulomb parts if needed,
    # but OpenMM's NonbondedForce usually returns the sum as one.
    # Actually we can set the force group of components.
    
    omm_components = {}
    for i, force in enumerate(omm_system.getForces()):
        # This will give the energy for this group (force)
        state = simulation.context.getState(getEnergy=True, groups=1 << i)
        e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        name = force.__class__.__name__
        
        if isinstance(force, openmm.NonbondedForce):
             # Try to isolate if possible or keep as one
             omm_components["NonbondedForce_Total"] = e
        else:
             omm_components[name] = omm_components.get(name, 0.0) + e

    # --- NEW DIAGNOSTICS ---
    print(f"\n--- DIAGNOSTICS START ---")
    
    # [NEW] 1. Print Improper Params
    print(f"\n[1] JAX improper_params sum: {jnp.sum(jnp.abs(protein_system.improper_params))}")
    print(f"    JAX improper_params[0]: {protein_system.improper_params[0]}")
    
    # [NEW] 2. Check Charges
    print(f"\n[2] JAX charges sum: {jnp.sum(protein_system.charges)}")
    print(f"    JAX charges squared sum: {jnp.sum(protein_system.charges**2)}")

    # [NEW] 3. Check OMM Nonbonded Exceptions
    print(f"\n[3] OMM Nonbonded Exceptions count: {nb_force.getNumExceptions()}")
    # Compare with JAX exclusions
    jax_excl_count = 0
    if hasattr(protein_system, 'excl_indices'):
        jax_excl_count = jnp.count_nonzero(protein_system.excl_indices != -1)
    print(f"    JAX Exclusion indices count (non -1): {jax_excl_count}")

    # 1. Torsion Params
    print(f"\n[4] Torsion / Improper Params:")
    # Get PeriodicTorsionForce
    torsion_force = [f for f in omm_system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)][0]
    num_torsions = torsion_force.getNumTorsions()
    print(f"OMM Total Torsions: {num_torsions}")
    print(f"JAX Total Improp: {len(protein_system.improper_params)}")
    
    for i in range(min(9, num_torsions)):
        params = torsion_force.getTorsionParameters(i)
        p1, p2, p3, p4, peri, phase, k = params
        print(f"Torsion {i} (OMM): {p1, p2, p3, p4} k={k.value_in_unit(unit.kilojoules_per_mole):.4f}")
        if i < len(protein_system.improper_params):
             # JAX params (assuming improper_params structure)
             jp = protein_system.improper_params[i]
             print(f"  Torsion {i} (JAX): {jp}")

    # 2. Bond Data
    print(f"\n[5] Bond Data:")
    print(f"JAX total bonds: {len(protein_system.bonds)}")
    for i, b in enumerate(protein_system.bonds[:20]):
        print(f"Bond {i}: {b}")
    
    # Check for atom 0 connectivity
    atom0_bonds = [b for b in protein_system.bonds if 0 in b]
    print(f"Atom 0 bonds: {len(atom0_bonds)} -> {atom0_bonds}")

    # 3. LJ Parameters
    print(f"\n[6] LJ Params (Atom 0):")
    print(f"JAX Atom 0: sigma={protein_system.sigmas[0]:.4f}, eps={protein_system.epsilons[0]:.4f}")
    # OMM params for atom 0
    # NonbondedForce usually uses index
    for i in range(nb_force.getNumParticles()):
        if i == 0:
            q, s, e = nb_force.getParticleParameters(i)
            print(f"OMM Atom 0: sigma={s.value_in_unit(unit.nanometers)*10:.4f}, eps={e.value_in_unit(unit.kilojoules_per_mole)*0.239006:.4f}")
    
    # 4. Exclusion mapping (Atom 0)
    print(f"\n[7] Exclusion mapping (Atom 0):")
    # Manually compute
    def get_neighbors(atom, bonds):
        neighbors = set()
        for b in bonds:
            if int(b[0]) == atom: neighbors.add(int(b[1]))
            if int(b[1]) == atom: neighbors.add(int(b[0]))
        return neighbors

    neighbors_12 = get_neighbors(0, protein_system.bonds)
    neighbors_13 = set()
    for n in neighbors_12:
        neighbors_13.update(get_neighbors(n, protein_system.bonds))
    neighbors_13 -= neighbors_12
    neighbors_13 -= {0}
    
    print(f"Atom 0 neighbors (1-2): {len(neighbors_12)}")
    print(f"Atom 0 neighbors (1-3): {len(neighbors_13)}")
    
    # Also look at nb_force exceptions
    atom0_exc = 0
    for i in range(nb_force.getNumExceptions()):
        p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
        if p1 == 0 or p2 == 0:
            atom0_exc += 1
    print(f"Atom 0 OMM Exceptions: {atom0_exc}")
    print(f"--- DIAGNOSTICS END ---\n")
    # JAX Decomposition
    jax_lj = energy_fns["lj"](pos)
    jax_elec = energy_fns["electrostatics"](pos)
    # jax_elec returns (e_gb, e_direct, born_radii)
    e_gb, e_direct, born_radii = jax_elec
    
    # [NEW] 4. Print JAX ACE Term
    e_ace = jnp.sum(generalized_born.compute_ace_nonpolar_energy(protein_system.radii, born_radii))
    print(f"DEBUG JAX ACE Energy: {e_ace:.4f}")

    print(f"--- Global Constant Checks ---")
    print(f"JAX COULOMB_CONSTANT: 332.0637")
    print(f"JAX pme_alpha (implicit): 0.0")

    print(f"--- Nonbonded Comparison ---")
    print(f"JAX LJ: {jax_lj:.4f}")
    print(f"JAX Elec Direct: {e_direct:.4f}")
    print(f"JAX GB: {e_gb:.4f}")
    print(f"JAX Total NB: {jax_lj + e_direct + e_gb:.4f}")
    
    omm_total = omm_components.get("NonbondedForce_Total", 0.0) + omm_components.get("CustomGBForce", 0.0)
    print(f"OMM Total NB: {omm_total:.4f}")
    print(f"Diff (JAX - OMM): {(jax_lj + e_direct + e_gb) - omm_total:.4f}")

    # CMAP
    jax_cmap = energy_fns["cmap"](pos)
    omm_cmap = omm_components.get("CMAPTorsionForce", 0.0)
    print(f"--- CMAP Comparison ---")
    print(f"JAX CMAP: {jax_cmap:.4f}")
    print(f"OMM CMAP: {omm_cmap:.4f}")
    if abs(jax_cmap) > 1e-6:
        print(f"Ratio OMM/JAX: {omm_cmap/jax_cmap:.4f}")

    # Torsion
    jax_torsion = energy_fns["dihedral"](pos) + energy_fns["improper"](pos)
    omm_torsion = omm_components.get("PeriodicTorsionForce", 0.0)
    print(f"--- Torsion Comparison ---")
    print(f"JAX Torsion: {jax_torsion:.4f}")
    print(f"OMM Torsion: {omm_torsion:.4f}")
    print(f"Diff: {jax_torsion - omm_torsion:.4f}")

    # Investigate LJ parameters
    print(f"--- LJ Params Check ---")
    print(f"ExclusionSpec counts: 12-13: {len(exclusion_spec.idx_12_13)}, 14: {len(exclusion_spec.idx_14)}")
    
    # Check for dense exclusions
    if hasattr(protein_system, "dense_excl_scale_vdw") and protein_system.dense_excl_scale_vdw is not None:
        print(f"Dense exclusion mask found! Shape: {protein_system.dense_excl_scale_vdw.shape}")
        print(f"Non-zero elements in dense mask: {np.count_nonzero(protein_system.dense_excl_scale_vdw < 1.0)}")
    else:
        print("No dense exclusion mask found.")

    # Check a few exceptions
    nb_force = [f for f in omm_system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    print(f"--- Exceptions Check ---")
    for i in range(min(10, nb_force.getNumExceptions())):
        p1, p2, q, s, e = nb_force.getExceptionParameters(i)
        q = q.value_in_unit(unit.elementary_charge**2)
        s = s.value_in_unit(unit.nanometers)
        e = e.value_in_unit(unit.kilojoules_per_mole)
        
        # JAX LB
        q1, q2 = float(protein_system.charges[p1]), float(protein_system.charges[p2])
        s1, s2 = float(protein_system.sigmas[p1]), float(protein_system.sigmas[p2])
        e1, e2 = float(protein_system.epsilons[p1]), float(protein_system.epsilons[p2])
        
        # Determine if it's 1-4 or 1-2/1-3
        is_14 = False
        for pair in exclusion_spec.idx_14:
            if (pair[0] == p1 and pair[1] == p2) or (pair[0] == p2 and pair[1] == p1):
                is_14 = True
                break
        
        if is_14:
            jax_q = q1 * q2 * exclusion_spec.scale_14_elec
            jax_s = 0.5 * (s1 + s2)
            jax_e = np.sqrt(e1 * e2) * exclusion_spec.scale_14_vdw
        else:
            jax_q = 0.0
            jax_s = 0.0 # Standard for fully excluded
            jax_e = 0.0
        
        print(f"Exc {i} ({p1}, {p2}) {'1-4' if is_14 else '1-2/1-3'}:")
        print(f"  OMM: q={q:.6f}, s={s*10:.6f}, e={e*0.239006:.6f}")
        print(f"  JAX: q={jax_q:.6f}, s={jax_s:.6f}, e={jax_e:.6f}")
    
if __name__ == "__main__":
    debug_parity()
