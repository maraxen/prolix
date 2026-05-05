
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
from prolix.physics import bonded, system, neighbor_list as nl

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
    
    # OpenMM Setup
    pdb_file = app.PDBFile(tmp_path)
    topology = pdb_file.topology
    omm_positions = pdb_file.positions
    os.unlink(tmp_path)

    omm_ff = app.ForceField(ff_xml_path, "implicit/obc2.xml")
    omm_system = omm_ff.createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False
    )

    for i, force in enumerate(omm_system.getForces()):
        force.setForceGroup(i)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(omm_positions)

    omm_components = {}
    for i, force in enumerate(omm_system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups=1 << i)
        force_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        fname = force.__class__.__name__
        omm_components[fname] = omm_components.get(fname, 0.0) + force_energy

    # JAX Decomposition
    jax_lj = energy_fns["lj"](pos)
    jax_elec = energy_fns["electrostatics"](pos)
    # jax_elec returns (e_gb, e_direct, born_radii)
    e_gb, e_direct, _ = jax_elec
    
    print(f"--- Nonbonded Comparison ---")
    print(f"JAX LJ: {jax_lj:.4f}")
    print(f"JAX Elec Direct: {e_direct:.4f}")
    print(f"JAX GB: {e_gb:.4f}")
    
    omm_nb = omm_components.get("NonbondedForce", 0.0)
    omm_gb = omm_components.get("CustomGBForce", 0.0)
    print(f"OMM Nonbonded (LJ+Coul): {omm_nb:.4f}")
    print(f"OMM GB: {omm_gb:.4f}")
    
    print(f"Nonbonded Diff: {jax_lj + e_direct - omm_nb:.4f}")
    print(f"GB Diff: {e_gb - omm_gb:.4f}")

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
