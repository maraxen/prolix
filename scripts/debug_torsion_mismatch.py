
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space
import openmm
import openmm.app as app
import openmm.unit as unit
from prolix.physics import system
from priox.physics import force_fields
from priox.md import jax_md_bridge
from priox.io.parsing import biotite as parsing_biotite
import biotite.structure.io.pdb as pdb

# Enable x64
jax.config.update("jax_enable_x64", True)

PDB_PATH = "data/pdb/1UAO.pdb"
FF_EQX_PATH = "data/force_fields/protein19SB.eqx"
OPENMM_XMLS = [
    "openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml",
    'implicit/obc2.xml'
]

def debug_torsion():
    print("Debugging Torsion Mismatch...")
    
    # 1. Load Structure
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    
    # 2. OpenMM System
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file_bio = pdb.PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology
        positions = pdb_file.positions

    omm_ff = app.ForceField(*OPENMM_XMLS)
    omm_system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
    
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    simulation = app.Simulation(topology, omm_system, integrator, openmm.Platform.getPlatformByName('Reference'))
    simulation.context.setPositions(positions)
    
    state = simulation.context.getState(getEnergy=True)
    print(f"OpenMM Total Energy: {state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)}")
    
    # 3. JAX System
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    # Extract topology info
    residues = []
    atom_names = []
    atom_counts = []
    for i, chain in enumerate(topology.chains()):
        for j, res in enumerate(chain.residues()):
            residues.append(res.name)
            count = 0
            for atom in res.atoms():
                name = atom.name
                if i == 0 and j == 0 and name == "H": name = "H1"
                atom_names.append(name)
                count += 1
            atom_counts.append(count)
            
    if residues:
        residues[0] = "N" + residues[0]
        residues[-1] = "C" + residues[-1]
            
    system_params = jax_md_bridge.parameterize_system(ff, residues, atom_names, atom_counts)
    
    jax_positions = jnp.array(positions.value_in_unit(unit.angstrom))
    
    # 4. Investigate Torsion 11-10-13-14
    # Note: These are 0-indexed indices.
    target_indices = (11, 10, 13, 14) 
    print(f"\nTarget Torsion: {target_indices}")
    
    # Get JAX Params
    print("JAX Params:")
    dih = system_params['dihedrals']
    params = system_params['dihedral_params']
    
    jax_terms = []
    for i in range(len(dih)):
        idx = tuple(int(x) for x in dih[i])
        # Canonicalize
        if idx[0] > idx[3]: c_idx = (idx[3], idx[2], idx[1], idx[0])
        else: c_idx = idx
        
        t_target = target_indices
        if t_target[0] > t_target[3]: c_target = (t_target[3], t_target[2], t_target[1], t_target[0])
        else: c_target = t_target
        
        if c_idx == c_target:
            p = params[i]
            print(f"  Indices: {idx} Param: n={p[0]} phase={p[1]} k={p[2]}")
            jax_terms.append(p)
            
    # Get OpenMM Params
    print("OpenMM Params:")
    omm_terms = []
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(i)
                idx = (p1, p2, p3, p4)
                if p1 > p4: c_idx = (p4, p3, p2, p1)
                else: c_idx = idx
                
                t_target = target_indices
                if t_target[0] > t_target[3]: c_target = (t_target[3], t_target[2], t_target[1], t_target[0])
                else: c_target = t_target
                
                if c_idx == c_target:
                     print(f"  Indices: {idx} Param: n={per} phase={phase} k={k}")
                     omm_terms.append((per, phase.value_in_unit(unit.radians), k.value_in_unit(unit.kilocalories_per_mole)))

    # Calculate Phi
    r = jax_positions
    r_i, r_j, r_k, r_l = r[target_indices[0]], r[target_indices[1]], r[target_indices[2]], r[target_indices[3]]
    
    def calc_phi(r_i, r_j, r_k, r_l):
        b0 = r_j - r_i
        b1 = r_k - r_j
        b2 = r_l - r_k
        b1_norm = jnp.linalg.norm(b1)
        b1_unit = b1 / b1_norm
        v = b0 - jnp.dot(b0, b1_unit) * b1_unit
        w = b2 - jnp.dot(b2, b1_unit) * b1_unit
        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1_unit, v), w)
        return jnp.arctan2(y, x)

    phi = calc_phi(r_i, r_j, r_k, r_l)
    print(f"\nCalculated Phi: {phi} radians ({np.degrees(phi)} degrees)")
    
    # Calculate Energy Manually
    e_jax_sum = 0.0
    print("\nJAX Manual Energy:")
    for p in jax_terms:
        # E = k * (1 + cos(n*phi - phase))
        e = p[2] * (1.0 + jnp.cos(p[0] * phi - p[1]))
        print(f"  n={p[0]} k={p[2]:.4f} E={e:.4f}")
        e_jax_sum += e
    print(f"Total JAX Manual: {e_jax_sum:.4f} kcal/mol")
    
    e_omm_sum = 0.0
    print("\nOpenMM Manual Energy:")
    for p in omm_terms:
        e = p[2] * (1.0 + np.cos(p[0] * phi - p[1]))
        print(f"  n={p[0]} k={p[2]:.4f} E={e:.4f}")
        e_omm_sum += e
    print(f"Total OpenMM Manual: {e_omm_sum:.4f} kcal/mol")
    
    # Create Isolated OpenMM System for this Torsion
    print("\nIsolating Torsion in OpenMM...")
    iso_system = openmm.System()
    for _ in range(topology.getNumAtoms()):
        iso_system.addParticle(1.0) # mass doesn't matter
    
    force = openmm.PeriodicTorsionForce()
    # Add ONLY the target torsion terms
    for p in omm_terms:
        # Indices 11, 10, 13, 14
        # OpenMM usually requires specific ordering?
        # Use the ordering we found or the target?
        # The params we extracted matched the target indices.
        force.addTorsion(target_indices[0], target_indices[1], target_indices[2], target_indices[3], int(p[0]), p[1], p[2]*4.184) # kJ/mol input if we used OpenMM unit system? 
        # Wait, creating System usually takes k in kJ/mol?
        # p[2] is in kcal/mol (converted above).
        # OpenMM expects inputs in internal units? No, Quantity.
        # Let's use Quantity.
    
    # Let's recreate properly using Quantity
    # Find PeriodicTorsionForce in original system
    f_orig = None
    for f in omm_system.getForces():
        if isinstance(f, openmm.PeriodicTorsionForce):
            f_orig = f
            break
            
    if f_orig is None:
        print("Error: No PeriodicTorsionForce found in original system.")
        return

    print("\nTerm-by-Term Isolation:")
    for idx_term, p in enumerate(omm_terms):
        sys_iso = openmm.System()
        for _ in range(topology.getNumAtoms()):
            sys_iso.addParticle(1.0)
            
        f_iso = openmm.PeriodicTorsionForce()
        f_iso.addTorsion(target_indices[0], target_indices[1], target_indices[2], target_indices[3], int(p[0]), p[1], p[2]*4.184) # Convert back to kJ/mol for OpenMM input?
        # WAIT. p[2] was converted to kcal/mol in omm_terms list.
        # But addTorsion expects Quantity (if units enabled) or kJ/mol (if implicit).
        # We manually converted p[2] to kcal/mol for printing/manual calc.
        # But here we are passing it to addTorsion.
        # We should pass Quantity.
        # Let's retrieve original quantity from force or convert properly.
        # Better: use original units.
        
        # Or just multiply by 4.184 to get kJ/mol value (assuming implicitly kJ/mol or using Quantity).
        # OpenMM addTorsion args: per, phase, k (Quantity or float in kJ/mol if using implicit units?)
        # Since we use app.Simulation with topology, the system is in standard units?
        # No, System is just System.
        # If we didn't define units, it assumes standard (kJ/mol, nm).
        
        # In the previous run, I used:
        # force.addTorsion(..., k) 
        # where k was from f_orig.getTorsionParameters(i) which returns Quantity!
        # And I passed that Quantity directly.
        # That was CORRECT.
        
        # But here p comes from omm_terms which I manually stripped units from!
        # omm_terms.append((per, phase.value_in_unit(...), k.value_in_unit(...)))
        # So p[2] is float (kcal/mol).
        # We need to pass Quantity to addTorsion or kJ/mol float.
        # Let's pass equivalent kJ/mol float.
        k_kj = p[2] * 4.184
        f_iso.addTorsion(target_indices[0], target_indices[1], target_indices[2], target_indices[3], int(p[0]), p[1], k_kj)
        
        sys_iso.addForce(f_iso)
        
        integ = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        sim = app.Simulation(topology, sys_iso, integ, openmm.Platform.getPlatformByName('Reference'))
        sim.context.setPositions(positions)
        e_iso = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        # Manual E
        e_man = p[2] * (1.0 + np.cos(p[0] * phi - p[1]))
        
        print(f"  n={p[0]} Manual={e_man:.4f} OpenMM={e_iso:.4f} Diff={e_iso - e_man:.4f}")

if __name__ == "__main__":
    debug_torsion()
