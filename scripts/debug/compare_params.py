import os
import openmm.app as app
import openmm
import jax.numpy as jnp
import numpy as np
from proxide import parse_structure, OutputSpec

def compare_parameters():
    pdb_path = "data/pdb/1UAO.pdb"
    ff_xml = "amber14/protein.ff14SB.xml"
    solv_xml = "implicit/obc2.xml"
    
    # 1. OpenMM Setup
    pdb = app.PDBFile(pdb_path)
    omm_ff = app.ForceField(ff_xml, solv_xml)
    omm_system = omm_ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    
    nb_force = [f for f in omm_system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    gb_force = [f for f in omm_system.getForces() if f.__class__.__name__ in ["CustomGBForce", "GBSAOBC2Force"] or "GB" in f.__class__.__name__][0]
    
    # 2. JAX Setup (via Proxide)
    # We need to find the absolute path for proxide to work correctly in this environment
    import openmm.app as app_mod
    base_path = os.path.dirname(app_mod.__file__)
    ff_path = os.path.join(base_path, "data", ff_xml)
    
    spec = OutputSpec(parameterize_md=True, force_field=ff_path, add_hydrogens=False)
    protein = parse_structure(pdb_path, spec)
    
    print("=== PARAMETER PARITY CHECK (1UAO) ===")
    
    # --- TORSIONS ---
    # OpenMM torsions
    torsion_force = [f for f in omm_system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)][0]
    omm_torsion_count = torsion_force.getNumTorsions()
    
    jax_dih = protein.dihedrals if protein.dihedrals is not None else protein.proper_dihedrals
    jax_torsion_count = len(jax_dih) if jax_dih is not None else 0
    print(f"Torsions: OpenMM={omm_torsion_count}, JAX={jax_torsion_count}")
    
    # --- NONBONDED EXCEPTIONS ---
    omm_exceptions = {}
    for i in range(nb_force.getNumExceptions()):
        p1, p2, q, s, e = nb_force.getExceptionParameters(i)
        pair = tuple(sorted((p1, p2)))
        omm_exceptions[pair] = (s.value_in_unit(openmm.unit.nanometer), e.value_in_unit(openmm.unit.kilojoule_per_mole))
        
    print(f"Exclusions/Exceptions: OpenMM Total={len(omm_exceptions)}")
    
    # Check for Proline 1-4s (often indices 88-93 or similar)
    # Let's see if any OMM exception has epsilon=0 where JAX might have it > 0
    zero_eps_omm = [pair for pair, params in omm_exceptions.items() if params[1] == 0]
    print(f"OpenMM Zero-Epsilon Exceptions: {len(zero_eps_omm)}")
    
    # --- GB RADII ---
    omm_radii = []
    for i in range(gb_force.getNumParticles()):
        params = gb_force.getParticleParameters(i)
        # CustomGBForce parameters are floats
        omm_radii.append(params[1])
    
    omm_radii = np.array(omm_radii)
    jax_radii = np.array(protein.radii) if protein.radii is not None else np.array([])
    
    print(f"GB Radii: OMM shape={omm_radii.shape}, JAX shape={jax_radii.shape}")
    if jax_radii.size > 0:
        print(f"  OMM (first 5): {omm_radii[:5]}")
        print(f"  JAX (first 5): {jax_radii[:5]}")
    
    # --- EXCLUSION SCALES ---
    jax_excl_vdw = np.array(protein.scale_matrix_vdw) if hasattr(protein, "scale_matrix_vdw") else None
    print(f"JAX scale_matrix_vdw: {'Present' if jax_excl_vdw is not None else 'Missing'}")
    
    if hasattr(protein, "excl_scales_vdw"):
        print(f"JAX excl_scales_vdw present")

if __name__ == "__main__":
    compare_parameters()
