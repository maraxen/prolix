import os
import sys
import numpy as np
from termcolor import colored

# OpenMM Imports
try:
    import openmm
    from openmm import app, unit
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# Prolix/Proxide Imports
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat
import proxide

# Configuration
PDB_PATH = "data/pdb/1UAO.pdb"
import proxide
FF_XML_PATH = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")

def verify_torsions():
    print(colored("===========================================================", "cyan"))
    print(colored("   Verifying Torsion Multi-Definition Parity", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Load Structure with OpenMM (using PDBFixer to match Prolix test suite exactly)
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=PDB_PATH)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, tmp)
        tmp_path = tmp.name

    # 2. Setup OpenMM System
    print(colored("\n[1] Setting up OpenMM System...", "yellow"))
    pdb_file = app.PDBFile(tmp_path)
    topology = pdb_file.topology
    omm_ff = app.ForceField(FF_XML_PATH, "implicit/obc2.xml")
    omm_system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff)

    omm_torsions = {}
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                t = tuple(sorted((p1, p2, p3, p4)))
                
                if t not in omm_torsions: omm_torsions[t] = []
                omm_torsions[t].append({"period": period, "k": k.value_in_unit(unit.kilocalories_per_mole)})

    print(f"OpenMM Total Torsion Terms: {sum(len(v) for v in omm_torsions.values())}")
    print(f"OpenMM Unique Torsion Quads: {len(omm_torsions)}")

    # 3. Setup JAX MD System via Proxide
    print(colored("\n[2] Setting up JAX MD System...", "yellow"))
    spec = OutputSpec(coord_format=CoordFormat.Full, add_hydrogens=False, parameterize_md=True, force_field=FF_XML_PATH)
    protein = parse_structure(tmp_path, spec=spec)
    os.unlink(tmp_path)

    jax_torsions = {}
    
    def process_dihs(dihs, params):
        if dihs is not None:
            for i in range(len(dihs)):
                p1, p2, p3, p4 = map(int, dihs[i])
                period, phase, k = map(float, params[i])
                t = tuple(sorted((p1, p2, p3, p4)))
                
                if t not in jax_torsions: jax_torsions[t] = []
                jax_torsions[t].append({"period": int(period), "k": k})

    process_dihs(protein.proper_dihedrals, protein.dihedral_params)
    process_dihs(protein.impropers, protein.improper_params)

    print(f"JAX Total Torsion Terms: {sum(len(v) for v in jax_torsions.values())}")
    print(f"JAX Unique Torsion Quads: {len(jax_torsions)}")
    
    if protein.dihedral_params is not None:
        print(f"Raw JAX proper_dihedrals count: {len(protein.proper_dihedrals)}")
        print(f"Raw JAX dihedral_params count: {len(protein.dihedral_params)}")

    # 4. Compare
    print(colored("\n[3] Comparing...", "yellow"))
    omm_quads = set(omm_torsions.keys())
    jax_quads = set(jax_torsions.keys())
    
    missing_quads = omm_quads - jax_quads
    extra_quads = jax_quads - omm_quads
    
    total_missing_terms = 0
    total_extra_terms = 0
    
    for q in missing_quads:
        total_missing_terms += len(omm_torsions[q])
    for q in extra_quads:
        total_extra_terms += len(jax_torsions[q])
        
    mismatch_count = 0
    for q in (omm_quads & jax_quads):
        if len(omm_torsions[q]) != len(jax_torsions[q]):
            mismatch_count += 1
            if len(jax_torsions[q]) < len(omm_torsions[q]):
                total_missing_terms += (len(omm_torsions[q]) - len(jax_torsions[q]))
            else:
                total_extra_terms += (len(jax_torsions[q]) - len(omm_torsions[q]))

    print(f"Quads unique to OpenMM: {len(missing_quads)}")
    print(f"Quads unique to JAX: {len(extra_quads)}")
    print(f"Quads with Term Count Mismatch: {mismatch_count}")
    print(f"Total Missing Terms in JAX: {total_missing_terms}")
    print(f"Total Extra Terms in JAX: {total_extra_terms}")

    if missing_quads:
        print("\nExamples of quads missing in JAX (With Atom Names):")
        atom_metadata = []
        for i in range(len(protein.atom_names)):
            atom_metadata.append(f"{protein.residue_index[i]}:{protein.atom_names[i]}")

        for q in list(missing_quads)[:5]:
            q_names = [atom_metadata[i] if i < len(atom_metadata) else f"ERR:{i}" for i in q]
            print(f"  {q} ({'-'.join(q_names)}): OpenMM terms={omm_torsions[q]}")
            
    if extra_quads:
        print("\nExamples of extra quads in JAX (Indices):")
        for q in list(extra_quads)[:5]:
            print(f"  {q}: JAX terms={jax_torsions[q]}")

    if total_missing_terms == 0:
        print(colored("\nSUCCESS: No missing torsion terms in JAX!", "green"))
    else:
        print(colored(f"\nFAILURE: {total_missing_terms} terms missing in JAX.", "red"))

if __name__ == "__main__":
    verify_torsions()
