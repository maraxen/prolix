import os
import proxide
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat
import jax.numpy as jnp

PDB_PATH = "data/pdb/1UAO.pdb"
FF_XML_PATH = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")

def inspect_proxide_output():
    spec = OutputSpec(
        coord_format=CoordFormat.Full, 
        add_hydrogens=True, 
        parameterize_md=True, 
        force_field=FF_XML_PATH
    )
    # Note: 1UAO.pdb might need cleaning or just use it if it exists
    if not os.path.exists(PDB_PATH):
        print(f"PDB not found at {PDB_PATH}")
        return

    protein = parse_structure(PDB_PATH, spec=spec)
    
    print(f"Proper Dihedrals shape: {protein.proper_dihedrals.shape}")
    print(f"Dihedral Params shape: {protein.dihedral_params.shape}")
    
    print(f"Impropers shape: {protein.impropers.shape}")
    print(f"Improper Params shape: {protein.improper_params.shape}")

    try:
        print(f"All Dihedrals shape: {protein.dihedrals.shape}")
    except:
        print("protein.dihedrals not available or has no shape")
    
    if len(protein.dihedral_params) > 0:
        print(f"First dihedral param: {protein.dihedral_params[0]}")
        print(f"First dihedral param type: {type(protein.dihedral_params[0])}")
        try:
            print(f"First dihedral param shape: {protein.dihedral_params[0].shape}")
        except:
            pass

if __name__ == "__main__":
    inspect_proxide_output()
