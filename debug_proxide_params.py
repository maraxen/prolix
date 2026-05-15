import os
import proxide
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat
import jax.numpy as jnp

PDB_PATH = "data/pdb/1UAO.pdb"
FF_XML_PATH = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")

def inspect_sigmas():
    spec = OutputSpec(
        coord_format=CoordFormat.Full, 
        add_hydrogens=True, 
        parameterize_md=True, 
        force_field=FF_XML_PATH
    )
    protein = parse_structure(PDB_PATH, spec=spec)
    
    print(f"Num atoms: {protein.num_atoms}")
    print(f"First 10 sigmas: {protein.sigmas[:10]}")
    print(f"First 10 epsilons: {protein.epsilons[:10]}")
    print(f"First 10 charges: {protein.charges[:10]}")
    
    # Check atom names to see what they are
    print(f"First 10 atom names: {protein.atom_names[:10]}")

if __name__ == "__main__":
    inspect_sigmas()
