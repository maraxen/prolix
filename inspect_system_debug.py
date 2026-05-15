import os
import pickle
import jax.numpy as jnp
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat

def inspect_system():
    # Use the same setup as debug_parity_gaps.py
    import proxide
    ff_xml_path = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")
    
    # Use a small PDB (Chignolin)
    pdb_path = "benchmarks/chignolin.pdb"
    if not os.path.exists(pdb_path):
        # Fallback to any pdb in data
        pdb_path = "data/pdb/1UAO.pdb"
        
    spec = OutputSpec(coord_format=CoordFormat.Full, add_hydrogens=False, parameterize_md=True, force_field=ff_xml_path)
    system = parse_structure(pdb_path, spec=spec)
    
    print(f"--- System Inspection ---")
    print(f"Attributes: {dir(system)}")
    
    if hasattr(system, "coords"):
        print(f"Coords shape: {system.coords.shape}")
    
    for attr in ["bonds", "angles", "dihedrals", "proper_dihedrals", "impropers", "improper_dihedrals", "excl_indices"]:
        val = getattr(system, attr, None)
        if val is not None:
            if hasattr(val, "shape"):
                print(f"{attr}: {val.shape}")
            else:
                print(f"{attr}: {type(val)}")
        else:
            print(f"{attr}: None")
            
    if hasattr(system, "dihedral_params") and system.dihedral_params is not None:
        print(f"dihedral_params: {system.dihedral_params.shape}")
    if hasattr(system, "proper_dihedral_params") and system.proper_dihedral_params is not None:
        print(f"proper_dihedral_params: {system.proper_dihedral_params.shape}")

if __name__ == "__main__":
    inspect_system()
