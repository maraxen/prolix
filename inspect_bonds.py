import os
import jax.numpy as jnp
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat

def inspect_bonds():
    import proxide
    ff_xml_path = os.path.join(os.path.dirname(proxide.__file__), "assets", "protein.ff19SB.xml")
    pdb_path = "data/pdb/1UAO.pdb"
    spec = OutputSpec(coord_format=CoordFormat.Full, add_hydrogens=False, parameterize_md=True, force_field=ff_xml_path)
    system = parse_structure(pdb_path, spec=spec)
    
    print(f"Total bonds: {len(system.bonds)}")
    print(f"First 20 bonds: {system.bonds[:20]}")
    
    # Check atom 0 neighbors
    adj = []
    for b in system.bonds:
        if b[0] == 0: adj.append(b[1])
        if b[1] == 0: adj.append(b[0])
    print(f"Atom 0 neighbors: {adj}")

if __name__ == "__main__":
    inspect_bonds()
