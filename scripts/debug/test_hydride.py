from priox.io.parsing import biotite
import biotite.structure.io.pdb as pdb
import os

PDB_PATH = "data/pdb/1UBQ.pdb"
if not os.path.exists(PDB_PATH):
    print(f"File not found: {PDB_PATH}")
    # Try absolute path or check where we are
    print(f"CWD: {os.getcwd()}")
else:
    print(f"Loading structure from {PDB_PATH}...")
    try:
        atom_array = biotite.load_structure_with_hydride(PDB_PATH, model=1)
        print(f"Loaded {len(atom_array)} atoms.")
    except Exception as e:
        print(f"Error: {e}")
