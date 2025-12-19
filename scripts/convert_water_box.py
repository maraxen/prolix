
import os

import numpy as np


def convert_pdb_to_npz(pdb_path: str, output_path: str):
    print(f"Converting {pdb_path} to {output_path}...")

    positions = []
    box_size = np.array([0.0, 0.0, 0.0])

    with open(pdb_path) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("CRYST1"):
            # CRYST1   30.000   30.000   30.000 ...
            box_size[0] = float(line[6:15])
            box_size[1] = float(line[15:24])
            box_size[2] = float(line[24:33])
        elif line.startswith("ATOM"):
            # ATOM      1  O   HOH A   1       4.125  13.679 ...
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            positions.append([x, y, z])

    positions_array = np.array(positions, dtype=np.float32)
    box_size_array = np.array(box_size, dtype=np.float32)

    np.savez(output_path, positions=positions_array, box_size=box_size_array)
    print(f"Saved {len(positions_array)} atoms with box {box_size_array}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdb_path = os.path.join(base_dir, "data", "water_boxes", "tip3p.pdb")
    output_path = os.path.join(base_dir, "data", "water_boxes", "tip3p.npz")

    if not os.path.exists(pdb_path):
        print(f"Error: {pdb_path} not found.")
    else:
        convert_pdb_to_npz(pdb_path, output_path)
