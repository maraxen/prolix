
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from prolix.physics.solvation import load_tip3p_box


def test_load():
    print("Attempting to load water box...")
    box = load_tip3p_box()
    print("Success!")
    print(f"Number of atoms: {box.positions.shape[0]}")
    print(f"Box size: {box.box_size}")

    if box.positions.shape[0] > 0 and box.box_size[0] > 0:
        print("Verification passed.")
    else:
        print("Verification failed: Empty box.")
        sys.exit(1)

if __name__ == "__main__":
    test_load()
