
import jax.numpy as jnp
import pytest

from prolix.physics import solvation


def test_add_ions():
    # Mock system: 1 Solute atom (Charge -2), box of waters
    solute_pos = jnp.array([[10.0, 10.0, 10.0]])
    solute_charge = -2.0

    # Create fake waters (100 waters)
    # We need to construct positions for 300 atoms.
    # Just uniform grid
    waters = []
    for i in range(100):
        o = [float(i), 0.0, 0.0]
        h1 = [float(i), 0.0, 1.0]
        h2 = [float(i), 0.0, -1.0]
        waters.extend([o, h1, h2])

    water_pos = jnp.array(waters)
    positions = jnp.concatenate([solute_pos, water_pos])

    # Indices: Solute is 0. Waters are 1..300.
    water_indices = jnp.arange(1, 301)

    box_size = jnp.array([100.0, 100.0, 100.0]) # Huge box

    # 1. Neutralization Check
    # Should add 2 Positive Ions (Na+)
    new_pos, atom_names, res_names = solvation.add_ions(
        positions, water_indices, solute_charge,
        neutralize=True, ionic_strength=0.0
    )

    # Check counts
    # Original: 1 + 300 = 301 atoms.
    # New: 1 + (98 * 3) + (2 * 1) = 1 + 294 + 2 = 297 atoms.
    assert len(new_pos) == 297

    # Check topology
    # atom_names should be length 296 (waters/ions only)
    assert len(atom_names) == 296

    # Count NA
    na_count = atom_names.count("NA")
    assert na_count == 2

    # 2. Salt Concentration Check
    # 0.15 M.
    # Box 100 A = 10 nm. V = 1000 nm^3.
    # 1 L = 1e24 nm^3.
    # V(L) = 1e-21.
    # N = 0.15 * 1e-21 * 6e23 = 0.15 * 600 = 90 ions.
    # Wait, box 100A is huge?
    # 100A = 10nm.
    # 10x10x10 nm^3 = 1000 nm^3.
    # 1 L = 10^24 nm^3.
    # V = 1e-21 L.
    # N = C * V * N_A = 0.15 * 1e-21 * 6.022e23 = 0.15 * 602.2 = 90.3 -> 90.

    # Note: add_ions adds salt PAIRS. So +90, -90.
    # Plus neutralization (2 NA).
    # Total NA: 92. Total CL: 90.

    # Need enough waters. 100 waters is not enough for 182 ions.
    # Let's reduce box size or calc expected failure.

    with pytest.raises(ValueError, match="Not enough waters"):
        solvation.add_ions(
            positions, water_indices, solute_charge,
            neutralize=True, ionic_strength=0.15, box_size=box_size
        )

    # Valid concentration
    # Smaller box: 10x10x10 A.
    # V = 1e-27 L.
    # N = 0.15 * 1e-27 * 6e23 ~ 0.
    # Let's try ridiculous concentration for 100 waters.
    # Or just use count check.

    # box 40A.
    # V = 64000 A^3 = 6.4e-23 L.
    # N = 0.15 * 6.4e-23 * 6e23 = 5.78 -> 6 pairs.

    box_small = jnp.array([40.0, 40.0, 40.0])
    new_pos, atom_names, res_names = solvation.add_ions(
        positions, water_indices, solute_charge,
        neutralize=True, ionic_strength=0.15, box_size=box_small
    )

    na_count = atom_names.count("NA")
    cl_count = atom_names.count("CL")
    print(f"NA: {na_count}, CL: {cl_count}")

    # Neutralize: +2 NA.
    # Salt: +6 NA, +6 CL.
    # Total: 8 NA, 6 CL.
    assert na_count == 8
    assert cl_count == 6

if __name__ == "__main__":
    test_add_ions()
