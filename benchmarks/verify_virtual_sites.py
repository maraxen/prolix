import os

import jax.numpy as jnp
from proxide.physics.force_fields import load_force_field

from prolix.physics import virtual_sites

FF_PATH = "data/force_fields/waters_ions_tip4p.eqx"

def test_tip4p_vs():
    if not os.path.exists(FF_PATH):
        print(f"Skipping VS test: {FF_PATH} not found.")
        return

    print("Loading TIP4P force field...")
    ff = load_force_field(FF_PATH)

    # Check if TP4 or TIP4P residue exists and has VS
    # Grep showed "TP4" or similar?
    # Let's inspect ff.virtual_sites
    vs_dict = ff.virtual_sites
    print(f"Virtual Sites found for residues: {list(vs_dict.keys())}")

    # Assuming TIP4P residue name is 'TP4' or similar from XML
    # In XML convert log: "Found 1 Virtual Sites in SWM4" (likely TIP4P-like?)
    # "Found 1 Virtual Sites in TP4E"
    # "Found 1 Virtual Sites in TP45"

    res_name = None
    if "TP4" in vs_dict: res_name = "TP4"
    elif "TIP4" in vs_dict: res_name = "TIP4"
    elif "TP4E" in vs_dict: res_name = "TP4E"
    elif "TP45" in vs_dict: res_name = "TP45"
    elif "SWM4" in vs_dict: res_name = "SWM4"

    if res_name is None:
        print("No standard TIP4P residue found in FF virtual_sites.")
        return

    print(f"Testing VS for residue: {res_name}")
    vs_data_list = vs_dict[res_name]
    vs_data = vs_data_list[0]

    # Extract params
    # vs_data is dict: {type, siteName, atoms, p, wo, wx, wy}
    # Create arrays
    vs_def = jnp.array([[0, 0, 1, 2]], dtype=jnp.int32) # vs_idx=0 (overlap O?), parents=0,1,2 (O,H,H)
    # Wait, vs usually separate particle?
    # In 'localCoords', the vs is a particle that *moves*.
    # TIP4P has 4 particles: O, H, H, M.
    # O is 0, H1 is 1, H2 is 2, M is 3.
    # The M site depends on O, H, H.

    # So vs_idx should be 3.
    # Parents: 0 (O), 1 (H1), 2 (H2).
    vs_def = jnp.array([[3, 0, 1, 2]], dtype=jnp.int32)

    p = vs_data["p"] # list
    wo = vs_data["wo"]
    wx = vs_data["wx"]
    wy = vs_data["wy"]

    params_flat = jnp.array(p + wo + wx + wy, dtype=jnp.float32)
    vs_params = params_flat[None, :] # (1, 12)

    # Define Positions (Standard water geometry)
    # O at origin
    # H1 at (0.9572, 0, 0) (Angstrom) - temporary
    # H2 at (x, y, 0) with angle 104.52

    d_OH = 0.9572
    angle = 104.52 * jnp.pi / 180.0

    r_O = jnp.array([0.0, 0.0, 0.0])
    r_H1 = jnp.array([d_OH, 0.0, 0.0])
    r_H2 = jnp.array([d_OH * jnp.cos(angle), d_OH * jnp.sin(angle), 0.0])
    r_M = jnp.array([0.0, 0.0, 0.0]) # Initial dummy

    coords = jnp.stack([r_O, r_H1, r_H2, r_M])

    print("Initial Coords:")
    print(coords)

    # Reconstruct
    new_coords = virtual_sites.reconstruct_virtual_sites(coords, vs_def, vs_params)

    print("\nReconstructed Coords:")
    print(new_coords)

    # Verify M position
    m_pos = new_coords[3]
    print(f"\nM site position: {m_pos}")

    # For TIP4P, M is on bisector of H-O-H, moved towards H?
    # Actually M is usually shifted along bisector from O.
    # d_OM is ~0.15 A (0.015 nm).

    # Calculate bisector
    v1 = r_H1 - r_O
    v2 = r_H2 - r_O
    bisector = (v1 + v2)
    bisector = bisector / jnp.linalg.norm(bisector)

    # Project M onto bisector
    # In local coords frame:
    # Origin O (weights 1,0,0)
    # X, Y defined by weighted sums?
    # TIP4P VS typical: p1=d_OM (along bisector/local Z?), p2=0, p3=0?
    # Let's see computed params.

    print(f"\nVS Params (p): {p}")
    print(f"VS Params (wo): {wo}")
    print(f"VS Params (wx): {wx}")
    print(f"VS Params (wy): {wy}")

    # Distance O-M
    dist_OM = jnp.linalg.norm(m_pos - r_O)
    print(f"Distance O-M: {dist_OM:.5f} A")

    # Check against p value (usually stored in param p1)
    if abs(dist_OM - abs(p[0])) < 0.01:
         print(f"PASS: Distance matches p1 parameter ({p[0]})")
    else:
         print(f"CHECK: Distance {dist_OM} vs param {p[0]}")

    # Check if on bisector (approx) or in plane
    # Assuming planar molecule, Z component should be 0
    if abs(m_pos[2]) < 1e-4:
        print("PASS: M site is planar (Z ~ 0)")
    else:
        print(f"FAIL: M site out of plane (Z={m_pos[2]})")

if __name__ == "__main__":
    test_tip4p_vs()
