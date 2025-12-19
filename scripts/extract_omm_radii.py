"""Extract GBSA radii from OpenMM for all atom classes used in protein.ff19SB.
"""
import os
import xml.etree.ElementTree as ET

import openmm
from openmm import app, unit


def extract_radii():
    print("Extracting GBSA radii from OpenMM...")

    # We need an ff19SB XML + implicit solvent
    xml_path = "openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"

    ff = app.ForceField(xml_path, "implicit/obc2.xml")

    # Parse ff19SB to get all atom types/classes
    tree = ET.parse(xml_path)
    root = tree.getroot()

    type_to_class = {}
    for t in root.findall("AtomTypes/Type"):
        type_to_class[t.attrib["name"]] = t.attrib["class"]

    print(f"Found {len(type_to_class)} atom types.")

    # We need to build a system with all residues to get all radii.
    # Let's use a simple polypeptide covering common residues.

    # Build a minimal system for each residue type
    residue_names = set()
    for res in root.findall("Residues/Residue"):
        residue_names.add(res.attrib["name"])

    print(f"Found {len(residue_names)} residue types.")

    # We'll collect radii by atom class
    class_to_radii = {}

    # Use Ala as a simple test

    # Build a simple ALA tripeptide
    # We can use the forcefield's templates directly
    # But it's easier to use a known PDB

    pdb_path = "data/pdb/1UAO.pdb"
    if not os.path.exists(pdb_path):
        print(f"PDB not found at {pdb_path}")
        return

    pdb = app.PDBFile(pdb_path)
    topology = pdb.topology
    positions = pdb.positions

    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None
    )

    # Find the GBSAOBCForce or CustomGBForce
    gb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.GBSAOBCForce) or isinstance(force, openmm.CustomGBForce):
            gb_force = force
            break

    if gb_force is None:
        print("No GBSA force found!")
        return

    print(f"Found {gb_force.__class__.__name__} with {gb_force.getNumParticles()} particles.")

    # Extract radii
    atoms = list(topology.atoms())

    for i in range(gb_force.getNumParticles()):
        atom = atoms[i]
        res_name = atom.residue.name
        atom_name = atom.name

        # Get atom class
        # The ff19SB uses special terminal residues (N/C prefixed)
        # For now, we'll just use res_name + atom_name as key

        if isinstance(gb_force, openmm.GBSAOBCForce):
            charge, radius, scale = gb_force.getParticleParameters(i)
            radius_A = radius.value_in_unit(unit.angstrom)
            scale_val = scale
        else:
            # CustomGBForce - params depend on definition
            params = gb_force.getParticleParameters(i)
            # Usually (charge, or, sr) for OBC
            radius_A = params[1] * 10.0  # nm -> A
            scale_val = params[2]

        key = f"{res_name}_{atom_name}"
        if key not in class_to_radii:
            class_to_radii[key] = (radius_A, scale_val)

    print(f"Extracted {len(class_to_radii)} unique (res, atom) radii.")

    # Print mean
    radii_vals = [r for r, s in class_to_radii.values()]
    mean_r = sum(radii_vals) / len(radii_vals)
    print(f"Mean radius: {mean_r:.4f} A")

    # Compare to current proxide values
    from proxide.physics import force_fields
    ff_eqx = force_fields.load_force_field("data/force_fields/protein19SB.eqx")

    print("\nComparing to existing force field...")
    if hasattr(ff_eqx, "radii_by_id"):
        print("Force field has radii_by_id")
    else:
        print("Force field MISSING radii_by_id")

    # See if gb_radii is set elsewhere?
    # Check jax_md_bridge
    print("\nDone.")

if __name__ == "__main__":
    extract_radii()
