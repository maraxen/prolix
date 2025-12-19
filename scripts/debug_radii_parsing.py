
import dataclasses
import os
import xml.etree.ElementTree as ET

from openmm import app
from proxide.physics.force_fields import FullForceField


def test_parse_radii():
    # 1. Find obc2.xml
    print("Looking for obc2.xml...")
    obc2_path = None
    try:
        data_dir = os.path.join(os.path.dirname(app.__file__), "data")
        candidate = os.path.join(data_dir, "implicit", "obc2.xml")
        if os.path.exists(candidate):
            obc2_path = candidate
    except Exception as e:
        print(f"Error finding openmm data: {e}")

    if not obc2_path:
        print("Could not find obc2.xml in standard locations.")
        return

    print(f"Found obc2.xml at {obc2_path}")

    tree = ET.parse(obc2_path)
    root = tree.getroot()

    print(f"Parsing {obc2_path}...")
    print(f"Root Tag: {root.tag}")
    print(f"Root Children: {[child.tag for child in root]}")

    radii_map = {} # class -> radius

    # Check GBSAOBCForce
    for force in root.findall("GBSAOBCForce"):
        print("Found GBSAOBCForce")
        atoms = force.findall("Atom")
        print(f"  Found {len(atoms)} atoms.")
        for i, atom in enumerate(atoms):
            if i < 5:
                print(f"  Atom {i} attributes: {atom.attrib}")

            cls = atom.attrib.get("class", atom.attrib.get("type"))
            radius = float(atom.attrib["radius"])
            scale = float(atom.attrib["scale"])
            if cls:
                radii_map[cls] = (radius, scale)

    print(f"Extracted {len(radii_map)} radii classes.")
    for k in list(radii_map.keys())[:5]:
        print(f"  {k}: {radii_map[k]}")

    # 2. Inspect FullForceField
    print("\nInspecting FullForceField...")
    fields = dataclasses.fields(FullForceField)
    field_names = [f.name for f in fields]
    print(f"Fields: {field_names}")

    if "radii_by_id" in field_names:
        print("SUCCESS: FullForceField already has 'radii_by_id'.")
    else:
        print("FAILURE: FullForceField MISSING 'radii_by_id'. Must modify proxide source.")

if __name__ == "__main__":
    test_parse_radii()
