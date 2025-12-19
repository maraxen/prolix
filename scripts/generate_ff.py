
import os

from convert_all_xmls import parse_xml_to_eqx
from openmm import app


def generate_ff():
    # Find amber14/protein.ff14SB.xml
    app_dir = os.path.dirname(app.__file__)
    data_dir = os.path.join(app_dir, "data")

    xmls_to_convert = [
        os.path.join(data_dir, "amber14", "protein.ff14SB.xml"),
        # OpenMM might not ship ff19SB by default in 'data/amber14'?
        # It's usually in openmmforcefields package if installed.
    ]

    # Check openmmforcefields for ff19SB
    try:
        import openmmforcefields
        ff_dir = os.path.dirname(openmmforcefields.__file__)
        ff19sb_path = os.path.join(ff_dir, "ffxml", "amber", "protein.ff19SB.xml")
        if os.path.exists(ff19sb_path):
            xmls_to_convert.append(ff19sb_path)
        else:
            print(f"Warning: protein.ff19SB.xml not found at {ff19sb_path}")
    except ImportError:
        print("openmmforcefields not installed, checking for local clone...")
        if not os.path.exists("openmmforcefields"):
            print("Cloning openmmforcefields repo...")
            os.system("git clone --depth 1 https://github.com/openmm/openmmforcefields.git")

        ff19sb_path = os.path.join("openmmforcefields", "openmmforcefields", "ffxml", "amber", "protein.ff19SB.xml")
        if os.path.exists(ff19sb_path):
            xmls_to_convert.append(ff19sb_path)
        else:
            print(f"Warning: protein.ff19SB.xml not found at {ff19sb_path}")

    output_dir = "../../proxide/src/proxide/physics/force_fields/eqx"
    os.makedirs(output_dir, exist_ok=True)

    for xml_path in xmls_to_convert:
        if not os.path.exists(xml_path):
            print(f"Error: {xml_path} not found.")
            continue

        print(f"Converting {xml_path}...")
        parse_xml_to_eqx(xml_path, output_dir)

        # Rename if needed
        ff_name = os.path.basename(xml_path).replace(".xml", "").replace(".ff", "")
        if ff_name == "protein14SB":
             # Create alias ff14SB.eqx
             src = os.path.join(output_dir, "protein14SB.eqx")
             dst = os.path.join(output_dir, "ff14SB.eqx")
             if os.path.exists(src):
                 import shutil
                 shutil.copy(src, dst)
                 print(f"Created alias {dst}")

if __name__ == "__main__":
    generate_ff()
