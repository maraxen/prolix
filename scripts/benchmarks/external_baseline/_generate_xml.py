"""Shared utility: prolix molecule JSON → OpenMM-compatible XML files.

Used by DMFF, TorchMD (XML→YAML), and espaloma harnesses as the common
input-generation layer. Not a benchmark itself.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

log = logging.getLogger(__name__)

_ELEMENT_SYMBOLS = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}

_ATOMIC_MASSES = {
    1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    15: 30.974, 16: 32.06, 17: 35.45, 35: 79.904, 53: 126.904,
}


def generate_xml_for_mol(json_path: Path, xml_path: Path) -> None:
    """Read one prolix JSON, write one OpenMM XML."""
    with open(json_path) as f:
        mol = json.load(f)

    ff = ET.Element("ForceField")

    atom_types_el = ET.SubElement(ff, "AtomTypes")
    for atom in mol["atom_types"]:
        idx = atom["idx"]
        Z = atom["Z"]
        symbol = _ELEMENT_SYMBOLS.get(Z, "C")
        mass = _ATOMIC_MASSES.get(Z, 12.0)
        ET.SubElement(atom_types_el, "Type", {
            "name": f"A{idx}",
            "class": f"A{idx}",
            "element": symbol,
            "mass": str(mass),
        })

    residues_el = ET.SubElement(ff, "Residues")
    residue_el = ET.SubElement(residues_el, "Residue", {"name": "MOL"})
    for atom in mol["atom_types"]:
        idx = atom["idx"]
        ET.SubElement(residue_el, "Atom", {"name": f"A{idx}", "type": f"A{idx}"})
    for bond in mol["bonds"]:
        ET.SubElement(residue_el, "Bond", {
            "atomName1": f"A{bond['i']}",
            "atomName2": f"A{bond['j']}",
        })

    bond_force_el = ET.SubElement(ff, "HarmonicBondForce")
    for bond in mol["bonds"]:
        length_nm = bond["r0"] * 0.1
        k_kJ_per_nm2 = bond["k"] * 4.184 * 100 * 2
        ET.SubElement(bond_force_el, "Bond", {
            "type1": f"A{bond['i']}",
            "type2": f"A{bond['j']}",
            "length": f"{length_nm:.8f}",
            "k": f"{k_kJ_per_nm2:.6f}",
        })

    angle_force_el = ET.SubElement(ff, "HarmonicAngleForce")
    for angle in mol["angles"]:
        angle_rad = angle["theta0_deg"] * math.pi / 180.0
        k_kJ_per_rad2 = angle["k_theta"] * 4.184 * 2
        ET.SubElement(angle_force_el, "Angle", {
            "type1": f"A{angle['i']}",
            "type2": f"A{angle['j']}",
            "type3": f"A{angle['k']}",
            "angle": f"{angle_rad:.8f}",
            "k": f"{k_kJ_per_rad2:.6f}",
        })

    torsion_force_el = ET.SubElement(ff, "PeriodicTorsionForce")
    for tor in mol.get("proper_torsions", []):
        periodicities = tor["periodicity"]
        phases_deg = tor["phase_deg"]
        k_phis = tor["k_phi"]
        for n, phase_deg, k_phi in zip(periodicities, phases_deg, k_phis):
            phase_rad = phase_deg * math.pi / 180.0
            k_kJ = k_phi * 4.184
            ET.SubElement(torsion_force_el, "Proper", {
                "type1": f"A{tor['i']}",
                "type2": f"A{tor['j']}",
                "type3": f"A{tor['k']}",
                "type4": f"A{tor['l']}",
                "periodicity1": str(n),
                "phase1": f"{phase_rad:.8f}",
                "k1": f"{k_kJ:.6f}",
            })

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    raw = ET.tostring(ff, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent=" ", encoding=None)
    # minidom adds an XML declaration line; strip it for cleaner output
    lines = pretty.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    xml_path.write_text("\n".join(lines) + "\n")


def generate_xml_dir(subset_dir: Path, xml_dir: Path, lane: str = "lane_a") -> list[Path]:
    """Process all mol_*.params_init.json in subset_dir/lane/, write XMLs to xml_dir/.

    Returns list of written XML paths sorted by mol index.
    """
    lane_dir = subset_dir / lane
    json_paths = sorted(lane_dir.glob("mol_*.params_init.json"))
    written: list[Path] = []
    for json_path in json_paths:
        with open(json_path) as f:
            mol_id = json.load(f)["molecule_id"]
        xml_path = xml_dir / f"{mol_id}.xml"
        generate_xml_for_mol(json_path, xml_path)
        log.debug("wrote %s", xml_path)
        written.append(xml_path)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenMM XML files from prolix molecule JSON files."
    )
    parser.add_argument("--subset-dir", type=Path, required=True,
                        help="Root subset directory (e.g. data/ani1x_subset)")
    parser.add_argument("--xml-dir", type=Path, required=True,
                        help="Output directory for XML files")
    parser.add_argument("--lane", default="lane_a",
                        help="Lane subdirectory name (default: lane_a)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(message)s")
    written = generate_xml_dir(args.subset_dir, args.xml_dir, lane=args.lane)
    print(f"generated {len(written)} XMLs → {args.xml_dir}")


if __name__ == "__main__":
    main()
