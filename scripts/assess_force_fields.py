import glob
import os
import xml.etree.ElementTree as ET

# Units (same as convert script)
KJ_TO_KCAL = 0.239005736
NM_TO_ANGSTROM = 10.0

UNSUPPORTED_TAGS = [
    "VirtualSite",
    "TwoParticleAverageSite",
    "ThreeParticleAverageSite",
    "LocalCoordinatesSite",
    "DrudeForce",
    "AmoebaBondForce",
    "AmoebaAngleForce",
    "AmoebaInPlaneAngleForce",
    "AmoebaTorsionForce",
    "AmoebaPiTorsionForce",
    "AmoebaStretchBendForce",
    "AmoebaOpBendForce",
    "AmoebaTorsionTorsionForce",
    "AmoebaMultipoleForce",
    "AmoebaGeneralizedKirkwoodForce",
    "AmoebaVdwForce",
    "AmoebaWcaDispersionForce",
    "NoseHooverIntegrator",
    "AndersenThermostat",
    "MonteCarloBarostat"
]

def assess_xml(xml_path):
    ff_name = os.path.basename(xml_path)
    report = {
        "file": ff_name,
        "path": xml_path,
        "unsupported_features": [],
        "parsing_error": None,
        "status": "PASS"
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.tag != "ForceField":
            report["status"] = "SKIP (Not ForceField)"
            return report

        # 1. Check for unsupported tags
        for elem in root.iter():
            if elem.tag in UNSUPPORTED_TAGS or elem.tag.startswith("Amoeba"):
                if elem.tag not in report["unsupported_features"]:
                    report["unsupported_features"].append(elem.tag)

        # 2. Attempt basic parsing (mimic convert_all_xmls.py logic)
        # We don't need to actually build the full model, just see if access patterns fail

        # Atom Types
        type_to_class = {t.attrib["name"]: t.attrib.get("class", t.attrib["name"]) for t in root.findall("AtomTypes/Type")}

        # LJ Parameters
        for a in root.findall("NonbondedForce/Atom"):
            # Some FFs might use different attributes, but standard is 'type' or 'class'
            pass

        # Residues
        for res in root.findall("Residues/Residue"):
            res_atom_names = []
            for atom in res.findall("Atom"):
                atom_name = atom.attrib["name"]
                atom_type = atom.attrib.get("type")

                # Check if atom type exists (a common failure point if logic assumes it does)
                # But actual converter is lenient here.

            # Bonds in residue
            for bond in res.findall("Bond"):
                if "from" in bond.attrib:
                    idx1 = int(bond.attrib["from"])
                    idx2 = int(bond.attrib["to"])
                    # Check range
                    # if idx1 >= len(res_atom_names): error

        # Harmonic Bond
        for b in root.findall("HarmonicBondForce/Bond"):
             # Just checking if attributes exist
             _ = float(b.attrib["length"])
             _ = float(b.attrib["k"])

        # Harmonic Angle
        for a in root.findall("HarmonicAngleForce/Angle"):
            _ = float(a.attrib["angle"])
            _ = float(a.attrib["k"])

        # Periodic Torsion
        for t in root.findall("PeriodicTorsionForce/Proper"):
            # Check for k, periodicity, phase
            for i in range(1, 7):
                if f"periodicity{i}" in t.attrib:
                     _ = int(t.attrib[f"periodicity{i}"])
                     _ = float(t.attrib[f"phase{i}"])
                     _ = float(t.attrib[f"k{i}"])

        # CMAP
        for c in root.findall("CMAPTorsionForce/Map"):
            # Check grid size validity
            energy_elem = c.find("Energy")
            if energy_elem is not None:
                txt = energy_elem.text
            else:
                txt = c.text

            if txt:
                vals = txt.split()
                size = int(len(vals)**0.5)
                if size*size != len(vals):
                    raise ValueError(f"Invalid CMAP grid size: {len(vals)}")

    except Exception as e:
        report["status"] = "FAIL"
        report["parsing_error"] = str(e)
        # report["traceback"] = traceback.format_exc() # Optional: too verbose for summary

    if report["unsupported_features"]:
        report["status"] = "WARN" if report["status"] == "PASS" else report["status"]

    return report

def main():
    target_patterns = [
        "openmmforcefields/openmmforcefields/ffxml/amber/*ff14SB*.xml",
        "openmmforcefields/openmmforcefields/ffxml/amber/*ff99SB*.xml",
        "openmmforcefields/openmmforcefields/ffxml/charmm/*charmm36*.xml"
    ]

    files = []
    for pattern in target_patterns:
        files.extend(glob.glob(pattern, recursive=True))

    files = sorted(list(set(files))) # Unique files

    print(f"Assesssing {len(files)} files...\n")
    print(f"{'File':<40} | {'Status':<10} | {'Unsupported / Error'}")
    print("-" * 100)

    results = []

    for f in files:
        res = assess_xml(f)
        results.append(res)

        info = ""
        if res["status"] == "FAIL":
            info = f"ERROR: {res['parsing_error']}"
        elif res["unsupported_features"]:
            info = f"UNSUPPORTED: {', '.join(res['unsupported_features'])}"

        print(f"{res['file']:<40} | {res['status']:<10} | {info}")

    # Summary
    print("\n--- Summary ---")
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    print(f"Total: {len(results)}")
    print(f"PASS: {pass_count}")
    print(f"WARN (Unsupported Features): {warn_count}")
    print(f"FAIL (Parsing Errors): {fail_count}")

if __name__ == "__main__":
    main()
