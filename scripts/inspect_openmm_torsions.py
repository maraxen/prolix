import os
import sys

from termcolor import colored

# OpenMM Imports
try:
    import openmm
    from openmm import app, unit
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# Configuration
PDB_PATH = "data/pdb/1UAO.pdb"
# Use relative path to protein.ff19SB.xml found in openmmforcefields
OPENMM_XMLS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
]

def analyze_torsions():
    print(colored("===========================================================", "cyan"))
    print(colored("   Inspecting OpenMM Torsions for 1UAO + ff19SB", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Load Structure
    print(colored("\n[1] Loading Structure...", "yellow"))
    pdb = app.PDBFile(PDB_PATH)
    topology = pdb.topology
    positions = pdb.positions

    # 2. Setup OpenMM System
    print(colored("\n[2] Setting up OpenMM System...", "yellow"))
    ff = app.ForceField(*OPENMM_XMLS)
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False
    )

    # 3. Analyze Torsions
    print(colored("\n[3] Analyzing Torsions...", "yellow"))

    # Map atom index to atom object for easy lookup
    atoms = list(topology.atoms())

    torsion_force = None
    for force in system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            torsion_force = force
            break

    if torsion_force is None:
        print("No PeriodicTorsionForce found!")
        return

    print(f"Total Torsions: {torsion_force.getNumTorsions()}")

    # Store torsions by atoms key
    torsion_map = {} # (i,j,k,l) -> list of (per, phase, k)

    for i in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, per, phase, k = torsion_force.getTorsionParameters(i)

        # Canonicalize key
        if p1 > p4:
            key = (p4, p3, p2, p1)
        else:
            key = (p1, p2, p3, p4)

        if key not in torsion_map:
            torsion_map[key] = []

        torsion_map[key].append({
            "periodicity": per,
            "phase": phase.value_in_unit(unit.degrees), # degrees for easier reading
            "k": k.value_in_unit(unit.kilocalories_per_mole)
        })

    # Find interesting torsions (e.g., C-N-CA-C)
    # Iterate through all gathered torsions and check atom names
    found_interesting = False

    # We'll look for C-N-CA-C specifically
    target_pattern = ["C", "N", "CA", "C"]

    for key, params in torsion_map.items():
        indices = list(key)
        names = [atoms[idx].name for idx in indices]

        # Check match (forward or reverse)
        is_match = False
        if names == target_pattern or names == target_pattern[::-1]:
            is_match = True

        if is_match or len(params) > 1: # Print MULTI-TERM torsions generally to see what's happening
            a1, a2, a3, a4 = [atoms[idx] for idx in indices]

            # Get Residue info
            res_str = f"{a1.residue.name}{a1.residue.index}"

            atom_str = f"{a1.name}-{a2.name}-{a3.name}-{a4.name}"
            print(f"\nTorsion: {atom_str} (Res: {res_str}) Indices: {indices}")
            print(f"  Atom Types (names): {names}")
            # Note: We don't easily get atom TYPES (classes) from OpenMM API after creation without inspecting Template/ForceField wrapper.
            # But we can see what was applied.

            for p in params:
                 print(f"    - Periodicity: {p['periodicity']}, Phase: {p['phase']:.2f}, k: {p['k']:.4f}")

            if is_match:
                found_interesting = True

    if not found_interesting:
        print("No C-N-CA-C torsions found!")

if __name__ == "__main__":
    analyze_torsions()
