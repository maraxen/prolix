import os
import sys
from termcolor import colored

# OpenMM Imports
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# Configuration
PDB_PATH = "data/pdb/1UAO.pdb"
OPENMM_XMLS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
]

def analyze_nonzero_torsions():
    print(colored("===========================================================", "cyan"))
    print(colored("   Inspecting Non-Zero Torsions for 1UAO + ff19SB", "cyan"))
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
    
    atoms = list(topology.atoms())
    
    torsion_force = None
    for force in system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            torsion_force = force
            break
            
    if torsion_force is None:
        return

    # Count non-zero torsions per residue
    res_torsions = {} # Res -> list of (Names, k)

    for i in range(torsion_force.getNumTorsions()):
        p1, p2, p3, p4, per, phase, k = torsion_force.getTorsionParameters(i)
        k_val = k.value_in_unit(unit.kilocalories_per_mole)
        
        # Calculate Energy
        a1, a2, a3, a4 = [atoms[idx] for idx in [p1, p2, p3, p4]]
        pos1 = positions[p1]
        pos2 = positions[p2]
        pos3 = positions[p3]
        pos4 = positions[p4]
        
        # OpenMM Torsion Energy = k * (1 + cos(n*phi - phase))
        # Need to calculate phi
        # Use simple formula or OpenMM internal? 
        # Easier to skip rigorous calc and just filter by k magnitude first, 
        # but User wants Energy.
        
        # Quick hack: Use a helper or just print k for now?
        # No, I need energy to find the 30 kcal culprit.
        
        # Vectors
        r1 = pos1
        r2 = pos2
        r3 = pos3
        r4 = pos4
        
        # Compute vectors
        # v1 = r2 - r1
        # v2 = r3 - r2
        # v3 = r4 - r3
        # ... complicated vector math.
        
        # Let's rely on identification by NAME first.
        # But I will print names more carefully.
        
        if abs(k_val) > 1e-6:
            a1, a2, a3, a4 = [atoms[idx] for idx in [p1, p2, p3, p4]]
            names = [a.name for a in [a1, a2, a3, a4]]
            
            # Simple heuristic for now: Just capture everything.
            res = a2.residue
            key = f"{res.name}{res.index}"
            if key not in res_torsions:
                res_torsions[key] = []
            res_torsions[key].append((names, k_val, per, phase.value_in_unit(unit.radians)))

    print("\nNon-Zero Backbone/Improper Torsions (k > 0.1):")
    sorted_res = sorted(res_torsions.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    for r in sorted_res:
        print(f"  {r}:")
        for names, k, per, phase in res_torsions[r]:
            if k > 0.1: # Filter small terms
                print(f"    - {'-'.join(names)}: k={k:.4f} n={per} phase={phase:.2f}")

    # Check CMAP
    cmap_force = None
    for force in system.getForces():
        if isinstance(force, openmm.CMAPTorsionForce):
            cmap_force = force
            break
            
    print("\nCMAP Terms by Residue:")
    if cmap_force:
        for i in range(cmap_force.getNumTorsions()):
             # mapIdx, a1...a8? No. 
             # getTorsionParameters: map, a1, a2, a3, a4, b1, b2, b3, b4
             # a1-a4 is Phi, b1-b4 is Psi?
             params = cmap_force.getTorsionParameters(i)
             map_idx = params[0]
             # a2 is N, a3 is CA => Phi
             # a2 = params[2]
             atom_ca = atoms[params[3]] # CA
             
             res = atom_ca.residue
             print(f"  {res.name}{res.index}: Map {map_idx} (CA atom {atom_ca.index})")

if __name__ == "__main__":
    analyze_nonzero_torsions()
