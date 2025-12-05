import os
import sys
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import equinox as eqx

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.convert_all_xmls import parse_xml_to_eqx
from priox.physics import force_fields

# Path to XML
xml_path = os.path.abspath("openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml")
# Update output dir to point to the local priox clone
output_dir = "priox/src/priox/physics/force_fields/eqx"

if not os.path.exists(xml_path):
    print(f"Error: XML not found at {xml_path}")
    sys.exit(1)

print(f"Regenerating protein19SB.eqx from {xml_path}...")
parse_xml_to_eqx(xml_path, output_dir)

# Now load and update with radii
ff_path = os.path.join(output_dir, "protein19SB.eqx")
print(f"Loading {ff_path} to update radii...")
ff = force_fields.load_force_field(ff_path)

print("Extracting GBSA radii from OpenMM...")
# Load FF with implicit solvent
try:
    omm_ff = app.ForceField(xml_path, 'implicit/obc2.xml')
except Exception:
    omm_ff = app.ForceField(xml_path, os.path.join(os.path.dirname(xml_path), 'implicit', 'obc2.xml'))

# Build a system with all residues to extract radii
radii_map = {} # (res, atom) -> (r, s)

# Get all templates
templates = omm_ff._templates
print(f"Found {len(templates)} templates in OpenMM FF.")

def create_system_for_res(res_name, cap_n=False, cap_c=False):
    topo = app.Topology()
    chain = topo.addChain()
    
    # Add ACE if needed
    if cap_n:
        ace = topo.addResidue("ACE", chain)
        ace_template = templates["ACE"]
        ace_atoms = {}
        for atom in ace_template.atoms:
            a = topo.addAtom(atom.name, app.Element.getBySymbol(atom.element.symbol) if atom.element else None, ace)
            ace_atoms[atom.name] = a
        for b in ace_template.bonds:
             if b[0] < len(ace_template.atoms) and b[1] < len(ace_template.atoms):
                topo.addBond(ace_atoms[ace_template.atoms[b[0]].name], ace_atoms[ace_template.atoms[b[1]].name])

    # Add Target Residue
    res = topo.addResidue(res_name, chain)
    template = templates[res_name]
    atoms = {}
    for atom in template.atoms:
        a = topo.addAtom(atom.name, app.Element.getBySymbol(atom.element.symbol) if atom.element else None, res)
        atoms[atom.name] = a
    for b in template.bonds:
        if b[0] < len(template.atoms) and b[1] < len(template.atoms):
            topo.addBond(atoms[template.atoms[b[0]].name], atoms[template.atoms[b[1]].name])

    # Add NME if needed
    if cap_c:
        nme = topo.addResidue("NME", chain)
        nme_template = templates["NME"]
        nme_atoms = {}
        for atom in nme_template.atoms:
            a = topo.addAtom(atom.name, app.Element.getBySymbol(atom.element.symbol) if atom.element else None, nme)
            nme_atoms[atom.name] = a
        for b in nme_template.bonds:
             if b[0] < len(nme_template.atoms) and b[1] < len(nme_template.atoms):
                topo.addBond(nme_atoms[nme_template.atoms[b[0]].name], nme_atoms[nme_template.atoms[b[1]].name])

    # Add Inter-residue bonds
    # ACE-Res
    if cap_n:
        # ACE C connects to Res N
        if "C" in ace_atoms and "N" in atoms:
            topo.addBond(ace_atoms["C"], atoms["N"])
            
    # Res-NME
    if cap_c:
        # Res C connects to NME N
        if "C" in atoms and "N" in nme_atoms:
            topo.addBond(atoms["C"], nme_atoms["N"])
            
    return topo

for res_name, template in templates.items():
    # Skip caps themselves for main loop (handled inside)
    if res_name in ["ACE", "NME"]:
        # We still want their radii, but they are special.
        # We can extract them from any system that uses them.
        pass

    # Try strategies
    strategies = [
        (True, True),   # ACE-Res-NME (Standard)
        (False, True),  # Res-NME (N-term)
        (True, False),  # ACE-Res (C-term)
        (False, False)  # Res (Ion/Cap)
    ]
    
    success = False
    for cap_n, cap_c in strategies:
        try:
            topo = create_system_for_res(res_name, cap_n, cap_c)
            system = omm_ff.createSystem(topo, nonbondedMethod=app.NoCutoff, constraints=None)
            
            # Extract Radii
            gb_force = None
            for f in system.getForces():
                if isinstance(f, openmm.GBSAOBCForce):
                    gb_force = f
                    break
                elif isinstance(f, openmm.CustomGBForce):
                    gb_force = f
                    break
            
            if gb_force:
                # Find the target residue in the topology
                # It's the 2nd residue if cap_n is True, else 1st
                target_res_idx = 1 if cap_n else 0
                # But if we failed to add ACE (e.g. if res is ACE), indices shift.
                # Actually, we construct topo explicitly.
                
                # Iterate atoms and find those belonging to res_name
                for i, atom in enumerate(topo.atoms()):
                    if atom.residue.name == res_name:
                        if isinstance(gb_force, openmm.GBSAOBCForce):
                            c, r, s = gb_force.getParticleParameters(i)
                            r_val = r.value_in_unit(unit.angstrom)
                            s_val = s
                        else:
                            params = gb_force.getParticleParameters(i)
                            # CustomGBForce (OBC2) stores:
                            # param 0: charge
                            # param 1: offset_radius (nm) = radius - offset
                            # param 2: scaled_radius (nm) = scale * (radius - offset)
                            
                            # We need intrinsic radius: r_val = (param 1 * 10) + 0.09
                            r_val = params[1] * 10.0 + 0.09
                            
                            # We need scale factor: s_val = param 2 / param 1
                            if abs(params[1]) > 1e-6:
                                s_val = params[2] / params[1]
                            else:
                                s_val = 0.0
                        
                        radii_map[(res_name, atom.name)] = (r_val, s_val)
                
                success = True
                break
        except Exception:
            continue
            
    if not success:
        print(f"  Failed to extract radii for {res_name}")

# Handle ACE/NME specifically if missed
if ("ACE", "C") not in radii_map:
    # Create ACE-NME
    try:
        topo = create_system_for_res("ACE", False, True) # ACE-NME
        system = omm_ff.createSystem(topo, nonbondedMethod=app.NoCutoff, constraints=None)
        # Extract...
        # (Simplified for brevity, assuming main loop catches most)
        pass
    except: pass

print(f"Extracted radii for {len(radii_map)} atoms.")

# Update Force Field
radii_list = []
scales_list = []
miss_count = 0

for i, key in enumerate(ff.id_to_atom_key):
    # key is (res, atom)
    # Handle tuple/list conversion if needed
    if isinstance(key, list): key = tuple(key)
    
    res, atom = key
    if (res, atom) in radii_map:
        r, s = radii_map[(res, atom)]
        radii_list.append(r)
        scales_list.append(s)
    else:
        # Try mapping terminals (NALA -> ALA)
        core_res = res
        if len(res) == 4 and res.startswith('N'): core_res = res[1:]
        if len(res) == 4 and res.startswith('C'): core_res = res[1:]
        
        if (core_res, atom) in radii_map:
             r, s = radii_map[(core_res, atom)]
             radii_list.append(r)
             scales_list.append(s)
        else:
            # print(f"Missing radii for {res} {atom}")
            radii_list.append(0.0)
            scales_list.append(0.0)
            miss_count += 1

print(f"Updated FF with {len(radii_list)} radii. Missing: {miss_count}")

new_ff = eqx.tree_at(
    lambda f: (f.radii_by_id, f.scales_by_id),
    ff,
    (jnp.array(radii_list), jnp.array(scales_list))
)

force_fields.save_force_field(ff_path, new_ff)
print(f"Saved updated force field to {ff_path}")

