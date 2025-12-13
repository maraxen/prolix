# File: scripts/convert_all_xmls.py
import os
import glob
import json
import xml.etree.ElementTree as ET
import jax.numpy as jnp
import equinox as eqx
from proxide.physics.force_fields import FullForceField, save_force_field
from proxide.physics.force_fields.components import (
    AtomTypeParams,
    BondPotentialParams,
    AnglePotentialParams,
    DihedralPotentialParams,
    CMAPParams,
    UreyBradleyParams,
    VirtualSiteParams,
    NonbondedGlobalParams,
    GAFFNonbondedParams,
)

# Units
KJ_TO_KCAL = 0.239005736
NM_TO_ANGSTROM = 10.0

def parse_xml_to_eqx(xml_path: str, output_dir: str):
    ff_name = os.path.basename(xml_path).replace('.xml', '').replace('.ff', '')
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Validate it's a force field file
        if root.tag != 'ForceField':
            return # Skip non-forcefield files silently

        print(f"--- Processing {ff_name} ---")
        
        # --- 0. GAFF Type Parsing (New) ---
        gaff_types = {}  # type -> (sigma, epsilon)
        for atom in root.findall('NonbondedForce/Atom'):
            atom_class = atom.attrib.get('class')
            # Fallback to 'type' if class not present, but for GAFF usually class is the type key
            if not atom_class and 'type' in atom.attrib:
                atom_class = atom.attrib['type']
                
            if atom_class:
                sigma_nm = float(atom.attrib.get('sigma', 1.0))
                epsilon_kj = float(atom.attrib.get('epsilon', 0.0))
                
                sigma = sigma_nm * NM_TO_ANGSTROM
                epsilon = epsilon_kj * KJ_TO_KCAL
                
                gaff_types[atom_class.lower()] = (sigma, epsilon)
        
        print(f"DEBUG: Found {len(gaff_types)} GAFF atom types with LJ params.")

        # --- 0.5 Global Nonbonded Parameters ---
        # Defaults
        nb_globals = {
            "coulomb14scale": 0.833333,
            "lj14scale": 0.5,
        }
        nb_force = root.find('NonbondedForce')
        if nb_force is not None:
             if 'coulomb14scale' in nb_force.attrib:
                 nb_globals["coulomb14scale"] = float(nb_force.attrib['coulomb14scale'])
             if 'lj14scale' in nb_force.attrib:
                 nb_globals["lj14scale"] = float(nb_force.attrib['lj14scale'])

        # --- 1. Basic Parameter Parsing ---
        type_to_class = {t.attrib['name']: t.attrib['class'] for t in root.findall('AtomTypes/Type')}
        print(f"DEBUG: Type to Class Sample: {list(type_to_class.items())[:5]}")
        
        class_to_lj = {}
        for a in root.findall('NonbondedForce/Atom'):
            key = a.attrib.get('class', a.attrib.get('type'))
            if key:
                class_to_lj[key] = (float(a.attrib.get('sigma', 1.0)), float(a.attrib.get('epsilon', 0.0)))

        # Initialize Data containers
        charges, sigmas, epsilons = [0.0], [1.0], [0.0]
        hyperparams = {
            "atom_key_to_id": {("UNK", "UNK"): 0}, "id_to_atom_key": [("UNK", "UNK")],
            "atom_class_map": {}, "atom_type_map": {}, "source_files": [os.path.basename(xml_path)],
            "bonds": [], "angles": [], "propers": [], "impropers": [], "cmap_torsions": [],
            "residue_templates": {}
        }
        current_id = 1

        # Parse Residues & Atoms
        for res in root.findall('Residues/Residue'):
            res_name = res.attrib['name']
            
            # Track atoms in this residue for bond mapping (index -> name)
            res_atom_names = []
            
            for atom in res.findall('Atom'):
                atom_name, atom_type = atom.attrib['name'], atom.attrib['type']
                res_atom_names.append(atom_name)
                
                key = (res_name, atom_name)
                
                if key not in hyperparams["atom_key_to_id"]:
                    atom_cls = type_to_class.get(atom_type, atom_type)
                    sigma, epsilon = class_to_lj.get(atom_cls, (1.0, 0.0))
                    
                    hyperparams["atom_key_to_id"][key] = current_id
                    hyperparams["id_to_atom_key"].append(key)
                    hyperparams["atom_class_map"][f"{res_name}_{atom_name}"] = atom_cls
                    hyperparams["atom_type_map"][f"{res_name}_{atom_name}"] = atom_type
                    charges.append(float(atom.attrib.get('charge', 0.0)))
                    sigmas.append(sigma * NM_TO_ANGSTROM)
                    epsilons.append(epsilon * KJ_TO_KCAL)
                    current_id += 1
            
            # Extract Residue Internal Bonds (Templates)
            res_bonds = []
            for bond in res.findall('Bond'):
                if 'from' in bond.attrib and 'to' in bond.attrib:
                    idx1 = int(bond.attrib['from'])
                    idx2 = int(bond.attrib['to'])
                    if idx1 < len(res_atom_names) and idx2 < len(res_atom_names):
                        name1 = res_atom_names[idx1]
                        name2 = res_atom_names[idx2]
                        res_bonds.append((name1, name2))
                elif 'atomName1' in bond.attrib and 'atomName2' in bond.attrib:
                    name1 = bond.attrib['atomName1']
                    name2 = bond.attrib['atomName2']
                    res_bonds.append((name1, name2))
            
            if res_bonds:
                hyperparams["residue_templates"][res_name] = res_bonds
        
        # Parse Standard Terms
        for b in root.findall('HarmonicBondForce/Bond'):
             c1 = b.attrib.get('class1', b.attrib.get('type1'))
             c2 = b.attrib.get('class2', b.attrib.get('type2'))
             if c1 and c2:
                 # Convert length nm -> A
                 length_nm = float(b.attrib['length'])
                 length_a = length_nm * NM_TO_ANGSTROM
                 
                 # Convert k kJ/mol/nm^2 -> kcal/mol/A^2
                 # k_kcal_A2 = k_kJ_nm2 * KJ_TO_KCAL / (NM_TO_ANGSTROM**2)
                 k_kj_nm2 = float(b.attrib['k'])
                 k_kcal_a2 = k_kj_nm2 * KJ_TO_KCAL / (NM_TO_ANGSTROM**2)
                 
                 hyperparams["bonds"].append((c1, c2, length_a, k_kcal_a2))
                 if len(hyperparams["bonds"]) == 1:
                     print(f"DEBUG: First Bond Converted: {c1}-{c2} L={length_a} k={k_kcal_a2}")

        for a in root.findall('HarmonicAngleForce/Angle'):
            c1, c2, c3 = [a.attrib.get(f'class{i}') or a.attrib.get(f'type{i}') for i in range(1, 4)]
            if c1 and c2 and c3:
                # Convert k kJ/mol/rad^2 -> kcal/mol/rad^2
                k_kj_rad2 = float(a.attrib['k'])
                k_kcal_rad2 = k_kj_rad2 * KJ_TO_KCAL
                
                hyperparams["angles"].append((c1, c2, c3, float(a.attrib['angle']), k_kcal_rad2))
        
        # Helper for torsions
        def parse_tor(tag_name):
            data = []
            for t in root.findall(f'PeriodicTorsionForce/{tag_name}'):
                classes = []
                for i in range(1, 5):
                    c = t.attrib.get(f'class{i}')
                    if c is None:
                        c = t.attrib.get(f'type{i}')
                    classes.append(c)
                
                if any(c is None for c in classes): continue
                terms = []
                for i in range(1, 7):
                    if f'periodicity{i}' in t.attrib:
                        terms.append((int(t.attrib[f'periodicity{i}']), float(t.attrib[f'phase{i}']), float(t.attrib[f'k{i}']) * KJ_TO_KCAL))
                if terms: 
                    data.append({'classes': tuple(classes), 'terms': terms})
                    if tag_name == 'Proper' and len(data) <= 100:
                        print(f"DEBUG: Proper Torsion Classes: {classes}")
                    
                    if tag_name == 'Proper':
                        c_tuple = tuple(classes)
                        if c_tuple[1] == 'protein-N' and c_tuple[2] == 'protein-CA':
                            print(f"DEBUG: Parsed *-N-CA-* Torsion: {classes} -> {terms}")
                        if set(classes) == {'protein-C', 'protein-XC', 'protein-CT', 'protein-CA'}:
                             print(f"DEBUG: Parsed C-XC-CT-CA Torsion (Any order): {classes} -> Term Count: {len(terms)}")
                             for t_val in terms:
                                 print(f"  Term: {t_val}")
            return data

        hyperparams['propers'] = parse_tor('Proper')
        hyperparams['impropers'] = parse_tor('Improper')

        # --- 1.5 Urey-Bradley Parsing (AmoebaUreyBradleyForce) ---
        ub_terms = []
        for ub_force in root.findall('AmoebaUreyBradleyForce'):
            for ub in ub_force.findall('UreyBradley'):
                c1 = ub.attrib.get('class1', ub.attrib.get('type1'))
                c2 = ub.attrib.get('class2', ub.attrib.get('type2'))
                if c1 and c2:
                    d_nm = float(ub.attrib['d'])
                    d_a = d_nm * NM_TO_ANGSTROM
                    k_kj_nm2 = float(ub.attrib['k'])
                    k_kcal_a2 = k_kj_nm2 * KJ_TO_KCAL / (NM_TO_ANGSTROM**2)
                    ub_terms.append((c1, c2, d_a, k_kcal_a2))
        
        hyperparams["urey_bradley_bonds"] = ub_terms
        if ub_terms:
            print(f"DEBUG: Parsed {len(ub_terms)} Urey-Bradley terms.")

        # --- 1.6 Virtual Site Parsing ---
        if "virtual_sites" not in hyperparams:
            hyperparams["virtual_sites"] = {}

        for res in root.findall('Residues/Residue'):
            res_name = res.attrib['name']
            vs_list = []
            for vs in res.findall('VirtualSite'):
                if vs.attrib.get('type') == 'localCoords':
                    try:
                        data = {
                            'type': 'localCoords',
                            'siteName': vs.attrib['siteName'], 
                            'atoms': [vs.attrib['atomName1'], vs.attrib['atomName2'], vs.attrib['atomName3']],
                            'p': [float(vs.attrib['p1']), float(vs.attrib['p2']), float(vs.attrib['p3'])], # nm
                            'wo': [float(vs.attrib.get('wo1',0)), float(vs.attrib.get('wo2',0)), float(vs.attrib.get('wo3',0))],
                            'wx': [float(vs.attrib.get('wx1',0)), float(vs.attrib.get('wx2',0)), float(vs.attrib.get('wx3',0))],
                            'wy': [float(vs.attrib.get('wy1',0)), float(vs.attrib.get('wy2',0)), float(vs.attrib.get('wy3',0))]
                        }
                        # Convert p from nm to Angstrom
                        data['p'] = [x * NM_TO_ANGSTROM for x in data['p']]
                        vs_list.append(data)
                    except Exception as e:
                        print(f"  WARNING: Failed to parse VirtualSite in {res_name}: {e}")
            
            if vs_list:
                hyperparams["virtual_sites"][res_name] = vs_list
                if len(hyperparams["virtual_sites"]) <= 5: 
                     print(f"  DEBUG: Found {len(vs_list)} Virtual Sites in {res_name}")

        # --- 2. CMAP Parsing ---
        cmap_grids = []
        for cmap_force in root.findall('CMAPTorsionForce'):
            # Parse Maps
            for map_tag in cmap_force.findall('Map'):
                energy_elem = map_tag.find('Energy')
                if energy_elem is not None:
                    energy_str = energy_elem.text.strip()
                else:
                    energy_str = map_tag.text.strip()
                    
                values = [float(x) for x in energy_str.split()]
                size = int(len(values)**0.5)
                if size * size != len(values):
                    print(f"  WARNING: Invalid CMAP grid size in {ff_name}, skipping map.")
                    continue
                    
                # Convert kJ/mol -> kcal/mol
                grid = jnp.array(values, dtype=jnp.float32).reshape(size, size) * KJ_TO_KCAL
                cmap_grids.append(grid)
                if len(cmap_grids) == 1:
                    print(f"DEBUG: First CMAP Grid Converted. Max Val: {jnp.max(grid)}")
            
            # Parse Torsions (Class mappings)
            for t in cmap_force.findall('Torsion'):
                classes = [t.attrib.get(f'class{i}') or t.attrib.get(f'type{i}') for i in range(1, 6)]
                if all(classes):
                    hyperparams["cmap_torsions"].append({
                        'classes': tuple(classes),
                        'map_index': int(t.attrib['map'])
                    })

        if cmap_grids:
            cmap_energy_grids = jnp.stack(cmap_grids)
        else:
            # Default empty grid if no CMAP
            cmap_energy_grids = jnp.zeros((0, 24, 24), dtype=jnp.float32)

        # --- GBSA Radii (Skipped for valid reasons discussed) ---
        radii_vals = [0.0] * len(charges)
        scale_vals = [0.0] * len(charges)
        
        # --- Create Components ---
        atom_params = AtomTypeParams(
            charges=jnp.asarray(charges),
            sigmas=jnp.asarray(sigmas),
            epsilons=jnp.asarray(epsilons),
            radii=jnp.asarray(radii_vals),
            scales=jnp.asarray(scale_vals),
            atom_key_to_id=hyperparams["atom_key_to_id"],
            id_to_atom_key=hyperparams["id_to_atom_key"],
            atom_class_map=hyperparams["atom_class_map"],
            atom_type_map=hyperparams["atom_type_map"]
        )

        bond_params = BondPotentialParams(params=hyperparams["bonds"])
        angle_params = AnglePotentialParams(params=hyperparams["angles"])
        dihedral_params = DihedralPotentialParams(
            propers=hyperparams["propers"], 
            impropers=hyperparams["impropers"]
        )
        cmap_params = CMAPParams(
            energy_grids=cmap_energy_grids,
            torsions=hyperparams["cmap_torsions"]
        )
        urey_bradley_params = UreyBradleyParams(params=hyperparams["urey_bradley_bonds"])
        virtual_site_params = VirtualSiteParams(definitions=hyperparams["virtual_sites"])

        global_params = NonbondedGlobalParams(
            coulomb14scale=nb_globals["coulomb14scale"],
            lj14scale=nb_globals["lj14scale"]
        )

        # Create GAFF Params if data exists
        gaff_params = None
        if gaff_types:
            sorted_types = sorted(gaff_types.keys())
            type_to_index = {t: i for i, t in enumerate(sorted_types)}
            
            sig_arr = [gaff_types[t][0] for t in sorted_types]
            eps_arr = [gaff_types[t][1] for t in sorted_types]
            
            gaff_params = GAFFNonbondedParams(
                type_to_index=type_to_index,
                sigmas=jnp.array(sig_arr, dtype=jnp.float32),
                epsilons=jnp.array(eps_arr, dtype=jnp.float32)
            )

        # Create and Save
        model = FullForceField(
            atom_params=atom_params,
            bond_params=bond_params,
            angle_params=angle_params,
            dihedral_params=dihedral_params,
            cmap_params=cmap_params,
            urey_bradley_params=urey_bradley_params,
            virtual_site_params=virtual_site_params,
            global_params=global_params,
            gaff_nonbonded_params=gaff_params,
            residue_templates=hyperparams["residue_templates"],
            source_files=hyperparams["source_files"]
        )

        output_path = os.path.join(output_dir, f"{ff_name}.eqx")
        save_force_field(output_path, model)
        print(f"✅ Saved {output_path}")

    except Exception as e:
        print(f"🛑 Error processing {ff_name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not os.path.exists("openmmforcefields"):
        print("Cloning repo...")
        os.system("git clone --depth 1 https://github.com/openmm/openmmforcefields.git")
    
    output_dir = "data/force_fields"
    os.makedirs(output_dir, exist_ok=True)
    
    # Recursive search for ANY xml file
    xml_files = glob.glob("openmmforcefields/openmmforcefields/**/*.xml", recursive=True)
    print(f"Found {len(xml_files)} potential XML files.")
    
    for xml in xml_files:
        if "gaff-2.11" not in xml: continue
        print(f"DEBUG: Explicitly processing {xml}")
        parse_xml_to_eqx(xml, output_dir)
