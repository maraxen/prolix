import jax
import jax.numpy as jnp
import equinox as eqx
from proxide.physics import force_fields

def debug_torsion_matching():
    ff_path = "data/force_fields/protein19SB.eqx"
    print(f"Loading Force Field from {ff_path}...")
    
    # Use load_force_field which handles JSON header + EQX body
    ff = force_fields.load_force_field(ff_path)
    
    # Check TRP HH2 class
    res = "TRP"
    atom = "HH2"
    key = f"{res}_{atom}"
    cls = ff.atom_class_map.get(key, "MISSING")
    print(f"Class for {res} {atom}: {cls}")
    
    # Check TRP CH2 class
    atom = "CH2"
    key = f"{res}_{atom}"
    cls = ff.atom_class_map.get(key, "MISSING")
    print(f"Class for {res} {atom}: {cls}")
    
    # Check TRP CZ3 class
    atom = "CZ3"
    key = f"{res}_{atom}"
    cls = ff.atom_class_map.get(key, "MISSING")
    print(f"Class for {res} {atom}: {cls}")
    
    # Check TRP CE3 class
    atom = "CE3"
    key = f"{res}_{atom}"
    cls = ff.atom_class_map.get(key, "MISSING")
    print(f"Class for {res} {atom}: {cls}")
    
    # Check what classes are in propers
    print("\nChecking propers for 'protein-HA' or 'HA'...")
    has_protein_ha = False
    has_ha = False
    sample_proper = None
    
    for proper in ff.propers:
        pc = proper["classes"]
        if "protein-HA" in pc:
            has_protein_ha = True
        if "HA" in pc:
            has_ha = True
            sample_proper = pc
            
    print(f"Has 'protein-HA': {has_protein_ha}")
    print(f"Has 'HA': {has_ha}")
    if sample_proper:
        print(f"Sample 'HA' proper: {sample_proper}")

    # Target path: CE3(CA) - CZ3(CA) - CH2(CA) - HH2(HA)
    # Classes: protein-CA - protein-CA - protein-CA - protein-HA
    target_classes = ["protein-CA", "protein-CA", "protein-CA", "protein-HA"]
    print(f"\nLooking for match for: {target_classes}")
    
    matches = []
    for proper in ff.propers:
        pc = proper["classes"]
        # Check forward
        match_fwd = True
        if (pc[0] != "" and pc[0] != target_classes[0]) or \
           (pc[1] != "" and pc[1] != target_classes[1]) or \
           (pc[2] != "" and pc[2] != target_classes[2]) or \
           (pc[3] != "" and pc[3] != target_classes[3]):
            match_fwd = False
            
        # Check backward
        match_bwd = True
        if (pc[0] != "" and pc[0] != target_classes[3]) or \
           (pc[1] != "" and pc[1] != target_classes[2]) or \
           (pc[2] != "" and pc[2] != target_classes[1]) or \
           (pc[3] != "" and pc[3] != target_classes[0]):
            match_bwd = False
            
        if match_fwd or match_bwd:
            matches.append(proper)
            
    print(f"Found {len(matches)} matches.")
    for m in matches:
        print(f"  Match: {m['classes']} -> {m['terms']}")

if __name__ == "__main__":
    debug_torsion_matching()
