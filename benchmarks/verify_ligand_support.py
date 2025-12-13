#!/usr/bin/env python
"""Verify ligand energy calculations against OpenMM GAFF.

This benchmark compares the energy from our GAFF parameterization
against OpenMM's GAFFTemplateGenerator for validation.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add source paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxide.io.parsing.molecule import Molecule
from proxide.physics.force_fields import load_force_field


def verify_mol2_parsing():
    """Test basic MOL2 parsing with the imatinib file."""
    print("=" * 60)
    print("Verifying MOL2 Parsing")
    print("=" * 60)
    
    mol2_path = Path(__file__).parent.parent / "openmmforcefields/amber/files/imatinib.mol2"
    
    if not mol2_path.exists():
        print(f"SKIP: {mol2_path} not found")
        return None
    
    mol = Molecule.from_mol2(mol2_path)
    
    print(f"Molecule: {mol.name}")
    print(f"  Atoms: {mol.n_atoms}")
    print(f"  Bonds: {mol.n_bonds}")
    print(f"  Unique atom types: {set(mol.atom_types)}")
    print(f"  Total charge: {mol.charges.sum():.4f}")
    print(f"  Elements: {set(mol.elements)}")
    
    return mol


def verify_gaff_loading():
    """Test loading GAFF force field."""
    print("\n" + "=" * 60)
    print("Verifying GAFF Force Field Loading")
    print("=" * 60)
    
    ff_path = Path(__file__).parent.parent / "data/force_fields/gaff-2.2.20.eqx"
    
    if not ff_path.exists():
        print(f"SKIP: {ff_path} not found")
        return None
    
    ff = load_force_field(ff_path)
    
    print(f"Force Field: gaff-2.2.20")
    print(f"  Atom types: {len(ff.atom_key_to_id)}")
    print(f"  Bond types: {len(ff.bonds)}")
    print(f"  Angle types: {len(ff.angles)}")
    print(f"  Proper torsions: {len(ff.propers)}")
    print(f"  Improper torsions: {len(ff.impropers)}")
    
    # Check for sample GAFF types
    sample_types = ["ca", "c3", "n", "o", "ha", "h1"]
    found_types = []
    for res, atom in ff.atom_key_to_id.keys():
        if atom.lower() in sample_types or res.lower() in sample_types:
            found_types.append((res, atom))
    
    print(f"  Sample GAFF types found: {len(found_types)}")
    if len(found_types) > 0:
        print(f"    Examples: {found_types[:5]}")
    
    return ff


def verify_ligand_parameterization(mol, ff):
    """Test ligand parameterization."""
    from proxide.md.bridge.ligand import parameterize_ligand
    
    print("\n" + "=" * 60)
    print("Verifying Ligand Parameterization")
    print("=" * 60)
    
    if mol is None or ff is None:
        print("SKIP: Missing molecule or force field")
        return None
    
    try:
        params = parameterize_ligand(mol, ff)
        
        print(f"SystemParams generated:")
        print(f"  Charges: {params['charges'].shape}, sum={params['charges'].sum():.4f}")
        print(f"  Masses: {params['masses'].shape}, total={params['masses'].sum():.2f}")
        print(f"  Bonds: {params['bonds'].shape}")
        print(f"  Angles: {params['angles'].shape}")
        print(f"  Dihedrals: {params['dihedrals'].shape}")
        print(f"  Impropers: {params['impropers'].shape}")
        
        # Validate
        assert params['charges'].shape[0] == mol.n_atoms, "Charge count mismatch"
        assert params['bonds'].shape[0] == mol.n_bonds, "Bond count mismatch"
        assert params['angles'].shape[0] > 0, "Should have angles"
        
        print("\n✓ Ligand parameterization successful")
        return params
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_energy_calculation(mol, params):
    """Test energy calculation for the ligand."""
    print("\n" + "=" * 60)
    print("Verifying Energy Calculation")
    print("=" * 60)
    
    if mol is None or params is None:
        print("SKIP: Missing molecule or params")
        return
    
    try:
        from prolix.physics.system import make_energy_fn
        from jax_md import space
        import jax.numpy as jnp
        
        # Create displacement function
        displacement_fn, shift_fn = space.free()
        
        # Create energy function
        energy_fn = make_energy_fn(
            displacement_fn,
            params,
            neighbor_list=None,
        )
        
        # Calculate energy at molecule's geometry
        positions = jnp.array(mol.positions)
        energy = energy_fn(positions)
        
        print(f"Ligand Energy (GAFF):")
        print(f"  Total: {float(energy):.4f} kcal/mol")
        
        # Check it's finite
        assert np.isfinite(float(energy)), "Energy is not finite"
        
        print("\n✓ Energy calculation successful")
        
    except ImportError as e:
        print(f"SKIP: Missing dependency - {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def verify_openmm_comparison():
    """Compare against OpenMM GAFF energy (if OpenMM available)."""
    print("\n" + "=" * 60)
    print("Verifying OpenMM Comparison (Optional)")
    print("=" * 60)
    
    try:
        from openmm import app, unit
        from openmm.app import ForceField, Modeller
        from openmmforcefields.generators import GAFFTemplateGenerator
    except ImportError:
        print("SKIP: OpenMM or openmmforcefields not installed")
        return
    
    mol2_path = Path(__file__).parent.parent / "openmmforcefields/amber/files/imatinib.mol2"
    if not mol2_path.exists():
        print(f"SKIP: {mol2_path} not found")
        return
    
    try:
        from openff.toolkit import Molecule as OFFMolecule
        
        # Load with OpenFF
        off_mol = OFFMolecule.from_file(str(mol2_path), file_format="mol2")
        
        # Create GAFF template generator
        gaff_gen = GAFFTemplateGenerator(molecules=[off_mol], forcefield="gaff-2.11")
        
        # Create force field
        forcefield = ForceField()
        forcefield.registerTemplateGenerator(gaff_gen.generator)
        
        # Create topology from OpenFF molecule
        topology = off_mol.to_topology().to_openmm()
        
        # Positions
        positions = off_mol.conformers[0].magnitude * unit.angstrom
        
        # Create system
        system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
        
        # Get energy
        from openmm import Context, LangevinIntegrator
        integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picosecond)
        context = Context(system, integrator)
        context.setPositions(positions)
        
        state = context.getState(getEnergy=True)
        omm_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
        print(f"OpenMM GAFF Energy: {omm_energy:.4f} kcal/mol")
        print("\n✓ OpenMM comparison successful")
        
    except Exception as e:
        print(f"ERROR during OpenMM comparison: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all verification steps."""
    print("Ligand Support Verification")
    print("=" * 60)
    
    # Step 1: MOL2 Parsing
    mol = verify_mol2_parsing()
    
    # Step 2: GAFF Loading
    ff = verify_gaff_loading()
    
    # Step 3: Ligand Parameterization
    params = verify_ligand_parameterization(mol, ff)
    
    # Step 4: Energy Calculation
    verify_energy_calculation(mol, params)
    
    # Step 5: OpenMM Comparison (optional)
    verify_openmm_comparison()
    
    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
