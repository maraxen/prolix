"""Verify GAFF Implementation End-to-End against OpenMM.

This script benchmarks the prolix GAFF implementation by comparing
energies and forces for a test ligand against OpenMM's reference implementation.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

# Prolix imports
from proxide.io.parsing.molecule import Molecule
from proxide.md.bridge.ligand import parameterize_ligand
from proxide.physics.force_fields import load_force_field

from prolix.physics.system import make_energy_fn

# Optional OpenMM imports
try:
    import openmm
    from openff.toolkit import Molecule as OFFMolecule
    from openmm import app, unit
    from openmmforcefields.generators import GAFFTemplateGenerator
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("WARNING: OpenMM or openmmforcefields not available. Skipping comparison.")

# Constants
KJ_TO_KCAL = 0.239005736

def run_gaff_comparison(mol2_path: str, ff_path: str, ff_xml_version: str = "gaff-2.11"):
    """Run comparison for a single molecule."""
    print(f"\n=== Benchmarking {os.path.basename(mol2_path)} ===")

    # 1. Load Molecule
    # Use Prolix parser as primary
    print(f"Loading molecule from {mol2_path}...")
    mol_prolix = Molecule.from_mol2(mol2_path)

    # If OpenFF available, use it to get ground truth positions (or trust mol2)
    if HAS_OPENMM:
        print("Using OpenFF to validate positions...")
        try:
            off_mol = OFFMolecule.from_file(mol2_path, file_format="mol2")
            conf = off_mol.conformers[0].magnitude
            mol_prolix.positions = jnp.array(conf, dtype=jnp.float32)
        except Exception as e:
            print(f"OpenFF loading failed: {e}. Using Prolix positions.")

    positions = mol_prolix.positions

    # 2. Load Force Field (Prolix)
    print(f"Loading force field from {ff_path}...")
    ff = load_force_field(ff_path)

    has_gaff_params = hasattr(ff, "gaff_nonbonded_params") and ff.gaff_nonbonded_params
    if has_gaff_params:
        print(f"✓ Force field has embedded GAFF LJ parameters ({len(ff.gaff_nonbonded_params.sigmas)} types)")
    else:
        print("! Force field DOES NOT have embedded GAFF LJ parameters (using fallback)")

    # 3. Parameterize System (Prolix)
    print("Parameterizing in prolix...")
    # Verify we are using the new params
    if has_gaff_params:
         # Check a few random types from molecule to see if they exist in params
         for at in mol_prolix.atom_types:
             if at in ff.gaff_nonbonded_params.type_to_index:
                 idx = ff.gaff_nonbonded_params.type_to_index[at]
                 sig = ff.gaff_nonbonded_params.sigmas[idx]
                 eps = ff.gaff_nonbonded_params.epsilons[idx]
                 print(f"  Debug: Atom Type '{at}' found in FF. Sig={sig:.4f}, Eps={eps:.6f}")
             else:
                 print(f"  Debug: Atom Type '{at}' NOT found in FF.")

    sys_params = parameterize_ligand(mol_prolix, ff)

    # Create energy function
    displacement_fn, shift_fn = space.free()

    # JAX MD expects displacement_fn
    energy_fn = make_energy_fn(
        displacement_fn,
        sys_params,
        implicit_solvent=False, # Match OpenMM NoCutoff (vacuum)
        dielectric_constant=1.0,
        use_pbc=False
    )

    # Compute Prolix Energy & Forces
    E_prolix, F_prolix = jax.value_and_grad(energy_fn)(positions)
    print(f"Prolix Energy: {E_prolix:.4f} kcal/mol")

    if not HAS_OPENMM:
        print("Skipping OpenMM comparison due to missing dependencies.")
        return

    # 4. Setup OpenMM System
    print(f"Setting up OpenMM system with {ff_xml_version}...")
    gaff_gen = GAFFTemplateGenerator(molecules=[off_mol], forcefield=ff_xml_version)
    omm_ff = app.ForceField()
    omm_ff.registerTemplateGenerator(gaff_gen.generator)

    topology = off_mol.to_topology().to_openmm()
    omm_system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff)

    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(omm_system, integrator)
    context.setPositions(conf * unit.angstrom)

    # Compute OpenMM Energy & Forces
    state = context.getState(getEnergy=True, getForces=True)
    E_omm_total = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    F_omm = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom)

    # Decompose OpenMM Energy (approximate by grouping forces)
    # Ideally we'd validte component-wise but OpenMM accumulates.
    # We can check specific force groups if we assigned them, but strict comparison of total is good first step.

    print("\nTotal Energy:")
    print(f"  Prolix: {E_prolix:.4f} kcal/mol")
    print(f"  OpenMM: {E_omm_total:.4f} kcal/mol")
    print(f"  Diff:   {abs(E_prolix - E_omm_total):.4f} kcal/mol")

    # Force Comparison
    F_diff = F_prolix - F_omm
    rmse = np.sqrt(np.mean(F_diff**2))
    max_diff = np.max(np.abs(F_diff))

    print("\nForces:")
    print(f"  RMSE:     {rmse:.4f} kcal/mol/A")
    print(f"  Max Diff: {max_diff:.4f} kcal/mol/A")

    if abs(E_prolix - E_omm_total) > 0.1:
        print("❌ Energy mismatch > 0.1 kcal/mol")
    else:
        print("✅ Energy match within tolerance")

    if rmse > 1.0:
        print("❌ Force RMSE > 1.0 kcal/mol/A")
    else:
        print("✅ Force match within tolerance")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol2", type=str, default="proxide/tests/io/parsing/imatinib.mol2")
    parser.add_argument("--ff", type=str, default="data/force_fields/gaff-2.11.eqx")
    parser.add_argument("--xml_version", type=str, default="gaff-2.11")
    args = parser.parse_args()

    # Ensure molecule file exists
    if not os.path.exists(args.mol2):
        print(f"Creating dummy mol2 at {args.mol2} if needed or checking path...")
        # For now assume it exists or user provides valid path

    run_gaff_comparison(args.mol2, args.ff, args.xml_version)
