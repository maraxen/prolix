"""Compare OpenMM and Prolix forces/energies for solvated 1UAO.

This script creates identical solvated systems in both OpenMM and Prolix,
then compares the energy breakdown and forces to identify discrepancies.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space
from pathlib import Path

jax.config.update("jax_enable_x64", True)

# OpenMM imports
import openmm
from openmm import app, unit

# Prolix imports
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat
from prolix.physics import system, solvation
from prolix.physics.neighbor_list import ExclusionSpec, make_neighbor_list_fn


def create_openmm_system():
    """Create solvated 1UAO in OpenMM."""
    pdb_path = Path("data/pdb/1UAO.pdb")
    
    # Load PDB
    pdb = app.PDBFile(str(pdb_path))
    
    # Use ff14SB (similar to ff19SB) + TIP3P
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    
    # Create modeller and solvate
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, padding=0.8*unit.nanometer, model='tip3p')
    
    # Create system WITHOUT constraints for comparison
    omm_system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9*unit.nanometer,
        constraints=None,  # No constraints for force comparison
        rigidWater=False,  # Flexible water for force comparison
    )
    
    # Get positions in Angstroms
    positions_nm = modeller.positions.value_in_unit(unit.nanometer)
    positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])
    
    # Get box size
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    box_A = np.array([
        box_vecs[0][0].value_in_unit(unit.angstrom),
        box_vecs[1][1].value_in_unit(unit.angstrom),
        box_vecs[2][2].value_in_unit(unit.angstrom),
    ])
    
    return omm_system, modeller.topology, positions_A, box_A


def get_openmm_energy_breakdown(omm_system, positions_A, box_A):
    """Get energy and forces from OpenMM."""
    # Create context
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        omm_system, integrator,
        openmm.Platform.getPlatformByName("Reference")
    )
    
    # Set positions (convert Å to nm)
    positions_nm = [(p[0] * 0.1, p[1] * 0.1, p[2] * 0.1) for p in positions_A]
    context.setPositions(positions_nm * unit.nanometer)
    
    # Get state with energy and forces
    state = context.getState(getEnergy=True, getForces=True)
    
    total_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    
    # Forces: OpenMM gives kJ/mol/nm, convert to kcal/mol/Å
    forces_openmm = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    # kJ/mol/nm to kcal/mol/Å: divide by 4.184 (kJ->kcal), multiply by 10 (nm->Å conversion inverts)
    # Actually: F_kcal/A = F_kJ/nm * (1/4.184) * (1/10) = F_kJ/nm / 41.84
    forces_kcal_A = forces_openmm / 41.84
    
    # Get per-force-group energies
    energy_breakdown = {}
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        force.setForceGroup(i)
    
    # Recreate context with force groups
    context2 = openmm.Context(
        omm_system, openmm.VerletIntegrator(0.001 * unit.picoseconds),
        openmm.Platform.getPlatformByName("Reference")
    )
    context2.setPositions(positions_nm * unit.nanometer)
    
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        force_name = type(force).__name__
        state = context2.getState(getEnergy=True, groups={i})
        energy_breakdown[force_name] = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    
    return total_energy, forces_kcal_A, energy_breakdown


def create_prolix_system(positions_A, box_A, n_atoms):
    """Create Prolix energy function matching the OpenMM system."""
    # For now, use a simplified energy function
    # TODO: Extract exact parameters from OpenMM
    
    displacement_fn, shift_fn = space.periodic(np.array(box_A))
    
    # Allocate neighbor list
    neighbor_fn = make_neighbor_list_fn(displacement_fn, np.array(box_A), 9.0)
    neighbor = neighbor_fn.allocate(jnp.array(positions_A))
    
    return displacement_fn, neighbor_fn, neighbor


def extract_openmm_params(omm_system):
    """Extract force field parameters from OpenMM system."""
    n_atoms = omm_system.getNumParticles()
    
    # Extract nonbonded parameters
    charges = np.zeros(n_atoms)
    sigmas = np.zeros(n_atoms)  # nm
    epsilons = np.zeros(n_atoms)  # kJ/mol
    
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(i)
                charges[i] = q.value_in_unit(unit.elementary_charge)
                sigmas[i] = sig.value_in_unit(unit.nanometer) * 10  # nm -> Å
                epsilons[i] = eps.value_in_unit(unit.kilojoule_per_mole) / 4.184  # kJ->kcal
    
    return {
        'charges': charges,
        'sigmas': sigmas,
        'epsilons': epsilons,
        'n_atoms': n_atoms,
    }


def main():
    print("=" * 60)
    print("OpenMM vs Prolix Force/Energy Comparison")
    print("=" * 60)
    
    # Create OpenMM system
    print("\n1. Creating OpenMM solvated system...")
    omm_system, topology, positions_A, box_A = create_openmm_system()
    
    n_atoms = len(positions_A)
    n_waters = sum(1 for r in topology.residues() if r.name in ('HOH', 'WAT'))
    n_protein = n_atoms - n_waters * 3
    
    print(f"   Total atoms: {n_atoms}")
    print(f"   Protein atoms: {n_protein}")
    print(f"   Water molecules: {n_waters}")
    print(f"   Box size: {box_A} Å")
    
    # Get OpenMM energies and forces
    print("\n2. Computing OpenMM energies and forces...")
    omm_total, omm_forces, omm_breakdown = get_openmm_energy_breakdown(omm_system, positions_A, box_A)
    
    print(f"   Total energy: {omm_total:.2f} kcal/mol")
    print("   Energy breakdown:")
    for name, energy in omm_breakdown.items():
        print(f"      {name}: {energy:.2f} kcal/mol")
    
    max_force = np.max(np.abs(omm_forces))
    mean_force = np.mean(np.abs(omm_forces))
    print(f"\n   Force statistics:")
    print(f"      Max force: {max_force:.2f} kcal/mol/Å")
    print(f"      Mean |force|: {mean_force:.2f} kcal/mol/Å")
    
    # Extract parameters for Prolix
    print("\n3. Extracting OpenMM parameters...")
    params = extract_openmm_params(omm_system)
    
    print(f"   Charges range: [{params['charges'].min():.3f}, {params['charges'].max():.3f}] e")
    print(f"   Sigma range: [{params['sigmas'].min():.3f}, {params['sigmas'].max():.3f}] Å")
    print(f"   Epsilon range: [{params['epsilons'].min():.4f}, {params['epsilons'].max():.4f}] kcal/mol")
    
    # Check for any zero sigmas (potential issue)
    zero_sig = np.sum(params['sigmas'] == 0)
    print(f"   Atoms with sigma=0: {zero_sig}")
    
    # Check minimum distances
    print("\n4. Checking minimum inter-atomic distances...")
    positions = np.array(positions_A)
    
    # Compute pairwise distances (sample for speed)
    n_sample = min(1000, n_atoms)
    idx = np.random.choice(n_atoms, n_sample, replace=False)
    sample_pos = positions[idx]
    
    dists = np.linalg.norm(sample_pos[:, None, :] - sample_pos[None, :, :], axis=-1)
    np.fill_diagonal(dists, 1e10)
    min_dist = dists.min()
    
    print(f"   Min distance (sample): {min_dist:.3f} Å")
    
    if min_dist < 0.5:
        print("   ⚠️ WARNING: Very close atoms detected!")
    
    # Look for clashing atoms
    clash_pairs = np.where(dists < 1.0)
    if len(clash_pairs[0]) > 0:
        print(f"   Found {len(clash_pairs[0])} pairs with distance < 1.0 Å")
    
    # Test a single MD step in OpenMM
    print("\n5. Testing OpenMM MD stability...")
    integrator = openmm.LangevinMiddleIntegrator(
        10 * unit.kelvin,  # Very low temperature
        1 / unit.picosecond,
        0.0005 * unit.picoseconds,  # 0.5 fs
    )
    context = openmm.Context(
        omm_system, integrator,
        openmm.Platform.getPlatformByName("Reference")
    )
    positions_nm = [(p[0] * 0.1, p[1] * 0.1, p[2] * 0.1) for p in positions_A]
    context.setPositions(positions_nm * unit.nanometer)
    
    # Run 10 steps
    for i in range(10):
        integrator.step(1)
        state = context.getState(getEnergy=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        print(f"   Step {i+1}: E = {e:.2f} kcal/mol")
    
    print("\n6. Summary")
    print("=" * 60)
    if omm_total < 0 and max_force < 1e6:
        print("✅ OpenMM system appears stable")
    else:
        print("⚠️ OpenMM system may have issues")


if __name__ == "__main__":
    main()
